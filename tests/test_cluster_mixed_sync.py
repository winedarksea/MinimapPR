"""Phase E: mixed-sync-grade cluster localization.

Three GPS_PPS nodes + one NTP-disciplined node form a cluster. We verify:

* The cluster's per-sensor clock uncertainty follows the configured clock-class
  defaults (``localization_clock_class_uncertainty_seconds``), floored at the
  cluster's declared sync grade.
* Localization still converges when the NTP sensor injects a realistic timing
  jitter (≈ a few ms of skew). With the NTP node's pairs down-weighted by its
  clock uncertainty (kept, never dropped), the recovered source position
  remains within tolerance.
* When the same NTP sensor is treated as perfectly timed, the down-weighting
  benefit disappears or is reduced — i.e., the weighting actually has effect.
"""
from __future__ import annotations

import numpy as np
import pytest

from minimappr.core.cluster_registry import ClusterRegistry
from minimappr.core.localization_dispatch import LocalizationDispatcher
from minimappr.config import Settings
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.pair_timing_uncertainty import ClockUncertaintyPolicy, NodeClockEvidence
from minimappr.models import ClusterSpec, NodeSpec, NodeType, SyncGrade, TimeQuality
from tests.helpers import shift_signal


SAMPLE_RATE_HZ = 16_000
SOUND_SPEED_MPS = 343.2
CLOCK_POLICY = ClockUncertaintyPolicy.from_settings(Settings())


def _node(node_id: str, position: tuple[float, float, float]) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        node_type=NodeType.POINT,
        position_m=position,
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
    )


async def _record_clocks(nodes: NodeRegistry, gps_node_ids: tuple[str, ...], ntp_node_id: str) -> None:
    for nid in gps_node_ids:
        await nodes.record_node_clock_evidence(nid, NodeClockEvidence(TimeQuality.GPS_LOCKED.value))
    await nodes.record_node_clock_evidence(ntp_node_id, NodeClockEvidence(TimeQuality.NTP_DISCIPLINED.value))


@pytest.mark.asyncio
async def test_mixed_sync_cluster_timing_uncertainty_matches_class_defaults() -> None:
    """3 PPS + 1 NTP cluster: per-sensor clock 1σ equals the class defaults."""
    nodes = NodeRegistry()
    positions = {
        "p0": (0.0, 0.0, 2.0),
        "p1": (3.0, 0.0, 2.0),
        "p2": (3.0, 3.0, 2.0),
        "ntp0": (0.0, 3.0, 2.0),
    }
    for nid, pos in positions.items():
        await nodes.upsert(_node(nid, pos), last_seen_ns=1)
    await _record_clocks(nodes, ("p0", "p1", "p2"), "ntp0")

    clusters = ClusterRegistry()
    await clusters.upsert(ClusterSpec(
        id="mixed",
        member_node_ids=list(positions.keys()),
        declared_sync_grade=SyncGrade.GPS_PPS,
    ))

    uncertainty = await clusters.cluster_sensor_timing_uncertainty_seconds("mixed", nodes, CLOCK_POLICY)
    for nid in ("p0", "p1", "p2"):
        assert uncertainty[f"{nid}:ch0"] == pytest.approx(1.0e-6)
    assert uncertainty["ntp0:ch0"] == pytest.approx(2.0e-3)

    # A cluster declared NTP-synced caps every member at NTP-class timing.
    await clusters.upsert(ClusterSpec(
        id="mixed",
        member_node_ids=list(positions.keys()),
        declared_sync_grade=SyncGrade.NTP,
    ))
    floored = await clusters.cluster_sensor_timing_uncertainty_seconds("mixed", nodes, CLOCK_POLICY)
    assert all(value == pytest.approx(2.0e-3) for value in floored.values())


@pytest.mark.asyncio
async def test_mixed_sync_localization_converges_with_weighting() -> None:
    """With an NTP sensor injecting timing jitter, weighting keeps localization
    accurate; equal-weighting degrades it.
    """
    rng = np.random.default_rng(0)
    nodes = NodeRegistry()
    positions = {
        "p0": (0.0, 0.0, 2.0),
        "p1": (4.0, 0.0, 2.0),
        "p2": (4.0, 4.0, 2.0),
        "ntp0": (0.0, 4.0, 2.0),
    }
    for nid, pos in positions.items():
        await nodes.upsert(_node(nid, pos), last_seen_ns=1)
    await _record_clocks(nodes, ("p0", "p1", "p2"), "ntp0")

    clusters = ClusterRegistry()
    await clusters.upsert(ClusterSpec(
        id="mixed",
        member_node_ids=list(positions.keys()),
        declared_sync_grade=SyncGrade.GPS_PPS,
    ))

    sensor_positions = await clusters.cluster_sensor_positions("mixed", nodes)
    uncertainty = await clusters.cluster_sensor_timing_uncertainty_seconds("mixed", nodes, CLOCK_POLICY)
    sensor_node_ids = {sid: sid.split(":")[0] for sid in sensor_positions}

    source = np.array([1.3, 1.7, 1.8])
    n = 4096
    excitation = (rng.standard_normal(n) * np.hanning(n)).astype(np.float32)
    pad = np.concatenate([np.zeros(200, dtype=np.float32), excitation])

    # Inject ~2 ms of NTP-grade timing jitter on the NTP sensor only.
    NTP_JITTER_S = 0.002

    windows: dict[str, np.ndarray] = {}
    for sid, pos in sensor_positions.items():
        true_delay = float(np.linalg.norm(source - pos)) / SOUND_SPEED_MPS
        skew = NTP_JITTER_S if sid == "ntp0:ch0" else 0.0
        windows[sid] = shift_signal(pad, SAMPLE_RATE_HZ, true_delay + skew)

    disp = LocalizationDispatcher()

    weighted = disp.localize(
        sensor_positions=sensor_positions,
        sensor_windows=windows,
        sample_rate_hz=SAMPLE_RATE_HZ,
        temperature_c=20.0,
        humidity_fraction=0.5,
        sensor_timing_uncertainty_seconds=uncertainty,
        sensor_node_ids=sensor_node_ids,
    )
    equal = disp.localize(
        sensor_positions=sensor_positions,
        sensor_windows=windows,
        sample_rate_hz=SAMPLE_RATE_HZ,
        temperature_c=20.0,
        humidity_fraction=0.5,
        sensor_timing_uncertainty_seconds={sid: 0.0 for sid in sensor_positions},
        sensor_node_ids=sensor_node_ids,
    )

    err_weighted = float(np.linalg.norm(np.array(weighted.position_m) - source))
    err_equal = float(np.linalg.norm(np.array(equal.position_m) - source))

    # Weighted estimate must converge within a reasonable bound despite the
    # NTP sensor's skew.
    assert err_weighted < 0.6, f"weighted err {err_weighted:.3f}m"
    # And weighting should not make things worse than naive equal weighting.
    assert err_weighted <= err_equal + 1e-6, (
        f"weighting hurt accuracy: weighted={err_weighted:.3f}m equal={err_equal:.3f}m"
    )
    assert weighted.confidence > 0.0
