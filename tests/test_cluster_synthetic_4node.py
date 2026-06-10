"""Phase E: synthetic 4-node 1-mic cluster localization.

Four virtual single-mic nodes are placed at the corners of a 2 m square. A
known source emits a broadband impulse; per-mic delays are computed
geometrically (c = 343.2 m/s) and applied. The full registry → cluster →
dispatcher pipeline is exercised, and the recovered position must lie within
0.30 m of truth.
"""
from __future__ import annotations

import numpy as np
import pytest

from minimappr.core.cluster_registry import ClusterRegistry
from minimappr.core.localization_dispatch import LocalizationDispatcher
from minimappr.core.node_registry import NodeRegistry
from minimappr.models import ClusterSpec, NodeSpec, NodeType, SyncGrade
from tests.helpers import shift_signal


SAMPLE_RATE_HZ = 16_000
SOUND_SPEED_MPS = 343.2


def _single_mic_node(node_id: str, position: tuple[float, float, float]) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        node_type=NodeType.POINT,
        position_m=position,
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
    )


def _impulse_excitation(n: int, seed: int = 7) -> np.ndarray:
    """Broadband random excitation, windowed — gives GCC-PHAT a clear peak."""
    rng = np.random.default_rng(seed)
    sig = rng.standard_normal(n).astype(np.float64) * np.hanning(n)
    return sig.astype(np.float32)


@pytest.mark.asyncio
async def test_4node_square_cluster_recovers_known_source() -> None:
    # 2 m square at z = 2.0 m.
    node_positions: dict[str, tuple[float, float, float]] = {
        "n0": (0.0, 0.0, 2.0),
        "n1": (2.0, 0.0, 2.0),
        "n2": (2.0, 2.0, 2.0),
        "n3": (0.0, 2.0, 2.0),
    }
    source_position = np.array([0.8, 1.1, 1.7])

    # Build registries.
    nodes = NodeRegistry()
    for nid, pos in node_positions.items():
        await nodes.upsert(_single_mic_node(nid, pos), last_seen_ns=1)
        await nodes.update_sensor_sync_grade(f"{nid}:ch0", SyncGrade.GPS_PPS)

    clusters = ClusterRegistry()
    await clusters.upsert(ClusterSpec(
        id="cluster_square",
        member_node_ids=list(node_positions.keys()),
        declared_sync_grade=SyncGrade.GPS_PPS,
    ))
    await clusters.update_node_memberships(nodes)

    # Resolve cluster topology.
    sensor_positions = await clusters.cluster_sensor_positions("cluster_square", nodes)
    sensor_weights = await clusters.cluster_sensor_weights("cluster_square", nodes)
    assert set(sensor_positions.keys()) == {f"n{i}:ch0" for i in range(4)}
    # All GPS_PPS → all weights == 1.0.
    assert all(w == pytest.approx(1.0) for w in sensor_weights.values())

    # Synthesize per-mic windows via geometric delay propagation.
    duration_s = 0.25
    n_samples = int(SAMPLE_RATE_HZ * duration_s)
    excitation = _impulse_excitation(n_samples)

    # Pre-pad so even the most distant sensor's negative offset still fits.
    pad = SAMPLE_RATE_HZ // 100  # 10 ms head room
    padded = np.concatenate([np.zeros(pad, dtype=np.float32), excitation])

    sensor_windows: dict[str, np.ndarray] = {}
    distances: dict[str, float] = {}
    for sid, pos in sensor_positions.items():
        distance = float(np.linalg.norm(source_position - pos))
        distances[sid] = distance
        delay_s = distance / SOUND_SPEED_MPS
        sensor_windows[sid] = shift_signal(padded, SAMPLE_RATE_HZ, delay_s)

    # Localize through the dispatcher (matches what fusion_node uses).
    dispatcher = LocalizationDispatcher()
    result = dispatcher.localize(
        sensor_positions=sensor_positions,
        sensor_windows=sensor_windows,
        sample_rate_hz=SAMPLE_RATE_HZ,
        temperature_c=20.0,
        humidity_fraction=0.5,
        sensor_weights=sensor_weights,
    )

    estimate = np.array(result.position_m)
    error_m = float(np.linalg.norm(estimate - source_position))
    assert error_m < 0.30, f"recovered {estimate.tolist()} vs truth {source_position.tolist()}, err={error_m:.3f}m"
    assert result.confidence > 0.0
    assert result.position_covariance_m2 is not None


@pytest.mark.asyncio
async def test_uniform_weights_match_legacy_behavior() -> None:
    """sensor_weights=None must reproduce the same result as uniform weights=1.0.

    Bit-identity guard for the legacy code path described in the plan.
    """
    positions: dict[str, np.ndarray] = {
        f"s{i}": np.array(p, dtype=np.float64)
        for i, p in enumerate([
            (0.0, 0.0, 2.0), (3.0, 0.0, 2.0), (3.0, 3.0, 2.0),
            (0.0, 3.0, 2.0), (1.5, 1.5, 3.0),
        ])
    }
    source = np.array([1.2, 1.4, 1.9])

    n = 4096
    rng = np.random.default_rng(11)
    excitation = (rng.standard_normal(n) * np.hanning(n)).astype(np.float32)
    pad = np.concatenate([np.zeros(200, dtype=np.float32), excitation])

    windows: dict[str, np.ndarray] = {}
    for sid, pos in positions.items():
        delay_s = float(np.linalg.norm(source - pos)) / SOUND_SPEED_MPS
        windows[sid] = shift_signal(pad, SAMPLE_RATE_HZ, delay_s)

    disp = LocalizationDispatcher()
    r_default = disp.localize(
        sensor_positions=positions, sensor_windows=windows,
        sample_rate_hz=SAMPLE_RATE_HZ, temperature_c=20.0, humidity_fraction=0.5,
    )
    r_uniform = disp.localize(
        sensor_positions=positions, sensor_windows=windows,
        sample_rate_hz=SAMPLE_RATE_HZ, temperature_c=20.0, humidity_fraction=0.5,
        sensor_weights={sid: 1.0 for sid in positions},
    )
    np.testing.assert_allclose(r_default.position_m, r_uniform.position_m, atol=1e-9)
    assert r_default.confidence == pytest.approx(r_uniform.confidence, abs=1e-9)
