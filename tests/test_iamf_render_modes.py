"""Phase E: IAMF render-mode coverage.

Exercises ``render_cluster_audio()`` for all three render modes:

* ``FOA_BED``       — tetra 4-mic, all PPS sensors. Geometry passes
                       ``foa_geometry_suitable``, output is a 4×N B-format bed.
* ``N_MONO_OBJECTS`` — 4-node single-mic cluster (large baseline). Output is
                       an ``NMonoBundle`` carrying per-sensor positions.
* ``AUTO``          — Mixed cluster (4 PPS + 2 NTP). PPS subset is FOA-eligible
                       so FOA bed is built from PPS sensors, while NTP sensors
                       are emitted as separate mono objects alongside.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from minimappr.core.ambi_atob import SIRITH_MIC_POSITIONS_M, foa_geometry_suitable
from minimappr.core.iamf_pipeline import ClusterRenderResult, render_cluster_audio
from minimappr.core.iamf_writer import NMonoBundle, write_n_mono_substreams
from minimappr.models import ClusterSpec, IamfRenderMode, NodeOrientation, NodeSpec, NodeType, SyncGrade
from tests.helpers import SIRITH_TETRA_SENSOR_OFFSETS_M


SAMPLE_RATE_HZ = 16_000


def _silence(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def _tone(n: int, freq_hz: float = 440.0) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE_HZ
    return (0.1 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_foa_bed_tetra_renders_bed() -> None:
    """4-mic tetra geometry routed through render_cluster_audio yields FOA bed."""
    n = 8192
    sensor_ids = [f"tetra:ch{i}" for i in range(4)]
    sensor_positions = {sid: SIRITH_MIC_POSITIONS_M[i] for i, sid in enumerate(sensor_ids)}
    channels = {sid: _tone(n) for sid in sensor_ids}
    grades = {sid: SyncGrade.GPS_PPS for sid in sensor_ids}

    spec = ClusterSpec(
        id="tetra1",
        member_node_ids=["tetra"],
        declared_sync_grade=SyncGrade.GPS_PPS,
        iamf_render_mode=IamfRenderMode.FOA_BED,
        max_baseline_m_for_foa=0.5,
    )

    # Sanity: tetra geometry is FOA-suitable under the 0.5 m baseline cap.
    suitable, _ = foa_geometry_suitable(SIRITH_MIC_POSITIONS_M, 0.5)
    assert suitable is True

    result = render_cluster_audio(spec, channels, sensor_positions, grades, SAMPLE_RATE_HZ)

    assert isinstance(result, ClusterRenderResult)
    assert result.render_mode == IamfRenderMode.FOA_BED
    assert result.foa_bed is not None
    assert result.foa_bed.shape == (4, n)
    assert result.foa_bed.dtype == np.float32
    assert result.mono_bundle is None
    assert result.ntp_mono_bundle is None
    assert result.foa_sensor_ids == sensor_ids


def test_n_mono_objects_explicit_mode_for_distributed_cluster() -> None:
    """Distributed 4-node single-mic cluster (large baselines): N-mono only."""
    n = 4096
    positions = {
        "n0:ch0": np.array([0.0, 0.0, 2.0]),
        "n1:ch0": np.array([2.0, 0.0, 2.0]),
        "n2:ch0": np.array([2.0, 2.0, 2.0]),
        "n3:ch0": np.array([0.0, 2.0, 2.0]),
    }
    channels = {sid: _tone(n) for sid in positions}
    grades = {sid: SyncGrade.GPS_PPS for sid in positions}

    spec = ClusterSpec(
        id="distributed",
        member_node_ids=["n0", "n1", "n2", "n3"],
        declared_sync_grade=SyncGrade.GPS_PPS,
        iamf_render_mode=IamfRenderMode.N_MONO_OBJECTS,
    )

    result = render_cluster_audio(spec, channels, positions, grades, SAMPLE_RATE_HZ)
    assert result.render_mode == IamfRenderMode.N_MONO_OBJECTS
    assert result.foa_bed is None
    bundle = result.mono_bundle
    assert isinstance(bundle, NMonoBundle)
    assert len(bundle.streams) == 4
    assert bundle.sample_rate_hz == SAMPLE_RATE_HZ

    # Sensor IDs and positions must round-trip via the position manifest.
    manifest = bundle.position_manifest()
    streams = manifest["streams"]
    assert {s["sensor_id"] for s in streams} == set(positions.keys())
    pos_by_sid = {s["sensor_id"]: s["position_m"] for s in streams}
    for sid, pos in positions.items():
        np.testing.assert_allclose(pos_by_sid[sid], list(pos), atol=1e-9)


def test_auto_mode_partitions_pps_into_foa_and_ntp_into_objects() -> None:
    """AUTO with 4 PPS + 2 NTP: PPS form FOA bed, NTP become mono objects."""
    n = 2048

    # 4 PPS sensors arranged in tetra geometry (FOA-suitable).
    pps_ids = [f"pps:ch{i}" for i in range(4)]
    pps_positions = {sid: SIRITH_MIC_POSITIONS_M[i] for i, sid in enumerate(pps_ids)}
    pps_grades = {sid: SyncGrade.GPS_PPS for sid in pps_ids}

    # 2 NTP sensors far away (irrelevant to FOA eligibility — they're excluded
    # from the FOA partition by grade, not by geometry).
    ntp_positions = {
        "ntp_a:ch0": np.array([10.0, 0.0, 2.0]),
        "ntp_b:ch0": np.array([0.0, 10.0, 2.0]),
    }
    ntp_grades = {sid: SyncGrade.NTP for sid in ntp_positions}

    channels = {sid: _tone(n) for sid in {**pps_positions, **ntp_positions}}
    positions = {**pps_positions, **ntp_positions}
    grades = {**pps_grades, **ntp_grades}

    spec = ClusterSpec(
        id="auto_mixed",
        member_node_ids=["pps", "ntp_a", "ntp_b"],
        declared_sync_grade=SyncGrade.GPS_PPS,
        iamf_render_mode=IamfRenderMode.AUTO,
        max_baseline_m_for_foa=0.5,
    )

    result = render_cluster_audio(spec, channels, positions, grades, SAMPLE_RATE_HZ)

    assert result.render_mode == IamfRenderMode.FOA_BED
    assert result.foa_bed is not None
    assert result.foa_bed.shape == (4, n)
    assert set(result.foa_sensor_ids or []) == set(pps_ids)

    # NTP sensors must surface as a separate N-mono bundle.
    ntp_bundle = result.ntp_mono_bundle
    assert isinstance(ntp_bundle, NMonoBundle)
    assert len(ntp_bundle.streams) == 2
    ntp_sids = {meta.sensor_id for meta, _ in ntp_bundle.streams}
    assert ntp_sids == set(ntp_positions.keys())
    for meta, _ in ntp_bundle.streams:
        assert meta.sync_grade == "ntp"


def test_auto_mode_uses_single_tetra_anchor_not_distributed_high_grade_foa() -> None:
    n = 2048
    anchor_ids = [f"anchor:ch{i}" for i in range(4)]
    anchor_positions = {sid: SIRITH_MIC_POSITIONS_M[i] for i, sid in enumerate(anchor_ids)}
    remote_positions = {
        "remote_a:ch0": np.array([8.0, 0.0, 0.0]),
        "remote_b:ch0": np.array([0.0, 8.0, 0.0]),
        "remote_c:ch0": np.array([8.0, 8.0, 0.0]),
        "remote_d:ch0": np.array([4.0, 8.0, 0.0]),
    }
    positions = {**anchor_positions, **remote_positions}
    channels = {sid: _tone(n) for sid in positions}
    grades = {sid: SyncGrade.GPS_PPS for sid in positions}
    spec = ClusterSpec(
        id="mixed_anchor",
        member_node_ids=["anchor", "remote_a", "remote_b", "remote_c", "remote_d"],
        declared_sync_grade=SyncGrade.GPS_PPS,
        iamf_render_mode=IamfRenderMode.AUTO,
        max_baseline_m_for_foa=0.5,
    )

    result = render_cluster_audio(spec, channels, positions, grades, SAMPLE_RATE_HZ)

    assert result.render_mode == IamfRenderMode.FOA_BED
    assert result.anchor_node_id == "anchor"
    assert result.foa_sensor_ids == anchor_ids
    assert result.ntp_mono_bundle is not None
    remote_sids = {meta.sensor_id for meta, _ in result.ntp_mono_bundle.streams}
    assert remote_sids == set(remote_positions)


def test_auto_mode_can_derive_anchor_geometry_from_oriented_node_spec() -> None:
    n = 2048
    anchor_ids = [f"anchor:ch{i}" for i in range(4)]
    channels = {sid: _tone(n) for sid in anchor_ids}
    grades = {sid: SyncGrade.GPS_PPS for sid in anchor_ids}
    raw_positions = {sid: np.asarray(SIRITH_TETRA_SENSOR_OFFSETS_M[i], dtype=np.float64) for i, sid in enumerate(anchor_ids)}
    node = NodeSpec(
        id="anchor",
        node_type=NodeType.SIRITH_TETRA,
        position_m=(10.0, 20.0, 0.0),
        sensor_offsets_m=list(SIRITH_TETRA_SENSOR_OFFSETS_M),
        orientation=NodeOrientation(yaw_deg=90.0),
    )
    spec = ClusterSpec(
        id="oriented_anchor",
        member_node_ids=["anchor"],
        declared_sync_grade=SyncGrade.GPS_PPS,
        iamf_render_mode=IamfRenderMode.AUTO,
        max_baseline_m_for_foa=0.5,
    )

    result = render_cluster_audio(
        spec,
        channels,
        raw_positions,
        grades,
        SAMPLE_RATE_HZ,
        node_specs={"anchor": node},
    )

    assert result.render_mode == IamfRenderMode.FOA_BED
    assert result.anchor_node_id == "anchor"
    assert result.foa_sensor_ids == anchor_ids


def test_auto_mode_falls_back_to_all_n_mono_when_no_foa_eligible_set() -> None:
    """AUTO with fewer than 4 PPS sensors → all-N-mono fallback."""
    n = 2048
    positions = {
        "p0:ch0": np.array([0.0, 0.0, 2.0]),
        "p1:ch0": np.array([1.0, 0.0, 2.0]),
        "n0:ch0": np.array([0.0, 1.0, 2.0]),
        "n1:ch0": np.array([1.0, 1.0, 2.0]),
    }
    grades = {
        "p0:ch0": SyncGrade.GPS_PPS,
        "p1:ch0": SyncGrade.GPS_PPS,
        "n0:ch0": SyncGrade.NTP,
        "n1:ch0": SyncGrade.NTP,
    }
    channels = {sid: _tone(n) for sid in positions}

    spec = ClusterSpec(
        id="auto_no_foa",
        member_node_ids=["p0", "p1", "n0", "n1"],
        declared_sync_grade=SyncGrade.GPS_PPS,
        iamf_render_mode=IamfRenderMode.AUTO,
    )

    result = render_cluster_audio(spec, channels, positions, grades, SAMPLE_RATE_HZ)
    assert result.render_mode == IamfRenderMode.N_MONO_OBJECTS
    assert result.foa_bed is None
    assert result.mono_bundle is not None
    assert len(result.mono_bundle.streams) == 4


def test_write_n_mono_substreams_produces_parsable_bundle() -> None:
    """The on-disk bundle keeps magic, manifest JSON, and per-stream PCM."""
    n = 256
    positions = {
        "n0:ch0": np.array([0.0, 0.0, 2.0]),
        "n1:ch0": np.array([2.0, 0.0, 2.0]),
        "n2:ch0": np.array([2.0, 2.0, 2.0]),
        "n3:ch0": np.array([0.0, 2.0, 2.0]),
    }
    channels = {sid: _silence(n) for sid in positions}
    grades = {sid: SyncGrade.GPS_PPS for sid in positions}
    spec = ClusterSpec(
        id="d", member_node_ids=list(positions),
        declared_sync_grade=SyncGrade.GPS_PPS,
        iamf_render_mode=IamfRenderMode.N_MONO_OBJECTS,
    )
    result = render_cluster_audio(spec, channels, positions, grades, SAMPLE_RATE_HZ)
    assert result.mono_bundle is not None

    blob, manifest_json = write_n_mono_substreams(result.mono_bundle)
    assert blob[:4] == b"NMBT"
    parsed = json.loads(manifest_json)
    assert parsed["sample_rate_hz"] == SAMPLE_RATE_HZ
    assert len(parsed["streams"]) == 4
    for s in parsed["streams"]:
        assert "sensor_id" in s and "position_m" in s and s["n_samples"] == n
