"""P9: one clock policy on both ingest paths.

The Rust-handoff re-solve (``core/rust_tdoa_solve.py``, fed the sidecar's
per-pair ``timing_uncertainty_seconds``) and the Python-ingest pair measurement
(``core/tdoa_measurements.py``, fed per-node clock evidence from the registry)
must weight the same pair set identically, and the Rust and Python per-class
defaults must agree.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from minimappr.config import Settings
from minimappr.core import localization_overhaul_config as overhaul_defaults
from minimappr.core.cluster_registry import ClusterRegistry
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.pair_timing_uncertainty import (
    ClockUncertaintyPolicy,
    cross_node_pair_timing_uncertainty_seconds,
    node_clock_evidence_from_frame,
    node_timing_uncertainty_seconds,
    solver_reference_time_std_seconds,
)
from minimappr.core.rust_tdoa_solve import pair_measurements_from_rust_tdoas, solve_localization_from_rust_tdoas
from minimappr.core.tdoa_measurements import measure_pair_tdoas, normalized_pair_quality
from minimappr.models import ClusterSpec, NodeSpec, NodeType, SyncGrade, TimeQuality

REPO_ROOT = Path(__file__).resolve().parents[1]
RUST_SOLVE_CONFIG_PATH = REPO_ROOT / "minimappr-ingest-sidecar" / "src" / "localization_solve_config.rs"
SAMPLE_RATE_HZ = 48_000
INTERPOLATION_FACTOR = 4
SOUND_SPEED_MPS = 343.0
CORRELATION_PEAK = 0.8
DEFAULT_POLICY = ClockUncertaintyPolicy.from_settings(Settings())

TETRA_OFFSETS_M = [(0.05, 0.05, 0.05), (0.05, -0.05, -0.05), (-0.05, 0.05, -0.05), (-0.05, -0.05, 0.05)]
NODE_ORIGINS_M = {
    "tetra-a": (0.0, 0.0, 1.5),
    "point-b": (50.0, 0.0, 2.0),
    "point-c": (0.0, 50.0, 1.0),
    "point-d": (50.0, 50.0, 2.5),
}
# What each node last reported; covers every rule: class default, holdover
# error above its class default, and an unrecognised class.
NODE_CLOCK_REPORTS = {
    "tetra-a": (TimeQuality.GPS_LOCKED, {}),
    "point-b": (TimeQuality.NTP_DISCIPLINED, {}),
    "point-c": (TimeQuality.GPS_HOLDOVER, {"clock_holdover": {"holdover_active": True, "predicted_error_ns": 300_000}}),
    "point-d": ("sundial", {}),
}


def _sensor_layout() -> tuple[list[str], dict[str, np.ndarray], dict[str, str]]:
    positions: dict[str, np.ndarray] = {}
    node_ids: dict[str, str] = {}
    for node_id, origin in NODE_ORIGINS_M.items():
        offsets = TETRA_OFFSETS_M if node_id.startswith("tetra") else [(0.0, 0.0, 0.0)]
        for index, offset in enumerate(offsets):
            sensor_id = f"{node_id}:ch{index}"
            positions[sensor_id] = np.add(origin, offset).astype(np.float64)
            node_ids[sensor_id] = node_id
    return sorted(positions), positions, node_ids


def _arrival_lag_seconds(source_m: np.ndarray, position_a: np.ndarray, position_b: np.ndarray) -> float:
    return float((np.linalg.norm(source_m - position_a) - np.linalg.norm(source_m - position_b)) / SOUND_SPEED_MPS)


async def _registry_sensor_uncertainty(cluster_declared_sync_grade: SyncGrade | None = None) -> dict[str, float]:
    """Python-ingest per-sensor clock 1-sigma, optionally via a cluster of every node."""
    registry = NodeRegistry()
    for node_id, origin in NODE_ORIGINS_M.items():
        offsets = TETRA_OFFSETS_M if node_id.startswith("tetra") else [(0.0, 0.0, 0.0)]
        await registry.upsert(
            NodeSpec(
                id=node_id,
                node_type=NodeType.SIRITH_TETRA if node_id.startswith("tetra") else NodeType.POINT,
                position_m=origin,
                sensor_offsets_m=offsets,
                capabilities=["audio"],
            ),
            last_seen_ns=1,
        )
        time_quality, timing_diagnostics = NODE_CLOCK_REPORTS[node_id]
        await registry.record_node_clock_evidence(node_id, node_clock_evidence_from_frame(time_quality, timing_diagnostics))
    if cluster_declared_sync_grade is None:
        return await registry.sensor_timing_uncertainty_seconds(DEFAULT_POLICY)
    clusters = ClusterRegistry()
    await clusters.upsert(
        ClusterSpec(id="site", member_node_ids=list(NODE_ORIGINS_M), declared_sync_grade=cluster_declared_sync_grade)
    )
    return await clusters.cluster_sensor_timing_uncertainty_seconds("site", registry, DEFAULT_POLICY)


def _rust_node_uncertainty() -> dict[str, float]:
    """The per-node uncertainty the sidecar computes from the same reports."""
    return {
        node_id: node_timing_uncertainty_seconds(node_clock_evidence_from_frame(*report), DEFAULT_POLICY)
        for node_id, report in NODE_CLOCK_REPORTS.items()
    }


def _rust_pair_tdoas(
    ordered_sensor_ids: list[str],
    positions: dict[str, np.ndarray],
    node_ids: dict[str, str],
    source_m: np.ndarray,
    node_uncertainty: dict[str, float],
    node_clock_offset_s: dict[str, float] | None = None,
) -> list[dict]:
    """Pairs as the sidecar emits them: 0 on one clock, quadrature across two."""
    clock_offset_s = node_clock_offset_s or {}
    pairs = []
    for (index_a, sensor_a), (index_b, sensor_b) in itertools.combinations(enumerate(ordered_sensor_ids), 2):
        node_a, node_b = node_ids[sensor_a], node_ids[sensor_b]
        cross_node = node_a != node_b
        pairs.append(
            {
                "ch_a": index_a,
                "ch_b": index_b,
                "lag_seconds": _arrival_lag_seconds(source_m, positions[sensor_a], positions[sensor_b])
                + clock_offset_s.get(node_a, 0.0)
                - clock_offset_s.get(node_b, 0.0),
                "confidence": CORRELATION_PEAK,
                "is_cross_node": cross_node,
                "timing_uncertainty_seconds": (
                    cross_node_pair_timing_uncertainty_seconds(node_uncertainty[node_a], node_uncertainty[node_b])
                    if cross_node
                    else 0.0
                ),
            }
        )
    return pairs


# None = cluster-unaware path; every declared grade must leave the weights at
# the sidecar's, which never receives the declared grade.
@pytest.mark.parametrize("cluster_declared_sync_grade", [None, *SyncGrade])
@pytest.mark.asyncio
async def test_same_pair_set_gets_the_same_weights_on_both_paths(cluster_declared_sync_grade: SyncGrade | None) -> None:
    ordered_sensor_ids, positions, node_ids = _sensor_layout()
    source_m = np.asarray([70.0, 40.0, 5.0])
    windows = {sensor_id: np.full(8, float(index)) for index, sensor_id in enumerate(ordered_sensor_ids)}

    def exact_gcc_phat(*, signal, reference_signal, **_kwargs):
        sensor_a = ordered_sensor_ids[int(signal[0])]
        sensor_b = ordered_sensor_ids[int(reference_signal[0])]
        return _arrival_lag_seconds(source_m, positions[sensor_a], positions[sensor_b]), CORRELATION_PEAK

    python_ingest = measure_pair_tdoas(
        sensor_positions=positions,
        sensor_windows=windows,
        sensor_ids=ordered_sensor_ids,
        sample_rate_hz=SAMPLE_RATE_HZ,
        sound_speed_mps=SOUND_SPEED_MPS,
        max_tau_s=0.35,
        interpolation_factor=INTERPOLATION_FACTOR,
        sensor_timing_uncertainty_seconds=await _registry_sensor_uncertainty(cluster_declared_sync_grade),
        gcc_phat_function=exact_gcc_phat,
        sensor_node_ids=node_ids,
        cross_node_max_tau_s=0.35,
    )
    rust_handoff, _ = pair_measurements_from_rust_tdoas(
        ordered_sensor_ids=ordered_sensor_ids,
        pair_tdoas=_rust_pair_tdoas(ordered_sensor_ids, positions, node_ids, source_m, _rust_node_uncertainty()),
        reference_time_std_seconds=solver_reference_time_std_seconds(SAMPLE_RATE_HZ, INTERPOLATION_FACTOR),
    )

    python_weights = {(m.sensor_a, m.sensor_b): m.weight for m in python_ingest}
    rust_weights = {(m.sensor_a, m.sensor_b): m.weight for m in rust_handoff}
    assert len(python_weights) == math.comb(len(ordered_sensor_ids), 2)
    assert python_weights.keys() == rust_weights.keys()
    for pair_id, weight in python_weights.items():
        assert weight == pytest.approx(rust_weights[pair_id], rel=1e-12), pair_id
    # Same-node pairs carry the correlation quality alone; every cross-node pair
    # is present but weaker (no pair excluded by clock class).
    tetra_pair = ("tetra-a:ch0", "tetra-a:ch1")
    assert python_weights[tetra_pair] == pytest.approx(normalized_pair_quality(CORRELATION_PEAK))
    assert all(weight < 1.0e-3 for (a, b), weight in python_weights.items() if node_ids[a] != node_ids[b])


def test_old_sidecar_pairs_without_timing_keep_pre_p9_weights_and_solve() -> None:
    ordered_sensor_ids, positions, node_ids = _sensor_layout()
    source_m = np.asarray([70.0, 40.0, 5.0])
    current_pairs = _rust_pair_tdoas(ordered_sensor_ids, positions, node_ids, source_m, _rust_node_uncertainty())
    old_sidecar_pairs = [
        {key: value for key, value in pair.items() if key not in ("is_cross_node", "timing_uncertainty_seconds")}
        for pair in current_pairs
    ]
    measurements, _ = pair_measurements_from_rust_tdoas(
        ordered_sensor_ids=ordered_sensor_ids,
        pair_tdoas=old_sidecar_pairs,
        reference_time_std_seconds=solver_reference_time_std_seconds(SAMPLE_RATE_HZ, INTERPOLATION_FACTOR),
    )
    assert [m.weight for m in measurements] == [max(0.01, normalized_pair_quality(CORRELATION_PEAK))] * len(measurements)

    def solve(pairs: list[dict]):
        return solve_localization_from_rust_tdoas(
            sensor_positions=positions,
            ordered_sensor_ids=ordered_sensor_ids,
            pair_tdoas=pairs,
            steering_direction=None,
            sound_speed_mps=SOUND_SPEED_MPS,
            sample_rate_hz=SAMPLE_RATE_HZ,
            interpolation_factor=INTERPOLATION_FACTOR,
            far_field_default_range_m=50.0,
        )

    explicit_none = [dict(pair, timing_uncertainty_seconds=None) for pair in old_sidecar_pairs]
    old_sidecar_solve = solve(old_sidecar_pairs)
    assert old_sidecar_solve.position_m == solve(explicit_none).position_m
    assert old_sidecar_solve.position_covariance_m2 == solve(explicit_none).position_covariance_m2


def test_rust_handoff_resolve_lets_gps_locked_pairs_dominate_and_widens_covariance() -> None:
    """point-d's clock is really 20 ms off. With its pairs weighted by its
    free-running uncertainty the GPS-locked pairs decide the fix; trusting it
    as locked does not. The covariance widens whenever weak clocks are present."""
    ordered_sensor_ids, positions, node_ids = _sensor_layout()
    source_m = np.asarray([70.0, 40.0, 5.0])
    free_running_d = {node_id: 1.0e-6 for node_id in NODE_ORIGINS_M} | {"point-d": 5.0e-2}
    all_locked = {node_id: 1.0e-6 for node_id in NODE_ORIGINS_M}

    def solve(node_uncertainty: dict[str, float], node_clock_offset_s: dict[str, float]):
        return solve_localization_from_rust_tdoas(
            sensor_positions=positions,
            ordered_sensor_ids=ordered_sensor_ids,
            pair_tdoas=_rust_pair_tdoas(
                ordered_sensor_ids, positions, node_ids, source_m, node_uncertainty, node_clock_offset_s
            ),
            steering_direction=None,
            sound_speed_mps=SOUND_SPEED_MPS,
            sample_rate_hz=SAMPLE_RATE_HZ,
            interpolation_factor=INTERPOLATION_FACTOR,
            far_field_default_range_m=50.0,
        )

    def error_m(result) -> float:
        return float(np.linalg.norm(np.asarray(result.position_m) - source_m))

    def covariance_trace(result) -> float:
        return float(np.trace(np.asarray(result.position_covariance_m2)))

    offset = {"point-d": 0.02}
    weighted = solve(free_running_d, offset)
    trusted_as_locked = solve(all_locked, offset)
    assert error_m(weighted) < 1.0
    assert error_m(trusted_as_locked) > 10.0 * max(error_m(weighted), 0.1)

    assert covariance_trace(solve(free_running_d, {})) > covariance_trace(solve(all_locked, {}))


def test_rust_and_python_clock_class_defaults_match() -> None:
    rust_source = RUST_SOLVE_CONFIG_PATH.read_text()
    class_json = re.search(r'const DEFAULT_CLOCK_CLASS_UNCERTAINTY_JSON: &str =\s*r#"(\{.*?\})"#;', rust_source)
    assert class_json is not None, "DEFAULT_CLOCK_CLASS_UNCERTAINTY_JSON not found in localization_solve_config.rs"
    assert json.loads(class_json.group(1)) == overhaul_defaults.default_localization_clock_class_uncertainty_seconds()
    assert json.loads(class_json.group(1)) == Settings().localization_clock_class_uncertainty_seconds

    unknown_default = re.search(r"localization_clock_unknown_uncertainty_seconds: ([0-9.eE+-]+),", rust_source)
    unknown_cli_default = re.search(
        r'env = "MINIMAPPR_LOCALIZATION_CLOCK_UNKNOWN_UNCERTAINTY_SECONDS",\s*default_value_t = ([0-9.eE+-]+)',
        rust_source,
    )
    assert unknown_default is not None and unknown_cli_default is not None
    for rust_literal in (unknown_default.group(1), unknown_cli_default.group(1)):
        assert float(rust_literal) == overhaul_defaults.DEFAULT_LOCALIZATION_CLOCK_UNKNOWN_UNCERTAINTY_SECONDS


def test_clock_class_defaults_name_every_current_time_quality() -> None:
    """Every class a node can report has a default; the legacy ``ntp_sync``
    value is an alias of ``ntp_disciplined`` on both sides, not its own class."""
    current_time_qualities = {quality.value for quality in TimeQuality} - {TimeQuality.NTP_SYNC.value}
    assert set(overhaul_defaults.default_localization_clock_class_uncertainty_seconds()) == current_time_qualities


def test_snapshot_handoff_resolve_passes_pair_timing_uncertainty_to_the_re_solve() -> None:
    """The python_cartesian handoff re-solve reads each admitted pair's
    ``timing_uncertainty_seconds`` from the manifest: a free-running point node
    widens the published covariance without moving an unbiased fix."""
    from minimappr.core.localized_handoff_solution import SnapshotGeometrySolutionResolver
    from tests.localization_solve_manifest_fixtures import (
        PLANAR_NODE_ID,
        POINT_NODE_ID,
        mixed_cluster_solve_localization_payload,
        planar_node_payload,
    )
    from tests.test_localized_handoff_geometry_and_solve import (
        SITE_FRAME_ID,
        SOURCE_POSITION_M,
        _request,
        _resolve,
        _solution_host,
    )

    def resolve(cross_node_timing_uncertainty_s: float):
        localization = mixed_cluster_solve_localization_payload(
            arrival_order=(PLANAR_NODE_ID, POINT_NODE_ID),
            source_position_m=SOURCE_POSITION_M,
            frame_id=SITE_FRAME_ID,
            reporting_node_id=PLANAR_NODE_ID,
            sidecar_position_m=(0.0, 0.0, 0.0),
        )
        for pair in localization["pair_tdoas"]:
            pair["timing_uncertainty_seconds"] = cross_node_timing_uncertainty_s if pair["is_cross_node"] else 0.0
        request = _request(localization, reporter_payload=planar_node_payload())
        return SnapshotGeometrySolutionResolver(_solution_host("python_cartesian")).resolve_handoff_solution(
            payload=request, geometry=_resolve(request), reporting_modality="localized"
        )

    locked = resolve(cross_node_pair_timing_uncertainty_seconds(1.0e-6, 1.0e-6))
    free_running_point = resolve(cross_node_pair_timing_uncertainty_seconds(1.0e-6, 5.0e-2))
    assert free_running_point.method == "python_cartesian_rust_tdoa"
    assert math.dist(free_running_point.position_m, SOURCE_POSITION_M) < 0.5
    assert np.trace(np.asarray(free_running_point.position_covariance_m2)) > 10.0 * np.trace(
        np.asarray(locked.position_covariance_m2)
    )


def test_point_only_network_on_weak_clocks_keeps_the_fix_and_widens_covariance() -> None:
    """Every pair clock-dominated (point nodes on NTP): a uniform weight change
    must not move the least-squares fix, only widen its covariance. Before the
    optimizer rescale, scipy stopped at its start point on such tiny residuals."""
    positions = {
        f"{node_id}:ch0": np.asarray(origin, dtype=np.float64)
        for node_id, origin in {
            "p0": (0.0, 0.0, 1.5),
            "p1": (50.0, 0.0, 2.0),
            "p2": (0.0, 50.0, 1.0),
            "p3": (50.0, 50.0, 2.5),
            "p4": (25.0, 25.0, 8.0),
        }.items()
    }
    ordered_sensor_ids = list(positions)
    node_ids = {sensor_id: sensor_id.split(":")[0] for sensor_id in ordered_sensor_ids}
    source_m = np.asarray([30.0, 20.0, 3.0])
    rng = np.random.default_rng(3)
    node_clock_offset_s = {node_id: float(rng.normal(0.0, 0.002)) for node_id in node_ids.values()}

    def solve(node_uncertainty_s: float):
        return solve_localization_from_rust_tdoas(
            sensor_positions=positions,
            ordered_sensor_ids=ordered_sensor_ids,
            pair_tdoas=_rust_pair_tdoas(
                ordered_sensor_ids,
                positions,
                node_ids,
                source_m,
                {node_id: node_uncertainty_s for node_id in node_ids.values()},
                node_clock_offset_s,
            ),
            steering_direction=None,
            sound_speed_mps=SOUND_SPEED_MPS,
            sample_rate_hz=SAMPLE_RATE_HZ,
            interpolation_factor=INTERPOLATION_FACTOR,
            far_field_default_range_m=50.0,
        )

    unweighted = solve(0.0)
    ntp_weighted = solve(DEFAULT_POLICY.class_uncertainty_for(TimeQuality.NTP_DISCIPLINED.value))
    np.testing.assert_allclose(ntp_weighted.position_m, unweighted.position_m, atol=1.0e-3)
    # The unweighted solve already widens through its (clock-scattered) residual;
    # the weighted one must state at least that much from the clock model.
    assert np.trace(np.asarray(ntp_weighted.position_covariance_m2)) > np.trace(
        np.asarray(unweighted.position_covariance_m2)
    )
