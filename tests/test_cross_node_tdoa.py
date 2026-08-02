"""Phase 5: true cross-node TDOA (tier c).

Synthetic, sample-accurate multi-node scenes exercising the widened cross-node tau
ceiling, the sync gate, and geometry weighting in ``measure_pair_tdoas`` +
``solve_cartesian_tdoa``.
"""

from __future__ import annotations

import numpy as np
import pytest

from minimappr.core.localization import LocalizationEngine, gcc_phat
from minimappr.core.tdoa_measurements import measure_pair_tdoas

from tests.helpers import synthesize_multinode_windows

SAMPLE_RATE_HZ = 48_000
SOUND_SPEED_MPS = 343.2


def _broadband(seed: int, duration_s: float = 0.4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.3, size=int(duration_s * SAMPLE_RATE_HZ)).astype(np.float32)


def _localize(source_m, nodes, *, cross_node_tau, sensor_weights=None, min_sync=None):
    positions, windows, node_ids = synthesize_multinode_windows(
        _broadband(3),
        SAMPLE_RATE_HZ,
        source_position_m=source_m,
        node_origins_m=nodes,
        tetra_node_ids=tuple(nodes),
        sound_speed_mps=SOUND_SPEED_MPS,
    )
    engine = LocalizationEngine(
        max_tau_s=0.02,
        interp_factor=8,
        cross_node_max_tau_s=cross_node_tau,
        cross_node_min_sync_weight=min_sync,
    )
    result = engine.localize(
        positions, windows, SAMPLE_RATE_HZ, 20.0, 0.5,
        sensor_weights=sensor_weights, sensor_node_ids=node_ids,
    )
    centroid = np.mean(np.vstack(list(positions.values())), axis=0)
    return result, centroid


def test_cross_node_tau_lift_unclamps_wide_baseline_lag() -> None:
    """The core tier-c enabler: a wide-baseline cross-node pair's lag exceeds the
    tight same-node 0.02 s cap. Without the cross-node ceiling the measured lag is
    clamped; with it, the true (large) geometric lag is recovered."""
    # Two single-mic nodes 30 m apart; source well off broadside so the cross-node
    # lag is large (~0.077 s), far beyond the 0.02 s same-node cap.
    oa = np.array([0.0, 0.0, 0.0])
    ob = np.array([30.0, 0.0, 0.0])
    source = np.array([200.0, 100.0, 0.0])
    true_lag_s = (
        float(np.linalg.norm(source - oa)) - float(np.linalg.norm(source - ob))
    ) / SOUND_SPEED_MPS
    assert abs(true_lag_s) > 0.02  # exceeds the same-node cap

    positions, windows, node_ids = synthesize_multinode_windows(
        _broadband(5),
        SAMPLE_RATE_HZ,
        source_position_m=source,
        node_origins_m={"a": tuple(oa), "b": tuple(ob)},
        sound_speed_mps=SOUND_SPEED_MPS,
    )
    common = dict(
        sensor_positions=positions,
        sensor_windows=windows,
        sensor_ids=sorted(positions),
        sample_rate_hz=SAMPLE_RATE_HZ,
        sound_speed_mps=SOUND_SPEED_MPS,
        max_tau_s=0.02,
        interpolation_factor=8,
        sensor_weights=None,
        gcc_phat_function=gcc_phat,
        sensor_node_ids=node_ids,
    )
    clamped = measure_pair_tdoas(**common)
    lifted = measure_pair_tdoas(**common, cross_node_max_tau_s=0.35)

    clamped_lag = abs(clamped[0].tdoa_seconds)
    lifted_lag = abs(lifted[0].tdoa_seconds)
    # Clamped cannot exceed the 0.02 s cap; lifted recovers the true ~0.077 s lag.
    assert clamped_lag <= 0.02 + 1e-3
    assert abs(lifted_lag - abs(true_lag_s)) < 0.01
    assert lifted_lag > clamped_lag


def test_two_node_tetra_cluster_localizes_100m_source() -> None:
    """Two tetra nodes 30 m apart localize a 100 m source (bearing intersection +
    cross-node TDOA) to within the cluster's expected accuracy."""
    nodes = {"tetra-a": (0.0, 0.0, 0.0), "tetra-b": (30.0, 0.0, 0.0)}
    source = np.array([80.0, 100.0, 0.0])
    result, _ = _localize(source, nodes, cross_node_tau=0.35)
    err = float(np.linalg.norm(np.asarray(result.position_m) - source))
    assert err < 20.0, f"position error {err:.1f} m"


def test_three_node_cross_node_tdoa_range_error_under_10pct() -> None:
    nodes = {
        "tetra-a": (0.0, 0.0, 0.0),
        "tetra-b": (100.0, 0.0, 0.0),
        "tetra-c": (50.0, 60.0, 0.0),
    }
    source = np.array([50.0, 300.0, 0.0])
    result, centroid = _localize(source, nodes, cross_node_tau=0.35)
    pos = np.asarray(result.position_m, dtype=np.float64)
    true_range = float(np.linalg.norm(source - centroid))
    est_range = float(np.linalg.norm(pos - centroid))
    assert abs(est_range - true_range) / true_range < 0.10


def test_ntp_nodes_excluded_at_min_sync_weight_one() -> None:
    """With min_sync_weight=1.0 (PPS/PTP-only), a node whose sensors carry a sub-1.0
    sync weight is excluded from cross-node pairs."""
    positions = {
        "a:ch0": np.array([0.0, 0.0, 0.0]),
        "a:ch1": np.array([0.05, 0.0, 0.0]),
        "a:ch2": np.array([0.0, 0.05, 0.0]),
        "a:ch3": np.array([0.0, 0.0, 0.05]),
        "b:ch0": np.array([30.0, 0.0, 0.0]),
    }
    node_ids = {sid: sid.split(":")[0] for sid in positions}
    windows = {sid: _broadband(1) for sid in positions}
    # Node b is NTP-grade (sync weight 0.5); node a is PPS (1.0).
    weights = {sid: (0.5 if sid.startswith("b") else 1.0) for sid in positions}

    excluded = measure_pair_tdoas(
        sensor_positions=positions,
        sensor_windows=windows,
        sensor_ids=sorted(positions),
        sample_rate_hz=SAMPLE_RATE_HZ,
        sound_speed_mps=SOUND_SPEED_MPS,
        max_tau_s=0.02,
        interpolation_factor=4,
        sensor_weights=weights,
        gcc_phat_function=gcc_phat,
        sensor_node_ids=node_ids,
        cross_node_max_tau_s=0.35,
        cross_node_min_sync_weight=1.0,
    )
    # No pair should straddle nodes a and b.
    assert not any(
        node_ids[m.sensor_a] != node_ids[m.sensor_b] for m in excluded
    )

    # At the default 0.25 gate the cross-node pair is admitted.
    admitted = measure_pair_tdoas(
        sensor_positions=positions,
        sensor_windows=windows,
        sensor_ids=sorted(positions),
        sample_rate_hz=SAMPLE_RATE_HZ,
        sound_speed_mps=SOUND_SPEED_MPS,
        max_tau_s=0.02,
        interpolation_factor=4,
        sensor_weights=weights,
        gcc_phat_function=gcc_phat,
        sensor_node_ids=node_ids,
        cross_node_max_tau_s=0.35,
        cross_node_min_sync_weight=0.25,
    )
    assert any(node_ids[m.sensor_a] != node_ids[m.sensor_b] for m in admitted)


def test_node_position_std_inflates_downweights_cross_node_pair() -> None:
    """A poorly-surveyed node (σ=5 m) contributes a lower-weighted cross-node pair."""
    positions = {
        "a:ch0": np.array([0.0, 0.0, 0.0]),
        "b:ch0": np.array([30.0, 0.0, 0.0]),
        "a:ch1": np.array([0.05, 0.0, 0.0]),
        "b:ch1": np.array([30.05, 0.0, 0.0]),
    }
    node_ids = {sid: sid.split(":")[0] for sid in positions}
    windows = {sid: _broadband(2) for sid in positions}

    def _cross_pair_weight(node_std):
        pairs = measure_pair_tdoas(
            sensor_positions=positions,
            sensor_windows=windows,
            sensor_ids=sorted(positions),
            sample_rate_hz=SAMPLE_RATE_HZ,
            sound_speed_mps=SOUND_SPEED_MPS,
            max_tau_s=0.02,
            interpolation_factor=4,
            sensor_weights=None,
            gcc_phat_function=gcc_phat,
            sensor_node_ids=node_ids,
            cross_node_max_tau_s=0.35,
            node_position_std_m=node_std,
        )
        cross = [m for m in pairs if node_ids[m.sensor_a] != node_ids[m.sensor_b]]
        return max(m.weight for m in cross)

    surveyed = _cross_pair_weight({sid: 0.0 for sid in positions})
    uncertain = _cross_pair_weight({sid: 5.0 for sid in positions})
    assert uncertain < surveyed


def test_build_localizer_from_localization_config_carries_cross_node_settings() -> None:
    """Regression for the silent config drop: production builds the localizer
    from a LocalizationConfig (runtime_bootstrap), and until these fields were
    carried onto it, getattr fallbacks returned None — the cross-node tau lift
    never reached the live engine, so a 7.9 m baseline was clipped by the
    0.02 s (~6.9 m) same-node cap."""
    from minimappr.config import Settings
    from minimappr.core.localization_dispatch import build_localizer_from_settings

    cfg = Settings().localization_config()
    assert cfg.localization_cross_node_tdoa_enabled is True
    dispatcher = build_localizer_from_settings(cfg)
    engine = dispatcher.algorithms["gcc_phat"] if hasattr(dispatcher, "algorithms") else dispatcher
    gcc = getattr(dispatcher, "_algorithms", None)
    # Locate the gcc engine regardless of dispatcher internals.
    candidates = [engine]
    if isinstance(gcc, dict):
        candidates.append(gcc.get("gcc_phat"))
    resolved = next(
        (c for c in candidates if isinstance(c, LocalizationEngine)),
        None,
    )
    assert resolved is not None, "gcc_phat LocalizationEngine not found on dispatcher"
    assert resolved.cross_node_max_tau_s == pytest.approx(0.35)
    assert resolved.cross_node_min_sync_weight == pytest.approx(0.25)


def test_pair_measurement_stats_count_cross_node_activity() -> None:
    """PairMeasurementStats must count measured cross-node pairs and sync-gate
    rejections — the FusionMetrics counters these feed read 0 forever before."""
    from minimappr.core.tdoa_measurements import PairMeasurementStats

    positions, windows, node_ids = synthesize_multinode_windows(
        _broadband(11),
        SAMPLE_RATE_HZ,
        source_position_m=(6.0, 3.0, 1.0),
        node_origins_m={"node-a": (0.0, 0.0, 0.0), "node-b": (12.0, 0.0, 0.0)},
        tetra_node_ids=("node-a", "node-b"),
        sound_speed_mps=SOUND_SPEED_MPS,
    )
    stats = PairMeasurementStats()
    measurements = measure_pair_tdoas(
        sensor_positions=positions,
        sensor_windows=windows,
        sensor_ids=sorted(positions),
        sample_rate_hz=SAMPLE_RATE_HZ,
        sound_speed_mps=SOUND_SPEED_MPS,
        max_tau_s=0.02,
        interpolation_factor=4,
        sensor_weights=None,
        gcc_phat_function=gcc_phat,
        sensor_node_ids=node_ids,
        cross_node_max_tau_s=0.35,
        stats=stats,
    )
    assert measurements
    assert stats.cross_node_pairs_measured > 0
    assert stats.cross_node_pairs_rejected_sync == 0

    # Sub-gate sync weights on one node reject every cross-node pair.
    gated_stats = PairMeasurementStats()
    weights = {sid: (0.05 if node_ids[sid] == "node-b" else 1.0) for sid in positions}
    measure_pair_tdoas(
        sensor_positions=positions,
        sensor_windows=windows,
        sensor_ids=sorted(positions),
        sample_rate_hz=SAMPLE_RATE_HZ,
        sound_speed_mps=SOUND_SPEED_MPS,
        max_tau_s=0.02,
        interpolation_factor=4,
        sensor_weights=weights,
        gcc_phat_function=gcc_phat,
        sensor_node_ids=node_ids,
        cross_node_max_tau_s=0.35,
        cross_node_min_sync_weight=0.25,
        stats=gated_stats,
    )
    assert gated_stats.cross_node_pairs_rejected_sync > 0
    assert gated_stats.cross_node_pairs_measured == 0


def test_engine_localize_surfaces_cross_node_pair_stats() -> None:
    """LocalizationResult carries the pair stats when node ids are supplied."""
    positions, windows, node_ids = synthesize_multinode_windows(
        _broadband(13),
        SAMPLE_RATE_HZ,
        source_position_m=(6.0, 3.0, 1.0),
        node_origins_m={"node-a": (0.0, 0.0, 0.0), "node-b": (12.0, 0.0, 0.0)},
        tetra_node_ids=("node-a", "node-b"),
        sound_speed_mps=SOUND_SPEED_MPS,
    )
    engine = LocalizationEngine(max_tau_s=0.02, interp_factor=8, cross_node_max_tau_s=0.35)
    result = engine.localize(
        positions,
        windows,
        SAMPLE_RATE_HZ,
        20.0,
        0.5,
        sensor_node_ids=node_ids,
    )
    assert result.cross_node_pairs_measured is not None
    assert result.cross_node_pairs_measured > 0
    assert result.cross_node_pairs_rejected_sync == 0


def test_localize_2d_is_node_aware() -> None:
    """Phase 5: the 2D tier — ~98% of a compact-array site's solves — was
    node-blind. It must (a) lift the tau ceiling for cross-node pairs so a wide
    baseline isn't clipped by the same-node cap, (b) apply the cross-node sync
    gate, and (c) stamp range-mode provenance so bearing-fusion gate G3 can
    engage on 2D solves."""
    positions, windows, node_ids = synthesize_multinode_windows(
        _broadband(17),
        SAMPLE_RATE_HZ,
        source_position_m=(6.0, 5.0, 0.0),
        node_origins_m={"node-a": (0.0, 0.0, 0.0), "node-b": (12.0, 0.0, 0.0)},
        tetra_node_ids=("node-a", "node-b"),
        sound_speed_mps=SOUND_SPEED_MPS,
    )
    engine = LocalizationEngine(
        max_tau_s=0.002,  # ~0.7 m same-node cap: would clip the 12 m baseline
        interp_factor=8,
        cross_node_max_tau_s=0.35,
        cross_node_min_sync_weight=0.25,
    )
    result = engine.localize_2d(
        positions,
        windows,
        SAMPLE_RATE_HZ,
        20.0,
        0.5,
        sensor_node_ids=node_ids,
    )
    assert result.cross_node_pairs_measured is not None
    assert result.cross_node_pairs_measured > 0
    # (c) range-mode provenance for bearing-fusion gate G3.
    assert result.range_projection_mode == "range_refined"
    assert result.range_observability is not None
    assert 0.0 <= result.range_observability <= 1.0


def test_localize_2d_sync_gate_excludes_unsynchronized_cross_node_sensors() -> None:
    positions, windows, node_ids = synthesize_multinode_windows(
        _broadband(19),
        SAMPLE_RATE_HZ,
        source_position_m=(6.0, 5.0, 0.0),
        node_origins_m={"node-a": (0.0, 0.0, 0.0), "node-b": (12.0, 0.0, 0.0)},
        tetra_node_ids=("node-a", "node-b"),
        sound_speed_mps=SOUND_SPEED_MPS,
    )
    engine = LocalizationEngine(
        max_tau_s=0.02,
        interp_factor=8,
        cross_node_max_tau_s=0.35,
        cross_node_min_sync_weight=0.25,
    )
    # node-b sensors carry a free-running-clock sync weight below the gate.
    weights = {sid: (0.05 if node_ids[sid] == "node-b" else 1.0) for sid in positions}
    result = engine.localize_2d(
        positions,
        windows,
        SAMPLE_RATE_HZ,
        20.0,
        0.5,
        sensor_weights=weights,
        sensor_node_ids=node_ids,
    )
    assert result.cross_node_pairs_rejected_sync is not None
    assert result.cross_node_pairs_rejected_sync > 0
    assert result.cross_node_pairs_measured == 0
    # Only node-a sensors contribute measurements (reference is on node-a: it
    # has the higher weights and identical RMS).
    assert all(node_ids[sid] == node_ids[result.reference_sensor] for sid in result.tdoa_s)
