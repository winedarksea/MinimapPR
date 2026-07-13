"""Deterministic reliability matrix for production cross-node GCC-PHAT localization."""

from __future__ import annotations

import numpy as np
import pytest

from minimappr.core.localization import LocalizationEngine, gcc_phat
from minimappr.core.localization_dispatch import LocalizationDispatcher
from minimappr.core.tdoa_measurements import measure_pair_tdoas
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.models import GeoPoint
from tests.helpers import CrossNodeSyntheticScene, synthesize_cross_node_scene


SAMPLE_RATE_HZ = 48_000
SOUND_SPEED_MPS = 343.2
WIDE_NODES_M = {
    "tetra-a": (0.0, 0.0, 1.5),
    "point-b": (50.0, 0.0, 2.0),
    "point-c": (0.0, 50.0, 1.0),
    "point-d": (50.0, 50.0, 2.5),
}


def _source_at_range(distance_m: float) -> np.ndarray:
    centroid = np.mean(np.asarray(list(WIDE_NODES_M.values()), dtype=np.float64), axis=0)
    direction = np.asarray([0.6, 0.7, 0.39], dtype=np.float64)
    return centroid + direction / np.linalg.norm(direction) * distance_m


def _localize(scene: CrossNodeSyntheticScene, *, sensor_weights=None, min_sync=0.25):
    return LocalizationEngine(
        max_tau_s=0.02,
        interp_factor=8,
        cross_node_max_tau_s=0.35,
        cross_node_min_sync_weight=min_sync,
    ).localize(
        sensor_positions=scene.reported_sensor_positions_m,
        sensor_windows=scene.sensor_windows,
        sample_rate_hz=SAMPLE_RATE_HZ,
        temperature_c=20.0,
        humidity_fraction=0.5,
        sensor_weights=sensor_weights,
        sensor_node_ids=scene.sensor_node_ids,
    )


@pytest.mark.parametrize("distance_m", [2.0, 10.0, 100.0, 300.0, 1000.0])
def test_clean_gps_pps_wide_array_distance_matrix(distance_m: float) -> None:
    scene = synthesize_cross_node_scene(
        source_position_m=_source_at_range(distance_m),
        node_origins_m=WIDE_NODES_M,
        tetra_node_ids=("tetra-a",),
        seed=19,
    )

    result = _localize(scene)
    error_m = float(np.linalg.norm(np.asarray(result.position_m) - scene.source_position_m))
    assert error_m <= max(5.0, 0.10 * distance_m)
    covariance = np.asarray(result.position_covariance_m2, dtype=np.float64)
    assert covariance.shape == (3, 3)
    assert np.all(np.isfinite(covariance))
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-9


@pytest.mark.parametrize(
    "node_origins_m,tetra_node_ids",
    [
        ({"tetra-a": (0.0, 0.0, 1.5), "tetra-b": (30.0, 0.0, 1.5)}, ("tetra-a", "tetra-b")),
        (
            {
                "tetra-a": (0.0, 0.0, 1.5),
                "tetra-b": (30.0, 0.0, 1.5),
                "tetra-c": (15.0, 25.0, 1.5),
            },
            ("tetra-a", "tetra-b", "tetra-c"),
        ),
    ],
)
def test_two_and_three_node_scenes_localize_with_cross_node_tdoa(
    node_origins_m: dict[str, tuple[float, float, float]],
    tetra_node_ids: tuple[str, ...],
) -> None:
    centroid = np.mean(np.asarray(list(node_origins_m.values()), dtype=np.float64), axis=0)
    scene = synthesize_cross_node_scene(
        source_position_m=centroid + np.asarray([40.0, 80.0, 10.0]),
        node_origins_m=node_origins_m,
        tetra_node_ids=tetra_node_ids,
        seed=17,
    )
    result = _localize(scene)
    assert float(np.linalg.norm(np.asarray(result.position_m) - scene.source_position_m)) < 20.0


@pytest.mark.parametrize(
    ("node_origins_m", "max_error_m"),
    [
        (
            {
                "point-a": (0.0, 0.0, 0.0),
                "point-b": (30.0, 0.0, 0.0),
                "point-c": (0.0, 30.0, 0.0),
            },
            2.0,
        ),
        (
            {
                "point-a": (0.0, 0.0, 0.0),
                "point-b": (30.0, 0.0, 0.0),
                "point-c": (0.0, 30.0, 0.0),
                "point-d": (30.0, 30.0, 0.0),
            },
            1.0,
        ),
    ],
)
def test_three_and_four_single_microphone_nodes_localize_in_2d(
    node_origins_m: dict[str, tuple[float, float, float]],
    max_error_m: float,
) -> None:
    """Point-only arrays use the production 2D fallback, not tetrahedral 3D solve."""
    scene = synthesize_cross_node_scene(
        source_position_m=(12.0, 9.0, 0.0),
        node_origins_m=node_origins_m,
        seed=41,
        additive_noise_std=0.03,
    )
    dispatcher = LocalizationDispatcher(
        algorithms={"gcc_phat": LocalizationEngine(max_tau_s=0.35, interp_factor=8)}
    )
    result = dispatcher.localize_2d(
        sensor_positions=scene.reported_sensor_positions_m,
        sensor_windows=scene.sensor_windows,
        sample_rate_hz=SAMPLE_RATE_HZ,
        temperature_c=20.0,
        humidity_fraction=0.5,
        fixed_z_m=0.0,
    )

    assert result.resolved_algorithm == "gcc_phat"
    assert float(np.linalg.norm(np.asarray(result.position_m) - scene.source_position_m)) < max_error_m
    covariance = np.asarray(result.position_covariance_m2, dtype=np.float64)
    assert np.all(np.isfinite(covariance))
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-9


def test_rust_auto_gps_flat_geometry_matches_server_coordinate_frame() -> None:
    """Rust's equirectangular auto-GPS frame must agree with server geometry locally."""
    origin = GeoPoint(lat=45.0, lon=-93.0, alt_m=2.0)
    frame = LocalCoordinateFrame(origin=origin, mode="flat")
    reported = GeoPoint(lat=45.0000180, lon=-92.9999746, alt_m=5.0)
    python_position = np.asarray(frame.geo_to_local(reported), dtype=np.float64)
    # Mirrors the Rust auto-GPS conversion for this local (<3 m) geometry.
    earth_radius_m = 6_371_000.0
    east_m = np.deg2rad(reported.lon - origin.lon) * earth_radius_m * np.cos(np.deg2rad(origin.lat))
    north_m = np.deg2rad(reported.lat - origin.lat) * earth_radius_m
    rust_position = np.asarray([east_m, north_m, reported.alt_m - origin.alt_m])
    assert np.linalg.norm(python_position - rust_position) < 0.02


def test_free_running_node_has_no_cross_node_measurements() -> None:
    scene = synthesize_cross_node_scene(
        source_position_m=_source_at_range(100.0),
        node_origins_m=WIDE_NODES_M,
        tetra_node_ids=("tetra-a",),
        seed=23,
    )
    weights = {
        sensor_id: 0.05 if node_id == "point-d" else 1.0
        for sensor_id, node_id in scene.sensor_node_ids.items()
    }
    measurements = measure_pair_tdoas(
        sensor_positions=scene.reported_sensor_positions_m,
        sensor_windows=scene.sensor_windows,
        sensor_ids=sorted(scene.sensor_windows),
        sample_rate_hz=SAMPLE_RATE_HZ,
        sound_speed_mps=SOUND_SPEED_MPS,
        max_tau_s=0.02,
        interpolation_factor=8,
        sensor_weights=weights,
        gcc_phat_function=gcc_phat,
        sensor_node_ids=scene.sensor_node_ids,
        cross_node_max_tau_s=0.35,
        cross_node_min_sync_weight=0.25,
    )
    assert all(
        "point-d" not in (measurement.sensor_a, measurement.sensor_b)
        or scene.sensor_node_ids[measurement.sensor_a] == scene.sensor_node_ids[measurement.sensor_b]
        for measurement in measurements
    )


@pytest.mark.parametrize(
    "fault_kwargs",
    [
        {"additive_noise_std": 0.05},
        {"reflection_delay_seconds": 0.006, "reflection_gain": 0.5},
        {"node_clock_offsets_s": {"point-b": 0.001}},
        {"node_position_errors_m": {"point-b": (3.0, 0.0, 0.0)}},
    ],
)
def test_degraded_scenes_remain_finite_and_do_not_claim_high_confidence(fault_kwargs: dict) -> None:
    scene = synthesize_cross_node_scene(
        source_position_m=_source_at_range(100.0),
        node_origins_m=WIDE_NODES_M,
        tetra_node_ids=("tetra-a",),
        seed=29,
        **fault_kwargs,
    )
    result = _localize(scene)
    covariance = np.asarray(result.position_covariance_m2, dtype=np.float64)
    assert np.all(np.isfinite(np.asarray(result.position_m)))
    assert np.all(np.isfinite(covariance))
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-9
    assert result.confidence < 0.20


def test_missing_cross_node_audio_cannot_produce_a_precise_fix() -> None:
    scene = synthesize_cross_node_scene(
        source_position_m=_source_at_range(100.0),
        node_origins_m=WIDE_NODES_M,
        tetra_node_ids=("tetra-a",),
        seed=31,
        missing_sensor_ids={"point-b:ch0", "point-c:ch0", "point-d:ch0"},
    )
    result = _localize(scene)
    error_m = float(np.linalg.norm(np.asarray(result.position_m) - scene.source_position_m))
    assert result.confidence == 0.0 or error_m > 20.0
