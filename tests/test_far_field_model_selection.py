"""Compact-array near-field collapse fix (Field Mode W2 step 0).

Before the fix, a single 5 cm tetra's Cartesian TDOA solve landed a few cm from
the array as RANGE_REFINED with range observability ~1.0, so every single-node
fix collapsed onto its node (docs/field_mode_progress/onset_srp.md).
"""

from __future__ import annotations

import itertools

import numpy as np

from minimappr.core.cartesian_tdoa import solve_cartesian_tdoa
from minimappr.core.far_field_model_selection import (
    COMPACT_ARRAY_MAX_APERTURE_M,
    far_field_alternative_if_curvature_unsupported,
)
from minimappr.core.range_projection import RANGE_REFINED, range_mode_is_unobservable
from minimappr.core.tdoa_measurements import PairTdoaMeasurement, reference_tdoas
from minimappr.sim.geometry import regular_tetrahedron_offsets_m

SOUND_SPEED_MPS = 343.2
SAMPLE_RATE_HZ = 16_000
INTERPOLATION_FACTOR = 4


def _tetra(origin_m) -> dict[str, np.ndarray]:
    return {
        f"{tuple(origin_m)}:ch{index}": np.asarray(origin_m, dtype=np.float64) + np.asarray(offset)
        for index, offset in enumerate(regular_tetrahedron_offsets_m(0.05))
    }


def _measurements(positions, source_m, noise_std_s, rng) -> list[PairTdoaMeasurement]:
    rows = []
    for sensor_a, sensor_b in itertools.combinations(sorted(positions), 2):
        tdoa_s = (
            np.linalg.norm(source_m - positions[sensor_a]) - np.linalg.norm(source_m - positions[sensor_b])
        ) / SOUND_SPEED_MPS
        rows.append(
            PairTdoaMeasurement(
                sensor_a=sensor_a,
                sensor_b=sensor_b,
                tdoa_seconds=float(tdoa_s + rng.normal(0.0, noise_std_s)),
                correlation_peak=0.5,
                weight=0.5,
            )
        )
    return rows


def _solve(positions, measurements):
    sensor_ids = sorted(positions)
    reference_sensor = sensor_ids[0]
    windows = {sensor_id: np.ones(4) * (2.0 if sensor_id == reference_sensor else 1.0) for sensor_id in sensor_ids}
    reference_sensor, reference_tdoa_s = reference_tdoas(
        measurements=measurements, sensor_windows=windows, sensor_ids=sensor_ids
    )
    return solve_cartesian_tdoa(
        sensor_positions=positions,
        measurements=measurements,
        sound_speed_mps=SOUND_SPEED_MPS,
        sample_rate_hz=SAMPLE_RATE_HZ,
        interpolation_factor=INTERPOLATION_FACTOR,
        reference_sensor=reference_sensor,
        reference_tdoa_s=reference_tdoa_s,
    )


def _bearing_error_deg(origin_m, estimate_m, truth_m) -> float:
    estimate = np.asarray(estimate_m) - origin_m
    truth = np.asarray(truth_m) - origin_m
    cosine = float(estimate @ truth / (np.linalg.norm(estimate) * np.linalg.norm(truth)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def test_single_tetra_far_source_no_longer_collapses_onto_the_node() -> None:
    rng = np.random.default_rng(3)
    origin_m = np.zeros(3)
    positions = _tetra(origin_m)
    source_m = np.asarray([6.0, 12.0, 2.0])
    for _ in range(5):
        solve = _solve(positions, _measurements(positions, source_m, 4e-6, rng))
        assert range_mode_is_unobservable(solve.range_projection_mode)
        assert np.linalg.norm(solve.position_m - origin_m) > 10.0
        assert _bearing_error_deg(origin_m, solve.position_m, source_m) < 10.0


def test_multi_node_near_field_fix_is_kept() -> None:
    rng = np.random.default_rng(4)
    positions = {**_tetra((0.0, 0.0, 0.0)), **_tetra((20.0, 0.0, 0.0)), **_tetra((10.0, 18.0, 0.0))}
    source_m = np.asarray([6.0, 12.0, 2.0])
    solve = _solve(positions, _measurements(positions, source_m, 4e-6, rng))
    assert solve.range_projection_mode == RANGE_REFINED
    assert np.linalg.norm(solve.position_m - source_m) < 1.0


def test_wide_arrays_are_never_offered_a_far_field_alternative() -> None:
    alternative = far_field_alternative_if_curvature_unsupported(
        centroid_m=np.zeros(3),
        near_field_position_m=np.asarray([0.5, 0.0, 0.0]),
        plane_wave_direction=np.asarray([1.0, 0.0, 0.0]),
        residual_function=lambda position_m: np.zeros(3),
        near_field_residual_rms_seconds=0.0,
        timing_noise_std_seconds=1e-5,
        far_field_radius_m=50.0,
        array_aperture_m=COMPACT_ARRAY_MAX_APERTURE_M * 2.0,
    )
    assert alternative is None


def test_near_field_with_strong_curvature_evidence_is_kept() -> None:
    # Residual is zero only at the near point; the far projection leaves a gap
    # far above the timing noise, so the near fix stands.
    near_point_m = np.asarray([0.2, 0.0, 0.0])
    alternative = far_field_alternative_if_curvature_unsupported(
        centroid_m=np.zeros(3),
        near_field_position_m=near_point_m,
        plane_wave_direction=np.asarray([1.0, 0.0, 0.0]),
        residual_function=lambda position_m: np.asarray([np.linalg.norm(position_m - near_point_m) * 1e-3]),
        near_field_residual_rms_seconds=0.0,
        timing_noise_std_seconds=1e-5,
        far_field_radius_m=50.0,
        array_aperture_m=0.05,
    )
    assert alternative is None


def test_solves_outside_the_collapse_zone_are_left_alone() -> None:
    alternative = far_field_alternative_if_curvature_unsupported(
        centroid_m=np.zeros(3),
        near_field_position_m=np.asarray([0.3, 0.0, 0.0]),
        plane_wave_direction=np.asarray([1.0, 0.0, 0.0]),
        residual_function=lambda position_m: np.zeros(3),
        near_field_residual_rms_seconds=0.0,
        timing_noise_std_seconds=1e-5,
        far_field_radius_m=50.0,
        array_aperture_m=0.05,
    )
    assert alternative is None
