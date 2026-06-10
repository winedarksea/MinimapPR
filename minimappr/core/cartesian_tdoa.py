"""Broadband Cartesian TDOA localization with explicit weak-axis handling."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from minimappr.core.localization_uncertainty import covariance_to_nested_list
from minimappr.core.tdoa_measurements import (
    PairTdoaMeasurement,
    measure_pair_tdoas,
    normalized_pair_quality,
    reference_tdoas,
)
from minimappr.models import LocalizationResult


EPSILON = 1.0e-12
MAX_REFINEMENT_START_COUNT = 4
MAX_REFINEMENT_EVALUATIONS = 20
RADIAL_UNOBSERVABLE_STD_TO_RANGE_RATIO = 1.0
COMPACT_ARRAY_WEAK_RANGE_APERTURE_MULTIPLIER = 2.0


@dataclass(frozen=True, slots=True)
class CartesianTdoaSolve:
    position_m: np.ndarray
    covariance_m2: np.ndarray
    confidence: float
    gdop: float
    residual_rms_seconds: float
    radial_observability: float


def _array_aperture_m(sensor_positions: dict[str, np.ndarray], sensor_ids: list[str]) -> float:
    return max(
        (
            float(np.linalg.norm(sensor_positions[a] - sensor_positions[b]))
            for a, b in itertools.combinations(sensor_ids, 2)
        ),
        default=0.0,
    )

def _weighted_residual_seconds(
    position_m: np.ndarray,
    *,
    sensor_positions: dict[str, np.ndarray],
    measurements: list[PairTdoaMeasurement],
    sound_speed_mps: float,
) -> np.ndarray:
    rows: list[float] = []
    for measurement in measurements:
        distance_a_m = float(np.linalg.norm(position_m - sensor_positions[measurement.sensor_a]))
        distance_b_m = float(np.linalg.norm(position_m - sensor_positions[measurement.sensor_b]))
        predicted_tdoa_s = (distance_a_m - distance_b_m) / sound_speed_mps
        rows.append(math.sqrt(measurement.weight) * (predicted_tdoa_s - measurement.tdoa_seconds))
    return np.asarray(rows, dtype=np.float64)


def _weighted_tdoa_jacobian(
    position_m: np.ndarray,
    *,
    sensor_positions: dict[str, np.ndarray],
    measurements: list[PairTdoaMeasurement],
    sound_speed_mps: float,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for measurement in measurements:
        offset_a = position_m - sensor_positions[measurement.sensor_a]
        offset_b = position_m - sensor_positions[measurement.sensor_b]
        distance_a_m = float(np.linalg.norm(offset_a)) + EPSILON
        distance_b_m = float(np.linalg.norm(offset_b)) + EPSILON
        gradient = (
            (offset_a / distance_a_m) - (offset_b / distance_b_m)
        ) / sound_speed_mps
        rows.append(math.sqrt(measurement.weight) * gradient)
    return np.vstack(rows)


def _broadband_direction(
    *,
    sensor_positions: dict[str, np.ndarray],
    measurements: list[PairTdoaMeasurement],
    sound_speed_mps: float,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    targets: list[float] = []
    weights: list[float] = []
    for measurement in measurements:
        rows.append(sensor_positions[measurement.sensor_b] - sensor_positions[measurement.sensor_a])
        targets.append(sound_speed_mps * measurement.tdoa_seconds)
        weights.append(math.sqrt(measurement.weight))
    weighted_rows = np.vstack(rows) * np.asarray(weights)[:, None]
    weighted_targets = np.asarray(targets, dtype=np.float64) * np.asarray(weights)
    try:
        direction, *_ = np.linalg.lstsq(weighted_rows, weighted_targets, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ValueError("TDOA bearing solve is singular") from exc
    norm = float(np.linalg.norm(direction))
    if norm < EPSILON:
        raise ValueError("TDOA bearing solve is degenerate")
    return direction / norm


def _algebraic_initial_position(
    *,
    sensor_positions: dict[str, np.ndarray],
    reference_sensor: str,
    reference_tdoa_s: dict[str, float],
    sound_speed_mps: float,
) -> np.ndarray | None:
    reference_position = sensor_positions[reference_sensor]
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for sensor_id, tdoa_seconds in sorted(reference_tdoa_s.items()):
        range_difference_m = sound_speed_mps * tdoa_seconds
        sensor_position = sensor_positions[sensor_id]
        rows.append(
            np.concatenate(
                (
                    2.0 * (sensor_position - reference_position),
                    np.asarray([2.0 * range_difference_m]),
                )
            )
        )
        targets.append(
            float(
                np.dot(sensor_position, sensor_position)
                - np.dot(reference_position, reference_position)
                - (range_difference_m * range_difference_m)
            )
        )
    if len(rows) < 3:
        return None
    try:
        solution, *_ = np.linalg.lstsq(np.vstack(rows), np.asarray(targets), rcond=None)
    except np.linalg.LinAlgError:
        return None
    position_m = solution[:3]
    return position_m if np.all(np.isfinite(position_m)) else None


def _radial_std_m(
    *,
    position_m: np.ndarray,
    centroid_m: np.ndarray,
    jacobian: np.ndarray,
    time_std_s: float,
) -> float:
    radial_offset = position_m - centroid_m
    radial_norm = float(np.linalg.norm(radial_offset))
    if radial_norm < EPSILON:
        return float("inf")
    radial_axis = radial_offset / radial_norm
    radial_information = float(np.linalg.norm(jacobian @ radial_axis))
    if radial_information < EPSILON:
        return float("inf")
    return time_std_s / radial_information


def _observability_boundary_position(
    *,
    centroid_m: np.ndarray,
    direction: np.ndarray,
    aperture_m: float,
    time_std_s: float,
    sensor_positions: dict[str, np.ndarray],
    measurements: list[PairTdoaMeasurement],
    sound_speed_mps: float,
) -> np.ndarray:
    minimum_radius_m = max(aperture_m * 0.5, 0.05)
    radius_m = minimum_radius_m
    maximum_radius_m = max(
        10.0,
        aperture_m * 100.0,
        (aperture_m * aperture_m) / max(sound_speed_mps * time_std_s, EPSILON) * 4.0,
    )
    for _ in range(24):
        candidate = centroid_m + direction * radius_m
        jacobian = _weighted_tdoa_jacobian(
            candidate,
            sensor_positions=sensor_positions,
            measurements=measurements,
            sound_speed_mps=sound_speed_mps,
        )
        radial_std_m = _radial_std_m(
            position_m=candidate,
            centroid_m=centroid_m,
            jacobian=jacobian,
            time_std_s=time_std_s,
        )
        if radial_std_m >= radius_m * RADIAL_UNOBSERVABLE_STD_TO_RANGE_RATIO:
            return candidate
        radius_m = min(radius_m * 1.6, maximum_radius_m)
        if radius_m >= maximum_radius_m:
            break
    return centroid_m + direction * maximum_radius_m


def _covariance_from_svd(
    *,
    jacobian: np.ndarray,
    time_std_s: float,
    minimum_position_std_m: float,
    maximum_position_std_m: float,
) -> np.ndarray:
    try:
        _, singular_values, right_vectors_t = np.linalg.svd(jacobian, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float64) * (maximum_position_std_m**2)
    variance_floor_m2 = minimum_position_std_m**2
    variance_ceiling_m2 = maximum_position_std_m**2
    variances = np.full(3, variance_ceiling_m2, dtype=np.float64)
    for index, singular_value in enumerate(singular_values[:3]):
        if singular_value > EPSILON:
            variances[index] = float(
                np.clip((time_std_s / singular_value) ** 2, variance_floor_m2, variance_ceiling_m2)
            )
    covariance = right_vectors_t.T @ np.diag(variances) @ right_vectors_t
    return 0.5 * (covariance + covariance.T)


def solve_cartesian_tdoa(
    *,
    sensor_positions: dict[str, np.ndarray],
    measurements: list[PairTdoaMeasurement],
    sound_speed_mps: float,
    sample_rate_hz: int,
    interpolation_factor: int,
    reference_sensor: str,
    reference_tdoa_s: dict[str, float],
) -> CartesianTdoaSolve:
    sensor_ids = sorted(sensor_positions)
    centroid_m = np.mean(np.vstack([sensor_positions[sensor_id] for sensor_id in sensor_ids]), axis=0)
    aperture_m = _array_aperture_m(sensor_positions, sensor_ids)
    sample_time_std_s = 1.0 / max(sample_rate_hz * max(interpolation_factor, 1), 1)
    minimum_position_std_m = max(sound_speed_mps * sample_time_std_s, 0.05)
    direction = _broadband_direction(
        sensor_positions=sensor_positions,
        measurements=measurements,
        sound_speed_mps=sound_speed_mps,
    )
    observability_boundary = _observability_boundary_position(
        centroid_m=centroid_m,
        direction=direction,
        aperture_m=aperture_m,
        time_std_s=sample_time_std_s,
        sensor_positions=sensor_positions,
        measurements=measurements,
        sound_speed_mps=sound_speed_mps,
    )
    initial_positions = [
        centroid_m,
        centroid_m + direction * max(aperture_m, minimum_position_std_m),
        observability_boundary,
    ]
    algebraic_position = _algebraic_initial_position(
        sensor_positions=sensor_positions,
        reference_sensor=reference_sensor,
        reference_tdoa_s=reference_tdoa_s,
        sound_speed_mps=sound_speed_mps,
    )
    if algebraic_position is not None:
        initial_positions.insert(0, algebraic_position)

    candidate_solutions: list[tuple[float, float, np.ndarray]] = []
    for initial_position in initial_positions[:MAX_REFINEMENT_START_COUNT]:
        try:
            solved = least_squares(
                lambda position: _weighted_residual_seconds(
                    position,
                    sensor_positions=sensor_positions,
                    measurements=measurements,
                    sound_speed_mps=sound_speed_mps,
                ),
                x0=initial_position,
                method="trf",
                loss="soft_l1",
                max_nfev=MAX_REFINEMENT_EVALUATIONS,
                xtol=1.0e-9,
                ftol=1.0e-9,
                gtol=1.0e-9,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        residual = _weighted_residual_seconds(
            solved.x,
            sensor_positions=sensor_positions,
            measurements=measurements,
            sound_speed_mps=sound_speed_mps,
        )
        cost = float(np.dot(residual, residual))
        if np.all(np.isfinite(solved.x)) and np.isfinite(cost):
            range_from_centroid_m = float(np.linalg.norm(solved.x - centroid_m))
            candidate_solutions.append((cost, range_from_centroid_m, solved.x))
    if not candidate_solutions:
        raise ValueError("Cartesian TDOA refinement failed")
    minimum_cost = min(candidate[0] for candidate in candidate_solutions)
    # Radially distinct solutions inside the timing noise floor are not distinguishable.
    statistically_equivalent_cost = minimum_cost + (
        len(measurements) * sample_time_std_s * sample_time_std_s
    )
    _, _, best_position = min(
        (
            candidate
            for candidate in candidate_solutions
            if candidate[0] <= statistically_equivalent_cost
        ),
        key=lambda candidate: candidate[1],
    )

    residual = _weighted_residual_seconds(
        best_position,
        sensor_positions=sensor_positions,
        measurements=measurements,
        sound_speed_mps=sound_speed_mps,
    )
    residual_rms_s = float(np.sqrt(np.mean(np.square(residual))))
    effective_time_std_s = max(residual_rms_s, sample_time_std_s)
    jacobian = _weighted_tdoa_jacobian(
        best_position,
        sensor_positions=sensor_positions,
        measurements=measurements,
        sound_speed_mps=sound_speed_mps,
    )
    best_range_m = float(np.linalg.norm(best_position - centroid_m))
    radial_std_m = _radial_std_m(
        position_m=best_position,
        centroid_m=centroid_m,
        jacobian=jacobian,
        time_std_s=effective_time_std_s,
    )
    if radial_std_m >= max(best_range_m, aperture_m, minimum_position_std_m):
        best_position = observability_boundary
        residual = _weighted_residual_seconds(
            best_position,
            sensor_positions=sensor_positions,
            measurements=measurements,
            sound_speed_mps=sound_speed_mps,
        )
        residual_rms_s = float(np.sqrt(np.mean(np.square(residual))))
        effective_time_std_s = max(residual_rms_s, sample_time_std_s)
        jacobian = _weighted_tdoa_jacobian(
            best_position,
            sensor_positions=sensor_positions,
            measurements=measurements,
            sound_speed_mps=sound_speed_mps,
        )

    solved_range_m = float(np.linalg.norm(best_position - centroid_m))
    maximum_position_std_m = max(25.0, solved_range_m * 4.0, aperture_m * 100.0)
    covariance_m2 = _covariance_from_svd(
        jacobian=jacobian,
        time_std_s=effective_time_std_s,
        minimum_position_std_m=minimum_position_std_m,
        maximum_position_std_m=maximum_position_std_m,
    )
    radial_axis = (best_position - centroid_m) / max(solved_range_m, EPSILON)
    if solved_range_m >= aperture_m * COMPACT_ARRAY_WEAK_RANGE_APERTURE_MULTIPLIER:
        # Evaluating at the bounded representative point must not invent radial precision.
        radial_variance_m2 = float(radial_axis @ covariance_m2 @ radial_axis)
        covariance_m2 += max(
            (maximum_position_std_m**2) - radial_variance_m2,
            0.0,
        ) * np.outer(radial_axis, radial_axis)
    eigenvalues = np.linalg.eigvalsh(covariance_m2)
    radial_variance_m2 = float(radial_axis @ covariance_m2 @ radial_axis)
    lateral_variance_m2 = max(
        (float(np.trace(covariance_m2)) - radial_variance_m2) / 2.0,
        minimum_position_std_m**2,
    )
    radial_observability = float(
        np.clip(math.sqrt(lateral_variance_m2 / max(radial_variance_m2, lateral_variance_m2)), 0.0, 1.0)
    )
    condition_observability = float(
        np.clip(math.sqrt(max(float(eigenvalues[0]), EPSILON) / max(float(eigenvalues[-1]), EPSILON)), 0.0, 1.0)
    )
    peak_quality = float(
        np.mean([normalized_pair_quality(measurement.correlation_peak) for measurement in measurements])
    )
    fit_scale_s = max(sample_time_std_s * 4.0, EPSILON)
    fit_quality = float(math.exp(-residual_rms_s / fit_scale_s))
    confidence = float(
        np.clip(
            peak_quality * fit_quality * math.sqrt(max(condition_observability, radial_observability * 0.25)),
            0.0,
            1.0,
        )
    )
    try:
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        inverse_squared = [
            1.0 / (singular_value * singular_value)
            for singular_value in singular_values
            if singular_value > EPSILON
        ]
        gdop = float(math.sqrt(sum(inverse_squared))) if len(inverse_squared) == 3 else float("inf")
    except np.linalg.LinAlgError:
        gdop = float("inf")
    return CartesianTdoaSolve(
        position_m=best_position,
        covariance_m2=covariance_m2,
        confidence=confidence,
        gdop=gdop,
        residual_rms_seconds=residual_rms_s,
        radial_observability=radial_observability,
    )


def localization_result_from_cartesian_solve(
    *,
    solve: CartesianTdoaSolve,
    reference_sensor: str,
    reference_tdoa_s: dict[str, float],
) -> LocalizationResult:
    position_m = solve.position_m
    return LocalizationResult(
        position_m=(float(position_m[0]), float(position_m[1]), float(position_m[2])),
        confidence=solve.confidence,
        gdop=solve.gdop,
        reference_sensor=reference_sensor,
        tdoa_s=reference_tdoa_s,
        position_covariance_m2=covariance_to_nested_list(solve.covariance_m2),
        residual_rms_seconds=solve.residual_rms_seconds,
    )
