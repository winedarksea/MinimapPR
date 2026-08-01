"""Broadband Cartesian TDOA localization with explicit weak-axis handling."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from minimappr.core.localization_uncertainty import covariance_to_nested_list
from minimappr.core.radial_range_refinement import (
    adaptive_radial_refinement,
    radial_standard_deviation,
)
from minimappr.core.range_projection import (
    RANGE_ASYMPTOTIC,
    RANGE_BEARING_PROJECTED,
    RANGE_BOUNDARY,
    RANGE_REFINED,
    BEARING_PROJECTED_CONFIDENCE_CAP,
    BEARING_PROJECTED_RANGE_OBSERVABILITY_CAP,
    UNOBSERVABLE_CONFIDENCE_CAP,
    UNOBSERVABLE_RANGE_OBSERVABILITY_CAP,
    normalize_range_mode,
)
from minimappr.core.spatial_constraints import (
    AmplitudeRatioConstraint,
    SpatialBearingConstraint,
    bearing_intersection_initial_position,
    broadband_direction_from_tdoas,
    spatial_constraint_jacobian,
    spatial_constraint_residual_seconds,
)
from minimappr.core.subspace_bearing import BearingPrior
from minimappr.core.tdoa_measurements import (
    PairTdoaMeasurement,
    measure_pair_tdoas,
    normalized_pair_quality,
    reference_tdoas,
)
from minimappr.models import LocalizationResult


EPSILON = 1.0e-12

# Default obs_factor floor for solves with TDOA measurements whose bearing is
# well-conditioned (Jacobian cond < 1e8). condition_observability compares the
# covariance's extreme eigenvalues, which for a compact (~5 cm) array is
# ~1e-5..1e-3 even on a perfectly healthy bearing solve — on the 2026-08-01
# live box it crushed every detection's confidence to ~0.002, excluding all
# compact-array tracks from IAMF spatial rendering (0.30 slot threshold).
# RANGE_BEARING_PROJECTED already escapes via obs_factor=1.0 for exactly this
# reason; the floor extends that reasoning to bearing-observed RANGE_REFINED
# solves. 1.0 = full escape (confidence = peak_quality × fit_quality);
# 0.0 = legacy behavior. The per-mode confidence caps still apply after it.
BEARING_OBSERVED_OBS_FACTOR_FLOOR = 1.0
MAX_REFINEMENT_START_COUNT = 4
MAX_REFINEMENT_EVALUATIONS = 100
MAX_RETRY_REFINEMENT_EVALUATIONS = 500
COMPACT_ARRAY_RADIAL_REFINEMENT_APERTURE_MULTIPLIER = 2.0
PRIOR_DOMINATED_ASYMPTOTIC_RADIAL_STD_FRACTION = 0.9

# Fallback observability cutoff applied when no bearing prior is available
# (mirrors Rust's unconditional ``radial_std_m >= radius_m`` far-field check
# in fit_far_field_range). Without a bearing prior, the radial search has no
# other safeguard and can settle on a finite "interior minimum" radius whose
# Fisher uncertainty (radial_std_m) is many times the radius itself -- i.e. a
# range estimate the data does not actually support. 5x is calibrated against
# the near-field fixtures in test_localization_benchmark.py and
# test_phase2_localization_soundscape.py (legitimate near-field refinements
# stay below ~3x) while still catching collapsed far-field solves (26x-330x).
DEFAULT_ASYMPTOTIC_RADIAL_STD_FRACTION = 5.0


_RANGE_PROJECTION_CONDITION_THRESHOLD: float = 1e6


@dataclass(frozen=True, slots=True)
class CartesianTdoaSolve:
    position_m: np.ndarray
    covariance_m2: np.ndarray
    confidence: float
    gdop: float
    residual_rms_seconds: float
    radial_observability: float
    range_projection_mode: str
    condition_number: float = float("inf")


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


def _bearing_prior_residual_seconds(
    position_m: np.ndarray,
    *,
    centroid_m: np.ndarray,
    bearing_prior: BearingPrior | None,
    sample_time_std_s: float,
) -> np.ndarray:
    if bearing_prior is None:
        return np.zeros(0, dtype=np.float64)
    offset_m = position_m - centroid_m
    offset_norm_m = float(np.linalg.norm(offset_m))
    if offset_norm_m < EPSILON:
        return np.zeros(3, dtype=np.float64)
    estimated_direction = offset_m / offset_norm_m
    angular_error = estimated_direction - bearing_prior.direction
    return angular_error * sample_time_std_s * math.sqrt(bearing_prior.strength)


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
    return np.vstack(rows) if rows else np.zeros((0, 3), dtype=np.float64)


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


def _covariance_from_svd(
    *,
    jacobian: np.ndarray,
    time_std_s: float,
    minimum_position_std_m: float,
    unobservable_position_std_m: float,
) -> np.ndarray:
    try:
        _, singular_values, right_vectors_t = np.linalg.svd(jacobian, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float64) * (unobservable_position_std_m**2)
    variance_floor_m2 = minimum_position_std_m**2
    unobservable_variance_m2 = unobservable_position_std_m**2
    variances = np.full(3, unobservable_variance_m2, dtype=np.float64)
    for index, singular_value in enumerate(singular_values[:3]):
        if singular_value > EPSILON:
            variances[index] = float(max((time_std_s / singular_value) ** 2, variance_floor_m2))
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
    bearing_prior: BearingPrior | None = None,
    bearing_constraints: list[SpatialBearingConstraint] | None = None,
    amplitude_constraints: list[AmplitudeRatioConstraint] | None = None,
    far_field_initial_range_m: float = 50.0,
    far_field_prior_radial_std_m: float | None = None,
    radial_refinement_enabled: bool = True,
    bearing_observed_obs_floor: float = BEARING_OBSERVED_OBS_FACTOR_FLOOR,
) -> CartesianTdoaSolve:
    bearing_constraints = bearing_constraints or []
    amplitude_constraints = amplitude_constraints or []
    sensor_ids = sorted(sensor_positions)
    centroid_m = np.mean(np.vstack([sensor_positions[sensor_id] for sensor_id in sensor_ids]), axis=0)
    aperture_m = _array_aperture_m(sensor_positions, sensor_ids)
    sample_time_std_s = 1.0 / max(sample_rate_hz * max(interpolation_factor, 1), 1)
    minimum_position_std_m = max(sound_speed_mps * sample_time_std_s, 0.05)
    has_tdoa_measurements = bool(measurements)
    if has_tdoa_measurements:
        direction = broadband_direction_from_tdoas(
            sensor_positions=sensor_positions,
            measurements=measurements,
            sound_speed_mps=sound_speed_mps,
        )
        initial_positions = [
            centroid_m,
            centroid_m + direction * max(aperture_m, minimum_position_std_m),
            centroid_m + direction * max(aperture_m * 10.0, 1.0),
        ]
    else:
        search_scale_m = max(aperture_m, minimum_position_std_m)
        direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        initial_positions = [
            centroid_m,
            centroid_m + np.asarray([search_scale_m, 0.0, 0.0]),
            centroid_m + np.asarray([0.0, search_scale_m, 0.0]),
            centroid_m + np.asarray([0.0, 0.0, search_scale_m]),
        ]
    bearing_intersection_position = bearing_intersection_initial_position(
        bearing_constraints
    )
    if bearing_intersection_position is not None:
        initial_positions.insert(0, bearing_intersection_position)
    if bearing_prior is not None:
        initial_positions.insert(
            0,
            centroid_m
            + bearing_prior.direction * max(aperture_m, minimum_position_std_m),
        )
    algebraic_position = (
        _algebraic_initial_position(
            sensor_positions=sensor_positions,
            reference_sensor=reference_sensor,
            reference_tdoa_s=reference_tdoa_s,
            sound_speed_mps=sound_speed_mps,
        )
        if has_tdoa_measurements
        else None
    )
    if algebraic_position is not None:
        initial_positions.insert(0, algebraic_position)

    candidate_solutions: list[tuple[float, np.ndarray]] = []
    def combined_residual_seconds(position_m: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (
                _weighted_residual_seconds(
                    position_m,
                    sensor_positions=sensor_positions,
                    measurements=measurements,
                    sound_speed_mps=sound_speed_mps,
                ),
                _bearing_prior_residual_seconds(
                    position_m,
                    centroid_m=centroid_m,
                    bearing_prior=bearing_prior,
                    sample_time_std_s=sample_time_std_s,
                ),
                spatial_constraint_residual_seconds(
                    position_m,
                    bearing_constraints=bearing_constraints,
                    amplitude_constraints=amplitude_constraints,
                    residual_scale_seconds=sample_time_std_s,
                ),
            )
        )

    for initial_position in initial_positions[:MAX_REFINEMENT_START_COUNT]:
        solved = None
        try:
            solved = least_squares(
                combined_residual_seconds,
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
        if not solved.success and np.all(np.isfinite(solved.x)):
            try:
                solved = least_squares(
                    combined_residual_seconds,
                    x0=solved.x,
                    method="trf",
                    loss="soft_l1",
                    max_nfev=MAX_RETRY_REFINEMENT_EVALUATIONS,
                    xtol=1.0e-11,
                    ftol=1.0e-11,
                    gtol=1.0e-11,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
        if not solved.success:
            continue
        residual = combined_residual_seconds(solved.x)
        cost = float(np.dot(residual, residual))
        if (
            np.all(np.isfinite(solved.x))
            and np.all(np.isfinite(residual))
            and np.all(np.isfinite(solved.jac))
            and np.isfinite(cost)
        ):
            candidate_solutions.append((cost, solved.x))
    if not candidate_solutions:
        raise ValueError("Cartesian TDOA refinement did not converge")
    _, best_position = min(candidate_solutions, key=lambda candidate: candidate[0])

    residual = combined_residual_seconds(best_position)
    residual_rms_s = float(np.sqrt(np.mean(np.square(residual))))
    effective_time_std_s = max(residual_rms_s, sample_time_std_s)
    range_projection_mode = RANGE_REFINED
    radial_refinement_std_m: float | None = None
    preliminary_jacobian = np.vstack(
        (
            _weighted_tdoa_jacobian(
                best_position,
                sensor_positions=sensor_positions,
                measurements=measurements,
                sound_speed_mps=sound_speed_mps,
            ),
            spatial_constraint_jacobian(
                best_position,
                bearing_constraints=bearing_constraints,
                amplitude_constraints=amplitude_constraints,
                residual_scale_seconds=sample_time_std_s,
            ),
        )
    )
    best_range_before_radial_refinement_m = float(
        np.linalg.norm(best_position - centroid_m)
    )
    preliminary_radial_std_m = radial_standard_deviation(
        position_m=best_position,
        centroid_m=centroid_m,
        jacobian=preliminary_jacobian,
        time_std_s=effective_time_std_s,
    )
    weak_cartesian_range = preliminary_radial_std_m >= max(
        best_range_before_radial_refinement_m,
        aperture_m,
        minimum_position_std_m,
    )
    compact_array_far_field_candidate = best_range_before_radial_refinement_m >= (
        aperture_m * COMPACT_ARRAY_RADIAL_REFINEMENT_APERTURE_MULTIPLIER
    )
    if (
        has_tdoa_measurements
        and radial_refinement_enabled
        and (weak_cartesian_range or compact_array_far_field_candidate)
    ):
        best_offset = best_position - centroid_m
        best_offset_norm = float(np.linalg.norm(best_offset))
        refinement_direction = best_offset / max(best_offset_norm, EPSILON)
        radial_refinement = adaptive_radial_refinement(
            centroid_m=centroid_m,
            direction=refinement_direction,
            initial_radius_m=max(
                best_offset_norm,
                far_field_initial_range_m,
                aperture_m,
                minimum_position_std_m,
            ),
            minimum_radius_m=max(aperture_m * 0.5, minimum_position_std_m),
            residual_function=combined_residual_seconds,
            effective_time_std_s=effective_time_std_s,
            asymptotic_radial_std_fraction=(
                PRIOR_DOMINATED_ASYMPTOTIC_RADIAL_STD_FRACTION
                if bearing_prior is not None
                else DEFAULT_ASYMPTOTIC_RADIAL_STD_FRACTION
            ),
        )
        if radial_refinement.fit_residual_rms_seconds < residual_rms_s * (1.0 - 1.0e-4):
            best_position = radial_refinement.position_m
            radial_refinement_std_m = radial_refinement.radial_std_m
            range_projection_mode = radial_refinement.projection_mode
            residual = combined_residual_seconds(best_position)
            residual_rms_s = float(np.sqrt(np.mean(np.square(residual))))
            effective_time_std_s = max(residual_rms_s, sample_time_std_s)
    jacobian = np.vstack(
        (
            _weighted_tdoa_jacobian(
                best_position,
                sensor_positions=sensor_positions,
                measurements=measurements,
                sound_speed_mps=sound_speed_mps,
            ),
            spatial_constraint_jacobian(
                best_position,
                bearing_constraints=bearing_constraints,
                amplitude_constraints=amplitude_constraints,
                residual_scale_seconds=sample_time_std_s,
            ),
        )
    )
    # Compute main Jacobian SVD once: used for condition number gate (Item D),
    # covariance, and GDOP to avoid three separate SVD calls.
    try:
        jacobian_singular_values = np.linalg.svd(jacobian, compute_uv=False)
        main_cond_num = float(
            jacobian_singular_values[0] / max(float(jacobian_singular_values[-1]), EPSILON)
        )
    except np.linalg.LinAlgError:
        jacobian_singular_values = np.zeros(3)
        main_cond_num = float("inf")
    # When the full Jacobian is severely ill-conditioned (e.g. nearly collinear
    # multi-node geometry or a compact single-array with a far source that the
    # radial refinement didn't catch), force RANGE_ASYMPTOTIC so the existing
    # bearing-condition check below can upgrade it to RANGE_BEARING_PROJECTED.
    if (
        range_projection_mode == RANGE_REFINED
        and has_tdoa_measurements
        and main_cond_num > _RANGE_PROJECTION_CONDITION_THRESHOLD
    ):
        range_projection_mode = RANGE_ASYMPTOTIC
    solved_range_m = float(np.linalg.norm(best_position - centroid_m))
    unobservable_position_std_m = max(
        radial_refinement_std_m or 0.0,
        solved_range_m,
        aperture_m,
        minimum_position_std_m,
    )
    covariance_m2 = _covariance_from_svd(
        jacobian=jacobian,
        time_std_s=effective_time_std_s,
        minimum_position_std_m=minimum_position_std_m,
        unobservable_position_std_m=unobservable_position_std_m,
    )
    radial_axis = (best_position - centroid_m) / max(solved_range_m, EPSILON)
    if radial_refinement_std_m is not None:
        radial_variance_m2 = float(radial_axis @ covariance_m2 @ radial_axis)
        covariance_m2 += max(
            (radial_refinement_std_m**2) - radial_variance_m2,
            0.0,
        ) * np.outer(radial_axis, radial_axis)
    eigenvalues = np.linalg.eigvalsh(covariance_m2)
    radial_variance_m2 = float(radial_axis @ covariance_m2 @ radial_axis)
    lateral_variance_m2 = max(
        (float(np.trace(covariance_m2)) - radial_variance_m2) / 2.0,
        minimum_position_std_m**2,
    )
    # When the range axis is unobservable (RANGE_ASYMPTOTIC) but the bearing is
    # well-determined (Jacobian condition number < 1e8), upgrade to
    # RANGE_BEARING_PROJECTED and replace the covariance with an explicit
    # "cone of uncertainty" — tight laterally (from GCC-PHAT angular precision),
    # very large radially (4× solved range or 200 m, whichever is greater).
    try:
        bearing_cond = float(np.linalg.cond(preliminary_jacobian))
    except np.linalg.LinAlgError:
        bearing_cond = float("inf")
    if range_projection_mode == RANGE_ASYMPTOTIC and has_tdoa_measurements:
        if bearing_cond < 1e8:
            range_projection_mode = RANGE_BEARING_PROJECTED
            # Cone radial (range) std: loosest of 4× the solved range, the amplitude
            # prior's radial std (std_factor × prior_range, passed by the caller when
            # the amplitude prior is active; otherwise the raw projection distance),
            # and a 200 m floor. Mirrors srp_phat.rs's cone construction.
            prior_radial_std_m = (
                far_field_prior_radial_std_m
                if far_field_prior_radial_std_m is not None
                else far_field_initial_range_m
            )
            cone_radial_std_m = max(solved_range_m * 4.0, prior_radial_std_m, 200.0)
            cone_lateral_std_m = math.sqrt(lateral_variance_m2)
            covariance_m2 = (
                cone_lateral_std_m**2 * (np.eye(3, dtype=np.float64) - np.outer(radial_axis, radial_axis))
                + cone_radial_std_m**2 * np.outer(radial_axis, radial_axis)
            )
            eigenvalues = np.linalg.eigvalsh(covariance_m2)
            radial_variance_m2 = cone_radial_std_m**2
            lateral_variance_m2 = cone_lateral_std_m**2
    radial_observability = float(
        np.clip(math.sqrt(lateral_variance_m2 / max(radial_variance_m2, lateral_variance_m2)), 0.0, 1.0)
    )
    condition_observability = float(
        np.clip(math.sqrt(max(float(eigenvalues[0]), EPSILON) / max(float(eigenvalues[-1]), EPSILON)), 0.0, 1.0)
    )
    radial_observability = min(radial_observability, condition_observability)
    _norm_mode = normalize_range_mode(range_projection_mode)
    if _norm_mode in {RANGE_ASYMPTOTIC, RANGE_BOUNDARY}:
        radial_observability = min(radial_observability, UNOBSERVABLE_RANGE_OBSERVABILITY_CAP)
    elif _norm_mode == RANGE_BEARING_PROJECTED:
        radial_observability = min(radial_observability, BEARING_PROJECTED_RANGE_OBSERVABILITY_CAP)
    if measurements:
        peak_quality = float(
            np.mean(
                [
                    normalized_pair_quality(measurement.correlation_peak)
                    for measurement in measurements
                ]
            )
        )
    elif bearing_constraints:
        peak_quality = float(
            np.mean([constraint.quality for constraint in bearing_constraints])
        )
    else:
        peak_quality = 0.5
    fit_scale_s = max(sample_time_std_s * 4.0, EPSILON)
    fit_quality = float(math.exp(-residual_rms_s / fit_scale_s))
    if _norm_mode == RANGE_BEARING_PROJECTED:
        # The banana covariance has an intentionally huge radial eigenvalue, so
        # condition_observability is artificially tiny and must not penalise the
        # bearing quality. Use only peak and fit quality for the base score.
        obs_factor = 1.0
    else:
        obs_factor = math.sqrt(max(condition_observability, radial_observability * 0.25))
        # Same reasoning as the branch above, extended to bearing-observed
        # solves that kept a range estimate (RANGE_REFINED and friends): a
        # compact array's covariance conditioning penalises a perfectly good
        # bearing solve by orders of magnitude. The per-mode caps below still
        # bound ASYMPTOTIC/BOUNDARY at UNOBSERVABLE_CONFIDENCE_CAP.
        if has_tdoa_measurements and bearing_cond < 1e8:
            obs_factor = max(obs_factor, bearing_observed_obs_floor)
    confidence = float(np.clip(peak_quality * fit_quality * obs_factor, 0.0, 1.0))
    if _norm_mode in {RANGE_ASYMPTOTIC, RANGE_BOUNDARY}:
        confidence = min(confidence, UNOBSERVABLE_CONFIDENCE_CAP)
    elif _norm_mode == RANGE_BEARING_PROJECTED:
        confidence = min(confidence, BEARING_PROJECTED_CONFIDENCE_CAP)
    inverse_squared = [
        1.0 / (sv * sv)
        for sv in jacobian_singular_values
        if sv > EPSILON
    ]
    gdop = float(math.sqrt(sum(inverse_squared))) if len(inverse_squared) == 3 else float("inf")
    return CartesianTdoaSolve(
        position_m=best_position,
        covariance_m2=covariance_m2,
        confidence=confidence,
        gdop=gdop,
        residual_rms_seconds=residual_rms_s,
        radial_observability=radial_observability,
        range_projection_mode=range_projection_mode,
        condition_number=main_cond_num,
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
        range_observability=solve.radial_observability,
        residual_rms_seconds=solve.residual_rms_seconds,
        range_projection_mode=solve.range_projection_mode,
        condition_number=solve.condition_number if math.isfinite(solve.condition_number) else None,
    )
