"""Per-cluster Levenberg-Marquardt refinement of an SRP onset cluster.

Each cluster's frames vote for per-pair TDOAs: the local maximum of every pair
curve near the lag predicted at the cluster's SRP seed, aggregated across frames
by weighted median. Those TDOAs go through the existing Cartesian solver
(`solve_cartesian_tdoa`: LM, covariance, range-projection contract, compact-
array collapse guard), so S2 reuses rather than duplicates the solver. The LM
answer is kept only if it does not lower the cluster's summed SRP power; LM may
otherwise jump to a different basin than the one the cluster voted for.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from scipy.optimize import least_squares

from minimappr.core.cartesian_tdoa import solve_cartesian_tdoa
from minimappr.core.localization_uncertainty import (
    covariance_from_jacobian,
    covariance_to_nested_list,
    range_observability_from_covariance,
)
from minimappr.core.onset_srp_types import OnsetCluster, OnsetFrame
from minimappr.core.range_projection import (
    BEARING_PROJECTED_RANGE_OBSERVABILITY_CAP,
    RANGE_BEARING_PROJECTED,
    RANGE_REFINED,
    range_mode_is_unobservable,
)
from minimappr.core.tdoa_measurements import PairTdoaMeasurement, solve_reference_tdoas

# LM may lose this fraction of the seed's SRP score (curve interpolation noise)
# and still be preferred for its covariance and sub-grid precision.
REFINEMENT_MIN_SCORE_RATIO = 0.9
# Minimum per-pair search half-width around the seed's predicted lag, in samples.
MIN_LAG_SEARCH_HALF_WIDTH_SAMPLES = 2.0
# Correlation-peak floor for measurement weights (mirrors normalized_pair_quality).
MIN_MEASUREMENT_WEIGHT = 0.05
# Fit quality = exp(-residual / (this many interpolated sample periods)), as in
# cartesian_tdoa's confidence.
FIT_QUALITY_SAMPLE_PERIODS = 4.0


@dataclass(frozen=True, slots=True)
class RefinedClusterPosition:
    position_m: np.ndarray
    covariance_m2: list[list[float]] | None
    confidence: float
    gdop: float
    range_projection_mode: str
    range_observability: float | None
    residual_rms_seconds: float | None
    reference_sensor: str
    tdoa_s: dict[str, float]
    refined_by_least_squares: bool


def _predicted_lag_s(position_m, sensor_positions, sensor_a, sensor_b, sound_speed_mps) -> float:
    return float(
        (np.linalg.norm(position_m - sensor_positions[sensor_b]) - np.linalg.norm(position_m - sensor_positions[sensor_a]))
        / sound_speed_mps
    )


def cluster_srp_score(
    position_m: np.ndarray,
    cluster: OnsetCluster,
    frames: dict[int, OnsetFrame],
    sensor_positions: dict[str, np.ndarray],
    sound_speed_mps: float,
) -> float:
    score = 0.0
    for peak in cluster.peaks:
        frame = frames[peak.frame_index]
        total = sum(curve.weight_sum for curve in frame.curves) or 1.0
        for curve in frame.curves:
            lag = _predicted_lag_s(position_m, sensor_positions, curve.sensor_a, curve.sensor_b, sound_speed_mps)
            score += peak.weight * float(curve.value_at(np.asarray([lag]))[0]) * curve.weight_sum / total
    return score


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def vote_pair_tdoas(
    seed_position_m: np.ndarray,
    cluster: OnsetCluster,
    frames: dict[int, OnsetFrame],
    sensor_positions: dict[str, np.ndarray],
    sound_speed_mps: float,
    sample_rate_hz: int,
) -> list[PairTdoaMeasurement]:
    """Per-pair TDOA votes near the seed's predicted lags, aggregated by weighted median."""
    spread_s = 0.0
    for peak in cluster.peaks:
        for curve in frames[peak.frame_index].curves:
            spread_s = max(
                spread_s,
                abs(
                    _predicted_lag_s(peak.position_m, sensor_positions, curve.sensor_a, curve.sensor_b, sound_speed_mps)
                    - _predicted_lag_s(seed_position_m, sensor_positions, curve.sensor_a, curve.sensor_b, sound_speed_mps)
                ),
            )
    half_width_s = spread_s + MIN_LAG_SEARCH_HALF_WIDTH_SAMPLES / sample_rate_hz
    votes: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    for peak in cluster.peaks:
        for curve in frames[peak.frame_index].curves:
            predicted = _predicted_lag_s(seed_position_m, sensor_positions, curve.sensor_a, curve.sensor_b, sound_speed_mps)
            lag, value = curve.fine_peak_lag_near(predicted, half_width_s)
            if value <= 0.0:
                continue
            key, sign = ((curve.sensor_a, curve.sensor_b), 1.0)
            if curve.sensor_b < curve.sensor_a:
                key, sign = ((curve.sensor_b, curve.sensor_a), -1.0)
            votes.setdefault(key, []).append((sign * lag, peak.weight * value, value))
    measurements = []
    for (sensor_a, sensor_b), rows in votes.items():
        lags = np.asarray([row[0] for row in rows])
        weights = np.asarray([row[1] for row in rows])
        mean_value = float(np.mean([row[2] for row in rows]))
        # PairTdoaMeasurement predicts (d_a - d_b) / c, i.e. the lag of sensor_a
        # behind sensor_b; our curves give sensor_b behind sensor_a.
        measurements.append(
            PairTdoaMeasurement(
                sensor_a=sensor_b,
                sensor_b=sensor_a,
                tdoa_seconds=_weighted_median(lags, weights),
                correlation_peak=mean_value,
                weight=float(np.clip(mean_value, MIN_MEASUREMENT_WEIGHT, 1.0)),
            )
        )
    return measurements


def _seed_only_result(
    seed_position_m: np.ndarray,
    cluster: OnsetCluster,
    centroid_m: np.ndarray,
    is_bearing_only: bool,
    reference_sensor: str,
    minimum_std_m: float,
) -> RefinedClusterPosition:
    positions = np.vstack([peak.position_m for peak in cluster.peaks])
    spread = np.cov(positions.T) if positions.shape[0] > 1 else np.zeros((3, 3))
    covariance = spread + np.eye(3) * minimum_std_m**2
    mean_value = float(np.mean([peak.value for peak in cluster.peaks]))
    return RefinedClusterPosition(
        position_m=seed_position_m,
        covariance_m2=covariance_to_nested_list(covariance),
        confidence=float(np.clip(mean_value, 0.0, 1.0)),
        gdop=float("nan"),
        range_projection_mode=RANGE_BEARING_PROJECTED if is_bearing_only else RANGE_REFINED,
        range_observability=BEARING_PROJECTED_RANGE_OBSERVABILITY_CAP if is_bearing_only else None,
        residual_rms_seconds=None,
        reference_sensor=reference_sensor,
        tdoa_s={},
        refined_by_least_squares=False,
    )


def refine_cluster_position(
    *,
    cluster: OnsetCluster,
    frames: dict[int, OnsetFrame],
    sensor_positions: dict[str, np.ndarray],
    seed_position_m: np.ndarray,
    centroid_m: np.ndarray,
    seed_is_bearing_only: bool,
    sound_speed_mps: float,
    sample_rate_hz: int,
    interpolation_factor: int,
    far_field_default_range_m: float,
    minimum_position_std_m: float,
) -> RefinedClusterPosition:
    reference_counts = Counter(frames[index].reference_sensor for index in cluster.frame_indices)
    reference_sensor = reference_counts.most_common(1)[0][0]
    measurements = vote_pair_tdoas(
        seed_position_m, cluster, frames, sensor_positions, sound_speed_mps, sample_rate_hz
    )
    involved = sorted({m.sensor_a for m in measurements} | {m.sensor_b for m in measurements})
    seed_result = _seed_only_result(
        seed_position_m, cluster, centroid_m, seed_is_bearing_only, reference_sensor, minimum_position_std_m
    )
    if len(measurements) < 3 or reference_sensor not in involved:
        return seed_result
    reference_tdoa_s = solve_reference_tdoas(
        measurements=measurements, sensor_ids=involved, reference_sensor=reference_sensor
    )
    seed_score = cluster_srp_score(seed_position_m, cluster, frames, sensor_positions, sound_speed_mps)
    if seed_is_bearing_only:
        refined = _refine_bearing_with_cartesian_solver(
            measurements=measurements,
            involved=involved,
            sensor_positions=sensor_positions,
            centroid_m=centroid_m,
            reference_sensor=reference_sensor,
            reference_tdoa_s=reference_tdoa_s,
            sound_speed_mps=sound_speed_mps,
            sample_rate_hz=sample_rate_hz,
            interpolation_factor=interpolation_factor,
            far_field_default_range_m=far_field_default_range_m,
        )
    else:
        refined = _refine_position_from_seed(
            measurements=measurements,
            sensor_positions=sensor_positions,
            seed_position_m=seed_position_m,
            centroid_m=centroid_m,
            reference_sensor=reference_sensor,
            reference_tdoa_s=reference_tdoa_s,
            mean_peak_value=float(np.mean([peak.value for peak in cluster.peaks])),
            sound_speed_mps=sound_speed_mps,
            sample_rate_hz=sample_rate_hz,
            interpolation_factor=interpolation_factor,
            minimum_position_std_m=minimum_position_std_m,
        )
    if refined is None:
        return seed_result
    refined_score = cluster_srp_score(refined.position_m, cluster, frames, sensor_positions, sound_speed_mps)
    if refined_score < REFINEMENT_MIN_SCORE_RATIO * seed_score:
        return seed_result
    return refined


def _refine_bearing_with_cartesian_solver(
    *,
    measurements,
    involved,
    sensor_positions,
    centroid_m,
    reference_sensor,
    reference_tdoa_s,
    sound_speed_mps,
    sample_rate_hz,
    interpolation_factor,
    far_field_default_range_m,
) -> RefinedClusterPosition | None:
    """Single compact array: the solver's bearing (range unobservable) at the
    far-field default range, the range convention of RANGE_BEARING_PROJECTED."""
    try:
        solve = solve_cartesian_tdoa(
            sensor_positions={sensor_id: sensor_positions[sensor_id] for sensor_id in involved},
            measurements=measurements,
            sound_speed_mps=sound_speed_mps,
            sample_rate_hz=sample_rate_hz,
            interpolation_factor=interpolation_factor,
            reference_sensor=reference_sensor,
            reference_tdoa_s=reference_tdoa_s,
            far_field_initial_range_m=far_field_default_range_m,
        )
    except (ValueError, np.linalg.LinAlgError):
        return None
    offset = np.asarray(solve.position_m, dtype=np.float64) - centroid_m
    norm = float(np.linalg.norm(offset))
    if norm <= 0.0 or not np.isfinite(norm):
        return None
    position = solve.position_m
    covariance = solve.covariance_m2
    if range_mode_is_unobservable(solve.range_projection_mode):
        # The radial search can publish km-to-Mm ranges on an asymptotic cost;
        # rescale the cone to the default far-field range along the same bearing.
        scale = far_field_default_range_m / norm
        position = centroid_m + offset * scale
        covariance = np.asarray(covariance) * scale**2
    return RefinedClusterPosition(
        position_m=np.asarray(position, dtype=np.float64),
        covariance_m2=covariance_to_nested_list(np.asarray(covariance)),
        confidence=float(solve.confidence),
        gdop=float(solve.gdop),
        range_projection_mode=solve.range_projection_mode,
        range_observability=float(solve.radial_observability),
        residual_rms_seconds=float(solve.residual_rms_seconds),
        reference_sensor=reference_sensor,
        tdoa_s=dict(reference_tdoa_s),
        refined_by_least_squares=True,
    )


def _refine_position_from_seed(
    *,
    measurements,
    sensor_positions,
    seed_position_m,
    centroid_m,
    reference_sensor,
    reference_tdoa_s,
    mean_peak_value,
    sound_speed_mps,
    sample_rate_hz,
    interpolation_factor,
    minimum_position_std_m,
) -> RefinedClusterPosition | None:
    """Local LM from the cluster's SRP seed (multi-node: range observable).

    A local start, unlike the Cartesian solver's multi-start, keeps the answer in
    the basin the cluster voted for.
    """
    sample_time_s = 1.0 / (sample_rate_hz * max(1, interpolation_factor))
    rows = [
        (sensor_positions[m.sensor_a], sensor_positions[m.sensor_b], m.tdoa_seconds, np.sqrt(m.weight))
        for m in measurements
    ]

    def residuals(position_m: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                weight * ((np.linalg.norm(position_m - a) - np.linalg.norm(position_m - b)) / sound_speed_mps - tdoa)
                for a, b, tdoa, weight in rows
            ]
        )

    try:
        solved = least_squares(
            residuals, x0=np.asarray(seed_position_m, dtype=np.float64), method="trf", loss="soft_l1",
            f_scale=sample_time_s,
        )
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not solved.success or not np.all(np.isfinite(solved.x)):
        return None
    residual_rms_s = float(np.sqrt(np.mean(np.square(residuals(solved.x)))))
    covariance = covariance_from_jacobian(
        solved.jac, residual_rms_s, minimum_time_std_s=sample_time_s, minimum_std_m=minimum_position_std_m
    )
    if covariance is None:
        return None
    fit_quality = float(np.exp(-residual_rms_s / (FIT_QUALITY_SAMPLE_PERIODS * sample_time_s)))
    singular_values = np.linalg.svd(solved.jac, compute_uv=False)
    gdop = (
        float(np.sqrt(np.sum(1.0 / singular_values**2)))
        if singular_values.size == 3 and np.all(singular_values > 0.0)
        else float("inf")
    )
    return RefinedClusterPosition(
        position_m=np.asarray(solved.x, dtype=np.float64),
        covariance_m2=covariance_to_nested_list(covariance),
        confidence=float(np.clip(mean_peak_value * fit_quality, 0.0, 1.0)),
        gdop=gdop,
        range_projection_mode=RANGE_REFINED,
        range_observability=range_observability_from_covariance(covariance, solved.x - centroid_m),
        residual_rms_seconds=residual_rms_s,
        reference_sensor=reference_sensor,
        tdoa_s=dict(reference_tdoa_s),
        refined_by_least_squares=True,
    )
