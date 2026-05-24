"""Phase 2 localization algorithms: SRP-PHAT, MUSIC, and ESPRIT."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from minimappr.core.localization import LocalizationError, gcc_phat, speed_of_sound_mps
from minimappr.core.localization_uncertainty import (
    covariance_from_jacobian,
    covariance_to_nested_list,
    directional_covariance_m2,
    range_observability_from_covariance,
)
from minimappr.models import LocalizationResult
from minimappr.utils.audio import rms


EPSILON = 1e-9


def _minimum_position_std_m(sound_speed_mps: float, sample_rate_hz: int, interp: int) -> float:
    sample_period_s = 1.0 / max(sample_rate_hz * max(interp, 1), 1)
    return max(sound_speed_mps * sample_period_s, 0.05)


def _common_sensor_ids(
    sensor_positions: dict[str, np.ndarray],
    sensor_windows: dict[str, np.ndarray],
    *,
    min_required: int,
) -> list[str]:
    ids = sorted(sensor_id for sensor_id in sensor_positions if sensor_id in sensor_windows)
    if len(ids) < min_required:
        raise LocalizationError(f"Need at least {min_required} active sensors for localization")
    return ids


def _reference_sensor(sensor_windows: dict[str, np.ndarray], sensor_ids: list[str]) -> str:
    energies = {sensor_id: rms(sensor_windows[sensor_id]) for sensor_id in sensor_ids}
    return max(energies.items(), key=lambda item: item[1])[0]


def _predicted_tdoa(
    *,
    position: np.ndarray,
    sensor_positions: dict[str, np.ndarray],
    reference_sensor: str,
    meas_ids: list[str],
    sound_speed: float,
) -> dict[str, float]:
    ref_pos = sensor_positions[reference_sensor]
    dist_ref = np.linalg.norm(position - ref_pos) + EPSILON
    out: dict[str, float] = {}
    for sensor_id in meas_ids:
        dist_i = np.linalg.norm(position - sensor_positions[sensor_id]) + EPSILON
        out[sensor_id] = float((dist_i - dist_ref) / sound_speed)
    return out


def _tdoa_measurements(
    *,
    sensor_windows: dict[str, np.ndarray],
    sensor_ids: list[str],
    reference_sensor: str,
    sample_rate_hz: int,
    max_tau_s: float,
    interp: int,
) -> tuple[dict[str, float], list[float]]:
    ref_signal = sensor_windows[reference_sensor]
    tdoa_s: dict[str, float] = {}
    peaks: list[float] = []
    for sensor_id in sensor_ids:
        if sensor_id == reference_sensor:
            continue
        tau_s, peak = gcc_phat(
            signal=sensor_windows[sensor_id],
            reference_signal=ref_signal,
            sample_rate_hz=sample_rate_hz,
            max_tau_s=max_tau_s,
            interp=max(1, interp),
        )
        tdoa_s[sensor_id] = float(tau_s)
        peaks.append(float(peak))
    return tdoa_s, peaks


def _solve_position_from_tdoa(
    *,
    sensor_positions: dict[str, np.ndarray],
    reference_sensor: str,
    tdoa_s: dict[str, float],
    sound_speed: float,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    sensor_ids = sorted(sensor_positions.keys())
    if x0 is None:
        x0 = np.mean(np.vstack([sensor_positions[sensor_id] for sensor_id in sensor_ids]), axis=0)
    meas_ids = sorted(tdoa_s.keys())
    if not meas_ids:
        raise LocalizationError("No TDOA measurements available for solve")

    def residuals(position: np.ndarray) -> np.ndarray:
        predicted = _predicted_tdoa(
            position=position,
            sensor_positions=sensor_positions,
            reference_sensor=reference_sensor,
            meas_ids=meas_ids,
            sound_speed=sound_speed,
        )
        rows = [predicted[sensor_id] - tdoa_s[sensor_id] for sensor_id in meas_ids]
        return np.asarray(rows, dtype=np.float64)

    solved = least_squares(residuals, x0=x0, method="trf", loss="soft_l1")
    if not solved.success:
        raise LocalizationError("Least-squares position solve failed")

    position = solved.x
    residual = residuals(position)
    rmse_s = float(np.sqrt(np.mean(np.square(residual))))
    return position, rmse_s


def _tdoa_jacobian(
    *,
    position: np.ndarray,
    sensor_positions: dict[str, np.ndarray],
    reference_sensor: str,
    meas_ids: list[str],
    sound_speed: float,
) -> np.ndarray:
    ref_pos = sensor_positions[reference_sensor]
    dist_ref = np.linalg.norm(position - ref_pos) + EPSILON
    rows: list[np.ndarray] = []
    for sensor_id in meas_ids:
        pos_i = sensor_positions[sensor_id]
        dist_i = np.linalg.norm(position - pos_i) + EPSILON
        grad = ((position - pos_i) / (dist_i * sound_speed)) - ((position - ref_pos) / (dist_ref * sound_speed))
        rows.append(grad)
    return np.vstack(rows)


def _gdop(
    *,
    position: np.ndarray,
    sensor_positions: dict[str, np.ndarray],
    reference_sensor: str,
    meas_ids: list[str],
    sound_speed: float,
) -> float:
    jacobian = _tdoa_jacobian(
        position=position,
        sensor_positions=sensor_positions,
        reference_sensor=reference_sensor,
        meas_ids=meas_ids,
        sound_speed=sound_speed,
    )
    info = jacobian.T @ jacobian
    if np.linalg.cond(info) > 1e12:
        return float("inf")
    covariance = np.linalg.pinv(info)
    trace = float(np.trace(covariance))
    return float(np.sqrt(max(0.0, trace)))


def _confidence_from_fit(
    *,
    rmse_s: float,
    tdoa_s: dict[str, float],
    peaks: list[float],
    contrast: float,
) -> float:
    tau_scale = max(max((abs(v) for v in tdoa_s.values()), default=1e-5), 1e-5)
    fit_score = max(0.0, 1.0 - (rmse_s / tau_scale))
    peak_score = float(np.clip(np.mean(peaks), 0.0, 1.0)) if peaks else 0.5
    contrast_score = float(np.clip(contrast, 0.0, 1.0))
    score = (0.6 * fit_score) + (0.25 * peak_score) + (0.15 * contrast_score)
    return float(np.clip(score, 0.0, 1.0))


def _aperture_m(sensor_positions: dict[str, np.ndarray], sensor_ids: list[str]) -> float:
    positions = [sensor_positions[sensor_id] for sensor_id in sensor_ids]
    if len(positions) < 2:
        return 0.0
    return float(
        max(
            np.linalg.norm(positions[i] - positions[j])
            for i in range(len(positions))
            for j in range(i + 1, len(positions))
        )
    )


def _dominant_frequency_hz(
    signal: np.ndarray,
    sample_rate_hz: int,
    *,
    freq_min_hz: float,
    freq_max_hz: float,
) -> float:
    if signal.size <= 1:
        return max(300.0, freq_min_hz)
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate_hz)
    mask = (freqs >= freq_min_hz) & (freqs <= freq_max_hz)
    if not np.any(mask):
        return max(300.0, min(freq_max_hz, sample_rate_hz * 0.25))
    masked = spectrum[mask]
    if masked.size == 0:
        return max(300.0, min(freq_max_hz, sample_rate_hz * 0.25))
    idx = int(np.argmax(masked))
    return float(freqs[mask][idx])


def _estimate_source_count_mdl(eigenvalues: np.ndarray, snapshot_count: int) -> int:
    m = eigenvalues.size
    if m <= 2 or snapshot_count <= m:
        return 1

    lam = np.clip(np.sort(np.real(eigenvalues))[::-1], 1e-12, None)
    mdl_scores: list[float] = []
    for k in range(m):
        noise = lam[k:]
        p = noise.size
        if p <= 0:
            mdl_scores.append(float("inf"))
            continue
        gm = float(np.exp(np.mean(np.log(noise))))
        am = float(np.mean(noise))
        ratio = max(gm / (am + 1e-12), 1e-12)
        score = -snapshot_count * p * math.log(ratio) + 0.5 * k * (2 * m - k) * math.log(snapshot_count)
        mdl_scores.append(score)
    estimate = int(np.argmin(mdl_scores))
    return max(1, min(m - 1, estimate))


@dataclass(slots=True)
class FarFieldRangeEstimate:
    position: np.ndarray
    radius_m: float
    radial_std_m: float
    range_observability: float
    residual_rms_seconds: float
    range_projection_mode: str


def _far_field_range_fit(
    *,
    direction: np.ndarray,
    centroid: np.ndarray,
    sensor_positions: dict[str, np.ndarray],
    reference_sensor: str,
    measured_tdoa_s: dict[str, float],
    sound_speed: float,
    initial_range_m: float,
    max_range_m: float,
    minimum_time_std_s: float,
) -> FarFieldRangeEstimate:
    meas_ids = sorted(measured_tdoa_s.keys())
    if not meas_ids:
        raise LocalizationError("Cannot estimate range without TDOA measurements")
    direction = direction / (np.linalg.norm(direction) + EPSILON)

    def residuals(radius_vec: np.ndarray) -> np.ndarray:
        radius = float(radius_vec[0])
        position = centroid + direction * radius
        predicted = _predicted_tdoa(
            position=position,
            sensor_positions=sensor_positions,
            reference_sensor=reference_sensor,
            meas_ids=meas_ids,
            sound_speed=sound_speed,
        )
        return np.asarray(
            [predicted[sensor_id] - measured_tdoa_s[sensor_id] for sensor_id in meas_ids],
            dtype=np.float64,
        )

    solved = least_squares(
        residuals,
        x0=np.asarray([initial_range_m], dtype=np.float64),
        bounds=(np.asarray([0.05], dtype=np.float64), np.asarray([max_range_m], dtype=np.float64)),
        method="trf",
        loss="soft_l1",
    )
    if not solved.success:
        radius = initial_range_m
        radial_std_m = max(initial_range_m, max_range_m * 0.5, 1.0)
    else:
        radius = float(solved.x[0])
        rmse_at_solution_s = float(
            np.sqrt(np.mean(np.square(residuals(np.asarray([radius], dtype=np.float64)))))
        )
        radial_covariance_m2 = covariance_from_jacobian(
            np.asarray(solved.jac, dtype=np.float64).reshape(-1, 1),
            rmse_at_solution_s,
            minimum_time_std_s=minimum_time_std_s,
            minimum_std_m=max(sound_speed * minimum_time_std_s, 1.0),
        )
        radial_std_m = (
            float(np.sqrt(max(float(radial_covariance_m2[0, 0]), EPSILON)))
            if radial_covariance_m2 is not None
            else max(initial_range_m, 1.0)
        )
    position = centroid + direction * radius
    rmse_s = float(np.sqrt(np.mean(np.square(residuals(np.asarray([radius], dtype=np.float64))))))
    if not solved.success:
        range_observability = 0.0
    else:
        range_scale_m = max(abs(radius), 1.0)
        range_observability = float(np.clip(range_scale_m / (range_scale_m + radial_std_m), 0.0, 1.0))
    prior_radius_tolerance_m = max(0.5, initial_range_m * 0.02)
    range_projection_mode = (
        "prior_projected"
        if (
            not solved.success
            or abs(radius - initial_range_m) <= prior_radius_tolerance_m
            or range_observability < 0.10
            or radial_std_m >= max(radius, 1.0)
        )
        else "range_refined"
    )
    return FarFieldRangeEstimate(
        position=position,
        radius_m=float(radius),
        radial_std_m=float(radial_std_m),
        range_observability=range_observability,
        residual_rms_seconds=rmse_s,
        range_projection_mode=range_projection_mode,
    )


def _angular_search_directions(azimuth_step_deg: float, elevation_step_deg: float) -> np.ndarray:
    azimuths = np.deg2rad(np.arange(-180.0, 180.0, max(1.0, azimuth_step_deg)))
    elevations = np.deg2rad(np.arange(-65.0, 66.0, max(1.0, elevation_step_deg)))
    directions: list[np.ndarray] = []
    for elevation in elevations:
        cos_el = math.cos(float(elevation))
        sin_el = math.sin(float(elevation))
        for azimuth in azimuths:
            directions.append(
                np.asarray(
                    [cos_el * math.cos(float(azimuth)), cos_el * math.sin(float(azimuth)), sin_el],
                    dtype=np.float64,
                )
            )
    return np.vstack(directions)


def _far_field_covariance(
    *,
    direction: np.ndarray,
    range_estimate: FarFieldRangeEstimate,
    angular_std_rad: float,
    aperture_m: float,
    minimum_position_std_m: float,
) -> np.ndarray | None:
    lateral_std_m = max(
        minimum_position_std_m,
        range_estimate.radius_m * angular_std_rad,
        aperture_m * 0.5,
    )
    radial_std_m = max(range_estimate.radial_std_m, lateral_std_m)
    return directional_covariance_m2(
        direction,
        lateral_std_m=lateral_std_m,
        radial_std_m=radial_std_m,
        minimum_std_m=minimum_position_std_m,
    )


def _phat_correlation(
    signal: np.ndarray,
    reference_signal: np.ndarray,
    sample_rate_hz: int,
    max_tau_s: float,
    interp: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = signal.size + reference_signal.size
    if n <= 0:
        return np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)
    sig = np.fft.rfft(signal, n=n)
    ref = np.fft.rfft(reference_signal, n=n)
    response = sig * np.conj(ref)
    denom = np.abs(response)
    denom[denom < 1e-12] = 1e-12
    corr = np.fft.irfft(response / denom, n=interp * n)
    max_shift = int(interp * sample_rate_hz * max_tau_s)
    max_shift = max(1, min(max_shift, corr.size // 2))
    corr = np.concatenate((corr[-max_shift:], corr[: max_shift + 1]))
    lags = np.arange(-max_shift, max_shift + 1, dtype=np.float64) / float(interp * sample_rate_hz)
    return lags, corr.astype(np.float64)


def _grid_from_bounds(
    *,
    mins: np.ndarray,
    maxs: np.ndarray,
    resolution_m: float,
    max_points: int,
) -> np.ndarray:
    xs = np.arange(mins[0], maxs[0] + resolution_m * 0.5, resolution_m)
    ys = np.arange(mins[1], maxs[1] + resolution_m * 0.5, resolution_m)
    zs = np.arange(mins[2], maxs[2] + resolution_m * 0.5, resolution_m)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="xy"), axis=-1).reshape(-1, 3)
    if grid.shape[0] > max_points:
        step = int(math.ceil(grid.shape[0] / max_points))
        grid = grid[::step]
    return grid.astype(np.float64)


@dataclass(slots=True)
class SRPPhatLocalizer:
    max_tau_s: float = 0.02
    grid_resolution_m: float = 0.5
    search_padding_m: float = 2.0
    interp: int = 4
    max_grid_points: int = 60_000
    tight_array_aperture_m: float = 0.35
    far_field_default_range_m: float = 50.0
    far_field_max_range_m: float = 250.0
    far_field_azimuth_step_deg: float = 6.0
    far_field_elevation_step_deg: float = 8.0

    def localize(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
    ) -> LocalizationResult:
        sensor_ids = _common_sensor_ids(sensor_positions, sensor_windows, min_required=4)
        reference_sensor = _reference_sensor(sensor_windows, sensor_ids)
        sound_speed = speed_of_sound_mps(temperature_c=temperature_c, humidity_fraction=humidity_fraction)
        measured_tdoa, peaks = _tdoa_measurements(
            sensor_windows=sensor_windows,
            sensor_ids=sensor_ids,
            reference_sensor=reference_sensor,
            sample_rate_hz=sample_rate_hz,
            max_tau_s=self.max_tau_s,
            interp=self.interp,
        )
        meas_ids = sorted(measured_tdoa.keys())

        pos_matrix = np.vstack([sensor_positions[sensor_id] for sensor_id in sensor_ids])
        array_centroid = np.mean(pos_matrix, axis=0)
        array_aperture_m = _aperture_m(sensor_positions, sensor_ids)
        minimum_position_std_m = _minimum_position_std_m(sound_speed, sample_rate_hz, self.interp)

        pair_terms: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for sensor_a, sensor_b in itertools.combinations(sensor_ids, 2):
            pos_a = sensor_positions[sensor_a]
            pos_b = sensor_positions[sensor_b]
            pair_dist = float(np.linalg.norm(pos_a - pos_b))
            pair_tau = min(self.max_tau_s, (pair_dist / sound_speed) + (1.0 / sample_rate_hz))
            lags, corr = _phat_correlation(
                sensor_windows[sensor_a],
                sensor_windows[sensor_b],
                sample_rate_hz=sample_rate_hz,
                max_tau_s=max(pair_tau, 1.0 / sample_rate_hz),
                interp=max(1, self.interp),
            )
            pair_terms.append((pos_a, pos_b, lags, corr))

        if not pair_terms:
            raise LocalizationError("SRP-PHAT could not form sensor pairs")

        if array_aperture_m <= self.tight_array_aperture_m:
            mins = np.min(pos_matrix, axis=0) - self.search_padding_m
            maxs = np.max(pos_matrix, axis=0) + self.search_padding_m
            grid = _grid_from_bounds(
                mins=mins,
                maxs=maxs,
                resolution_m=max(0.05, self.grid_resolution_m),
                max_points=max(1, self.max_grid_points),
            )
            if grid.size == 0:
                raise LocalizationError("SRP-PHAT grid is empty")

            near_scores = np.zeros(grid.shape[0], dtype=np.float64)
            for pos_a, pos_b, lags, corr in pair_terms:
                lag0 = float(lags[0])
                step = float(lags[1] - lags[0]) if lags.size > 1 else (1.0 / sample_rate_hz)
                dist_a = np.linalg.norm(grid - pos_a[None, :], axis=1)
                dist_b = np.linalg.norm(grid - pos_b[None, :], axis=1)
                tau = (dist_a - dist_b) / sound_speed
                indices = np.rint((tau - lag0) / step).astype(np.int64)
                valid = (indices >= 0) & (indices < corr.size)
                near_scores[valid] += corr[indices[valid]]

            near_best_idx = int(np.argmax(near_scores))
            near_grid_position = grid[near_best_idx]
            try:
                near_best_position, near_rmse_s = _solve_position_from_tdoa(
                    sensor_positions=sensor_positions,
                    reference_sensor=reference_sensor,
                    tdoa_s=measured_tdoa,
                    sound_speed=sound_speed,
                    x0=near_grid_position,
                )
            except LocalizationError:
                near_best_position = near_grid_position
                near_predicted = _predicted_tdoa(
                    position=near_best_position,
                    sensor_positions=sensor_positions,
                    reference_sensor=reference_sensor,
                    meas_ids=meas_ids,
                    sound_speed=sound_speed,
                )
                near_residual = np.asarray(
                    [near_predicted[sensor_id] - measured_tdoa[sensor_id] for sensor_id in meas_ids],
                    dtype=np.float64,
                )
                near_rmse_s = float(np.sqrt(np.mean(np.square(near_residual))))

            near_covariance_m2 = covariance_from_jacobian(
                _tdoa_jacobian(
                    position=near_best_position,
                    sensor_positions=sensor_positions,
                    reference_sensor=reference_sensor,
                    meas_ids=meas_ids,
                    sound_speed=sound_speed,
                ),
                near_rmse_s,
                minimum_time_std_s=1.0 / max(sample_rate_hz * max(self.interp, 1), 1),
                minimum_std_m=max(self.grid_resolution_m * 0.5, minimum_position_std_m),
            )
            near_range_observability = range_observability_from_covariance(
                near_covariance_m2,
                near_best_position - array_centroid,
            )
            boundary_margin_m = max(0.05, self.grid_resolution_m)
            boundary_clamped = bool(
                np.any(np.abs(near_grid_position - mins) <= boundary_margin_m)
                or np.any(np.abs(maxs - near_grid_position) <= boundary_margin_m)
            )
            if boundary_clamped and near_range_observability is not None:
                near_range_observability = float(min(near_range_observability, 0.25))

            relative_tdoa_by_sensor = {reference_sensor: 0.0, **measured_tdoa}
            direction_rows: list[np.ndarray] = []
            direction_targets: list[float] = []
            for sensor_a, sensor_b in itertools.combinations(sensor_ids, 2):
                direction_rows.append(sensor_positions[sensor_b] - sensor_positions[sensor_a])
                direction_targets.append(
                    sound_speed
                    * (relative_tdoa_by_sensor[sensor_a] - relative_tdoa_by_sensor[sensor_b])
                )

            direction_matrix = np.vstack(direction_rows)
            target_vector = np.asarray(direction_targets, dtype=np.float64)
            best_direction, *_ = np.linalg.lstsq(direction_matrix, target_vector, rcond=None)
            direction_norm = float(np.linalg.norm(best_direction))
            if direction_norm < 1e-8:
                raise LocalizationError("SRP-PHAT tight-array direction solve failed")
            best_direction = best_direction / direction_norm
            direction_fit_residual = float(
                np.linalg.norm(direction_matrix @ best_direction - target_vector)
                / (np.linalg.norm(target_vector) + EPSILON)
            )
            far_scores = np.asarray([max(0.0, 1.0 - direction_fit_residual)], dtype=np.float64)
            far_best_idx = 0
            range_estimate = _far_field_range_fit(
                direction=best_direction,
                centroid=array_centroid,
                sensor_positions=sensor_positions,
                reference_sensor=reference_sensor,
                measured_tdoa_s=measured_tdoa,
                sound_speed=sound_speed,
                initial_range_m=max(0.5, self.far_field_default_range_m),
                max_range_m=max(self.far_field_default_range_m, self.far_field_max_range_m),
                minimum_time_std_s=1.0 / max(sample_rate_hz * max(self.interp, 1), 1),
            )
            far_best_position = range_estimate.position
            far_rmse_s = range_estimate.residual_rms_seconds
            far_covariance_m2 = _far_field_covariance(
                direction=best_direction,
                range_estimate=range_estimate,
                angular_std_rad=float(np.clip(direction_fit_residual, 0.05, 0.6)),
                aperture_m=array_aperture_m,
                minimum_position_std_m=minimum_position_std_m,
            )
            far_range_observability = range_estimate.range_observability

            scores = near_scores
            best_idx = near_best_idx
            best_position = near_best_position
            rmse_s = near_rmse_s
            covariance_m2 = near_covariance_m2
            range_observability = near_range_observability
            range_projection_mode = "bounded_grid_boundary" if boundary_clamped else "range_refined"
            far_selection_margin = 0.5 if boundary_clamped else 0.2
            if np.isfinite(far_rmse_s) and (
                not np.isfinite(rmse_s) or far_rmse_s <= (rmse_s * far_selection_margin)
            ):
                scores = far_scores
                best_idx = far_best_idx
                best_position = far_best_position
                rmse_s = far_rmse_s
                covariance_m2 = far_covariance_m2
                range_observability = far_range_observability
                range_projection_mode = range_estimate.range_projection_mode
        else:
            mins = np.min(pos_matrix, axis=0) - self.search_padding_m
            maxs = np.max(pos_matrix, axis=0) + self.search_padding_m
            grid = _grid_from_bounds(
                mins=mins,
                maxs=maxs,
                resolution_m=max(0.05, self.grid_resolution_m),
                max_points=max(1, self.max_grid_points),
            )
            if grid.size == 0:
                raise LocalizationError("SRP-PHAT grid is empty")

            scores = np.zeros(grid.shape[0], dtype=np.float64)
            for pos_a, pos_b, lags, corr in pair_terms:
                lag0 = float(lags[0])
                step = float(lags[1] - lags[0]) if lags.size > 1 else (1.0 / sample_rate_hz)
                dist_a = np.linalg.norm(grid - pos_a[None, :], axis=1)
                dist_b = np.linalg.norm(grid - pos_b[None, :], axis=1)
                tau = (dist_a - dist_b) / sound_speed
                indices = np.rint((tau - lag0) / step).astype(np.int64)
                valid = (indices >= 0) & (indices < corr.size)
                scores[valid] += corr[indices[valid]]

            best_idx = int(np.argmax(scores))
            best_position = grid[best_idx]
            predicted = _predicted_tdoa(
                position=best_position,
                sensor_positions=sensor_positions,
                reference_sensor=reference_sensor,
                meas_ids=meas_ids,
                sound_speed=sound_speed,
            )
            residual = np.asarray(
                [predicted[sensor_id] - measured_tdoa[sensor_id] for sensor_id in meas_ids],
                dtype=np.float64,
            )
            rmse_s = float(np.sqrt(np.mean(np.square(residual))))
            covariance_m2 = covariance_from_jacobian(
                _tdoa_jacobian(
                    position=best_position,
                    sensor_positions=sensor_positions,
                    reference_sensor=reference_sensor,
                    meas_ids=meas_ids,
                    sound_speed=sound_speed,
                ),
                rmse_s,
                minimum_time_std_s=1.0 / max(sample_rate_hz * max(self.interp, 1), 1),
                minimum_std_m=max(self.grid_resolution_m * 0.5, minimum_position_std_m),
            )
            range_observability = range_observability_from_covariance(
                covariance_m2,
                best_position - array_centroid,
            )
            range_projection_mode = "range_refined"

        gdop = _gdop(
            position=best_position,
            sensor_positions=sensor_positions,
            reference_sensor=reference_sensor,
            meas_ids=meas_ids,
            sound_speed=sound_speed,
        )

        median_score = float(np.median(scores))
        best_score = float(scores[best_idx])
        contrast = (best_score - median_score) / (abs(best_score) + EPSILON)
        confidence = _confidence_from_fit(
            rmse_s=rmse_s,
            tdoa_s=measured_tdoa,
            peaks=peaks,
            contrast=contrast,
        )

        return LocalizationResult(
            position_m=(float(best_position[0]), float(best_position[1]), float(best_position[2])),
            confidence=confidence,
            gdop=gdop,
            reference_sensor=reference_sensor,
            tdoa_s=measured_tdoa,
            position_covariance_m2=covariance_to_nested_list(covariance_m2),
            range_observability=range_observability,
            residual_rms_seconds=rmse_s,
            range_projection_mode=range_projection_mode,
        )


@dataclass(slots=True)
class MusicLocalizer:
    max_tau_s: float = 0.02
    azimuth_step_deg: float = 6.0
    elevation_step_deg: float = 8.0
    freq_min_hz: float = 300.0
    freq_max_hz: float = 3500.0
    source_count: int | None = None
    interp: int = 4
    far_field_default_range_m: float = 50.0
    far_field_max_range_m: float = 250.0

    def localize(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
    ) -> LocalizationResult:
        sensor_ids = _common_sensor_ids(sensor_positions, sensor_windows, min_required=4)
        sound_speed = speed_of_sound_mps(temperature_c=temperature_c, humidity_fraction=humidity_fraction)
        reference_sensor = _reference_sensor(sensor_windows, sensor_ids)
        measured_tdoa, peaks = _tdoa_measurements(
            sensor_windows=sensor_windows,
            sensor_ids=sensor_ids,
            reference_sensor=reference_sensor,
            sample_rate_hz=sample_rate_hz,
            max_tau_s=self.max_tau_s,
            interp=self.interp,
        )
        meas_ids = sorted(measured_tdoa.keys())

        x = np.vstack([sensor_windows[sensor_id] for sensor_id in sensor_ids]).astype(np.float64)
        covariance = (x @ x.T) / max(1, x.shape[1])
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        source_count = int(self.source_count) if self.source_count is not None else _estimate_source_count_mdl(
            eigenvalues, x.shape[1]
        )
        source_count = max(1, min(len(sensor_ids) - 1, source_count))
        noise_subspace = eigenvectors[:, source_count:]
        if noise_subspace.size == 0:
            noise_subspace = eigenvectors[:, -1:]

        reference_signal = sensor_windows[reference_sensor]
        dominant_freq_hz = _dominant_frequency_hz(
            reference_signal,
            sample_rate_hz,
            freq_min_hz=self.freq_min_hz,
            freq_max_hz=self.freq_max_hz,
        )

        sensor_matrix = np.vstack([sensor_positions[sensor_id] for sensor_id in sensor_ids])
        centroid = np.mean(sensor_matrix, axis=0)
        offsets = sensor_matrix - centroid[None, :]

        az_step = max(1.0, self.azimuth_step_deg)
        el_step = max(1.0, self.elevation_step_deg)
        azimuths = np.deg2rad(np.arange(-180.0, 180.0, az_step))
        elevations = np.deg2rad(np.arange(-65.0, 66.0, el_step))

        best_score = -1.0
        best_dir = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        scores: list[float] = []

        for elevation in elevations:
            cos_el = math.cos(float(elevation))
            sin_el = math.sin(float(elevation))
            for azimuth in azimuths:
                direction = np.asarray(
                    [cos_el * math.cos(float(azimuth)), cos_el * math.sin(float(azimuth)), sin_el],
                    dtype=np.float64,
                )
                delays = (offsets @ direction) / sound_speed
                steering = np.exp(-1j * 2.0 * np.pi * dominant_freq_hz * delays)
                proj = noise_subspace.conj().T @ steering
                denom = float(np.vdot(proj, proj).real)
                spectrum = 1.0 / max(denom, 1e-12)
                scores.append(spectrum)
                if spectrum > best_score:
                    best_score = spectrum
                    best_dir = direction

        aperture = max(0.2, _aperture_m(sensor_positions, sensor_ids))
        minimum_position_std_m = _minimum_position_std_m(sound_speed, sample_rate_hz, self.interp)
        range_estimate = _far_field_range_fit(
            direction=best_dir,
            centroid=centroid,
            sensor_positions=sensor_positions,
            reference_sensor=reference_sensor,
            measured_tdoa_s=measured_tdoa,
            sound_speed=sound_speed,
            initial_range_m=max(0.5, self.far_field_default_range_m),
            max_range_m=max(self.far_field_default_range_m, self.far_field_max_range_m),
            minimum_time_std_s=1.0 / max(sample_rate_hz * max(self.interp, 1), 1),
        )
        position = range_estimate.position
        rmse_s = range_estimate.residual_rms_seconds
        gdop = _gdop(
            position=position,
            sensor_positions=sensor_positions,
            reference_sensor=reference_sensor,
            meas_ids=meas_ids,
            sound_speed=sound_speed,
        )

        mean_score = float(np.mean(scores)) if scores else 0.0
        contrast = (best_score - mean_score) / (abs(best_score) + EPSILON)
        confidence = _confidence_from_fit(
            rmse_s=rmse_s,
            tdoa_s=measured_tdoa,
            peaks=peaks,
            contrast=contrast,
        )
        covariance_m2 = _far_field_covariance(
            direction=best_dir,
            range_estimate=range_estimate,
            angular_std_rad=math.radians(max(self.azimuth_step_deg, self.elevation_step_deg)),
            aperture_m=aperture,
            minimum_position_std_m=minimum_position_std_m,
        )
        return LocalizationResult(
            position_m=(float(position[0]), float(position[1]), float(position[2])),
            confidence=confidence,
            gdop=gdop,
            reference_sensor=reference_sensor,
            tdoa_s=measured_tdoa,
            position_covariance_m2=covariance_to_nested_list(covariance_m2),
            range_observability=range_estimate.range_observability,
            residual_rms_seconds=rmse_s,
            range_projection_mode=range_estimate.range_projection_mode,
        )


@dataclass(slots=True)
class EspritLocalizer:
    max_tau_s: float = 0.02
    freq_min_hz: float = 300.0
    freq_max_hz: float = 3500.0
    interp: int = 4
    far_field_default_range_m: float = 50.0
    far_field_max_range_m: float = 250.0

    def localize(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
    ) -> LocalizationResult:
        sensor_ids = _common_sensor_ids(sensor_positions, sensor_windows, min_required=4)
        sound_speed = speed_of_sound_mps(temperature_c=temperature_c, humidity_fraction=humidity_fraction)
        reference_sensor = _reference_sensor(sensor_windows, sensor_ids)
        measured_tdoa, peaks = _tdoa_measurements(
            sensor_windows=sensor_windows,
            sensor_ids=sensor_ids,
            reference_sensor=reference_sensor,
            sample_rate_hz=sample_rate_hz,
            max_tau_s=self.max_tau_s,
            interp=self.interp,
        )
        meas_ids = sorted(measured_tdoa.keys())

        x = np.vstack([sensor_windows[sensor_id] for sensor_id in sensor_ids]).astype(np.float64)
        covariance = (x @ x.T) / max(1, x.shape[1])
        _, eigenvectors = np.linalg.eigh(covariance)
        signal_vector = eigenvectors[:, -1]

        ref_idx = sensor_ids.index(reference_sensor)
        sensor_matrix = np.vstack([sensor_positions[sensor_id] for sensor_id in sensor_ids])
        centroid = np.mean(sensor_matrix, axis=0)
        dominant_freq_hz = _dominant_frequency_hz(
            sensor_windows[reference_sensor],
            sample_rate_hz,
            freq_min_hz=self.freq_min_hz,
            freq_max_hz=self.freq_max_hz,
        )

        a_rows: list[np.ndarray] = []
        b_rows: list[float] = []
        ref_pos = sensor_matrix[ref_idx]
        for idx, sensor_id in enumerate(sensor_ids):
            if sensor_id == reference_sensor:
                continue
            baseline = sensor_matrix[idx] - ref_pos
            phase = float(np.angle(signal_vector[idx] * np.conj(signal_vector[ref_idx])))
            delay_s = -phase / (2.0 * math.pi * dominant_freq_hz + EPSILON)
            a_rows.append(baseline)
            b_rows.append(sound_speed * delay_s)

        if len(a_rows) < 3:
            raise LocalizationError("ESPRIT requires at least three independent baselines")
        a = np.vstack(a_rows)
        b = np.asarray(b_rows, dtype=np.float64)
        direction, *_ = np.linalg.lstsq(a, b, rcond=None)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            raise LocalizationError("ESPRIT produced degenerate direction estimate")
        direction = direction / norm

        aperture = max(0.2, _aperture_m(sensor_positions, sensor_ids))
        minimum_position_std_m = _minimum_position_std_m(sound_speed, sample_rate_hz, self.interp)
        range_estimate = _far_field_range_fit(
            direction=direction,
            centroid=centroid,
            sensor_positions=sensor_positions,
            reference_sensor=reference_sensor,
            measured_tdoa_s=measured_tdoa,
            sound_speed=sound_speed,
            initial_range_m=max(0.5, self.far_field_default_range_m),
            max_range_m=max(self.far_field_default_range_m, self.far_field_max_range_m),
            minimum_time_std_s=1.0 / max(sample_rate_hz * max(self.interp, 1), 1),
        )
        position = range_estimate.position
        rmse_s = range_estimate.residual_rms_seconds
        gdop = _gdop(
            position=position,
            sensor_positions=sensor_positions,
            reference_sensor=reference_sensor,
            meas_ids=meas_ids,
            sound_speed=sound_speed,
        )

        linear_residual = float(np.linalg.norm(a @ direction - b) / (np.linalg.norm(b) + EPSILON))
        contrast = max(0.0, 1.0 - linear_residual)
        confidence = _confidence_from_fit(
            rmse_s=rmse_s,
            tdoa_s=measured_tdoa,
            peaks=peaks,
            contrast=contrast,
        )
        covariance_m2 = _far_field_covariance(
            direction=direction,
            range_estimate=range_estimate,
            angular_std_rad=float(np.clip(linear_residual, 0.05, 0.6)),
            aperture_m=aperture,
            minimum_position_std_m=minimum_position_std_m,
        )
        return LocalizationResult(
            position_m=(float(position[0]), float(position[1]), float(position[2])),
            confidence=confidence,
            gdop=gdop,
            reference_sensor=reference_sensor,
            tdoa_s=measured_tdoa,
            position_covariance_m2=covariance_to_nested_list(covariance_m2),
            range_observability=range_estimate.range_observability,
            residual_rms_seconds=rmse_s,
            range_projection_mode=range_estimate.range_projection_mode,
        )
