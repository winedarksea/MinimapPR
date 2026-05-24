"""TDOA-based source localization using GCC-PHAT and least squares."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from minimappr.core.localization_uncertainty import (
    covariance_from_jacobian,
    covariance_to_nested_list,
    embed_planar_covariance_m2,
    range_observability_from_covariance,
)
from minimappr.models import LocalizationResult
from minimappr.utils.audio import rms


class LocalizationError(RuntimeError):
    pass


def _minimum_position_std_m(sound_speed_mps: float, sample_rate_hz: int, interp_factor: int) -> float:
    sample_period_s = 1.0 / max(sample_rate_hz * max(interp_factor, 1), 1)
    return max(sound_speed_mps * sample_period_s, 0.05)


def speed_of_sound_mps(temperature_c: float, humidity_fraction: float) -> float:
    humidity_percent = max(0.0, min(1.0, humidity_fraction)) * 100.0
    return 331.3 + (0.606 * temperature_c) + (0.0124 * humidity_percent)


def dominant_frequency_hz(window: np.ndarray, sample_rate_hz: int) -> float:
    samples = np.asarray(window, dtype=np.float64).reshape(-1)
    if samples.size <= 1 or sample_rate_hz <= 0:
        return 0.0

    centered = samples - float(np.mean(samples))
    if not np.any(np.abs(centered) > 1e-12):
        return 0.0

    tapered = centered * np.hanning(centered.size)
    spectrum = np.fft.rfft(tapered)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(tapered.size, d=1.0 / float(sample_rate_hz))
    if power.size <= 1 or freqs.size <= 1:
        return 0.0

    power = power[1:]
    freqs = freqs[1:]
    total_power = float(np.sum(power))
    if total_power <= 1e-12:
        return 0.0
    return float(np.sum(freqs * power) / total_power)


def gcc_phat(
    signal: np.ndarray,
    reference_signal: np.ndarray,
    sample_rate_hz: int,
    max_tau_s: float,
    interp: int = 4,
) -> tuple[float, float]:
    n = signal.size + reference_signal.size
    if n <= 0:
        return 0.0, 0.0

    sig = np.fft.rfft(signal, n=n)
    ref = np.fft.rfft(reference_signal, n=n)
    response = sig * np.conj(ref)

    denom = np.abs(response)
    denom[denom < 1e-12] = 1e-12
    cross_corr = np.fft.irfft(response / denom, n=interp * n)

    max_shift = int(interp * sample_rate_hz * max_tau_s)
    max_shift = max(1, min(max_shift, cross_corr.size // 2))

    cross_corr = np.concatenate((cross_corr[-max_shift:], cross_corr[: max_shift + 1]))
    peak_index = int(np.argmax(np.abs(cross_corr)))
    shift = peak_index - max_shift
    tau = shift / float(interp * sample_rate_hz)

    peak = float(np.abs(cross_corr[peak_index]))
    return tau, peak


@dataclass(slots=True)
class LocalizationEngine:
    max_tau_s: float = 0.02
    interp_factor: int = 4

    def _validate_inputs(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
    ) -> None:
        for sensor_id, position in sensor_positions.items():
            pos = np.asarray(position, dtype=np.float64)
            if pos.shape != (3,):
                raise LocalizationError(f"Sensor position for {sensor_id} must be a 3-vector")
            if not np.all(np.isfinite(pos)):
                raise LocalizationError(f"Sensor position for {sensor_id} contains non-finite values")

        for sensor_id, window in sensor_windows.items():
            samples = np.asarray(window, dtype=np.float64)
            if samples.ndim != 1 or samples.size == 0:
                raise LocalizationError(f"Sensor window for {sensor_id} must be a non-empty 1-D array")
            if not np.all(np.isfinite(samples)):
                raise LocalizationError(f"Sensor window for {sensor_id} contains NaN/Inf values")

    def localize(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
        sensor_weights: dict[str, float] | None = None,
    ) -> LocalizationResult:
        if len(sensor_windows) < 4:
            raise LocalizationError("Need at least 4 active sensors for 3D TDOA localization")
        self._validate_inputs(sensor_positions=sensor_positions, sensor_windows=sensor_windows)

        w = sensor_weights or {}
        sensor_ids = sorted(sensor_windows.keys())
        energies = {sensor_id: rms(sensor_windows[sensor_id]) for sensor_id in sensor_ids}
        reference_sensor = max(energies.items(), key=lambda item: item[1])[0]
        reference_signal = sensor_windows[reference_sensor]
        ref_pos = sensor_positions[reference_sensor]
        ref_weight = w.get(reference_sensor, 1.0)

        tdoa_s: dict[str, float] = {}
        pair_weights: dict[str, float] = {}
        peaks: list[float] = []
        meas_ids: list[str] = []

        for sensor_id in sensor_ids:
            if sensor_id == reference_sensor:
                continue
            tau_s, peak = gcc_phat(
                signal=sensor_windows[sensor_id],
                reference_signal=reference_signal,
                sample_rate_hz=sample_rate_hz,
                max_tau_s=self.max_tau_s,
                interp=max(1, self.interp_factor),
            )
            if not np.isfinite(tau_s) or not np.isfinite(peak):
                raise LocalizationError(f"Non-finite GCC-PHAT output for sensor {sensor_id}")
            pw = min(ref_weight, w.get(sensor_id, 1.0))
            tdoa_s[sensor_id] = tau_s
            pair_weights[sensor_id] = pw
            peaks.append(peak * pw)
            meas_ids.append(sensor_id)

        if len(meas_ids) < 3:
            raise LocalizationError("Insufficient independent TDOA measurements")

        sound_speed = speed_of_sound_mps(temperature_c=temperature_c, humidity_fraction=humidity_fraction)

        points = np.vstack([sensor_positions[sid] for sid in sensor_ids])
        x0 = np.mean(points, axis=0)

        def residuals(position: np.ndarray) -> np.ndarray:
            dist_ref = np.linalg.norm(position - ref_pos) + 1e-9
            rows = []
            for sid in meas_ids:
                dist_i = np.linalg.norm(position - sensor_positions[sid]) + 1e-9
                predicted = (dist_i - dist_ref) / sound_speed
                rows.append(pair_weights.get(sid, 1.0) * (predicted - tdoa_s[sid]))
            return np.asarray(rows, dtype=np.float64)

        solved = least_squares(residuals, x0=x0, method="trf", loss="soft_l1")
        if not solved.success:
            raise LocalizationError("TDOA least-squares solve failed")

        position = solved.x
        if not np.all(np.isfinite(position)):
            raise LocalizationError("Localization solver returned non-finite position")
        residual = residuals(position)
        if not np.all(np.isfinite(residual)):
            raise LocalizationError("Localization residual contains non-finite values")
        rmse_s = float(np.sqrt(np.mean(np.square(residual))))
        tau_scale = max(max(abs(v) for v in tdoa_s.values()), 1e-5)
        confidence = max(0.0, min(1.0, 1.0 - (rmse_s / tau_scale)))

        gdop = self._gdop(
            position=position,
            meas_ids=meas_ids,
            sensor_positions=sensor_positions,
            reference_sensor=reference_sensor,
            sound_speed=sound_speed,
        )
        if np.isnan(gdop):
            raise LocalizationError("GDOP computation produced NaN")

        # Correlation peak quality can significantly degrade when SNR is poor.
        if peaks:
            confidence *= float(np.clip(np.mean(peaks), 0.0, 1.0))

        minimum_position_std_m = _minimum_position_std_m(
            sound_speed,
            sample_rate_hz,
            self.interp_factor,
        )
        position_covariance_m2 = covariance_from_jacobian(
            solved.jac,
            rmse_s,
            minimum_time_std_s=1.0 / max(sample_rate_hz * max(self.interp_factor, 1), 1),
            minimum_std_m=minimum_position_std_m,
        )
        range_observability = range_observability_from_covariance(
            position_covariance_m2,
            position - np.mean(points, axis=0),
        )

        return LocalizationResult(
            position_m=(float(position[0]), float(position[1]), float(position[2])),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            gdop=gdop,
            reference_sensor=reference_sensor,
            tdoa_s=tdoa_s,
            position_covariance_m2=covariance_to_nested_list(position_covariance_m2),
            range_observability=range_observability,
            residual_rms_seconds=rmse_s,
        )

    def localize_2d(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
        fixed_z_m: float | None = None,
        sensor_weights: dict[str, float] | None = None,
    ) -> LocalizationResult:
        if len(sensor_windows) < 3:
            raise LocalizationError("Need at least 3 active sensors for 2D TDOA localization")
        self._validate_inputs(sensor_positions=sensor_positions, sensor_windows=sensor_windows)

        w = sensor_weights or {}
        sensor_ids = sorted(sensor_windows.keys())
        energies = {sensor_id: rms(sensor_windows[sensor_id]) for sensor_id in sensor_ids}
        reference_sensor = max(energies.items(), key=lambda item: item[1])[0]
        reference_signal = sensor_windows[reference_sensor]
        ref_pos = sensor_positions[reference_sensor]
        ref_weight = w.get(reference_sensor, 1.0)

        tdoa_s: dict[str, float] = {}
        pair_weights: dict[str, float] = {}
        peaks: list[float] = []
        meas_ids: list[str] = []

        for sensor_id in sensor_ids:
            if sensor_id == reference_sensor:
                continue
            tau_s, peak = gcc_phat(
                signal=sensor_windows[sensor_id],
                reference_signal=reference_signal,
                sample_rate_hz=sample_rate_hz,
                max_tau_s=self.max_tau_s,
                interp=max(1, self.interp_factor),
            )
            if not np.isfinite(tau_s) or not np.isfinite(peak):
                raise LocalizationError(f"Non-finite GCC-PHAT output for sensor {sensor_id}")
            pw = min(ref_weight, w.get(sensor_id, 1.0))
            tdoa_s[sensor_id] = tau_s
            pair_weights[sensor_id] = pw
            peaks.append(peak * pw)
            meas_ids.append(sensor_id)

        if len(meas_ids) < 2:
            raise LocalizationError("Insufficient independent TDOA measurements for 2D solve")

        sound_speed = speed_of_sound_mps(temperature_c=temperature_c, humidity_fraction=humidity_fraction)
        points = np.vstack([sensor_positions[sid] for sid in sensor_ids])
        z_m = float(fixed_z_m if fixed_z_m is not None else np.mean(points[:, 2]))
        xy0 = np.mean(points[:, :2], axis=0)

        def residuals_xy(xy: np.ndarray) -> np.ndarray:
            pos = np.asarray([xy[0], xy[1], z_m], dtype=np.float64)
            dist_ref = np.linalg.norm(pos - ref_pos) + 1e-9
            rows = []
            for sid in meas_ids:
                dist_i = np.linalg.norm(pos - sensor_positions[sid]) + 1e-9
                predicted = (dist_i - dist_ref) / sound_speed
                rows.append(pair_weights.get(sid, 1.0) * (predicted - tdoa_s[sid]))
            return np.asarray(rows, dtype=np.float64)

        solved = least_squares(residuals_xy, x0=xy0, method="trf", loss="soft_l1")
        if not solved.success:
            raise LocalizationError("2D TDOA least-squares solve failed")

        xy = solved.x
        if not np.all(np.isfinite(xy)):
            raise LocalizationError("2D localization solver returned non-finite position")
        position = np.asarray([xy[0], xy[1], z_m], dtype=np.float64)
        residual = residuals_xy(xy)
        if not np.all(np.isfinite(residual)):
            raise LocalizationError("2D localization residual contains non-finite values")
        rmse_s = float(np.sqrt(np.mean(np.square(residual))))
        tau_scale = max(max(abs(v) for v in tdoa_s.values()), 1e-5)
        confidence = max(0.0, min(1.0, 1.0 - (rmse_s / tau_scale)))
        if peaks:
            confidence *= float(np.clip(np.mean(peaks), 0.0, 1.0))

        gdop = self._gdop_2d(
            position=position,
            meas_ids=meas_ids,
            sensor_positions=sensor_positions,
            reference_sensor=reference_sensor,
            sound_speed=sound_speed,
        )
        if np.isnan(gdop):
            raise LocalizationError("2D GDOP computation produced NaN")

        minimum_position_std_m = _minimum_position_std_m(
            sound_speed,
            sample_rate_hz,
            self.interp_factor,
        )
        planar_covariance_m2 = covariance_from_jacobian(
            solved.jac,
            rmse_s,
            minimum_time_std_s=1.0 / max(sample_rate_hz * max(self.interp_factor, 1), 1),
            minimum_std_m=minimum_position_std_m,
        )
        position_covariance_m2 = embed_planar_covariance_m2(
            planar_covariance_m2,
            vertical_std_m=minimum_position_std_m,
            minimum_std_m=minimum_position_std_m,
        )
        range_observability = range_observability_from_covariance(
            position_covariance_m2,
            position - np.mean(points, axis=0),
        )

        return LocalizationResult(
            position_m=(float(position[0]), float(position[1]), float(position[2])),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            gdop=gdop,
            reference_sensor=reference_sensor,
            tdoa_s=tdoa_s,
            position_covariance_m2=covariance_to_nested_list(position_covariance_m2),
            range_observability=range_observability,
            residual_rms_seconds=rmse_s,
        )

    @staticmethod
    def _gdop(
        position: np.ndarray,
        meas_ids: list[str],
        sensor_positions: dict[str, np.ndarray],
        reference_sensor: str,
        sound_speed: float,
    ) -> float:
        ref_pos = sensor_positions[reference_sensor]
        dist_ref = np.linalg.norm(position - ref_pos) + 1e-9

        jacobian_rows: list[np.ndarray] = []
        for sid in meas_ids:
            pos_i = sensor_positions[sid]
            dist_i = np.linalg.norm(position - pos_i) + 1e-9

            grad = ((position - pos_i) / (dist_i * sound_speed)) - (
                (position - ref_pos) / (dist_ref * sound_speed)
            )
            jacobian_rows.append(grad)

        jacobian = np.vstack(jacobian_rows)
        info = jacobian.T @ jacobian
        if np.linalg.cond(info) > 1e12:
            return float("inf")

        covariance = np.linalg.pinv(info)
        trace = float(np.trace(covariance))
        return float(np.sqrt(max(0.0, trace)))

    @staticmethod
    def _gdop_2d(
        position: np.ndarray,
        meas_ids: list[str],
        sensor_positions: dict[str, np.ndarray],
        reference_sensor: str,
        sound_speed: float,
    ) -> float:
        ref_pos = sensor_positions[reference_sensor]
        dist_ref = np.linalg.norm(position - ref_pos) + 1e-9
        jacobian_rows: list[np.ndarray] = []
        for sid in meas_ids:
            pos_i = sensor_positions[sid]
            dist_i = np.linalg.norm(position - pos_i) + 1e-9
            grad3 = ((position - pos_i) / (dist_i * sound_speed)) - (
                (position - ref_pos) / (dist_ref * sound_speed)
            )
            jacobian_rows.append(grad3[:2])

        jacobian = np.vstack(jacobian_rows)
        info = jacobian.T @ jacobian
        if np.linalg.cond(info) > 1e12:
            return float("inf")
        covariance = np.linalg.pinv(info)
        trace = float(np.trace(covariance))
        return float(np.sqrt(max(0.0, trace)))
