"""Beamforming primitives for Phase 2 localization and classification."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


EPSILON = 1e-12


def _ordered_sensor_ids(
    sensor_positions: dict[str, np.ndarray],
    sensor_windows: dict[str, np.ndarray],
) -> list[str]:
    return sorted(sensor_id for sensor_id in sensor_positions if sensor_id in sensor_windows)


def _fractional_shift(samples: np.ndarray, sample_rate_hz: int, shift_s: float) -> np.ndarray:
    if samples.size <= 1:
        return samples.astype(np.float32, copy=True)
    t = np.arange(samples.size, dtype=np.float64) / float(sample_rate_hz)
    shifted_t = t - shift_s
    shifted = np.interp(shifted_t, t, samples, left=0.0, right=0.0)
    return shifted.astype(np.float32)


def _steering_delays_s(
    *,
    sensor_positions: dict[str, np.ndarray],
    sensor_ids: list[str],
    steer_position_m: tuple[float, float, float],
    sound_speed_mps: float,
) -> np.ndarray:
    target = np.asarray(steer_position_m, dtype=np.float64)
    distances = np.asarray(
        [np.linalg.norm(sensor_positions[sensor_id] - target) for sensor_id in sensor_ids],
        dtype=np.float64,
    )
    delays = distances / max(sound_speed_mps, 1.0)
    delays -= np.min(delays)
    return delays


@dataclass(slots=True)
class DelayAndSumBeamformer:
    def beamform(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        steer_position_m: tuple[float, float, float],
        sound_speed_mps: float = 343.2,
        frequency_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        del frequency_weights
        sensor_ids = _ordered_sensor_ids(sensor_positions, sensor_windows)
        if not sensor_ids:
            return np.zeros(0, dtype=np.float32)
        if len(sensor_ids) == 1:
            return sensor_windows[sensor_ids[0]].astype(np.float32, copy=True)

        delays = _steering_delays_s(
            sensor_positions=sensor_positions,
            sensor_ids=sensor_ids,
            steer_position_m=steer_position_m,
            sound_speed_mps=sound_speed_mps,
        )
        aligned = []
        for sensor_id, delay_s in zip(sensor_ids, delays, strict=True):
            aligned.append(_fractional_shift(sensor_windows[sensor_id], sample_rate_hz, shift_s=-float(delay_s)))
        stacked = np.vstack(aligned)
        weights = np.ones(stacked.shape[0], dtype=np.float64)
        weights /= np.sum(weights)
        output = np.sum(stacked * weights[:, None], axis=0)
        return output.astype(np.float32)


@dataclass(slots=True)
class MVDRBeamformer:
    diagonal_loading: float = 1e-3
    freq_min_hz: float = 200.0
    freq_max_hz: float = 5000.0

    def beamform(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        steer_position_m: tuple[float, float, float],
        sound_speed_mps: float = 343.2,
        frequency_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        sensor_ids = _ordered_sensor_ids(sensor_positions, sensor_windows)
        if not sensor_ids:
            return np.zeros(0, dtype=np.float32)
        if len(sensor_ids) == 1:
            return sensor_windows[sensor_ids[0]].astype(np.float32, copy=True)

        x = np.vstack([sensor_windows[sensor_id] for sensor_id in sensor_ids]).astype(np.float64)
        n = x.shape[1]
        if n == 0:
            return np.zeros(0, dtype=np.float32)

        delays = _steering_delays_s(
            sensor_positions=sensor_positions,
            sensor_ids=sensor_ids,
            steer_position_m=steer_position_m,
            sound_speed_mps=sound_speed_mps,
        )

        spectrum = np.fft.rfft(x, axis=1)
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
        y_spec = np.zeros(freqs.size, dtype=np.complex128)

        identity = np.eye(len(sensor_ids), dtype=np.complex128)
        loading = max(self.diagonal_loading, 1e-9)
        if frequency_weights is not None and frequency_weights.size == freqs.size:
            freq_weight = np.asarray(frequency_weights, dtype=np.float64)
        else:
            freq_weight = np.ones(freqs.size, dtype=np.float64)

        for idx, freq_hz in enumerate(freqs):
            x_f = spectrum[:, idx]
            if freq_hz <= 0.0:
                y_spec[idx] = np.mean(x_f)
                continue
            if freq_hz < self.freq_min_hz or freq_hz > self.freq_max_hz:
                y_spec[idx] = np.mean(x_f)
                continue

            steering = np.exp(-1j * 2.0 * math.pi * freq_hz * delays)
            covariance = np.outer(x_f, np.conj(x_f)) + (loading * identity)
            inv_cov = np.linalg.pinv(covariance)
            numer = inv_cov @ steering
            denom = np.vdot(steering, numer)
            if abs(denom) < EPSILON:
                y_spec[idx] = np.mean(x_f)
                continue
            weights = numer / denom
            y_spec[idx] = np.vdot(weights, x_f) * float(freq_weight[idx])

        y = np.fft.irfft(y_spec, n=n)
        return y.astype(np.float32)
