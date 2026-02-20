from __future__ import annotations

import numpy as np

from minimappr.core.localization import LocalizationEngine


def _shift_signal(signal: np.ndarray, sample_rate_hz: int, delay_s: float) -> np.ndarray:
    n = signal.size
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    shifted_t = t - delay_s
    return np.interp(shifted_t, t, signal, left=0.0, right=0.0).astype(np.float32)


def test_tdoa_localization_recovers_source_position() -> None:
    sample_rate_hz = 16000
    duration_s = 0.2
    n = int(sample_rate_hz * duration_s)
    t = np.arange(n, dtype=np.float64) / sample_rate_hz

    rng = np.random.default_rng(5)
    excitation = rng.standard_normal(n) * np.hanning(n)
    sound_speed = 343.2

    sensor_positions = {
        "s0": np.array([0.0, 0.0, 2.0]),
        "s1": np.array([6.0, 0.0, 2.0]),
        "s2": np.array([6.0, 0.4, 2.0]),
        "s3": np.array([6.2, 0.2, 2.2]),
        "s4": np.array([5.8, 0.2, 1.8]),
    }
    source = np.array([2.7, 3.3, 1.4])

    distances = {sensor_id: float(np.linalg.norm(source - pos)) for sensor_id, pos in sensor_positions.items()}
    windows = {}
    for sensor_id, distance in distances.items():
        windows[sensor_id] = _shift_signal(excitation, sample_rate_hz, distance / sound_speed)

    engine = LocalizationEngine(max_tau_s=0.03)
    result = engine.localize(
        sensor_positions=sensor_positions,
        sensor_windows=windows,
        sample_rate_hz=sample_rate_hz,
        temperature_c=20.0,
        humidity_fraction=0.5,
    )

    estimate = np.array(result.position_m)
    error = float(np.linalg.norm(estimate - source))

    assert error < 0.9
    assert result.confidence > 0.1
    assert np.isfinite(result.gdop)
