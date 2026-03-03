from __future__ import annotations

import numpy as np

from minimappr.models import LocalizationResult


def shift_signal(signal: np.ndarray, sample_rate_hz: int, delay_s: float) -> np.ndarray:
    """Delay or advance a 1-D signal by *delay_s* using linear interpolation."""
    n = signal.size
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    shifted_t = t - delay_s
    return np.interp(shifted_t, t, signal, left=0.0, right=0.0).astype(np.float32)


class StubLocalizer:
    """Simple deterministic localizer stub for dispatcher strategy tests."""

    def __init__(self, name: str, confidence: float) -> None:
        self.name = name
        self.confidence = confidence
        self.calls = 0

    def localize(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
    ) -> LocalizationResult:
        del sensor_windows, sample_rate_hz, temperature_c, humidity_fraction
        self.calls += 1
        reference_sensor = sorted(sensor_positions.keys())[0]
        return LocalizationResult(
            position_m=(0.0, 0.0, 0.0),
            confidence=self.confidence,
            gdop=1.0,
            reference_sensor=reference_sensor,
            tdoa_s={},
        )
