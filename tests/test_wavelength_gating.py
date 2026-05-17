from __future__ import annotations

import numpy as np
import pytest

from minimappr.core.localization import dominant_frequency_hz
from minimappr.core.localization_dispatch import LocalizationDispatcher
from tests.helpers import SIRITH_TETRA_SENSOR_OFFSETS_M, StubLocalizer


def _tetra_positions() -> dict[str, np.ndarray]:
    return {
        f"tetra:ch{index}": np.asarray(offset_m, dtype=np.float64)
        for index, offset_m in enumerate(SIRITH_TETRA_SENSOR_OFFSETS_M)
    }


def _tone_windows(
    positions: dict[str, np.ndarray],
    *,
    frequency_hz: float,
    sample_rate_hz: int,
    duration_seconds: float = 0.085,
) -> dict[str, np.ndarray]:
    sample_count = int(round(duration_seconds * sample_rate_hz))
    time_axis = np.arange(sample_count, dtype=np.float32) / float(sample_rate_hz)
    tone = np.sin(2.0 * np.pi * float(frequency_hz) * time_axis).astype(np.float32)
    return {sensor_id: tone.copy() for sensor_id in positions}


def test_dominant_frequency_hz_tracks_narrowband_tone() -> None:
    sample_rate_hz = 16_000
    time_axis = np.arange(4096, dtype=np.float32) / float(sample_rate_hz)
    tone = np.sin(2.0 * np.pi * 1200.0 * time_axis).astype(np.float32)

    estimate_hz = dominant_frequency_hz(tone, sample_rate_hz)

    assert estimate_hz == pytest.approx(1200.0, abs=80.0)


def test_dispatch_keeps_full_confidence_below_alias_cutoff() -> None:
    positions = _tetra_positions()
    sample_rate_hz = 16_000
    windows = _tone_windows(positions, frequency_hz=1200.0, sample_rate_hz=sample_rate_hz)
    dispatcher = LocalizationDispatcher(
        strategy="fixed",
        default_algorithm="gcc_phat",
        wavelength_gating_enabled=True,
        wavelength_penalty_floor=0.25,
        algorithms={"gcc_phat": StubLocalizer("gcc", 0.8)},
    )

    result = dispatcher.localize(positions, windows, sample_rate_hz, 20.0, 0.5)

    assert result.wavelength_factor == pytest.approx(1.0, abs=0.02)
    assert result.confidence == pytest.approx(0.8, abs=0.02)
    assert result.alias_cutoff_hz is not None
    assert result.dominant_frequency_hz is not None
    assert result.alias_cutoff_hz > result.dominant_frequency_hz


def test_dispatch_penalizes_confidence_above_alias_cutoff() -> None:
    positions = _tetra_positions()
    sample_rate_hz = 48_000
    windows = _tone_windows(positions, frequency_hz=12_000.0, sample_rate_hz=sample_rate_hz)
    dispatcher = LocalizationDispatcher(
        strategy="fixed",
        default_algorithm="gcc_phat",
        wavelength_gating_enabled=True,
        wavelength_penalty_floor=0.25,
        algorithms={"gcc_phat": StubLocalizer("gcc", 0.8)},
    )

    result = dispatcher.localize(positions, windows, sample_rate_hz, 20.0, 0.5)

    assert result.wavelength_factor is not None
    assert 0.25 <= result.wavelength_factor < 0.4
    assert result.confidence == pytest.approx(0.8 * result.wavelength_factor, abs=0.03)
    assert result.alias_cutoff_hz is not None
    assert result.dominant_frequency_hz is not None
    assert result.alias_cutoff_hz < result.dominant_frequency_hz