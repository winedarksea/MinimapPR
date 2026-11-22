"""Sub-band flux onset detection (Field Mode W2-onset-srp)."""

from __future__ import annotations

import numpy as np

from minimappr.config import Settings
from minimappr.core.onset_detection import build_subband_history, detect_onsets

SAMPLE_RATE_HZ = 16_000


def _detect(signal: np.ndarray):
    config = Settings().onset_srp_clustering_config()
    history = build_subband_history(
        signal,
        sample_rate_hz=SAMPLE_RATE_HZ,
        hop_seconds=config.hop_seconds,
        band_min_hz=config.band_min_hz,
        band_max_hz=config.band_max_hz,
        band_count=config.subband_count,
    )
    return detect_onsets(
        history,
        sample_rate_hz=SAMPLE_RATE_HZ,
        flux_threshold_mad_multiplier=config.flux_threshold_mad_multiplier,
        min_separation_seconds=config.min_separation_seconds,
        max_onsets=config.max_onsets_per_window,
        active_band_snr_db=config.active_band_snr_db,
        frame_seconds=config.frame_seconds,
    )


def _tone(start_s: float, duration_s: float, frequency_hz: float, amplitude: float, total_s: float) -> np.ndarray:
    out = np.zeros(int(total_s * SAMPLE_RATE_HZ))
    t = np.arange(int(duration_s * SAMPLE_RATE_HZ)) / SAMPLE_RATE_HZ
    start = int(start_s * SAMPLE_RATE_HZ)
    out[start : start + t.size] = amplitude * np.sin(2 * np.pi * frequency_hz * t) * np.hanning(t.size) ** 0.1
    return out


def test_onsets_land_at_note_starts_with_their_own_band() -> None:
    rng = np.random.default_rng(1)
    signal = rng.normal(0, 0.002, int(1.5 * SAMPLE_RATE_HZ))
    signal += _tone(0.3, 0.4, 2300.0, 0.1, 1.5) + _tone(0.5, 0.3, 4000.0, 0.1, 1.5)
    onsets = sorted(_detect(signal), key=lambda onset: onset.sample_index)
    times = [onset.sample_index / SAMPLE_RATE_HZ for onset in onsets]
    assert any(abs(t - 0.3) < 0.04 for t in times)
    assert any(abs(t - 0.5) < 0.04 for t in times)
    second = min(onsets, key=lambda onset: abs(onset.sample_index / SAMPLE_RATE_HZ - 0.5))
    # The second note starts under the first: its band must be its own, not the union.
    assert second.band_min_hz > 3000.0 and second.band_max_hz > 4000.0


def test_faint_note_is_found_relative_to_the_noise_floor() -> None:
    rng = np.random.default_rng(2)
    noise_rms = 1e-5
    signal = rng.normal(0, noise_rms, int(1.5 * SAMPLE_RATE_HZ)) + _tone(0.6, 0.3, 3000.0, 2e-4, 1.5)
    onsets = _detect(signal)
    assert any(abs(onset.sample_index / SAMPLE_RATE_HZ - 0.6) < 0.04 for onset in onsets)


def test_pure_noise_yields_few_onsets() -> None:
    rng = np.random.default_rng(3)
    onsets = _detect(rng.normal(0, 0.01, int(1.5 * SAMPLE_RATE_HZ)))
    assert len(onsets) <= 2
