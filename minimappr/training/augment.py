"""Waveform augmentation + synthetic ambience for the drone-head trainer.

Ported from the audio_app reference (``scripts/synthetic_ambient.py``):

* :func:`synthesize_ambient_windows` — deterministic synthetic AMBIENT windows
  rotating over four noise profiles; used as extra negatives to harden the
  benign-background class against false positives.
* :func:`augment_waveform` — deterministic per-(content, seed) time-shift +
  SNR-relative additive noise + attenuation-only gain, used to multiply scarce
  real examples (notably the 4 coyote recordings).

Both are **train-only** by policy — val/test folds see real audio only.
"""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np

SAMPLE_RATE = 16_000

AMBIENT_NOISE_PROFILE_NAMES = [
    "band_limited_white_brown",
    "band_limited_pink_brown",
    "lfo_modulated_noise",
    "silence_or_dropout",
]

SYNTHETIC_AMBIENT_LOW_CUTOFF_HZ = 100.0
SYNTHETIC_AMBIENT_HIGH_CUTOFF_HZ = 4000.0


# --------------------------------------------------------------------------- #
# Synthetic ambience (ported verbatim from synthetic_ambient.py)
# --------------------------------------------------------------------------- #
def fft_bandpass_filter(
    samples: np.ndarray,
    sample_rate: int,
    low_cutoff_hz: float = SYNTHETIC_AMBIENT_LOW_CUTOFF_HZ,
    high_cutoff_hz: float = SYNTHETIC_AMBIENT_HIGH_CUTOFF_HZ,
) -> np.ndarray:
    """Band-limit synthetic ambience to a detector-audio-like passband."""
    if samples.size == 0:
        return samples.astype(np.float32)
    spectrum = np.fft.rfft(samples.astype(np.float32))
    frequencies_hz = np.fft.rfftfreq(samples.size, d=1.0 / float(sample_rate))
    passband_mask = (frequencies_hz >= low_cutoff_hz) & (frequencies_hz <= high_cutoff_hz)
    spectrum[~passband_mask] = 0.0
    return np.fft.irfft(spectrum, n=samples.size).astype(np.float32)


def synthesize_ambient_windows(
    count: int,
    window_seconds: float,
    seed: int,
    sample_rate: int = SAMPLE_RATE,
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(variant_key, waveform)`` deterministic synthetic AMBIENT windows.

    Rotates over the four profiles (white/brown weighted twice per the reference
    quota). ``variant_key`` is self-describing (``synth-{profile}-{index}``) so it
    can seed a stable embedding cache key.
    """
    if count <= 0:
        return
    window_size = int(round(window_seconds * sample_rate))
    rng = np.random.default_rng(seed)
    generators: list[tuple[str, Callable[[np.random.Generator, int, int], np.ndarray]]] = [
        ("band_limited_white_brown", _band_limited_white_brown_window),
        ("band_limited_white_brown", _band_limited_white_brown_window),
        ("band_limited_pink_brown", _band_limited_pink_brown_window),
        ("lfo_modulated_noise", _lfo_modulated_noise_window),
        ("silence_or_dropout", _silence_or_dropout_window),
    ]
    for index in range(count):
        profile, generator = generators[index % len(generators)]
        wave = _clip_float32(generator(rng, window_size, sample_rate))
        yield f"synth-{profile}-{index}", wave


def _band_limited_white_brown_window(rng, window_size, sample_rate):
    white = rng.normal(0.0, 1.0, window_size).astype(np.float32)
    brown = _brown_noise(rng, window_size)
    white_weight = float(rng.uniform(0.35, 0.8))
    mixed = white_weight * white + (1.0 - white_weight) * brown
    return _scale_to_rms(
        fft_bandpass_filter(mixed, sample_rate), target_rms=float(rng.uniform(0.05, 0.15))
    )


def _band_limited_pink_brown_window(rng, window_size, sample_rate):
    pink = _pink_noise(rng, window_size)
    brown = _brown_noise(rng, window_size)
    pink_weight = float(rng.uniform(0.45, 0.75))
    mixed = pink_weight * pink + (1.0 - pink_weight) * brown
    return _scale_to_rms(
        fft_bandpass_filter(mixed, sample_rate), target_rms=float(rng.uniform(0.05, 0.15))
    )


def _lfo_modulated_noise_window(rng, window_size, sample_rate):
    white = rng.normal(0.0, 1.0, window_size).astype(np.float32)
    pink = _pink_noise(rng, window_size)
    brown = _brown_noise(rng, window_size)
    mixed = 0.35 * white + 0.4 * pink + 0.25 * brown
    band_limited = fft_bandpass_filter(mixed, sample_rate)
    time_seconds = np.arange(window_size, dtype=np.float32) / float(sample_rate)
    lfo_hz = float(rng.uniform(1.0, 3.0))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    modulation_depth = float(rng.uniform(0.35, 0.75))
    envelope = 1.0 - modulation_depth * (0.5 + 0.5 * np.sin(2.0 * np.pi * lfo_hz * time_seconds + phase))
    modulated = band_limited * envelope.astype(np.float32)
    return _scale_to_rms(modulated, target_rms=float(rng.uniform(0.05, 0.15)))


def _silence_or_dropout_window(rng, window_size, sample_rate):
    if bool(rng.integers(0, 2)):
        return np.zeros(window_size, dtype=np.float32)
    base = _scale_to_rms(
        fft_bandpass_filter(_pink_noise(rng, window_size), sample_rate),
        target_rms=float(rng.uniform(0.035, 0.08)),
    )
    dropout_start = int(
        rng.integers(window_size // 8, max(window_size // 8 + 1, window_size * 5 // 8))
    )
    dropout_length = int(
        rng.integers(window_size // 12, max(window_size // 12 + 1, window_size // 3))
    )
    dropout_end = min(window_size, dropout_start + dropout_length)
    envelope = np.ones(window_size, dtype=np.float32)
    envelope[dropout_start:dropout_end] = 0.0
    return base * envelope


def _brown_noise(rng, window_size):
    brown = np.cumsum(rng.normal(0.0, 0.08, window_size)).astype(np.float32)
    return _normalize_peak(brown)


def _pink_noise(rng, window_size):
    white_spectrum = np.fft.rfft(rng.normal(0.0, 1.0, window_size).astype(np.float32))
    frequencies = np.fft.rfftfreq(window_size)
    scale = np.ones_like(frequencies, dtype=np.float32)
    non_zero = frequencies > 0
    scale[non_zero] = 1.0 / np.sqrt(frequencies[non_zero])
    scale[~non_zero] = 0.0
    pink = np.fft.irfft(white_spectrum * scale, n=window_size).astype(np.float32)
    return _normalize_peak(pink)


def _normalize_peak(samples):
    max_abs = float(np.max(np.abs(samples))) if samples.size else 0.0
    if max_abs <= 1e-8:
        return np.zeros_like(samples, dtype=np.float32)
    return (samples / max_abs).astype(np.float32)


def _scale_to_rms(samples, target_rms):
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float32)))) if samples.size else 0.0
    if rms <= 1e-8:
        return np.zeros_like(samples, dtype=np.float32)
    return (samples * (target_rms / rms)).astype(np.float32)


def _clip_float32(samples):
    return np.clip(samples, -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Real-example augmentation
# --------------------------------------------------------------------------- #
def augment_waveform(wave: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic augmentation of ``wave`` for the given ``seed``.

    Applies, in order: time-shift by U(-10%, +10%) with zero-fill; additive
    Gaussian noise at U(25, 40) dB SNR relative to the window RMS; random
    attenuation-only gain of U(-20, 0) dB. Deterministic per (content-length,
    seed) so results are cacheable.
    """
    wave = np.asarray(wave, dtype=np.float32)
    n = wave.size
    if n == 0:
        return wave.copy()
    rng = np.random.default_rng(seed)

    # Time-shift ±10% with zero-fill.
    max_shift = int(0.1 * n)
    shift = int(rng.integers(-max_shift, max_shift + 1)) if max_shift > 0 else 0
    shifted = np.zeros_like(wave)
    if shift > 0:
        shifted[shift:] = wave[: n - shift]
    elif shift < 0:
        shifted[: n + shift] = wave[-shift:]
    else:
        shifted[:] = wave

    # Additive Gaussian noise at target SNR (dB) relative to signal RMS.
    signal_rms = float(np.sqrt(np.mean(np.square(shifted, dtype=np.float64)) + 1e-12))
    snr_db = float(rng.uniform(25.0, 40.0))
    noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    noisy = shifted + rng.normal(0.0, noise_rms, n).astype(np.float32)

    # Attenuation-only gain U(-20, 0) dB.
    gain_db = float(rng.uniform(-20.0, 0.0))
    gained = noisy * (10.0 ** (gain_db / 20.0))

    return _clip_float32(gained)


__all__ = [
    "AMBIENT_NOISE_PROFILE_NAMES",
    "augment_waveform",
    "fft_bandpass_filter",
    "synthesize_ambient_windows",
]
