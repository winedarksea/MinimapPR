"""Leave-cluster-out masking for LOCO attribution (Field Mode W3-attribution, S5).

Replaces the time-frequency regions of *other* clusters in a classification
signal with noise at the signal's own local floor (never zeros: a hole in the
spectrogram is out of the classifier's training distribution, while the
noise floor is what that region would hold if the other bird had been silent).

A region starts at a cluster onset's predicted arrival on the classifying node
and runs while the onset's band stays active above the floor, so a 0.6 s trill
is covered without knowing its duration up front. The classifier still sees a
full-length natural recording.

Works on in-memory numpy arrays only; nothing is written to disk.
Pure and synchronous: callers run it off the event loop (asyncio.to_thread).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import istft, stft

PERCENT = 100.0
POWER_FLOOR = 1e-30
STFT_OVERLAP_FRACTION = 0.75


@dataclass(frozen=True, slots=True)
class MaskRegionSeed:
    """Onset of another cluster inside the signal (seconds from signal start)."""

    onset_seconds: float
    band_min_hz: float
    band_max_hz: float


@dataclass(frozen=True, slots=True)
class MaskingParameters:
    stft_frame_seconds: float
    noise_floor_percentile: float
    active_snr_db: float
    release_seconds: float
    max_region_seconds: float
    band_padding_hz: float
    pre_roll_seconds: float


@dataclass(frozen=True, slots=True)
class MaskedSignal:
    samples: np.ndarray
    masked_cell_fraction: float  # share of time-frequency cells replaced
    region_seconds: tuple[float, ...]  # duration masked per seed


def _rayleigh_scale_from_percentile(magnitude_percentile: np.ndarray, percentile: float) -> np.ndarray:
    """Complex-Gaussian noise has Rayleigh magnitudes; recover the scale from a percentile."""
    probability = min(max(percentile / PERCENT, 1e-6), 1.0 - 1e-6)
    return magnitude_percentile / math.sqrt(-2.0 * math.log(1.0 - probability))


def _region_frame_span(
    band_power: np.ndarray,
    floor_power: float,
    start_frame: int,
    *,
    active_ratio: float,
    release_frames: int,
    max_frames: int,
) -> int:
    """Frames from start until the band stays below floor * active_ratio for release_frames."""
    end = start_frame
    quiet_run = 0
    last = min(band_power.size, start_frame + max_frames)
    for frame in range(start_frame, last):
        end = frame + 1
        if band_power[frame] >= floor_power * active_ratio:
            quiet_run = 0
        else:
            quiet_run += 1
            if quiet_run >= release_frames:
                end = frame + 1 - quiet_run
                break
    return max(end, start_frame + 1)


def mask_regions_with_noise_floor(
    signal: np.ndarray,
    *,
    sample_rate_hz: int,
    seeds: list[MaskRegionSeed],
    parameters: MaskingParameters,
    rng: np.random.Generator,
) -> MaskedSignal:
    samples = np.asarray(signal, dtype=np.float64)
    if samples.ndim > 1:
        samples = samples.mean(axis=0)
    if not seeds or samples.size == 0:
        return MaskedSignal(samples.copy(), 0.0, ())
    frame_samples = max(16, int(round(parameters.stft_frame_seconds * sample_rate_hz)))
    frame_samples = min(frame_samples, samples.size)
    overlap_samples = int(frame_samples * STFT_OVERLAP_FRACTION)
    hop_samples = frame_samples - overlap_samples
    frequencies, frame_times, spectrum = stft(
        samples, fs=sample_rate_hz, nperseg=frame_samples, noverlap=overlap_samples, boundary="even"
    )
    magnitude = np.abs(spectrum)
    floor_scale = _rayleigh_scale_from_percentile(
        np.percentile(magnitude, parameters.noise_floor_percentile, axis=1), parameters.noise_floor_percentile
    )
    mask = np.zeros(spectrum.shape, dtype=bool)
    active_ratio = 10.0 ** (parameters.active_snr_db / 10.0)
    release_frames = max(1, int(round(parameters.release_seconds * sample_rate_hz / hop_samples)))
    max_frames = max(1, int(round(parameters.max_region_seconds * sample_rate_hz / hop_samples)))
    region_seconds = []
    for seed in seeds:
        bins = (frequencies >= seed.band_min_hz - parameters.band_padding_hz) & (
            frequencies <= seed.band_max_hz + parameters.band_padding_hz
        )
        if not np.any(bins):
            region_seconds.append(0.0)
            continue
        start_time = seed.onset_seconds - parameters.pre_roll_seconds
        start_frame = int(np.searchsorted(frame_times, start_time, side="left"))
        if start_frame >= frame_times.size:
            region_seconds.append(0.0)
            continue
        band_power = np.mean(magnitude[bins] ** 2, axis=0)
        floor_power = float(np.mean(2.0 * floor_scale[bins] ** 2)) + POWER_FLOOR
        # A note that started before the signal only matters if it is still sounding at its start.
        if start_time < frame_times[0] and band_power[0] < floor_power * active_ratio:
            region_seconds.append(0.0)
            continue
        end_frame = _region_frame_span(
            band_power,
            floor_power,
            start_frame,
            active_ratio=active_ratio,
            release_frames=release_frames,
            max_frames=max_frames,
        )
        mask[np.ix_(bins, np.arange(start_frame, end_frame))] = True
        region_seconds.append((end_frame - start_frame) * hop_samples / sample_rate_hz)
    # Fill from the STFT of real white noise, shaped per bin to the floor. Independent
    # random cells are not a consistent STFT: the inverse transform would discard
    # ~4 dB of them at 75 % overlap and the fill would sit below the floor.
    _, _, unit_noise = stft(
        rng.standard_normal(samples.size), fs=sample_rate_hz, nperseg=frame_samples, noverlap=overlap_samples,
        boundary="even",
    )
    unit_scale = _rayleigh_scale_from_percentile(
        np.percentile(np.abs(unit_noise), parameters.noise_floor_percentile, axis=1), parameters.noise_floor_percentile
    )
    noise = unit_noise * (floor_scale / np.maximum(unit_scale, POWER_FLOOR))[:, None]
    replaced = np.where(mask, noise, spectrum)
    _, rebuilt = istft(replaced, fs=sample_rate_hz, nperseg=frame_samples, noverlap=overlap_samples)
    output = np.zeros_like(samples)
    usable = min(output.size, rebuilt.size)
    output[:usable] = rebuilt[:usable]
    return MaskedSignal(
        samples=output.astype(np.float32 if np.asarray(signal).dtype == np.float32 else np.float64),
        masked_cell_fraction=float(np.mean(mask)),
        region_seconds=tuple(region_seconds),
    )
