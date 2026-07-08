"""Parametric FOA enhancement for compact omnidirectional tetra arrays."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from minimappr.spatial_audio.geometry import SIRITH_MIC_POSITIONS_M, rotate_positions
from minimappr.spatial_audio.linear_atob import atob_foa
from minimappr.spatial_audio.profiles import AmbisonicsProfile, get_profile
from minimappr.spatial_audio.stft import istft_channels, sqrt_hann_window, stft_channels


_SQRT2 = math.sqrt(2.0)


def encode_ambisonics(
    channels: NDArray[np.float32 | np.float64],
    sample_rate_hz: int,
    *,
    profile: str | AmbisonicsProfile = "parametric_v2",
    orientation: object | None = None,
    mic_positions_m: NDArray[np.float64] | None = None,
) -> NDArray[np.float32]:
    selected_profile = get_profile(profile)
    positions = SIRITH_MIC_POSITIONS_M if mic_positions_m is None else np.asarray(mic_positions_m, dtype=np.float64)
    if orientation is not None:
        positions = rotate_positions(positions, orientation)

    frame_size = _frame_size_for_rate(sample_rate_hz, selected_profile.frame_duration_ms)
    linear = atob_foa(
        channels,
        sample_rate_hz,
        block_size=frame_size,
        hop=max(1, frame_size // 4),
        mic_positions_m=positions,
    )
    if selected_profile.max_parametric_blend <= 0.0:
        return _scale_true_peak(linear, selected_profile.output_peak_target)

    enhanced = enhance_foa_parametric(
        linear,
        sample_rate_hz,
        profile=selected_profile,
        frame_size=frame_size,
    )
    return _scale_true_peak(enhanced, selected_profile.output_peak_target)


def enhance_foa_parametric(
    foa_linear: NDArray[np.float32 | np.float64],
    sample_rate_hz: int,
    *,
    profile: AmbisonicsProfile,
    frame_size: int | None = None,
) -> NDArray[np.float32]:
    if foa_linear.ndim != 2 or foa_linear.shape[0] != 4:
        raise ValueError("foa_linear must have shape (4, samples)")
    if foa_linear.shape[1] == 0:
        return foa_linear.astype(np.float32, copy=True)

    frame_size = frame_size or _frame_size_for_rate(sample_rate_hz, profile.frame_duration_ms)
    hop_size = max(1, int(round(frame_size * profile.hop_fraction)))
    window = sqrt_hann_window(frame_size)
    spectra = stft_channels(
        np.asarray(foa_linear, dtype=np.float64),
        frame_size=frame_size,
        hop_size=hop_size,
        window=window,
    )
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate_hz)
    output_spectra = spectra.copy()

    directions = _smoothed_intensity_directions(
        spectra,
        hop_size=hop_size,
        sample_rate_hz=sample_rate_hz,
        smoothing_ms=profile.intensity_smoothing_ms,
    )
    diffuseness = _smoothed_diffuseness(
        spectra,
        directions,
        hop_size=hop_size,
        sample_rate_hz=sample_rate_hz,
        smoothing_ms=profile.diffuseness_smoothing_ms,
    )

    low_hz = float(profile.min_parametric_hz)
    high_hz = 0.5 * sample_rate_hz * float(profile.max_parametric_fraction_of_nyquist)
    active_bins = (freqs >= low_hz) & (freqs <= high_hz)

    w = spectra[0]
    linear_xyz = spectra[1:4]
    parametric_xyz = np.zeros_like(linear_xyz)
    for axis in range(3):
        parametric_xyz[axis] = _SQRT2 * directions[:, axis][:, np.newaxis] * w

    energy = np.abs(w) ** 2 + np.sum(np.abs(linear_xyz) ** 2, axis=0)
    confidence = energy / (energy + np.percentile(energy, 35) + 1e-12)
    blend = profile.max_parametric_blend * (1.0 - diffuseness) * confidence
    blend = np.where(confidence >= profile.min_confidence_for_blend, blend, 0.0)
    blend[:, ~active_bins] = 0.0
    blend = np.clip(blend, 0.0, profile.max_parametric_blend)

    output_spectra[0] = spectra[0]
    output_spectra[1:4] = (1.0 - blend[np.newaxis, :, :]) * linear_xyz + (
        blend[np.newaxis, :, :] * parametric_xyz
    )

    return istft_channels(
        output_spectra,
        frame_size=frame_size,
        hop_size=hop_size,
        n_samples=foa_linear.shape[1],
        window=window,
    )


def _smoothed_intensity_directions(
    spectra: NDArray[np.complex128],
    *,
    hop_size: int,
    sample_rate_hz: int,
    smoothing_ms: float,
) -> NDArray[np.float64]:
    w = spectra[0]
    xyz = spectra[1:4]
    raw = np.real(np.conj(w)[np.newaxis, :, :] * xyz)
    frame_vectors = np.sum(raw, axis=2).T
    alpha = _ema_alpha(hop_size, sample_rate_hz, smoothing_ms)
    smoothed = np.zeros_like(frame_vectors, dtype=np.float64)
    previous = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    for frame_index, vector in enumerate(frame_vectors):
        if np.linalg.norm(vector) > 1e-12:
            previous = (alpha * previous) + ((1.0 - alpha) * vector)
        norm = float(np.linalg.norm(previous))
        smoothed[frame_index] = previous / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])
    return smoothed


def _smoothed_diffuseness(
    spectra: NDArray[np.complex128],
    directions: NDArray[np.float64],
    *,
    hop_size: int,
    sample_rate_hz: int,
    smoothing_ms: float,
) -> NDArray[np.float64]:
    w = spectra[0]
    xyz = spectra[1:4]
    projected = np.sum(xyz * directions.T[:, :, np.newaxis], axis=0)
    directional_energy = np.abs(projected) ** 2
    total_velocity_energy = np.sum(np.abs(xyz) ** 2, axis=0)
    raw = 1.0 - (directional_energy / (total_velocity_energy + 1e-12))
    raw = np.clip(raw, 0.0, 1.0)
    alpha = _ema_alpha(hop_size, sample_rate_hz, smoothing_ms)
    smoothed = np.zeros_like(raw)
    previous = raw[0]
    for frame_index in range(raw.shape[0]):
        previous = (alpha * previous) + ((1.0 - alpha) * raw[frame_index])
        smoothed[frame_index] = previous
    return smoothed


def _ema_alpha(hop_size: int, sample_rate_hz: int, smoothing_ms: float) -> float:
    tau_s = max(1e-3, smoothing_ms / 1000.0)
    hop_s = hop_size / float(sample_rate_hz)
    return float(math.exp(-hop_s / tau_s))


def _frame_size_for_rate(sample_rate_hz: int, frame_duration_ms: float) -> int:
    target = max(256, int(round(sample_rate_hz * frame_duration_ms / 1000.0)))
    return 1 << int(math.ceil(math.log2(target)))


def _scale_true_peak(channels: NDArray[np.float32], target_peak: float) -> NDArray[np.float32]:
    peak = float(np.max(np.abs(channels))) if channels.size else 0.0
    if peak <= target_peak or peak <= 1e-12:
        return channels.astype(np.float32, copy=False)
    return (channels.astype(np.float32) * (float(target_peak) / peak)).astype(np.float32)
