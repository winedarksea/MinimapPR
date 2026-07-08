"""COLA-friendly STFT helpers for offline and future streaming encoders."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sqrt_hann_window(frame_size: int) -> NDArray[np.float64]:
    if frame_size <= 0:
        raise ValueError("frame_size must be positive")
    return np.sqrt(np.hanning(frame_size).astype(np.float64))


def stft_channels(
    channels: NDArray[np.float32 | np.float64],
    *,
    frame_size: int,
    hop_size: int,
    window: NDArray[np.float64] | None = None,
) -> NDArray[np.complex128]:
    if channels.ndim != 2:
        raise ValueError("channels must have shape (channels, samples)")
    if hop_size <= 0 or frame_size <= 0:
        raise ValueError("frame_size and hop_size must be positive")
    if hop_size > frame_size:
        raise ValueError("hop_size must be <= frame_size")

    win = sqrt_hann_window(frame_size) if window is None else np.asarray(window, dtype=np.float64)
    if win.shape != (frame_size,):
        raise ValueError("window length must equal frame_size")

    n_channels, n_samples = channels.shape
    n_frames = max(1, int(np.ceil(max(1, n_samples - frame_size) / hop_size)) + 1)
    spectra = np.zeros((n_channels, n_frames, frame_size // 2 + 1), dtype=np.complex128)
    for frame_index in range(n_frames):
        start = frame_index * hop_size
        end = min(start + frame_size, n_samples)
        frame = np.zeros((n_channels, frame_size), dtype=np.float64)
        if end > start:
            frame[:, : end - start] = channels[:, start:end]
        spectra[:, frame_index, :] = np.fft.rfft(frame * win[np.newaxis, :], axis=1)
    return spectra


def istft_channels(
    spectra: NDArray[np.complex128],
    *,
    frame_size: int,
    hop_size: int,
    n_samples: int,
    window: NDArray[np.float64] | None = None,
) -> NDArray[np.float32]:
    if spectra.ndim != 3:
        raise ValueError("spectra must have shape (channels, frames, bins)")
    win = sqrt_hann_window(frame_size) if window is None else np.asarray(window, dtype=np.float64)
    if win.shape != (frame_size,):
        raise ValueError("window length must equal frame_size")

    n_channels, n_frames, _ = spectra.shape
    output = np.zeros((n_channels, n_samples), dtype=np.float64)
    norm = np.zeros(n_samples, dtype=np.float64)
    for frame_index in range(n_frames):
        start = frame_index * hop_size
        end = min(start + frame_size, n_samples)
        if end <= start:
            continue
        frame = np.fft.irfft(spectra[:, frame_index, :], n=frame_size, axis=1)
        actual = end - start
        output[:, start:end] += frame[:, :actual] * win[:actual][np.newaxis, :]
        norm[start:end] += win[:actual] ** 2

    safe_norm = np.where(norm > 1e-12, norm, 1.0)
    return (output / safe_norm[np.newaxis, :]).astype(np.float32)


def cola_error(frame_size: int, hop_size: int) -> float:
    win = sqrt_hann_window(frame_size)
    acc = np.zeros(frame_size * 4, dtype=np.float64)
    for start in range(0, frame_size * 3, hop_size):
        acc[start : start + frame_size] += win ** 2
    middle = acc[frame_size : frame_size * 2]
    return float(np.max(np.abs(middle - np.mean(middle))))
