"""Audio utilities for decoding, analysis, and snippets."""

from __future__ import annotations

import base64
import binascii
import wave
from pathlib import Path

import numpy as np


EPSILON = 1e-12


def decode_pcm16le_b64(samples_b64: str, channels: int) -> np.ndarray:
    try:
        raw = base64.b64decode(samples_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 audio payload") from exc
    if not raw:
        raise ValueError("Decoded PCM payload is empty")
    if len(raw) % 2 != 0:
        raise ValueError("Decoded PCM payload byte length must be even for pcm16le")
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if data.size % channels != 0:
        raise ValueError("Decoded PCM sample count is not divisible by channels")
    return data.reshape(-1, channels).T


def encode_pcm16le_b64(channels_first: np.ndarray) -> str:
    if channels_first.ndim != 2:
        raise ValueError("Expected channels-first array")
    interleaved = np.clip(channels_first.T, -1.0, 1.0)
    pcm = (interleaved * 32767.0).astype("<i2").tobytes()
    return base64.b64encode(pcm).decode("ascii")


def mono_mix(channels_first: np.ndarray) -> np.ndarray:
    if channels_first.ndim != 2:
        raise ValueError("Expected channels-first array")
    return np.mean(channels_first, axis=0)


def trigger_rms(channels_first: np.ndarray) -> float:
    """Estimate trigger energy without inter-channel cancellation.

    Array microphones capture the same source with small phase offsets. Averaging
    channels before RMS can cancel narrowband content, especially birdsong, and
    hide real events from the trigger gate. Use the strongest per-channel RMS
    instead so multichannel arrays trigger on audible events the same way a
    single hot sensor would.
    """
    if channels_first.ndim != 2:
        raise ValueError("Expected channels-first array")
    if channels_first.shape[0] == 0:
        return 0.0
    return max(rms(channel) for channel in channels_first)


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples)) + EPSILON))


def spectral_features(samples: np.ndarray, sample_rate_hz: int) -> dict[str, float]:
    if samples.size == 0:
        return {
            "rms": 0.0,
            "zcr": 0.0,
            "crest": 0.0,
            "centroid_hz": 0.0,
            "flatness": 0.0,
            "bandwidth_hz": 0.0,
        }

    x = samples.astype(np.float32)
    energy_rms = rms(x)
    peak = float(np.max(np.abs(x)) + EPSILON)
    crest = peak / (energy_rms + EPSILON)

    zc = np.abs(np.diff(np.signbit(x))).sum()
    zcr = float(zc / max(1, x.size - 1))

    spectrum = np.abs(np.fft.rfft(x)) + EPSILON
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sample_rate_hz)
    mag_sum = float(np.sum(spectrum))
    centroid = float(np.sum(freqs * spectrum) / mag_sum)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / mag_sum))

    geometric_mean = float(np.exp(np.mean(np.log(spectrum))))
    arithmetic_mean = float(np.mean(spectrum))
    flatness = geometric_mean / (arithmetic_mean + EPSILON)

    return {
        "rms": energy_rms,
        "zcr": zcr,
        "crest": crest,
        "centroid_hz": centroid,
        "flatness": float(flatness),
        "bandwidth_hz": bandwidth,
    }


def write_wav_mono(path: Path, samples: np.ndarray, sample_rate_hz: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(samples, -1.0, 1.0)
    pcm = (data * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate_hz)
        wav.writeframes(pcm.tobytes())


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = int(wav.getnchannels())
        sample_width = int(wav.getsampwidth())
        sample_rate_hz = int(wav.getframerate())
        frame_count = int(wav.getnframes())
        raw = wav.readframes(frame_count)

    if channels <= 0:
        raise ValueError("WAV must have at least one channel")

    if sample_width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if data.size % channels != 0:
        raise ValueError("WAV payload does not align to channel count")

    frames = data.reshape(-1, channels)
    mono = np.mean(frames, axis=1, dtype=np.float32)
    return mono.astype(np.float32), sample_rate_hz
