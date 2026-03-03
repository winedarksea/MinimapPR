"""First-order ambisonic encoding and offline soundscape rendering."""

from __future__ import annotations

import io
import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EPSILON = 1e-12
FOA_CHANNELS = 4


@dataclass(slots=True)
class SpatialSourceFrame:
    samples: np.ndarray
    position_m: tuple[float, float, float]
    label: str | None = None
    gain: float = 1.0
    source_id: str | None = None


@dataclass(slots=True)
class AmbisonicSpatialEncoder:
    distance_reference_m: float = 1.0
    min_distance_m: float = 0.35
    max_distance_gain: float = 1.0

    def encode_sources(
        self,
        sources: list[SpatialSourceFrame],
        *,
        listener_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        if not sources:
            return np.zeros((FOA_CHANNELS, 0), dtype=np.float32)

        max_len = max(int(source.samples.size) for source in sources)
        bformat = np.zeros((FOA_CHANNELS, max_len), dtype=np.float64)
        listener = np.asarray(listener_position_m, dtype=np.float64)

        for source in sources:
            if source.samples.size == 0:
                continue
            padded = np.zeros(max_len, dtype=np.float64)
            padded[: source.samples.size] = np.asarray(source.samples, dtype=np.float64)

            relative = np.asarray(source.position_m, dtype=np.float64) - listener
            distance = float(np.linalg.norm(relative))
            if distance < self.min_distance_m:
                distance = self.min_distance_m

            azimuth = float(math.atan2(relative[1], relative[0]))
            horizontal = float(np.hypot(relative[0], relative[1]))
            elevation = float(math.atan2(relative[2], max(horizontal, EPSILON)))

            # SN3D first-order B-format basis.
            w = 1.0 / math.sqrt(2.0)
            x = math.cos(azimuth) * math.cos(elevation)
            y = math.sin(azimuth) * math.cos(elevation)
            z = math.sin(elevation)

            distance_gain = min(
                self.max_distance_gain,
                self.distance_reference_m / max(distance, self.min_distance_m),
            )
            source_gain = float(source.gain) * distance_gain

            bformat[0] += source_gain * w * padded
            bformat[1] += source_gain * x * padded
            bformat[2] += source_gain * y * padded
            bformat[3] += source_gain * z * padded

        return np.clip(bformat, -1.0, 1.0).astype(np.float32)


def foa_to_5_1(foa_channels_first: np.ndarray) -> np.ndarray:
    if foa_channels_first.ndim != 2 or foa_channels_first.shape[0] != FOA_CHANNELS:
        raise ValueError("Expected FOA channels-first array with shape (4, n)")
    w, x, y, z = foa_channels_first
    del z
    left = (0.707 * w) + (0.45 * x) + (0.45 * y)
    right = (0.707 * w) + (0.45 * x) - (0.45 * y)
    center = (0.707 * w) + (0.55 * x)
    lfe = np.zeros_like(w)
    surround_left = (0.707 * w) - (0.35 * x) + (0.55 * y)
    surround_right = (0.707 * w) - (0.35 * x) - (0.55 * y)
    out = np.vstack([left, right, center, lfe, surround_left, surround_right])
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def write_wav_multichannel(path: Path, channels_first: np.ndarray, sample_rate_hz: int) -> None:
    if channels_first.ndim != 2:
        raise ValueError("Expected channels-first audio data")
    path.parent.mkdir(parents=True, exist_ok=True)
    interleaved = np.clip(channels_first.T, -1.0, 1.0)
    pcm = (interleaved * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(int(channels_first.shape[0]))
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate_hz))
        wav.writeframes(pcm.tobytes())


def wav_multichannel_bytes(channels_first: np.ndarray, sample_rate_hz: int) -> bytes:
    if channels_first.ndim != 2:
        raise ValueError("Expected channels-first audio data")
    interleaved = np.clip(channels_first.T, -1.0, 1.0)
    pcm = (interleaved * 32767.0).astype("<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(int(channels_first.shape[0]))
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate_hz))
        wav.writeframes(pcm.tobytes())
    return out.getvalue()


@dataclass(slots=True)
class SoundscapeRenderResult:
    bformat: np.ndarray
    rendered_sources: int
    suppressed_sources: int


@dataclass(slots=True)
class SoundscapeRenderer:
    encoder: AmbisonicSpatialEncoder
    suppress_labels: set[str] | None = None

    def render(
        self,
        sources: list[SpatialSourceFrame],
        *,
        listener_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        suppress_labels: set[str] | None = None,
    ) -> SoundscapeRenderResult:
        configured = {label.strip().lower() for label in (self.suppress_labels or set())}
        extra = {label.strip().lower() for label in (suppress_labels or set())}
        blocked = configured.union(extra)

        kept: list[SpatialSourceFrame] = []
        suppressed = 0
        for source in sources:
            label = (source.label or "").strip().lower()
            if label and label in blocked:
                suppressed += 1
                continue
            kept.append(source)

        encoded = self.encoder.encode_sources(kept, listener_position_m=listener_position_m)
        return SoundscapeRenderResult(
            bformat=encoded,
            rendered_sources=len(kept),
            suppressed_sources=suppressed,
        )

    def export_bformat_wav(self, path: Path, bformat: np.ndarray, sample_rate_hz: int) -> None:
        write_wav_multichannel(path=path, channels_first=bformat, sample_rate_hz=sample_rate_hz)

    def export_5_1_wav(self, path: Path, bformat: np.ndarray, sample_rate_hz: int) -> np.ndarray:
        downmix = foa_to_5_1(bformat)
        write_wav_multichannel(path=path, channels_first=downmix, sample_rate_hz=sample_rate_hz)
        return downmix
