from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from minimappr.models import LocalizationResult
from minimappr.utils.audio import read_wav_mono, rms, write_wav_mono


# Mirrored from firmware/nodes/sirith_tetrahedral/include/node_config.h.
# Keep the test oracle aligned with the deployed Sirith tetra microphone geometry.
SIRITH_TETRA_SENSOR_OFFSETS_M: tuple[tuple[float, float, float], ...] = (
    (-0.016238, 0.025000, -0.010205),
    (0.027063, 0.000000, -0.010205),
    (-0.016238, -0.025000, -0.010205),
    (0.005413, 0.000000, 0.030615),
)


def shift_signal(signal: np.ndarray, sample_rate_hz: int, delay_s: float) -> np.ndarray:
    """Delay or advance a 1-D signal by *delay_s* using linear interpolation."""
    n = signal.size
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    shifted_t = t - delay_s
    return np.interp(shifted_t, t, signal, left=0.0, right=0.0).astype(np.float32)


def load_wav_fixture_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load a mono WAV fixture as float32 samples in [-1, 1]."""
    return read_wav_mono(path)


def resample_signal(signal: np.ndarray, input_sample_rate_hz: int, output_sample_rate_hz: int) -> np.ndarray:
    """Resample deterministically using a rational polyphase filter."""
    if input_sample_rate_hz == output_sample_rate_hz:
        return signal.astype(np.float32, copy=True)

    gcd = int(np.gcd(input_sample_rate_hz, output_sample_rate_hz))
    up = output_sample_rate_hz // gcd
    down = input_sample_rate_hz // gcd
    return resample_poly(signal, up=up, down=down).astype(np.float32)


def synthesize_delayed_array_channels(
    mono_signal: np.ndarray,
    sample_rate_hz: int,
    *,
    source_position_m: tuple[float, float, float],
    sensor_offsets_m: tuple[tuple[float, float, float], ...] = SIRITH_TETRA_SENSOR_OFFSETS_M,
    array_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    sound_speed_mps: float = 343.2,
) -> np.ndarray:
    """Create a channels-first array using geometric propagation delays."""
    source = np.asarray(source_position_m, dtype=np.float64)
    origin = np.asarray(array_origin_m, dtype=np.float64)
    channels: list[np.ndarray] = []
    for offset in sensor_offsets_m:
        sensor_position = origin + np.asarray(offset, dtype=np.float64)
        propagation_delay_s = float(np.linalg.norm(source - sensor_position) / sound_speed_mps)
        channels.append(shift_signal(mono_signal, sample_rate_hz, propagation_delay_s))
    return np.stack(channels, axis=0).astype(np.float32)


def geometric_array_propagation_delays_s(
    *,
    source_position_m: tuple[float, float, float],
    sensor_offsets_m: tuple[tuple[float, float, float], ...] = SIRITH_TETRA_SENSOR_OFFSETS_M,
    array_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    sound_speed_mps: float = 343.2,
) -> np.ndarray:
    """Return absolute propagation delays from the source to each array sensor."""
    source = np.asarray(source_position_m, dtype=np.float64)
    origin = np.asarray(array_origin_m, dtype=np.float64)
    return np.asarray(
        [
            float(np.linalg.norm(source - (origin + np.asarray(offset, dtype=np.float64))) / sound_speed_mps)
            for offset in sensor_offsets_m
        ],
        dtype=np.float64,
    )


def write_delayed_array_wav_files(
    output_dir: Path,
    *,
    mono_signal: np.ndarray,
    sample_rate_hz: int,
    source_position_m: tuple[float, float, float],
    sensor_offsets_m: tuple[tuple[float, float, float], ...] = SIRITH_TETRA_SENSOR_OFFSETS_M,
    array_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    sound_speed_mps: float = 343.2,
    filename_prefix: str = "sensor",
) -> dict[str, Path]:
    """Materialize one mono WAV file per microphone for realistic array-fixture tests."""
    channels = synthesize_delayed_array_channels(
        mono_signal,
        sample_rate_hz,
        source_position_m=source_position_m,
        sensor_offsets_m=sensor_offsets_m,
        array_origin_m=array_origin_m,
        sound_speed_mps=sound_speed_mps,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sensor_paths: dict[str, Path] = {}
    for index, channel in enumerate(channels):
        sensor_id = f"{filename_prefix}{index}"
        path = output_dir / f"{sensor_id}.wav"
        write_wav_mono(path, channel, sample_rate_hz)
        sensor_paths[sensor_id] = path
    return sensor_paths


def load_sensor_wav_files(sensor_paths: dict[str, Path]) -> tuple[dict[str, np.ndarray], int]:
    """Load per-sensor mono WAVs and enforce a shared sample rate."""
    windows: dict[str, np.ndarray] = {}
    sample_rate_hz: int | None = None
    for sensor_id, path in sensor_paths.items():
        samples, current_sample_rate_hz = read_wav_mono(path)
        if sample_rate_hz is None:
            sample_rate_hz = current_sample_rate_hz
        elif current_sample_rate_hz != sample_rate_hz:
            raise ValueError("All sensor WAV files must share the same sample rate")
        windows[sensor_id] = samples
    if sample_rate_hz is None:
        raise ValueError("Expected at least one sensor WAV file")
    return windows, sample_rate_hz


def prepend_noise_padding_to_duration(
    signal: np.ndarray,
    sample_rate_hz: int,
    *,
    total_duration_seconds: float,
    noise_rms: float,
    seed: int = 0,
) -> np.ndarray:
    """Front-pad a signal with deterministic low-level white noise to a target duration."""
    target_samples = max(1, int(round(total_duration_seconds * sample_rate_hz)))
    if signal.size >= target_samples:
        return signal.astype(np.float32, copy=True)

    pad_samples = target_samples - signal.size
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=pad_samples).astype(np.float32)
    current_rms = rms(noise)
    if current_rms > 0.0:
        noise *= noise_rms / current_rms
    return np.concatenate([noise, signal.astype(np.float32, copy=False)]).astype(np.float32)


def prepend_noise_padding(
    signal: np.ndarray,
    *,
    pad_samples: int,
    noise_rms: float,
    seed: int = 0,
) -> np.ndarray:
    """Front-pad a signal with a fixed number of deterministic noise samples."""
    if pad_samples <= 0:
        return signal.astype(np.float32, copy=True)

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=pad_samples).astype(np.float32)
    current_rms = rms(noise)
    if current_rms > 0.0:
        noise *= noise_rms / current_rms
    return np.concatenate([noise, signal.astype(np.float32, copy=False)]).astype(np.float32)


def split_channels_into_frames(
    channels_first: np.ndarray,
    *,
    sample_rate_hz: int,
    start_time_ns: int,
    frame_samples: int,
) -> list[tuple[int, np.ndarray]]:
    """Split channels-first audio into fixed-size frames, zero-padding the tail."""
    if channels_first.ndim != 2:
        raise ValueError("Expected channels-first array")
    if frame_samples <= 0:
        raise ValueError("frame_samples must be > 0")

    channels, total_samples = channels_first.shape
    del channels
    frame_duration_ns = int(round((frame_samples / sample_rate_hz) * 1_000_000_000))
    frames: list[tuple[int, np.ndarray]] = []
    for frame_index, start_sample in enumerate(range(0, total_samples, frame_samples)):
        frame = channels_first[:, start_sample : start_sample + frame_samples].astype(np.float32, copy=True)
        if frame.shape[1] < frame_samples:
            padding = np.zeros((frame.shape[0], frame_samples - frame.shape[1]), dtype=np.float32)
            frame = np.concatenate([frame, padding], axis=1)
        frame_start_ns = start_time_ns + frame_index * frame_duration_ns
        frames.append((frame_start_ns, frame))
    return frames


class StubLocalizer:
    """Simple deterministic localizer stub for dispatcher strategy tests."""

    def __init__(self, name: str, confidence: float) -> None:
        self.name = name
        self.confidence = confidence
        self.calls = 0
        self.localize_2d_calls = 0

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

    def localize_2d(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
        fixed_z_m: float | None = None,
    ) -> LocalizationResult:
        del sensor_windows, sample_rate_hz, temperature_c, humidity_fraction, fixed_z_m
        self.localize_2d_calls += 1
        reference_sensor = sorted(sensor_positions.keys())[0]
        return LocalizationResult(
            position_m=(0.0, 0.0, 0.0),
            confidence=self.confidence,
            gdop=1.0,
            reference_sensor=reference_sensor,
            tdoa_s={},
        )


class FailingLocalizer:
    """Localizer that always raises LocalizationError."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def localize(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
    ) -> LocalizationResult:
        from minimappr.core.localization import LocalizationError

        self.calls += 1
        raise LocalizationError(f"{self.name} always fails")
