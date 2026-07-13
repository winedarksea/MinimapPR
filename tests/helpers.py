from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class CrossNodeSyntheticScene:
    """Deterministic, physically delayed multi-node scene for TDOA regressions.

    ``true_sensor_positions_m`` drives propagation, while ``reported_sensor_positions_m``
    is what a localizer receives. Keeping those distinct lets tests model GPS survey error
    without corrupting the acoustic ground truth.
    """

    source_position_m: np.ndarray
    true_sensor_positions_m: dict[str, np.ndarray]
    reported_sensor_positions_m: dict[str, np.ndarray]
    sensor_windows: dict[str, np.ndarray]
    sensor_node_ids: dict[str, str]


def synthesize_cross_node_scene(
    *,
    source_position_m: tuple[float, float, float] | np.ndarray,
    node_origins_m: dict[str, tuple[float, float, float] | np.ndarray],
    tetra_node_ids: tuple[str, ...] = (),
    sample_rate_hz: int = 48_000,
    duration_seconds: float = 0.4,
    seed: int = 0,
    sound_speed_mps: float = 343.2,
    additive_noise_std: float = 0.0,
    reflection_delay_seconds: float | None = None,
    reflection_gain: float = 0.0,
    node_clock_offsets_s: dict[str, float] | None = None,
    node_position_errors_m: dict[str, tuple[float, float, float] | np.ndarray] | None = None,
    missing_sensor_ids: set[str] | None = None,
) -> CrossNodeSyntheticScene:
    """Generate a broadband scene with deterministic acoustic and metadata faults.

    A single global propagation reference preserves inter-node delay. Reflection and
    noise are deliberately simple but repeatable; they exercise correlation ambiguity
    without making the tests depend on a room simulator.
    """
    rng = np.random.default_rng(seed)
    sample_count = int(round(sample_rate_hz * duration_seconds))
    excitation = (rng.standard_normal(sample_count) * np.hanning(sample_count)).astype(np.float32)
    if reflection_delay_seconds is not None and reflection_gain:
        excitation = excitation + float(reflection_gain) * shift_signal(
            excitation, sample_rate_hz, float(reflection_delay_seconds)
        )

    true_positions, windows, node_ids = synthesize_multinode_windows(
        excitation,
        sample_rate_hz,
        source_position_m=source_position_m,
        node_origins_m=node_origins_m,
        tetra_node_ids=tetra_node_ids,
        sound_speed_mps=sound_speed_mps,
    )
    clock_offsets = node_clock_offsets_s or {}
    missing = missing_sensor_ids or set()
    reported_positions: dict[str, np.ndarray] = {}
    adjusted_windows: dict[str, np.ndarray] = {}
    position_errors = node_position_errors_m or {}
    for sensor_id, true_position in true_positions.items():
        node_id = node_ids[sensor_id]
        error_m = np.asarray(position_errors.get(node_id, (0.0, 0.0, 0.0)), dtype=np.float64)
        reported_positions[sensor_id] = np.asarray(true_position, dtype=np.float64) + error_m
        window = windows[sensor_id]
        if node_id in clock_offsets:
            window = shift_signal(window, sample_rate_hz, float(clock_offsets[node_id]))
        if sensor_id in missing:
            window = np.zeros_like(window)
        if additive_noise_std:
            window = window + rng.normal(0.0, additive_noise_std, size=window.size).astype(np.float32)
        adjusted_windows[sensor_id] = window.astype(np.float32, copy=False)

    return CrossNodeSyntheticScene(
        source_position_m=np.asarray(source_position_m, dtype=np.float64),
        true_sensor_positions_m=true_positions,
        reported_sensor_positions_m=reported_positions,
        sensor_windows=adjusted_windows,
        sensor_node_ids=node_ids,
    )


def shift_signal(signal: np.ndarray, sample_rate_hz: int, delay_s: float) -> np.ndarray:
    """Delay or advance a 1-D signal by *delay_s* using linear interpolation."""
    n = signal.size
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    shifted_t = t - delay_s
    return np.interp(shifted_t, t, signal, left=0.0, right=0.0).astype(np.float32)


def synthesize_multinode_windows(
    excitation: np.ndarray,
    sample_rate_hz: int,
    *,
    source_position_m: tuple[float, float, float] | np.ndarray,
    node_origins_m: dict[str, tuple[float, float, float] | np.ndarray],
    tetra_node_ids: tuple[str, ...] = (),
    sound_speed_mps: float = 343.2,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    """Build synchronized multi-node sensor windows for a single point source.

    Nodes in *tetra_node_ids* are expanded into four microphones using
    ``SIRITH_TETRA_SENSOR_OFFSETS_M``; every other node is a single-microphone
    ("point") node placed at its origin. Sensor IDs follow the ``"{node}:ch{i}"``
    convention used elsewhere in the suite.

    Unlike the free-running mixed-node fixtures (which subtract a *per-node*
    minimum delay), this helper subtracts a single **global** minimum delay so
    inter-node arrival timing is preserved. That cross-node timing is exactly the
    parallax that lets a synchronized multi-node cluster observe range. Amplitude
    is uniform across sensors so far-range cases are not gated by per-channel RMS.

    Returns ``(sensor_positions, sensor_windows, sensor_node_ids)``.
    """
    source = np.asarray(source_position_m, dtype=np.float64)

    sensor_positions: dict[str, np.ndarray] = {}
    sensor_node_ids: dict[str, str] = {}
    for node_id, origin in node_origins_m.items():
        origin_m = np.asarray(origin, dtype=np.float64)
        if node_id in tetra_node_ids:
            for index, offset_m in enumerate(SIRITH_TETRA_SENSOR_OFFSETS_M):
                sensor_id = f"{node_id}:ch{index}"
                sensor_positions[sensor_id] = origin_m + np.asarray(offset_m, dtype=np.float64)
                sensor_node_ids[sensor_id] = node_id
        else:
            sensor_id = f"{node_id}:ch0"
            sensor_positions[sensor_id] = origin_m
            sensor_node_ids[sensor_id] = node_id

    absolute_delays_s = {
        sensor_id: float(np.linalg.norm(source - position) / sound_speed_mps)
        for sensor_id, position in sensor_positions.items()
    }
    global_min_delay_s = min(absolute_delays_s.values())
    max_relative_delay_s = max(absolute_delays_s.values()) - global_min_delay_s

    # Pad so the most-delayed sensor's content still fits inside the window.
    head_samples = sample_rate_hz // 100  # 10 ms head room
    tail_samples = int(np.ceil(max_relative_delay_s * sample_rate_hz)) + head_samples
    padded = np.concatenate(
        [
            np.zeros(head_samples, dtype=np.float32),
            np.asarray(excitation, dtype=np.float32),
            np.zeros(tail_samples, dtype=np.float32),
        ]
    )

    sensor_windows = {
        sensor_id: shift_signal(padded, sample_rate_hz, delay_s - global_min_delay_s)
        for sensor_id, delay_s in absolute_delays_s.items()
    }
    return sensor_positions, sensor_windows, sensor_node_ids


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
