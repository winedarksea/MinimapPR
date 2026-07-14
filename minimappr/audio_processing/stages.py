"""Reusable audio-processing stages with explicit state and channel semantics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfilt

from minimappr.audio_processing.levels import apply_bounded_rms_gain
from minimappr.interfaces import AudioPreprocessor


def _clamp_cutoff(cutoff_hz: float, sample_rate_hz: int) -> float:
    return float(min(max(cutoff_hz, 1.0), 0.5 * sample_rate_hz * 0.95))


@dataclass(slots=True)
class _CausalButterworthStage(AudioPreprocessor):
    order: int = 4
    _state: dict[int, np.ndarray] = field(default_factory=dict)
    _sample_rate_hz: int | None = None

    def _filter(self, samples: np.ndarray, sample_rate_hz: int, channel_idx: int, sos: np.ndarray) -> np.ndarray:
        if self._sample_rate_hz != sample_rate_hz:
            self.reset()
            self._sample_rate_hz = sample_rate_hz
        zi = self._state.get(channel_idx)
        expected_shape = (sos.shape[0], 2)
        if zi is None or zi.shape != expected_shape:
            zi = np.zeros(expected_shape, dtype=np.float64)
        output, final_state = sosfilt(sos, samples.astype(np.float64, copy=False), zi=zi)
        self._state[channel_idx] = final_state
        return output.astype(np.float32)

    def reset(self) -> None:
        self._state.clear()
        self._sample_rate_hz = None


@dataclass(slots=True)
class HighpassFilterStage(_CausalButterworthStage):
    cutoff_hz: float = 50.0

    def process(self, samples, sample_rate_hz, *, node_id=None, channel_idx=0):
        del node_id
        if self.cutoff_hz <= 0.0 or samples.size < 16:
            return samples
        cutoff = _clamp_cutoff(self.cutoff_hz, sample_rate_hz)
        sos = butter(self.order, cutoff, btype="highpass", fs=sample_rate_hz, output="sos")
        return self._filter(samples, sample_rate_hz, channel_idx, sos)


@dataclass(slots=True)
class LowpassFilterStage(_CausalButterworthStage):
    cutoff_hz: float = 0.0

    def process(self, samples, sample_rate_hz, *, node_id=None, channel_idx=0):
        del node_id
        if self.cutoff_hz <= 0.0 or samples.size < 16:
            return samples
        cutoff = _clamp_cutoff(self.cutoff_hz, sample_rate_hz)
        sos = butter(self.order, cutoff, btype="lowpass", fs=sample_rate_hz, output="sos")
        return self._filter(samples, sample_rate_hz, channel_idx, sos)


@dataclass(slots=True)
class BandpassFilterStage(_CausalButterworthStage):
    low_hz: float = 50.0
    high_hz: float = 8_000.0

    def process(self, samples, sample_rate_hz, *, node_id=None, channel_idx=0):
        del node_id
        if samples.size < 16:
            return samples
        low = _clamp_cutoff(self.low_hz, sample_rate_hz)
        high = _clamp_cutoff(self.high_hz, sample_rate_hz)
        if low >= high:
            return samples
        # Cascaded HP+LP is the canonical Rust wire behavior, including order.
        highpass = butter(self.order, low, btype="highpass", fs=sample_rate_hz, output="sos")
        lowpass = butter(self.order, high, btype="lowpass", fs=sample_rate_hz, output="sos")
        return self._filter(samples, sample_rate_hz, channel_idx, np.vstack((highpass, lowpass)))


@dataclass(slots=True)
class GainStage(AudioPreprocessor):
    multiplier: float = 1.0

    def process(self, samples, sample_rate_hz, *, node_id=None, channel_idx=0):
        del node_id, sample_rate_hz, channel_idx
        if samples.size == 0 or self.multiplier == 1.0:
            return samples
        return (samples * self.multiplier).astype(np.float32)

    def reset(self) -> None:
        return


@dataclass(slots=True)
class ChannelGainStage(AudioPreprocessor):
    multipliers_by_channel: tuple[float, ...]

    def process(self, samples, sample_rate_hz, *, node_id=None, channel_idx=0):
        del node_id, sample_rate_hz
        if channel_idx >= len(self.multipliers_by_channel) or samples.size == 0:
            return samples
        multiplier = self.multipliers_by_channel[channel_idx]
        if multiplier == 1.0:
            return samples
        return (samples * multiplier).astype(np.float32)

    def reset(self) -> None:
        return


@dataclass(slots=True)
class DCBlockStage(AudioPreprocessor):
    """Stateful one-pole 5 Hz DC blocker, matching the Rust sidecar."""

    cutoff_hz: float = 5.0
    _previous_input: dict[int, float] = field(default_factory=dict)
    _previous_output: dict[int, float] = field(default_factory=dict)

    def process(self, samples, sample_rate_hz, *, node_id=None, channel_idx=0):
        del node_id
        if samples.size == 0:
            return samples
        alpha = float(np.exp(-2.0 * np.pi * self.cutoff_hz / max(sample_rate_hz, 1)))
        previous_input = self._previous_input.get(channel_idx, 0.0)
        previous_output = self._previous_output.get(channel_idx, 0.0)
        output = np.empty_like(samples, dtype=np.float32)
        for index, raw_value in enumerate(samples):
            value = float(raw_value)
            filtered = value - previous_input + alpha * previous_output
            output[index] = filtered
            previous_input, previous_output = value, filtered
        self._previous_input[channel_idx] = previous_input
        self._previous_output[channel_idx] = previous_output
        return output

    def reset(self) -> None:
        self._previous_input.clear()
        self._previous_output.clear()


@dataclass(slots=True)
class MeanCenterStage(AudioPreprocessor):
    def process(self, samples, sample_rate_hz, *, node_id=None, channel_idx=0):
        del node_id, sample_rate_hz, channel_idx
        if samples.size == 0:
            return samples
        return (samples.astype(np.float32, copy=False) - np.mean(samples, dtype=np.float32)).astype(np.float32)

    def reset(self) -> None:
        return


class _DCRemovalCompatibilityMeta(type(MeanCenterStage)):
    def __instancecheck__(cls, instance):
        return isinstance(instance, (MeanCenterStage, DCBlockStage))


class DCRemovalStage(MeanCenterStage, metaclass=_DCRemovalCompatibilityMeta):
    """Backward-compatible window mean removal; use ``dc_block`` at ingest."""


@dataclass(slots=True)
class BoundedRmsGainStage(AudioPreprocessor):
    target_rms_dbfs: float
    max_gain_db: float
    peak_ceiling_dbfs: float = -1.0
    boost_only: bool = True

    def process(self, samples, sample_rate_hz, *, node_id=None, channel_idx=0):
        del node_id, sample_rate_hz, channel_idx
        output, _ = apply_bounded_rms_gain(
            samples,
            target_rms=10.0 ** (self.target_rms_dbfs / 20.0),
            max_gain=10.0 ** (self.max_gain_db / 20.0),
            peak_ceiling=10.0 ** (self.peak_ceiling_dbfs / 20.0),
            boost_only=self.boost_only,
        )
        return output

    def reset(self) -> None:
        return


@dataclass(slots=True)
class NormalizationStage(AudioPreprocessor):
    target_level: float = 1.0
    mode: str = "peak"

    def process(self, samples, sample_rate_hz, *, node_id=None, channel_idx=0):
        del node_id, sample_rate_hz, channel_idx
        if samples.size == 0:
            return samples
        current = (
            float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
            if self.mode == "rms"
            else float(np.max(np.abs(samples)))
        )
        return samples if current < 1e-12 else (samples * self.target_level / current).astype(np.float32)

    def reset(self) -> None:
        return


@dataclass(slots=True)
class SpectralGateStage(AudioPreprocessor):
    threshold_factor: float = 1.5
    block_size: int = 1024
    min_gain: float = 0.08

    def process(self, samples, sample_rate_hz, *, node_id=None, channel_idx=0):
        del node_id, sample_rate_hz, channel_idx
        if samples.size < 16:
            return samples
        spectrum = np.fft.rfft(np.asarray(samples, dtype=np.float64))
        magnitude = np.abs(spectrum)
        floor = self.threshold_factor * float(np.median(magnitude))
        if floor < 1e-12:
            return samples
        gain = np.maximum(magnitude**2 / (magnitude**2 + floor**2), self.min_gain)
        return np.fft.irfft(spectrum * gain, n=samples.size).astype(np.float32)

    def reset(self) -> None:
        return


@dataclass(slots=True)
class PassthroughStage(AudioPreprocessor):
    def process(self, samples, sample_rate_hz, *, node_id=None, channel_idx=0):
        del sample_rate_hz, node_id, channel_idx
        return samples

    def reset(self) -> None:
        return
