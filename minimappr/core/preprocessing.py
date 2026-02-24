"""Audio preprocessing chain with pluggable filter stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

from minimappr.config import Settings
from minimappr.interfaces import AudioPreprocessor
from minimappr.models import NodeSpec


def _clamp_cutoff(cutoff_hz: float, sample_rate_hz: int) -> float:
    nyquist = 0.5 * float(sample_rate_hz)
    return float(min(max(cutoff_hz, 1.0), nyquist * 0.95))


@dataclass(slots=True)
class HighpassFilterStage(AudioPreprocessor):
    cutoff_hz: float
    order: int = 4

    def process(
        self,
        samples: np.ndarray,
        sample_rate_hz: int,
        *,
        node_id: str | None = None,
    ) -> np.ndarray:
        del node_id
        if self.cutoff_hz <= 0.0 or samples.size < 16:
            return samples
        cutoff = _clamp_cutoff(self.cutoff_hz, sample_rate_hz)
        sos = butter(self.order, cutoff, btype="highpass", fs=float(sample_rate_hz), output="sos")
        return sosfiltfilt(sos, samples).astype(np.float32)


@dataclass(slots=True)
class LowpassFilterStage(AudioPreprocessor):
    cutoff_hz: float
    order: int = 4

    def process(
        self,
        samples: np.ndarray,
        sample_rate_hz: int,
        *,
        node_id: str | None = None,
    ) -> np.ndarray:
        del node_id
        if self.cutoff_hz <= 0.0 or samples.size < 16:
            return samples
        cutoff = _clamp_cutoff(self.cutoff_hz, sample_rate_hz)
        sos = butter(self.order, cutoff, btype="lowpass", fs=float(sample_rate_hz), output="sos")
        return sosfiltfilt(sos, samples).astype(np.float32)


@dataclass(slots=True)
class AudioPreprocessingChain(AudioPreprocessor):
    stages: list[AudioPreprocessor]

    def process(
        self,
        samples: np.ndarray,
        sample_rate_hz: int,
        *,
        node_id: str | None = None,
    ) -> np.ndarray:
        output = samples.astype(np.float32, copy=False)
        for stage in self.stages:
            output = stage.process(output, sample_rate_hz, node_id=node_id)
        return output


class NodePreprocessorFactory:
    """Build per-node preprocessing chains from Settings and node properties."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def for_node(self, node: NodeSpec) -> AudioPreprocessor:
        if not self._settings.preprocess_enabled:
            return AudioPreprocessingChain(stages=[])

        cfg = self._node_cfg(node.properties)
        if cfg.get("enabled") is False:
            return AudioPreprocessingChain(stages=[])

        highpass = float(cfg.get("highpass_hz", self._settings.audio_highpass_hz))
        lowpass = float(cfg.get("lowpass_hz", self._settings.audio_lowpass_hz))

        stages: list[AudioPreprocessor] = []
        if highpass > 0.0:
            stages.append(HighpassFilterStage(cutoff_hz=highpass))
        if lowpass > 0.0:
            stages.append(LowpassFilterStage(cutoff_hz=lowpass))
        return AudioPreprocessingChain(stages=stages)

    @staticmethod
    def _node_cfg(properties: dict[str, Any]) -> dict[str, Any]:
        raw = properties.get("preprocess")
        if isinstance(raw, dict):
            return raw
        return {}

