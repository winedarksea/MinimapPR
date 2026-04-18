"""Audio preprocessing chain with pluggable filter stages.

Every stage implements the :class:`~minimappr.interfaces.AudioPreprocessor`
protocol.  Stages are composed into an :class:`AudioPreprocessingChain` which
itself satisfies the same protocol, so chains nest arbitrarily.

Built-in stages
~~~~~~~~~~~~~~~~
* :class:`HighpassFilterStage`   – Butterworth highpass
* :class:`LowpassFilterStage`    – Butterworth lowpass
* :class:`BandpassFilterStage`   – Butterworth bandpass (convenience wrapper)
* :class:`DCRemovalStage`        – zero-mean removal
* :class:`NormalizationStage`    – peak or RMS normalization
* :class:`SpectralGateStage`     – simple spectral noise gate

Adding a custom stage
~~~~~~~~~~~~~~~~~~~~~~
Implement the ``AudioPreprocessor`` protocol::

    @dataclass(slots=True)
    class MyStage(AudioPreprocessor):
        some_param: float = 1.0

        def process(self, samples, sample_rate_hz, *, node_id=None):
            return my_transform(samples, self.some_param)

Then insert it into any chain — the classification preprocessor, the per-node
preprocessor, or an entirely new pipeline — using
:func:`build_preprocessing_chain`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

from minimappr.config import LocalizationConfig, Settings
from minimappr.interfaces import AudioPreprocessor
from minimappr.models import NodeSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp_cutoff(cutoff_hz: float, sample_rate_hz: int) -> float:
    nyquist = 0.5 * float(sample_rate_hz)
    return float(min(max(cutoff_hz, 1.0), nyquist * 0.95))


# ---------------------------------------------------------------------------
# Built-in stages — all satisfy AudioPreprocessor protocol
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class HighpassFilterStage(AudioPreprocessor):
    """Butterworth highpass filter."""

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
    """Butterworth lowpass filter."""

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
class BandpassFilterStage(AudioPreprocessor):
    """Butterworth bandpass — convenience wrapper combining low+high cut."""

    low_hz: float
    high_hz: float
    order: int = 4

    def process(
        self,
        samples: np.ndarray,
        sample_rate_hz: int,
        *,
        node_id: str | None = None,
    ) -> np.ndarray:
        del node_id
        if samples.size < 16:
            return samples
        low = _clamp_cutoff(self.low_hz, sample_rate_hz)
        high = _clamp_cutoff(self.high_hz, sample_rate_hz)
        if low >= high:
            return samples
        sos = butter(self.order, [low, high], btype="bandpass", fs=float(sample_rate_hz), output="sos")
        return sosfiltfilt(sos, samples).astype(np.float32)


@dataclass(slots=True)
class DCRemovalStage(AudioPreprocessor):
    """Remove DC offset (zero-mean the signal)."""

    def process(
        self,
        samples: np.ndarray,
        sample_rate_hz: int,
        *,
        node_id: str | None = None,
    ) -> np.ndarray:
        del node_id, sample_rate_hz
        if samples.size == 0:
            return samples
        return (samples - np.mean(samples)).astype(np.float32)


@dataclass(slots=True)
class NormalizationStage(AudioPreprocessor):
    """Peak or RMS normalization.

    *mode* can be ``"peak"`` (scale so max |sample| == *target_level*) or
    ``"rms"`` (scale so RMS == *target_level*).
    """

    target_level: float = 1.0
    mode: str = "peak"  # "peak" or "rms"

    def process(
        self,
        samples: np.ndarray,
        sample_rate_hz: int,
        *,
        node_id: str | None = None,
    ) -> np.ndarray:
        del node_id, sample_rate_hz
        if samples.size == 0:
            return samples

        if self.mode == "rms":
            current = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        else:
            current = float(np.max(np.abs(samples)))

        if current < 1e-12:
            return samples
        scale = self.target_level / current
        return (samples * scale).astype(np.float32)


@dataclass(slots=True)
class SpectralGateStage(AudioPreprocessor):
    """Simple spectral noise gate.

    FFT bins whose magnitude is below *threshold_factor* × median magnitude
    are zeroed, then the signal is reconstructed via inverse FFT.
    Useful for suppressing broadband background noise before classification.
    """

    threshold_factor: float = 1.5

    def process(
        self,
        samples: np.ndarray,
        sample_rate_hz: int,
        *,
        node_id: str | None = None,
    ) -> np.ndarray:
        del node_id, sample_rate_hz
        if samples.size < 16:
            return samples
        spectrum = np.fft.rfft(samples.astype(np.float64))
        magnitudes = np.abs(spectrum)
        median_mag = float(np.median(magnitudes))
        gate_mask = magnitudes >= (self.threshold_factor * median_mag)
        gated_spectrum = spectrum * gate_mask
        return np.fft.irfft(gated_spectrum, n=samples.size).astype(np.float32)


@dataclass(slots=True)
class GainStage(AudioPreprocessor):
    """Apply a simple amplitude multiplier.

    Useful for scaling up faint signals prior to feature extraction.
    """

    multiplier: float = 1.0

    def process(
        self,
        samples: np.ndarray,
        sample_rate_hz: int,
        *,
        node_id: str | None = None,
    ) -> np.ndarray:
        del node_id, sample_rate_hz
        if samples.size == 0 or self.multiplier == 1.0:
            return samples
        return (samples * self.multiplier).astype(np.float32)


# ---------------------------------------------------------------------------
# Chain — composes arbitrary stages into a single AudioPreprocessor
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AudioPreprocessingChain(AudioPreprocessor):
    """Ordered sequence of ``AudioPreprocessor`` stages executed in series.

    The chain itself satisfies the protocol, so chains nest arbitrarily::

        inner = AudioPreprocessingChain([DCRemovalStage(), HighpassFilterStage(50.0)])
        outer = AudioPreprocessingChain([inner, NormalizationStage()])
    """

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


# ---------------------------------------------------------------------------
# Stage registry — maps short names to constructors
# ---------------------------------------------------------------------------

#: Registry of built-in stage names → factory callables.
#: Custom stages can be registered at runtime via ``register_stage``.
_STAGE_REGISTRY: dict[str, type] = {
    "highpass": HighpassFilterStage,
    "lowpass": LowpassFilterStage,
    "bandpass": BandpassFilterStage,
    "dc_remove": DCRemovalStage,
    "gain": GainStage,
    "normalize": NormalizationStage,
    "spectral_gate": SpectralGateStage,
}


def register_stage(name: str, cls: type) -> None:
    """Register a custom preprocessing stage class by name.

    The class must satisfy the ``AudioPreprocessor`` protocol.
    After registration, the stage can be referenced by name in
    :func:`build_preprocessing_chain` and config-driven pipelines.
    """
    _STAGE_REGISTRY[name] = cls


def available_stages() -> list[str]:
    """Return sorted list of registered stage names."""
    return sorted(_STAGE_REGISTRY.keys())


def create_stage(name: str, **kwargs: Any) -> AudioPreprocessor:
    """Instantiate a preprocessing stage by registered name.

    Raises ``ValueError`` if the name is not registered.
    """
    cls = _STAGE_REGISTRY.get(name.strip().lower())
    if cls is None:
        raise ValueError(
            f"Unknown preprocessing stage {name!r}. "
            f"Available: {available_stages()}"
        )
    return cls(**kwargs)  # type: ignore[call-arg]


def build_preprocessing_chain(
    stage_specs: list[dict[str, Any]],
) -> AudioPreprocessingChain:
    """Build a chain from a list of stage specifications.

    Each spec is a dict with a ``"name"`` key and optional kwargs::

        build_preprocessing_chain([
            {"name": "dc_remove"},
            {"name": "bandpass", "low_hz": 100, "high_hz": 4000},
            {"name": "normalize", "mode": "rms", "target_level": 0.5},
        ])

    This is the primary entry point for config-driven pipeline construction.
    """
    stages: list[AudioPreprocessor] = []
    for spec in stage_specs:
        spec = dict(spec)  # shallow copy
        name = spec.pop("name")
        stages.append(create_stage(name, **spec))
    return AudioPreprocessingChain(stages=stages)


# ---------------------------------------------------------------------------
# Per-node preprocessor factory (used by ingest pipeline)
# ---------------------------------------------------------------------------

class NodePreprocessorFactory:
    """Build per-node preprocessing chains from Settings and node properties."""

    def __init__(self, settings: Settings | LocalizationConfig) -> None:
        self._settings = settings.localization_config() if isinstance(settings, Settings) else settings

    def for_node(self, node: NodeSpec) -> AudioPreprocessor:
        if not self._settings.preprocess_enabled:
            return AudioPreprocessingChain(stages=[])

        cfg = self._node_cfg(node.properties)
        if cfg.get("enabled") is False:
            return AudioPreprocessingChain(stages=[])

        highpass = float(cfg.get("highpass_hz", self._settings.audio_highpass_hz))
        lowpass = float(cfg.get("lowpass_hz", self._settings.audio_lowpass_hz))
        gain = float(cfg.get("gain_multiplier", self._settings.ingest_gain_multiplier))

        stages: list[AudioPreprocessor] = []
        if gain != 1.0:
            stages.append(GainStage(multiplier=gain))
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


# ---------------------------------------------------------------------------
# Classification preprocessor factory
# ---------------------------------------------------------------------------

def create_classification_preprocessor(
    config: LocalizationConfig,
    *,
    extra_stages: list[AudioPreprocessor] | None = None,
) -> AudioPreprocessor | None:
    """Build an optional preprocessing chain applied *after* beamforming
    and *before* classification.

    Returns ``None`` when no stages are configured, so callers can skip
    the step entirely for zero overhead.

    Parameters
    ----------
    config:
        Localization configuration — ``pre_classification_highpass_hz`` and
        ``pre_classification_lowpass_hz`` drive the default filters.
    extra_stages:
        Additional custom stages appended after the default filters.
        Pass any ``AudioPreprocessor`` implementation here for full
        flexibility without modifying config or this function.
    """
    stages: list[AudioPreprocessor] = []
    if config.pre_classification_highpass_hz > 0.0:
        stages.append(HighpassFilterStage(cutoff_hz=config.pre_classification_highpass_hz))
    if config.pre_classification_lowpass_hz > 0.0:
        stages.append(LowpassFilterStage(cutoff_hz=config.pre_classification_lowpass_hz))
    if extra_stages:
        stages.extend(extra_stages)
    if not stages:
        return None
    return AudioPreprocessingChain(stages=stages)


def create_localization_preprocessor(
    config: LocalizationConfig,
    *,
    extra_stages: list[AudioPreprocessor] | None = None,
) -> AudioPreprocessor | None:
    """Build an optional preprocessing chain applied only on the localization path."""
    stages: list[AudioPreprocessor] = []
    if (
        config.localization_band_min_hz > 0.0
        and config.localization_band_max_hz > config.localization_band_min_hz
    ):
        stages.append(
            BandpassFilterStage(
                low_hz=config.localization_band_min_hz,
                high_hz=config.localization_band_max_hz,
            )
        )
    if extra_stages:
        stages.extend(extra_stages)
    if not stages:
        return None
    return AudioPreprocessingChain(stages=stages)
