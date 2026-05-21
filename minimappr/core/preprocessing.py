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
from scipy.signal import butter, sosfilt

from minimappr.config import LocalizationConfig, Settings
from minimappr.interfaces import AudioPreprocessor
from minimappr.models import NodeSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Causality is required for parity with the Rust sidecar's per-stream stateful
# biquad cascade. `sosfilt` is single-pass forward-only; `sosfiltfilt` (the
# previous choice) was forward+backward, producing zero phase delay but
# non-causal output. The parity tests assert numerical equivalence between
# Python and Rust, which is only achievable with a causal filter on both sides.
# Edge transients at the start of each call are absorbed by carrying the
# per-(stage, channel) `zi` state across `process()` invocations.

def _clamp_cutoff(cutoff_hz: float, sample_rate_hz: int) -> float:
    nyquist = 0.5 * float(sample_rate_hz)
    return float(min(max(cutoff_hz, 1.0), nyquist * 0.95))


def _design_sos(
    *,
    btype: str,
    cutoff_hz: float | tuple[float, float],
    order: int,
    sample_rate_hz: int,
) -> np.ndarray:
    """Design a Butterworth SOS for the given band — shared by all filter stages
    so that the design path is identical regardless of the variant."""
    return butter(order, cutoff_hz, btype=btype, fs=float(sample_rate_hz), output="sos")


def _apply_sosfilt(
    sos: np.ndarray,
    samples: np.ndarray,
    state: dict[int, np.ndarray],
    channel_idx: int,
) -> np.ndarray:
    """Apply `sosfilt` carrying per-channel `zi` state across calls. State for
    a channel is initialized to zeros on first encounter and updated in-place."""
    n_sections = sos.shape[0]
    zi = state.get(channel_idx)
    if zi is None or zi.shape != (n_sections, 2):
        zi = np.zeros((n_sections, 2), dtype=np.float64)
    out, zf = sosfilt(sos, samples.astype(np.float64, copy=False), zi=zi)
    state[channel_idx] = zf
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Built-in stages — all satisfy AudioPreprocessor protocol
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class HighpassFilterStage(AudioPreprocessor):
    """Butterworth highpass filter — causal `sosfilt` with per-channel zi state."""

    cutoff_hz: float
    order: int = 4
    _state: dict[int, np.ndarray] = field(default_factory=dict)
    _designed_for_sample_rate_hz: int | None = None

    def process(
        self,
        samples: np.ndarray,
        sample_rate_hz: int,
        *,
        node_id: str | None = None,
        channel_idx: int = 0,
    ) -> np.ndarray:
        del node_id
        if self.cutoff_hz <= 0.0 or samples.size < 16:
            return samples
        if self._designed_for_sample_rate_hz != sample_rate_hz:
            self._state.clear()
            self._designed_for_sample_rate_hz = sample_rate_hz
        cutoff = _clamp_cutoff(self.cutoff_hz, sample_rate_hz)
        sos = _design_sos(
            btype="highpass",
            cutoff_hz=cutoff,
            order=self.order,
            sample_rate_hz=sample_rate_hz,
        )
        return _apply_sosfilt(sos, samples, self._state, channel_idx)

    def reset(self) -> None:
        self._state.clear()
        self._designed_for_sample_rate_hz = None


@dataclass(slots=True)
class LowpassFilterStage(AudioPreprocessor):
    """Butterworth lowpass filter — causal `sosfilt` with per-channel zi state."""

    cutoff_hz: float
    order: int = 4
    _state: dict[int, np.ndarray] = field(default_factory=dict)
    _designed_for_sample_rate_hz: int | None = None

    def process(
        self,
        samples: np.ndarray,
        sample_rate_hz: int,
        *,
        node_id: str | None = None,
        channel_idx: int = 0,
    ) -> np.ndarray:
        del node_id
        if self.cutoff_hz <= 0.0 or samples.size < 16:
            return samples
        if self._designed_for_sample_rate_hz != sample_rate_hz:
            self._state.clear()
            self._designed_for_sample_rate_hz = sample_rate_hz
        cutoff = _clamp_cutoff(self.cutoff_hz, sample_rate_hz)
        sos = _design_sos(
            btype="lowpass",
            cutoff_hz=cutoff,
            order=self.order,
            sample_rate_hz=sample_rate_hz,
        )
        return _apply_sosfilt(sos, samples, self._state, channel_idx)

    def reset(self) -> None:
        self._state.clear()
        self._designed_for_sample_rate_hz = None


@dataclass(slots=True)
class BandpassFilterStage(AudioPreprocessor):
    """Butterworth bandpass — convenience wrapper combining low+high cut."""

    low_hz: float
    high_hz: float
    order: int = 4
    _state: dict[int, np.ndarray] = field(default_factory=dict)
    _designed_for_sample_rate_hz: int | None = None

    def process(
        self,
        samples: np.ndarray,
        sample_rate_hz: int,
        *,
        node_id: str | None = None,
        channel_idx: int = 0,
    ) -> np.ndarray:
        del node_id
        if samples.size < 16:
            return samples
        low = _clamp_cutoff(self.low_hz, sample_rate_hz)
        high = _clamp_cutoff(self.high_hz, sample_rate_hz)
        if low >= high:
            return samples
        if self._designed_for_sample_rate_hz != sample_rate_hz:
            self._state.clear()
            self._designed_for_sample_rate_hz = sample_rate_hz
        sos = _design_sos(
            btype="bandpass",
            cutoff_hz=(low, high),
            order=self.order,
            sample_rate_hz=sample_rate_hz,
        )
        return _apply_sosfilt(sos, samples, self._state, channel_idx)

    def reset(self) -> None:
        self._state.clear()
        self._designed_for_sample_rate_hz = None


@dataclass(slots=True)
class DCRemovalStage(AudioPreprocessor):
    """Remove DC offset (zero-mean the signal)."""

    def process(
        self,
        samples: np.ndarray,
        sample_rate_hz: int,
        *,
        node_id: str | None = None,
        channel_idx: int = 0,
    ) -> np.ndarray:
        del node_id, sample_rate_hz, channel_idx
        if samples.size == 0:
            return samples
        return (samples - np.mean(samples)).astype(np.float32)

    def reset(self) -> None:
        # No state to clear.
        return


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
        channel_idx: int = 0,
    ) -> np.ndarray:
        del node_id, sample_rate_hz, channel_idx
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

    def reset(self) -> None:
        return


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
        channel_idx: int = 0,
    ) -> np.ndarray:
        del node_id, sample_rate_hz, channel_idx
        if samples.size < 16:
            return samples
        spectrum = np.fft.rfft(samples.astype(np.float64))
        magnitudes = np.abs(spectrum)
        median_mag = float(np.median(magnitudes))
        gate_mask = magnitudes >= (self.threshold_factor * median_mag)
        gated_spectrum = spectrum * gate_mask
        return np.fft.irfft(gated_spectrum, n=samples.size).astype(np.float32)

    def reset(self) -> None:
        return


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
        channel_idx: int = 0,
    ) -> np.ndarray:
        del node_id, sample_rate_hz, channel_idx
        if samples.size == 0 or self.multiplier == 1.0:
            return samples
        return (samples * self.multiplier).astype(np.float32)

    def reset(self) -> None:
        return


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
        channel_idx: int = 0,
    ) -> np.ndarray:
        output = samples.astype(np.float32, copy=False)
        for stage in self.stages:
            output = stage.process(
                output, sample_rate_hz, node_id=node_id, channel_idx=channel_idx
            )
        return output

    def reset(self) -> None:
        """Clear cached filter state on every stage that supports it. Called by
        the per-node preprocessor cache when the node's override config changes."""
        for stage in self.stages:
            reset_fn = getattr(stage, "reset", None)
            if callable(reset_fn):
                reset_fn()


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


def build_chain_from_rust_stages(
    rust_stages: list[dict[str, Any]],
) -> AudioPreprocessingChain:
    """Build a chain from the cross-language `PreprocessStage` JSON shape.

    This is the canonical shape mirrored between Python `NodeAudioOverride.stages`
    and Rust `NodeAudioConfig.stages` (see `dsp_worker.rs::PreprocessStage`).
    Each spec is a `{"type": "...", ...kwargs}` mapping using snake_case
    variant names. `passthrough` and `dc_block` carry no extra fields::

        build_chain_from_rust_stages([
            {"type": "gain", "db": 6.0},
            {"type": "highpass", "cutoff_hz": 100.0, "order": 4},
            {"type": "lowpass", "cutoff_hz": 4000.0, "order": 4},
        ])

    Unknown variants raise ValueError so a malformed override fails fast on the
    next ingest call rather than silently being dropped.
    """
    stages: list[AudioPreprocessor] = []
    for raw in rust_stages:
        spec = dict(raw)
        stage_type = str(spec.pop("type", "")).strip().lower()
        if stage_type == "passthrough":
            continue
        if stage_type == "gain":
            db = float(spec.get("db", 0.0))
            stages.append(GainStage(multiplier=10.0 ** (db / 20.0)))
            continue
        if stage_type == "highpass":
            stages.append(
                HighpassFilterStage(
                    cutoff_hz=float(spec["cutoff_hz"]),
                    order=int(spec.get("order", 4)),
                )
            )
            continue
        if stage_type == "lowpass":
            stages.append(
                LowpassFilterStage(
                    cutoff_hz=float(spec["cutoff_hz"]),
                    order=int(spec.get("order", 4)),
                )
            )
            continue
        if stage_type == "bandpass":
            stages.append(
                BandpassFilterStage(
                    low_hz=float(spec["low_hz"]),
                    high_hz=float(spec["high_hz"]),
                    order=int(spec.get("order", 4)),
                )
            )
            continue
        if stage_type == "dc_block":
            stages.append(DCRemovalStage())
            continue
        raise ValueError(
            f"Unknown PreprocessStage type {stage_type!r}. "
            "Expected one of: gain, highpass, lowpass, bandpass, dc_block, passthrough."
        )
    return AudioPreprocessingChain(stages=stages)


# ---------------------------------------------------------------------------
# Per-node preprocessor factory (used by ingest pipeline)
# ---------------------------------------------------------------------------

class NodePreprocessorFactory:
    """Build per-node preprocessing chains from Settings and node properties.

    Chains are cached per `node_id` so per-stream filter state survives between
    `for_node()` calls (each ingest frame). On override change, the cached chain
    is invalidated *and* its filter state reset — preventing transient artifacts
    from the previous configuration's state leaking into the new one.
    """

    def __init__(self, settings: Settings | LocalizationConfig) -> None:
        self._settings = settings.localization_config() if isinstance(settings, Settings) else settings
        self._node_overrides: dict[str, dict] = {}
        # Chain cache + the signature used to build each cached entry. When the
        # signature changes (override mutation, properties change), we drop and
        # rebuild — calling reset() on the old chain first so any references
        # held elsewhere also see cleared state.
        self._chain_cache: dict[str, AudioPreprocessor] = {}
        self._chain_signature: dict[str, tuple] = {}

    def set_node_override(self, node_id: str, override: dict | None) -> None:
        """Apply or clear a runtime per-node DSP override (gain_db, hp_hz, lp_hz,
        stages, ...). Invalidates the cached chain for this node so the next
        `for_node()` call rebuilds with fresh coefficients and zero filter state."""
        if override is None:
            self._node_overrides.pop(node_id, None)
        else:
            self._node_overrides[node_id] = override
        self._invalidate(node_id)

    def _invalidate(self, node_id: str) -> None:
        prior = self._chain_cache.pop(node_id, None)
        self._chain_signature.pop(node_id, None)
        if prior is not None:
            reset_fn = getattr(prior, "reset", None)
            if callable(reset_fn):
                reset_fn()

    def for_node(self, node: NodeSpec) -> AudioPreprocessor:
        signature = self._compute_signature(node)
        cached = self._chain_cache.get(node.id)
        if cached is not None and self._chain_signature.get(node.id) == signature:
            return cached
        # Signature changed (or never built) — drop any stale chain.
        self._invalidate(node.id)
        chain = self._build_chain(node)
        self._chain_cache[node.id] = chain
        self._chain_signature[node.id] = signature
        return chain

    def _build_chain(self, node: NodeSpec) -> AudioPreprocessor:
        if not self._settings.preprocess_enabled:
            return AudioPreprocessingChain(stages=[])

        cfg = self._node_cfg(node.properties)
        if cfg.get("enabled") is False:
            return AudioPreprocessingChain(stages=[])

        override = self._node_overrides.get(node.id, {})

        # New canonical path: if the override carries an explicit `stages` list
        # in the cross-language PreprocessStage shape, honor it verbatim.
        explicit_stages = override.get("stages") if isinstance(override, dict) else None
        if isinstance(explicit_stages, list):
            return build_chain_from_rust_stages(explicit_stages)

        highpass = float(override.get("hp_hz", cfg.get("highpass_hz", self._settings.audio_highpass_hz)))
        lowpass = float(override.get("lp_hz", cfg.get("lowpass_hz", self._settings.audio_lowpass_hz)))
        base_gain = float(cfg.get("gain_multiplier", self._settings.ingest_gain_multiplier))
        gain_db = float(override.get("gain_db", 0.0))
        gain = base_gain * (10.0 ** (gain_db / 20.0))

        stages: list[AudioPreprocessor] = []
        if gain != 1.0:
            stages.append(GainStage(multiplier=gain))
        if highpass > 0.0:
            stages.append(HighpassFilterStage(cutoff_hz=highpass))
        if lowpass > 0.0:
            stages.append(LowpassFilterStage(cutoff_hz=lowpass))
        return AudioPreprocessingChain(stages=stages)

    def _compute_signature(self, node: NodeSpec) -> tuple:
        """Build a hashable signature representing every input that affects the
        chain shape. Any change → cache miss → fresh chain (and reset state)."""
        cfg = self._node_cfg(node.properties)
        override = self._node_overrides.get(node.id, {})
        # Convert lists/dicts to hashable form via repr — coarse but safe.
        return (
            bool(self._settings.preprocess_enabled),
            cfg.get("enabled", True),
            float(cfg.get("highpass_hz", self._settings.audio_highpass_hz)),
            float(cfg.get("lowpass_hz", self._settings.audio_lowpass_hz)),
            float(cfg.get("gain_multiplier", self._settings.ingest_gain_multiplier)),
            override.get("hp_hz"),
            override.get("lp_hz"),
            override.get("gain_db"),
            repr(override.get("stages")),
        )

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
