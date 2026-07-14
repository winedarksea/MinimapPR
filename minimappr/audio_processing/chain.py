"""Chain construction, validation, and per-node ingest ownership."""

from __future__ import annotations

import logging
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from minimappr.audio_processing.stages import (
    BandpassFilterStage,
    BoundedRmsGainStage,
    ChannelGainStage,
    DCBlockStage,
    DCRemovalStage,
    GainStage,
    HighpassFilterStage,
    LowpassFilterStage,
    MeanCenterStage,
    NormalizationStage,
    PassthroughStage,
    SpectralGateStage,
)
from minimappr.config import LocalizationConfig, Settings
from minimappr.interfaces import AudioPreprocessor
from minimappr.models import NodeSpec

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AudioProcessingChain(AudioPreprocessor):
    stages: list[AudioPreprocessor]

    def process(self, samples, sample_rate_hz, *, node_id=None, channel_idx=0):
        output = np.asarray(samples, dtype=np.float32)
        for stage in self.stages:
            output = stage.process(output, sample_rate_hz, node_id=node_id, channel_idx=channel_idx)
        return output

    def reset(self) -> None:
        for stage in self.stages:
            reset = getattr(stage, "reset", None)
            if callable(reset):
                reset()


_STAGE_REGISTRY: dict[str, type] = {
    "highpass": HighpassFilterStage,
    "lowpass": LowpassFilterStage,
    "bandpass": BandpassFilterStage,
    "dc_block": DCBlockStage,
    "dc_remove": DCRemovalStage,
    "mean_center": MeanCenterStage,
    "gain": GainStage,
    "channel_gain": ChannelGainStage,
    "bounded_rms_gain": BoundedRmsGainStage,
    "normalize": NormalizationStage,
    "spectral_gate": SpectralGateStage,
    "passthrough": PassthroughStage,
}


def register_stage(name: str, cls: type) -> None:
    _STAGE_REGISTRY[name.strip().lower()] = cls


def available_stages() -> list[str]:
    return sorted(_STAGE_REGISTRY)


def create_stage(name: str, **kwargs: Any) -> AudioPreprocessor:
    cls = _STAGE_REGISTRY.get(name.strip().lower())
    if cls is None:
        raise ValueError(f"Unknown preprocessing stage {name!r}. Available: {available_stages()}")
    return cls(**kwargs)


def _canonical_stage(raw: dict[str, Any], *, ingest: bool) -> AudioPreprocessor:
    spec = dict(raw)
    stage_type = str(spec.pop("type", spec.pop("name", ""))).strip().lower()
    if ingest and stage_type in {"bounded_rms_gain", "normalize", "spectral_gate", "mean_center", "dc_remove"}:
        raise ValueError(f"Adaptive or window-relative stage {stage_type!r} is not allowed at ingest")
    if stage_type == "gain":
        if "db" in spec:
            gain_db = float(spec["db"])
            if not np.isfinite(gain_db) or not -60.0 <= gain_db <= 60.0:
                raise ValueError("gain db must be finite and in [-60, 60]")
            return GainStage(multiplier=10.0 ** (gain_db / 20.0))
        return GainStage(multiplier=float(spec.get("multiplier", 1.0)))
    if stage_type == "channel_gain":
        gains_db = spec.get("db_by_channel", spec.get("gains_db", []))
        if not gains_db or any(
            not np.isfinite(float(value)) or not -60.0 <= float(value) <= 60.0
            for value in gains_db
        ):
            raise ValueError("channel gains must be finite, non-empty, and in [-60, 60]")
        return ChannelGainStage(tuple(10.0 ** (float(value) / 20.0) for value in gains_db))
    if stage_type in {"dc_block", "dc_remove", "mean_center", "passthrough"}:
        return create_stage(stage_type)
    if stage_type in {"highpass", "lowpass"}:
        cutoff_hz = float(spec.get("cutoff_hz", 0.0))
        order = int(spec.get("order", 4))
        if not np.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
            raise ValueError("filter cutoff_hz must be finite and positive")
        if order < 2 or order > 12 or order % 2:
            raise ValueError("filter order must be even and in [2, 12]")
    return create_stage(stage_type, **spec)


def build_preprocessing_chain(stage_specs: list[dict[str, Any]], *, ingest: bool = False) -> AudioProcessingChain:
    return AudioProcessingChain([_canonical_stage(spec, ingest=ingest) for spec in stage_specs])


def build_chain_from_rust_stages(rust_stages: list[dict[str, Any]]) -> AudioProcessingChain:
    stages: list[AudioPreprocessor] = []
    for raw in rust_stages:
        stage_type = str(raw.get("type", "")).strip().lower()
        if stage_type == "passthrough":
            continue
        if stage_type == "bandpass":
            order = int(raw.get("order", 4))
            stages.extend(
                [
                    HighpassFilterStage(cutoff_hz=float(raw["low_hz"]), order=order),
                    LowpassFilterStage(cutoff_hz=float(raw["high_hz"]), order=order),
                ]
            )
            continue
        try:
            stages.append(_canonical_stage(raw, ingest=True))
        except ValueError as exc:
            if "Unknown preprocessing stage" in str(exc):
                raise ValueError(f"Unknown PreprocessStage type {stage_type!r}") from exc
            raise
    return AudioProcessingChain(stages)


class NodePreprocessorFactory:
    """Build stateful chains once per node, with explicit override precedence."""

    def __init__(self, settings: Settings | LocalizationConfig) -> None:
        self._settings = settings.localization_config() if isinstance(settings, Settings) else settings
        self._node_overrides: dict[str, dict] = (
            {node_id: dict(value) for node_id, value in settings.node_audio_overrides.items()}
            if isinstance(settings, Settings)
            else {}
        )
        self._chain_cache: dict[str, AudioProcessingChain] = {}
        self._chain_signature: dict[str, tuple] = {}
        self._latest_metrics: dict[str, dict[str, Any]] = {}

    def set_node_override(self, node_id: str, override: dict | None) -> None:
        if override is None:
            self._node_overrides.pop(node_id, None)
        else:
            self._node_overrides[node_id] = dict(override)
        self._invalidate(node_id)

    def _invalidate(self, node_id: str) -> None:
        chain = self._chain_cache.pop(node_id, None)
        self._chain_signature.pop(node_id, None)
        if chain is not None:
            chain.reset()

    def for_node(self, node: NodeSpec) -> AudioProcessingChain:
        signature = self._signature(node)
        if node.id in self._chain_cache and self._chain_signature.get(node.id) == signature:
            return self._chain_cache[node.id]
        self._invalidate(node.id)
        chain = self._build(node)
        self._chain_cache[node.id] = chain
        self._chain_signature[node.id] = signature
        return chain

    def resolved_default_stages(self) -> list[dict[str, Any]]:
        if not self._settings.preprocess_enabled:
            return []
        stages: list[dict[str, Any]] = []
        gain = float(self._settings.ingest_gain_multiplier)
        if gain != 1.0:
            stages.append({"type": "gain", "db": 20.0 * float(np.log10(gain))})
        if self._settings.audio_highpass_hz > 0.0:
            stages.append({"type": "highpass", "cutoff_hz": self._settings.audio_highpass_hz, "order": 4})
        if self._settings.audio_lowpass_hz > 0.0:
            stages.append({"type": "lowpass", "cutoff_hz": self._settings.audio_lowpass_hz, "order": 4})
        return stages

    def record_frame_metrics(self, node_id: str, metrics: dict[str, Any]) -> None:
        previous = self._latest_metrics.get(node_id, {})
        updated = dict(metrics)
        updated["clipping_risk_sample_count_total"] = int(
            previous.get("clipping_risk_sample_count_total", 0)
        ) + int(metrics.get("clipping_risk_sample_count", 0))
        signature = repr(self._chain_signature.get(node_id, ()))
        updated["profile_fingerprint"] = hashlib.sha256(signature.encode()).hexdigest()[:16]
        chain = self._chain_cache.get(node_id)
        scalar_gain_db = 0.0
        channel_gain_stage = None
        if chain is not None:
            for stage in chain.stages:
                if isinstance(stage, GainStage):
                    scalar_gain_db += 20.0 * float(np.log10(stage.multiplier))
                elif isinstance(stage, ChannelGainStage):
                    channel_gain_stage = stage
        if channel_gain_stage is None:
            updated["fixed_gain_db_by_channel"] = [scalar_gain_db] if scalar_gain_db else []
        else:
            updated["fixed_gain_db_by_channel"] = [
                scalar_gain_db + 20.0 * float(np.log10(multiplier))
                for multiplier in channel_gain_stage.multipliers_by_channel
            ]
        self._latest_metrics[node_id] = updated

    def metrics_snapshot(self) -> dict[str, dict[str, Any]]:
        return {node_id: dict(metrics) for node_id, metrics in self._latest_metrics.items()}

    def fixed_gain_db_for_sensor(self, sensor_id: str) -> float:
        parsed_node_id, separator, channel_text = sensor_id.rpartition(":ch")
        node_id = parsed_node_id if separator else sensor_id
        channel_index = int(channel_text) if separator and channel_text.isdigit() else 0
        override = self._node_overrides.get(node_id, {})
        stages = override.get("stages") if isinstance(override, dict) else None
        if isinstance(stages, list):
            gain_db = 0.0
            for stage in stages:
                if stage.get("type") == "gain":
                    gain_db += float(stage.get("db", 0.0))
                elif stage.get("type") == "channel_gain":
                    values = stage.get("db_by_channel", [])
                    if channel_index < len(values):
                        gain_db += float(values[channel_index])
            return gain_db
        values = override.get("channel_gains_db", override.get("mic_gains_db"))
        if isinstance(values, list) and channel_index < len(values):
            return float(values[channel_index])
        base_gain_db = 20.0 * float(np.log10(self._settings.ingest_gain_multiplier))
        return base_gain_db + float(override.get("gain_db", 0.0))

    def _build(self, node: NodeSpec) -> AudioProcessingChain:
        if not self._settings.preprocess_enabled:
            return AudioProcessingChain([])
        node_config = self._node_config(node)
        if node_config.get("enabled") is False:
            return AudioProcessingChain([])
        override = self._node_overrides.get(node.id, {})
        explicit_stages = override.get("stages")
        if isinstance(explicit_stages, list):
            return build_chain_from_rust_stages(explicit_stages)

        stages: list[dict[str, Any]] = []
        mic_gains_db = override.get("channel_gains_db", override.get("mic_gains_db"))
        if isinstance(mic_gains_db, list) and mic_gains_db:
            stages.append({"type": "channel_gain", "db_by_channel": mic_gains_db})
        elif override:
            gain_db = float(override.get("gain_db", 0.0))
            if abs(gain_db) > 1e-12:
                stages.append({"type": "gain", "db": gain_db})
        else:
            base_gain = float(
                node_config.get("gain_multiplier", 1.0)
                if node_config
                else self._settings.ingest_gain_multiplier
            )
            if abs(base_gain - 1.0) > 1e-12:
                stages.append({"type": "gain", "db": 20.0 * float(np.log10(base_gain))})
        # A node-specific config is a complete replacement. Missing filter
        # fields mean disabled, not inheritance from the global chain.
        highpass = float(
            override.get("hp_hz", 0.0)
            if override
            else node_config.get("highpass_hz", 0.0)
            if node_config
            else self._settings.audio_highpass_hz
        )
        lowpass = float(
            override.get("lp_hz", 0.0)
            if override
            else node_config.get("lowpass_hz", 0.0)
            if node_config
            else self._settings.audio_lowpass_hz
        )
        if highpass > 0.0:
            stages.append({"type": "highpass", "cutoff_hz": highpass, "order": 4})
        if lowpass > 0.0:
            stages.append({"type": "lowpass", "cutoff_hz": lowpass, "order": 4})
        return build_preprocessing_chain(stages, ingest=True)

    def _signature(self, node: NodeSpec) -> tuple:
        return (
            self._settings.preprocess_enabled,
            self._settings.ingest_gain_multiplier,
            self._settings.audio_highpass_hz,
            self._settings.audio_lowpass_hz,
            repr(self._node_config(node)),
            repr(self._node_overrides.get(node.id)),
        )

    @staticmethod
    def _node_config(node: NodeSpec) -> dict[str, Any]:
        raw = node.properties.get("preprocess") if isinstance(node.properties, dict) else None
        return raw if isinstance(raw, dict) else {}


def create_classification_preprocessor(config: LocalizationConfig, *, extra_stages=None):
    stages: list[AudioPreprocessor] = []
    if config.pre_classification_highpass_hz > 0.0:
        stages.append(HighpassFilterStage(cutoff_hz=config.pre_classification_highpass_hz))
    if config.pre_classification_lowpass_hz > 0.0:
        stages.append(LowpassFilterStage(cutoff_hz=config.pre_classification_lowpass_hz))
    stages.extend(extra_stages or [])
    return AudioProcessingChain(stages) if stages else None


def create_localization_preprocessor(config: LocalizationConfig, *, extra_stages=None):
    stages: list[AudioPreprocessor] = []
    if config.localization_band_min_hz > 0.0 and config.localization_band_max_hz > config.localization_band_min_hz:
        stages.append(BandpassFilterStage(low_hz=config.localization_band_min_hz, high_hz=config.localization_band_max_hz))
    stages.extend(extra_stages or [])
    return AudioProcessingChain(stages) if stages else None
