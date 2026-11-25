"""Data carriers passed between FusionNode pipeline stages.

Kept apart from ``fusion_node.py`` so stage collaborators (for example
``localized_render_handoff.py``) can build them without importing the
coordinator, which imports those collaborators.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from minimappr.core.audio_buffer import AudioCoverageStats
from minimappr.models import DetectionEvent, TimeQuality, TrackState


@dataclass(slots=True)
class EventCandidate:
    id: str
    source_node_id: str
    event_time_ns: int
    sample_rate_hz: int
    source_type: str
    time_quality: TimeQuality
    source_observation_ids: list[str]
    # Wall-clock time the candidate was enqueued onto `_localization_queue`.
    # Used to compute queue-wait latency (mirrors the Rust sidecar's
    # `IngestDiagnostics.record_queue_wait_ms`, see diagnostics.rs) for the
    # `/api/v1/diagnostics/summary` side-by-side comparison.
    enqueued_ns: int = 0


@dataclass(slots=True)
class LocalizedCandidate:
    candidate: EventCandidate
    localization_branch: LocalizationBranch | None
    selected_sensor_ids: list[str]
    selected_windows: dict[str, np.ndarray]
    selected_positions: dict[str, np.ndarray]
    classification_selected_windows: dict[str, np.ndarray]
    omni_reference_sensor: str
    omni_reference_signal: np.ndarray
    omni_position_m: tuple[float, float, float]
    omni_classification_reference_signal: np.ndarray
    localization_audio_quality: dict[str, AudioCoverageStats]
    classification_audio_quality: dict[str, AudioCoverageStats]
    classification_audio_quality_source: str
    environment: dict[str, Any]
    extra_classification_features: dict[str, Any] = field(default_factory=dict)
    # Admission score in [0,1] computed once at classification-enqueue time (see
    # core/classification_priority.py). Ordering only; never affects inference.
    classification_significance: float = 0.0
    # Field Mode W3-attribution: ClusterAttributionContext from the onset stage (None = no attribution).
    onset_attribution_context: Any = None
    # Closeout B2: parked by window label reconciliation; a sibling or the timeout emits it.
    held_for_window_label_reconciliation: bool = False
    # No render or clip to classify (too few sensors, beamform error): "could not
    # classify", not "classified as nothing", so a held fix applies geometry-only.
    classification_unavailable: bool = False
    # P7: work a classification worker runs instead of the beamformed classification
    # (a point-network Rust render's clip classification, ``localized_render_reporting``);
    # returns the detections to deliver. None for every Python-ingest product.
    deferred_classification: Callable[[], Awaitable[list["DetectionProduct"]]] | None = None


@dataclass(slots=True)
class DetectionProduct:
    detection: DetectionEvent
    track: TrackState | None
    suppressed_by_zone: bool
    suppression_reasons: list[str]
    pipeline_item_id: str
    event_time_ns: int
    entity_id: str | None = None  # Field Mode W3-entity


@dataclass(slots=True)
class LocalizationBranch:
    localization_position_m: tuple[float, float, float]
    localization_confidence: float
    localization_gdop: float
    localization_position_covariance_m2: list[list[float]] | None
    localization_range_observability: float | None
    localization_residual_rms_seconds: float | None
    localization_range_projection_mode: str | None
    reference_sensor: str
    reference_signal: np.ndarray
    classification_reference_signal: np.ndarray
    tdoa_s: dict[str, float]
    localization_method: str
    capability_tier: str
    wavelength_factor: float | None = None
    dominant_frequency_hz: float | None = None
    alias_cutoff_hz: float | None = None
