"""Fusion node runtime for staged ingest/localization/classification/rules.

FusionNode is a thin coordinator that delegates to focused subsystem
components:

- ``IngestProcessor`` — PCM decode, preprocessing, buffer insertion,
  observation persistence, and trigger evaluation.
- ``ClassificationOrchestrator`` — omni vs. beamformed classification path
  selection, label resolution, and confidence comparison.
- ``DetectionAssembler`` — builds ``DetectionEvent``, writes snippets,
  persists detections and pings.
- ``RetentionPolicy`` — retention tier determination.

FusionNode owns the pipeline queues, worker lifecycle, and stage routing.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import aiosqlite
import numpy as np

from minimappr.api.rust_dsp_manifests import LocalizedClassifierRenderRequest
from minimappr.classifiers.base import AudioClassifier
from minimappr.config import Settings
from minimappr.core.amplitude_range import amplitude_range_prior_m
from minimappr.core.assembly import DetectionAssembler
from minimappr.core.multi_node_bearing_fusion import (
    BearingFusionResult,
    BearingFusionStore,
    BearingObservation,
    fuse_bearings,
)
from minimappr.core.audio_buffer import AudioCoverageStats, MultiSensorBuffer
from minimappr.core.beamforming import create_beamformer
from minimappr.core.classification_chunking import ClassificationChunkingPolicy
from minimappr.core.classification import ClassificationOrchestrator
from minimappr.core.degradation import CapabilityDegradationModel
from minimappr.core.environment import StaticEnvironmentProvider
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.cluster_registry import ClusterRegistry
from minimappr.core.ingest import EnvironmentUpdater, IngestProcessor
from minimappr.core.live_ingest_state import LiveIngestState
from minimappr.core.ambi_atob import alias_cutoff_from_positions
from minimappr.core.localization import LocalizationError
from minimappr.core.localization_uncertainty import (
    apply_frequency_covariance_scaling,
    clamp_covariance_eigenvalues_range_proportional,
    covariance_to_nested_list,
    range_observability_from_covariance,
)
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.pipeline_realtime import PipelineRealtimeTracker
from minimappr.core.range_projection import (
    LEGACY_PRIOR_PROJECTED,
    RANGE_ASYMPTOTIC,
    RANGE_BEARING_PROJECTED,
    RANGE_BOUNDARY,
    RANGE_REFINED,
    apply_unobservable_range_haircut,
    normalize_range_mode,
)
from minimappr.core.rust_tdoa_solve import solve_localization_from_rust_tdoas
from minimappr.core.preprocessing import (
    NodePreprocessorFactory,
    create_classification_preprocessor,
    create_localization_preprocessor,
)
from minimappr.core.reporting_fusion import ReportingFusionPolicy
from minimappr.core.retention import RetentionPolicy
from minimappr.core.rules import ConfigRuleEngine, LoggingRuleActionHandler, WebsocketRuleActionHandler
from minimappr.core.taxonomy import RuntimeTaxonomyProvider
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.interfaces import (
    ActionDescriptor,
    Beamformer,
    EnvironmentProvider,
    Localizer,
    RuleActionHandler,
    RuleEngine,
    StorageBackend,
    TaxonomyProvider,
)
from minimappr.models import (
    DetectionEvent,
    EnvironmentSampleIn,
    GeoPoint,
    IngestFrameRequest,
    IngestFrameResponse,
    NodeSpec,
    RetentionTier,
    TimeQuality,
    TrackState,
    TranscriptRecord,
)
from minimappr.utils.audio import rms


logger = logging.getLogger(__name__)

# Absolute distance-from-origin backstop for the localization sanity gate. The
# primary gate measures range from the contributing-sensor centroid (Phase 2), so a
# node surveyed far from origin still gets its full 1 km envelope; this backstop only
# catches runaway coordinates well beyond any plausible deployment footprint.
_SANITY_GATE_ORIGIN_BACKSTOP_M = 5000.0


# ---------------------------------------------------------------------------
# Internal pipeline stage data carriers
# ---------------------------------------------------------------------------

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


@dataclass(slots=True)
class DetectionProduct:
    detection: DetectionEvent
    track: TrackState | None
    suppressed_by_zone: bool
    suppression_reasons: list[str]
    pipeline_item_id: str
    event_time_ns: int


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


@dataclass(slots=True)
class FusionMetrics:
    ingest_requests: int = 0
    frames_accepted: int = 0
    frames_rejected: int = 0
    frames_zero_padded_degraded: int = 0
    frame_sequence_gaps: int = 0
    triggers_enqueued: int = 0
    triggers_dropped_queue_full: int = 0
    localization_stage_in: int = 0
    localization_stage_out: int = 0
    localization_failures: int = 0
    classification_stage_in: int = 0
    classification_stage_out: int = 0
    classification_failures: int = 0
    rules_stage_in: int = 0
    rules_stage_out: int = 0
    rules_failures: int = 0
    detections_emitted: int = 0
    detections_suppressed_by_zone: int = 0
    stage_drops_backpressure: int = 0
    beamforming_failures: int = 0
    environment_samples_ingested: int = 0
    environment_samples_persisted: int = 0
    localization_fallback_count: int = 0
    localization_tier_full_3d_count: int = 0
    localization_tier_2d_count: int = 0
    localization_config_bypassed_count: int = 0
    localization_band_aliased_count: int = 0
    localization_prior_projected_count: int = 0
    localization_range_asymptotic_count: int = 0
    localization_range_boundary_count: int = 0
    localization_single_node_python_solved_count: int = 0
    localization_single_node_python_fallback_count: int = 0
    localization_solver_unconverged_count: int = 0
    localization_covariance_missing_count: int = 0
    localization_range_observability_low_count: int = 0
    localization_covariance_range_capped_count: int = 0
    localization_amplitude_prior_used_count: int = 0
    localization_amplitude_prior_clamped_count: int = 0
    track_multi_node_association_count: int = 0
    tracks_multi_node_active: int = 0
    localization_bearing_fusion_attempt_count: int = 0
    localization_bearing_fusion_fused_count: int = 0
    localization_bearing_fusion_degenerate_count: int = 0
    localization_bearing_fusion_stale_count: int = 0
    last_bearing_fusion_contributor_count: int = 0
    localization_cross_node_pairs_measured_count: int = 0
    localization_cross_node_pairs_rejected_sync_count: int = 0
    localization_candidates_coalesced_count: int = 0
    localization_cross_node_wait_timeout_count: int = 0
    last_cross_node_pair_count: int = 0
    localization_rejected_out_of_range_count: int = 0
    localization_stage_total_time_ms: float = 0.0
    localization_stage_max_time_ms: float = 0.0
    # Ingest-to-DSP-result latency, split into queue wait vs. compute time so
    # "slow because of load" and "slow because of compute" are distinguishable.
    # Mirrors the Rust sidecar's `IngestDiagnostics` (`diagnostics.rs`) —
    # field names/semantics are kept parallel for the
    # `/api/v1/diagnostics/summary` side-by-side comparison.
    ingest_queue_wait_total_ms: float = 0.0
    ingest_queue_wait_max_ms: float = 0.0
    ingest_queue_wait_count: int = 0
    ingest_processing_total_ms: float = 0.0
    ingest_processing_max_ms: float = 0.0
    ingest_processing_count: int = 0
    stage_timeout_count: int = 0
    last_localization_algorithm: str = "gcc_phat"
    last_attempted_algorithm: str = "gcc_phat"
    classification_reuse_hits: int = 0
    beamform_renders: int = 0
    beamform_failures: int = 0
    birdnet_chunk_dispatches_suppressed: int = 0
    # Silent-drop visibility: each pipeline stage may return without emitting a
    # downstream item (e.g. localization buffer miss, classifier suppression).
    # These dicts make the previously-invisible drops countable per reason.
    localization_drops_by_reason: dict[str, int] = field(default_factory=dict)
    classification_drops_by_reason: dict[str, int] = field(default_factory=dict)
    rules_drops_by_reason: dict[str, int] = field(default_factory=dict)
    # Worker exception visibility: unexpected exceptions caught by each worker loop.
    # Keyed by exception class name so persistent bugs surface without drowning logs.
    localization_exceptions_by_type: dict[str, int] = field(default_factory=dict)
    classification_exceptions_by_type: dict[str, int] = field(default_factory=dict)
    rules_exceptions_by_type: dict[str, int] = field(default_factory=dict)
    # Wall-clock timestamps used to derive "active drought" — triggers firing
    # while detections stall. Both default to 0 so status() can detect "never".
    last_detection_emission_ns: int = 0
    last_trigger_enqueue_ns: int = 0


# ---------------------------------------------------------------------------
# FusionNode — thin coordinator
# ---------------------------------------------------------------------------


class FusionNode:
    """Pipeline coordinator delegating to focused subsystem components.

    Public API (unchanged from pre-decomposition):
      - ``start()`` / ``stop()`` — lifecycle
      - ``ingest(request)`` — frame ingestion entry point
      - ``status()`` — operational status dict
      - ``housekeeping_tick(now_ns)`` — periodic maintenance
    """

    def __init__(
        self,
        settings: Settings,
        registry: NodeRegistry,
        buffer: MultiSensorBuffer,
        localizer: Localizer,
        classifier: AudioClassifier,
        tracker: TrackManager,
        storage: StorageBackend,
        live_callback,
        coordinate_frame: LocalCoordinateFrame,
        zone_matcher: ZoneMatcher,
        *,
        cluster_registry: ClusterRegistry | None = None,
        taxonomy_provider: TaxonomyProvider | None = None,
        rules_engine: RuleEngine | None = None,
        action_handlers: dict[str, RuleActionHandler] | None = None,
        environment_provider: EnvironmentProvider | None = None,
        preprocessor_factory: NodePreprocessorFactory | None = None,
        degradation_model: CapabilityDegradationModel | None = None,
        beamformer: Beamformer | None = None,
    ) -> None:
        self.settings = settings
        self.classifier_config = settings.classifier_config()
        self.localization_config = settings.localization_config()
        self.fusion_config = settings.fusion_config()
        self.registry = registry
        self.cluster_registry = cluster_registry
        self.buffer = buffer
        self.localizer = localizer
        self.classifier = classifier
        self.tracker = tracker
        self.storage = storage
        self.live_callback = live_callback
        self.coordinate_frame = coordinate_frame
        self.zone_matcher = zone_matcher
        self.taxonomy_provider = taxonomy_provider or RuntimeTaxonomyProvider.from_config_file(settings.taxonomy_config_path)
        self.rules_engine = rules_engine or ConfigRuleEngine(settings.rules_config_path)
        self.environment_provider = environment_provider or StaticEnvironmentProvider(
            temperature_c=settings.default_temperature_c,
            humidity_fraction=settings.default_humidity,
        )
        self.preprocessor_factory = preprocessor_factory or NodePreprocessorFactory(settings)
        self.degradation_model = degradation_model or CapabilityDegradationModel(
            min_sensors_for_3d=settings.min_sensors_for_3d,
            min_sensors_for_2d=settings.min_sensors_for_2d,
        )
        self.beamformer = beamformer or self._create_beamformer()
        self.classification_preprocessor = create_classification_preprocessor(self.localization_config)
        self.localization_preprocessor = create_localization_preprocessor(self.localization_config)
        self.retention_policy = RetentionPolicy(
            permanent_labels=set(self.fusion_config.retention_permanent_labels),
            security_long_confidence=self.fusion_config.retention_long_security_confidence,
        )
        self.reporting_policy = ReportingFusionPolicy(
            storage=storage,
            reporting_window_seconds=self.fusion_config.reporting_window_seconds,
            omni_suppression_scope=self.fusion_config.omni_suppression_scope,
            omni_suppression_max_distance_m=self.fusion_config.omni_suppression_max_distance_m,
            taxonomy_provider=self.taxonomy_provider,
        )
        self._action_handlers = action_handlers or {
            "cop": WebsocketRuleActionHandler(live_callback),
            "log": LoggingRuleActionHandler(),
        }
        self._live_ingest_state = LiveIngestState()

        # -- extracted subsystem components ------------------------------------
        self._ingest_processor = IngestProcessor(
            localization_config=self.localization_config,
            fusion_config=self.fusion_config,
            registry=registry,
            preprocessor_factory=self.preprocessor_factory,
            buffer=buffer,
            storage=storage,
            coordinate_frame=coordinate_frame,
            live_ingest_state=self._live_ingest_state,
            environment_updater=EnvironmentUpdater(self.environment_provider),
            persist_observations_on_ingest=bool(settings.persist_observations_on_ingest),
            node_position_kalman_q=settings.node_position_kalman_q,
            node_position_kalman_q_stationary=settings.node_position_kalman_q_stationary,
            node_position_kalman_r=settings.node_position_kalman_r,
            node_position_kalman_init_p=settings.node_position_kalman_init_p,
            node_position_gps_gate_m=settings.node_position_gps_gate_m,
            node_position_kde_bandwidth_m=settings.node_position_kde_bandwidth_m,
            node_position_kde_reservoir_capacity=settings.node_position_kde_reservoir_capacity,
            node_position_kde_warmup_fixes=settings.node_position_kde_warmup_fixes,
            node_position_kde_recompute_seconds=settings.node_position_kde_recompute_seconds,
            node_position_kde_checkpoint_seconds=settings.node_position_kde_checkpoint_seconds,
            node_position_kde_acceptance_radius_m=settings.node_position_kde_acceptance_radius_m,
        )
        self._classification_orchestrator = ClassificationOrchestrator(
            classifier=classifier,
            storage=storage,
            taxonomy_provider=self.taxonomy_provider,
            environment_provider=self.environment_provider,
            beamformer=self.beamformer,
            classification_preprocessor=self.classification_preprocessor,
            beamformed_classification_min_sensor_count=settings.beamformed_classification_min_sensor_count,
            beamformed_classification_confidence_margin=settings.beamformed_classification_confidence_margin,
            stage_timeout_seconds=self.classifier_config.stage_timeout_seconds,
            on_beamform_error=self._record_beamform_failure,
        )
        self._detection_assembler = DetectionAssembler(
            storage=storage,
            coordinate_frame=coordinate_frame,
            zone_matcher=zone_matcher,
            registry=registry,
            preprocessor_factory=self.preprocessor_factory,
            retention_policy=self.retention_policy,
            snippet_dir=settings.snippet_dir,
            snippet_retention_seconds=settings.snippet_retention_seconds,
            classifier_audio_retention_seconds={
                "yamnet": settings.retention_yamnet_audio_seconds,
                "birdnet": settings.retention_birdnet_audio_seconds,
                "drone_head": settings.retention_drone_audio_seconds,
            },
            event_stale_seconds=settings.event_stale_seconds,
        )

        self._candidate_counter = itertools.count(1)
        self._taxonomy_refresh_ns = 0
        self._per_node_frame_metrics: dict[str, dict[str, int]] = {}

        self._localization_queue: asyncio.Queue[EventCandidate | None] = asyncio.Queue(
            maxsize=max(1, self.fusion_config.localization_queue_size)
        )
        self._classification_queue: asyncio.Queue[LocalizedCandidate | None] = asyncio.Queue(
            maxsize=max(1, self.fusion_config.classification_queue_size)
        )
        self._rules_queue: asyncio.Queue[DetectionProduct | None] = asyncio.Queue(
            maxsize=max(1, self.fusion_config.rules_queue_size)
        )

        self._localization_workers: list[asyncio.Task] = []
        self._classification_workers: list[asyncio.Task] = []
        self._rules_workers: list[asyncio.Task] = []
        self._metrics = FusionMetrics()
        # Phase 4: windowed multi-node bearing triangulation store (server-side,
        # feature-flagged off by default). Bounded + TTL-pruned.
        self._bearing_fusion_store = BearingFusionStore(
            max_entries=256,
            ttl_seconds=self.settings.multi_node_bearing_ttl_seconds,
        )
        self._realtime_tracker = PipelineRealtimeTracker(
            ("localization", "classification", "rules")
        )
        self._classification_chunking_policy = self._build_classification_chunking_policy()
        self._backpressure_warning_last_logged_s: dict[str, float] = {}
        self._backpressure_warning_interval_seconds = 5.0
        self._drop_warning_last_logged_s: dict[tuple[str, str], float] = {}
        self._drop_warning_interval_seconds = 10.0
        self._degraded_audio_warning_last_logged_s: dict[str, float] = {}
        self._degraded_audio_warning_interval_seconds = 10.0
        self._last_error: str | None = None
        self._started = False
        self._stopping = False
        # Per-node firmware transport baselines support edge-triggered degraded
        # CBIT reports and a bounded, explicit recovery report.
        self._last_firmware_runner_counters: dict[str, tuple[int, int]] = {}
        self._firmware_transport_degraded_since_ns: dict[str, int] = {}
        self._firmware_transport_recovery_ns = int(
            settings.node_degraded_after_seconds * 1_000_000_000
        )
        # Continuous omni scanner (Phase 4) — constructed lazily in start() so the
        # omni_continuous classifier (models) is only built when enabled.
        self._omni_scanner: Any = None
        # Speech capture (Phase 6) — constructed eagerly (no model load until a
        # session actually transcribes) so triggers can fire from the first frame.
        self._speech_capture: Any = None
        if self.settings.stt_enabled:
            from minimappr.core.speech_capture import SpeechCaptureManager

            self._speech_capture = SpeechCaptureManager(
                settings=self.settings, buffer=self.buffer, storage=self.storage
            )
            self._speech_capture.add_consumer(self._on_transcript)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            return

        self._stopping = False
        await self._ingest_processor.hydrate_position_estimator_states()
        await self._refresh_taxonomy(force=True)
        worker_count = max(1, self.fusion_config.worker_count)
        self._localization_workers = [
            asyncio.create_task(self._localization_worker_loop(index), name=f"fusion-localize-{index}")
            for index in range(worker_count)
        ]
        self._classification_workers = [
            asyncio.create_task(self._classification_worker_loop(index), name=f"fusion-classify-{index}")
            for index in range(worker_count)
        ]
        self._rules_workers = [
            asyncio.create_task(self._rules_worker_loop(index), name=f"fusion-rules-{index}")
            for index in range(1)
        ]
        await self._start_omni_scanner()
        self._started = True

    async def _start_omni_scanner(self) -> None:
        if not self.settings.omni_scan_enabled or self._omni_scanner is not None:
            return
        try:
            from minimappr.classifiers.factory import create_context_classifier
            from minimappr.classifiers.routing import CONTEXT_OMNI_CONTINUOUS
            from minimappr.core.omni_scanner import ContinuousOmniScanner

            classifier = create_context_classifier(self.settings, CONTEXT_OMNI_CONTINUOUS)
            self._omni_scanner = ContinuousOmniScanner(
                settings=self.settings,
                classifier=classifier,
                registry=self.registry,
                buffer=self.buffer,
                sink=self.ingest_continuous_scan_result,
            )
            await self._omni_scanner.start()
        except Exception:  # noqa: BLE001 — scanner is best-effort, must not block startup
            logger.exception("Failed to start continuous omni scanner; continuing without it.")
            self._omni_scanner = None

    @property
    def accepted_frame_count(self) -> int:
        return self._ingest_processor.accepted_frame_count

    def node_position_estimator_diagnostics(self, node_id: str) -> dict[str, Any]:
        return self._ingest_processor.position_estimator_diagnostics(node_id)

    async def reset_node_position_estimator(self, node_id: str) -> None:
        await self._ingest_processor.reset_position_estimator(node_id)

    def rebind_runtime_dependencies(self, *, classifier: AudioClassifier, coordinate_frame: LocalCoordinateFrame) -> None:
        self.classifier = classifier
        self.coordinate_frame = coordinate_frame
        self._ingest_processor.replace_coordinate_frame(coordinate_frame)
        self._classification_orchestrator.replace_classifier(classifier)
        self._detection_assembler.replace_coordinate_frame(coordinate_frame)

    def set_action_handler(self, destination: str, handler: RuleActionHandler) -> None:
        """Register (or replace) a rule-action handler for a destination.

        Used to wire optional subsystems (e.g. effectors) that may not exist
        yet at FusionNode construction time.
        """
        self._action_handlers[destination] = handler

    async def stop(self) -> None:
        if not self._started:
            return

        self._stopping = True
        self._started = False

        if self._omni_scanner is not None:
            try:
                await self._omni_scanner.stop()
            finally:
                self._omni_scanner = None

        if self._speech_capture is not None:
            self._speech_capture.close()

        # Bound shutdown wait time so lifespan teardown cannot hang indefinitely
        # if in-flight work stalls inside an external backend.
        join_timeout_s = 5.0
        try:
            await asyncio.wait_for(self._localization_queue.join(), timeout=join_timeout_s)
            await asyncio.wait_for(self._classification_queue.join(), timeout=join_timeout_s)
            await asyncio.wait_for(self._rules_queue.join(), timeout=join_timeout_s)
        except asyncio.TimeoutError:
            self._last_error = "Timeout waiting for pipeline queues to drain during shutdown"

        try:
            for _ in self._localization_workers:
                await asyncio.wait_for(self._localization_queue.put(None), timeout=join_timeout_s)
            for _ in self._classification_workers:
                await asyncio.wait_for(self._classification_queue.put(None), timeout=join_timeout_s)
            for _ in self._rules_workers:
                await asyncio.wait_for(self._rules_queue.put(None), timeout=join_timeout_s)
        except asyncio.TimeoutError:
            self._last_error = "Timeout enqueueing worker shutdown sentinels"

        workers = [
            *self._localization_workers,
            *self._classification_workers,
            *self._rules_workers,
        ]
        if workers:
            try:
                await asyncio.wait_for(asyncio.gather(*workers, return_exceptions=True), timeout=join_timeout_s)
            except asyncio.TimeoutError:
                for task in workers:
                    task.cancel()
                try:
                    await asyncio.wait_for(asyncio.gather(*workers, return_exceptions=True), timeout=join_timeout_s)
                except asyncio.TimeoutError:
                    # If workers still do not acknowledge cancellation, continue shutdown.
                    self._last_error = (
                        "Timed out waiting for worker tasks during shutdown; "
                        "proceeded after cancellation"
                    )
                else:
                    self._last_error = "Timeout waiting for worker tasks during shutdown; cancelled remaining workers"

        self._localization_workers.clear()
        self._classification_workers.clear()
        self._rules_workers.clear()
        self._stopping = False

    # ------------------------------------------------------------------
    # Ingest — delegates to IngestProcessor
    # ------------------------------------------------------------------

    async def ingest(self, request: IngestFrameRequest) -> IngestFrameResponse:
        return await self._ingest_request(request, decoded_audio=None)

    async def ingest_decoded(self, request: IngestFrameRequest, decoded_audio: np.ndarray) -> IngestFrameResponse:
        return await self._ingest_request(request, decoded_audio=decoded_audio)

    async def refresh_live_node(
        self,
        node: NodeSpec,
        *,
        last_seen_ns: int | None = None,
    ) -> NodeSpec:
        """Refresh runtime liveness and persist only a changed registration."""
        normalized_node, position_geo = self._ingest_processor._normalize_node_spec(node)
        effective_last_seen_ns = time.time_ns() if last_seen_ns is None else last_seen_ns
        await self.registry.upsert(normalized_node, effective_last_seen_ns)
        await self._persist_node_registration_if_changed(
            normalized_node,
            last_seen_ns=effective_last_seen_ns,
            position_geo=position_geo,
        )
        return normalized_node

    async def record_live_audio_summary(self, *, node_id: str, summary: dict[str, Any]) -> None:
        await self._live_ingest_state.record_audio_summary(node_id=node_id, summary=summary)

    async def live_audio_summary(self, node_id: str) -> dict[str, Any] | None:
        return await self._live_ingest_state.audio_summary(node_id)

    async def ingest_environment_sample(self, *, node_id: str, sample: EnvironmentSampleIn) -> bool:
        """Update live environment state and persist its bounded history."""
        if not sample.has_any_measurement():
            return False
        timestamp_ns = sample.timestamp_ns or time.time_ns()
        metadata = {"source": sample.source, **sample.metadata} if sample.source else dict(sample.metadata)
        ingest_fn = getattr(self.environment_provider, "ingest_sample", None)
        if callable(ingest_fn):
            ingest_fn(
                node_id=node_id,
                timestamp_ns=timestamp_ns,
                temperature_c=sample.temperature_c,
                humidity_fraction=sample.humidity_fraction,
                pressure_pa=sample.pressure_pa,
                wind_speed_mps=sample.wind_speed_mps,
                wind_dir_deg=sample.wind_dir_deg,
                solar_lux=sample.solar_lux,
                location_m=None,
                metadata=metadata,
            )
        if not await self._live_ingest_state.should_persist_environment_sample(
            node_id=node_id,
            timestamp_ns=timestamp_ns,
        ):
            return False
        await self.storage.insert_environment(
            node_id=node_id,
            timestamp_ns=timestamp_ns,
            temperature_c=sample.temperature_c,
            pressure_pa=sample.pressure_pa,
            humidity_fraction=sample.humidity_fraction,
            wind_speed_mps=sample.wind_speed_mps,
            wind_dir_deg=sample.wind_dir_deg,
            solar_lux=sample.solar_lux,
            metadata=metadata,
        )
        return True

    async def _persist_node_registration_if_changed(
        self,
        node: NodeSpec,
        *,
        last_seen_ns: int,
        position_geo: GeoPoint | None,
    ) -> None:
        if not await self._live_ingest_state.should_persist_node_registration(node):
            return
        await self.storage.upsert_node(
            spec=node,
            last_seen_ns=last_seen_ns,
            position_geo=position_geo,
        )

    async def ingest_localized_render(self, payload: LocalizedClassifierRenderRequest) -> None:
        if self._stopping:
            raise ValueError("Fusion node is stopping")

        server_received_ns = time.time_ns()

        normalized_audio = np.asarray(payload.decoded_audio, dtype=np.float32)
        if normalized_audio.ndim != 1 or normalized_audio.size == 0:
            raise ValueError("Localized render audio must be a non-empty mono signal")
        if payload.sample_rate_hz <= 0:
            raise ValueError("Localized render sample_rate_hz must be positive")

        metadata = dict(payload.node.metadata)
        metadata["time_quality"] = payload.time_quality.value
        node = payload.node.model_copy(update={"metadata": metadata})
        if node.position_m is None and node.position_geo is not None:
            node = node.model_copy(
                update={"position_m": self.coordinate_frame.geo_to_local(node.position_geo)}
            )

        # A classifier render may retain the DSP localization payload for
        # diagnostics even after the sidecar deliberately selected an omni render
        # (for example, low-confidence SRP-PHAT). That payload is not a reportable
        # position. Treat the render contract as authoritative so an ill-conditioned
        # solve cannot turn an omni detection into a far-away localized track.
        reporting_modality = payload.reporting_modality
        if payload.render_kind == "birdnet_omni_fallback" or payload.fallback_reason is not None:
            reporting_modality = "omni"
        if reporting_modality == "omni" and node.position_m is None:
            raise ValueError("Omni classifier render requires node.position_m")

        # Localized classifier renders are derived products, not raw sensor frames.
        # Keep event-time semantics for localization/classification, but update
        # node heartbeat freshness from server receipt time so online/offline
        # status reflects current connectivity rather than source clock skew.
        runtime = await self.registry.upsert(node, server_received_ns)
        position_geo = node.position_geo
        if position_geo is None and node.position_m is not None:
            position_geo = self.coordinate_frame.local_to_geo(node.position_m)
        await self._persist_node_registration_if_changed(
            node,
            last_seen_ns=server_received_ns,
            position_geo=position_geo,
        )
        sensor_descriptors = await self.registry.sensors_for_node(node.id)
        render_duration_ns = int(
            round(normalized_audio.size * (1_000_000_000.0 / float(payload.sample_rate_hz)))
        )
        render_start_time_ns = server_received_ns - max(0, render_duration_ns)
        for descriptor in sensor_descriptors:
            await self.buffer.append(
                sensor_id=descriptor.sensor_id,
                sample_rate_hz=payload.sample_rate_hz,
                start_time_ns=render_start_time_ns,
                samples=normalized_audio,
            )
        selected_positions = {
            descriptor.sensor_id: descriptor.position_m for descriptor in sensor_descriptors
        }
        selected_sensor_ids = list(runtime.sensor_ids)
        if not selected_sensor_ids:
            raise ValueError(f"Node {node.id!r} did not register any sensors")

        reference_sensor = selected_sensor_ids[0]

        # Resolve the single-node localization. By default we trust the Rust
        # sidecar's own SRP-PHAT estimate; when configured for "python_cartesian"
        # we re-home the *solve* to Python using the sidecar's pairwise TDOAs +
        # bearing so this path shares the multi-node Cartesian estimator. Either
        # way the canonical range mode + observability/confidence haircut apply.
        localization_position_m = payload.localization_position_m
        localization_confidence = payload.localization_confidence
        localization_gdop = payload.localization_gdop
        localization_position_covariance_m2 = payload.localization_position_covariance_m2
        localization_range_observability = payload.localization_range_observability
        localization_residual_rms_seconds = payload.localization_residual_rms_seconds
        raw_range_projection_mode = payload.localization_range_projection_mode
        localization_method = payload.localization_method

        if (
            self.settings.localization_single_node_solver == "python_cartesian"
            and reporting_modality == "localized"
            and payload.localization_pair_tdoas
            and payload.localization_sound_speed_mps is not None
        ):
            # Amplitude/SNR-informed range prior (Phase 1c): when enabled and the
            # sidecar reported a received level, derive the projection distance from
            # inverse-square spreading instead of the fixed config default. Only
            # affects unobservable-range projection (asymptotic / bearing cone apex);
            # a range_refined solve ignores far_field_initial_range_m entirely.
            far_field_range_m = self.settings.localization_far_field_default_range_m
            far_field_prior_radial_std_m: float | None = None
            cfg = self.localization_config
            if (
                cfg.localization_amplitude_range_prior_enabled
                and payload.localization_received_level_dbfs is not None
            ):
                prior_range_m, prior_clamped = amplitude_range_prior_m(
                    payload.localization_received_level_dbfs,
                    reference_source_level_db=cfg.localization_amplitude_reference_level_db,
                    min_range_m=cfg.localization_amplitude_prior_min_range_m,
                    max_range_m=cfg.localization_amplitude_prior_max_range_m,
                )
                far_field_range_m = prior_range_m
                far_field_prior_radial_std_m = (
                    cfg.localization_amplitude_prior_std_factor * prior_range_m
                )
                self._metrics.localization_amplitude_prior_used_count += 1
                if prior_clamped:
                    self._metrics.localization_amplitude_prior_clamped_count += 1
            solved = solve_localization_from_rust_tdoas(
                sensor_positions=selected_positions,
                ordered_sensor_ids=selected_sensor_ids,
                pair_tdoas=payload.localization_pair_tdoas,
                steering_direction=payload.localization_steering_direction,
                sound_speed_mps=payload.localization_sound_speed_mps,
                sample_rate_hz=payload.sample_rate_hz,
                interpolation_factor=self.settings.gcc_phat_interp_factor,
                far_field_default_range_m=far_field_range_m,
                far_field_prior_radial_std_m=far_field_prior_radial_std_m,
                bearing_strength=cfg.localization_node_bearing_strength,
            )
            if solved is not None:
                localization_position_m = solved.position_m
                localization_confidence = solved.confidence
                localization_gdop = solved.gdop
                localization_position_covariance_m2 = solved.position_covariance_m2
                localization_range_observability = solved.range_observability
                localization_residual_rms_seconds = solved.residual_rms_seconds
                raw_range_projection_mode = solved.range_projection_mode
                localization_method = "python_cartesian_rust_tdoa"
                self._metrics.localization_single_node_python_solved_count += 1
            else:
                self._metrics.localization_single_node_python_fallback_count += 1

        rust_contributing_centroid_m = (
            np.mean(
                np.vstack(
                    [np.asarray(selected_positions[sid], dtype=np.float64) for sid in selected_positions]
                ),
                axis=0,
            )
            if selected_positions
            else None
        )
        localization_rejected_out_of_range = (
            reporting_modality == "localized"
            and not self._localization_position_is_reportable(
                localization_position_m,
                rust_contributing_centroid_m,
            )
        )
        if localization_rejected_out_of_range:
            # This ingress path receives a precomputed position and used to bypass
            # _build_localization_branch(), where the normal solver path is gated.
            # Preserve the classification as an omni observation at its node instead
            # of allowing a bogus geographic position into a localized track.
            self._metrics.localization_rejected_out_of_range_count += 1
            self._record_silent_drop(stage="localization", reason="position_out_of_range")
            reporting_modality = "omni"

        # Frequency-dependent lateral covariance scaling for the single-node path.
        # Angular resolution degrades when the dominant signal frequency falls well
        # below the array's spatial-aliasing cutoff, so widen the covariance
        # perpendicular to the bearing. The sidecar reports the dominant frequency
        # (absent on un-rebuilt sidecars → skip); the alias cutoff is derived from
        # node geometry. Mirrors the multi-node dispatcher's Item B scaling.
        if (
            localization_position_covariance_m2 is not None
            and payload.localization_dominant_frequency_hz is not None
            and payload.localization_sound_speed_mps is not None
        ):
            positions = np.vstack(
                [
                    np.asarray(selected_positions[sid], dtype=np.float64)
                    for sid in sorted(selected_positions)
                ]
            )
            alias_cutoff_hz = alias_cutoff_from_positions(
                positions, c_sound=payload.localization_sound_speed_mps
            )
            centroid_m = np.mean(positions, axis=0)
            bearing_vec = np.asarray(localization_position_m, dtype=np.float64) - centroid_m
            scaled_cov = apply_frequency_covariance_scaling(
                covariance_m2=np.asarray(localization_position_covariance_m2, dtype=np.float64),
                bearing_unit_vec=bearing_vec,
                dominant_frequency_hz=payload.localization_dominant_frequency_hz,
                alias_cutoff_hz=alias_cutoff_hz,
            )
            localization_position_covariance_m2 = covariance_to_nested_list(scaled_cov)

        # Canonicalize the range mode and apply the path-agnostic haircut so a
        # far-field single-node estimate cannot pass through over-confident.
        localization_range_projection_mode = normalize_range_mode(raw_range_projection_mode)
        self._record_range_projection_metrics(raw_range_projection_mode)
        localization_confidence, localization_range_observability = apply_unobservable_range_haircut(
            mode=localization_range_projection_mode,
            confidence=localization_confidence,
            range_observability=localization_range_observability,
        )
        localization_position_covariance_m2 = self._clamp_localization_covariance(
            localization_position_covariance_m2,
            position_m=localization_position_m,
            contributing_centroid_m=rust_contributing_centroid_m,
        )

        rust_audio_quality = (
            {reference_sensor: payload.audio_quality}
            if payload.audio_quality is not None
            else {}
        )
        capability_tier = (
            "full_3d"
            if reporting_modality == "localized"
            and len(selected_sensor_ids) >= self.settings.min_sensors_for_3d
            else "classification_only"
        )
        candidate = EventCandidate(
            id=f"rust-{payload.manifest_id}",
            source_node_id=node.id,
            event_time_ns=payload.event_time_ns,
            sample_rate_hz=payload.sample_rate_hz,
            source_type=payload.source_type,
            time_quality=payload.time_quality,
            source_observation_ids=list(payload.source_observation_ids),
        )
        rust_extra_features: dict[str, Any] = {"rust_manifest_id": payload.manifest_id}
        if payload.render_kind is not None:
            rust_extra_features["rust_render_kind"] = payload.render_kind
        if payload.render_start_ns is not None:
            rust_extra_features["rust_render_start_ns"] = int(payload.render_start_ns)
        if payload.render_end_ns is not None:
            rust_extra_features["rust_render_end_ns"] = int(payload.render_end_ns)
        if payload.fallback_reason is not None:
            rust_extra_features["rust_fallback_reason"] = payload.fallback_reason
        if localization_rejected_out_of_range:
            rust_extra_features["localization_rejected_reason"] = "position_out_of_range"
        if reporting_modality == "omni":
            # Keep the raw sidecar localization only as provenance above; an omni
            # detection is positioned at its source node and cannot create a track.
            localization_position_m = tuple(float(value) for value in node.position_m)
            localization_confidence = self.fusion_config.fallback_localization_confidence
            localization_gdop = float("inf")
            localization_position_covariance_m2 = None
            localization_range_observability = None
            localization_residual_rms_seconds = None
            localization_range_projection_mode = None
            localization_method = "rust_classifier_render_fallback"
        rust_localization_branch = LocalizationBranch(
            localization_position_m=localization_position_m,
            localization_confidence=localization_confidence,
            localization_gdop=localization_gdop,
            reference_sensor=reference_sensor,
            reference_signal=normalized_audio,
            classification_reference_signal=normalized_audio,
            tdoa_s={},
            localization_method=localization_method,
            capability_tier=capability_tier,
            localization_position_covariance_m2=localization_position_covariance_m2,
            localization_range_observability=localization_range_observability,
            localization_residual_rms_seconds=localization_residual_rms_seconds,
            localization_range_projection_mode=localization_range_projection_mode,
        )
        if reporting_modality == "localized":
            rust_localization_branch = await self._maybe_fuse_multi_node_bearing(
                rust_localization_branch,
                node_id=node.id,
                contributing_centroid_m=rust_contributing_centroid_m,
                event_time_ns=payload.event_time_ns,
                sound_speed_mps=payload.localization_sound_speed_mps,
            )
        localized_product = LocalizedCandidate(
            candidate=candidate,
            localization_branch=rust_localization_branch,
            selected_sensor_ids=selected_sensor_ids,
            selected_windows={},
            selected_positions=selected_positions,
            classification_selected_windows={},
            omni_reference_sensor=reference_sensor,
            omni_reference_signal=normalized_audio,
            omni_position_m=localization_position_m,
            omni_classification_reference_signal=normalized_audio,
            localization_audio_quality=rust_audio_quality,
            classification_audio_quality=rust_audio_quality,
            classification_audio_quality_source="rust_classifier_render",
            environment=dict(payload.environment),
            extra_classification_features=rust_extra_features,
        )
        self._record_degraded_audio_quality_metrics(
            candidate=candidate,
            source_window_type=localized_product.classification_audio_quality_source,
            audio_quality=localized_product.classification_audio_quality,
        )
        # Rust localized renders should be classified synchronously so callers can
        # observe the resulting detection immediately after ingest_localized_render returns.
        if self._classification_chunking_policy is not None and payload.authoritative_classification is None:
            if self._should_suppress_chunked_classification(localized_product):
                self._metrics.birdnet_chunk_dispatches_suppressed += 1
                return

        if payload.authoritative_classification is not None:
            classified = await self._classification_orchestrator.adopt_authoritative_classification(
                classification=payload.authoritative_classification,
                event_time_ns=payload.event_time_ns,
                classification_signal=normalized_audio,
            )
            classified.classification.features["rust_classification_authoritative"] = True
        else:
            classified = await self._classification_orchestrator.classify_omni_only(
                reference_signal=normalized_audio,
                sample_rate_hz=payload.sample_rate_hz,
                event_time_ns=payload.event_time_ns,
            )
        classified.classification.features.update(rust_extra_features)
        await self._maybe_trigger_speech_capture(
            node_id=node.id,
            sensor_id=reference_sensor,
            event_time_ns=payload.event_time_ns,
            scores=classified.classification.scores,
        )
        detection_product = await self._assemble_reporting_branch(
            product=localized_product,
            classified=classified,
            reporting_modality=reporting_modality,
            localization_position_m=localization_position_m,
            localization_confidence=localization_confidence,
            localization_gdop=localization_gdop,
            localization_position_covariance_m2=localization_position_covariance_m2,
            localization_range_observability=localization_range_observability,
            localization_residual_rms_seconds=localization_residual_rms_seconds,
            localization_range_projection_mode=localization_range_projection_mode,
            reference_sensor=reference_sensor,
            reference_signal=normalized_audio,
            tdoa_s={},
            capability_tier=capability_tier,
            localization_method=localization_method,
        )
        if detection_product is None:
            return
        await self._process_rules_and_delivery(detection_product)

    async def ingest_continuous_scan_result(self, scan) -> None:
        """Sink for :class:`ContinuousOmniScanner`.

        Adopts the pre-computed ``omni_continuous`` classification authoritatively
        and routes it through the standard omni reporting-branch assembly so the
        site-wide omni dedupe (reporting_fusion) suppresses duplicates of a
        concurrently trigger-driven detection automatically.
        """
        if self._stopping:
            return
        node = scan.node
        if node.position_m is None:
            return

        sensor_descriptors = await self.registry.sensors_for_node(node.id)
        if not sensor_descriptors:
            return
        selected_positions = {
            descriptor.sensor_id: descriptor.position_m for descriptor in sensor_descriptors
        }
        selected_sensor_ids = [descriptor.sensor_id for descriptor in sensor_descriptors]
        reference_sensor = (
            scan.sensor_id if scan.sensor_id in selected_positions else selected_sensor_ids[0]
        )
        audio = np.asarray(scan.audio, dtype=np.float32)
        node_position_m = tuple(float(value) for value in node.position_m)

        candidate = EventCandidate(
            id=f"omni-scan-{node.id}-{scan.end_time_ns}",
            source_node_id=node.id,
            event_time_ns=scan.end_time_ns,
            sample_rate_hz=scan.sample_rate_hz,
            # source_type identifies the physical upstream producer.  A
            # continuous scan is still raw sensor audio; its scan provenance
            # is retained below in classification features instead of
            # inventing a value outside the persisted DetectionEvent contract.
            source_type="raw_sensor",
            time_quality=TimeQuality.FREE_RUNNING,
            source_observation_ids=[],
        )
        localization_branch = LocalizationBranch(
            localization_position_m=node_position_m,
            localization_confidence=self.fusion_config.fallback_localization_confidence,
            localization_gdop=float("inf"),
            localization_position_covariance_m2=None,
            localization_range_observability=None,
            localization_residual_rms_seconds=None,
            localization_range_projection_mode=None,
            reference_sensor=reference_sensor,
            reference_signal=audio,
            classification_reference_signal=audio,
            tdoa_s={},
            localization_method="omni_continuous_scan",
            capability_tier="classification_only",
        )
        product = LocalizedCandidate(
            candidate=candidate,
            localization_branch=localization_branch,
            selected_sensor_ids=selected_sensor_ids,
            selected_windows={},
            selected_positions=selected_positions,
            classification_selected_windows={},
            omni_reference_sensor=reference_sensor,
            omni_reference_signal=audio,
            omni_position_m=node_position_m,
            omni_classification_reference_signal=audio,
            localization_audio_quality={},
            classification_audio_quality={},
            classification_audio_quality_source="omni_continuous_scan",
            environment={},
            extra_classification_features={"omni_scan_rms": float(scan.rms)},
        )
        classified = await self._classification_orchestrator.adopt_authoritative_classification(
            classification=scan.classification,
            event_time_ns=scan.end_time_ns,
            classification_signal=audio,
            classification_path="omni_continuous",
        )
        classified.classification.features["omni_continuous_scan"] = True
        await self._maybe_trigger_speech_capture(
            node_id=node.id,
            sensor_id=reference_sensor,
            event_time_ns=scan.end_time_ns,
            scores=classified.classification.scores,
        )
        detection_product = await self._assemble_reporting_branch(
            product=product,
            classified=classified,
            reporting_modality="omni",
            localization_position_m=node_position_m,
            localization_confidence=self.fusion_config.fallback_localization_confidence,
            localization_gdop=float("inf"),
            localization_position_covariance_m2=None,
            localization_range_observability=None,
            localization_residual_rms_seconds=None,
            localization_range_projection_mode=None,
            reference_sensor=reference_sensor,
            reference_signal=audio,
            tdoa_s={},
            capability_tier="classification_only",
            localization_method="omni_continuous_scan",
        )
        if detection_product is None:
            return
        await self._process_rules_and_delivery(detection_product)

    async def _ingest_request(
        self,
        request: IngestFrameRequest,
        *,
        decoded_audio: np.ndarray | None,
    ) -> IngestFrameResponse:
        if self._stopping:
            raise ValueError("Fusion node is stopping")
        self._metrics.ingest_requests += 1
        try:
            result = await self._ingest_processor.process_frame(
                request,
                storage_batch_ctx=self._storage_batch,
                decoded_audio=decoded_audio,
            )
        except ValueError:
            self._metrics.frames_rejected += 1
            raise

        self._record_ingest_result_metrics(request, result)
        await self._queue_trigger_candidate(request, result)

        return result.response

    # ------------------------------------------------------------------
    # Status & housekeeping
    # ------------------------------------------------------------------

    async def status(self) -> dict[str, Any]:
        nodes = await self.registry.list_nodes()
        now_ns = time.time_ns()
        realtime = self._realtime_tracker.snapshot(now_ns=now_ns)
        buffer_state = await self.buffer.snapshot_state()
        health = self._compute_health_snapshot(now_ns=now_ns)
        # Phase 3: cross-node track fusion telemetry (counter + live gauge).
        if hasattr(self.tracker, "multi_node_association_count"):
            self._metrics.track_multi_node_association_count = (
                self.tracker.multi_node_association_count()
            )
        if hasattr(self.tracker, "multi_node_active_count"):
            self._metrics.tracks_multi_node_active = self.tracker.multi_node_active_count()
        return {
            "started": self._started,
            "queue": {
                "localization_depth": self._localization_queue.qsize(),
                "classification_depth": self._classification_queue.qsize(),
                "rules_depth": self._rules_queue.qsize(),
                "localization_max": self._localization_queue.maxsize,
                "classification_max": self._classification_queue.maxsize,
                "rules_max": self._rules_queue.maxsize,
            },
            "workers": {
                "configured": max(1, self.fusion_config.worker_count),
                "localization_running": sum(1 for task in self._localization_workers if not task.done()),
                "classification_running": sum(1 for task in self._classification_workers if not task.done()),
                "rules_running": sum(1 for task in self._rules_workers if not task.done()),
            },
            "last_trigger_ns": self._ingest_processor.last_trigger_ns,
            "last_error": self._last_error,
            "registered_nodes": len(nodes),
            "metrics": asdict(self._metrics),
            "realtime": realtime,
            "offline_replay_mode": self.fusion_config.offline_replay_mode,
            "drop_on_backpressure": self.fusion_config.drop_on_backpressure,
            "buffer_state": buffer_state,
            "health": health,
            "omni_scanner": self._omni_scanner.stats() if self._omni_scanner is not None else {"enabled": bool(self.settings.omni_scan_enabled), "running": False},
        }

    def observe_firmware_runner_stats(
        self,
        node_id: str,
        timing_diagnostics: dict,
    ) -> dict | None:
        """Return edge-triggered degraded and recovered firmware transport states."""
        overflows = int(timing_diagnostics.get("runner_queue_overflows") or 0)
        dropped = int(timing_diagnostics.get("runner_frames_dropped") or 0)
        now_ns = time.monotonic_ns()
        last = self._last_firmware_runner_counters.get(node_id)
        self._last_firmware_runner_counters[node_id] = (overflows, dropped)
        if last is None:
            return None
        overflow_delta = max(0, overflows - last[0])
        dropped_delta = max(0, dropped - last[1])
        if overflow_delta > 0 or dropped_delta > 0:
            self._firmware_transport_degraded_since_ns[node_id] = now_ns
            return {
                "state": "degraded",
                "new_queue_overflows": overflow_delta,
                "new_frames_dropped": dropped_delta,
                "total_queue_overflows": overflows,
                "total_frames_dropped": dropped,
            }
        degraded_since_ns = self._firmware_transport_degraded_since_ns.get(node_id)
        if degraded_since_ns is not None and now_ns - degraded_since_ns >= self._firmware_transport_recovery_ns:
            del self._firmware_transport_degraded_since_ns[node_id]
            return {
                "state": "recovered",
                "total_queue_overflows": overflows,
                "total_frames_dropped": dropped,
            }
        return None

    def _compute_health_snapshot(self, *, now_ns: int) -> dict[str, Any]:
        """Derive watchdog signals from FusionMetrics emission timestamps.

        `active_drought` flags the silent-stall state that motivated the
        observability work: triggers continue to fire while emissions stall.
        """
        last_emit_ns = self._metrics.last_detection_emission_ns
        last_trig_ns = self._metrics.last_trigger_enqueue_ns
        seconds_since_emit = (now_ns - last_emit_ns) / 1e9 if last_emit_ns else None
        seconds_since_trig = (now_ns - last_trig_ns) / 1e9 if last_trig_ns else None
        active_drought = (
            last_trig_ns > 0
            and (last_emit_ns == 0 or last_trig_ns > last_emit_ns + 60 * 1_000_000_000)
        )
        return {
            "seconds_since_last_emission": seconds_since_emit,
            "seconds_since_last_trigger": seconds_since_trig,
            "active_drought": active_drought,
        }

    def node_frame_metrics(self) -> dict[str, dict[str, int]]:
        """Return a snapshot of per-node frame ingest counts."""
        return dict(self._per_node_frame_metrics)

    def apply_node_audio_override(self, node_id: str, override: dict | None) -> None:
        """Apply or clear a runtime per-node DSP override on the preprocessor factory."""
        self._ingest_processor._preprocessor_factory.set_node_override(node_id, override)

    async def housekeeping_tick(self, now_ns: int) -> None:
        await self.zone_matcher.refresh_if_due(now_ns=now_ns)
        await self._refresh_taxonomy(now_ns=now_ns)
        tracks = await self.tracker.snapshot(now_ns=now_ns)
        for track in tracks:
            track.position_geo = self.coordinate_frame.local_to_geo(track.position_m)
            await self.storage.upsert_track(track)

    # ------------------------------------------------------------------
    # Worker loops — unchanged pipeline stage routing
    # ------------------------------------------------------------------

    async def _localization_worker_loop(self, worker_id: int) -> None:
        del worker_id
        timeout_s = self.classifier_config.stage_timeout_seconds
        while True:
            candidate = await self._localization_queue.get()
            if candidate is None:
                self._localization_queue.task_done()
                return
            if candidate.enqueued_ns:
                queue_wait_ms = max(0.0, (time.time_ns() - candidate.enqueued_ns) / 1_000_000.0)
                self._metrics.ingest_queue_wait_total_ms += queue_wait_ms
                self._metrics.ingest_queue_wait_max_ms = max(
                    self._metrics.ingest_queue_wait_max_ms,
                    queue_wait_ms,
                )
                self._metrics.ingest_queue_wait_count += 1
            self._realtime_tracker.mark_started(stage_name="localization", item_id=candidate.id)
            self._metrics.localization_stage_in += 1
            try:
                localization_started_ns = time.perf_counter_ns()
                product = await asyncio.wait_for(
                    self._localize_candidate(candidate),
                    timeout=timeout_s,
                )
                elapsed_ms = (time.perf_counter_ns() - localization_started_ns) / 1_000_000.0
                self._metrics.localization_stage_total_time_ms += elapsed_ms
                self._metrics.localization_stage_max_time_ms = max(
                    self._metrics.localization_stage_max_time_ms,
                    elapsed_ms,
                )
                self._metrics.ingest_processing_total_ms += elapsed_ms
                self._metrics.ingest_processing_max_ms = max(
                    self._metrics.ingest_processing_max_ms,
                    elapsed_ms,
                )
                self._metrics.ingest_processing_count += 1
                if product is not None:
                    if await self._enqueue_stage(self._classification_queue, product):
                        self._metrics.localization_stage_out += 1
            except asyncio.TimeoutError:
                self._metrics.stage_timeout_count += 1
                self._last_error = f"Localization stage timed out after {timeout_s:.3f}s"
                logger.warning(
                    "Localization stage timed out",
                    extra={
                        "candidate_id": candidate.id,
                        "source_node_id": candidate.source_node_id,
                        "timeout_seconds": timeout_s,
                    },
                )
            except Exception as exc:  # pragma: no cover - resilience path
                self._metrics.localization_failures += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._record_worker_exception(stage="localization", exc=exc, candidate_id=candidate.id)
            finally:
                self._realtime_tracker.mark_finished(stage_name="localization", item_id=candidate.id)
                self._localization_queue.task_done()

    async def _classification_worker_loop(self, worker_id: int) -> None:
        del worker_id
        timeout_s = self.classifier_config.stage_timeout_seconds
        while True:
            product = await self._classification_queue.get()
            if product is None:
                self._classification_queue.task_done()
                return
            self._realtime_tracker.mark_started(
                stage_name="classification",
                item_id=product.candidate.id,
            )
            self._metrics.classification_stage_in += 1
            try:
                if self._should_suppress_chunked_classification(product):
                    self._metrics.birdnet_chunk_dispatches_suppressed += 1
                    self._record_silent_drop(
                        stage="classification",
                        reason="chunk_suppressed",
                        candidate_id=product.candidate.id,
                        event_time_ns=product.candidate.event_time_ns,
                        source_node_id=product.candidate.source_node_id,
                    )
                    continue
                detection_products = await asyncio.wait_for(
                    self._classify_and_assemble(product),
                    timeout=timeout_s,
                )
                classification_failed = self._detection_products_had_backend_failure(detection_products)
                self._record_chunk_dispatch_outcome(
                    product=product,
                    detection_products=detection_products,
                    classification_failed=classification_failed,
                )
                if classification_failed:
                    self._metrics.classification_failures += 1
                    self._record_silent_drop(
                        stage="classification",
                        reason="backend_failure",
                        candidate_id=product.candidate.id,
                        event_time_ns=product.candidate.event_time_ns,
                        source_node_id=product.candidate.source_node_id,
                    )
                elif not detection_products:
                    self._record_silent_drop(
                        stage="classification",
                        reason="empty_classification",
                        candidate_id=product.candidate.id,
                        event_time_ns=product.candidate.event_time_ns,
                        source_node_id=product.candidate.source_node_id,
                    )
                else:
                    for detection_product in detection_products:
                        if await self._enqueue_stage(self._rules_queue, detection_product):
                            self._metrics.classification_stage_out += 1
            except asyncio.TimeoutError:
                self._record_chunk_dispatch_outcome(
                    product=product,
                    detection_products=[],
                    classification_failed=True,
                )
                self._metrics.stage_timeout_count += 1
                self._last_error = f"Classification stage timed out after {timeout_s:.3f}s"
                logger.warning(
                    "Classification stage timed out",
                    extra={
                        "candidate_id": product.candidate.id,
                        "source_node_id": product.candidate.source_node_id,
                        "timeout_seconds": timeout_s,
                    },
                )
            except Exception as exc:
                self._record_chunk_dispatch_outcome(
                    product=product,
                    detection_products=[],
                    classification_failed=True,
                )
                self._metrics.classification_failures += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._record_worker_exception(stage="classification", exc=exc, candidate_id=product.candidate.id)
            finally:
                self._realtime_tracker.mark_finished(
                    stage_name="classification",
                    item_id=product.candidate.id,
                )
                self._classification_queue.task_done()

    async def _rules_worker_loop(self, worker_id: int) -> None:
        del worker_id
        while True:
            product = await self._rules_queue.get()
            if product is None:
                self._rules_queue.task_done()
                return
            self._realtime_tracker.mark_started(
                stage_name="rules",
                item_id=product.pipeline_item_id,
            )
            self._metrics.rules_stage_in += 1
            try:
                await self._process_rules_and_delivery(product)
                self._metrics.rules_stage_out += 1
            except Exception as exc:  # pragma: no cover - resilience path
                self._metrics.rules_failures += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._record_worker_exception(stage="rules", exc=exc)
            finally:
                self._realtime_tracker.mark_finished(
                    stage_name="rules",
                    item_id=product.pipeline_item_id,
                )
                self._rules_queue.task_done()

    # ------------------------------------------------------------------
    # Localization stage — sensor selection, environment, localizer call
    # ------------------------------------------------------------------

    async def _await_window_coverage(
        self,
        *,
        sensor_ids: list[str],
        center_time_ns: int,
        window_seconds: float,
        timeout_s: float,
        poll_interval_s: float = 0.005,
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Yield until at least min_sensors_for_2d sensors have audio covering
        [center − window/2, center + window/2], or the deadline expires.

        Returns (ready, last_snapshot). Replaces the legacy fixed 40 ms
        grace-sleep retry which dropped 96 % of triggers under typical
        per-sensor ingest jitter (see plan: valiant-launching-whale).
        """
        half_window_ns = int(window_seconds / 2 * 1_000_000_000)
        target_end_ns = center_time_ns + half_window_ns
        target_start_ns = center_time_ns - half_window_ns
        # Wait until we have coverage for the smaller of (sensors registered
        # on this node) and min_sensors_for_2d. Capping at the registered
        # count preserves the single-sensor / classification-only path that
        # the legacy code allowed through with whatever window was present.
        min_needed = max(
            1,
            min(len(sensor_ids), int(self.localization_config.min_sensors_for_2d)),
        )
        deadline = time.monotonic() + max(0.0, timeout_s)
        snapshot: list[dict[str, Any]] = []
        while True:
            snapshot = await self.buffer.snapshot_state(sensor_ids=sensor_ids)
            covered = 0
            min_start_ns: int | None = None
            for state in snapshot:
                start_ns = state.get("start_time_ns")
                end_ns = state.get("end_time_ns")
                if state.get("present") and start_ns is not None and end_ns is not None:
                    if start_ns <= target_start_ns and end_ns >= target_end_ns:
                        covered += 1
                    if min_start_ns is None or start_ns < min_start_ns:
                        min_start_ns = start_ns
            if covered >= min_needed:
                return True, snapshot
            # If the event's trailing edge sits before the oldest buffered
            # sample, waiting can't recover it — exit immediately so the
            # caller can drop with `event_too_old` instead of stalling.
            if min_start_ns is not None and target_end_ns < min_start_ns:
                return False, snapshot
            if time.monotonic() >= deadline:
                return False, snapshot
            await asyncio.sleep(poll_interval_s)

    async def _localize_candidate(self, candidate: EventCandidate) -> LocalizedCandidate | None:
        sensor_positions = await self.registry.sensor_positions()
        sensor_grades = await self.registry.sensor_sync_grades()
        sensor_weights: dict[str, float] | None = {
            sensor_id: grade.weight()
            for sensor_id, grade in sensor_grades.items()
            if sensor_id in sensor_positions
        }

        if (
            self.localization_config.cluster_aware_localization
            and self.cluster_registry is not None
        ):
            cluster = await self.cluster_registry.cluster_for_node(candidate.source_node_id)
            if cluster is not None:
                cluster_positions = await self.cluster_registry.cluster_sensor_positions(
                    cluster.id, self.registry
                )
                if cluster_positions:
                    sensor_positions = cluster_positions
                    sensor_weights = await self.cluster_registry.cluster_sensor_weights(
                        cluster.id, self.registry
                    )

        sensor_ids = sorted(sensor_positions.keys())
        if not sensor_ids:
            self._record_silent_drop(
                stage="localization",
                reason="no_sensors",
                candidate_id=candidate.id,
                event_time_ns=candidate.event_time_ns,
                source_node_id=candidate.source_node_id,
            )
            return None

        window_seconds = self.localization_config.localization_window_seconds
        timeout_s = self.localization_config.localization_buffer_wait_max_seconds
        min_localization_sensors = max(2, int(self.localization_config.min_sensors_for_2d))
        # Wait for the per-node sensor buffers to actually cover the requested
        # window before asking for it. The legacy fixed 40 ms grace-sleep
        # retry was insufficient under steady-state ingest jitter and dropped
        # ~96 % of triggers in production with no diagnosable signal.
        ready, coverage_snapshot = await self._await_window_coverage(
            sensor_ids=sensor_ids,
            center_time_ns=candidate.event_time_ns,
            window_seconds=window_seconds,
            timeout_s=timeout_s,
        )
        if not ready:
            half_window_ns = int(window_seconds / 2 * 1_000_000_000)
            target_start_ns = candidate.event_time_ns - half_window_ns
            present_starts = [
                state["start_time_ns"]
                for state in coverage_snapshot
                if state.get("present") and state.get("start_time_ns") is not None
            ]
            min_start_ns = min(present_starts) if present_starts else None
            # Two distinct failure modes share the same drop path:
            #   * `event_too_old` — buffer pruned past target; waiting can't help.
            #   * `buffer_lag_timeout` — sensors never caught up within timeout_s.
            reason = (
                "event_too_old"
                if (min_start_ns is not None and target_start_ns < min_start_ns)
                else "buffer_lag_timeout"
            )
            self._record_silent_drop(
                stage="localization",
                reason=reason,
                candidate_id=candidate.id,
                event_time_ns=candidate.event_time_ns,
                source_node_id=candidate.source_node_id,
                sample_rate_hz=candidate.sample_rate_hz,
                sensors_requested=len(sensor_ids),
                timeout_s=timeout_s,
                buffer_snapshot=coverage_snapshot,
            )
            return None

        windows = await self.buffer.get_synchronized_window(
            sensor_ids=sensor_ids,
            center_time_ns=candidate.event_time_ns,
            window_seconds=window_seconds,
            sample_rate_hz=candidate.sample_rate_hz,
        )
        if not windows:
            # Rare residual race: coverage was present at the snapshot but the
            # synchronized fetch returned empty (e.g. concurrent rate-mismatch
            # wipe between snapshot and fetch). Keep the original drop reason
            # so an unexpected regression to "no_window" remains visible.
            buffer_snapshot = await self.buffer.snapshot_state(sensor_ids=sensor_ids)
            self._record_silent_drop(
                stage="localization",
                reason="no_window",
                candidate_id=candidate.id,
                event_time_ns=candidate.event_time_ns,
                source_node_id=candidate.source_node_id,
                sample_rate_hz=candidate.sample_rate_hz,
                sensors_requested=len(sensor_ids),
                buffer_snapshot=buffer_snapshot,
            )
            return None

        energies = {sensor_id: rms(sig) for sensor_id, sig in windows.items()}
        threshold = self.localization_config.trigger_rms * self.fusion_config.sensor_energy_threshold_multiplier
        selected_ids = [sid for sid, energy in energies.items() if energy > threshold]
        # If thresholding under-selects, keep the strongest sensors needed for a localization attempt.
        if len(selected_ids) < min_localization_sensors and len(energies) >= min_localization_sensors:
            ranked = sorted(energies.items(), key=lambda item: item[1], reverse=True)
            selected_ids = [sensor_id for sensor_id, _ in ranked[:min_localization_sensors]]
        if len(selected_ids) < 1:
            self._record_silent_drop(
                stage="localization",
                reason="low_energy",
                candidate_id=candidate.id,
                event_time_ns=candidate.event_time_ns,
                source_node_id=candidate.source_node_id,
                threshold=threshold,
                max_sensor_energy=max(energies.values()) if energies else 0.0,
                sensors_evaluated=len(energies),
            )
            return None

        tier = self.degradation_model.tier_for_sensor_count(len(selected_ids))
        selected_windows = {sensor_id: windows[sensor_id] for sensor_id in selected_ids}
        selected_positions = {sensor_id: sensor_positions[sensor_id] for sensor_id in selected_ids}
        selected_sensor_node_ids = await self.registry.sensor_node_ids(selected_ids)
        selected_sensor_gain_offsets_db = await self.registry.sensor_gain_offsets_db(
            selected_ids
        )
        # Half-space (D7) is a single coplanar node's own up/down constraint; it
        # only applies unambiguously when every selected sensor belongs to one node.
        selected_node_ids = set(selected_sensor_node_ids.values())
        selected_half_space = (
            await self.registry.node_half_space(next(iter(selected_node_ids)))
            if len(selected_node_ids) == 1
            else None
        )
        localization_audio_quality = await self.buffer.get_synchronized_window_coverage_stats(
            sensor_ids=selected_ids,
            center_time_ns=candidate.event_time_ns,
            window_seconds=self.localization_config.localization_window_seconds,
            sample_rate_hz=candidate.sample_rate_hz,
        )
        classification_windows = await self._classification_windows_for_event(
            candidate=candidate,
            selected_sensor_ids=selected_ids,
            fallback_windows=selected_windows,
        )
        classification_audio_quality, classification_audio_quality_source = await self._classification_quality_for_event(
            candidate=candidate,
            selected_sensor_ids=selected_ids,
            localization_audio_quality=localization_audio_quality,
            classification_windows=classification_windows,
        )
        self._record_degraded_audio_quality_metrics(
            candidate=candidate,
            source_window_type=classification_audio_quality_source,
            audio_quality=classification_audio_quality,
        )

        environment_location = self._sensor_centroid(selected_positions)
        conditions = self.environment_provider.get_conditions(environment_location)
        environment_summary = {
            "temperature_c": conditions.temperature_c,
            "humidity_fraction": conditions.humidity_fraction,
            "pressure_pa": conditions.pressure_pa,
            "wind_speed_mps": conditions.wind_speed_mps,
            "wind_dir_deg": conditions.wind_dir_deg,
            "metadata": dict(conditions.metadata or {}),
        }
        fallback_candidate = self._build_reference_sensor_candidate(
            candidate=candidate,
            energies=energies,
            sensor_positions=sensor_positions,
            selected_ids=selected_ids,
            selected_windows=selected_windows,
            classification_windows=classification_windows,
            environment_summary=environment_summary,
        )
        if self.localization_config.skip_localization_for_classification:
            return fallback_candidate

        localization_windows = selected_windows
        if self.localization_preprocessor is not None:
            localization_windows = {
                sensor_id: self.localization_preprocessor.process(window, candidate.sample_rate_hz)
                for sensor_id, window in selected_windows.items()
            }

        localization_kwargs: dict[str, object] = {}
        localize_parameters = inspect.signature(self.localizer.localize).parameters
        if sensor_weights is not None and "sensor_weights" in localize_parameters:
            localization_kwargs["sensor_weights"] = {
                sensor_id: sensor_weights.get(sensor_id, 1.0)
                for sensor_id in selected_ids
            }
        if "sensor_node_ids" in localize_parameters:
            localization_kwargs["sensor_node_ids"] = selected_sensor_node_ids
        if "sensor_gain_offsets_db" in localize_parameters:
            localization_kwargs["sensor_gain_offsets_db"] = (
                selected_sensor_gain_offsets_db
            )
        if selected_half_space is not None and "half_space" in localize_parameters:
            localization_kwargs["half_space"] = selected_half_space
        localization_2d_kwargs: dict[str, object] = {}
        if hasattr(self.localizer, "localize_2d"):
            localize_2d_parameters = inspect.signature(self.localizer.localize_2d).parameters
            if sensor_weights is not None and "sensor_weights" in localize_2d_parameters:
                localization_2d_kwargs["sensor_weights"] = {
                    sensor_id: sensor_weights.get(sensor_id, 1.0)
                    for sensor_id in selected_ids
                }
        localization_branch: LocalizationBranch | None = None
        contributing_centroid_m = (
            np.mean(
                np.vstack(
                    [np.asarray(pos, dtype=np.float64) for pos in selected_positions.values()]
                ),
                axis=0,
            )
            if selected_positions
            else None
        )
        if tier == "full_3d":
            self._metrics.localization_tier_full_3d_count += 1
            try:
                localization = await asyncio.to_thread(
                    self.localizer.localize,
                    selected_positions,
                    localization_windows,
                    candidate.sample_rate_hz,
                    conditions.temperature_c,
                    conditions.humidity_fraction,
                    **localization_kwargs,
                )
                localization_branch = self._build_localization_branch(
                    localization=localization,
                    selected_windows=selected_windows,
                    classification_windows=classification_windows,
                    capability_tier=tier,
                    contributing_centroid_m=contributing_centroid_m,
                )
            except LocalizationError as exc:
                self._metrics.localization_failures += 1
                if "did not converge" in str(exc):
                    self._metrics.localization_solver_unconverged_count += 1
        elif tier == "2d" and hasattr(self.localizer, "localize_2d"):
            self._metrics.localization_tier_2d_count += 1
            if getattr(self.localizer, "default_algorithm", "gcc_phat") != "gcc_phat":
                self._metrics.localization_config_bypassed_count += 1
            try:
                mean_z = float(np.mean([pos[2] for pos in selected_positions.values()]))
                localization = await asyncio.to_thread(
                    self.localizer.localize_2d,
                    selected_positions,
                    localization_windows,
                    candidate.sample_rate_hz,
                    conditions.temperature_c,
                    conditions.humidity_fraction,
                    mean_z,
                    **localization_2d_kwargs,
                )
                localization_branch = self._build_localization_branch(
                    localization=localization,
                    selected_windows=selected_windows,
                    classification_windows=classification_windows,
                    capability_tier=tier,
                    contributing_centroid_m=contributing_centroid_m,
                )
            except LocalizationError:
                self._metrics.localization_failures += 1

        if localization_branch is not None:
            localization_branch = await self._maybe_fuse_multi_node_bearing(
                localization_branch,
                node_id=candidate.source_node_id,
                contributing_centroid_m=contributing_centroid_m,
                event_time_ns=candidate.event_time_ns,
                sound_speed_mps=None,
            )

        return LocalizedCandidate(
            candidate=candidate,
            localization_branch=localization_branch,
            selected_sensor_ids=selected_ids,
            selected_windows=selected_windows,
            selected_positions=selected_positions,
            classification_selected_windows=classification_windows,
            omni_reference_sensor=fallback_candidate.omni_reference_sensor,
            omni_reference_signal=fallback_candidate.omni_reference_signal,
            omni_position_m=fallback_candidate.omni_position_m,
            omni_classification_reference_signal=fallback_candidate.omni_classification_reference_signal,
            localization_audio_quality=localization_audio_quality,
            classification_audio_quality=classification_audio_quality,
            classification_audio_quality_source=classification_audio_quality_source,
            environment=environment_summary,
        )

    # ------------------------------------------------------------------
    # Classification + assembly — delegates to orchestrator and assembler
    # ------------------------------------------------------------------

    async def _classify_and_assemble(self, product: LocalizedCandidate) -> list[DetectionProduct]:
        """Classify a localized candidate's beamformed render.

        Raw omni inference is intentionally reserved for ContinuousOmniScanner.
        This keeps triggered work spatially meaningful and avoids re-running
        BirdNET/YAMNet over the same ambient window.
        """
        detection_products: list[DetectionProduct] = []
        extra_features = product.extra_classification_features

        if product.localization_branch is not None:
            localized = await self._classification_orchestrator.classify_beamformed_only(
                sample_rate_hz=product.candidate.sample_rate_hz,
                capability_tier=product.localization_branch.capability_tier,
                selected_sensor_ids=product.selected_sensor_ids,
                selected_positions=product.selected_positions,
                selected_windows=product.classification_selected_windows,
                localization_position_m=product.localization_branch.localization_position_m,
                event_time_ns=product.candidate.event_time_ns,
                alias_cutoff_hz=product.localization_branch.alias_cutoff_hz,
            )
            if localized is None:
                return detection_products
            self._metrics.beamform_renders += 1
            if extra_features:
                localized.classification.features.update(extra_features)
            await self._maybe_trigger_speech_capture(
                node_id=product.candidate.source_node_id,
                sensor_id=product.localization_branch.reference_sensor,
                event_time_ns=product.candidate.event_time_ns,
                scores=localized.classification.scores,
            )
            maybe_detection = await self._assemble_reporting_branch(
                product=product,
                classified=localized,
                reporting_modality="localized",
                localization_position_m=product.localization_branch.localization_position_m,
                localization_confidence=product.localization_branch.localization_confidence,
                localization_gdop=product.localization_branch.localization_gdop,
                localization_position_covariance_m2=product.localization_branch.localization_position_covariance_m2,
                localization_range_observability=product.localization_branch.localization_range_observability,
                localization_residual_rms_seconds=product.localization_branch.localization_residual_rms_seconds,
                localization_range_projection_mode=product.localization_branch.localization_range_projection_mode,
                reference_sensor=product.localization_branch.reference_sensor,
                reference_signal=product.localization_branch.reference_signal,
                tdoa_s=product.localization_branch.tdoa_s,
                capability_tier=product.localization_branch.capability_tier,
                localization_method=product.localization_branch.localization_method,
            )
            if maybe_detection is not None:
                detection_products.append(maybe_detection)
        return detection_products

    async def _classify_omni_branch(
        self,
        *,
        product: LocalizedCandidate,
        localized_result,
    ):
        if localized_result is not None and self._can_reuse_localized_result_for_omni(
            product=product,
            localized_result=localized_result,
        ):
            self._metrics.classification_reuse_hits += 1
            return localized_result
        return await self._classification_orchestrator.classify_omni_only(
            reference_signal=product.omni_classification_reference_signal,
            sample_rate_hz=product.candidate.sample_rate_hz,
            event_time_ns=product.candidate.event_time_ns,
        )

    @staticmethod
    def _can_reuse_localized_result_for_omni(
        *,
        product: LocalizedCandidate,
        localized_result,
    ) -> bool:
        if product.localization_branch is None:
            return False
        if localized_result.beamformed_classification is not None:
            return False
        if localized_result.classification_path != "omni":
            return False
        return product.localization_branch.reference_sensor == product.omni_reference_sensor

    async def _classification_windows_for_event(
        self,
        *,
        candidate: EventCandidate,
        selected_sensor_ids: list[str],
        fallback_windows: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        if (
            self.localization_config.classification_window_seconds
            <= self.localization_config.localization_window_seconds
        ):
            return fallback_windows

        trailing_windows = await self.buffer.get_synchronized_window_ending_at(
            sensor_ids=selected_sensor_ids,
            end_time_ns=candidate.event_time_ns,
            window_seconds=self.localization_config.classification_window_seconds,
            sample_rate_hz=candidate.sample_rate_hz,
        )
        # Use trailing windows per-sensor where available so that sensors with
        # sufficient buffered history give BirdNET longer audio.  Previously the
        # all-or-nothing count check caused every sensor to fall back to the short
        # localization window whenever any single sensor lacked a full trailing window.
        merged: dict[str, np.ndarray] = {}
        for sensor_id in selected_sensor_ids:
            if sensor_id in trailing_windows:
                merged[sensor_id] = trailing_windows[sensor_id]
            elif sensor_id in fallback_windows:
                merged[sensor_id] = fallback_windows[sensor_id]
        return merged

    async def _classification_quality_for_event(
        self,
        *,
        candidate: EventCandidate,
        selected_sensor_ids: list[str],
        localization_audio_quality: dict[str, AudioCoverageStats],
        classification_windows: dict[str, np.ndarray],
    ) -> tuple[dict[str, AudioCoverageStats], str]:
        if (
            self.localization_config.classification_window_seconds
            <= self.localization_config.localization_window_seconds
        ):
            return localization_audio_quality, "localization_centered"

        trailing_quality = await self.buffer.get_synchronized_window_ending_at_coverage_stats(
            sensor_ids=selected_sensor_ids,
            end_time_ns=candidate.event_time_ns,
            window_seconds=self.localization_config.classification_window_seconds,
            sample_rate_hz=candidate.sample_rate_hz,
        )
        merged: dict[str, AudioCoverageStats] = {}
        trailing_count = 0
        for sensor_id in selected_sensor_ids:
            if sensor_id in classification_windows and sensor_id in trailing_quality:
                merged[sensor_id] = trailing_quality[sensor_id]
                trailing_count += 1
            elif sensor_id in localization_audio_quality:
                merged[sensor_id] = localization_audio_quality[sensor_id]
        if trailing_count == 0:
            source = "localization_centered"
        elif trailing_count == len(merged):
            source = "classification_trailing"
        else:
            source = "mixed"
        return merged, source

    def _record_degraded_audio_quality_metrics(
        self,
        *,
        candidate: EventCandidate,
        source_window_type: str,
        audio_quality: dict[str, AudioCoverageStats],
    ) -> None:
        degraded_stats = {
            sensor_id: stats
            for sensor_id, stats in audio_quality.items()
            if stats.degraded
        }
        if not degraded_stats:
            return
        # Always count every degraded sensor; the cumulative metric is the
        # authoritative signal for monitoring.
        self._metrics.frames_zero_padded_degraded += len(degraded_stats)
        # Rate-limit the log per source node — a persistently degraded node
        # would otherwise emit this warning on every candidate and saturate the
        # log ring buffer (observed ~7 warnings/sec on a single-node deployment).
        rate_key = candidate.source_node_id or ""
        now_s = time.monotonic()
        last_logged_s = self._degraded_audio_warning_last_logged_s.get(rate_key, 0.0)
        if now_s - last_logged_s < self._degraded_audio_warning_interval_seconds:
            return
        self._degraded_audio_warning_last_logged_s[rate_key] = now_s
        logger.warning(
            "Zero-padded degraded audio coverage detected",
            extra={
                "candidate_id": candidate.id,
                "source_node_id": candidate.source_node_id,
                "source_window_type": source_window_type,
                "degraded_sensor_ids": sorted(degraded_stats.keys()),
                "max_missing_ratio": max(stats.missing_ratio for stats in degraded_stats.values()),
                "max_gap_seconds": max(stats.max_gap_seconds for stats in degraded_stats.values()),
                "degraded_count_total": self._metrics.frames_zero_padded_degraded,
            },
        )

    @staticmethod
    def _nearest_reference_sensor(
        selected_positions: dict[str, np.ndarray],
        position_m: tuple[float, float, float],
    ) -> str | None:
        """Sensor id whose position is closest to ``position_m`` (argmin distance)."""
        if not selected_positions:
            return None
        target = np.asarray(position_m, dtype=np.float64)
        best_id: str | None = None
        best_dist = float("inf")
        for sensor_id, pos in selected_positions.items():
            dist = float(np.linalg.norm(np.asarray(pos, dtype=np.float64) - target))
            if dist < best_dist:
                best_dist = dist
                best_id = sensor_id
        return best_id

    def _nearest_node_omni_sensor(self, product: LocalizedCandidate) -> str | None:
        """Return the nearest sensor id for nearest_node_omni classification.

        Only active when ``classification_audio_source == "nearest_node_omni"`` and
        the localized branch's confidence clears ``min_localization_confidence``;
        otherwise ``None`` so the caller falls back to the loudest-mic/omni path.
        """
        if self.settings.classification_audio_source != "nearest_node_omni":
            return None
        branch = product.localization_branch
        if branch is None:
            return None
        if branch.localization_confidence < self.settings.min_localization_confidence:
            return None
        return self._nearest_reference_sensor(
            product.selected_positions,
            branch.localization_position_m,
        )

    def _build_reference_sensor_candidate(
        self,
        *,
        candidate: EventCandidate,
        energies: dict[str, float],
        sensor_positions: dict[str, np.ndarray],
        selected_ids: list[str],
        selected_windows: dict[str, np.ndarray],
        classification_windows: dict[str, np.ndarray],
        environment_summary: dict[str, Any],
    ) -> LocalizedCandidate:
        reference_sensor = max(energies.items(), key=lambda item: item[1])[0]
        ref_signal = selected_windows[reference_sensor]
        ref_pos = sensor_positions[reference_sensor]
        return LocalizedCandidate(
            candidate=candidate,
            localization_branch=None,
            selected_sensor_ids=selected_ids,
            selected_windows=selected_windows,
            selected_positions={sensor_id: sensor_positions[sensor_id] for sensor_id in selected_ids},
            classification_selected_windows=classification_windows,
            omni_reference_sensor=reference_sensor,
            omni_reference_signal=ref_signal,
            omni_position_m=(float(ref_pos[0]), float(ref_pos[1]), float(ref_pos[2])),
            omni_classification_reference_signal=classification_windows.get(reference_sensor, ref_signal),
            localization_audio_quality={},
            classification_audio_quality={},
            classification_audio_quality_source="localization_centered",
            environment=environment_summary,
        )

    def _clamp_localization_covariance(
        self,
        covariance_m2,
        *,
        position_m,
        contributing_centroid_m,
    ):
        """Range-proportional covariance clamp shared by both ingest seams.

        Effective per-axis std ceiling scales with distance from the contributing
        sensors, so a 1 m near-field solve stays tight while a 1 km bearing cone can
        carry honest uncertainty (up to the absolute ceiling). ``std_factor <= 0``
        reproduces the legacy fixed clamp exactly.
        """
        cfg = self.localization_config
        range_m = 0.0
        if contributing_centroid_m is not None and position_m is not None:
            pos = np.asarray(position_m, dtype=np.float64).reshape(-1)
            cen = np.asarray(contributing_centroid_m, dtype=np.float64).reshape(-1)
            if pos.size == 3 and cen.size == 3 and np.all(np.isfinite(pos)) and np.all(np.isfinite(cen)):
                range_m = float(np.linalg.norm(pos - cen))
        capped, range_capped = clamp_covariance_eigenvalues_range_proportional(
            covariance_m2,
            range_m=range_m,
            std_factor=cfg.localization_std_range_factor,
            floor_std_m=cfg.localization_position_std_floor_m,
            ceiling_std_m=cfg.localization_max_position_std_m,
        )
        if range_capped:
            self._metrics.localization_covariance_range_capped_count += 1
        return capped

    async def _maybe_fuse_multi_node_bearing(
        self,
        branch: LocalizationBranch,
        *,
        node_id: str,
        contributing_centroid_m: np.ndarray | None,
        event_time_ns: int,
        sound_speed_mps: float | None,
    ) -> LocalizationBranch:
        """Opportunistically upgrade a single-node cone via bearing triangulation.

        Server-side, feature-flagged (Phase 4). Registers this cone as a bearing
        observation and, if another node's cone corroborates within the window,
        intersects them to recover range — upgrading the branch in place. Earlier
        detections are not retro-edited (Phase 3 track association merges them); this
        is purely opportunistic on the second-arriving candidate, so cadence is
        unchanged. Returns the branch unmodified when disabled, ineligible, or
        degenerate.
        """

        if not self.settings.multi_node_bearing_fusion_enabled:
            return branch
        if contributing_centroid_m is None or branch.localization_position_covariance_m2 is None:
            return branch
        mode = branch.localization_range_projection_mode
        obs_low = (
            branch.localization_range_observability is not None
            and branch.localization_range_observability < 0.10
        )
        # Only unobservable-range single-node cones benefit; a refined range is left
        # alone unless its observability is low (marginal range).
        if not (mode == RANGE_BEARING_PROJECTED or (mode == RANGE_REFINED and obs_low)):
            return branch

        position = np.asarray(branch.localization_position_m, dtype=np.float64)
        centroid = np.asarray(contributing_centroid_m, dtype=np.float64).reshape(-1)
        bearing = position - centroid
        norm = float(np.linalg.norm(bearing))
        if centroid.size != 3 or norm < 1e-9 or not np.all(np.isfinite(position)):
            return branch
        direction = bearing / norm
        cov = np.asarray(branch.localization_position_covariance_m2, dtype=np.float64)
        if cov.shape != (3, 3) or not np.all(np.isfinite(cov)):
            return branch
        radial_var = float(direction @ cov @ direction)
        lateral_std_m = float(np.sqrt(max((float(np.trace(cov)) - radial_var) / 2.0, 1e-4)))

        now_ns = time.time_ns()
        ttl_ns = int(self.settings.multi_node_bearing_ttl_seconds * 1e9)
        observation = BearingObservation(
            node_id=node_id,
            origin_m=centroid,
            direction=direction,
            lateral_std_m=lateral_std_m,
            confidence=float(branch.localization_confidence),
            event_time_ns=int(event_time_ns),
            range_prior_m=norm,
            expiry_ns=now_ns + ttl_ns,
        )
        await self._bearing_fusion_store.register(observation, now_ns=now_ns)
        corroborators = await self._bearing_fusion_store.corroborators(
            observation,
            now_ns=now_ns,
            window_seconds=self.settings.multi_node_bearing_window_seconds,
        )
        if not corroborators:
            return branch

        self._metrics.localization_bearing_fusion_attempt_count += 1
        result = fuse_bearings(
            [observation, *corroborators],
            min_separation_deg=self.settings.multi_node_bearing_min_separation_deg,
            max_condition=self.settings.multi_node_bearing_max_condition,
            max_range_m=self.localization_config.localization_max_range_m,
            sound_speed_mps=sound_speed_mps or 343.0,
            window_seconds=self.settings.multi_node_bearing_window_seconds,
        )
        if not isinstance(result, BearingFusionResult):
            if result == "stale":
                self._metrics.localization_bearing_fusion_stale_count += 1
            else:
                self._metrics.localization_bearing_fusion_degenerate_count += 1
            return branch

        fused_position = result.position_m
        fused_cov = self._clamp_localization_covariance(
            result.covariance_m2.tolist(),
            position_m=fused_position,
            contributing_centroid_m=centroid,
        )
        observability = range_observability_from_covariance(
            result.covariance_m2, fused_position - centroid
        )
        self._metrics.localization_bearing_fusion_fused_count += 1
        self._metrics.last_bearing_fusion_contributor_count = len(result.contributor_node_ids)
        return replace(
            branch,
            localization_position_m=(
                float(fused_position[0]),
                float(fused_position[1]),
                float(fused_position[2]),
            ),
            localization_position_covariance_m2=fused_cov,
            localization_range_observability=observability,
            localization_range_projection_mode=RANGE_REFINED,
            localization_confidence=min(
                0.95, max(float(branch.localization_confidence), result.confidence)
            ),
            localization_method="multi_node_bearing_triangulation",
        )

    def _record_range_projection_metrics(self, mode: str | None) -> None:
        """Count range-projection states canonically across both ingest paths."""
        canonical = normalize_range_mode(mode)
        if canonical == RANGE_ASYMPTOTIC:
            self._metrics.localization_range_asymptotic_count += 1
        elif canonical == RANGE_BOUNDARY:
            self._metrics.localization_range_boundary_count += 1
        # Legacy telemetry: surfaces sidecars that have not yet been rebuilt with
        # the canonical vocabulary (they still get the haircut via normalization).
        if mode == LEGACY_PRIOR_PROJECTED:
            self._metrics.localization_prior_projected_count += 1

    def _localization_position_is_reportable(
        self,
        position_m,
        contributing_centroid_m: np.ndarray | None,
    ) -> bool:
        """Apply the shared physical localization envelope to an inbound position."""
        position = np.asarray(position_m, dtype=np.float64).reshape(-1)
        if position.size != 3 or not np.all(np.isfinite(position)):
            return False
        if float(np.linalg.norm(position)) > _SANITY_GATE_ORIGIN_BACKSTOP_M:
            return False

        max_range_m = self.localization_config.localization_max_range_m
        if max_range_m <= 0.0:
            return True
        centroid = (
            np.asarray(contributing_centroid_m, dtype=np.float64).reshape(-1)
            if contributing_centroid_m is not None
            else np.zeros(3, dtype=np.float64)
        )
        return centroid.size != 3 or float(np.linalg.norm(position - centroid)) <= max_range_m

    def _build_localization_branch(
        self,
        *,
        localization,
        selected_windows: dict[str, np.ndarray],
        classification_windows: dict[str, np.ndarray],
        capability_tier: str,
        contributing_centroid_m: np.ndarray | None = None,
    ) -> LocalizationBranch | None:
        # Sanity gate: drop unphysical solver blowups before they become a track. The
        # candidate falls back to its omni branch (no localized position) instead.
        # Range is measured from the contributing-sensor centroid (how far the source
        # is from the array), NOT the site origin — a node surveyed far from origin
        # must still be allowed a legitimate 1 km solve. A 5 km absolute origin
        # backstop still catches runaway coordinates.
        if not FusionNode._localization_position_is_reportable(
            self,
            localization.position_m,
            contributing_centroid_m,
        ):
            self._metrics.localization_rejected_out_of_range_count += 1
            self._record_silent_drop(stage="localization", reason="position_out_of_range")
            return None

        reference_signal = selected_windows[localization.reference_sensor]
        localization_method = self._current_localizer_name()
        self._metrics.last_localization_algorithm = localization.resolved_algorithm or localization_method
        self._metrics.last_attempted_algorithm = localization.attempted_algorithm or localization_method
        if localization.position_covariance_m2 is None:
            self._metrics.localization_covariance_missing_count += 1
        if localization.range_observability is not None and localization.range_observability < 0.10:
            self._metrics.localization_range_observability_low_count += 1
        self._record_range_projection_metrics(localization.range_projection_mode)
        if (
            localization.attempted_algorithm
            and localization.resolved_algorithm
            and localization.attempted_algorithm != localization.resolved_algorithm
        ):
            self._metrics.localization_fallback_count += 1
        if localization.wavelength_factor is not None and localization.wavelength_factor < 1.0:
            self._metrics.localization_band_aliased_count += 1
        range_projection_mode = normalize_range_mode(localization.range_projection_mode)
        localization_confidence, localization_range_observability = apply_unobservable_range_haircut(
            mode=range_projection_mode,
            confidence=localization.confidence,
            range_observability=localization.range_observability,
        )
        # Cap covariance so unobservable-range / cone solves can't carry km-scale
        # ellipses into tracking. Catches every covariance path (jacobian + cone).
        # The ceiling scales with distance from the contributing sensors so honest
        # uncertainty survives at range while near-field solves stay tight.
        capped_covariance_m2 = self._clamp_localization_covariance(
            localization.position_covariance_m2,
            position_m=localization.position_m,
            contributing_centroid_m=contributing_centroid_m,
        )
        return LocalizationBranch(
            localization_position_m=localization.position_m,
            localization_confidence=localization_confidence,
            localization_gdop=localization.gdop,
            localization_position_covariance_m2=capped_covariance_m2,
            localization_range_observability=localization_range_observability,
            localization_residual_rms_seconds=localization.residual_rms_seconds,
            localization_range_projection_mode=range_projection_mode,
            reference_sensor=localization.reference_sensor,
            reference_signal=reference_signal,
            classification_reference_signal=classification_windows.get(
                localization.reference_sensor,
                reference_signal,
            ),
            tdoa_s=localization.tdoa_s,
            localization_method=localization_method,
            capability_tier=capability_tier,
            wavelength_factor=localization.wavelength_factor,
            dominant_frequency_hz=localization.dominant_frequency_hz,
            alias_cutoff_hz=localization.alias_cutoff_hz,
        )

    @staticmethod
    def _merge_source_observation_ids(
        existing_source_observation_ids: list[str],
        new_source_observation_ids: list[str],
    ) -> list[str]:
        return list(dict.fromkeys([*existing_source_observation_ids, *new_source_observation_ids]))

    @staticmethod
    def _classification_features_for_reporting(
        product: LocalizedCandidate,
        *,
        classified,
        reporting_modality: str,
    ) -> dict[str, Any]:
        classification_features = dict(classified.classification.features)
        if reporting_modality != "localized" or product.localization_branch is None:
            return classification_features

        for feature_name, feature_value in (
            ("wavelength_factor", product.localization_branch.wavelength_factor),
            ("dominant_frequency_hz", product.localization_branch.dominant_frequency_hz),
            ("alias_cutoff_hz", product.localization_branch.alias_cutoff_hz),
        ):
            if feature_value is not None:
                classification_features[feature_name] = feature_value
        return classification_features

    async def _assemble_reporting_branch(
        self,
        *,
        product: LocalizedCandidate,
        classified,
        reporting_modality: str,
        localization_position_m: tuple[float, float, float],
        localization_confidence: float,
        localization_gdop: float,
        localization_position_covariance_m2: list[list[float]] | None,
        localization_range_observability: float | None,
        localization_residual_rms_seconds: float | None,
        localization_range_projection_mode: str | None,
        reference_sensor: str,
        reference_signal: np.ndarray,
        tdoa_s: dict[str, float],
        capability_tier: str,
        localization_method: str,
    ) -> DetectionProduct | None:
        if classified.beamforming_error:
            self._metrics.beamforming_failures += 1

        source_node_id = await self.registry.node_id_for_sensor(reference_sensor)
        audio_quality = self._audio_quality_for_detection(product, reference_sensor)
        branch_details = {
            "label": classified.classification.label,
            "confidence": classified.classification.confidence,
            "classification_path": classified.classification_path,
            "capability_tier": capability_tier,
            "event_time_ns": product.candidate.event_time_ns,
            "localization_method": localization_method,
            "localization_confidence": localization_confidence,
            "gdop": localization_gdop,
            "suppressed": False,
        }
        decision = await self.reporting_policy.decide(
            event_time_ns=product.candidate.event_time_ns,
            source_node_id=source_node_id,
            label=classified.classification.label,
            reporting_modality=reporting_modality,
            branch_details=branch_details,
        )
        if decision.action == "suppress":
            return None
        if decision.action == "enrich_existing":
            existing_detection = DetectionEvent.model_validate(decision.existing_detection)
            existing_feature_summary = dict(existing_detection.feature_summary)
            existing_feature_summary["branch_evidence"] = decision.branch_evidence
            existing_detection.feature_summary = existing_feature_summary
            existing_detection.source_observation_ids = self._merge_source_observation_ids(
                existing_detection.source_observation_ids,
                product.candidate.source_observation_ids,
            )
            await self.storage.update_detection(
                detection=existing_detection,
                snippet_path=existing_detection.snippet_path,
                snippet_expires_ns=None,
                retention_tier=existing_detection.retention_tier.value,
            )
            return None

        persist_mode = "update" if decision.action == "upgrade_existing" else "insert"
        try:
            classification_features = self._classification_features_for_reporting(
                product,
                classified=classified,
                reporting_modality=reporting_modality,
            )
            assembly = await self._detection_assembler.assemble(
                localization_position_m=localization_position_m,
                localization_confidence=localization_confidence,
                localization_gdop=localization_gdop,
                localization_position_covariance_m2=localization_position_covariance_m2,
                localization_range_observability=localization_range_observability,
                localization_residual_rms_seconds=localization_residual_rms_seconds,
                localization_range_projection_mode=localization_range_projection_mode,
                reference_sensor=reference_sensor,
                tdoa_s=tdoa_s,
                selected_sensor_ids=product.selected_sensor_ids,
                reference_signal=reference_signal,
                capability_tier=capability_tier,
                localization_method=localization_method,
                environment=product.environment,
                classification_label=classified.classification.label,
                classification_confidence=classified.classification.confidence,
                classification_scores=classified.classification.scores,
                classification_features=classification_features,
                classification_path=classified.classification_path,
                omni_confidence=classified.omni_classification.confidence,
                beamformed_classification_confidence=(
                    classified.beamformed_classification.confidence
                    if classified.beamformed_classification is not None
                    else None
                ),
                beamformed_classification_label=(
                    classified.beamformed_classification.label
                    if classified.beamformed_classification is not None
                    else None
                ),
                beamforming_error=classified.beamforming_error,
                classification_signal=classified.classification_signal,
                label_category=classified.label_category,
                iff_category=classified.iff_category,
                label_id=classified.label_id,
                report_window_start_ns=decision.report_window_start_ns,
                report_window_end_ns=decision.report_window_end_ns,
                reporting_modality=decision.reporting_modality,
                branch_evidence=decision.branch_evidence,
                event_time_ns=product.candidate.event_time_ns,
                source_type=product.candidate.source_type,
                time_quality=product.candidate.time_quality,
                source_observation_ids=product.candidate.source_observation_ids,
                sample_rate_hz=product.candidate.sample_rate_hz,
                audio_quality=audio_quality,
                existing_detection=decision.existing_detection,
                persist_mode=persist_mode,
                tracker=self.tracker,
                storage_batch_ctx=self._storage_batch,
            )
        except aiosqlite.IntegrityError as exc:
            # A concurrent worker won the canonical insert for this reporting window.
            if "uq_detections_reporting_window_canonical" in str(exc):
                return None
            raise
        if assembly.suppressed_by_zone:
            self._metrics.detections_suppressed_by_zone += 1
        return DetectionProduct(
            detection=assembly.detection,
            track=assembly.track,
            suppressed_by_zone=assembly.suppressed_by_zone,
            suppression_reasons=assembly.suppression_reasons,
            pipeline_item_id=assembly.detection.id,
            event_time_ns=assembly.detection.timestamp_ns,
        )

    @staticmethod
    def _audio_quality_for_detection(product: LocalizedCandidate, reference_sensor: str) -> dict[str, Any]:
        stats = product.classification_audio_quality.get(reference_sensor)
        source_window_type = product.classification_audio_quality_source
        if stats is None:
            stats = product.localization_audio_quality.get(reference_sensor)
            source_window_type = "localization_centered"
        if stats is None:
            return {
                "source_window_type": source_window_type,
                "degraded": True,
                "warning": True,
                "coverage_ratio": 0.0,
                "missing_ratio": 1.0,
                "max_gap_seconds": 0.0,
                "sample_count": 0,
                "covered_samples": 0,
                "missing_samples": 0,
            }
        return stats.to_feature_summary(source_window_type=source_window_type)

    # ------------------------------------------------------------------
    # Rules & delivery stage
    # ------------------------------------------------------------------

    async def _process_rules_and_delivery(self, product: DetectionProduct) -> None:
        detection = product.detection
        track = product.track

        if product.suppressed_by_zone:
            self._record_silent_drop(
                stage="rules",
                reason="suppressed_by_zone",
                candidate_id=product.pipeline_item_id,
                event_time_ns=product.event_time_ns,
                source_node_id=detection.source_node_id,
                suppression_reasons=list(product.suppression_reasons or []),
            )
            return

        # Skip broadcasting unknown/0% confidence detections — they provide no
        # actionable information for the operator and clutter the live feed.
        # The DB row is still written so the reporting-window policy can track
        # it and upgrade to a real species label if one arrives later.
        if detection.label == "unknown" and detection.label_confidence == 0.0:
            self._record_silent_drop(
                stage="rules",
                reason="unknown_zero_confidence",
                candidate_id=product.pipeline_item_id,
                event_time_ns=product.event_time_ns,
                source_node_id=detection.source_node_id,
            )
            return

        payload = {
            "event_id": detection.event_id,
            "event_type": "detection",
            "source_type": detection.source_type,
            "node_id": detection.source_node_id,
            "toa_ns": detection.toa_ns,
            "tor_ns": detection.tor_ns,
            "time_quality": detection.time_quality.value,
            "stale_ns": detection.stale_ns,
            "position": {
                "local_m": list(detection.position_m),
                "geo": detection.position_geo.model_dump(mode="json") if detection.position_geo else None,
            },
            "provenance": {
                "source_observation_ids": detection.source_observation_ids,
                "track_id": detection.track_id,
            },
            "type": "detection",
            "detection": detection.model_dump(mode="json"),
            "track": track.model_dump(mode="json") if track is not None else None,
            "server_time_ns": time.time_ns(),
        }
        await self.live_callback(payload)
        self._metrics.detections_emitted += 1
        self._metrics.last_detection_emission_ns = time.time_ns()
        self._realtime_tracker.record_completed(event_time_ns=product.event_time_ns)

        evaluations = await self.rules_engine.evaluate(detection=detection, track=track)
        for evaluation in evaluations:
            if not evaluation.matched:
                continue
            for descriptor in evaluation.descriptors:
                await self._dispatch_rule_action(
                    descriptor=descriptor,
                    rule_id=evaluation.rule_id,
                    detection=detection,
                    track=track,
                )

    async def _dispatch_rule_action(
        self,
        *,
        descriptor: ActionDescriptor,
        rule_id: str,
        detection: DetectionEvent | None,
        track: TrackState | None,
    ) -> None:
        timestamp_ns = time.time_ns()
        payload = {
            "action_type": descriptor.action_type,
            "destination": descriptor.destination,
            "priority": descriptor.priority,
            "payload": descriptor.payload,
        }
        alert_id = await self.storage.insert_alert(
            timestamp_ns=timestamp_ns,
            rule_id=rule_id,
            detection_id=detection.id if detection else None,
            track_id=track.id if track else None,
            destination=descriptor.destination,
            priority=descriptor.priority,
            status="sent",
            payload=payload,
        )
        handler = self._action_handlers.get(descriptor.destination)
        if handler is None:
            handler = self._action_handlers.get("log")
        if handler is None:
            return
        status = "sent"
        patch: dict[str, Any] = {}
        try:
            result = await handler.handle(
                descriptor,
                detection=detection,
                track=track,
                alert_id=alert_id,
                rule_id=rule_id,
            )
            patch = result if isinstance(result, dict) else {}
            delivered = bool(patch.get("delivered", True))
            if not delivered:
                status = "escalated"
        except Exception as exc:  # pragma: no cover - resilience path
            status = "escalated"
            patch = {"error": f"{type(exc).__name__}: {exc}"}
        await self.storage.update_alert_status(
            alert_id=alert_id,
            status=status,
            updated_ns=time.time_ns(),
            payload_patch=patch,
        )

    async def _on_transcript(self, record: TranscriptRecord) -> None:
        payload = {
            "type": "transcript",
            "event_type": "transcript",
            "transcript": record.model_dump(mode="json"),
            "server_time_ns": time.time_ns(),
        }
        await self.live_callback(payload)

        evaluations = await self.rules_engine.evaluate_transcript(record)
        for evaluation in evaluations:
            if not evaluation.matched:
                continue
            for descriptor in evaluation.descriptors:
                await self._dispatch_rule_action(
                    descriptor=descriptor,
                    rule_id=evaluation.rule_id,
                    detection=None,
                    track=None,
                )

    async def _maybe_trigger_speech_capture(
        self,
        *,
        node_id: str,
        sensor_id: str,
        event_time_ns: int,
        scores: dict[str, float],
    ) -> None:
        if self._speech_capture is None:
            return
        speech_confidence = scores.get("Speech")
        if speech_confidence is None:
            return
        await self._speech_capture.maybe_trigger(
            node_id=node_id,
            sensor_id=sensor_id,
            event_time_ns=event_time_ns,
            speech_confidence=float(speech_confidence),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_beamform_failure(self, error: str) -> None:
        self._metrics.beamform_failures += 1

    def _create_beamformer(self) -> Beamformer | None:
        """Create the classification beamformer with recall-biased settings.

        Uses the ``create_beamformer`` factory from the beamforming module,
        applying ``classifier_diagonal_loading_scale`` so that MVDR widens
        its main lobe for higher recall — preferable when the output feeds
        a classifier that benefits from capturing more target energy.
        """
        if not self.settings.beamformed_classification_enabled:
            return None
        if self.settings.beamformer_type == "band_split_das":
            from minimappr.core.beamforming import BandSplitDasRenderer, BandSplitRenderConfig

            return BandSplitDasRenderer(
                config=BandSplitRenderConfig(
                    highpass_hz=self.settings.beamform_render_highpass_hz,
                    low_crossover_width_hz=self.settings.beamform_low_crossover_width_hz,
                    high_crossover_width_min_hz=self.settings.beamform_high_crossover_width_min_hz,
                    high_crossover_width_fraction=self.settings.beamform_high_crossover_width_fraction,
                )
            )
        return create_beamformer(
            beamformer_type=self.settings.beamformer_type,
            diagonal_loading=self.settings.mvdr_diagonal_loading,
            classifier_diagonal_loading_scale=self.localization_config.classifier_diagonal_loading_scale,
        )

    def _build_classification_chunking_policy(self) -> ClassificationChunkingPolicy | None:
        if not self.fusion_config.birdnet_chunked_dispatch_enabled:
            return None
        if not self.settings.birdnet_enabled:
            return None
        stride_seconds = max(
            self.localization_config.localization_window_seconds,
            self.localization_config.classification_window_seconds - self.fusion_config.birdnet_chunk_overlap_seconds,
        )
        return ClassificationChunkingPolicy(
            stride_seconds=stride_seconds,
            max_retries_per_chunk=self.fusion_config.birdnet_chunk_max_retries_per_chunk,
            min_retry_progress_seconds=self.fusion_config.birdnet_chunk_min_retry_progress_seconds,
        )

    def _should_suppress_chunked_classification(self, product: LocalizedCandidate) -> bool:
        if self._classification_chunking_policy is None:
            return False
        return not self._classification_chunking_policy.should_dispatch(
            source_node_id=product.candidate.source_node_id,
            event_time_ns=product.candidate.event_time_ns,
        )

    @staticmethod
    def _detection_products_had_backend_failure(detection_products: list[DetectionProduct]) -> bool:
        return bool(detection_products) and all(
            product.detection.feature_summary.get("reason") == "classification_error"
            for product in detection_products
        )

    def _record_chunk_dispatch_outcome(
        self,
        *,
        product: LocalizedCandidate,
        detection_products: list[DetectionProduct],
        classification_failed: bool,
    ) -> None:
        if self._classification_chunking_policy is None:
            return
        produced_actionable_detection = any(
            detection_product.detection.label != "unknown"
            and detection_product.detection.label_confidence > 0.0
            for detection_product in detection_products
        )
        self._classification_chunking_policy.record_dispatch_outcome(
            source_node_id=product.candidate.source_node_id,
            event_time_ns=product.candidate.event_time_ns,
            produced_actionable_detection=produced_actionable_detection,
            allow_retry_for_non_actionable=(
                not classification_failed
                or self.fusion_config.birdnet_chunk_retry_on_classifier_error
            ),
        )

    def _current_localizer_name(self) -> str:
        getter = getattr(self.localizer, "last_algorithm_name", None)
        if callable(getter):
            try:
                value = getter()
            except Exception:  # pragma: no cover - resilience path
                value = None
            if isinstance(value, str) and value:
                return value
        return type(self.localizer).__name__

    async def _refresh_taxonomy(self, now_ns: int | None = None, force: bool = False) -> None:
        now = now_ns if now_ns is not None else time.time_ns()
        refresh_interval_ns = int(self.fusion_config.taxonomy_refresh_interval_seconds * 1_000_000_000)
        if not force and now - self._taxonomy_refresh_ns < refresh_interval_ns:
            return
        rows = await self.storage.list_labels()
        if hasattr(self.taxonomy_provider, "merge_labels"):
            self.taxonomy_provider.merge_labels(rows)
        self._taxonomy_refresh_ns = now

    def _record_ingest_result_metrics(self, request: IngestFrameRequest, result: Any) -> None:
        self._metrics.frames_accepted += 1
        if result.environment_counts is not None:
            self._metrics.environment_samples_ingested += result.environment_counts.ingested
            self._metrics.environment_samples_persisted += result.environment_counts.persisted
        if result.sequence_gap_count > 0:
            self._metrics.frame_sequence_gaps += result.sequence_gap_count

        per_node_metrics = self._per_node_frame_metrics.setdefault(
            request.node.id,
            {"frames_accepted": 0, "frame_gaps": 0, "last_frame_ns": 0},
        )
        per_node_metrics["frames_accepted"] += 1
        per_node_metrics["last_frame_ns"] = max(
            per_node_metrics["last_frame_ns"],
            request.frame.start_time_ns,
        )
        if result.sequence_gap_count > 0:
            per_node_metrics["frame_gaps"] += result.sequence_gap_count

        result.response.queue_depth = self._localization_queue.qsize()

    async def _queue_trigger_candidate(self, request: IngestFrameRequest, result: Any) -> None:
        if not result.triggered:
            return

        enqueued_ns = time.time_ns()
        candidate = EventCandidate(
            id=f"evt-{next(self._candidate_counter):08d}",
            source_node_id=request.node.id,
            event_time_ns=result.event_time_ns,
            sample_rate_hz=result.sample_rate_hz,
            source_type=result.source_type,
            time_quality=result.time_quality,
            source_observation_ids=result.observation_ids or [],
            enqueued_ns=enqueued_ns,
        )
        if await self._enqueue_stage(self._localization_queue, candidate):
            self._ingest_processor.confirm_trigger(result.event_time_ns)
            self._metrics.triggers_enqueued += 1
            self._metrics.last_trigger_enqueue_ns = enqueued_ns
            result.response.queued_event_id = candidate.id
            return

        # Queue full — report as not triggered to the caller.
        result.response.triggered = False
        self._metrics.triggers_dropped_queue_full += 1

    def _record_stage_enqueue(
        self,
        *,
        stage_name: str,
        item_id: str,
        event_time_ns: int,
    ) -> None:
        self._realtime_tracker.mark_enqueued(
            stage_name=stage_name,
            item_id=item_id,
            event_time_ns=event_time_ns,
        )

    def _record_stage_backpressure_drop(
        self,
        *,
        event_time_ns: int,
        item_id: str,
        queue: asyncio.Queue,
        stage_name: str,
    ) -> None:
        self._metrics.stage_drops_backpressure += 1
        now_s = time.monotonic()
        last_logged_s = self._backpressure_warning_last_logged_s.get(stage_name, 0.0)
        if now_s - last_logged_s >= self._backpressure_warning_interval_seconds:
            self._backpressure_warning_last_logged_s[stage_name] = now_s
            logger.warning(
                "Dropping pipeline item due to fusion backpressure",
                extra={
                    "stage_name": stage_name,
                    "item_id": item_id,
                    "event_time_ns": event_time_ns,
                    "queue_size": queue.qsize(),
                    "queue_maxsize": queue.maxsize,
                },
            )

    _DROP_REASON_COUNTER_FIELDS: dict[str, str] = {
        "localization": "localization_drops_by_reason",
        "classification": "classification_drops_by_reason",
        "rules": "rules_drops_by_reason",
    }

    def _record_silent_drop(
        self,
        *,
        stage: str,
        reason: str,
        candidate_id: str | None = None,
        event_time_ns: int | None = None,
        **extras: Any,
    ) -> None:
        """Count and rate-limit-log a silent stage drop.

        A "silent drop" is a stage exit that produces no downstream item but is
        not a timeout or raised exception — historically these were invisible
        and produced a hung-pipeline symptom with no diagnosable signal.
        """
        field_name = self._DROP_REASON_COUNTER_FIELDS.get(stage)
        if field_name is None:
            return
        counters: dict[str, int] = getattr(self._metrics, field_name)
        counters[reason] = counters.get(reason, 0) + 1

        rate_key = (stage, reason)
        now_s = time.monotonic()
        last_logged_s = self._drop_warning_last_logged_s.get(rate_key, 0.0)
        if now_s - last_logged_s < self._drop_warning_interval_seconds:
            return
        self._drop_warning_last_logged_s[rate_key] = now_s
        log_extra: dict[str, Any] = {
            "stage_name": stage,
            "drop_reason": reason,
            "candidate_id": candidate_id,
            "event_time_ns": event_time_ns,
            "drop_count_for_reason": counters[reason],
        }
        log_extra.update(extras)
        logger.warning("Silent pipeline drop", extra=log_extra)

    _EXCEPTION_COUNTER_FIELDS: dict[str, str] = {
        "localization": "localization_exceptions_by_type",
        "classification": "classification_exceptions_by_type",
        "rules": "rules_exceptions_by_type",
    }

    def _record_worker_exception(
        self,
        *,
        stage: str,
        exc: BaseException,
        **extras: Any,
    ) -> None:
        """Count and rate-limit-log an unexpected worker exception.

        Mirrors the _record_silent_drop throttling pattern: same 10s interval,
        same throttle dict, keyed by (\"exception\", stage, exc_type) so a
        persistent bug floods neither logs nor the counter dict.
        """
        field_name = self._EXCEPTION_COUNTER_FIELDS.get(stage)
        if field_name is None:
            return
        counters: dict[str, int] = getattr(self._metrics, field_name)
        exc_type = type(exc).__name__
        counters[exc_type] = counters.get(exc_type, 0) + 1

        rate_key = ("exception", stage, exc_type)
        now_s = time.monotonic()
        last_logged_s = self._drop_warning_last_logged_s.get(rate_key, 0.0)
        if now_s - last_logged_s < self._drop_warning_interval_seconds:
            return
        self._drop_warning_last_logged_s[rate_key] = now_s
        logger.warning(
            "Fusion worker exception",
            extra={
                "stage_name": stage,
                "exception_type": exc_type,
                "exception_message": str(exc),
                "count_for_type": counters[exc_type],
                **extras,
            },
            exc_info=exc,
        )

    async def _enqueue_stage(self, queue: asyncio.Queue, item: Any) -> bool:
        tracker_stage_name = self._queue_stage_name(queue)
        tracker_item_id = self._stage_item_id(item)
        tracker_event_time_ns = self._stage_item_event_time_ns(item)
        if self.fusion_config.offline_replay_mode or not self.fusion_config.drop_on_backpressure:
            await queue.put(item)
            self._record_stage_enqueue(
                stage_name=tracker_stage_name,
                item_id=tracker_item_id,
                event_time_ns=tracker_event_time_ns,
            )
            return True
        try:
            queue.put_nowait(item)
            self._record_stage_enqueue(
                stage_name=tracker_stage_name,
                item_id=tracker_item_id,
                event_time_ns=tracker_event_time_ns,
            )
            return True
        except asyncio.QueueFull:
            self._record_stage_backpressure_drop(
                event_time_ns=tracker_event_time_ns,
                item_id=tracker_item_id,
                queue=queue,
                stage_name=tracker_stage_name,
            )
            return False

    def _queue_stage_name(self, queue: asyncio.Queue) -> str:
        stage_map: dict[asyncio.Queue, str] = {
            self._localization_queue: "localization",
            self._classification_queue: "classification",
            self._rules_queue: "rules",
        }
        return stage_map[queue]

    @staticmethod
    def _stage_item_id(item: Any) -> str:
        if isinstance(item, EventCandidate):
            return item.id
        if isinstance(item, LocalizedCandidate):
            return item.candidate.id
        if isinstance(item, DetectionProduct):
            return item.pipeline_item_id
        raise TypeError(f"Unsupported pipeline item type: {type(item).__name__}")

    @staticmethod
    def _stage_item_event_time_ns(item: Any) -> int:
        if isinstance(item, EventCandidate):
            return item.event_time_ns
        if isinstance(item, LocalizedCandidate):
            return item.candidate.event_time_ns
        if isinstance(item, DetectionProduct):
            return item.event_time_ns
        raise TypeError(f"Unsupported pipeline item type: {type(item).__name__}")

    @asynccontextmanager
    async def _storage_batch(self):
        begin_batch = getattr(self.storage, "begin_batch", None)
        if callable(begin_batch):
            async with begin_batch():
                yield
            return
        yield

    @staticmethod
    def _sensor_centroid(
        selected_positions: dict[str, np.ndarray],
    ) -> tuple[float, float, float] | None:
        if not selected_positions:
            return None
        stacked = np.stack([np.asarray(pos, dtype=np.float64) for pos in selected_positions.values()], axis=0)
        centroid = np.mean(stacked, axis=0)
        return (float(centroid[0]), float(centroid[1]), float(centroid[2]))
