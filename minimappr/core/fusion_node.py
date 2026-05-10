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
import itertools
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

import aiosqlite
import numpy as np

from minimappr.api.rust_dsp_manifests import LocalizedClassifierRenderRequest
from minimappr.classifiers.base import AudioClassifier
from minimappr.config import Settings
from minimappr.core.assembly import DetectionAssembler
from minimappr.core.audio_buffer import AudioCoverageStats, MultiSensorBuffer
from minimappr.core.beamforming import create_beamformer
from minimappr.core.classification_chunking import ClassificationChunkingPolicy
from minimappr.core.classification import ClassificationOrchestrator
from minimappr.core.degradation import CapabilityDegradationModel
from minimappr.core.environment import StaticEnvironmentProvider
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.ingest import EnvironmentUpdater, IngestProcessor
from minimappr.core.localization import LocalizationError
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.pipeline_realtime import PipelineRealtimeTracker
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
    GeoPoint,
    IngestFrameRequest,
    IngestFrameResponse,
    RetentionTier,
    TimeQuality,
    TrackState,
)
from minimappr.utils.audio import rms


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
    reference_sensor: str
    reference_signal: np.ndarray
    classification_reference_signal: np.ndarray
    tdoa_s: dict[str, float]
    localization_method: str
    capability_tier: str


@dataclass(slots=True)
class FusionMetrics:
    ingest_requests: int = 0
    frames_accepted: int = 0
    frames_rejected: int = 0
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
    last_localization_algorithm: str = "gcc_phat"
    last_attempted_algorithm: str = "gcc_phat"
    classification_reuse_hits: int = 0
    birdnet_chunk_dispatches_suppressed: int = 0


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
        taxonomy_provider: TaxonomyProvider | None = None,
        rules_engine: RuleEngine | None = None,
        action_handlers: dict[str, RuleActionHandler] | None = None,
        environment_provider: EnvironmentProvider | None = None,
        preprocessor_factory: NodePreprocessorFactory | None = None,
        degradation_model: CapabilityDegradationModel | None = None,
        beamformer: Beamformer | None = None,
    ) -> None:
        self.settings = settings
        self.localization_config = settings.localization_config()
        self.fusion_config = settings.fusion_config()
        self.registry = registry
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
        self.preprocessor_factory = preprocessor_factory or NodePreprocessorFactory(self.localization_config)
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
        )
        self._action_handlers = action_handlers or {
            "cop": WebsocketRuleActionHandler(live_callback),
            "log": LoggingRuleActionHandler(),
        }

        # -- extracted subsystem components ------------------------------------
        self._ingest_processor = IngestProcessor(
            localization_config=self.localization_config,
            fusion_config=self.fusion_config,
            registry=registry,
            buffer=buffer,
            storage=storage,
            coordinate_frame=coordinate_frame,
            preprocessor_factory=self.preprocessor_factory,
            environment_updater=EnvironmentUpdater(self.environment_provider),
            persist_observations_on_ingest=bool(settings.persist_observations_on_ingest),
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
            classifier_backend_name=settings.classifier_backend,
        )
        self._detection_assembler = DetectionAssembler(
            storage=storage,
            coordinate_frame=coordinate_frame,
            zone_matcher=zone_matcher,
            registry=registry,
            retention_policy=self.retention_policy,
            snippet_dir=settings.snippet_dir,
            snippet_retention_seconds=settings.snippet_retention_seconds,
            event_stale_seconds=settings.event_stale_seconds,
        )

        self._candidate_counter = itertools.count(1)
        self._taxonomy_refresh_ns = 0

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
        self._realtime_tracker = PipelineRealtimeTracker(
            ("localization", "classification", "rules")
        )
        self._classification_chunking_policy = self._build_classification_chunking_policy()
        self._last_error: str | None = None
        self._started = False
        self._stopping = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            return

        self._stopping = False
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
        self._started = True

    @property
    def accepted_frame_count(self) -> int:
        return self._ingest_processor.accepted_frame_count

    def rebind_runtime_dependencies(self, *, classifier: AudioClassifier, coordinate_frame: LocalCoordinateFrame) -> None:
        self.classifier = classifier
        self.coordinate_frame = coordinate_frame
        self._ingest_processor.replace_coordinate_frame(coordinate_frame)
        self._classification_orchestrator.replace_classifier(classifier)
        self._detection_assembler.replace_coordinate_frame(coordinate_frame)

    async def stop(self) -> None:
        if not self._started:
            return

        self._stopping = True
        self._started = False

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

        # Localized classifier renders are derived products, not raw sensor frames.
        # Keep event-time semantics for localization/classification, but update
        # node heartbeat freshness from server receipt time so online/offline
        # status reflects current connectivity rather than source clock skew.
        runtime = await self.registry.upsert(node, server_received_ns)
        position_geo = node.position_geo
        if position_geo is None and node.position_m is not None:
            position_geo = self.coordinate_frame.local_to_geo(node.position_m)
        await self.storage.upsert_node(
            spec=node,
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
        rust_audio_quality = (
            {reference_sensor: payload.audio_quality}
            if payload.audio_quality is not None
            else {}
        )
        capability_tier = (
            "full_3d"
            if payload.reporting_modality == "localized"
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
        localized_product = LocalizedCandidate(
            candidate=candidate,
            localization_branch=LocalizationBranch(
                localization_position_m=payload.localization_position_m,
                localization_confidence=payload.localization_confidence,
                localization_gdop=payload.localization_gdop,
                reference_sensor=reference_sensor,
                reference_signal=normalized_audio,
                classification_reference_signal=normalized_audio,
                tdoa_s={},
                localization_method=payload.localization_method,
                capability_tier=capability_tier,
            ),
            selected_sensor_ids=selected_sensor_ids,
            selected_windows={},
            selected_positions=selected_positions,
            classification_selected_windows={},
            omni_reference_sensor=reference_sensor,
            omni_reference_signal=normalized_audio,
            omni_position_m=payload.localization_position_m,
            omni_classification_reference_signal=normalized_audio,
            localization_audio_quality=rust_audio_quality,
            classification_audio_quality=rust_audio_quality,
            classification_audio_quality_source="rust_classifier_render",
            environment=dict(payload.environment),
            extra_classification_features=rust_extra_features,
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
        detection_product = await self._assemble_reporting_branch(
            product=localized_product,
            classified=classified,
            reporting_modality=payload.reporting_modality,
            localization_position_m=payload.localization_position_m,
            localization_confidence=payload.localization_confidence,
            localization_gdop=payload.localization_gdop,
            reference_sensor=reference_sensor,
            reference_signal=normalized_audio,
            tdoa_s={},
            capability_tier=capability_tier,
            localization_method=payload.localization_method,
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

        self._metrics.frames_accepted += 1
        if result.environment_counts is not None:
            self._metrics.environment_samples_ingested += result.environment_counts.ingested
            self._metrics.environment_samples_persisted += result.environment_counts.persisted

        # Update queue depth on the response before returning.
        result.response.queue_depth = self._localization_queue.qsize()

        if result.triggered:
            candidate = EventCandidate(
                id=f"evt-{next(self._candidate_counter):08d}",
                source_node_id=request.node.id,
                event_time_ns=result.event_time_ns,
                sample_rate_hz=result.sample_rate_hz,
                source_type=result.source_type,
                time_quality=result.time_quality,
                source_observation_ids=result.observation_ids or [],
            )
            if await self._enqueue_stage(self._localization_queue, candidate):
                self._ingest_processor.confirm_trigger(result.event_time_ns)
                self._metrics.triggers_enqueued += 1
                result.response.queued_event_id = candidate.id
            else:
                # Queue full — report as not triggered to the caller.
                result.response.triggered = False
                self._metrics.triggers_dropped_queue_full += 1

        return result.response

    # ------------------------------------------------------------------
    # Status & housekeeping
    # ------------------------------------------------------------------

    async def status(self) -> dict[str, Any]:
        nodes = await self.registry.list_nodes()
        realtime = self._realtime_tracker.snapshot(now_ns=time.time_ns())
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
        }

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
        while True:
            candidate = await self._localization_queue.get()
            if candidate is None:
                self._localization_queue.task_done()
                return
            self._realtime_tracker.mark_started(stage_name="localization", item_id=candidate.id)
            self._metrics.localization_stage_in += 1
            try:
                product = await self._localize_candidate(candidate)
                if product is not None:
                    if await self._enqueue_stage(self._classification_queue, product):
                        self._metrics.localization_stage_out += 1
            except Exception as exc:  # pragma: no cover - resilience path
                self._metrics.localization_failures += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
            finally:
                self._realtime_tracker.mark_finished(stage_name="localization", item_id=candidate.id)
                self._localization_queue.task_done()

    async def _classification_worker_loop(self, worker_id: int) -> None:
        del worker_id
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
                    continue
                detection_products = await self._classify_and_assemble(product)
                classification_failed = self._detection_products_had_backend_failure(detection_products)
                self._record_chunk_dispatch_outcome(
                    product=product,
                    detection_products=detection_products,
                    classification_failed=classification_failed,
                )
                if classification_failed:
                    self._metrics.classification_failures += 1
                else:
                    for detection_product in detection_products:
                        if await self._enqueue_stage(self._rules_queue, detection_product):
                            self._metrics.classification_stage_out += 1
            except Exception as exc:
                self._record_chunk_dispatch_outcome(
                    product=product,
                    detection_products=[],
                    classification_failed=True,
                )
                self._metrics.classification_failures += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
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
            finally:
                self._realtime_tracker.mark_finished(
                    stage_name="rules",
                    item_id=product.pipeline_item_id,
                )
                self._rules_queue.task_done()

    # ------------------------------------------------------------------
    # Localization stage — sensor selection, environment, localizer call
    # ------------------------------------------------------------------

    async def _localize_candidate(self, candidate: EventCandidate) -> LocalizedCandidate | None:
        sensor_positions = await self.registry.sensor_positions()
        sensor_ids = sorted(sensor_positions.keys())
        if not sensor_ids:
            return None

        windows = await self.buffer.get_synchronized_window(
            sensor_ids=sensor_ids,
            center_time_ns=candidate.event_time_ns,
            window_seconds=self.localization_config.localization_window_seconds,
            sample_rate_hz=candidate.sample_rate_hz,
        )
        min_localization_sensors = max(2, int(self.localization_config.min_sensors_for_2d))
        if len(windows) < min_localization_sensors and len(sensor_ids) >= min_localization_sensors:
            # Ingest arrives per-node; a short grace period captures sibling node frames for the same event time.
            grace_s = min(0.05, max(0.005, self.localization_config.localization_window_seconds * 0.5))
            await asyncio.sleep(grace_s)
            windows = await self.buffer.get_synchronized_window(
                sensor_ids=sensor_ids,
                center_time_ns=candidate.event_time_ns,
                window_seconds=self.localization_config.localization_window_seconds,
                sample_rate_hz=candidate.sample_rate_hz,
            )
        if not windows:
            return None

        energies = {sensor_id: rms(sig) for sensor_id, sig in windows.items()}
        threshold = self.localization_config.trigger_rms * self.fusion_config.sensor_energy_threshold_multiplier
        selected_ids = [sid for sid, energy in energies.items() if energy > threshold]
        # If thresholding under-selects, keep the strongest sensors needed for a localization attempt.
        if len(selected_ids) < min_localization_sensors and len(energies) >= min_localization_sensors:
            ranked = sorted(energies.items(), key=lambda item: item[1], reverse=True)
            selected_ids = [sensor_id for sensor_id, _ in ranked[:min_localization_sensors]]
        if len(selected_ids) < 1:
            return None

        tier = self.degradation_model.tier_for_sensor_count(len(selected_ids))
        selected_windows = {sensor_id: windows[sensor_id] for sensor_id in selected_ids}
        selected_positions = {sensor_id: sensor_positions[sensor_id] for sensor_id in selected_ids}
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

        localization_branch: LocalizationBranch | None = None
        if tier == "full_3d":
            try:
                localization = await asyncio.to_thread(
                    self.localizer.localize,
                    selected_positions,
                    localization_windows,
                    candidate.sample_rate_hz,
                    conditions.temperature_c,
                    conditions.humidity_fraction,
                )
                localization_branch = self._build_localization_branch(
                    localization=localization,
                    selected_windows=selected_windows,
                    classification_windows=classification_windows,
                    capability_tier=tier,
                )
            except LocalizationError:
                self._metrics.localization_failures += 1
        elif tier == "2d" and hasattr(self.localizer, "localize_2d"):
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
                )
                localization_branch = self._build_localization_branch(
                    localization=localization,
                    selected_windows=selected_windows,
                    classification_windows=classification_windows,
                    capability_tier=tier,
                )
            except LocalizationError:
                self._metrics.localization_failures += 1

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
        """Classify localized and omni branches, then apply reporting fusion."""
        detection_products: list[DetectionProduct] = []
        extra_features = product.extra_classification_features

        if product.localization_branch is not None:
            localized = await self._classification_orchestrator.classify(
                reference_signal=product.localization_branch.classification_reference_signal,
                sample_rate_hz=product.candidate.sample_rate_hz,
                capability_tier=product.localization_branch.capability_tier,
                selected_sensor_ids=product.selected_sensor_ids,
                selected_positions=product.selected_positions,
                selected_windows=product.classification_selected_windows,
                localization_position_m=product.localization_branch.localization_position_m,
                event_time_ns=product.candidate.event_time_ns,
            )
            if extra_features:
                localized.classification.features.update(extra_features)
            maybe_detection = await self._assemble_reporting_branch(
                product=product,
                classified=localized,
                reporting_modality="localized",
                localization_position_m=product.localization_branch.localization_position_m,
                localization_confidence=product.localization_branch.localization_confidence,
                localization_gdop=product.localization_branch.localization_gdop,
                reference_sensor=product.localization_branch.reference_sensor,
                reference_signal=product.localization_branch.reference_signal,
                tdoa_s=product.localization_branch.tdoa_s,
                capability_tier=product.localization_branch.capability_tier,
                localization_method=product.localization_branch.localization_method,
            )
            if maybe_detection is not None:
                detection_products.append(maybe_detection)

        localized_result_for_omni = localized if product.localization_branch is not None else None
        omni = await self._classify_omni_branch(
            product=product,
            localized_result=localized_result_for_omni,
        )
        if extra_features and omni is not localized_result_for_omni:
            omni.classification.features.update(extra_features)
        maybe_detection = await self._assemble_reporting_branch(
            product=product,
            classified=omni,
            reporting_modality="omni",
            localization_position_m=product.omni_position_m,
            localization_confidence=self.fusion_config.fallback_localization_confidence,
            localization_gdop=float("inf"),
            reference_sensor=product.omni_reference_sensor,
            reference_signal=product.omni_reference_signal,
            tdoa_s={},
            capability_tier="classification_only",
            localization_method="omni_reporting_window",
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

    def _build_localization_branch(
        self,
        *,
        localization,
        selected_windows: dict[str, np.ndarray],
        classification_windows: dict[str, np.ndarray],
        capability_tier: str,
    ) -> LocalizationBranch:
        reference_signal = selected_windows[localization.reference_sensor]
        localization_method = self._current_localizer_name()
        self._metrics.last_localization_algorithm = localization.resolved_algorithm or localization_method
        self._metrics.last_attempted_algorithm = localization.attempted_algorithm or localization_method
        if (
            localization.attempted_algorithm
            and localization.resolved_algorithm
            and localization.attempted_algorithm != localization.resolved_algorithm
        ):
            self._metrics.localization_fallback_count += 1
        return LocalizationBranch(
            localization_position_m=localization.position_m,
            localization_confidence=localization.confidence,
            localization_gdop=localization.gdop,
            reference_sensor=localization.reference_sensor,
            reference_signal=reference_signal,
            classification_reference_signal=classification_windows.get(
                localization.reference_sensor,
                reference_signal,
            ),
            tdoa_s=localization.tdoa_s,
            localization_method=localization_method,
            capability_tier=capability_tier,
        )

    async def _assemble_reporting_branch(
        self,
        *,
        product: LocalizedCandidate,
        classified,
        reporting_modality: str,
        localization_position_m: tuple[float, float, float],
        localization_confidence: float,
        localization_gdop: float,
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
            existing_detection.source_observation_ids = list(
                dict.fromkeys(
                    [
                        *existing_detection.source_observation_ids,
                        *product.candidate.source_observation_ids,
                    ]
                )
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
            assembly = await self._detection_assembler.assemble(
                localization_position_m=localization_position_m,
                localization_confidence=localization_confidence,
                localization_gdop=localization_gdop,
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
                classification_features=dict(classified.classification.features),
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
            return

        # Skip broadcasting unknown/0% confidence detections — they provide no
        # actionable information for the operator and clutter the live feed.
        # The DB row is still written so the reporting-window policy can track
        # it and upgrade to a real species label if one arrives later.
        if detection.label == "unknown" and detection.label_confidence == 0.0:
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
            result = await handler.handle(descriptor, detection=detection, track=track)
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_beamformer(self) -> Beamformer | None:
        """Create the classification beamformer with recall-biased settings.

        Uses the ``create_beamformer`` factory from the beamforming module,
        applying ``classifier_diagonal_loading_scale`` so that MVDR widens
        its main lobe for higher recall — preferable when the output feeds
        a classifier that benefits from capturing more target energy.
        """
        if not self.settings.beamformed_classification_enabled:
            return None
        return create_beamformer(
            beamformer_type=self.settings.beamformer_type,
            diagonal_loading=self.settings.mvdr_diagonal_loading,
            classifier_diagonal_loading_scale=self.localization_config.classifier_diagonal_loading_scale,
        )

    def _build_classification_chunking_policy(self) -> ClassificationChunkingPolicy | None:
        if not self.fusion_config.birdnet_chunked_dispatch_enabled:
            return None
        if self.settings.classifier_backend.strip().lower() != "birdnet":
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

    async def _enqueue_stage(self, queue: asyncio.Queue, item: Any) -> bool:
        tracker_stage_name = self._queue_stage_name(queue)
        tracker_item_id = self._stage_item_id(item)
        tracker_event_time_ns = self._stage_item_event_time_ns(item)
        if self.fusion_config.offline_replay_mode or not self.fusion_config.drop_on_backpressure:
            await queue.put(item)
            self._realtime_tracker.mark_enqueued(
                stage_name=tracker_stage_name,
                item_id=tracker_item_id,
                event_time_ns=tracker_event_time_ns,
            )
            return True
        try:
            queue.put_nowait(item)
            self._realtime_tracker.mark_enqueued(
                stage_name=tracker_stage_name,
                item_id=tracker_item_id,
                event_time_ns=tracker_event_time_ns,
            )
            return True
        except asyncio.QueueFull:
            self._metrics.stage_drops_backpressure += 1
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
