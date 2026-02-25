"""Fusion node runtime for staged ingest/localization/classification/rules."""

from __future__ import annotations

import asyncio
import itertools
import math
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from minimappr.classifiers.base import AudioClassifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.beamforming import DelayAndSumBeamformer, MVDRBeamformer
from minimappr.core.degradation import CapabilityDegradationModel
from minimappr.core.environment import StaticEnvironmentProvider
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization import LocalizationError
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.preprocessing import NodePreprocessorFactory
from minimappr.core.rules import ConfigRuleEngine, LoggingRuleActionHandler, WebsocketRuleActionHandler
from minimappr.core.taxonomy import RuntimeTaxonomyProvider
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.interfaces import (
    ActionDescriptor,
    AudioPreprocessor,
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
)
from minimappr.utils.audio import decode_pcm16le_b64, mono_mix, rms, write_wav_mono


@dataclass(slots=True)
class EventCandidate:
    id: str
    event_time_ns: int
    sample_rate_hz: int
    source_type: str
    time_quality: TimeQuality
    source_observation_ids: list[str]


@dataclass(slots=True)
class LocalizedCandidate:
    candidate: EventCandidate
    localization_position_m: tuple[float, float, float]
    localization_confidence: float
    localization_gdop: float
    reference_sensor: str
    tdoa_s: dict[str, float]
    selected_sensor_ids: list[str]
    selected_windows: dict[str, np.ndarray]
    selected_positions: dict[str, np.ndarray]
    reference_signal: np.ndarray
    localization_method: str
    capability_tier: str
    environment: dict[str, Any]


@dataclass(slots=True)
class DetectionProduct:
    detection: DetectionEvent
    track: TrackState | None
    suppressed_by_zone: bool
    suppression_reasons: list[str]


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


@dataclass(slots=True)
class RetentionPolicy:
    permanent_labels: set[str]
    security_long_confidence: float

    def tier_for_detection(
        self,
        *,
        label: str,
        label_category: str,
        confidence: float,
        suppressed_by_zone: bool,
    ) -> str:
        if suppressed_by_zone:
            return RetentionTier.EPHEMERAL.value
        normalized = label.strip().lower()
        if normalized in self.permanent_labels:
            return RetentionTier.PERMANENT.value
        if label_category == "security" and confidence >= self.security_long_confidence:
            return RetentionTier.LONG.value
        return RetentionTier.SHORT.value


class FusionNode:
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
        self.retention_policy = RetentionPolicy(
            permanent_labels=set(self.fusion_config.retention_permanent_labels),
            security_long_confidence=self.fusion_config.retention_long_security_confidence,
        )
        self._action_handlers = action_handlers or {
            "cop": WebsocketRuleActionHandler(live_callback),
            "log": LoggingRuleActionHandler(),
        }

        self._last_trigger_ns = 0
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
        self._last_error: str | None = None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return

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

    async def stop(self) -> None:
        if not self._started:
            return

        await self._localization_queue.join()
        await self._classification_queue.join()
        await self._rules_queue.join()

        for _ in self._localization_workers:
            await self._localization_queue.put(None)
        for _ in self._classification_workers:
            await self._classification_queue.put(None)
        for _ in self._rules_workers:
            await self._rules_queue.put(None)

        if self._localization_workers:
            await asyncio.gather(*self._localization_workers, return_exceptions=True)
        if self._classification_workers:
            await asyncio.gather(*self._classification_workers, return_exceptions=True)
        if self._rules_workers:
            await asyncio.gather(*self._rules_workers, return_exceptions=True)

        self._localization_workers.clear()
        self._classification_workers.clear()
        self._rules_workers.clear()
        self._started = False

    async def ingest(self, request: IngestFrameRequest) -> IngestFrameResponse:
        self._metrics.ingest_requests += 1
        raw_node = request.node
        frame = request.frame

        try:
            audio = decode_pcm16le_b64(frame.samples_b64, frame.channels)
        except Exception as exc:
            self._metrics.frames_rejected += 1
            raise ValueError(f"Invalid PCM payload: {exc}") from exc

        if audio.shape[0] != frame.channels:
            self._metrics.frames_rejected += 1
            raise ValueError("Decoded channel count does not match frame.channels")

        normalized_node, geo_position = self._normalize_node_spec(raw_node)
        preprocessor: AudioPreprocessor = self.preprocessor_factory.for_node(normalized_node)
        processed = np.zeros_like(audio, dtype=np.float32)
        for channel_idx in range(audio.shape[0]):
            processed[channel_idx] = preprocessor.process(
                audio[channel_idx],
                frame.sample_rate_hz,
                node_id=normalized_node.id,
            )

        runtime = await self.registry.upsert(normalized_node, frame.start_time_ns)
        tor_ns = frame.tor_ns if frame.tor_ns is not None else time.time_ns()
        environment_sample = self._extract_environment_sample(
            request=request,
            node=normalized_node,
            toa_ns=frame.toa_ns or frame.start_time_ns,
            tor_ns=tor_ns,
        )
        observation_ids: list[str] = []

        async with self._storage_batch():
            await self.storage.upsert_node(spec=normalized_node, last_seen_ns=frame.start_time_ns, position_geo=geo_position)
            if environment_sample is not None:
                self._metrics.environment_samples_ingested += 1
                await self.storage.insert_environment(
                    node_id=normalized_node.id,
                    timestamp_ns=environment_sample["timestamp_ns"],
                    temperature_c=environment_sample["temperature_c"],
                    pressure_pa=environment_sample["pressure_pa"],
                    humidity_fraction=environment_sample["humidity_fraction"],
                    wind_speed_mps=environment_sample["wind_speed_mps"],
                    wind_dir_deg=environment_sample["wind_dir_deg"],
                    solar_lux=environment_sample["solar_lux"],
                    metadata=environment_sample["metadata"],
                )
                self._metrics.environment_samples_persisted += 1
                self._update_environment_provider(environment_sample)
            for channel_index, sensor_id in enumerate(runtime.sensor_ids):
                await self.buffer.append(
                    sensor_id=sensor_id,
                    sample_rate_hz=frame.sample_rate_hz,
                    start_time_ns=frame.start_time_ns,
                    samples=processed[channel_index],
                )
                observation_id = await self.storage.insert_observation(
                    node_id=normalized_node.id,
                    sensor_id=sensor_id,
                    sensor_type="audio",
                    source_type=frame.source_type,
                    toa_ns=frame.toa_ns or frame.start_time_ns,
                    tor_ns=tor_ns,
                    time_quality=frame.time_quality.value,
                    sample_rate_hz=frame.sample_rate_hz,
                    channel_index=channel_index,
                    frame_sequence=frame.sequence,
                    metadata={
                        "frame_start_ns": frame.start_time_ns,
                        "frame_channels": frame.channels,
                        "encoding": frame.encoding,
                        "preprocess": normalized_node.properties.get("preprocess", {}),
                    },
                )
                observation_ids.append(observation_id)
                await self.registry.record_observation(sensor_id=sensor_id, observation_id=observation_id)
        self._metrics.frames_accepted += 1

        frame_energy = rms(mono_mix(processed))
        frame_duration_ns = int((processed.shape[1] / frame.sample_rate_hz) * 1_000_000_000)
        half_window_ns = int(self.localization_config.localization_window_seconds * 0.5 * 1_000_000_000)
        center_offset_ns = max(0, frame_duration_ns - half_window_ns)
        center_time_ns = frame.start_time_ns + center_offset_ns

        triggered = False
        queued_event_id: str | None = None
        cooldown_ns = int(self.localization_config.trigger_cooldown_seconds * 1_000_000_000)

        if frame_energy >= self.localization_config.trigger_rms and center_time_ns - self._last_trigger_ns >= cooldown_ns:
            candidate = EventCandidate(
                id=f"evt-{next(self._candidate_counter):08d}",
                event_time_ns=center_time_ns,
                sample_rate_hz=frame.sample_rate_hz,
                source_type=frame.source_type,
                time_quality=frame.time_quality,
                source_observation_ids=observation_ids,
            )
            if await self._enqueue_stage(self._localization_queue, candidate):
                self._last_trigger_ns = center_time_ns
                self._metrics.triggers_enqueued += 1
                triggered = True
                queued_event_id = candidate.id
            else:
                self._metrics.triggers_dropped_queue_full += 1

        return IngestFrameResponse(
            accepted=True,
            triggered=triggered,
            frame_energy=frame_energy,
            detection_id=None,
            queued_event_id=queued_event_id,
            queue_depth=self._localization_queue.qsize(),
        )

    async def status(self) -> dict[str, Any]:
        nodes = await self.registry.list_nodes()
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
            "last_trigger_ns": self._last_trigger_ns,
            "last_error": self._last_error,
            "registered_nodes": len(nodes),
            "metrics": asdict(self._metrics),
            "offline_replay_mode": self.fusion_config.offline_replay_mode,
        }

    async def housekeeping_tick(self, now_ns: int) -> None:
        await self.zone_matcher.refresh_if_due(now_ns=now_ns)
        await self._refresh_taxonomy(now_ns=now_ns)
        tracks = await self.tracker.snapshot(now_ns=now_ns)
        for track in tracks:
            track.position_geo = self.coordinate_frame.local_to_geo(track.position_m)
            await self.storage.upsert_track(track)

    async def _localization_worker_loop(self, worker_id: int) -> None:
        del worker_id
        while True:
            candidate = await self._localization_queue.get()
            if candidate is None:
                self._localization_queue.task_done()
                return
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
                self._localization_queue.task_done()

    async def _classification_worker_loop(self, worker_id: int) -> None:
        del worker_id
        while True:
            product = await self._classification_queue.get()
            if product is None:
                self._classification_queue.task_done()
                return
            self._metrics.classification_stage_in += 1
            try:
                detection_product = await self._classify_and_track(product)
                if detection_product is not None:
                    if await self._enqueue_stage(self._rules_queue, detection_product):
                        self._metrics.classification_stage_out += 1
            except Exception as exc:  # pragma: no cover - resilience path
                self._metrics.classification_failures += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
            finally:
                self._classification_queue.task_done()

    async def _rules_worker_loop(self, worker_id: int) -> None:
        del worker_id
        while True:
            product = await self._rules_queue.get()
            if product is None:
                self._rules_queue.task_done()
                return
            self._metrics.rules_stage_in += 1
            try:
                await self._process_rules_and_delivery(product)
                self._metrics.rules_stage_out += 1
            except Exception as exc:  # pragma: no cover - resilience path
                self._metrics.rules_failures += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
            finally:
                self._rules_queue.task_done()

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
        if not windows:
            return None

        energies = {sensor_id: rms(sig) for sensor_id, sig in windows.items()}
        threshold = self.localization_config.trigger_rms * self.fusion_config.sensor_energy_threshold_multiplier
        selected_ids = [sid for sid, energy in energies.items() if energy > threshold]
        # If thresholding under-selects, keep the strongest sensors needed for a localization attempt.
        min_localization_sensors = max(2, int(self.localization_config.min_sensors_for_2d))
        if len(selected_ids) < min_localization_sensors and len(energies) >= min_localization_sensors:
            ranked = sorted(energies.items(), key=lambda item: item[1], reverse=True)
            selected_ids = [sensor_id for sensor_id, _ in ranked[:min_localization_sensors]]
        if len(selected_ids) < 1:
            return None

        tier = self.degradation_model.tier_for_sensor_count(len(selected_ids))
        selected_windows = {sensor_id: windows[sensor_id] for sensor_id in selected_ids}
        selected_positions = {sensor_id: sensor_positions[sensor_id] for sensor_id in selected_ids}

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
        if tier == "full_3d":
            try:
                localization = await asyncio.to_thread(
                    self.localizer.localize,
                    selected_positions,
                    selected_windows,
                    candidate.sample_rate_hz,
                    conditions.temperature_c,
                    conditions.humidity_fraction,
                )
                reference_signal = selected_windows[localization.reference_sensor]
                localization_method = self._current_localizer_name()
                return LocalizedCandidate(
                    candidate=candidate,
                    localization_position_m=localization.position_m,
                    localization_confidence=localization.confidence,
                    localization_gdop=localization.gdop,
                    reference_sensor=localization.reference_sensor,
                    tdoa_s=localization.tdoa_s,
                    selected_sensor_ids=selected_ids,
                    selected_windows=selected_windows,
                    selected_positions=selected_positions,
                    reference_signal=reference_signal,
                    localization_method=localization_method,
                    capability_tier=tier,
                    environment=environment_summary,
                )
            except LocalizationError:
                self._metrics.localization_failures += 1
                return None

        if tier == "2d" and hasattr(self.localizer, "localize_2d"):
            try:
                mean_z = float(np.mean([pos[2] for pos in selected_positions.values()]))
                localization = await asyncio.to_thread(
                    self.localizer.localize_2d,
                    selected_positions,
                    selected_windows,
                    candidate.sample_rate_hz,
                    conditions.temperature_c,
                    conditions.humidity_fraction,
                    mean_z,
                )
                reference_signal = selected_windows[localization.reference_sensor]
                localization_method = self._current_localizer_name()
                return LocalizedCandidate(
                    candidate=candidate,
                    localization_position_m=localization.position_m,
                    localization_confidence=localization.confidence,
                    localization_gdop=localization.gdop,
                    reference_sensor=localization.reference_sensor,
                    tdoa_s=localization.tdoa_s,
                    selected_sensor_ids=selected_ids,
                    selected_windows=selected_windows,
                    selected_positions=selected_positions,
                    reference_signal=reference_signal,
                    localization_method=localization_method,
                    capability_tier=tier,
                    environment=environment_summary,
                )
            except LocalizationError:
                self._metrics.localization_failures += 1
                return None

        reference_sensor = max(energies.items(), key=lambda item: item[1])[0]
        ref_signal = windows[reference_sensor]
        ref_pos = sensor_positions[reference_sensor]
        return LocalizedCandidate(
            candidate=candidate,
            localization_position_m=(float(ref_pos[0]), float(ref_pos[1]), float(ref_pos[2])),
            localization_confidence=self.fusion_config.fallback_localization_confidence,
            localization_gdop=float("inf"),
            reference_sensor=reference_sensor,
            tdoa_s={},
            selected_sensor_ids=selected_ids,
            selected_windows=selected_windows,
            selected_positions=selected_positions,
            reference_signal=ref_signal,
            localization_method="fallback_reference_sensor",
            capability_tier=tier,
            environment=environment_summary,
        )

    async def _classify_and_track(self, product: LocalizedCandidate) -> DetectionProduct | None:
        """Classify and track a localized candidate.

        Beamforming is skipped for `alerting_only` capability tier or when there are not enough sensors.
        """
        omni_classification = await asyncio.to_thread(
            self.classifier.classify,
            product.reference_signal,
            product.candidate.sample_rate_hz,
        )
        classification = omni_classification
        classification_signal = product.reference_signal
        classification_path = "omni"
        beamformed_classification = None
        beamforming_error: str | None = None

        if (
            self.beamformer is not None
            and product.capability_tier != "alerting_only"
            and len(product.selected_sensor_ids) >= self.settings.beamformed_classification_min_sensor_count
        ):
            try:
                sound_speed = self.environment_provider.get_speed_of_sound(product.localization_position_m)
                beamformed_signal = await asyncio.to_thread(
                    self.beamformer.beamform,
                    product.selected_positions,
                    product.selected_windows,
                    product.candidate.sample_rate_hz,
                    product.localization_position_m,
                    sound_speed,
                )
                beamformed_classification = await asyncio.to_thread(
                    self.classifier.classify,
                    beamformed_signal,
                    product.candidate.sample_rate_hz,
                )
                margin = max(0.0, self.settings.beamformed_classification_confidence_margin)
                if beamformed_classification.confidence > (omni_classification.confidence + margin):
                    classification = beamformed_classification
                    classification_signal = beamformed_signal
                    classification_path = f"beamformed:{self.settings.beamformer_type}"
            except Exception as exc:  # pragma: no cover - resilience path
                beamforming_error = f"{type(exc).__name__}: {exc}"
                self._metrics.beamforming_failures += 1

        label_category = self.taxonomy_provider.category_for_label(classification.label)
        iff_category = self.taxonomy_provider.iff_for_category(label_category)
        label_id = await self.storage.upsert_label(
            name=classification.label,
            category=label_category,
            source=self.settings.classifier_backend,
            created_ns=product.candidate.event_time_ns,
        )
        if hasattr(self.taxonomy_provider, "register_label"):
            self.taxonomy_provider.register_label(classification.label, label_category)

        detection_geo = self.coordinate_frame.local_to_geo(product.localization_position_m)
        zone_ids = await self.zone_matcher.match_geo_point(
            lat=detection_geo.lat,
            lon=detection_geo.lon,
            now_ns=product.candidate.event_time_ns,
        )
        suppressed_by_zone, suppression_reasons = await self.zone_matcher.evaluate_detection_policies(
            zone_ids=zone_ids,
            label=classification.label,
            label_category=label_category,
            now_ns=product.candidate.event_time_ns,
        )

        track: TrackState | None = None
        if not suppressed_by_zone and product.capability_tier != "alerting_only":
            track = await self.tracker.update(
                timestamp_ns=product.candidate.event_time_ns,
                position_m=product.localization_position_m,
                label=classification.label,
                label_category=label_category,
                iff_category=iff_category,
                label_id=label_id,
                confidence=classification.confidence,
                sensor_count=len(product.selected_sensor_ids),
                capability_tier=product.capability_tier,
            )
            track.position_geo = self.coordinate_frame.local_to_geo(track.position_m)

        latest_observation_ids = await self.registry.latest_observation_ids(product.selected_sensor_ids)
        source_observation_ids = list(
            dict.fromkeys([*product.candidate.source_observation_ids, *latest_observation_ids])
        )
        source_node_id = await self.registry.node_id_for_sensor(product.reference_sensor)
        tor_ns = time.time_ns()
        stale_ns = product.candidate.event_time_ns + int(self.settings.event_stale_seconds * 1_000_000_000)

        gain_offset_db = await self.registry.gain_offset_db_for_sensor(product.reference_sensor)
        ref_rms = max(rms(product.reference_signal), 1e-9)
        spl_db = float(20.0 * math.log10(ref_rms)) + gain_offset_db
        retention_tier = self.retention_policy.tier_for_detection(
            label=classification.label,
            label_category=label_category,
            confidence=classification.confidence,
            suppressed_by_zone=suppressed_by_zone,
        )
        feature_summary = dict(classification.features)
        feature_summary["capability_tier"] = product.capability_tier
        feature_summary["localization_method"] = product.localization_method
        feature_summary["classification_path"] = classification_path
        feature_summary["omni_confidence"] = omni_classification.confidence
        feature_summary["environment"] = product.environment
        if beamformed_classification is not None:
            feature_summary["beamformed_confidence"] = beamformed_classification.confidence
            feature_summary["beamformed_label"] = beamformed_classification.label
        if beamforming_error:
            feature_summary["beamforming_error"] = beamforming_error
        if suppression_reasons:
            feature_summary["zone_suppression"] = suppression_reasons

        detection = DetectionEvent(
            id=f"det-{uuid.uuid4().hex[:12]}",
            source_type=product.candidate.source_type,
            source_node_id=source_node_id,
            timestamp_ns=product.candidate.event_time_ns,
            toa_ns=product.candidate.event_time_ns,
            tor_ns=tor_ns,
            time_quality=product.candidate.time_quality,
            stale_ns=stale_ns,
            position_m=product.localization_position_m,
            position_geo=detection_geo,
            position_covariance_m2=track.position_covariance_m2 if track is not None else None,
            confidence=product.localization_confidence,
            gdop=product.localization_gdop,
            label_id=label_id,
            label=classification.label,
            label_category=label_category,
            iff_category=iff_category,
            label_confidence=classification.confidence,
            spl_db=spl_db,
            track_id=track.id if track is not None and not suppressed_by_zone else None,
            source_sensors=sorted(product.selected_sensor_ids),
            source_observation_ids=source_observation_ids,
            zone_ids=zone_ids,
            reference_sensor=product.reference_sensor,
            tdoa_s=product.tdoa_s,
            classifier_scores=classification.scores,
            feature_summary=feature_summary,
            retention_tier=RetentionTier(retention_tier),
            snippet_path=None,
        )

        snippet_path: str | None = None
        snippet_expires_ns: int | None = None
        if self.settings.snippet_retention_seconds > 0 and retention_tier not in {"ephemeral", "experiment"}:
            snippet_file = Path(self.settings.snippet_dir) / f"{detection.id}.wav"
            await asyncio.to_thread(write_wav_mono, snippet_file, classification_signal, product.candidate.sample_rate_hz)
            snippet_path = str(snippet_file)
            snippet_expires_ns = product.candidate.event_time_ns + int(
                self.settings.snippet_retention_seconds * 1_000_000_000
            )
            detection.snippet_path = snippet_path

        async with self._storage_batch():
            if track is not None and not suppressed_by_zone:
                await self.storage.upsert_track(track)

            await self.storage.insert_detection(
                detection=detection,
                snippet_path=snippet_path,
                snippet_expires_ns=snippet_expires_ns,
                retention_tier=retention_tier,
            )
            await self.storage.insert_ping(
                timestamp_ns=detection.timestamp_ns,
                ping_type="acoustic",
                label=detection.label,
                label_id=detection.label_id,
                spl_db=detection.spl_db,
                position_m=detection.position_m,
                position_geo=detection.position_geo,
                source_detection_id=detection.id,
                source_observation_id=detection.source_observation_ids[0] if detection.source_observation_ids else None,
                source_track_id=detection.track_id,
                retention_tier=retention_tier,
                metadata={
                    "label_category": detection.label_category,
                    "confidence": detection.label_confidence,
                    "zone_ids": detection.zone_ids,
                    "capability_tier": product.capability_tier,
                },
            )

            if track is not None and not suppressed_by_zone:
                await self.storage.insert_track_update(
                    track=track,
                    timestamp_ns=product.candidate.event_time_ns,
                    event_id=detection.event_id,
                    update_type="detection",
                    detection_id=detection.id,
                    observation_ids=source_observation_ids,
                    metadata={
                        "gdop": detection.gdop,
                        "reference_sensor": detection.reference_sensor,
                        "capability_tier": product.capability_tier,
                    },
                )
        if suppressed_by_zone:
            self._metrics.detections_suppressed_by_zone += 1
        return DetectionProduct(
            detection=detection,
            track=track,
            suppressed_by_zone=suppressed_by_zone,
            suppression_reasons=suppression_reasons,
        )

    async def _process_rules_and_delivery(self, product: DetectionProduct) -> None:
        detection = product.detection
        track = product.track

        if product.suppressed_by_zone:
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

    def _create_beamformer(self) -> Beamformer | None:
        if not self.settings.beamformed_classification_enabled:
            return None
        beamformer_name = str(self.settings.beamformer_type).strip().lower()
        if beamformer_name == "mvdr":
            return MVDRBeamformer(diagonal_loading=self.settings.mvdr_diagonal_loading)
        return DelayAndSumBeamformer()

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
        if self.fusion_config.offline_replay_mode or not self.fusion_config.drop_on_backpressure:
            await queue.put(item)
            return True
        try:
            queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            self._metrics.stage_drops_backpressure += 1
            return False

    @asynccontextmanager
    async def _storage_batch(self):
        begin_batch = getattr(self.storage, "begin_batch", None)
        if callable(begin_batch):
            async with begin_batch():
                yield
            return
        yield

    def _extract_environment_sample(
        self,
        *,
        request: IngestFrameRequest,
        node: NodeSpec,
        toa_ns: int,
        tor_ns: int,
    ) -> dict[str, Any] | None:
        environment: EnvironmentSampleIn | None = request.environment
        timestamp_ns = toa_ns
        temperature_c: float | None = None
        humidity_fraction: float | None = None
        pressure_pa: float | None = None
        wind_speed_mps: float | None = None
        wind_dir_deg: float | None = None
        solar_lux: float | None = None
        metadata: dict[str, Any] = {}

        if environment is not None and environment.has_any_measurement():
            timestamp_ns = environment.timestamp_ns or toa_ns
            temperature_c = environment.temperature_c
            humidity_fraction = environment.humidity_fraction
            pressure_pa = environment.pressure_pa
            wind_speed_mps = environment.wind_speed_mps
            wind_dir_deg = environment.wind_dir_deg
            solar_lux = environment.solar_lux
            if environment.source:
                metadata["source"] = environment.source
            if environment.metadata:
                metadata.update(environment.metadata)
        else:
            from_metadata = self._extract_environment_from_node_metadata(node.metadata)
            if from_metadata is None:
                return None
            temperature_c = from_metadata.get("temperature_c")
            humidity_fraction = from_metadata.get("humidity_fraction")
            pressure_pa = from_metadata.get("pressure_pa")
            wind_speed_mps = from_metadata.get("wind_speed_mps")
            wind_dir_deg = from_metadata.get("wind_dir_deg")
            solar_lux = from_metadata.get("solar_lux")
            metadata = from_metadata.get("metadata", {})

        if all(
            value is None
            for value in (
                temperature_c,
                humidity_fraction,
                pressure_pa,
                wind_speed_mps,
                wind_dir_deg,
                solar_lux,
            )
        ):
            return None

        if humidity_fraction is not None:
            humidity_fraction = max(0.0, min(1.0, float(humidity_fraction)))
        metadata.setdefault("tor_ns", tor_ns)

        return {
            "node_id": node.id,
            "timestamp_ns": timestamp_ns,
            "temperature_c": temperature_c,
            "humidity_fraction": humidity_fraction,
            "pressure_pa": pressure_pa,
            "wind_speed_mps": wind_speed_mps,
            "wind_dir_deg": wind_dir_deg,
            "solar_lux": solar_lux,
            "metadata": metadata,
            "location_m": node.position_m,
        }

    def _extract_environment_from_node_metadata(self, metadata: dict[str, Any]) -> dict[str, Any] | None:
        root = metadata if isinstance(metadata, dict) else {}
        nested = root.get("environment")
        env = nested if isinstance(nested, dict) else {}

        def _value(keys: tuple[str, ...]) -> float | None:
            for key in keys:
                candidate = self._coerce_float(env.get(key))
                if candidate is not None:
                    return candidate
            for key in keys:
                candidate = self._coerce_float(root.get(key))
                if candidate is not None:
                    return candidate
            return None

        temperature_c = _value(("temperature_c", "temp_c"))
        humidity_fraction = _value(("humidity_fraction",))
        if humidity_fraction is None:
            humidity_raw = _value(("humidity", "humidity_percent"))
            if humidity_raw is not None:
                humidity_fraction = humidity_raw / 100.0 if humidity_raw > 1.0 else humidity_raw

        pressure_pa = _value(("pressure_pa",))
        wind_speed_mps = _value(("wind_speed_mps",))
        wind_dir_deg = _value(("wind_dir_deg",))
        solar_lux = _value(("solar_lux",))

        if all(
            value is None
            for value in (
                temperature_c,
                humidity_fraction,
                pressure_pa,
                wind_speed_mps,
                wind_dir_deg,
                solar_lux,
            )
        ):
            return None

        source = (
            env.get("temperature_source")
            or root.get("temperature_source")
            or env.get("source")
            or root.get("source")
        )
        out_meta = {"ingest": "node_metadata"}
        if isinstance(source, str) and source.strip():
            out_meta["source"] = source.strip()
        return {
            "temperature_c": temperature_c,
            "humidity_fraction": humidity_fraction,
            "pressure_pa": pressure_pa,
            "wind_speed_mps": wind_speed_mps,
            "wind_dir_deg": wind_dir_deg,
            "solar_lux": solar_lux,
            "metadata": out_meta,
        }

    def _update_environment_provider(self, sample: dict[str, Any]) -> None:
        ingest_fn = getattr(self.environment_provider, "ingest_sample", None)
        if not callable(ingest_fn):
            return
        ingest_fn(
            node_id=sample["node_id"],
            timestamp_ns=sample["timestamp_ns"],
            temperature_c=sample["temperature_c"],
            humidity_fraction=sample["humidity_fraction"],
            pressure_pa=sample["pressure_pa"],
            wind_speed_mps=sample["wind_speed_mps"],
            wind_dir_deg=sample["wind_dir_deg"],
            solar_lux=sample["solar_lux"],
            location_m=sample["location_m"],
            metadata=sample["metadata"],
        )

    @staticmethod
    def _sensor_centroid(
        selected_positions: dict[str, np.ndarray],
    ) -> tuple[float, float, float] | None:
        if not selected_positions:
            return None
        stacked = np.stack([np.asarray(pos, dtype=np.float64) for pos in selected_positions.values()], axis=0)
        centroid = np.mean(stacked, axis=0)
        return (float(centroid[0]), float(centroid[1]), float(centroid[2]))

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _normalize_node_spec(self, spec: NodeSpec) -> tuple[NodeSpec, GeoPoint]:
        if spec.position_m is not None:
            local_pos = spec.position_m
            geo = spec.position_geo or self.coordinate_frame.local_to_geo(local_pos)
            normalized = spec.model_copy(update={"position_m": local_pos, "position_geo": geo})
            return normalized, geo

        if spec.position_geo is None:
            raise ValueError("NodeSpec must include position_m or position_geo")

        local_pos = self.coordinate_frame.geo_to_local(spec.position_geo)
        normalized = spec.model_copy(update={"position_m": local_pos})
        return normalized, spec.position_geo
