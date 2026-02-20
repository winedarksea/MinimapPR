"""Fusion node runtime for staged ingest/localization/classification/tracking."""

from __future__ import annotations

import asyncio
import itertools
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from minimappr.classifiers.base import AudioClassifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.localization import LocalizationEngine, LocalizationError
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.models import DetectionEvent, IngestFrameRequest, IngestFrameResponse
from minimappr.storage.db import Storage
from minimappr.utils.audio import decode_pcm16le_b64, mono_mix, rms, write_wav_mono


@dataclass(slots=True)
class EventCandidate:
    id: str
    event_time_ns: int
    sample_rate_hz: int


@dataclass(slots=True)
class FusionMetrics:
    ingest_requests: int = 0
    frames_accepted: int = 0
    frames_rejected: int = 0
    triggers_enqueued: int = 0
    triggers_dropped_queue_full: int = 0
    events_dequeued: int = 0
    events_processed: int = 0
    events_failed: int = 0
    localization_failures: int = 0
    detections_emitted: int = 0


class FusionNode:
    def __init__(
        self,
        settings: Settings,
        registry: NodeRegistry,
        buffer: MultiSensorBuffer,
        localizer: LocalizationEngine,
        classifier: AudioClassifier,
        tracker: TrackManager,
        storage: Storage,
        live_callback,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.buffer = buffer
        self.localizer = localizer
        self.classifier = classifier
        self.tracker = tracker
        self.storage = storage
        self.live_callback = live_callback

        self._last_trigger_ns = 0
        self._candidate_counter = itertools.count(1)

        queue_size = max(1, settings.fusion_event_queue_size)
        self._queue: asyncio.Queue[EventCandidate | None] = asyncio.Queue(maxsize=queue_size)
        self._workers: list[asyncio.Task] = []
        self._metrics = FusionMetrics()
        self._last_error: str | None = None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return

        worker_count = max(1, self.settings.fusion_worker_count)
        self._workers = [
            asyncio.create_task(self._worker_loop(worker_id=index), name=f"fusion-worker-{index}")
            for index in range(worker_count)
        ]
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return

        await self._queue.join()

        for _ in self._workers:
            await self._queue.put(None)

        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)

        self._workers.clear()
        self._started = False

    async def ingest(self, request: IngestFrameRequest) -> IngestFrameResponse:
        self._metrics.ingest_requests += 1

        node = request.node
        frame = request.frame

        try:
            audio = decode_pcm16le_b64(frame.samples_b64, frame.channels)
        except Exception as exc:
            self._metrics.frames_rejected += 1
            raise ValueError(f"Invalid PCM payload: {exc}") from exc

        if audio.shape[0] != frame.channels:
            self._metrics.frames_rejected += 1
            raise ValueError("Decoded channel count does not match frame.channels")

        runtime = await self.registry.upsert(node, frame.start_time_ns)
        for channel_index, sensor_id in enumerate(runtime.sensor_ids):
            await self.buffer.append(
                sensor_id=sensor_id,
                sample_rate_hz=frame.sample_rate_hz,
                start_time_ns=frame.start_time_ns,
                samples=audio[channel_index],
            )

        await self.storage.upsert_node(spec=node, last_seen_ns=frame.start_time_ns)

        self._metrics.frames_accepted += 1

        frame_energy = rms(mono_mix(audio))
        frame_duration_ns = int((audio.shape[1] / frame.sample_rate_hz) * 1_000_000_000)
        half_window_ns = int(self.settings.localization_window_seconds * 0.5 * 1_000_000_000)

        # Bias event center slightly earlier than frame midpoint so the requested
        # localization window is available immediately after appending this frame.
        center_offset_ns = max(0, frame_duration_ns - half_window_ns)
        center_time_ns = frame.start_time_ns + center_offset_ns

        triggered = False
        queued_event_id: str | None = None
        cooldown_ns = int(self.settings.trigger_cooldown_seconds * 1_000_000_000)

        if frame_energy >= self.settings.trigger_rms and center_time_ns - self._last_trigger_ns >= cooldown_ns:
            candidate = EventCandidate(
                id=f"evt-{next(self._candidate_counter):08d}",
                event_time_ns=center_time_ns,
                sample_rate_hz=frame.sample_rate_hz,
            )
            try:
                self._queue.put_nowait(candidate)
                self._last_trigger_ns = center_time_ns
                self._metrics.triggers_enqueued += 1
                triggered = True
                queued_event_id = candidate.id
            except asyncio.QueueFull:
                self._metrics.triggers_dropped_queue_full += 1

        return IngestFrameResponse(
            accepted=True,
            triggered=triggered,
            frame_energy=frame_energy,
            detection_id=None,
            queued_event_id=queued_event_id,
            queue_depth=self._queue.qsize(),
        )

    async def status(self) -> dict:
        nodes = await self.registry.list_nodes()
        return {
            "started": self._started,
            "queue": {
                "depth": self._queue.qsize(),
                "max_depth": self._queue.maxsize,
            },
            "workers": {
                "configured": max(1, self.settings.fusion_worker_count),
                "running": sum(1 for task in self._workers if not task.done()),
            },
            "last_trigger_ns": self._last_trigger_ns,
            "last_error": self._last_error,
            "registered_nodes": len(nodes),
            "metrics": asdict(self._metrics),
        }

    async def housekeeping_tick(self, now_ns: int) -> None:
        tracks = await self.tracker.snapshot(now_ns=now_ns)
        for track in tracks:
            await self.storage.upsert_track(track)

    async def _worker_loop(self, worker_id: int) -> None:
        del worker_id
        while True:
            candidate = await self._queue.get()
            if candidate is None:
                self._queue.task_done()
                return

            self._metrics.events_dequeued += 1

            try:
                detection = await self._process_candidate(candidate)
                self._metrics.events_processed += 1
                if detection is not None:
                    self._metrics.detections_emitted += 1
            except Exception as exc:  # pragma: no cover - resilience path
                self._metrics.events_failed += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
            finally:
                self._queue.task_done()

    async def _process_candidate(self, candidate: EventCandidate) -> DetectionEvent | None:
        sensor_positions = await self.registry.sensor_positions()
        sensor_ids = sorted(sensor_positions.keys())
        if len(sensor_ids) < 4:
            return None

        windows = await self.buffer.get_synchronized_window(
            sensor_ids=sensor_ids,
            center_time_ns=candidate.event_time_ns,
            window_seconds=self.settings.localization_window_seconds,
            sample_rate_hz=candidate.sample_rate_hz,
        )
        if len(windows) < 4:
            return None

        energies = {sensor_id: rms(sig) for sensor_id, sig in windows.items()}
        selected_ids = [sensor_id for sensor_id, energy in energies.items() if energy > (self.settings.trigger_rms * 0.45)]
        if len(selected_ids) < 4:
            selected_ids = [sid for sid, _ in sorted(energies.items(), key=lambda item: item[1], reverse=True)[:4]]

        selected_windows = {sensor_id: windows[sensor_id] for sensor_id in selected_ids}
        selected_positions = {sensor_id: sensor_positions[sensor_id] for sensor_id in selected_ids}

        try:
            localization = await asyncio.to_thread(
                self.localizer.localize,
                selected_positions,
                selected_windows,
                candidate.sample_rate_hz,
                self.settings.default_temperature_c,
                self.settings.default_humidity,
            )
        except LocalizationError:
            self._metrics.localization_failures += 1
            return None

        reference_signal = selected_windows[localization.reference_sensor]
        classification = await asyncio.to_thread(self.classifier.classify, reference_signal, candidate.sample_rate_hz)

        track = await self.tracker.update(
            timestamp_ns=candidate.event_time_ns,
            position_m=localization.position_m,
            label=classification.label,
            confidence=classification.confidence,
            sensor_count=len(selected_ids),
        )

        detection = DetectionEvent(
            id=f"det-{uuid.uuid4().hex[:12]}",
            timestamp_ns=candidate.event_time_ns,
            position_m=localization.position_m,
            confidence=localization.confidence,
            gdop=localization.gdop,
            label=classification.label,
            label_confidence=classification.confidence,
            track_id=track.id,
            source_sensors=sorted(selected_ids),
            reference_sensor=localization.reference_sensor,
            tdoa_s=localization.tdoa_s,
            classifier_scores=classification.scores,
            feature_summary=classification.features,
            snippet_path=None,
        )

        snippet_path: str | None = None
        snippet_expires_ns: int | None = None
        if self.settings.snippet_retention_seconds > 0:
            snippet_file = Path(self.settings.snippet_dir) / f"{detection.id}.wav"
            await asyncio.to_thread(write_wav_mono, snippet_file, reference_signal, candidate.sample_rate_hz)
            snippet_path = str(snippet_file)
            snippet_expires_ns = candidate.event_time_ns + int(self.settings.snippet_retention_seconds * 1_000_000_000)
            detection.snippet_path = snippet_path

        await self.storage.insert_detection(
            detection=detection,
            snippet_path=snippet_path,
            snippet_expires_ns=snippet_expires_ns,
        )
        await self.storage.upsert_track(track)

        payload = {
            "type": "detection",
            "detection": detection.model_dump(),
            "track": track.model_dump(),
            "server_time_ns": time.time_ns(),
        }
        await self.live_callback(payload)
        return detection
