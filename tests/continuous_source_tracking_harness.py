"""Run a continuous-source scene through the sidecar manifest path (E6).

Every delivery goes through ``IngestStreamConsumer._dispatch_sse_event`` (the
SSE message path) into a real ``FusionNode`` via ``HttpIngestTransport``, so
it crosses contract parsing, the render handoff, the position-observation
seam, classification, reporting and the tracker exactly as live manifests do.
Settings are the defaults apart from storage paths and
``localization_single_node_solver="rust"``: the stored position is the
sidecar's estimate (the scene's fix, with its covariance). The Python re-solve
on this small cluster rightly calls range unobservable at fly-by distances,
which starts no track; range observability is E5's subject, not this one's.

After each delivery the harness yields to the event loop (so a queued
observation pipeline can drain) and diffs a full tracker snapshot, hidden
tracks included, to record which track the delivery updated. A delivery or
manifest failure is a ``ContinuousSourceHarnessError``, never a gate result,
so a strict xfail on the gate cannot absorb a broken harness.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minimappr.api.stream_consumer import IngestStreamConsumer, StreamConsumerConfig
from minimappr.api.transports import HttpIngestTransport
from minimappr.classifiers.heuristic import HeuristicClassifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.fusion_node import FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization import LocalizationEngine
from minimappr.core.localization_provenance import site_frame_id_for_coordinate_frame
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.ingest_sidecar_runtime import (
    sidecar_classification_window_seconds,
    sidecar_classifier_render_min_interval_seconds,
)
from minimappr.models import GeoPoint, TrackState
from minimappr.sim.continuous_source_scenes import ContinuousSourceScene, SceneDelivery, SceneTiming
from minimappr.sim.continuous_track_update_metrics import TrackUpdateObservation
from minimappr.storage.db import Storage
from tests.continuous_source_manifests import SCENE_START_NS, delivery_manifest

SITE_ORIGIN = GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0)
UNUSED_SIDECAR_BASE_URL = "http://127.0.0.1:9"
# Event-loop turns after each delivery: enough for an in-loop observation
# queue to drain one item through the tracker lock.
PIPELINE_SETTLE_EVENT_LOOP_TURNS = 50
STREAM_CONSUMER_LOGGER_NAME = "minimappr.api.stream_consumer"
SCENE_SINGLE_NODE_SOLVER = "rust"


class ContinuousSourceHarnessError(RuntimeError):
    """The harness itself failed (delivery raised, manifest rejected, records
    inconsistent); never a pipeline verdict."""


def default_scene_timing() -> SceneTiming:
    """Scene cadence from Settings defaults: one event solve per trigger
    cooldown, one covering render per sidecar classifier-render interval."""
    settings = Settings()
    classification_window_seconds = sidecar_classification_window_seconds(settings)
    return SceneTiming(
        solve_interval_seconds=settings.trigger_cooldown_seconds,
        render_interval_seconds=sidecar_classifier_render_min_interval_seconds(
            settings, classification_window_seconds=classification_window_seconds
        ),
    )


@dataclass(frozen=True, slots=True)
class TrackSnapshotEntry:
    """Value copy of a track: the tracker mutates its states in place."""

    track_id: str
    update_count: int
    last_seen_ns: int
    label: str
    label_category: str
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]

    @classmethod
    def from_track(cls, track: TrackState) -> "TrackSnapshotEntry":
        return cls(
            track_id=track.id,
            update_count=int(track.update_count),
            last_seen_ns=int(track.last_seen_ns),
            label=track.label,
            label_category=track.label_category,
            position_m=tuple(float(value) for value in track.position_m),
            velocity_mps=tuple(float(value) for value in track.velocity_mps),
        )


@dataclass(slots=True)
class SceneRunRecord:
    scene: ContinuousSourceScene
    scene_start_ns: int
    observations: list[TrackUpdateObservation] = field(default_factory=list)
    final_tracks: dict[str, TrackSnapshotEntry] = field(default_factory=dict)
    visible_track_ids: set[str] = field(default_factory=set)
    # Track state right after each delivery, by delivery index.
    tracks_after_delivery: list[dict[str, TrackSnapshotEntry]] = field(default_factory=list)

    def tracks_updated_by_solve(self, solve_id: str) -> list[str]:
        return list(dict.fromkeys(obs.track_id for obs in self.observations if obs.solve_id == solve_id))

    def track_for_source(self, source_id: str) -> TrackSnapshotEntry | None:
        """The final state of the track most of the source's solves updated."""
        counts: dict[str, int] = {}
        for observation in self.observations:
            if self.scene.solve(observation.solve_id).source_id == source_id:
                counts[observation.track_id] = counts.get(observation.track_id, 0) + 1
        if not counts:
            return None
        return self.final_tracks.get(max(counts, key=counts.get))


class _FailureRecordingTransport:
    """Forwards every transport call (including ones added later) and records
    exceptions the stream consumer would otherwise log and swallow."""

    def __init__(self, inner: HttpIngestTransport) -> None:
        self._inner = inner
        self.failures: list[str] = []

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._inner, name)
        if not inspect.iscoroutinefunction(attribute):
            return attribute

        async def recording_call(*args, **kwargs):
            try:
                return await attribute(*args, **kwargs)
            except Exception as exc:
                self.failures.append(f"{name}: {type(exc).__name__}: {exc}")
                raise

        return recording_call


class _WarningCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


async def started_scene_fusion(tmp_path: Path, **settings_overrides: Any) -> tuple[FusionNode, Storage]:
    settings_fields = {
        "db_path": tmp_path / "continuous_source.db",
        "snippet_dir": tmp_path / "snippets",
        "snippet_retention_seconds": 0,
        "localization_single_node_solver": SCENE_SINGLE_NODE_SOLVER,
        **settings_overrides,
    }
    settings = Settings(**settings_fields)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)
    storage = Storage(settings.db_path)
    await storage.initialize()
    fusion = FusionNode(
        settings=settings,
        registry=NodeRegistry(),
        buffer=MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds),
        localizer=LocalizationEngine(max_tau_s=0.03),
        classifier=HeuristicClassifier(),
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=lambda payload: asyncio.sleep(0, result=None),
        coordinate_frame=LocalCoordinateFrame(origin=SITE_ORIGIN, mode="flat"),
        zone_matcher=ZoneMatcher(storage=storage),
    )
    await fusion.start()
    return fusion, storage


async def _tracker_state(fusion: FusionNode, *, include_hidden: bool) -> dict[str, TrackSnapshotEntry]:
    tracks = await fusion.tracker.snapshot(include_hidden=include_hidden)
    return {track.id: TrackSnapshotEntry.from_track(track) for track in tracks}


async def _settle_pipeline() -> None:
    for _ in range(PIPELINE_SETTLE_EVENT_LOOP_TURNS):
        await asyncio.sleep(0)


async def deliver_through_stream_consumer(
    fusion: FusionNode,
    scene: ContinuousSourceScene,
    deliveries: list[SceneDelivery] | None = None,
    *,
    scene_start_ns: int = SCENE_START_NS,
) -> SceneRunRecord:
    transport = _FailureRecordingTransport(HttpIngestTransport(fusion))
    consumer = IngestStreamConsumer(
        config=StreamConsumerConfig(sidecar_base_url=UNUSED_SIDECAR_BASE_URL),
        ingest_transport=transport,
        coordinate_frame=fusion.coordinate_frame,
    )
    frame_id = site_frame_id_for_coordinate_frame(fusion.coordinate_frame)
    warnings = _WarningCollector()
    consumer_logger = logging.getLogger(STREAM_CONSUMER_LOGGER_NAME)
    consumer_logger.addHandler(warnings)
    record = SceneRunRecord(scene=scene, scene_start_ns=scene_start_ns)
    previous = await _tracker_state(fusion, include_hidden=True)
    try:
        for delivery_index, delivery in enumerate(scene.deliveries if deliveries is None else deliveries):
            manifest = delivery_manifest(
                scene, delivery, delivery_index=delivery_index, frame_id=frame_id, scene_start_ns=scene_start_ns
            )
            await consumer._dispatch_sse_event(
                event_type="message", event_id=str(delivery_index), data_lines=[json.dumps(manifest)]
            )
            await _settle_pipeline()
            current = await _tracker_state(fusion, include_hidden=True)
            for track_id, entry in current.items():
                before = previous.get(track_id)
                increase = entry.update_count - (before.update_count if before is not None else 0)
                if increase > 0:
                    record.observations.append(
                        TrackUpdateObservation(
                            delivery_index=delivery_index,
                            delivery_kind=delivery.kind,
                            solve_id=delivery.solve_id,
                            track_id=track_id,
                            update_count_increase=increase,
                            track_last_seen_ns=entry.last_seen_ns,
                        )
                    )
            record.tracks_after_delivery.append(current)
            previous = current
    finally:
        consumer_logger.removeHandler(warnings)
    if transport.failures or warnings.messages:
        raise ContinuousSourceHarnessError(
            f"delivery failures {transport.failures}; stream consumer warnings {warnings.messages}"
        )
    await _settle_pipeline()
    record.final_tracks = await _tracker_state(fusion, include_hidden=True)
    record.visible_track_ids = set(await _tracker_state(fusion, include_hidden=False))
    if record.final_tracks != previous:
        raise ContinuousSourceHarnessError(
            "tracker changed after the last delivery settled: the observation pipeline lags the per-delivery "
            "snapshot, so updates were attributed to the wrong solve (raise PIPELINE_SETTLE_EVENT_LOOP_TURNS)"
        )
    return record


async def run_scene(tmp_path: Path, scene: ContinuousSourceScene, **settings_overrides: Any) -> SceneRunRecord:
    fusion, storage = await started_scene_fusion(tmp_path, **settings_overrides)
    try:
        return await deliver_through_stream_consumer(fusion, scene)
    finally:
        await fusion.stop()
        await storage.close()
