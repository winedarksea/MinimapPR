"""Field Mode W3-entity: glue between FusionNode and the bird-entity registry.

FusionNode calls three seams, all off the ingest hot path (classification and
rules workers):

1. ``decide`` (pre-assembly, instead of ``ReportingFusionPolicy.decide``) for
   detections whose category ``applies_to``; absorbed bouts emit an entity
   event and never reach rules.
2. ``on_detection_surfaced`` (rules stage, after every existing surfacing
   filter — zone suppression, hidden/sub-display, unknown/0.0) observes the
   detection into its entity, emits the entity event and decides whether rules
   fire now, after the grace window, or not at all (already notified).
3. ``list_entities`` for ``GET /api/v1/entities``.

A maintenance task expires entities and steps them back (track expiry,
uncertainty growth, staleness). Its clock is event time advanced by wall time
since the last detection, so replayed bundles (past timestamps) age correctly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable

from minimappr.core.assembly_track_resolution import resolve_detection_track
from minimappr.core.bird_entity_model import Vec3
from minimappr.core.bird_entity_projection import bird_entity_event_payload, bird_entity_to_model
from minimappr.core.bird_entity_registry import (
    BirdEntityChange,
    BirdEntityObservation,
    BirdEntityRegistry,
)
from minimappr.core.entity_reporting_config import EntityReportingConfig
from minimappr.core.entity_reporting_policy import EntityReportingPolicy
from minimappr.core.notification_grace_window import FireRulesCallback, NotificationGraceWindow
from minimappr.core.reporting_fusion import ReportingDecision
from minimappr.models import BirdEntityListResponse, DetectionEvent, TrackState

_logger = logging.getLogger(__name__)

LiveCallback = Callable[[dict], Awaitable[None]]


class EntityReportingCoordinator:
    def __init__(
        self,
        *,
        config: EntityReportingConfig,
        storage: Any,
        node_registry: Any,
        tracker: Any,
        live_callback: LiveCallback,
        coordinate_frame_getter: Callable[[], Any],
        fire_rules: FireRulesCallback,
        canonical_label: Callable[[str], str] | None = None,
        wall_clock_ns: Callable[[], int] = time.time_ns,
        registry: BirdEntityRegistry | None = None,
    ) -> None:
        self._config = config
        self._node_registry = node_registry
        self._tracker = tracker
        self._live_callback = live_callback
        self._coordinate_frame_getter = coordinate_frame_getter
        self._canonical_label = canonical_label or (lambda label: label)
        self._wall_clock_ns = wall_clock_ns
        self.registry = registry or BirdEntityRegistry(config)
        self.policy = EntityReportingPolicy(
            storage=storage, registry=self.registry, canonical_label=self._canonical_label
        )
        self.grace_window = NotificationGraceWindow(
            grace_seconds=config.notification_grace_seconds, fire_rules=fire_rules
        )
        self._last_event_time_ns: int | None = None
        self._wall_ns_at_last_event: int = 0
        self._maintenance_task: asyncio.Task | None = None
        self.event_counts: dict[str, int] = {}

    @property
    def config(self) -> EntityReportingConfig:
        return self._config

    def applies_to(self, label_category: str | None) -> bool:
        return self._config.applies_to(label_category)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._maintenance_task is None:
            self._maintenance_task = asyncio.create_task(self._maintenance_loop(), name="entity-maintenance")

    async def stop(self) -> None:
        task, self._maintenance_task = self._maintenance_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self.grace_window.stop()

    async def _maintenance_loop(self) -> None:
        while True:
            await asyncio.sleep(self._config.maintenance_interval_seconds)
            try:
                await self.maintain_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - keep the loop alive; log loudly
                _logger.exception("entity maintenance failed")

    # ------------------------------------------------------------------
    # Seams
    # ------------------------------------------------------------------

    async def decide(
        self,
        *,
        event_time_ns: int,
        source_node_id: str | None,
        label: str,
        reporting_modality: str,
        spatial_display_mode: str,
        branch_details: dict[str, Any],
        localization_position_m: Vec3 | None,
        below_display_threshold: bool,
        localization_position_covariance_m2: list[list[float]] | None = None,
        localization_range_projection_mode: str | None = None,
    ) -> ReportingDecision:
        self._note_event_time(event_time_ns)
        node_positions = await self._node_positions()
        decision, change = await self.policy.decide(
            event_time_ns=event_time_ns,
            source_node_id=source_node_id,
            reporting_node_position_m=node_positions.get(source_node_id or ""),
            label=label,
            reporting_modality=reporting_modality,
            spatial_display_mode=spatial_display_mode,
            branch_details=branch_details,
            localization_position_m=localization_position_m,
            below_display_threshold=below_display_threshold,
            localization_position_covariance_m2=localization_position_covariance_m2,
            localization_range_projection_mode=localization_range_projection_mode,
        )
        if change is not None:
            await self._emit(change)
        return decision

    async def feed_absorbed_bout_to_tracker(self, *, entity_id: str, **resolve_kwargs: Any) -> TrackState | None:
        """An absorbed (unpositioned) bout skips assembly, but the tracker still
        gets it: W2 partial binding updates the bound track's species belief and
        nudges its position. Same decision function assembly uses."""
        if self._tracker is None:
            return None
        resolution = await resolve_detection_track(
            tracker=self._tracker,
            registry=self._node_registry,
            spatial_display_mode="node_only",
            suppressed_by_zone=False,
            **resolve_kwargs,
        )
        track = resolution.track
        entity = self.registry.get(entity_id)
        if track is not None and entity is not None and entity.track_id == track.id:
            self.registry.refresh_species_from_track(entity, track)
        return track

    async def on_detection_surfaced(
        self,
        *,
        detection: DetectionEvent,
        track: TrackState | None,
        entity_id: str | None,
    ) -> None:
        """Observe a surfaced detection; fire rules once per entity (grace-aware)."""
        self._note_event_time(detection.timestamp_ns)
        node_positions = await self._node_positions()
        observation = BirdEntityObservation(
            event_time_ns=detection.timestamp_ns,
            label=self._canonical_label(detection.label),
            confidence=float(detection.label_confidence),
            label_category=detection.label_category,
            spatial_display_mode=detection.spatial_display_mode,
            reporting_node_id=detection.source_node_id,
            reporting_node_position_m=node_positions.get(detection.source_node_id or ""),
            detection_position_m=tuple(detection.position_m),  # type: ignore[arg-type]
            detection_covariance_m2=detection.position_covariance_m2,
            detection_id=detection.id,
            track=track,
        )
        change = self.registry.observe(observation, preferred_entity_id=entity_id)
        await self._emit(change)
        entity = change.entity
        grace = self.grace_window
        if entity.is_spatial:
            if grace.is_pending(entity.entity_id):
                await grace.release_for_spatial(entity.entity_id, detection, track)
            else:
                await grace.notify_now(entity.entity_id, detection, track)
        else:
            await grace.hold_or_fire(entity.entity_id, detection, track)

    async def maintain_once(self, now_ns: int | None = None) -> list[BirdEntityChange]:
        if now_ns is None:
            if self._last_event_time_ns is None:
                return []
            now_ns = self._last_event_time_ns + (self._wall_clock_ns() - self._wall_ns_at_last_event)
        visible_tracks = None
        if self._tracker is not None:
            visible_tracks = {track.id: track for track in await self._tracker.snapshot()}
        changes = self.registry.maintain(now_ns, visible_tracks)
        for change in changes:
            if change.kind == "expired":
                self.grace_window.forget(change.entity.entity_id)
            await self._emit(change)
        self.grace_window.retain_only({entity.entity_id for entity in self.registry.entities()})
        return changes

    async def list_entities(self) -> BirdEntityListResponse:
        local_to_geo = self._local_to_geo()
        return BirdEntityListResponse(
            enabled=True,
            generated_ns=self._wall_clock_ns(),
            entities=[
                bird_entity_to_model(
                    entity,
                    local_to_geo=local_to_geo,
                    sigma_multiplier=self._config.uncertainty_sigma_multiplier,
                )
                for entity in self.registry.entities()
            ],
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _note_event_time(self, event_time_ns: int) -> None:
        if self._last_event_time_ns is None or event_time_ns >= self._last_event_time_ns:
            self._last_event_time_ns = int(event_time_ns)
            self._wall_ns_at_last_event = self._wall_clock_ns()

    async def _node_positions(self) -> dict[str, Vec3]:
        if self._node_registry is None:
            return {}
        nodes = await self._node_registry.list_nodes()
        return {
            node.spec.id: tuple(float(v) for v in node.spec.position_m[:3])  # type: ignore[misc]
            for node in nodes
            if node.spec.position_m is not None
        }

    def _local_to_geo(self):
        frame = self._coordinate_frame_getter() if self._coordinate_frame_getter else None
        return getattr(frame, "local_to_geo", None)

    async def _emit(self, change: BirdEntityChange) -> None:
        self.event_counts[change.kind] = self.event_counts.get(change.kind, 0) + 1
        payload = bird_entity_event_payload(
            change,
            local_to_geo=self._local_to_geo(),
            sigma_multiplier=self._config.uncertainty_sigma_multiplier,
            server_time_ns=self._wall_clock_ns(),
        )
        await self._live_callback(payload)


def build_entity_reporting_coordinator(
    *,
    settings: Any,
    storage: Any,
    node_registry: Any,
    tracker: Any,
    live_callback: LiveCallback,
    coordinate_frame_getter: Callable[[], Any],
    fire_rules: FireRulesCallback,
    taxonomy_provider: Any = None,
) -> EntityReportingCoordinator | None:
    """None unless ``entity_reporting_enabled`` is literally True (flag off = legacy path)."""
    if getattr(settings, "entity_reporting_enabled", False) is not True:
        return None
    resolver = getattr(taxonomy_provider, "canonical_label", None)
    return EntityReportingCoordinator(
        config=settings.entity_reporting_config(),
        storage=storage,
        node_registry=node_registry,
        tracker=tracker,
        live_callback=live_callback,
        coordinate_frame_getter=coordinate_frame_getter,
        fire_rules=fire_rules,
        canonical_label=(lambda label: str(resolver(label))) if callable(resolver) else None,
    )
