"""Field Mode W3-entity: in-memory bird-entity registry (the state machine).

Sits on top of the tracker: tracks, species belief (``species_belief`` /
``species_top_label``) and partial binding decide *which bird* a detection is;
the registry turns that into what a user sees: one entity per bird that moves
heard_nearby -> direction_known -> pinpointed, with a sliding per-entity
absorption window replacing ``ReportingFusionPolicy``'s fixed buckets.

Synchronous and IO-free. Callers (``EntityReportingCoordinator``, the scenario
harness) resolve node positions and run it on one event loop, so no lock is
needed. Only detections that already passed the surfacing filters reach
``observe`` — hidden tentative tracks (W2 two-threshold) never create entities.
"""

from __future__ import annotations

import dataclasses
import uuid
from dataclasses import dataclass
from typing import Callable, Literal

from minimappr.core.bird_entity_absorption_geometry import concurrent_fix_conflict, remember_fix
from minimappr.core.bird_entity_model import (
    ENTITY_STATE_DIRECTION_KNOWN,
    ENTITY_STATE_HEARD_NEARBY,
    ENTITY_STATE_ONE_STEP_BACK,
    ENTITY_STATE_PINPOINTED,
    ENTITY_STATE_RANK,
    BirdEntityBearingEstimate,
    BirdEntityRecord,
    BirdEntitySpatialState,
    EntityFixSample,
    Vec3,
    bearing_estimate_toward,
    horizontal_major_sigma_m,
)
from minimappr.core.bird_entity_matching import (
    SPATIAL_DISPLAY_BEARING_ONLY,
    SPATIAL_DISPLAY_LOCALIZED,
    spatial_mismatch_score,
)
from minimappr.core.entity_reporting_config import EntityReportingConfig
from minimappr.models import TrackState, TrackStatus

NANOSECONDS_PER_SECOND = 1_000_000_000

BirdEntityChangeKind = Literal["created", "updated", "absorbed", "state_changed", "expired"]


@dataclass(frozen=True, slots=True)
class BirdEntityObservation:
    event_time_ns: int
    label: str
    confidence: float
    label_category: str
    spatial_display_mode: str
    reporting_node_id: str | None
    reporting_node_position_m: Vec3 | None
    detection_position_m: Vec3 | None = None
    detection_covariance_m2: list[list[float]] | None = None
    detection_id: str | None = None
    track: TrackState | None = None


@dataclass(frozen=True, slots=True)
class BirdEntityChange:
    kind: BirdEntityChangeKind
    entity: BirdEntityRecord
    previous_state: BirdEntitySpatialState | None = None


@dataclass(frozen=True, slots=True)
class _ObservedSpatialEstimate:
    state: BirdEntitySpatialState
    position_m: Vec3 | None = None
    covariance_m2: list[list[float]] | None = None
    horizontal_uncertainty_m: float | None = None
    bearing: BirdEntityBearingEstimate | None = None


def _default_entity_id() -> str:
    return f"ent-{uuid.uuid4().hex[:12]}"


class BirdEntityRegistry:
    def __init__(
        self,
        config: EntityReportingConfig,
        *,
        entity_id_factory: Callable[[], str] = _default_entity_id,
    ) -> None:
        self._config = config
        self._entity_id_factory = entity_id_factory
        self._entities: dict[str, BirdEntityRecord] = {}
        self._absorption_window_ns = int(config.absorption_window_seconds * NANOSECONDS_PER_SECOND)
        self._expiry_ns = int(config.expiry_seconds * NANOSECONDS_PER_SECOND)
        self._spatial_hold_ns = int(config.spatial_state_hold_seconds * NANOSECONDS_PER_SECOND)

    @property
    def config(self) -> EntityReportingConfig:
        return self._config

    def entities(self) -> list[BirdEntityRecord]:
        return sorted(self._entities.values(), key=lambda entity: entity.last_heard_ns, reverse=True)

    def get(self, entity_id: str | None) -> BirdEntityRecord | None:
        return self._entities.get(entity_id) if entity_id else None

    def entity_for_track(self, track_id: str | None) -> BirdEntityRecord | None:
        if not track_id:
            return None
        return next((entity for entity in self._entities.values() if entity.track_id == track_id), None)

    # ------------------------------------------------------------------
    # Matching (the sliding absorption window)
    # ------------------------------------------------------------------

    def find_absorbing_entity(
        self,
        *,
        label: str,
        event_time_ns: int,
        spatial_display_mode: str,
        reporting_node_id: str | None,
        reporting_node_position_m: Vec3 | None,
        detection_position_m: Vec3 | None,
        track_id: str | None = None,
        detection_covariance_m2: list[list[float]] | None = None,
    ) -> BirdEntityRecord | None:
        """The entity this detection belongs to, or None (a new entity).

        A track already bound to an entity wins outright (the tracker has said
        it is the same bird). Otherwise: same species (top label, or a belief
        of at least ``absorption_min_species_belief``), last heard within the
        window, spatially consistent (covariance-gated for fixes), and not
        contradicted by a concurrent fix of the entity; the lowest normalised
        mismatch wins.
        """
        bound = self.entity_for_track(track_id)
        if bound is not None:
            return bound
        fix_sample = self._fix_sample(event_time_ns, spatial_display_mode, detection_position_m, detection_covariance_m2)
        best_entity: BirdEntityRecord | None = None
        best_score = float("inf")
        for entity in self._entities.values():
            if not entity.believes_label(label, self._config.absorption_min_species_belief):
                continue
            if event_time_ns - entity.last_heard_ns > self._absorption_window_ns:
                continue
            if concurrent_fix_conflict(entity, fix_sample, self._config):
                continue
            score = spatial_mismatch_score(
                entity,
                self._config,
                spatial_display_mode=spatial_display_mode,
                reporting_node_id=reporting_node_id,
                reporting_node_position_m=reporting_node_position_m,
                detection_position_m=detection_position_m,
                detection_covariance_m2=detection_covariance_m2,
            )
            if score is not None and score < best_score:
                best_entity, best_score = entity, score
        return best_entity

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def absorb_unlocalized(
        self,
        entity_id: str,
        *,
        event_time_ns: int,
        label: str,
        confidence: float,
        reporting_node_id: str | None,
    ) -> BirdEntityChange | None:
        """A bout that failed to localize joins its entity: recency + pulse, no demotion."""
        entity = self._entities.get(entity_id)
        if entity is None:
            return None
        entity.last_heard_ns = max(entity.last_heard_ns, event_time_ns)
        entity.last_pulse_ns = max(entity.last_pulse_ns, event_time_ns)
        entity.absorbed_detection_count += 1
        self._note_reporting_node(entity, reporting_node_id)
        if entity.track_id is None:
            self._merge_detection_belief(entity, label, confidence)
        return BirdEntityChange(kind="absorbed", entity=entity, previous_state=entity.state)

    def observe(
        self, observation: BirdEntityObservation, *, preferred_entity_id: str | None = None
    ) -> BirdEntityChange:
        track = observation.track
        fix_sample = self._fix_sample(
            observation.event_time_ns,
            observation.spatial_display_mode,
            observation.detection_position_m,
            observation.detection_covariance_m2,
        )
        entity = self.get(preferred_entity_id)
        if entity is None:
            entity, refused_concurrently = self._entity_for_track_if_same_species(observation, fix_sample)
            if refused_concurrently:
                # The track carries another bird: this one stands on its own fix.
                observation = dataclasses.replace(observation, track=None)
                track = None
        if entity is None:
            entity = self.find_absorbing_entity(
                label=observation.label,
                event_time_ns=observation.event_time_ns,
                spatial_display_mode=observation.spatial_display_mode,
                reporting_node_id=observation.reporting_node_id,
                reporting_node_position_m=observation.reporting_node_position_m,
                detection_position_m=observation.detection_position_m,
                detection_covariance_m2=observation.detection_covariance_m2,
            )
        estimate = self._observed_estimate(observation)
        if entity is None:
            created = self._create(observation, estimate)
            if fix_sample is not None:
                remember_fix(created, fix_sample, self._config)
            return BirdEntityChange(kind="created", entity=created)
        if fix_sample is not None:
            remember_fix(entity, fix_sample, self._config)
        previous_state = entity.state
        entity.last_heard_ns = max(entity.last_heard_ns, observation.event_time_ns)
        entity.last_pulse_ns = max(entity.last_pulse_ns, observation.event_time_ns)
        entity.detection_count += 1
        entity.reporting_node_id = observation.reporting_node_id or entity.reporting_node_id
        entity.reporting_node_position_m = observation.reporting_node_position_m or entity.reporting_node_position_m
        self._note_reporting_node(entity, observation.reporting_node_id)
        if track is not None:
            self._bind_track(entity, track.id)
        if entity.canonical_detection_id is None:
            entity.canonical_detection_id = observation.detection_id
        self._apply_species(entity, observation)
        if ENTITY_STATE_RANK[estimate.state] >= ENTITY_STATE_RANK[entity.state]:
            self._apply_estimate(entity, estimate, observation.event_time_ns)
        if entity.state != previous_state:
            entity.last_state_change_ns = observation.event_time_ns
            return BirdEntityChange(kind="state_changed", entity=entity, previous_state=previous_state)
        return BirdEntityChange(kind="updated", entity=entity, previous_state=previous_state)

    def _entity_for_track_if_same_species(
        self, observation: BirdEntityObservation, fix_sample: EntityFixSample | None
    ) -> tuple[BirdEntityRecord | None, bool]:
        """(the track's entity or None, refused-for-a-concurrent-fix).

        A track that merged two birds (legacy CV tracker across species, any
        tracker across a flock) must not drag the entity: follow it when the
        entity believes the label or the log-odds track's top species (at the
        floor), never when a moments-old entity fix contradicts this one.
        """
        track = observation.track
        entity = self.entity_for_track(track.id if track else None)
        if entity is None:
            return None, False
        if concurrent_fix_conflict(entity, fix_sample, self._config):
            return None, True
        floor = self._config.absorption_min_species_belief
        if entity.believes_label(observation.label, floor):
            return entity, False
        if track.species_belief and entity.believes_label(track.species_top_label or track.label, floor):
            return entity, False
        return None, False

    @staticmethod
    def _fix_sample(
        event_time_ns: int,
        spatial_display_mode: str,
        position_m: Vec3 | None,
        covariance_m2: list[list[float]] | None,
    ) -> EntityFixSample | None:
        if spatial_display_mode != SPATIAL_DISPLAY_LOCALIZED or position_m is None:
            return None
        return EntityFixSample(
            event_time_ns=int(event_time_ns),
            position_m=tuple(float(v) for v in position_m[:3]),  # type: ignore[arg-type]
            covariance_m2=covariance_m2,
        )

    def maintain(
        self, now_ns: int, visible_tracks_by_id: dict[str, TrackState] | None
    ) -> list[BirdEntityChange]:
        """Expiry and step-back. *visible_tracks_by_id* None = tracker unknown (skip track checks)."""
        changes: list[BirdEntityChange] = []
        for entity_id, entity in list(self._entities.items()):
            if now_ns - entity.last_heard_ns > self._expiry_ns:
                del self._entities[entity_id]
                changes.append(BirdEntityChange(kind="expired", entity=entity, previous_state=entity.state))
                continue
            previous_state = entity.state
            if entity.track_id is not None and visible_tracks_by_id is not None:
                track = visible_tracks_by_id.get(entity.track_id)
                if track is None or track.status == TrackStatus.DROPPED.value:
                    entity.track_id = None
                    self._step_back(entity)
                else:
                    self._refresh_from_track(entity, track)
            hold_reference_ns = max(entity.last_spatial_update_ns or 0, entity.last_state_change_ns)
            if entity.is_spatial and now_ns - hold_reference_ns > self._spatial_hold_ns:
                self._step_back(entity)
            if entity.state != previous_state:
                entity.last_state_change_ns = now_ns
                changes.append(BirdEntityChange(kind="state_changed", entity=entity, previous_state=previous_state))
        return changes

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _create(self, observation: BirdEntityObservation, estimate: _ObservedSpatialEstimate) -> BirdEntityRecord:
        if len(self._entities) >= self._config.max_tracked_entities:
            oldest = min(self._entities.values(), key=lambda entity: entity.last_heard_ns)
            del self._entities[oldest.entity_id]
        entity = BirdEntityRecord(
            entity_id=self._entity_id_factory(),
            state=ENTITY_STATE_HEARD_NEARBY,
            label_category=observation.label_category,
            species_top_label=observation.label,
            species_top_probability=float(observation.confidence),
            species_belief={observation.label: float(observation.confidence)},
            first_heard_ns=observation.event_time_ns,
            last_heard_ns=observation.event_time_ns,
            reporting_node_id=observation.reporting_node_id,
            reporting_node_position_m=observation.reporting_node_position_m,
            canonical_detection_id=observation.detection_id,
            last_state_change_ns=observation.event_time_ns,
            detection_count=1,
            last_pulse_ns=observation.event_time_ns,
        )
        self._note_reporting_node(entity, observation.reporting_node_id)
        if observation.track is not None:
            self._bind_track(entity, observation.track.id)
        self._apply_species(entity, observation)
        self._apply_estimate(entity, estimate, observation.event_time_ns)
        self._entities[entity.entity_id] = entity
        return entity

    def _bind_track(self, entity: BirdEntityRecord, track_id: str) -> None:
        # A track belongs to one entity; a stale binding elsewhere is released.
        previous_owner = self.entity_for_track(track_id)
        if previous_owner is not None and previous_owner is not entity:
            previous_owner.track_id = None
        entity.track_id = track_id

    def _observed_estimate(self, observation: BirdEntityObservation) -> _ObservedSpatialEstimate:
        track = observation.track
        if track is not None:
            return self._estimate_from_position(
                observation, tuple(track.position_m), track.position_covariance_m2  # type: ignore[arg-type]
            )
        position_m = observation.detection_position_m
        if observation.spatial_display_mode == SPATIAL_DISPLAY_LOCALIZED and position_m is not None:
            return self._estimate_from_position(observation, position_m, observation.detection_covariance_m2)
        if (
            observation.spatial_display_mode == SPATIAL_DISPLAY_BEARING_ONLY
            and position_m is not None
            and observation.reporting_node_position_m is not None
        ):
            return _ObservedSpatialEstimate(
                state=ENTITY_STATE_DIRECTION_KNOWN,
                position_m=position_m,
                covariance_m2=observation.detection_covariance_m2,
                bearing=self._bearing(
                    observation.reporting_node_id,
                    observation.reporting_node_position_m,
                    position_m,
                    observation.detection_covariance_m2,
                    range_observable=False,
                ),
            )
        return _ObservedSpatialEstimate(state=ENTITY_STATE_HEARD_NEARBY)

    def _estimate_from_position(
        self,
        observation: BirdEntityObservation,
        position_m: Vec3,
        covariance_m2: list[list[float]] | None,
    ) -> _ObservedSpatialEstimate:
        sigma_m = horizontal_major_sigma_m(covariance_m2)
        node_position_m = observation.reporting_node_position_m
        too_uncertain = sigma_m is not None and sigma_m > self._config.pinpointed_max_horizontal_uncertainty_m
        if too_uncertain and node_position_m is not None:
            return _ObservedSpatialEstimate(
                state=ENTITY_STATE_DIRECTION_KNOWN,
                position_m=position_m,
                covariance_m2=covariance_m2,
                horizontal_uncertainty_m=sigma_m,
                bearing=self._bearing(
                    observation.reporting_node_id, node_position_m, position_m, covariance_m2, range_observable=True
                ),
            )
        return _ObservedSpatialEstimate(
            state=ENTITY_STATE_PINPOINTED,
            position_m=position_m,
            covariance_m2=covariance_m2,
            horizontal_uncertainty_m=sigma_m,
        )

    def _bearing(
        self,
        node_id: str | None,
        node_position_m: Vec3,
        target_position_m: Vec3,
        covariance_m2: list[list[float]] | None,
        *,
        range_observable: bool,
    ) -> BirdEntityBearingEstimate:
        config = self._config
        return bearing_estimate_toward(
            origin_node_id=node_id or "",
            origin_position_m=node_position_m,
            target_position_m=target_position_m,
            covariance_m2=covariance_m2,
            range_observable=range_observable,
            default_half_width_deg=config.bearing_default_half_width_deg,
            min_half_width_deg=config.bearing_min_half_width_deg,
            max_half_width_deg=config.bearing_max_half_width_deg,
            sigma_multiplier=config.uncertainty_sigma_multiplier,
        )

    @staticmethod
    def _apply_estimate(entity: BirdEntityRecord, estimate: _ObservedSpatialEstimate, event_time_ns: int) -> None:
        if estimate.state == ENTITY_STATE_HEARD_NEARBY:
            return
        entity.state = estimate.state
        entity.position_m = estimate.position_m
        entity.position_covariance_m2 = estimate.covariance_m2
        entity.horizontal_uncertainty_m = estimate.horizontal_uncertainty_m
        entity.bearing = estimate.bearing
        entity.last_spatial_update_ns = event_time_ns

    def _apply_species(self, entity: BirdEntityRecord, observation: BirdEntityObservation) -> None:
        track = observation.track
        if track is not None and track.species_belief:
            entity.species_belief = dict(track.species_belief)
            entity.species_top_label = track.species_top_label or track.label
            entity.species_top_probability = float(track.species_top_probability or track.confidence)
        elif track is not None and track.label != "unknown":
            entity.species_belief = {**entity.species_belief, track.label: float(track.confidence)}
            entity.species_top_label = track.label
            entity.species_top_probability = float(track.confidence)
        elif entity.track_id is None:
            self._merge_detection_belief(entity, observation.label, observation.confidence)

    @staticmethod
    def _merge_detection_belief(entity: BirdEntityRecord, label: str, confidence: float) -> None:
        # Without a tracker belief, keep each species' best score; the top is the max.
        entity.species_belief[label] = max(entity.species_belief.get(label, 0.0), float(confidence))
        top_label, top_probability = max(entity.species_belief.items(), key=lambda item: item[1])
        entity.species_top_label, entity.species_top_probability = top_label, top_probability

    @staticmethod
    def _note_reporting_node(entity: BirdEntityRecord, node_id: str | None) -> None:
        if node_id and node_id not in entity.reporting_node_ids:
            entity.reporting_node_ids.append(node_id)

    @staticmethod
    def refresh_species_from_track(entity: BirdEntityRecord, track: TrackState) -> None:
        if track.species_belief:
            entity.species_belief = dict(track.species_belief)
            entity.species_top_label = track.species_top_label or track.label
            entity.species_top_probability = float(track.species_top_probability or track.confidence)

    def _refresh_from_track(self, entity: BirdEntityRecord, track: TrackState) -> None:
        """Follow the bound track; uncertainty past the pinpoint limit demotes to a fan."""
        self.refresh_species_from_track(entity, track)
        if not entity.is_spatial:
            return
        entity.position_m = tuple(float(v) for v in track.position_m)  # type: ignore[assignment]
        entity.position_covariance_m2 = track.position_covariance_m2
        entity.horizontal_uncertainty_m = horizontal_major_sigma_m(track.position_covariance_m2)
        sigma_m = entity.horizontal_uncertainty_m
        if (
            entity.state == ENTITY_STATE_PINPOINTED
            and sigma_m is not None
            and sigma_m > self._config.pinpointed_max_horizontal_uncertainty_m
        ):
            self._step_back(entity)

    def _step_back(self, entity: BirdEntityRecord) -> None:
        target = ENTITY_STATE_ONE_STEP_BACK[entity.state]
        if target == ENTITY_STATE_DIRECTION_KNOWN and entity.position_m is not None and entity.reporting_node_position_m is not None:
            entity.bearing = self._bearing(
                entity.reporting_node_id,
                entity.reporting_node_position_m,
                entity.position_m,
                entity.position_covariance_m2,
                range_observable=True,
            )
            entity.state = ENTITY_STATE_DIRECTION_KNOWN
            return
        # Heard nearby: the map is spatial-only, so the entity drops its geometry.
        entity.state = ENTITY_STATE_HEARD_NEARBY
        entity.position_m = None
        entity.position_covariance_m2 = None
        entity.horizontal_uncertainty_m = None
        entity.bearing = None
