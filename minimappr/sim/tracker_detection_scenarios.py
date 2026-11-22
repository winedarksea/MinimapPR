"""Tracker-level scenarios: synthetic detection sequences with realistic fixes.

Replay bundles currently collapse every fix onto a node (server_baseline.md),
so they cannot exercise a tracker. These scenarios skip audio and localization
and feed ``resolve_detection_track`` (the real assembly decision: tracker
update, partial binding, hidden tracks) with detections whose positions are the
truth plus seeded Gaussian fix jitter, and whose scores mimic BirdNET (weak
bouts, occasional confusions). Output detections are scored with
``replay_entity_metrics`` against per-bout ground-truth events.

BirdNET emulation: with ``birdnet_two_threshold_enabled`` off, a bout whose top
score is under the display floor becomes ``unknown``/0.0 (what the
classifier returns today; the display floor is ``birdnet_trigger_min_confidence``);
with it on, bouts down to the localize-to-track
floor are kept and flagged ``below_display_threshold``. A detection *surfaces*
(is scored) unless it is unknown/0.0 or below display without a visible track
(both are silent drops in ``FusionNode._process_rules_and_delivery``). Each
surfaced detection carries its visible track's label, i.e. what the UI shows
for that entity.

With ``entity_reporting_enabled`` the W3 bird-entity registry runs on top, as
in ``FusionNode``: an unpositioned bout matching an entity is absorbed before
assembly (it still feeds partial binding), every other surfaced bout is
observed into its entity, and each detection carries ``entity_id`` plus the
entity's label, so ``replay_entity_metrics`` scores entities instead of tracks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from minimappr.calibration.ground_truth_events import GroundTruthEvent, GroundTruthSpecies
from minimappr.config import Settings
from minimappr.core.assembly_track_resolution import resolve_detection_track
from minimappr.core.bird_entity_registry import BirdEntityObservation, BirdEntityRegistry
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.models import NodeSpec, NodeType
from minimappr.sim.replay_entity_metrics import (
    EntityAssociationConfig,
    ASSOCIATION_KIND_CALL,
    EntityMetrics,
    associate_detections,
    compute_entity_metrics,
)

NANOSECONDS_PER_SECOND = 1_000_000_000
SCENARIO_START_NS = 1_700_000_000_000_000_000
BOUT_DURATION_S = 0.5
WILDLIFE = "wildlife"
# Same 3-node layout as synthetic_entity_scenarios.
DEFAULT_NODE_POSITIONS_M = {
    "node-a": (0.0, 0.0, 0.0),
    "node-b": (20.0, 0.0, 0.0),
    "node-c": (10.0, 18.0, 0.0),
}
SCORING_ASSOCIATION_RADIUS_M = 5.0
SCORING_TIME_SLOP_S = 1.0
FAR_FIELD_DISTANCE_M = 1_000.0


@dataclass(frozen=True)
class SyntheticBout:
    time_s: float
    individual_id: str
    true_position_m: tuple[float, float, float]
    scores: dict[str, float]
    # "localized" (full fix with jitter) or "omni" (no position; heard by node).
    mode: str = "localized"
    reporting_node_id: str = "node-a"


@dataclass(frozen=True)
class TrackerScenario:
    name: str
    species_by_individual: dict[str, str]
    bouts: list[SyntheticBout]
    fix_jitter_m: float = 1.0
    node_positions_m: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: dict(DEFAULT_NODE_POSITIONS_M)
    )


@dataclass
class TrackerScenarioResult:
    scenario: str
    surfaced_detections: int
    distinct_visible_tracks: int
    distinct_entities: int
    # Fraction of call-associated surfaced detections showing the true label.
    surfaced_label_accuracy: float | None
    metrics: EntityMetrics


async def _registry_for(scenario: TrackerScenario) -> NodeRegistry:
    registry = NodeRegistry()
    for node_id, position in scenario.node_positions_m.items():
        await registry.upsert(
            NodeSpec(id=node_id, node_type=NodeType.POINT, position_m=position,
                     sensor_offsets_m=[(0.0, 0.0, 0.0)], capabilities=["audio"]),
            last_seen_ns=SCENARIO_START_NS,
        )
    return registry


def ground_truth_events(scenario: TrackerScenario) -> list[GroundTruthEvent]:
    events = []
    for index, bout in enumerate(scenario.bouts):
        start_ns = SCENARIO_START_NS + int(bout.time_s * NANOSECONDS_PER_SECOND)
        species = scenario.species_by_individual[bout.individual_id]
        events.append(
            GroundTruthEvent(
                event_id=f"{scenario.name}-{index}",
                individual_id=bout.individual_id,
                label=species,
                label_category=WILDLIFE,
                species=[GroundTruthSpecies(label=species, label_category=WILDLIFE)],
                start_ns=start_ns,
                end_ns=start_ns + int(BOUT_DURATION_S * NANOSECONDS_PER_SECOND),
                waypoints_local_m=[(start_ns, np.asarray(bout.true_position_m, dtype=np.float64))],
            )
        )
    return events


def _emulated_birdnet(scores: dict[str, float], settings: Settings) -> tuple[str, float, dict, bool]:
    label, confidence = max(scores.items(), key=lambda item: item[1])
    display_floor = settings.birdnet_trigger_min_confidence
    if settings.birdnet_two_threshold_enabled:
        if confidence < settings.birdnet_localize_to_track_min_confidence:
            return "unknown", 0.0, {}, False
        return label, confidence, dict(scores), confidence < display_floor
    if confidence < display_floor:
        return "unknown", 0.0, {}, False
    kept = {key: value for key, value in scores.items() if value >= display_floor}
    return label, confidence, kept, False


async def run_tracker_scenario(
    scenario: TrackerScenario, settings_overrides: dict, *, seed: int = 0
) -> tuple[list[dict], TrackManager]:
    settings = Settings(**settings_overrides)
    manager = TrackManager(settings)
    registry = await _registry_for(scenario)
    rng = np.random.default_rng(seed)
    detections: list[dict] = []
    entity_registry = (
        BirdEntityRegistry(settings.entity_reporting_config(), entity_id_factory=_sequential_entity_ids())
        if settings.entity_reporting_enabled
        else None
    )
    for index, bout in enumerate(scenario.bouts):
        timestamp_ns = SCENARIO_START_NS + int((bout.time_s + BOUT_DURATION_S / 2) * NANOSECONDS_PER_SECOND)
        label, confidence, scores, below_display = _emulated_birdnet(bout.scores, settings)
        node_position = scenario.node_positions_m[bout.reporting_node_id]
        sensor_id = next(
            sensor for sensor, node in (await registry.sensor_node_ids()).items()
            if node == bout.reporting_node_id
        )
        if bout.mode == "localized":
            jitter = rng.normal(0.0, scenario.fix_jitter_m, size=3) * np.array([1.0, 1.0, 0.3])
            position = tuple(float(v) for v in np.asarray(bout.true_position_m) + jitter)
            spatial_mode, tier, modality = "localized", "full_3d", "localized"
        else:
            position, spatial_mode, tier, modality = node_position, "node_only", "classification_only", "omni"
        absorbing_entity = None
        if entity_registry is not None and modality == "omni" and not below_display and label != "unknown":
            absorbing_entity = entity_registry.find_absorbing_entity(
                label=label, event_time_ns=timestamp_ns, spatial_display_mode=spatial_mode,
                reporting_node_id=bout.reporting_node_id, reporting_node_position_m=node_position,
                detection_position_m=None,
            )
        resolution = await resolve_detection_track(
            tracker=manager, registry=registry, spatial_display_mode=spatial_mode,
            suppressed_by_zone=False, capability_tier=tier, event_time_ns=timestamp_ns,
            localization_position_m=position, localization_position_covariance_m2=None,
            classification_label=label, classification_confidence=confidence,
            classification_scores=scores,
            classification_features={"below_display_threshold": below_display} if below_display else {},
            label_category=WILDLIFE, iff_category="unknown", label_id=None,
            selected_sensor_ids=[sensor_id], source_node_id=bout.reporting_node_id,
            reference_sensor=sensor_id, tdoa_s={}, environment={},
        )
        track = resolution.track
        entity = None
        if absorbing_entity is not None:
            entity_registry.absorb_unlocalized(
                absorbing_entity.entity_id, event_time_ns=timestamp_ns, label=label,
                confidence=confidence, reporting_node_id=bout.reporting_node_id,
            )
            if track is not None and track.id == absorbing_entity.track_id:
                entity_registry.refresh_species_from_track(absorbing_entity, track)
            entity = absorbing_entity
        surfaced = not (label == "unknown" and confidence == 0.0) and not (below_display and track is None)
        if not surfaced and entity is None:
            continue
        if entity_registry is not None and entity is None and label_category_is_entity(settings):
            entity = entity_registry.observe(BirdEntityObservation(
                event_time_ns=timestamp_ns, label=label, confidence=confidence, label_category=WILDLIFE,
                spatial_display_mode=spatial_mode, reporting_node_id=bout.reporting_node_id,
                reporting_node_position_m=node_position, detection_position_m=position,
                detection_id=f"det-{index}", track=track,
            )).entity
        if entity is not None:
            detections.append({
                "id": f"det-{index}", "timestamp_ns": timestamp_ns, "position_m": list(position),
                "reporting_modality": modality, "source_node_id": bout.reporting_node_id,
                "track_id": track.id if track is not None else None, "entity_id": entity.entity_id,
                "entity_state": entity.state, "label": entity.species_top_label,
                "label_confidence": float(entity.species_top_probability), "label_category": WILDLIFE,
            })
            continue
        detections.append({
            "id": f"det-{index}",
            "timestamp_ns": timestamp_ns,
            "position_m": list(position),
            "reporting_modality": modality,
            "source_node_id": bout.reporting_node_id,
            "track_id": track.id if track is not None else None,
            "label": track.label if track is not None else label,
            "label_confidence": float(track.confidence if track is not None else confidence),
            "label_category": WILDLIFE,
        })
    return detections, manager


def _sequential_entity_ids():
    counter = iter(range(1, 1_000_000))
    return lambda: f"ent-{next(counter)}"


def label_category_is_entity(settings: Settings) -> bool:
    return settings.entity_reporting_config().applies_to(WILDLIFE)


async def score_tracker_scenario(
    scenario: TrackerScenario, settings_overrides: dict, *, seed: int = 0
) -> TrackerScenarioResult:
    detections, _manager = await run_tracker_scenario(scenario, settings_overrides, seed=seed)
    centroid = np.mean(np.asarray(list(scenario.node_positions_m.values())), axis=0)
    config = EntityAssociationConfig.from_expectations(
        {"association_radius_m": SCORING_ASSOCIATION_RADIUS_M,
         "association_time_slop_s": SCORING_TIME_SLOP_S},
        far_field_distance_m=FAR_FIELD_DISTANCE_M, match_mode="exact", array_centroid_m=centroid,
    )
    events = ground_truth_events(scenario)
    metrics = compute_entity_metrics(detections, events, config)
    call_associations = [
        association for association in associate_detections(detections, events, config)
        if association.kind == ASSOCIATION_KIND_CALL
    ]
    correct = sum(
        1 for association in call_associations
        if association.event.label_matches(association.detection["label"], WILDLIFE, "exact")
    )
    return TrackerScenarioResult(
        scenario=scenario.name,
        surfaced_detections=len(detections),
        distinct_visible_tracks=len({d["track_id"] for d in detections if d["track_id"]}),
        distinct_entities=len({d["entity_id"] for d in detections if d.get("entity_id")}),
        surfaced_label_accuracy=correct / len(call_associations) if call_associations else None,
        metrics=metrics,
    )
