"""Entity-aware replay metrics: group detections per ground-truth individual.

Association rule (every detection is assigned at most one ground-truth event):

1. *Temporal gate*: event ``E`` is a candidate for detection ``d`` when
   ``E.start_ns - slop <= d.timestamp_ns <= E.end_ns + slop``
   (``association_time_slop_s``).
2. *Localized* detections (``position_m`` present and ``reporting_modality !=
   "omni"``) also need a *spatial gate* against ``E.position_at(d.timestamp_ns)``:
   near-field events (truth within ``far_field_distance_m`` of the array
   centroid) need ``‖d − truth‖ <= association_radius_m``; far-field events need
   centroid-bearing error ``<= far_field_bearing_gate_deg``. Among passing
   candidates the smallest gate ratio (error / gate) wins. Real (non-echo)
   events are tried first; only when none pass are ``is_echo_source`` events
   tried, giving an *echo* association. Otherwise the detection is unassociated.
3. *Omni / unpositioned* detections are associated by time only, to the
   label-matching candidate with the nearest window midpoint; with no
   label-matching candidate they associate only if exactly one candidate exists.

Entity key (what the UI would surface as one thing): ``("entity", entity_id)``
when the detection carries a bird-entity id (W3 entity reporting); else ``("track", track_id)``
when the detection has a track; ``("omni", source_node_id, label)`` for omni
detections (one "heard nearby" row per node and species); otherwise
``("detection", id)``. An entity's label is that of its highest
``label_confidence`` detection (ties: latest timestamp).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from minimappr.calibration.ground_truth_events import GroundTruthEvent

NANOSECONDS_PER_SECOND = 1_000_000_000
ASSOCIATION_KIND_CALL = "call"
ASSOCIATION_KIND_ECHO = "echo"
ASSOCIATION_KIND_NONE = "unassociated"

DEFAULT_ENTITY_EXPECTATIONS: dict[str, Any] = {
    "association_time_slop_s": 2.0,
    "association_radius_m": 15.0,
    "far_field_bearing_gate_deg": 15.0,
    # Spatial gate on horizontal (E, N) distance only: the map is 2-D, and a
    # planar layout (every node at one height) barely constrains height.
    "horizontal_only": False,
    # Gates are opt-in (None = report only) until a baseline exists per bundle family.
    "min_localized_and_classified_recall": None,
    "max_label_swap_rate": None,
    "max_duplicate_entities": None,
    "max_echo_promotions": None,
    # Dual of duplicates: one entity (track) absorbing several birds. Without it a
    # single site-wide track would satisfy recall with zero duplicates.
    "max_entities_spanning_multiple_individuals": None,
}


@dataclass(frozen=True)
class EntityAssociationConfig:
    association_time_slop_ns: int
    association_radius_m: float
    far_field_bearing_gate_deg: float
    far_field_distance_m: float
    match_mode: str
    array_centroid_m: np.ndarray
    horizontal_only: bool = False

    @classmethod
    def from_expectations(
        cls,
        entity_expectations: dict[str, Any],
        *,
        far_field_distance_m: float,
        match_mode: str,
        array_centroid_m: np.ndarray,
    ) -> "EntityAssociationConfig":
        merged = {**DEFAULT_ENTITY_EXPECTATIONS, **(entity_expectations or {})}
        association_time_slop_s = float(merged["association_time_slop_s"])
        association_radius_m = float(merged["association_radius_m"])
        far_field_bearing_gate_deg = float(merged["far_field_bearing_gate_deg"])
        # Gate ratios divide by these; a zero/negative gate would make every
        # association fail (or pass) silently rather than flag a bad bundle.
        if not association_time_slop_s >= 0.0:
            raise ValueError("entities.association_time_slop_s must be >= 0")
        if not association_radius_m > 0.0:
            raise ValueError("entities.association_radius_m must be > 0")
        if not far_field_bearing_gate_deg > 0.0:
            raise ValueError("entities.far_field_bearing_gate_deg must be > 0")
        return cls(
            association_time_slop_ns=int(association_time_slop_s * NANOSECONDS_PER_SECOND),
            association_radius_m=association_radius_m,
            far_field_bearing_gate_deg=far_field_bearing_gate_deg,
            far_field_distance_m=float(far_field_distance_m),
            match_mode=match_mode,
            array_centroid_m=np.asarray(array_centroid_m, dtype=np.float64),
            horizontal_only=bool(merged["horizontal_only"]),
        )


@dataclass
class DetectionAssociation:
    detection: dict
    entity_key: tuple
    is_localized: bool
    kind: str = ASSOCIATION_KIND_NONE
    event: GroundTruthEvent | None = None
    spatially_gated: bool = False


@dataclass
class EntityMetrics:
    """Entity-level replay metrics (definitions in module + format docs).

    - ``localized_and_classified_recall``: fraction of real call events ``E`` for
      which some detection associated with ``E`` belongs to an entity that (a) has
      at least one detection spatially gated to an event of ``E``'s individual and
      (b) whose entity label is correct for ``E``. ``None`` with no call events.
    - ``localized_and_classified_individual_recall``: fraction of individuals with
      at least one such event. Per-event recall undercounts designs that keep one
      row per bird (entity reporting updates its canonical row in place).
    - ``label_swap_rate``: swaps / swap-eligible detections. A spatially gated
      detection of individual ``I`` is *swap-eligible* when another individual
      whose accepted species differ is active at the detection time; it is a
      *swap* when its label is wrong for ``I`` but right for such a concurrent
      individual. ``None`` with no eligible detections.
    - ``duplicate_entities``: Σ over individuals of max(0, distinct entities
      among its call-associated detections − 1).
    - ``echo_promotions``: entities with ≥1 echo-associated detection and no
      call-associated detection.
    """

    call_event_count: int = 0
    echo_event_count: int = 0
    individual_count: int = 0
    localized_and_classified_event_count: int = 0
    localized_and_classified_recall: float | None = None
    localized_and_classified_individual_count: int = 0
    localized_and_classified_individual_recall: float | None = None
    swap_eligible_detection_count: int = 0
    label_swap_count: int = 0
    label_swap_rate: float | None = None
    duplicate_entities: int = 0
    entity_count_by_individual: dict[str, int] = field(default_factory=dict)
    echo_promotions: int = 0
    unassociated_localized_entities: int = 0
    entities_spanning_multiple_individuals: int = 0


def _is_localized(detection: dict) -> bool:
    # Length check instead of truthiness: position_m may be a numpy array.
    position_m = detection.get("position_m")
    return (
        position_m is not None
        and len(position_m) >= 3
        and detection.get("reporting_modality") != "omni"
    )


def entity_key_for_detection(detection: dict, detection_index: int) -> tuple:
    # Field Mode W3-entity: a bird-entity id (entity reporting on) is the UI unit.
    entity_id = detection.get("entity_id")
    if entity_id:
        return ("entity", str(entity_id))
    track_id = detection.get("track_id")
    if track_id:
        return ("track", str(track_id))
    if not _is_localized(detection):
        return ("omni", str(detection.get("source_node_id") or ""), str(detection.get("label") or ""))
    detection_id = detection.get("id")
    if detection_id is None or detection_id == "":
        detection_id = detection.get("event_id")
    if detection_id is None or detection_id == "":
        # Separate namespace so an index never collides with another row's id.
        return ("detection_index", detection_index)
    return ("detection", str(detection_id))


def bearing_deg(origin: np.ndarray, target: np.ndarray) -> float:
    delta = target - origin
    return float(np.degrees(np.arctan2(delta[0], delta[1])) % 360.0)


def angular_difference_deg(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def _spatial_gate_ratio(
    position_m: np.ndarray, event: GroundTruthEvent, timestamp_ns: int, config: EntityAssociationConfig
) -> float:
    truth_m = event.position_at(timestamp_ns)
    centroid = config.array_centroid_m
    if float(np.linalg.norm(truth_m - centroid)) > config.far_field_distance_m:
        bearing_error = angular_difference_deg(
            bearing_deg(centroid, position_m), bearing_deg(centroid, truth_m)
        )
        return bearing_error / config.far_field_bearing_gate_deg
    delta = position_m - truth_m
    if config.horizontal_only:
        delta = delta[:2]
    return float(np.linalg.norm(delta)) / config.association_radius_m


def _best_spatial_match(
    position_m: np.ndarray,
    timestamp_ns: int,
    candidates: list[GroundTruthEvent],
    config: EntityAssociationConfig,
) -> GroundTruthEvent | None:
    best_event, best_ratio = None, None
    for event in candidates:
        ratio = _spatial_gate_ratio(position_m, event, timestamp_ns, config)
        if ratio <= 1.0 and (best_ratio is None or ratio < best_ratio):
            best_event, best_ratio = event, ratio
    return best_event


def associate_detections(
    detections: list[dict],
    events: list[GroundTruthEvent],
    config: EntityAssociationConfig,
) -> list[DetectionAssociation]:
    """Apply the module-level association rule to every detection."""
    call_events = [event for event in events if not event.is_echo_source]
    echo_events = [event for event in events if event.is_echo_source]
    associations: list[DetectionAssociation] = []
    for index, detection in enumerate(detections):
        timestamp_ns = int(detection.get("timestamp_ns") or 0)
        association = DetectionAssociation(
            detection=detection,
            entity_key=entity_key_for_detection(detection, index),
            is_localized=_is_localized(detection),
        )
        slop_ns = config.association_time_slop_ns
        active_calls = [e for e in call_events if e.is_active_at(timestamp_ns, slop_ns)]
        if association.is_localized:
            position_m = np.asarray(detection["position_m"], dtype=np.float64)
            matched = _best_spatial_match(position_m, timestamp_ns, active_calls, config)
            if matched is not None:
                association.kind, association.event = ASSOCIATION_KIND_CALL, matched
                association.spatially_gated = True
            else:
                active_echoes = [e for e in echo_events if e.is_active_at(timestamp_ns, slop_ns)]
                echo = _best_spatial_match(position_m, timestamp_ns, active_echoes, config)
                if echo is not None:
                    association.kind, association.event = ASSOCIATION_KIND_ECHO, echo
                    association.spatially_gated = True
        else:
            label_matching = [
                e
                for e in active_calls
                if e.label_matches(
                    detection.get("label"), detection.get("label_category"), config.match_mode
                )
            ]
            pool = label_matching or (active_calls if len(active_calls) == 1 else [])
            if pool:
                association.kind = ASSOCIATION_KIND_CALL
                association.event = min(
                    pool, key=lambda e: abs(e.midpoint_ns - timestamp_ns)
                )
        associations.append(association)
    return associations


def _entity_labels(associations: list[DetectionAssociation]) -> dict[tuple, dict]:
    """Representative detection per entity: highest label_confidence, then latest."""
    best: dict[tuple, dict] = {}
    for association in associations:
        detection = association.detection
        rank = (
            float(detection.get("label_confidence") or 0.0),
            int(detection.get("timestamp_ns") or 0),
        )
        current = best.get(association.entity_key)
        if current is None or rank > current["rank"]:
            best[association.entity_key] = {"rank": rank, "detection": detection}
    return {key: value["detection"] for key, value in best.items()}


def _swap_rivals(
    detection_event: GroundTruthEvent,
    timestamp_ns: int,
    call_events: list[GroundTruthEvent],
    config: EntityAssociationConfig,
) -> list[GroundTruthEvent]:
    """Concurrent events of other individuals whose accepted species differ."""
    own = (
        detection_event.accepted_labels
        if config.match_mode == "exact"
        else detection_event.accepted_label_categories
    )
    rivals = []
    for event in call_events:
        if event.individual_id == detection_event.individual_id:
            continue
        if not event.is_active_at(timestamp_ns, config.association_time_slop_ns):
            continue
        theirs = event.accepted_labels if config.match_mode == "exact" else event.accepted_label_categories
        if not theirs <= own:
            rivals.append(event)
    return rivals


def compute_entity_metrics(
    detections: list[dict],
    events: list[GroundTruthEvent],
    config: EntityAssociationConfig,
) -> EntityMetrics:
    associations = associate_detections(detections, events, config)
    call_events = [event for event in events if not event.is_echo_source]
    metrics = EntityMetrics(
        call_event_count=len(call_events),
        echo_event_count=len(events) - len(call_events),
        individual_count=len({event.individual_id for event in call_events}),
    )
    entity_representatives = _entity_labels(associations)

    individuals_by_entity: dict[tuple, set[str]] = {}
    kinds_by_entity: dict[tuple, set[str]] = {}
    spatially_gated_individuals_by_entity: dict[tuple, set[str]] = {}
    for association in associations:
        kinds_by_entity.setdefault(association.entity_key, set()).add(association.kind)
        if association.kind != ASSOCIATION_KIND_CALL:
            continue
        individual_id = association.event.individual_id
        individuals_by_entity.setdefault(association.entity_key, set()).add(individual_id)
        if association.spatially_gated:
            spatially_gated_individuals_by_entity.setdefault(association.entity_key, set()).add(
                individual_id
            )

    # Localized-and-classified recall.
    satisfied_event_ids: set[str] = set()
    for association in associations:
        if association.kind != ASSOCIATION_KIND_CALL:
            continue
        event = association.event
        key = association.entity_key
        if event.individual_id not in spatially_gated_individuals_by_entity.get(key, set()):
            continue
        representative = entity_representatives[key]
        if event.label_matches(
            representative.get("label"), representative.get("label_category"), config.match_mode
        ):
            satisfied_event_ids.add(event.event_id)
    metrics.localized_and_classified_event_count = len(satisfied_event_ids)
    if call_events:
        metrics.localized_and_classified_recall = len(satisfied_event_ids) / len(call_events)
        satisfied_individuals = {e.individual_id for e in call_events if e.event_id in satisfied_event_ids}
        metrics.localized_and_classified_individual_count = len(satisfied_individuals)
        metrics.localized_and_classified_individual_recall = len(satisfied_individuals) / metrics.individual_count

    # Label swaps (per spatially gated detection).
    for association in associations:
        if association.kind != ASSOCIATION_KIND_CALL or not association.spatially_gated:
            continue
        detection = association.detection
        if not detection.get("label"):
            continue
        timestamp_ns = int(detection.get("timestamp_ns") or 0)
        rivals = _swap_rivals(association.event, timestamp_ns, call_events, config)
        if not rivals:
            continue
        metrics.swap_eligible_detection_count += 1
        label, category = detection.get("label"), detection.get("label_category")
        if not association.event.label_matches(label, category, config.match_mode) and any(
            rival.label_matches(label, category, config.match_mode) for rival in rivals
        ):
            metrics.label_swap_count += 1
    if metrics.swap_eligible_detection_count:
        metrics.label_swap_rate = metrics.label_swap_count / metrics.swap_eligible_detection_count

    # Duplicates, echo promotions, spurious entities.
    entities_by_individual: dict[str, set[tuple]] = {}
    for key, individual_ids in individuals_by_entity.items():
        for individual_id in individual_ids:
            entities_by_individual.setdefault(individual_id, set()).add(key)
        if len(individual_ids) > 1:
            metrics.entities_spanning_multiple_individuals += 1
    metrics.entity_count_by_individual = {
        individual_id: len(keys) for individual_id, keys in sorted(entities_by_individual.items())
    }
    metrics.duplicate_entities = sum(
        max(0, count - 1) for count in metrics.entity_count_by_individual.values()
    )
    localized_entity_keys = {a.entity_key for a in associations if a.is_localized}
    for key, kinds in kinds_by_entity.items():
        if ASSOCIATION_KIND_CALL in kinds:
            continue
        if ASSOCIATION_KIND_ECHO in kinds:
            metrics.echo_promotions += 1
        elif key in localized_entity_keys:
            metrics.unassociated_localized_entities += 1
    return metrics


def entity_threshold_errors(metrics: EntityMetrics, entity_expectations: dict[str, Any]) -> list[str]:
    """Human-readable gate failures; thresholds left at None are not enforced."""
    merged = {**DEFAULT_ENTITY_EXPECTATIONS, **(entity_expectations or {})}
    errors: list[str] = []
    min_recall = merged.get("min_localized_and_classified_recall")
    if min_recall is not None and metrics.call_event_count:
        recall = metrics.localized_and_classified_recall or 0.0
        if recall < float(min_recall):
            errors.append(f"localized-and-classified recall {recall:.2f} below {min_recall}")
    max_swap_rate = merged.get("max_label_swap_rate")
    if max_swap_rate is not None and metrics.label_swap_rate is not None:
        if metrics.label_swap_rate > float(max_swap_rate):
            errors.append(
                f"label swap rate {metrics.label_swap_rate:.2f} above {max_swap_rate} "
                f"({metrics.label_swap_count}/{metrics.swap_eligible_detection_count})"
            )
    max_duplicates = merged.get("max_duplicate_entities")
    if max_duplicates is not None and metrics.duplicate_entities > int(max_duplicates):
        errors.append(f"duplicate entities {metrics.duplicate_entities} above {max_duplicates}")
    max_echoes = merged.get("max_echo_promotions")
    if max_echoes is not None and metrics.echo_promotions > int(max_echoes):
        errors.append(f"echo promotions {metrics.echo_promotions} above {max_echoes}")
    max_merged = merged.get("max_entities_spanning_multiple_individuals")
    if max_merged is not None and metrics.entities_spanning_multiple_individuals > int(max_merged):
        errors.append(
            f"entities spanning multiple individuals "
            f"{metrics.entities_spanning_multiple_individuals} above {max_merged}"
        )
    return errors
