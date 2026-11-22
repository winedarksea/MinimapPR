"""Ground-truth event model for calibration bundles (ground_truth.json v1 + v2).

v2 (see docs/field_mode_progress/bundle_format_v2.md) is a strict superset of
v1: every v1 event is a valid v2 event. Additions, all optional per event:

- ``individual_id``: events sharing it are bouts of one bird (default: own event_id).
- ``species``: list of ``{"label", "label_category", "status": "true"|"candidate"}``
  (default: one ``true`` entry from ``label``/``label_category``).
- ``is_echo_source`` + ``echo_of_event_id`` + ``reflector``: a virtual image source
  (reflection/sidelobe position) that must NOT surface as a bird.
- ``geometry.position_m`` (site-local ENU metres) as an alternative to
  ``position_geo``, and ``geometry.type == "trajectory"`` with time-stamped
  ``waypoints`` (linear interpolation, clamped at the ends).

`validate_raw_ground_truth_event` is structural (no site frame needed) and runs
in `load_bundle`; `parse_ground_truth_events` resolves positions into the site
frame for scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from minimappr.core.geo import LocalCoordinateFrame
from minimappr.models import GeoPoint

GROUND_TRUTH_SCHEMA_VERSION = 2
SUPPORTED_GROUND_TRUTH_SCHEMA_VERSIONS = frozenset({1, 2})
SUPPORTED_GEOMETRY_TYPES = frozenset({"static", "trajectory"})
SPECIES_STATUS_TRUE = "true"
SPECIES_STATUS_CANDIDATE = "candidate"
SUPPORTED_SPECIES_STATUSES = frozenset({SPECIES_STATUS_TRUE, SPECIES_STATUS_CANDIDATE})
UNKNOWN_LABEL_CATEGORY = "unknown"


@dataclass(frozen=True)
class GroundTruthSpecies:
    label: str
    label_category: str = UNKNOWN_LABEL_CATEGORY
    status: str = SPECIES_STATUS_TRUE


@dataclass
class GroundTruthEvent:
    """One ground-truth emission window, with positions in the site-local ENU frame."""

    event_id: str
    individual_id: str
    label: str
    label_category: str
    species: list[GroundTruthSpecies]
    start_ns: int
    end_ns: int
    # (t_ns, position_m) sorted by t_ns; a static event has exactly one entry.
    waypoints_local_m: list[tuple[int, np.ndarray]]
    is_echo_source: bool = False
    echo_of_event_id: str | None = None
    reflector: dict[str, Any] | None = None
    source_node_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def accepted_labels(self) -> frozenset[str]:
        """Labels scored as correct for this event (true and candidate species)."""
        return frozenset(entry.label for entry in self.species)

    @property
    def accepted_label_categories(self) -> frozenset[str]:
        return frozenset(entry.label_category for entry in self.species)

    @property
    def midpoint_ns(self) -> int:
        return (self.start_ns + self.end_ns) // 2

    def is_active_at(self, timestamp_ns: int, slop_ns: int) -> bool:
        return self.start_ns - slop_ns <= timestamp_ns <= self.end_ns + slop_ns

    def position_at(self, timestamp_ns: int) -> np.ndarray:
        """Linear interpolation between waypoints; clamped outside the path span."""
        waypoints = self.waypoints_local_m
        if len(waypoints) == 1 or timestamp_ns <= waypoints[0][0]:
            return waypoints[0][1]
        if timestamp_ns >= waypoints[-1][0]:
            return waypoints[-1][1]
        for (t0, p0), (t1, p1) in zip(waypoints, waypoints[1:]):
            if t0 <= timestamp_ns <= t1:
                if t1 == t0:
                    return p1
                fraction = (timestamp_ns - t0) / (t1 - t0)
                return p0 + fraction * (p1 - p0)
        return waypoints[-1][1]

    def label_matches(self, label: str | None, label_category: str | None, match_mode: str) -> bool:
        """`exact` compares labels; any other mode compares label categories."""
        if match_mode == "exact":
            return label is not None and label in self.accepted_labels
        return label_category is not None and label_category in self.accepted_label_categories


def site_frame_from_manifest(manifest: dict) -> LocalCoordinateFrame:
    site = manifest["site"]
    return LocalCoordinateFrame(
        origin=GeoPoint(
            lat=site["origin"]["lat"],
            lon=site["origin"]["lon"],
            alt_m=site["origin"].get("alt_m", 0.0),
        ),
        mode=site.get("coordinate_mode", "flat"),
    )


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and bool(np.isfinite(value))


def _validate_point(container: Any, where: str) -> None:
    if not isinstance(container, dict):
        raise ValueError(f"ground-truth geometry {where} must be an object")
    position_m = container.get("position_m")
    if position_m is not None:
        # position_m wins over position_geo, so only it needs to be well-formed.
        if not isinstance(position_m, (list, tuple)) or len(position_m) != 3 or not all(
            _is_finite_number(value) for value in position_m
        ):
            raise ValueError(f"ground-truth geometry {where} position_m must be [x, y, z] numbers")
        return
    position_geo = container.get("position_geo")
    if not isinstance(position_geo, dict):
        raise ValueError(f"ground-truth geometry {where} needs position_geo or position_m")
    # Exported manual rows can carry lat/lon = null; fail at load, not mid-scoring.
    if not (_is_finite_number(position_geo.get("lat")) and _is_finite_number(position_geo.get("lon"))):
        raise ValueError(f"ground-truth geometry {where} position_geo needs numeric lat and lon")


def _int_ns(container: dict, key: str, event_id: Any) -> int:
    try:
        return int(container[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ground-truth event {event_id} {key} must be an integer") from exc


def validate_raw_ground_truth_event(event: dict) -> None:
    """Structural validation of one ground_truth.json event; raises ValueError."""
    event_id = event.get("event_id")
    if not event_id:
        raise ValueError("ground-truth event missing event_id")
    if "start_ns" not in event or "end_ns" not in event:
        raise ValueError(f"ground-truth event {event_id} missing start_ns/end_ns")
    if _int_ns(event, "end_ns", event_id) < _int_ns(event, "start_ns", event_id):
        raise ValueError(f"ground-truth event {event_id} ends before it starts")
    species = event.get("species")
    if species is not None:
        if not isinstance(species, list) or not species:
            raise ValueError(f"ground-truth event {event_id} species must be a non-empty list")
        for entry in species:
            if not isinstance(entry, dict) or not entry.get("label"):
                raise ValueError(f"ground-truth event {event_id} species entries need a label")
            if entry.get("status", SPECIES_STATUS_TRUE) not in SUPPORTED_SPECIES_STATUSES:
                raise ValueError(
                    f"ground-truth event {event_id} species status {entry.get('status')!r} unsupported"
                )
        species_labels = {entry["label"] for entry in species}
        if event.get("label") and event["label"] not in species_labels:
            # Otherwise exact-mode scoring would silently reject the event's own label.
            raise ValueError(
                f"ground-truth event {event_id} label {event['label']!r} is not in its species list"
            )
    elif not event.get("label"):
        raise ValueError(f"ground-truth event {event_id} needs label or species")
    if event.get("is_echo_source") and not event.get("echo_of_event_id"):
        raise ValueError(f"ground-truth echo event {event_id} needs echo_of_event_id")

    geometry = event.get("geometry") or {}
    geometry_type = geometry.get("type")
    if geometry_type not in SUPPORTED_GEOMETRY_TYPES:
        raise ValueError(f"unsupported ground-truth geometry type: {geometry_type!r}")
    if geometry_type == "static":
        _validate_point(geometry, f"of event {event_id}")
        return
    waypoints = geometry.get("waypoints")
    if not isinstance(waypoints, list) or not waypoints:
        raise ValueError(f"ground-truth trajectory geometry of event {event_id} has no waypoints")
    previous_t_ns: int | None = None
    for waypoint in waypoints:
        if not isinstance(waypoint, dict) or "t_ns" not in waypoint:
            raise ValueError(f"ground-truth trajectory geometry of event {event_id}: waypoint missing t_ns")
        _validate_point(waypoint, f"waypoint of event {event_id}")
        t_ns = _int_ns(waypoint, "t_ns", event_id)
        if previous_t_ns is not None and t_ns < previous_t_ns:
            raise ValueError(
                f"ground-truth trajectory geometry of event {event_id}: waypoints not time-sorted"
            )
        previous_t_ns = t_ns


def validate_ground_truth_payload(ground_truth: dict) -> list[dict]:
    """Validate ground_truth.json and return its raw event list."""
    schema_version = int(ground_truth.get("schema_version", 1))
    if schema_version not in SUPPORTED_GROUND_TRUTH_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported ground_truth schema_version: {schema_version}")
    events = list(ground_truth.get("events", []))
    event_ids = set()
    for event in events:
        validate_raw_ground_truth_event(event)
        if event["event_id"] in event_ids:
            raise ValueError(f"duplicate ground-truth event_id {event['event_id']!r}")
        event_ids.add(event["event_id"])
    echo_event_ids = {event["event_id"] for event in events if event.get("is_echo_source")}
    for event in events:
        if not event.get("is_echo_source"):
            continue
        echo_of = event.get("echo_of_event_id")
        if echo_of not in event_ids:
            raise ValueError(
                f"ground-truth echo event {event['event_id']} references unknown event {echo_of!r}"
            )
        if echo_of in echo_event_ids:
            # Also rejects self-references: an echo must point at a real (call) event.
            raise ValueError(
                f"ground-truth echo event {event['event_id']} must reference a real event, "
                f"not echo {echo_of!r}"
            )
    return events


def _point_to_local(container: dict, frame: LocalCoordinateFrame) -> np.ndarray:
    if container.get("position_m") is not None:
        return np.asarray(container["position_m"], dtype=np.float64)
    geo = container["position_geo"]
    return np.asarray(
        frame.geo_to_local(
            GeoPoint(lat=geo["lat"], lon=geo["lon"], alt_m=geo.get("alt_m") or 0.0)
        ),
        dtype=np.float64,
    )


def _species_from_raw(event: dict) -> list[GroundTruthSpecies]:
    raw_species = event.get("species")
    event_label_category = event.get("label_category") or UNKNOWN_LABEL_CATEGORY
    if not raw_species:
        return [GroundTruthSpecies(label=event["label"], label_category=event_label_category)]
    # Entries inherit the event-level category; defaulting them to "unknown" made
    # category-mode scoring reject every detection of a correctly labelled event.
    return [
        GroundTruthSpecies(
            label=entry["label"],
            label_category=entry.get("label_category") or event_label_category,
            status=entry.get("status", SPECIES_STATUS_TRUE),
        )
        for entry in raw_species
    ]


def parse_ground_truth_event(event: dict, frame: LocalCoordinateFrame) -> GroundTruthEvent:
    validate_raw_ground_truth_event(event)
    species = _species_from_raw(event)
    primary = next((s for s in species if s.status == SPECIES_STATUS_TRUE), species[0])
    geometry = event["geometry"]
    if geometry["type"] == "static":
        waypoints = [(int(event["start_ns"]), _point_to_local(geometry, frame))]
    else:
        waypoints = [
            (int(waypoint["t_ns"]), _point_to_local(waypoint, frame))
            for waypoint in geometry["waypoints"]
        ]
    return GroundTruthEvent(
        event_id=str(event["event_id"]),
        individual_id=str(event.get("individual_id") or event["event_id"]),
        label=event.get("label") or primary.label,
        label_category=event.get("label_category") or primary.label_category,
        species=species,
        start_ns=int(event["start_ns"]),
        end_ns=int(event["end_ns"]),
        waypoints_local_m=waypoints,
        is_echo_source=bool(event.get("is_echo_source", False)),
        echo_of_event_id=event.get("echo_of_event_id"),
        reflector=event.get("reflector"),
        source_node_id=event.get("source_node_id"),
        raw=event,
    )


def parse_ground_truth_events(
    events: list[dict], frame: LocalCoordinateFrame
) -> list[GroundTruthEvent]:
    return [parse_ground_truth_event(event, frame) for event in events]
