"""Ground-truth and range-ladder checks for bundle validation.

Events are ``ground_truth.json`` v1/v2 (``calibration/ground_truth_events.py``).
The protocol adds one optional key, ``geometry.position_std_m``: the 1-sigma
accuracy of the truth position (tape from a surveyed peg, phone GPS, drone
GPS log). Scoring does not read it yet; validation uses it to flag truth too
coarse to grade a range-resolved fix at that event's range.

Range-ladder rungs match the E5 matrix (``tests/range_matrix_cells.py``); an
event's range is from its position at mid-event to the nearest localizing
node (reference nodes placed at the source are excluded, as in replay).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from minimappr.calibration.bundle_validation_findings import (
    BundleFindingCode as Code,
    BundleValidationConfig,
    BundleValidationFinding as Finding,
)
from minimappr.calibration.ground_truth_events import (
    GroundTruthEvent,
    parse_ground_truth_event,
    validate_ground_truth_payload,
)
from minimappr.core.geo import LocalCoordinateFrame

NANOSECONDS_PER_SECOND = 1_000_000_000


def reference_node_ids(raw_events: list[Mapping[str, Any]]) -> set[str]:
    return {str(event["source_node_id"]) for event in raw_events if event.get("source_node_id")}


def valid_ground_truth_events(ground_truth_payload: Mapping[str, Any] | None) -> list[dict]:
    """Raw events when the payload validates, else none (the check reports why)."""
    if ground_truth_payload is None:
        return []
    try:
        return validate_ground_truth_payload(dict(ground_truth_payload))
    except (ValueError, TypeError):
        return []


def range_ladder_rung_for(range_m: float, config: BundleValidationConfig) -> float | None:
    if range_m <= 0.0:
        return None
    log_range = math.log10(range_m)
    nearest_rung = min(config.range_ladder_rungs_m, key=lambda rung: abs(math.log10(rung) - log_range))
    return nearest_rung if abs(math.log10(nearest_rung) - log_range) <= config.range_rung_half_width_decades else None


def _window_findings(event: Mapping[str, Any], window_ns: tuple[int, int]) -> list[Finding]:
    start_ns, end_ns = int(event["start_ns"]), int(event["end_ns"])
    event_id = str(event["event_id"])
    if end_ns < window_ns[0] or start_ns > window_ns[1]:
        return [Finding(Code.GROUND_TRUTH_OUTSIDE_CAPTURE, "event lies entirely outside the capture window", event_id=event_id)]
    if start_ns < window_ns[0] or end_ns > window_ns[1]:
        return [Finding(Code.GROUND_TRUTH_PARTIALLY_OUTSIDE_CAPTURE, "event runs past the capture window", event_id=event_id)]
    return []


def _trajectory_findings(event: GroundTruthEvent, config: BundleValidationConfig) -> list[Finding]:
    if event.raw["geometry"]["type"] != "trajectory":
        return []
    waypoint_times_ns = [t_ns for t_ns, _ in event.waypoints_local_m]
    findings: list[Finding] = []
    if waypoint_times_ns[0] > event.start_ns or waypoint_times_ns[-1] < event.end_ns:
        findings.append(
            Finding(
                Code.GROUND_TRUTH_TRAJECTORY_INCOMPLETE,
                "waypoints do not span the event; truth is clamped to the end points outside the logged path",
                event_id=event.event_id,
            )
        )
    largest_spacing_seconds = max(np.diff(waypoint_times_ns), default=0) / NANOSECONDS_PER_SECOND
    if largest_spacing_seconds > config.maximum_trajectory_waypoint_spacing_seconds:
        findings.append(
            Finding(
                Code.GROUND_TRUTH_TRAJECTORY_SPARSE,
                f"waypoints up to {largest_spacing_seconds:.1f} s apart "
                f"(limit {config.maximum_trajectory_waypoint_spacing_seconds:.1f} s); log the path faster",
                event_id=event.event_id,
                details={"largest_waypoint_spacing_seconds": largest_spacing_seconds},
            )
        )
    return findings


def _range_to_nearest_node_m(event: GroundTruthEvent, node_positions_m: Mapping[str, np.ndarray]) -> float | None:
    if not node_positions_m:
        return None
    position = event.position_at(event.midpoint_ns)
    return float(min(np.linalg.norm(position - node_position) for node_position in node_positions_m.values()))


def _range_ladder_findings(ranges_by_event_id: Mapping[str, float], config: BundleValidationConfig) -> list[Finding]:
    counts = {rung: 0 for rung in config.range_ladder_rungs_m}
    off_ladder_event_ids: list[str] = []
    for event_id, range_m in ranges_by_event_id.items():
        rung = range_ladder_rung_for(range_m, config)
        if rung is None:
            off_ladder_event_ids.append(event_id)
        else:
            counts[rung] += 1
    count_by_rung_label = {f"{rung:g}m": count for rung, count in counts.items()}
    findings = [
        Finding(
            Code.RANGE_LADDER_COVERAGE,
            f"events per range rung {count_by_rung_label}",
            details={"event_count_by_rung": count_by_rung_label},
        )
    ]
    for rung, count in counts.items():
        if count == 0:
            findings.append(Finding(Code.RANGE_LADDER_RUNG_EMPTY, f"no events at the {rung:g} m rung", details={"rung_m": rung}))
        elif count < config.minimum_events_per_range_rung:
            findings.append(
                Finding(
                    Code.RANGE_LADDER_RUNG_UNDERCOVERED,
                    f"{count} event(s) at the {rung:g} m rung; record at least {config.minimum_events_per_range_rung}",
                    details={"rung_m": rung, "event_count": count},
                )
            )
    if off_ladder_event_ids:
        findings.append(
            Finding(
                Code.GROUND_TRUTH_OUTSIDE_RANGE_LADDER,
                f"{len(off_ladder_event_ids)} event(s) fall between or beyond the ladder rungs",
                details={"event_ids": sorted(off_ladder_event_ids)},
            )
        )
    return findings


def _truth_accuracy_findings(event: GroundTruthEvent, range_m: float | None, config: BundleValidationConfig) -> list[Finding]:
    position_std_m = event.raw["geometry"].get("position_std_m")
    if not isinstance(position_std_m, (int, float)) or range_m is None or range_m <= 0.0:
        return []
    if position_std_m > config.maximum_truth_position_std_relative_to_range * range_m:
        return [
            Finding(
                Code.GROUND_TRUTH_ACCURACY_COARSE_FOR_RANGE,
                f"truth position_std_m {position_std_m:g} m at {range_m:.1f} m range cannot grade a range-resolved fix",
                event_id=event.event_id,
                details={"position_std_m": float(position_std_m), "range_m": range_m},
            )
        ]
    return []


def check_ground_truth(
    ground_truth_payload: Mapping[str, Any] | None,
    *,
    window_ns: tuple[int, int] | None,
    frame: LocalCoordinateFrame | None,
    bundle_node_ids: set[str],
    localizing_node_positions_m: Mapping[str, np.ndarray],
    config: BundleValidationConfig,
) -> list[Finding]:
    if ground_truth_payload is None:
        return [Finding(Code.GROUND_TRUTH_MISSING, "bundle has no ground_truth.json")]
    try:
        raw_events = validate_ground_truth_payload(dict(ground_truth_payload))
    except (ValueError, TypeError) as exc:
        return [Finding(Code.GROUND_TRUTH_INVALID, str(exc))]
    real_events = [event for event in raw_events if not event.get("is_echo_source")]
    if not real_events:
        return [Finding(Code.GROUND_TRUTH_MISSING, "no ground-truth events to score against")]
    findings: list[Finding] = []
    events_without_truth_accuracy: list[str] = []
    ranges_by_event_id: dict[str, float] = {}
    for raw_event in raw_events:
        event_id = str(raw_event["event_id"])
        source_node_id = raw_event.get("source_node_id")
        if source_node_id and source_node_id not in bundle_node_ids:
            findings.append(
                Finding(Code.GROUND_TRUTH_SOURCE_NODE_UNKNOWN, f"source_node_id {source_node_id!r} is not a bundle node", event_id=event_id)
            )
        if window_ns is not None:
            findings += _window_findings(raw_event, window_ns)
        if frame is None:
            continue
        event = parse_ground_truth_event(raw_event, frame)
        farthest_m = max(float(np.linalg.norm(position)) for _, position in event.waypoints_local_m)
        if not math.isfinite(farthest_m) or farthest_m > config.maximum_distance_from_site_origin_m:
            findings.append(
                Finding(
                    Code.GROUND_TRUTH_POSITION_OUTSIDE_SITE,
                    f"truth position is {farthest_m:.0f} m from the site origin; was it logged in another frame?",
                    event_id=event_id,
                )
            )
            continue
        findings += _trajectory_findings(event, config)
        if event.is_echo_source:
            continue
        if raw_event["geometry"].get("position_std_m") is None:
            events_without_truth_accuracy.append(event_id)
        range_m = _range_to_nearest_node_m(event, localizing_node_positions_m)
        findings += _truth_accuracy_findings(event, range_m, config)
        if range_m is not None:
            ranges_by_event_id[event_id] = range_m
    if events_without_truth_accuracy:
        findings.append(
            Finding(
                Code.GROUND_TRUTH_POSITION_STD_MISSING,
                f"{len(events_without_truth_accuracy)} event(s) state no geometry.position_std_m (truth accuracy)",
                details={"event_ids": sorted(events_without_truth_accuracy)},
            )
        )
    if frame is not None and localizing_node_positions_m:
        findings += _range_ladder_findings(ranges_by_event_id, config)
    return findings
