"""Track continuity metrics: update gaps, identity switches and fragmentation.

Inputs are track updates (one per position applied to a track) and the
ground-truth emissions. Each update is attributed to the nearest truth
individual active at its time, within ``association_radius_m``; updates with
no such individual are counted as unassociated (phantom track motion).

Definitions, per truth emission window ``[start, end]`` (a drone fly-by is one
long window; a bird's calls are several short ones):

* **update gaps**: the silences start->first update, between consecutive
  updates, and last update->end. An emission with no update is one gap of its
  whole duration. This is the metric behind "a drone track moves about once
  per 30 s".
* **fragments**: runs of updates split by a gap longer than
  ``fragment_break_gap_seconds``; fragmentations = fragments - 1.
* **tracked time fraction**: total duration of gaps no longer than the break
  gap, over the window duration.

Per truth *individual* (all of its emissions in time order):

* **identity switches**: consecutive attributed updates whose track ids
  differ. Counted across emissions, so a perched bird that calls again and is
  given a new track is a switch (what ``stationary_hold`` exists to prevent).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from minimappr.calibration.ground_truth_events import GroundTruthEvent
from minimappr.sim.replay_gate_config import TrackMetricsConfig

NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class TrackUpdate:
    track_id: str
    timestamp_ns: int
    position_m: tuple[float, float, float]


@dataclass
class IndividualTrackResult:
    individual_id: str
    emission_count: int = 0
    attributed_update_count: int = 0
    distinct_track_ids: tuple[str, ...] = ()
    identity_switches: int = 0
    fragment_count: int = 0
    fragmentations: int = 0
    max_update_gap_seconds: float | None = None
    tracked_time_fraction: float | None = None


@dataclass
class TrackMetricsReport:
    individual_results: list[IndividualTrackResult] = field(default_factory=list)
    update_count: int = 0
    unassociated_update_count: int = 0
    tracks_spanning_multiple_individuals: int = 0
    total_identity_switches: int = 0
    total_fragmentations: int = 0
    max_update_gap_seconds: float | None = None
    update_gap_percentile_seconds: float | None = None
    min_tracked_time_fraction: float | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors


def track_updates_from_detections(detections: list[dict]) -> list[TrackUpdate]:
    """Stored detection rows with a track id and a position, as track updates."""
    updates = []
    for detection in detections:
        track_id = detection.get("track_id")
        position = detection.get("position_m")
        if not track_id or position is None or len(position) < 3:
            continue
        updates.append(
            TrackUpdate(
                track_id=str(track_id),
                timestamp_ns=int(detection.get("timestamp_ns") or 0),
                position_m=tuple(float(axis) for axis in position[:3]),
            )
        )
    return updates


def _attribute_update(
    update: TrackUpdate, emissions: list[GroundTruthEvent], config: TrackMetricsConfig, axes: slice
) -> GroundTruthEvent | None:
    slop_ns = int(config.association_time_slop_seconds * NANOSECONDS_PER_SECOND)
    position = np.asarray(update.position_m, dtype=np.float64)
    best_event, best_distance = None, None
    for event in emissions:
        if not event.is_active_at(update.timestamp_ns, slop_ns):
            continue
        distance = float(np.linalg.norm((position - event.position_at(update.timestamp_ns))[axes]))
        if distance <= config.association_radius_m and (best_distance is None or distance < best_distance):
            best_event, best_distance = event, distance
    return best_event


def _emission_gaps_seconds(event: GroundTruthEvent, update_times_ns: list[int]) -> list[float]:
    # Updates attributed through the time slop may sit just outside the window;
    # clamp so they neither create negative gaps nor extend the window.
    clamped_times_ns = sorted(min(max(t, event.start_ns), event.end_ns) for t in update_times_ns)
    boundaries_ns = [event.start_ns, *clamped_times_ns, event.end_ns]
    return [(later - earlier) / NANOSECONDS_PER_SECOND for earlier, later in zip(boundaries_ns, boundaries_ns[1:])]


def compute_track_metrics(
    updates: list[TrackUpdate],
    ground_truth_events: list[GroundTruthEvent],
    config: TrackMetricsConfig,
) -> TrackMetricsReport:
    axes = slice(0, 2) if config.horizontal_only else slice(0, 3)
    emissions = [event for event in ground_truth_events if not event.is_echo_source]
    report = TrackMetricsReport(update_count=len(updates))
    updates_by_event_id: dict[str, list[TrackUpdate]] = defaultdict(list)
    individuals_by_track_id: dict[str, set[str]] = defaultdict(set)

    for update in updates:
        event = _attribute_update(update, emissions, config, axes)
        if event is None:
            report.unassociated_update_count += 1
            continue
        updates_by_event_id[event.event_id].append(update)
        individuals_by_track_id[update.track_id].add(event.individual_id)

    emissions_by_individual: dict[str, list[GroundTruthEvent]] = defaultdict(list)
    for event in emissions:
        emissions_by_individual[event.individual_id].append(event)

    pooled_gaps_seconds: list[float] = []
    for individual_id, individual_emissions in emissions_by_individual.items():
        individual_emissions.sort(key=lambda event: event.start_ns)
        result = IndividualTrackResult(individual_id=individual_id, emission_count=len(individual_emissions))
        attributed: list[TrackUpdate] = []
        tracked_seconds, total_seconds = 0.0, 0.0
        for event in individual_emissions:
            event_updates = sorted(updates_by_event_id.get(event.event_id, []), key=lambda u: u.timestamp_ns)
            attributed.extend(event_updates)
            gaps_seconds = _emission_gaps_seconds(event, [update.timestamp_ns for update in event_updates])
            pooled_gaps_seconds.extend(gaps_seconds)
            interior_gaps_seconds = gaps_seconds[1:-1]
            if event_updates:
                result.fragment_count += 1 + sum(
                    gap > config.fragment_break_gap_seconds for gap in interior_gaps_seconds
                )
            tracked_seconds += sum(gap for gap in gaps_seconds if gap <= config.fragment_break_gap_seconds)
            total_seconds += (event.end_ns - event.start_ns) / NANOSECONDS_PER_SECOND
            event_max_gap_seconds = max(gaps_seconds)
            if result.max_update_gap_seconds is None or event_max_gap_seconds > result.max_update_gap_seconds:
                result.max_update_gap_seconds = event_max_gap_seconds
        attributed.sort(key=lambda update: update.timestamp_ns)
        result.attributed_update_count = len(attributed)
        result.distinct_track_ids = tuple(sorted({update.track_id for update in attributed}))
        result.identity_switches = sum(
            earlier.track_id != later.track_id for earlier, later in zip(attributed, attributed[1:])
        )
        result.fragmentations = max(0, result.fragment_count - 1) if result.fragment_count else 0
        result.tracked_time_fraction = tracked_seconds / total_seconds if total_seconds > 0.0 else None
        report.individual_results.append(result)

    report.tracks_spanning_multiple_individuals = sum(
        len(individual_ids) > 1 for individual_ids in individuals_by_track_id.values()
    )
    report.total_identity_switches = sum(result.identity_switches for result in report.individual_results)
    report.total_fragmentations = sum(result.fragmentations for result in report.individual_results)
    if pooled_gaps_seconds:
        report.max_update_gap_seconds = max(pooled_gaps_seconds)
        report.update_gap_percentile_seconds = float(np.percentile(pooled_gaps_seconds, config.update_gap_percentile))
    tracked_fractions = [r.tracked_time_fraction for r in report.individual_results if r.tracked_time_fraction is not None]
    report.min_tracked_time_fraction = min(tracked_fractions) if tracked_fractions else None
    report.errors = track_gate_errors(report, config)
    return report


def track_gate_errors(report: TrackMetricsReport, config: TrackMetricsConfig) -> list[str]:
    errors: list[str] = []
    if (
        config.max_update_gap_seconds is not None
        and report.max_update_gap_seconds is not None
        and report.max_update_gap_seconds > config.max_update_gap_seconds
    ):
        errors.append(f"max track update gap {report.max_update_gap_seconds:.2f} s above {config.max_update_gap_seconds} s")
    if config.max_identity_switches is not None and report.total_identity_switches > config.max_identity_switches:
        errors.append(f"identity switches {report.total_identity_switches} above {config.max_identity_switches}")
    if config.max_fragmentations is not None and report.total_fragmentations > config.max_fragmentations:
        errors.append(f"track fragmentations {report.total_fragmentations} above {config.max_fragmentations}")
    if (
        config.min_tracked_time_fraction is not None
        and report.min_tracked_time_fraction is not None
        and report.min_tracked_time_fraction < config.min_tracked_time_fraction
    ):
        errors.append(
            f"min tracked time fraction {report.min_tracked_time_fraction:.3f} below {config.min_tracked_time_fraction}"
        )
    return errors
