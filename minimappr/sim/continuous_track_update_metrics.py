"""Per-solve track-update accounting for continuous-source scenes (E6).

The harness snapshots the tracker after every delivery and records which
track each delivery updated (``TrackUpdateObservation``). From those records
this module scores, per source:

- whether every solve updated a track exactly once (a duplicate delivery of
  one solve must not apply its position twice);
- whether each update was stamped with the solve's window-centre time;
- the largest gap between the source's track updates, measured from its first
  solve to its last, so a source never updated scores its whole audible span;
- identity: the distinct tracks its updates landed on, switches between them,
  and tracks shared with another source.

``drone_update_gap_gate_failures`` turns those into the milestone 3 gate.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from minimappr.sim.continuous_source_scenes import SOURCE_KIND_DRONE, ContinuousSourceScene

NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True, slots=True)
class TrackUpdateObservation:
    delivery_index: int
    delivery_kind: str
    solve_id: str
    track_id: str
    update_count_increase: int
    track_last_seen_ns: int


@dataclass(slots=True)
class SourceTrackUpdateMetrics:
    source_id: str
    solve_count: int
    missed_solve_ids: list[str] = field(default_factory=list)
    multiply_applied_solve_ids: list[str] = field(default_factory=list)
    misstamped_solve_ids: list[str] = field(default_factory=list)
    track_ids_in_update_order: list[str] = field(default_factory=list)
    identity_switch_count: int = 0
    max_update_gap_seconds: float = 0.0
    updated_solve_count: int = 0

    @property
    def distinct_track_ids(self) -> list[str]:
        return list(dict.fromkeys(self.track_ids_in_update_order))

    def summary(self) -> str:
        return (
            f"{self.source_id}: {self.updated_solve_count}/{self.solve_count} solves updated, "
            f"max gap {self.max_update_gap_seconds:.2f} s, tracks {self.distinct_track_ids}, "
            f"switches {self.identity_switch_count}, missed {len(self.missed_solve_ids)}, "
            f"applied twice {len(self.multiply_applied_solve_ids)}, misstamped {len(self.misstamped_solve_ids)}"
        )


@dataclass(slots=True)
class SceneTrackUpdateMetrics:
    by_source: dict[str, SourceTrackUpdateMetrics]
    # Track id -> sources whose solves updated it, for tracks updated by more than one.
    shared_track_sources: dict[str, list[str]]

    def summary(self) -> str:
        lines = [metrics.summary() for metrics in self.by_source.values()]
        if self.shared_track_sources:
            lines.append(f"tracks shared between sources: {self.shared_track_sources}")
        return "; ".join(lines)


def _largest_gap_seconds(boundary_offsets: tuple[float, float], update_offsets: list[float]) -> float:
    first_solve_offset, last_solve_offset = boundary_offsets
    timeline = [first_solve_offset, *sorted(update_offsets), last_solve_offset]
    return float(max(np.diff(timeline))) if len(timeline) > 1 else 0.0


def score_track_updates(
    scene: ContinuousSourceScene,
    observations: list[TrackUpdateObservation],
    *,
    scene_start_ns: int,
) -> SceneTrackUpdateMetrics:
    applied_count_by_solve: Counter[str] = Counter()
    observations_by_solve: dict[str, list[TrackUpdateObservation]] = defaultdict(list)
    sources_by_track: dict[str, set[str]] = defaultdict(set)
    for observation in observations:
        applied_count_by_solve[observation.solve_id] += observation.update_count_increase
        observations_by_solve[observation.solve_id].append(observation)
        sources_by_track[observation.track_id].add(scene.solve(observation.solve_id).source_id)

    by_source: dict[str, SourceTrackUpdateMetrics] = {}
    for source_id in scene.sources:
        solves = scene.solves_for(source_id)
        metrics = SourceTrackUpdateMetrics(source_id=source_id, solve_count=len(solves))
        update_offsets: list[float] = []
        for solve in solves:
            applied_count = applied_count_by_solve[solve.solve_id]
            if applied_count == 0:
                metrics.missed_solve_ids.append(solve.solve_id)
                continue
            if applied_count > 1:
                metrics.multiply_applied_solve_ids.append(solve.solve_id)
            metrics.updated_solve_count += 1
            update_offsets.append(solve.solve_offset_seconds)
            first_application = observations_by_solve[solve.solve_id][0]
            window_centre_ns = scene_start_ns + int(round(solve.solve_offset_seconds * NANOSECONDS_PER_SECOND))
            if first_application.track_last_seen_ns != window_centre_ns:
                metrics.misstamped_solve_ids.append(solve.solve_id)
            metrics.track_ids_in_update_order.append(first_application.track_id)
        metrics.identity_switch_count = sum(
            1
            for previous, current in zip(metrics.track_ids_in_update_order, metrics.track_ids_in_update_order[1:])
            if previous != current
        )
        if solves:
            metrics.max_update_gap_seconds = _largest_gap_seconds(
                (solves[0].solve_offset_seconds, solves[-1].solve_offset_seconds), update_offsets
            )
        by_source[source_id] = metrics
    shared = {
        track_id: sorted(sources) for track_id, sources in sources_by_track.items() if len(sources) > 1
    }
    return SceneTrackUpdateMetrics(by_source=by_source, shared_track_sources=shared)


def drone_update_gap_gate_failures(
    scene: ContinuousSourceScene, metrics: SceneTrackUpdateMetrics, *, max_update_gap_seconds: float
) -> list[str]:
    """Every drone solve updates its drone's one track, stamped with the solve
    time, with no gap over the bound and no track shared with another source."""
    failures: list[str] = []
    for source_id in scene.source_ids_of_kind(SOURCE_KIND_DRONE):
        source_metrics = metrics.by_source[source_id]
        if source_metrics.max_update_gap_seconds > max_update_gap_seconds:
            failures.append(
                f"{source_id}: largest update gap {source_metrics.max_update_gap_seconds:.2f} s "
                f"> bound {max_update_gap_seconds:.2f} s"
            )
        if source_metrics.missed_solve_ids:
            failures.append(
                f"{source_id}: {len(source_metrics.missed_solve_ids)}/{source_metrics.solve_count} solves "
                f"never updated a track"
            )
        if source_metrics.multiply_applied_solve_ids:
            failures.append(f"{source_id}: solves applied more than once {source_metrics.multiply_applied_solve_ids}")
        if source_metrics.misstamped_solve_ids:
            failures.append(
                f"{source_id}: {len(source_metrics.misstamped_solve_ids)} updates not stamped with the solve time"
            )
        if len(source_metrics.distinct_track_ids) > 1:
            failures.append(
                f"{source_id}: split over tracks {source_metrics.distinct_track_ids} "
                f"({source_metrics.identity_switch_count} switches)"
            )
        for track_id in source_metrics.distinct_track_ids:
            if track_id in metrics.shared_track_sources:
                failures.append(f"{source_id}: track {track_id} also updated by {metrics.shared_track_sources[track_id]}")
    return failures
