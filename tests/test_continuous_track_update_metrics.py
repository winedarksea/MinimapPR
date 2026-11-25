"""Scoring behind the E6 drone update-gap gate, on hand-built update records."""

from __future__ import annotations

import pytest

from minimappr.sim.continuous_source_scenes import (
    DELIVERY_COVERING_CLASSIFIER_RENDER,
    DELIVERY_EVENT_LOCALIZATION_RESULT,
    SceneTiming,
    drone_fly_by,
)
from minimappr.sim.continuous_track_update_metrics import (
    NANOSECONDS_PER_SECOND,
    TrackUpdateObservation,
    drone_update_gap_gate_failures,
    score_track_updates,
)

SCENE_START_NS = 1_000 * NANOSECONDS_PER_SECOND
TIMING = SceneTiming(solve_interval_seconds=0.8, render_interval_seconds=28.0)
GAP_BOUND_SECONDS = 1.2


def _scene():
    return drone_fly_by(TIMING, duration_seconds=4.0, reporter_ids=("node-a", "node-b"))


def _update(index, solve, track_id, *, kind=DELIVERY_EVENT_LOCALIZATION_RESULT, stamp_offset_seconds=None):
    offset = solve.solve_offset_seconds if stamp_offset_seconds is None else stamp_offset_seconds
    return TrackUpdateObservation(
        delivery_index=index,
        delivery_kind=kind,
        solve_id=solve.solve_id,
        track_id=track_id,
        update_count_increase=1,
        track_last_seen_ns=SCENE_START_NS + int(round(offset * NANOSECONDS_PER_SECOND)),
    )


def test_every_solve_updating_one_track_passes_the_gate() -> None:
    scene = _scene()
    observations = [_update(index, solve, "trk-1") for index, solve in enumerate(scene.solves)]
    metrics = score_track_updates(scene, observations, scene_start_ns=SCENE_START_NS)
    drone = metrics.by_source["drone-1"]
    assert drone.updated_solve_count == drone.solve_count == len(scene.solves)
    assert drone.max_update_gap_seconds <= TIMING.largest_solve_interval_seconds + 1e-9
    assert drone_update_gap_gate_failures(scene, metrics, max_update_gap_seconds=GAP_BOUND_SECONDS) == []


def test_a_never_updated_drone_scores_its_whole_audible_span() -> None:
    scene = _scene()
    metrics = score_track_updates(scene, [], scene_start_ns=SCENE_START_NS)
    drone = metrics.by_source["drone-1"]
    assert drone.max_update_gap_seconds == pytest.approx(scene.solves[-1].solve_offset_seconds)
    assert len(drone.missed_solve_ids) == len(scene.solves)
    assert drone_update_gap_gate_failures(scene, metrics, max_update_gap_seconds=GAP_BOUND_SECONDS)


def test_one_skipped_solve_breaks_the_bound() -> None:
    scene = _scene()
    observations = [_update(index, solve, "trk-1") for index, solve in enumerate(scene.solves) if index != 2]
    metrics = score_track_updates(scene, observations, scene_start_ns=SCENE_START_NS)
    assert metrics.by_source["drone-1"].max_update_gap_seconds > GAP_BOUND_SECONDS
    assert metrics.by_source["drone-1"].missed_solve_ids == [scene.solves[2].solve_id]


def test_double_application_misstamping_and_track_splits_are_reported() -> None:
    scene = _scene()
    first, second, *rest = scene.solves
    observations = [
        _update(0, first, "trk-1"),
        _update(1, first, "trk-1", kind=DELIVERY_COVERING_CLASSIFIER_RENDER),
        _update(2, second, "trk-2", stamp_offset_seconds=second.solve_offset_seconds + 0.5),
        *(_update(3 + index, solve, "trk-1") for index, solve in enumerate(rest)),
    ]
    metrics = score_track_updates(scene, observations, scene_start_ns=SCENE_START_NS)
    drone = metrics.by_source["drone-1"]
    assert drone.multiply_applied_solve_ids == [first.solve_id]
    assert drone.misstamped_solve_ids == [second.solve_id]
    assert drone.distinct_track_ids == ["trk-1", "trk-2"]
    assert drone.identity_switch_count == 2
    failures = drone_update_gap_gate_failures(scene, metrics, max_update_gap_seconds=GAP_BOUND_SECONDS)
    assert len(failures) == 3
