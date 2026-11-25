"""Continuous-source tracking (localization overhaul E6, milestone 3 gate).

Scenes (``minimappr/sim/continuous_source_scenes.py``) go through the
sidecar manifest path into a real FusionNode
(``tests/continuous_source_tracking_harness.py``); the queue-pressure case
drives the Python-ingest worker loops
(``tests/same_label_callers_queue_pressure.py``).

Drone update-gap gate, fixed before measuring:

- Solve cadence C = ``Settings.trigger_cooldown_seconds`` (0.8 s): a drone is a
  continuous source, so it re-arms the live trigger on every dispatch and the
  sidecar emits one event solve per C, on the next 100 ms upload boundary. The
  scenes' solve intervals are therefore C or C + 0.1 s.
- Bound B = 1.5 C (1.2 s). It clears the largest solve interval (0.9 s) and
  sits under 2 C (1.6 s), the smallest gap one skipped solve can leave, so B is
  exceeded by any missed update.
- Gap = largest interval between a drone's track updates (event time), from
  its first solve to its last, so a drone never updated scores its whole
  audible span.
- With it: every drone solve updates exactly one track, once, stamped with the
  solve's window centre; each drone stays on one track (hover included) that
  no other source updates (no swap between the crossing drones).

Companion checks: drone tracks move with a velocity matching truth
(continuous_motion) and birds next to the drone hold position with no
velocity (stationary_hold, P4).

Long scenes (120 s, several renders and reporting windows) are opt-in with
``MINIMAPPR_ENABLE_CONTINUOUS_TRACKING_LONG_SCENES=1``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from minimappr.config import Settings
from minimappr.sim.continuous_source_scenes import (
    DELIVERY_COVERING_CLASSIFIER_RENDER,
    DELIVERY_EVENT_LOCALIZATION_RESULT,
    DELIVERY_RENDER_LOCALIZATION_RESULT,
    DRONE_LABEL,
    LONG_SCENE_SECONDS,
    SCENE_BUILDERS,
    SOURCE_KIND_BIRD,
    SOURCE_KIND_DRONE,
    ContinuousSourceScene,
    SceneDelivery,
    drone_fly_by,
    two_same_label_callers_in_one_cluster,
)
from minimappr.sim.continuous_track_update_metrics import drone_update_gap_gate_failures, score_track_updates
from tests.continuous_source_manifests import SCENE_REPORTER_IDS, SCENE_START_NS, window_centre_ns
from tests.continuous_source_tracking_harness import (
    SceneRunRecord,
    default_scene_timing,
    deliver_through_stream_consumer,
    run_scene,
    started_scene_fusion,
)
from tests.same_label_callers_queue_pressure import run_callers_under_classification_queue_pressure

DRONE_SOLVE_CADENCE_SECONDS = Settings().trigger_cooldown_seconds
DRONE_UPDATE_GAP_BOUND_SECONDS = 1.5 * DRONE_SOLVE_CADENCE_SECONDS
LONG_SCENES_ENV = "MINIMAPPR_ENABLE_CONTINUOUS_TRACKING_LONG_SCENES"

# continuous_motion: the track's velocity tracks the drone's. Loose enough for
# a constant-velocity Kalman on 1 m fixes; a held (zero) or frozen velocity fails.
CONTINUOUS_MOTION_SPEED_RATIO_RANGE = (0.65, 1.35)
CONTINUOUS_MOTION_MAX_HEADING_ERROR_DEG = 20.0
MIN_TRACK_UPDATES_FOR_MOTION_CHECK = 10
# stationary_hold carries no velocity state at all.
STATIONARY_HOLD_MAX_SPEED_MPS = 1e-6
STATIONARY_HOLD_MAX_PERCH_ERROR_STDS = 3.0
MIN_TRACK_UPDATES_FOR_HOLD_CHECK = 2
WILDLIFE_CATEGORY = "wildlife"
CALLER_TRACK_MATCH_RADIUS_M = 5.0

GATE_SCENE_NAMES = ("drone_fly_by", "hovering_drone", "crossing_drones", "drone_with_birds_calling")


class DroneUpdateGapGateViolation(AssertionError):
    """Raised only by a gate verdict, never by a harness failure."""


class TrackingModeViolation(AssertionError):
    pass


class DeliveryContractViolation(AssertionError):
    pass


def _build_scene(name: str, *, duration_seconds: float | None = None) -> ContinuousSourceScene:
    duration = {} if duration_seconds is None else {"duration_seconds": duration_seconds}
    return SCENE_BUILDERS[name](default_scene_timing(), reporter_ids=SCENE_REPORTER_IDS, **duration)


def _raise_if(failures: list[str], violation: type[AssertionError], context: str) -> None:
    if failures:
        raise violation(f"{context}: " + "; ".join(failures))


def _assert_drone_update_gap_gate(scene: ContinuousSourceScene, record: SceneRunRecord) -> None:
    metrics = score_track_updates(scene, record.observations, scene_start_ns=SCENE_START_NS)
    failures = drone_update_gap_gate_failures(scene, metrics, max_update_gap_seconds=DRONE_UPDATE_GAP_BOUND_SECONDS)
    _raise_if(failures, DroneUpdateGapGateViolation, f"{scene.name} [{metrics.summary()}]")


def test_scene_cadence_sits_inside_the_gate_bound_premises() -> None:
    """Every scene's solve intervals clear B, and one skipped solve would not."""
    for name in SCENE_BUILDERS:
        scene = _build_scene(name)
        for source_id in scene.source_ids_of_kind(SOURCE_KIND_DRONE):
            intervals = np.diff([solve.solve_offset_seconds for solve in scene.solves_for(source_id)])
            assert float(intervals.max()) < DRONE_UPDATE_GAP_BOUND_SECONDS, (name, source_id)
            assert float((intervals[:-1] + intervals[1:]).min()) > DRONE_UPDATE_GAP_BOUND_SECONDS, (name, source_id)
        # Renders arrive after later solves of their source: late classification is in every scene.
        assert scene.timing.render_latency_seconds > scene.timing.largest_solve_interval_seconds


@pytest.mark.parametrize("scene_name", GATE_SCENE_NAMES)
async def test_drone_update_gap_gate(tmp_path: Path, scene_name: str) -> None:
    scene = _build_scene(scene_name)
    record = await run_scene(tmp_path, scene)
    _assert_drone_update_gap_gate(scene, record)


@pytest.mark.skipif(os.environ.get(LONG_SCENES_ENV) != "1", reason=f"long scenes are opt-in ({LONG_SCENES_ENV}=1)")
@pytest.mark.parametrize("scene_name", list(SCENE_BUILDERS))
async def test_long_scene_drone_update_gap_gate(tmp_path: Path, scene_name: str) -> None:
    scene = _build_scene(scene_name, duration_seconds=LONG_SCENE_SECONDS)
    record = await run_scene(tmp_path, scene)
    _assert_drone_update_gap_gate(scene, record)


def _heading_error_deg(estimated: np.ndarray, truth: np.ndarray) -> float:
    cosine = float(np.dot(estimated, truth) / (np.linalg.norm(estimated) * np.linalg.norm(truth)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


@pytest.mark.parametrize("scene_name", ["drone_fly_by", "crossing_drones"])
async def test_drone_tracks_use_continuous_motion(tmp_path: Path, scene_name: str) -> None:
    scene = _build_scene(scene_name)
    record = await run_scene(tmp_path, scene)
    metrics = score_track_updates(scene, record.observations, scene_start_ns=SCENE_START_NS)
    failures = []
    for source_id in scene.source_ids_of_kind(SOURCE_KIND_DRONE):
        track = record.track_for_source(source_id)
        updates = metrics.by_source[source_id].updated_solve_count
        if track is None or updates < MIN_TRACK_UPDATES_FOR_MOTION_CHECK:
            failures.append(f"{source_id}: {updates} track updates, need {MIN_TRACK_UPDATES_FOR_MOTION_CHECK}")
            continue
        if track.label != DRONE_LABEL:
            failures.append(f"{source_id}: track label {track.label!r}")
        truth = np.asarray(scene.solves_for(source_id)[-1].true_velocity_mps)
        estimated = np.asarray(track.velocity_mps)
        speed_ratio = float(np.linalg.norm(estimated) / np.linalg.norm(truth))
        low, high = CONTINUOUS_MOTION_SPEED_RATIO_RANGE
        if not low <= speed_ratio <= high:
            failures.append(f"{source_id}: speed ratio {speed_ratio:.2f} outside [{low}, {high}]")
        elif _heading_error_deg(estimated, truth) > CONTINUOUS_MOTION_MAX_HEADING_ERROR_DEG:
            failures.append(f"{source_id}: heading off by {_heading_error_deg(estimated, truth):.1f} deg")
    _raise_if(failures, TrackingModeViolation, scene_name)


async def test_birds_near_a_drone_use_stationary_hold(tmp_path: Path) -> None:
    scene = _build_scene("drone_with_birds_calling")
    record = await run_scene(tmp_path, scene)
    metrics = score_track_updates(scene, record.observations, scene_start_ns=SCENE_START_NS)
    failures = []
    for source_id in scene.source_ids_of_kind(SOURCE_KIND_BIRD):
        track = record.track_for_source(source_id)
        updates = metrics.by_source[source_id].updated_solve_count
        if track is None or updates < MIN_TRACK_UPDATES_FOR_HOLD_CHECK:
            failures.append(f"{source_id}: {updates} track updates, need {MIN_TRACK_UPDATES_FOR_HOLD_CHECK}")
            continue
        perch = scene.solves_for(source_id)[0]
        if (track.label, track.label_category) != (scene.sources[source_id].label, WILDLIFE_CATEGORY):
            failures.append(f"{source_id}: track labelled {track.label!r}/{track.label_category!r}")
        speed = float(np.linalg.norm(track.velocity_mps))
        if speed > STATIONARY_HOLD_MAX_SPEED_MPS:
            failures.append(f"{source_id}: held track moves at {speed:.3f} m/s")
        perch_error_m = math.dist(track.position_m, perch.true_position_m)
        if perch_error_m > STATIONARY_HOLD_MAX_PERCH_ERROR_STDS * perch.position_std_m:
            failures.append(f"{source_id}: held {perch_error_m:.1f} m off its perch")
    _raise_if(failures, TrackingModeViolation, scene.name)


# --- delivery cases ---------------------------------------------------------

DUPLICATE_DELIVERY_ORDERS = {
    "event_then_render_pair": (
        DELIVERY_EVENT_LOCALIZATION_RESULT,
        DELIVERY_RENDER_LOCALIZATION_RESULT,
        DELIVERY_COVERING_CLASSIFIER_RENDER,
    ),
    "event_then_render": (DELIVERY_EVENT_LOCALIZATION_RESULT, DELIVERY_COVERING_CLASSIFIER_RENDER),
    "event_then_render_then_its_localization_result": (
        DELIVERY_EVENT_LOCALIZATION_RESULT,
        DELIVERY_COVERING_CLASSIFIER_RENDER,
        DELIVERY_RENDER_LOCALIZATION_RESULT,
    ),
}


def _fly_by_scene() -> ContinuousSourceScene:
    return drone_fly_by(default_scene_timing(), reporter_ids=SCENE_REPORTER_IDS)


async def _deliver(tmp_path: Path, scene: ContinuousSourceScene, deliveries: list[SceneDelivery]) -> SceneRunRecord:
    fusion, storage = await started_scene_fusion(tmp_path)
    try:
        return await deliver_through_stream_consumer(fusion, scene, deliveries)
    finally:
        await fusion.stop()
        await storage.close()


@pytest.mark.parametrize("order_name", sorted(DUPLICATE_DELIVERY_ORDERS))
async def test_duplicate_delivery_of_one_solve_gives_exactly_one_track_update(tmp_path: Path, order_name: str) -> None:
    scene = _fly_by_scene()
    solve = scene.solves[0]
    deliveries = [SceneDelivery(0.0, solve.solve_id, kind) for kind in DUPLICATE_DELIVERY_ORDERS[order_name]]
    record = await _deliver(tmp_path, scene, deliveries)

    applied = sum(observation.update_count_increase for observation in record.observations)
    assert applied == 1, record.observations
    (track_id,) = record.tracks_updated_by_solve(solve.solve_id)
    track = record.final_tracks[track_id]
    assert track.update_count == 1
    assert track.last_seen_ns == window_centre_ns(solve)
    assert track.label == DRONE_LABEL


async def test_duplicate_delivery_applies_the_position_on_arrival_and_only_the_label_from_the_render(
    tmp_path: Path,
) -> None:
    scene = _fly_by_scene()
    solve = scene.solves[0]
    kinds = DUPLICATE_DELIVERY_ORDERS["event_then_render_pair"]
    record = await _deliver(tmp_path, scene, [SceneDelivery(0.0, solve.solve_id, kind) for kind in kinds])

    failures = []
    event_updates = [obs for obs in record.observations if obs.delivery_kind == DELIVERY_EVENT_LOCALIZATION_RESULT]
    if [obs.track_last_seen_ns for obs in event_updates] != [window_centre_ns(solve)]:
        failures.append(f"event delivery updates {event_updates}")
    later_updates = [obs for obs in record.observations if obs.delivery_kind != DELIVERY_EVENT_LOCALIZATION_RESULT]
    if later_updates:
        failures.append(f"render deliveries re-applied the position: {later_updates}")
    if event_updates and record.final_tracks[event_updates[0].track_id].label != DRONE_LABEL:
        failures.append("render label not attached to the event's track")
    _raise_if(failures, DeliveryContractViolation, "duplicate delivery")


async def test_late_classification_attaches_the_label_without_rewinding_the_track(tmp_path: Path) -> None:
    scene = _fly_by_scene()
    solves = scene.solves[:5]
    covered = solves[1]
    deliveries = [SceneDelivery(0.0, solve.solve_id, DELIVERY_EVENT_LOCALIZATION_RESULT) for solve in solves]
    deliveries += [
        SceneDelivery(0.0, covered.solve_id, DELIVERY_RENDER_LOCALIZATION_RESULT),
        SceneDelivery(0.0, covered.solve_id, DELIVERY_COVERING_CLASSIFIER_RENDER),
    ]
    record = await _deliver(tmp_path, scene, deliveries)

    track_ids = list(dict.fromkeys(observation.track_id for observation in record.observations))
    if len(track_ids) != 1:
        raise DeliveryContractViolation(f"expected one drone track, got {track_ids} ({record.observations})")
    track = record.final_tracks[track_ids[0]]
    failures = []
    if track.update_count != len(solves):
        failures.append(f"update_count {track.update_count}, expected {len(solves)} (one per solve)")
    if track.last_seen_ns != window_centre_ns(solves[-1]):
        failures.append(f"last_seen {track.last_seen_ns - SCENE_START_NS} ns after start, expected the latest solve")
    latest_error_m = math.dist(track.position_m, solves[-1].measured_position_m)
    covered_error_m = math.dist(track.position_m, covered.measured_position_m)
    if latest_error_m >= covered_error_m:
        failures.append(f"track at the covered solve ({covered_error_m:.1f} m) not the latest ({latest_error_m:.1f} m)")
    if track.label != DRONE_LABEL:
        failures.append(f"label {track.label!r} not attached")
    if track.track_id not in record.visible_track_ids:
        failures.append("classified track still hidden")
    _raise_if(failures, DeliveryContractViolation, "late classification")


async def test_two_same_label_callers_keep_updating_under_classification_queue_pressure(tmp_path: Path) -> None:
    scene = two_same_label_callers_in_one_cluster(default_scene_timing(), reporter_ids=SCENE_REPORTER_IDS)
    record = await run_callers_under_classification_queue_pressure(tmp_path, scene)
    assert record.classification_queue_drops > 0, "scene put no pressure on the classification queue"

    failures = []
    matched_track_ids = []
    for source_id in scene.sources:
        solves = scene.solves_for(source_id)
        perch_m = solves[0].true_position_m
        nearby = [
            track for track in record.final_tracks.values()
            if math.dist(track.position_m, perch_m) <= CALLER_TRACK_MATCH_RADIUS_M
        ]
        if len(nearby) != 1:
            failures.append(f"{source_id}: {len(nearby)} tracks at its perch")
            continue
        track = nearby[0]
        matched_track_ids.append(track.track_id)
        if track.update_count != len(solves):
            failures.append(f"{source_id}: {track.update_count}/{len(solves)} calls updated its track")
        if track.last_seen_ns != window_centre_ns(solves[-1]):
            failures.append(f"{source_id}: track not updated by its last call")
    if len(set(matched_track_ids)) != len(matched_track_ids):
        failures.append(f"callers share a track: {matched_track_ids}")
    _raise_if(failures, DeliveryContractViolation, f"queue pressure ({record.classification_queue_drops} shed)")
