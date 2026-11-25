"""Milestone 3 review fixes for the one-observation-per-solve pipeline (P5).

The hold rule (a fix gating only a labelled stationary_hold track waits for its
classification), the label gates re-run at attach, hidden-track visibility,
abstentions, the evicted-solve memory and the shutdown / ordering nits. Cases
that started as the reviewer's probes say so.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from minimappr.config import Settings
from minimappr.core.assembly_track_resolution import label_observed_solve_track_without_report
from minimappr.core.fusion_node import FusionMetrics
from minimappr.core import localized_render_localization
from minimappr.core.localized_render_handoff_hooks import (
    SOLVE_ORIGIN_PYTHON_INGEST,
    SOLVE_ORIGIN_RUST_EVENT_LOCALIZATION,
)
from minimappr.core.localized_render_localization import solve_timing_sample_rate_hz
from minimappr.core.multi_node_bearing_fusion import BearingFusionStore, BearingObservation
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.position_observation_config import position_observation_setting_errors
from minimappr.core.position_observation_pipeline import SolveLedgerState
from minimappr.core.tracking import TrackManager
from minimappr.sim.continuous_source_scenes import (
    DELIVERY_COVERING_CLASSIFIER_RENDER,
    DELIVERY_EVENT_LOCALIZATION_RESULT,
    SceneDelivery,
    drone_fly_by,
)
from tests.continuous_source_manifests import SCENE_REPORTER_IDS
from tests.continuous_source_tracking_harness import (
    default_scene_timing,
    deliver_through_stream_consumer,
    started_scene_fusion,
)
from tests.test_localized_render_handoff_seams import _render, _started_fusion
from tests.test_position_observation_pipeline import (
    T0_NS,
    _classified_kwargs,
    _localized_product,
    _observe,
    _pipeline,
    _visible_track,
)

SECOND_NS = 1_000_000_000
ROBIN = "American Robin"
THRUSH = "Hermit Thrush"
WILDLIFE = "wildlife"
# Short enough for a test to wait out, long enough not to fire mid-step.
TEST_HELD_FIX_MAX_WAIT_SECONDS = 0.05


def _bird_kwargs(position_m, label: str, timestamp_ns: int, confidence: float = 0.8) -> dict:
    kwargs = _classified_kwargs(position_m, label=label, label_category=WILDLIFE, timestamp_ns=timestamp_ns)
    kwargs["confidence"] = confidence
    kwargs["classifier_scores"] = {label: confidence}
    return kwargs


def _abstention_kwargs(position_m, timestamp_ns: int = T0_NS) -> dict:
    kwargs = _classified_kwargs(position_m, label="unknown", label_category="unknown", timestamp_ns=timestamp_ns)
    kwargs["confidence"] = 0.0
    kwargs["classifier_scores"] = {}
    return kwargs


async def _perched_robin(tracker: TrackManager, position_m=(0.0, 0.0, 3.0)):
    return await tracker.update_with_classification(**_bird_kwargs(position_m, ROBIN, T0_NS - 2 * SECOND_NS))


async def _observe_python(pipeline, candidate_id: str, position_m, event_time_ns: int = T0_NS) -> None:
    await _observe(
        pipeline,
        _localized_product(candidate_id=candidate_id, position_m=position_m, event_time_ns=event_time_ns),
        SOLVE_ORIGIN_PYTHON_INGEST,
    )
    await pipeline.drain()


# -- hold rule ------------------------------------------------------------------


async def test_fix_on_a_perched_track_waits_for_its_label_then_applies_through_update() -> None:
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(tracker, metrics)
    robin = await _perched_robin(tracker)
    await _observe_python(pipeline, "evt-robin", (0.3, 0.0, 3.0))
    assert metrics.position_observations_held_for_label == 1
    assert (await tracker.snapshot())[0].update_count == 1

    track = await pipeline.track_for_classified_solve(
        solve_id="evt-robin", classified_update_kwargs=_bird_kwargs((0.3, 0.0, 3.0), ROBIN, T0_NS)
    )
    assert track.id == robin.id and track.update_count == 2
    assert metrics.position_observations_held_applied_with_label == 1


async def test_abstention_drops_a_held_fix_without_touching_the_track() -> None:
    """The gate_collinear_trail regression: an unsettled attribution must not move a perched bird."""
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(tracker, metrics)
    robin = await _perched_robin(tracker)
    position_before = robin.position_m
    await _observe_python(pipeline, "evt-overlap", (2.0, 0.0, 3.0))

    assert await pipeline.track_for_classified_solve(
        solve_id="evt-overlap", classified_update_kwargs=_abstention_kwargs((2.0, 0.0, 3.0))
    ) is None
    [track] = await tracker.snapshot()
    assert (track.update_count, track.position_m, track.label) == (1, position_before, ROBIN)
    assert metrics.position_observations_held_dropped == 1
    assert pipeline._ledger["evt-overlap"].state is SolveLedgerState.HELD_DROPPED


async def test_found_nothing_trackable_drops_a_held_fix() -> None:
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(tracker, metrics)
    await _perched_robin(tracker)
    await _observe_python(pipeline, "evt-empty", (0.3, 0.0, 3.0))
    await pipeline.classification_found_nothing_trackable("evt-empty")
    assert (await tracker.snapshot())[0].update_count == 1
    assert metrics.position_observations_held_dropped == 1


async def test_held_fix_without_a_classification_applies_geometry_only() -> None:
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(tracker, metrics)
    robin = await _perched_robin(tracker)
    await _observe_python(pipeline, "evt-shed", (0.3, 0.0, 3.0))

    await pipeline.release_held_position_without_classification("evt-shed")
    [track] = await tracker.snapshot()
    assert track.id == robin.id and track.update_count == 2 and track.label == ROBIN
    assert metrics.position_observations_held_applied_geometry_only == 1
    # Its classification can no longer arrive; a late one would only attach.
    await pipeline.release_held_position_without_classification("evt-shed")
    assert metrics.position_observations_held_applied_geometry_only == 1


async def test_held_fix_past_its_deadline_applies_geometry_only() -> None:
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(
        tracker, metrics, position_observation_held_fix_max_wait_seconds=TEST_HELD_FIX_MAX_WAIT_SECONDS
    )
    await _perched_robin(tracker)
    await _observe_python(pipeline, "evt-uncovered", (0.3, 0.0, 3.0))
    await asyncio.sleep(TEST_HELD_FIX_MAX_WAIT_SECONDS * 4)
    await pipeline.drain()
    assert (await tracker.snapshot())[0].update_count == 2
    assert metrics.position_observations_held_applied_geometry_only == 1


async def test_queue_eviction_releases_a_held_fix_geometry_only() -> None:
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(tracker, metrics)
    await _perched_robin(tracker)
    await _observe_python(pipeline, "evt-evicted", (0.3, 0.0, 3.0))
    pipeline.schedule_held_position_release("evt-evicted")
    await pipeline.drain()
    assert (await tracker.snapshot())[0].update_count == 2


async def test_moving_hidden_and_event_only_fixes_are_never_held() -> None:
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(tracker, metrics)
    drone = await _visible_track(tracker, (50.0, 0.0, 10.0))
    await _observe_python(pipeline, "evt-drone", (50.5, 0.0, 10.0))
    await _observe_python(pipeline, "evt-hidden-start", (-50.0, 0.0, 3.0))
    await _observe_python(pipeline, "evt-hidden-join", (-50.3, 0.0, 3.0), event_time_ns=T0_NS + SECOND_NS)
    robin = await _perched_robin(tracker)
    await _observe(
        pipeline,
        _localized_product(candidate_id="rust-evt", position_m=(0.3, 0.0, 3.0), solve_id="evt-only-1"),
        SOLVE_ORIGIN_RUST_EVENT_LOCALIZATION,
    )
    await pipeline.drain()

    counts = {track.id: track.update_count for track in await tracker.snapshot(include_hidden=True)}
    assert counts[drone.id] == 2 and counts[robin.id] == 2
    assert sorted(counts.values()) == [2, 2, 2]
    assert metrics.position_observations_held_for_label == 0


async def test_released_old_fix_never_moves_last_seen_backwards() -> None:
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(tracker, metrics)
    await _perched_robin(tracker)
    await _observe_python(pipeline, "evt-old", (0.3, 0.0, 3.0))
    newer = T0_NS + 5 * SECOND_NS
    await _observe_python(pipeline, "evt-new", (0.2, 0.0, 3.0), event_time_ns=newer)
    await pipeline.track_for_classified_solve(
        solve_id="evt-new", classified_update_kwargs=_bird_kwargs((0.2, 0.0, 3.0), ROBIN, newer)
    )
    await pipeline.release_held_position_without_classification("evt-old")
    [track] = await tracker.snapshot()
    assert track.update_count == 3
    assert track.last_seen_ns == newer


# -- finding 1: label-dependent association gates -------------------------------


async def test_held_fix_of_another_species_starts_its_own_track_not_the_perched_one() -> None:
    """Reviewer probe: a perched robin and an unclassified Hermit Thrush fix 2.8 m away."""
    settings = {"tracking_species_log_odds_enabled": True}
    tracker, metrics = TrackManager(Settings(**settings)), FusionMetrics()
    pipeline = _pipeline(tracker, metrics, **settings)
    await tracker.update_with_classification(**_bird_kwargs((0.0, 0.0, 3.0), ROBIN, T0_NS - 4 * SECOND_NS))
    robin = await tracker.update_with_classification(**_bird_kwargs((0.2, 0.0, 3.0), ROBIN, T0_NS - 2 * SECOND_NS))
    robin_position = robin.position_m
    await _observe_python(pipeline, "c-thrush", (3.0, 0.0, 3.0))

    thrush = await pipeline.track_for_classified_solve(
        solve_id="c-thrush", classified_update_kwargs=_bird_kwargs((3.0, 0.0, 3.0), THRUSH, T0_NS)
    )
    tracks = {track.id: track for track in await tracker.snapshot()}
    assert thrush.id != robin.id and thrush.label == THRUSH
    assert tracks[robin.id].label == ROBIN and tracks[robin.id].position_m == robin_position


async def test_attach_reruns_the_species_gate_and_treats_a_rejection_as_a_contradiction() -> None:
    settings = {"tracking_species_log_odds_enabled": True}
    tracker, metrics = TrackManager(Settings(**settings)), FusionMetrics()
    pipeline = _pipeline(tracker, metrics, **settings)
    # Both fixes land on one hidden track; the first is classified a confident robin.
    await _observe_python(pipeline, "evt-a", (0.0, 0.0, 3.0))
    await _observe_python(pipeline, "evt-b", (0.4, 0.0, 3.0), event_time_ns=T0_NS + SECOND_NS)
    for _ in range(3):
        robin = await pipeline.track_for_classified_solve(
            solve_id="evt-a", classified_update_kwargs=_bird_kwargs((0.0, 0.0, 3.0), ROBIN, T0_NS, confidence=0.95)
        )
    thrush = await pipeline.track_for_classified_solve(
        solve_id="evt-b",
        classified_update_kwargs=_bird_kwargs((0.4, 0.0, 3.0), THRUSH, T0_NS + SECOND_NS, confidence=0.95),
    )
    assert thrush.id != robin.id and thrush.label == THRUSH
    assert {track.id: track.label for track in await tracker.snapshot()}[robin.id] == ROBIN


# -- findings 2 and 3: hidden tracks and abstentions ----------------------------


async def test_dropped_hidden_track_stays_hidden_until_reaped() -> None:
    """Reviewer probe: aged to DROPPED, then the next observation used to prune it visible."""
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(tracker, metrics)
    await _observe_python(pipeline, "c-a", (50.0, 0.0, 3.0))
    await _observe_python(pipeline, "c-a2", (50.2, 0.0, 3.0), event_time_ns=T0_NS + SECOND_NS)
    _, drop_ns, reap_ns = tracker._lifecycle_ns("unknown")
    between_drop_and_reap = T0_NS + SECOND_NS + (drop_ns + reap_ns) // 2
    await _observe_python(pipeline, "c-b", (-50.0, 0.0, 3.0), event_time_ns=between_drop_and_reap)

    assert await tracker.snapshot(now_ns=between_drop_and_reap + 1) == []
    assert tracker._dormant == {}  # never parked: an abstention could revive it visible


async def test_abstention_retracts_the_hidden_track_its_own_fix_started() -> None:
    """Reviewer probe 2: an 'unknown', conf-0 classification revealed the track."""
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(tracker, metrics)
    await _observe_python(pipeline, "c-a", (50.0, 0.0, 3.0))
    assert await pipeline.track_for_classified_solve(
        solve_id="c-a", classified_update_kwargs=_abstention_kwargs((50.0, 0.0, 3.0))
    ) is None
    assert await tracker.snapshot(include_hidden=True) == []
    assert metrics.position_observation_provisional_tracks_retracted == 1


async def test_abstention_on_a_joined_hidden_track_keeps_it_hidden() -> None:
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(tracker, metrics)
    await _observe_python(pipeline, "c-a", (50.0, 0.0, 3.0))
    await _observe_python(pipeline, "c-b", (50.2, 0.0, 3.0), event_time_ns=T0_NS + SECOND_NS)
    track = await pipeline.track_for_classified_solve(
        solve_id="c-b", classified_update_kwargs=_abstention_kwargs((50.2, 0.0, 3.0), T0_NS + SECOND_NS)
    )
    assert track is not None and tracker.is_track_hidden(track.id) and track.label == "unknown"
    assert await tracker.snapshot() == []


async def test_abstention_never_overwrites_a_real_label() -> None:
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(tracker, metrics)
    drone = await _visible_track(tracker, (50.0, 0.0, 10.0))
    await _observe_python(pipeline, "c-drone", (50.5, 0.0, 10.0))
    track = await pipeline.track_for_classified_solve(
        solve_id="c-drone", classified_update_kwargs=_abstention_kwargs((50.5, 0.0, 10.0))
    )
    assert track.id == drone.id
    assert (track.label, track.label_category, track.confidence) == ("drone", "aircraft", drone.confidence)
    assert track.update_count == 2


# -- finding 7: evicted-solve memory --------------------------------------------


async def test_evicted_solve_is_still_a_duplicate_and_its_label_only_attaches() -> None:
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(
        tracker, metrics, position_observation_queue_max_items=1, position_observation_solve_ledger_max_entries=1
    )
    drone = await _visible_track(tracker, (50.0, 0.0, 10.0))
    await _observe_python(pipeline, "solve-a", (50.5, 0.0, 10.0))
    await _observe_python(pipeline, "solve-b", (-50.0, 0.0, 3.0))
    assert "solve-a" not in pipeline._ledger

    # The covering render of the evicted solve, then its classification.
    await _observe_python(pipeline, "solve-a", (50.5, 0.0, 10.0))
    assert metrics.position_observations_duplicate_solve == 1
    track = await pipeline.track_for_classified_solve(
        solve_id="solve-a", classified_update_kwargs=_classified_kwargs((50.5, 0.0, 10.0))
    )
    assert track.id == drone.id and track.update_count == 2


def test_solve_id_memory_must_outlive_the_ledger() -> None:
    errors = position_observation_setting_errors(
        SimpleNamespace(
            position_observation_solve_ledger_max_entries=4096,
            position_observation_queue_max_items=1024,
            position_observation_solve_id_memory_max_entries=10,
            position_observation_held_fix_max_wait_seconds=30.0,
        )
    )
    assert any("solve_id_memory" in error for error in errors)


# -- finding 8: unreported classifications still label their solve's track ------


async def test_suppressed_classification_reveals_the_hidden_track_of_its_solve() -> None:
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(tracker, metrics)
    await _observe_python(pipeline, "c-second-bird", (50.0, 0.0, 3.0))
    await label_observed_solve_track_without_report(
        tracker=tracker,
        registry=NodeRegistry(),
        solve_id="c-second-bird",
        position_observations=pipeline,
        capability_tier="full_3d",
        event_time_ns=T0_NS,
        localization_position_m=(50.0, 0.0, 3.0),
        localization_position_covariance_m2=None,
        classification_label=ROBIN,
        classification_confidence=0.8,
        classification_scores={ROBIN: 0.8},
        classification_features={},
        label_category=WILDLIFE,
        iff_category="unknown",
        label_id=None,
        selected_sensor_ids=["node-a-s0"],
        source_node_id="node-a",
        reference_sensor="node-a-s0",
        tdoa_s={},
        environment={},
    )
    [visible] = await tracker.snapshot()
    assert visible.label == ROBIN and visible.update_count == 1


def _always_suppress(decide):
    async def suppressing_decide(**kwargs):
        decision = await decide(**kwargs)
        decision.action = "suppress"
        return decision

    return suppressing_decide


async def test_reporting_suppress_decision_still_labels_the_solve_track(tmp_path: Path) -> None:
    fusion, storage = await _started_fusion(tmp_path)
    try:
        fusion.reporting_policy.decide = _always_suppress(fusion.reporting_policy.decide)
        # Heuristic "bird_like" is wildlife, which the entity coordinator routes.
        if fusion.entity_reporting is not None:
            fusion.entity_reporting.decide = _always_suppress(fusion.entity_reporting.decide)
        await fusion.ingest_localized_render(_render("manifest-suppressed", localization_position_m=(3.2, 4.0, 0.5)))
        await fusion._position_observations().drain()
        assert await storage.list_detections(limit=10) == []
        [track] = await fusion.tracker.snapshot()
        assert track.label != "unknown"
    finally:
        await fusion.stop()
        await storage.close()


async def test_chunk_suppressed_render_releases_its_held_fix_geometry_only(tmp_path: Path) -> None:
    fusion, storage = await _started_fusion(tmp_path)
    try:
        robin = await fusion.tracker.update_with_classification(
            **_bird_kwargs((3.0, 4.0, 0.5), ROBIN, _render("unused").event_time_ns - SECOND_NS)
        )
        fusion._classification_chunking_policy = object()
        fusion._should_suppress_chunked_classification = lambda product: True
        await fusion.ingest_localized_render(_render("manifest-held", localization_position_m=(3.2, 4.0, 0.5)))
        await fusion._position_observations().drain()
        [track] = await fusion.tracker.snapshot()
        assert track.id == robin.id and track.update_count == 2 and track.label == ROBIN
        # Abandoned before the drain held it, or released after: geometry-only either way.
        metrics = fusion._metrics
        assert metrics.position_observations_applied + metrics.position_observations_held_applied_geometry_only == 1
    finally:
        await fusion.stop()
        await storage.close()


# -- findings 4, 5, 6: render handoff --------------------------------------------


def test_solve_timing_uses_the_solve_rate_not_the_render_audio_rate() -> None:
    solve = SimpleNamespace(provenance=SimpleNamespace(solve_sample_rate_hz=16_000))
    assert solve_timing_sample_rate_hz(SimpleNamespace(localization_solve=solve, sample_rate_hz=48_000)) == 16_000
    assert solve_timing_sample_rate_hz(SimpleNamespace(localization_solve=None, sample_rate_hz=48_000)) == 48_000


async def test_python_re_solve_of_an_event_runs_off_the_event_loop(tmp_path: Path, monkeypatch) -> None:
    solver_threads: list[bool] = []
    original_solver = localized_render_localization.solve_localization_from_rust_tdoas

    def recording_solver(*args, **kwargs):
        solver_threads.append(threading.current_thread() is threading.main_thread())
        return original_solver(*args, **kwargs)

    monkeypatch.setattr(localized_render_localization, "solve_localization_from_rust_tdoas", recording_solver)
    scene = drone_fly_by(default_scene_timing(), reporter_ids=SCENE_REPORTER_IDS)
    fusion, storage = await started_scene_fusion(tmp_path, localization_single_node_solver="python_cartesian")
    try:
        delivery = SceneDelivery(0.0, scene.solves[0].solve_id, DELIVERY_EVENT_LOCALIZATION_RESULT)
        await deliver_through_stream_consumer(fusion, scene, [delivery])
        # Solved in a worker thread, then handed to the observation sink in order.
        assert solver_threads == [False]
        assert fusion._metrics.position_observations_received == 1
    finally:
        await fusion.stop()
        await storage.close()


async def test_covering_render_does_not_recount_its_event_rejection(tmp_path: Path) -> None:
    scene = drone_fly_by(default_scene_timing(), reporter_ids=SCENE_REPORTER_IDS)
    solve_id = scene.solves[0].solve_id
    fusion, storage = await started_scene_fusion(tmp_path)
    fusion._localization_quality_is_reportable = lambda **kwargs: "test_low_quality"
    try:
        await deliver_through_stream_consumer(
            fusion,
            scene,
            [
                SceneDelivery(0.0, solve_id, DELIVERY_EVENT_LOCALIZATION_RESULT),
                SceneDelivery(0.0, solve_id, DELIVERY_COVERING_CLASSIFIER_RENDER),
            ],
        )
        assert fusion._metrics.localization_rejected_low_quality_count == 1
    finally:
        await fusion.stop()
        await storage.close()


async def test_one_cone_per_node_and_event_in_the_bearing_store() -> None:
    store = BearingFusionStore()
    cone = BearingObservation(
        node_id="node-a",
        origin_m=np.zeros(3),
        direction=np.array([1.0, 0.0, 0.0]),
        lateral_std_m=1.0,
        confidence=0.8,
        event_time_ns=T0_NS,
        range_prior_m=10.0,
        expiry_ns=T0_NS + 10 * SECOND_NS,
    )
    await store.register(cone, now_ns=T0_NS)
    await store.register(cone, now_ns=T0_NS)
    assert len(store._observations) == 1


# -- nits ------------------------------------------------------------------------


async def test_close_cancels_the_drain_task_and_held_deadlines() -> None:
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(tracker, metrics)
    await _perched_robin(tracker)
    await _observe_python(pipeline, "evt-held", (0.3, 0.0, 3.0))
    entry = pipeline._ledger["evt-held"]
    assert entry.held_deadline is not None
    await _observe(pipeline, _localized_product(candidate_id="evt-pending", position_m=(90.0, 0.0, 3.0)), SOLVE_ORIGIN_PYTHON_INGEST)
    drain_task = pipeline._drain_task
    pipeline.close()
    await asyncio.sleep(0)
    assert drain_task.cancelled() or drain_task.done()
    assert entry.held_deadline is None


async def test_untracked_category_never_ends_a_labelled_track_with_the_category_gate_off() -> None:
    settings = {
        "association_category_gate_enabled": False,
        "tracking_mode_by_category": {**Settings().tracking_mode_by_category, "vehicle": "none"},
    }
    tracker, metrics = TrackManager(Settings(**settings)), FusionMetrics()
    pipeline = _pipeline(tracker, metrics, **settings)
    drone = await _visible_track(tracker, (50.0, 0.0, 10.0))
    await _observe_python(pipeline, "c-truck", (50.5, 0.0, 10.0))
    assert await pipeline.track_for_classified_solve(
        solve_id="c-truck",
        classified_update_kwargs=_classified_kwargs((50.5, 0.0, 10.0), label="truck", label_category="vehicle"),
    ) is None
    [kept] = await tracker.snapshot()
    assert kept.id == drone.id and kept.label == "drone"
