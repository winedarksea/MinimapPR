"""P5: one tracker update per solve, ahead of every classification-side drop.

Covers the pipeline contract (idempotent by solve id, bounded drop-oldest FIFO,
no coalescing, hidden tentative start, event-triggered-only start) on a real
TrackManager, and the FusionNode wiring (a suppressed or backlog-dropped
detection still moved its track).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from minimappr.config import Settings
from minimappr.core.assembly_track_resolution import resolve_detection_track
from minimappr.core.fusion_node import EventCandidate, FusionMetrics
from minimappr.core.fusion_stage_carriers import LocalizationBranch, LocalizedCandidate
from minimappr.core.localization_overhaul_config import (
    POSITION_OBSERVATION_POLICY_REQUIRE_CLASSIFICATION,
)
from minimappr.core.localization_provenance import LOCALIZATION_SOLVE_FEATURE_KEY
from minimappr.core.localized_render_handoff_hooks import (
    SOLVE_ORIGIN_PYTHON_INGEST,
    SOLVE_ORIGIN_RUST_CLASSIFIER_RENDER,
)
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.position_observation import position_observation_solve_id
from minimappr.core.position_observation_config import (
    position_observation_config_from_settings,
)
from minimappr.core.position_observation_pipeline import (
    PositionObservationPipeline,
    SolveLedgerState,
)
from minimappr.core.tracking import TrackManager
from minimappr.models import TimeQuality
from minimappr.sim.continuous_source_scenes import (
    DELIVERY_EVENT_LOCALIZATION_RESULT,
    drone_fly_by,
)
from tests.continuous_source_manifests import SCENE_REPORTER_IDS
from tests.continuous_source_tracking_harness import (
    default_scene_timing,
    deliver_through_stream_consumer,
    started_scene_fusion,
)
from tests.test_localized_render_handoff_seams import _render, _started_fusion

T0_NS = 1_739_950_000_000_000_000
EVENT_TRIGGER_ATTEMPT_KIND = "event_trigger_absolute_rms"
SCHEDULED_RENDER_ATTEMPT_KIND = "scheduled_classifier_render"


def _localized_product(
    *,
    candidate_id: str,
    position_m: tuple[float, float, float],
    event_time_ns: int = T0_NS,
    solve_id: str | None = None,
    attempt_kind: str | None = EVENT_TRIGGER_ATTEMPT_KIND,
    cluster_id: str | None = None,
    reporting_node_id: str = "node-a",
) -> LocalizedCandidate:
    signal = np.zeros(16, dtype=np.float32)
    features = {}
    if solve_id is not None:
        features[LOCALIZATION_SOLVE_FEATURE_KEY] = {
            "solve_id": solve_id,
            "attempt_kind": attempt_kind,
            "cluster_id": cluster_id,
            "reporting_node_id": reporting_node_id,
            "window_center_time_ns": event_time_ns,
        }
    return LocalizedCandidate(
        candidate=EventCandidate(
            id=candidate_id,
            source_node_id=reporting_node_id,
            event_time_ns=event_time_ns,
            sample_rate_hz=48_000,
            source_type="raw_sensor",
            time_quality=TimeQuality.GPS_LOCKED,
            source_observation_ids=[],
        ),
        localization_branch=LocalizationBranch(
            localization_position_m=position_m,
            localization_confidence=0.9,
            localization_gdop=1.0,
            localization_position_covariance_m2=None,
            localization_range_observability=None,
            localization_residual_rms_seconds=None,
            localization_range_projection_mode="range_refined",
            reference_sensor=f"{reporting_node_id}-s0",
            reference_signal=signal,
            classification_reference_signal=signal,
            tdoa_s={},
            localization_method="test_solver",
            capability_tier="full_3d",
        ),
        selected_sensor_ids=[f"{reporting_node_id}-s{index}" for index in range(4)],
        selected_windows={},
        selected_positions={},
        classification_selected_windows={},
        omni_reference_sensor=f"{reporting_node_id}-s0",
        omni_reference_signal=signal,
        omni_position_m=position_m,
        omni_classification_reference_signal=signal,
        localization_audio_quality={},
        classification_audio_quality={},
        classification_audio_quality_source="test",
        environment={},
        extra_classification_features=features,
    )


def _pipeline(tracker: TrackManager, metrics: FusionMetrics, **setting_overrides) -> PositionObservationPipeline:
    return PositionObservationPipeline(
        config=position_observation_config_from_settings(Settings(**setting_overrides)),
        tracker_provider=lambda: tracker,
        metrics_provider=lambda: metrics,
    )


def _classified_kwargs(position_m, *, label="drone", label_category="aircraft", timestamp_ns=T0_NS) -> dict:
    return {
        "timestamp_ns": timestamp_ns,
        "position_m": position_m,
        "measurement_covariance_m2": None,
        "label": label,
        "label_category": label_category,
        "iff_category": "unknown",
        "label_id": None,
        "confidence": 0.8,
        "sensor_count": 4,
        "capability_tier": "full_3d",
        "source_node_id": "node-a",
        "classifier_scores": {label: 0.8},
    }


async def _visible_track(tracker: TrackManager, position_m, *, label="drone", label_category="aircraft"):
    return await tracker.update_with_classification(
        **_classified_kwargs(position_m, label=label, label_category=label_category, timestamp_ns=T0_NS - 1_000_000_000)
    )


async def _observe(pipeline: PositionObservationPipeline, product: LocalizedCandidate, origin: str) -> None:
    await pipeline.observe_localized_solve(
        localized_product=product, reporting_modality="localized", solve_origin=origin
    )


async def test_one_solve_updates_the_tracker_once_when_event_and_covering_render_both_deliver_it() -> None:
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(tracker, metrics)
    existing = await _visible_track(tracker, (10.0, 0.0, 2.0))
    event_render = _localized_product(candidate_id="rust-event", position_m=(10.5, 0.0, 2.0), solve_id="solve-1")
    covering_render = _localized_product(candidate_id="rust-cover", position_m=(10.5, 0.0, 2.0), solve_id="solve-1")

    await _observe(pipeline, event_render, SOLVE_ORIGIN_RUST_CLASSIFIER_RENDER)
    await _observe(pipeline, covering_render, SOLVE_ORIGIN_RUST_CLASSIFIER_RENDER)
    await pipeline.drain()
    # Both deliveries are classified too; neither moves the track again.
    for product in (event_render, covering_render):
        classified_track = await pipeline.track_for_classified_solve(
            solve_id=position_observation_solve_id(product),
            classified_update_kwargs=_classified_kwargs((10.5, 0.0, 2.0)),
        )
        assert classified_track.id == existing.id

    [track] = await tracker.snapshot()
    assert track.update_count == 2  # the seeding update + this one solve
    assert metrics.position_observations_applied == 1
    assert metrics.position_observations_duplicate_solve == 1
    assert metrics.position_observation_labels_attached == 2


async def test_two_same_label_callers_in_one_cluster_at_the_same_time_both_update() -> None:
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(tracker, metrics)
    first_caller = await _visible_track(tracker, (0.0, 0.0, 2.0), label="robin", label_category="wildlife")
    second_caller = await _visible_track(tracker, (40.0, 0.0, 2.0), label="robin", label_category="wildlife")

    for solve_id, reporter, position in (
        ("solve-a", "node-a", (0.5, 0.0, 2.0)),
        ("solve-b", "node-b", (40.5, 0.0, 2.0)),
    ):
        await _observe(
            pipeline,
            _localized_product(
                candidate_id=f"rust-{solve_id}",
                position_m=position,
                solve_id=solve_id,
                cluster_id="cluster-1",
                reporting_node_id=reporter,
            ),
            SOLVE_ORIGIN_RUST_CLASSIFIER_RENDER,
        )
    await pipeline.drain()

    tracks = {track.id: track for track in await tracker.snapshot()}
    assert tracks[first_caller.id].update_count == 2
    assert tracks[second_caller.id].update_count == 2
    assert "node-b" in tracks[second_caller.id].contributor_node_ids
    assert metrics.position_observations_applied == 2


async def test_fix_gated_by_two_tracks_waits_for_its_label_aware_update() -> None:
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(tracker, metrics)
    robin = await _visible_track(tracker, (0.0, 0.0, 2.0), label="robin", label_category="wildlife")
    drone = await _visible_track(tracker, (3.0, 0.0, 2.0))
    await _observe(pipeline, _localized_product(candidate_id="evt-between", position_m=(1.6, 0.0, 2.0)), SOLVE_ORIGIN_PYTHON_INGEST)
    await pipeline.drain()
    assert metrics.position_observations_deferred_ambiguous == 1
    assert {track.update_count for track in await tracker.snapshot()} == {1}

    track = await pipeline.track_for_classified_solve(
        solve_id="evt-between",
        classified_update_kwargs=_classified_kwargs((1.6, 0.0, 2.0)),
    )
    assert track.id == drone.id
    counts = {track.id: track.update_count for track in await tracker.snapshot()}
    assert counts == {robin.id: 1, drone.id: 2}


async def test_fifo_overflow_drops_the_oldest_and_counts_it() -> None:
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(tracker, metrics, position_observation_queue_max_items=2)
    products = [
        _localized_product(candidate_id=f"evt-{index}", position_m=(30.0 * index, 0.0, 2.0)) for index in range(3)
    ]
    # No await yields between submissions, so the drain cannot run in between.
    for product in products:
        await _observe(pipeline, product, SOLVE_ORIGIN_PYTHON_INGEST)
    assert metrics.position_observations_dropped_queue_full == 1
    assert pipeline.queued_count() == 2
    await pipeline.drain()
    assert metrics.position_observations_applied == 2
    assert pipeline._ledger["evt-0"].state is SolveLedgerState.DROPPED_QUEUE_FULL

    # The dropped solve's classification still puts it on a track, once.
    track = await pipeline.track_for_classified_solve(
        solve_id="evt-0", classified_update_kwargs=_classified_kwargs((0.0, 0.0, 2.0))
    )
    assert track.update_count == 1
    assert metrics.position_observation_classified_updates == 1


async def test_unclassified_fix_starts_hidden_and_becomes_visible_on_classification() -> None:
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(tracker, metrics)
    product = _localized_product(candidate_id="evt-hidden", position_m=(5.0, 5.0, 1.0))

    await _observe(pipeline, product, SOLVE_ORIGIN_PYTHON_INGEST)
    await pipeline.drain()
    assert await tracker.snapshot() == []
    [hidden] = await tracker.snapshot(include_hidden=True)
    assert tracker.is_track_hidden(hidden.id)
    assert hidden.label == "unknown"

    classified = await pipeline.track_for_classified_solve(
        solve_id="evt-hidden", classified_update_kwargs=_classified_kwargs((5.0, 5.0, 1.0))
    )
    [visible] = await tracker.snapshot()
    assert visible.id == classified.id
    assert visible.label == "drone"
    assert visible.id == hidden.id and visible.update_count == 1  # revealed in place, not updated twice
    assert not tracker.is_track_hidden(visible.id)
    assert await tracker.snapshot(include_hidden=True) == [visible]


async def test_classification_reveals_the_provisional_track_in_place_without_moving_it() -> None:
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(tracker, metrics)
    await _observe(pipeline, _localized_product(candidate_id="evt-first", position_m=(5.0, 5.0, 1.0)), SOLVE_ORIGIN_PYTHON_INGEST)
    await _observe(
        pipeline,
        _localized_product(candidate_id="evt-second", position_m=(5.5, 5.0, 1.0), event_time_ns=T0_NS + 500_000_000),
        SOLVE_ORIGIN_PYTHON_INGEST,
    )
    await pipeline.drain()
    [provisional] = await tracker.snapshot(include_hidden=True)
    position_before, last_seen_before = provisional.position_m, provisional.last_seen_ns

    classified = await pipeline.track_for_classified_solve(
        solve_id="evt-first", classified_update_kwargs=_classified_kwargs((5.0, 5.0, 1.0))
    )
    assert classified.id == provisional.id
    assert (classified.update_count, classified.position_m, classified.last_seen_ns) == (
        2,
        position_before,
        last_seen_before,
    )
    assert await tracker.snapshot() == [classified]


async def test_non_event_triggered_solve_cannot_start_a_track() -> None:
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(tracker, metrics)
    scheduled = _localized_product(
        candidate_id="rust-scheduled",
        position_m=(5.0, 5.0, 1.0),
        solve_id="solve-scheduled",
        attempt_kind=SCHEDULED_RENDER_ATTEMPT_KIND,
    )
    legacy_render_without_provenance = _localized_product(candidate_id="rust-legacy", position_m=(60.0, 0.0, 1.0))

    await _observe(pipeline, scheduled, SOLVE_ORIGIN_RUST_CLASSIFIER_RENDER)
    await _observe(pipeline, legacy_render_without_provenance, SOLVE_ORIGIN_RUST_CLASSIFIER_RENDER)
    await pipeline.drain()

    assert await tracker.snapshot(include_hidden=True) == []
    assert metrics.position_observations_track_not_started == 2

    # It still updates a track that already exists.
    existing = await _visible_track(tracker, (100.0, 0.0, 1.0))
    await _observe(
        pipeline,
        _localized_product(
            candidate_id="rust-scheduled-2",
            position_m=(100.5, 0.0, 1.0),
            solve_id="solve-scheduled-2",
            attempt_kind=SCHEDULED_RENDER_ATTEMPT_KIND,
        ),
        SOLVE_ORIGIN_RUST_CLASSIFIER_RENDER,
    )
    await pipeline.drain()
    [track] = await tracker.snapshot()
    assert track.id == existing.id and track.update_count == 2


async def test_require_classification_policy_never_starts_an_unclassified_track() -> None:
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(
        tracker,
        metrics,
        position_observation_unclassified_track_policy=POSITION_OBSERVATION_POLICY_REQUIRE_CLASSIFICATION,
    )
    await _observe(pipeline, _localized_product(candidate_id="evt-1", position_m=(1.0, 1.0, 1.0)), SOLVE_ORIGIN_PYTHON_INGEST)
    await pipeline.drain()
    assert await tracker.snapshot(include_hidden=True) == []
    track = await pipeline.track_for_classified_solve(
        solve_id="evt-1", classified_update_kwargs=_classified_kwargs((1.0, 1.0, 1.0))
    )
    assert track.update_count == 1


async def test_classification_overtaking_its_queued_observation_applies_it_once() -> None:
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(tracker, metrics)
    existing = await _visible_track(tracker, (10.0, 0.0, 2.0))
    await _observe(pipeline, _localized_product(candidate_id="evt-fast", position_m=(10.5, 0.0, 2.0)), SOLVE_ORIGIN_PYTHON_INGEST)

    track = await pipeline.track_for_classified_solve(
        solve_id="evt-fast", classified_update_kwargs=_classified_kwargs((10.5, 0.0, 2.0))
    )
    await pipeline.drain()
    assert track.id == existing.id
    assert track.update_count == 2
    assert metrics.position_observations_applied == 1


async def test_classification_contradicting_the_geometric_match_updates_its_own_track() -> None:
    tracker, metrics = TrackManager(Settings()), FusionMetrics()
    pipeline = _pipeline(tracker, metrics)
    bird = await _visible_track(tracker, (10.0, 0.0, 2.0), label="robin", label_category="wildlife")
    await _observe(pipeline, _localized_product(candidate_id="evt-drone", position_m=(10.5, 0.0, 2.0)), SOLVE_ORIGIN_PYTHON_INGEST)
    await pipeline.drain()

    drone = await pipeline.track_for_classified_solve(
        solve_id="evt-drone", classified_update_kwargs=_classified_kwargs((10.5, 0.0, 2.0))
    )
    assert drone.id != bird.id
    assert drone.label == "drone"
    assert {track.id: track.label for track in await tracker.snapshot()}[bird.id] == "robin"


async def test_absorbed_entity_bout_of_an_observed_solve_attaches_without_moving_the_track() -> None:
    tracker, metrics = TrackManager(Settings(tracking_partial_constraint_binding_enabled=True)), FusionMetrics()
    pipeline = _pipeline(tracker, metrics)
    existing = await _visible_track(tracker, (10.0, 0.0, 2.0), label="robin", label_category="wildlife")
    await _observe(pipeline, _localized_product(candidate_id="evt-bout", position_m=(10.5, 0.0, 2.0)), SOLVE_ORIGIN_PYTHON_INGEST)
    await pipeline.drain()
    [after_observation] = await tracker.snapshot()
    position_after_observation = after_observation.position_m

    resolution = await resolve_detection_track(
        tracker=tracker,
        registry=NodeRegistry(),
        spatial_display_mode="node_only",
        suppressed_by_zone=False,
        capability_tier="full_3d",
        event_time_ns=T0_NS,
        localization_position_m=(10.5, 0.0, 2.0),
        localization_position_covariance_m2=None,
        classification_label="robin",
        classification_confidence=0.9,
        classification_scores={"robin": 0.9},
        classification_features={},
        label_category="wildlife",
        iff_category="unknown",
        label_id=None,
        selected_sensor_ids=["node-a-s0"],
        source_node_id="node-a",
        reference_sensor="node-a-s0",
        tdoa_s={},
        environment={},
        solve_id="evt-bout",
        position_observations=pipeline,
    )
    assert resolution.track is not None and resolution.track.id == existing.id
    assert resolution.track.update_count == 2
    assert resolution.track.position_m == position_after_observation


# -- FusionNode wiring ----------------------------------------------------------


async def test_classification_suppressed_render_still_updated_the_track(tmp_path: Path) -> None:
    fusion, storage = await _started_fusion(tmp_path)
    existing = await _visible_track(fusion.tracker, (3.0, 4.0, 0.5))
    fusion._classification_chunking_policy = object()
    fusion._should_suppress_chunked_classification = lambda product: True

    await fusion.ingest_localized_render(_render("manifest-p5-suppressed", localization_position_m=(3.2, 4.0, 0.5)))
    await fusion._position_observations().drain()

    assert await storage.list_detections(limit=10) == []
    [track] = await fusion.tracker.snapshot()
    assert track.id == existing.id
    assert track.update_count == 2
    assert track.last_seen_ns == _render("unused").event_time_ns
    await fusion.stop()
    await storage.close()


async def test_backlog_dropped_detection_still_updated_the_track(tmp_path: Path) -> None:
    fusion, storage = await _started_fusion(tmp_path)
    existing = await _visible_track(fusion.tracker, (3.0, 4.0, 0.5))
    product = _localized_product(candidate_id="evt-backlog", position_m=(3.2, 4.0, 0.5), reporting_node_id="sirith-seam")
    original_enqueue_stage = fusion._enqueue_stage

    async def _classification_backlog_drops(queue, item):
        if queue is fusion._classification_queue:
            return False
        return await original_enqueue_stage(queue, item)

    async def _localize_stub(candidate):
        return product

    fusion._enqueue_stage = _classification_backlog_drops
    fusion._localize_candidate = _localize_stub
    await fusion._localization_queue.put(product.candidate)
    await asyncio.wait_for(fusion._localization_queue.join(), timeout=5.0)
    await fusion._position_observations().drain()

    assert fusion._metrics.localization_stage_out == 0
    [track] = await fusion.tracker.snapshot()
    assert track.id == existing.id
    assert track.update_count == 2
    assert fusion._metrics.position_observations_applied == 1
    await fusion.stop()
    await storage.close()


async def test_event_only_localization_result_reaches_the_tracker_through_the_stream_consumer(tmp_path: Path) -> None:
    scene = drone_fly_by(default_scene_timing(), reporter_ids=SCENE_REPORTER_IDS)
    event_deliveries = [delivery for delivery in scene.deliveries if delivery.kind == DELIVERY_EVENT_LOCALIZATION_RESULT]
    fusion, storage = await started_scene_fusion(tmp_path)
    try:
        record = await deliver_through_stream_consumer(fusion, scene, event_deliveries[:3])
    finally:
        await fusion.stop()
        await storage.close()

    assert [observation.solve_id for observation in record.observations] == [
        delivery.solve_id for delivery in event_deliveries[:3]
    ]
    assert all(observation.update_count_increase == 1 for observation in record.observations)
    assert record.visible_track_ids == set()  # unclassified until a render's label arrives
