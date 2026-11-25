"""Milestone 3 review fixes on the render handoff: findings 4, 5 and 6.

The Python re-solve of an event-only solve runs off the event loop, both paths
time a solve at its own sample rate, and a covering render neither recounts its
event's rejections nor registers its bearing cone twice.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from minimappr.core import localized_render_localization
from minimappr.core.localized_render_localization import solve_timing_sample_rate_hz
from minimappr.core.multi_node_bearing_fusion import BearingFusionStore, BearingObservation
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

T0_NS = 1_739_950_000_000_000_000
SECOND_NS = 1_000_000_000


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
