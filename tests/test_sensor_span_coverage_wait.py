"""Coverage race fix: the shared span-coverage wait (core/sensor_span_coverage_wait.py).

A fake buffer advances each sensor's end time by a fixed step per poll, which
makes "node-b's frames arrive later than node-a's" deterministic without any
real ingest or wall-clock dependence beyond the stall budget / cap.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from minimappr.core.sensor_span_coverage_wait import (
    STOP_REASON_ALL_COVERED,
    STOP_REASON_STALLED,
    STOP_REASON_TOTAL_CAP,
    SensorStreamActivityTracker,
    await_sensor_span_coverage,
)

NODE_A = [f"node-a:ch{i}" for i in range(4)]
NODE_B = [f"node-b:ch{i}" for i in range(4)]
SPAN_START_NS = 0
SPAN_END_NS = 80_000_000
POLL_S = 0.002


class _SteppingBuffer:
    """Per-sensor end time advances by `step_ns_by_sensor` on every snapshot."""

    def __init__(self, end_ns_by_sensor: dict[str, int], step_ns_by_sensor: dict[str, int]) -> None:
        self.end_ns_by_sensor = dict(end_ns_by_sensor)
        self.step_ns_by_sensor = step_ns_by_sensor
        self.snapshot_count = 0

    async def snapshot_state(self, sensor_ids=None):
        self.snapshot_count += 1
        for sensor_id, step_ns in self.step_ns_by_sensor.items():
            self.end_ns_by_sensor[sensor_id] += step_ns
        return [
            {"sensor_id": s, "present": True, "start_time_ns": SPAN_START_NS - 1, "end_time_ns": self.end_ns_by_sensor[s]}
            for s in sensor_ids
        ]


def _late_node_b_buffer(node_b_step_ns: int) -> _SteppingBuffer:
    # node-a already covers the span; node-b is one 64 ms frame behind and catching up.
    end_ns = {s: SPAN_END_NS for s in NODE_A} | {s: SPAN_END_NS - 64_000_000 for s in NODE_B}
    return _SteppingBuffer(end_ns, {s: node_b_step_ns for s in NODE_B})


async def _wait(buffer, *, stall_s: float, cap_s: float, tracker=None, required: int = 3):
    return await await_sensor_span_coverage(
        buffer=buffer,
        sensor_ids=NODE_A + NODE_B,
        span_start_ns=SPAN_START_NS,
        span_end_ns=SPAN_END_NS,
        stall_budget_seconds=stall_s,
        total_cap_seconds=cap_s,
        poll_interval_seconds=POLL_S,
        activity_tracker=tracker,
        required_covered_count=required,
        required_coverage_deadline_seconds=cap_s,
    )


@pytest.mark.asyncio
async def test_pre_fix_semantics_reproduce_the_race_late_node_excluded() -> None:
    # Stall budget 0 = the old "return once min_sensors_for_2d are covered":
    # node-a's tetra alone satisfies it and node-b is silently dropped.
    result = await _wait(_late_node_b_buffer(8_000_000), stall_s=0.0, cap_s=1.0)
    assert result.covered_sensor_ids == NODE_A


@pytest.mark.asyncio
async def test_late_but_streaming_node_is_waited_for() -> None:
    result = await _wait(_late_node_b_buffer(8_000_000), stall_s=0.5, cap_s=2.0)
    assert sorted(result.covered_sensor_ids) == sorted(NODE_A + NODE_B)
    assert result.stop_reason == STOP_REASON_ALL_COVERED


@pytest.mark.asyncio
async def test_all_nodes_present_adds_no_wait() -> None:
    buffer = _SteppingBuffer({s: SPAN_END_NS for s in NODE_A + NODE_B}, {})
    started = time.monotonic()
    result = await _wait(buffer, stall_s=0.5, cap_s=1.0)
    assert result.stop_reason == STOP_REASON_ALL_COVERED
    assert buffer.snapshot_count == 1
    assert time.monotonic() - started < 0.05


@pytest.mark.asyncio
async def test_stalled_node_released_after_stall_budget() -> None:
    result = await _wait(_late_node_b_buffer(0), stall_s=0.1, cap_s=5.0)
    assert result.covered_sensor_ids == NODE_A
    assert result.stop_reason == STOP_REASON_STALLED
    assert sorted(result.released_sensor_ids) == sorted(NODE_B)
    # Released on the stall budget, nowhere near the 5 s cap (slack for a loaded box).
    assert 0.1 <= result.waited_seconds < 1.0


@pytest.mark.asyncio
async def test_slowly_advancing_node_cannot_exceed_total_cap() -> None:
    # node-b is 1 us short of covering and within its (long) stall budget, so it
    # stays eligible; only the total cap can end the wait.
    end_ns = {s: SPAN_END_NS for s in NODE_A} | {s: SPAN_END_NS - 1_000 for s in NODE_B}
    buffer = _SteppingBuffer(end_ns, {})
    started = time.monotonic()
    result = await _wait(buffer, stall_s=10.0, cap_s=0.2)
    elapsed = time.monotonic() - started
    assert result.stop_reason == STOP_REASON_TOTAL_CAP
    assert result.covered_sensor_ids == NODE_A
    assert 0.2 <= elapsed < 1.0


@pytest.mark.asyncio
async def test_dead_node_costs_nothing_on_later_waits_with_shared_tracker() -> None:
    tracker = SensorStreamActivityTracker()
    buffer = _late_node_b_buffer(0)
    first = await _wait(buffer, stall_s=0.1, cap_s=5.0, tracker=tracker)
    assert first.stop_reason == STOP_REASON_STALLED
    second = await _wait(buffer, stall_s=0.1, cap_s=5.0, tracker=tracker)
    assert second.covered_sensor_ids == NODE_A
    assert second.waited_seconds < 0.05


@pytest.mark.asyncio
async def test_node_lagging_more_audio_than_the_cap_is_skipped_immediately() -> None:
    # node-b streams in real time but is 5 s behind: it cannot cover within a 1 s cap.
    end_ns = {s: SPAN_END_NS for s in NODE_A} | {s: SPAN_END_NS - 5_000_000_000 for s in NODE_B}
    buffer = _SteppingBuffer(end_ns, {s: 1_000_000 for s in NODE_B})
    result = await _wait(buffer, stall_s=0.5, cap_s=1.0)
    assert result.covered_sensor_ids == NODE_A
    assert result.waited_seconds < 0.05


@pytest.mark.asyncio
async def test_required_count_still_waits_for_absent_sensors_until_deadline() -> None:
    # Legacy drop contract: fewer than min_sensors_for_2d covered -> wait to the deadline.
    buffer = _SteppingBuffer({NODE_A[0]: SPAN_END_NS}, {})

    async def _snapshot(sensor_ids=None):
        return [
            {"sensor_id": NODE_A[0], "present": True, "start_time_ns": -1, "end_time_ns": SPAN_END_NS},
            *({"sensor_id": s, "present": False} for s in sensor_ids[1:]),
        ]

    buffer.snapshot_state = _snapshot  # type: ignore[method-assign]
    result = await await_sensor_span_coverage(
        buffer=buffer,
        sensor_ids=NODE_A,
        span_start_ns=SPAN_START_NS,
        span_end_ns=SPAN_END_NS,
        stall_budget_seconds=0.5,
        total_cap_seconds=1.0,
        poll_interval_seconds=POLL_S,
        required_covered_count=3,
        required_coverage_deadline_seconds=0.08,
    )
    assert result.covered_sensor_ids == [NODE_A[0]]
    assert 0.08 <= result.waited_seconds < 0.5


# ---------------------------------------------------------------- FusionNode
SAMPLE_RATE_HZ = 16_000
FRAME_SAMPLES = 1024
FRAME_NS = int(FRAME_SAMPLES / SAMPLE_RATE_HZ * 1_000_000_000)
BASE_NS = 1_739_900_000_000_000_000
# Window (0.04 s) ends inside frame 2, which a lagging node-b has not delivered yet.
EVENT_CENTER_NS = BASE_NS + 2 * FRAME_NS + FRAME_NS // 4
WINDOW_SECONDS = 0.04


async def _fusion_with_real_buffer(tmp_path, *, stall_s: float, cap_s: float):
    from minimappr.classifiers.heuristic import HeuristicClassifier
    from minimappr.core.audio_buffer import MultiSensorBuffer
    from minimappr.core.fusion_node import FusionNode
    from minimappr.core.geo import LocalCoordinateFrame
    from minimappr.core.localization import LocalizationEngine
    from minimappr.core.node_registry import NodeRegistry
    from minimappr.core.tracking import TrackManager
    from minimappr.core.zones import ZoneMatcher
    from minimappr.models import GeoPoint
    from minimappr.storage.db import Storage
    from tests.test_fusion_node import _make_minimal_fusion_node

    settings = _make_minimal_fusion_node(tmp_path)
    storage = Storage(settings.db_path)
    await storage.initialize()
    buffer = MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds)
    fusion = FusionNode(
        settings=settings,
        registry=NodeRegistry(),
        buffer=buffer,
        localizer=LocalizationEngine(max_tau_s=0.03),
        classifier=HeuristicClassifier(),
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=lambda payload: asyncio.sleep(0, result=None),
        coordinate_frame=LocalCoordinateFrame(origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"),
        zone_matcher=ZoneMatcher(storage=storage),
    )
    fusion.localization_config.localization_coverage_stall_budget_seconds = stall_s
    fusion.localization_config.localization_coverage_total_cap_seconds = cap_s
    return fusion, buffer, storage


async def _append_frames(buffer, sensor_ids, frame_indices) -> None:
    for sensor_id in sensor_ids:
        for k in frame_indices:
            await buffer.append(
                sensor_id=sensor_id,
                sample_rate_hz=SAMPLE_RATE_HZ,
                start_time_ns=BASE_NS + k * FRAME_NS,
                samples=np.ones(FRAME_SAMPLES, dtype=np.float32),
            )


async def _fusion_wait(fusion):
    return await fusion._await_window_coverage(
        sensor_ids=NODE_A + NODE_B,
        center_time_ns=EVENT_CENTER_NS,
        window_seconds=WINDOW_SECONDS,
        timeout_s=0.3,
    )


@pytest.mark.asyncio
async def test_fusion_window_coverage_waits_for_late_node_frame(tmp_path) -> None:
    fusion, buffer, storage = await _fusion_with_real_buffer(tmp_path, stall_s=0.5, cap_s=1.0)
    try:
        await _append_frames(buffer, NODE_A, range(3))
        await _append_frames(buffer, NODE_B, range(2))

        async def _late_node_b_frame() -> None:
            await asyncio.sleep(0.05)
            await _append_frames(buffer, NODE_B, [2])

        late = asyncio.create_task(_late_node_b_frame())
        ready, snapshot = await _fusion_wait(fusion)
        await late
        assert ready is True
        span_end_ns = EVENT_CENTER_NS + int(WINDOW_SECONDS / 2 * 1_000_000_000)
        covered = {s["sensor_id"] for s in snapshot if s["end_time_ns"] >= span_end_ns}
        assert covered == set(NODE_A + NODE_B)
        assert fusion._metrics.localization_coverage_wait_stop_reasons == {STOP_REASON_ALL_COVERED: 1}
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_fusion_window_coverage_stalled_node_cannot_exceed_cap(tmp_path) -> None:
    # node-b present but dead: localization proceeds with node-a after the stall budget.
    fusion, buffer, storage = await _fusion_with_real_buffer(tmp_path, stall_s=0.1, cap_s=0.25)
    try:
        await _append_frames(buffer, NODE_A, range(3))
        await _append_frames(buffer, NODE_B, range(2))
        started = time.monotonic()
        ready, _ = await _fusion_wait(fusion)
        elapsed = time.monotonic() - started
        assert ready is True
        assert 0.1 <= elapsed < 0.3 + 0.5  # bounded by max(timeout_s, cap) plus load slack
        assert fusion._metrics.localization_coverage_wait_stop_reasons == {STOP_REASON_STALLED: 1}
        # Next event: the shared tracker already knows node-b is dead -> no added latency.
        started = time.monotonic()
        await _fusion_wait(fusion)
        assert time.monotonic() - started < 0.05
    finally:
        await storage.close()
