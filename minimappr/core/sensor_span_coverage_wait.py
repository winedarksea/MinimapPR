"""Non-blocking wait for per-sensor ring buffers to cover a time span.

Shared by the legacy single-window localization path (FusionNode) and the
onset-SRP stage. Contract: keep polling while any *uncovered* sensor that can
still cover the span is advancing; release a sensor once its buffer has not
advanced for `stall_budget_seconds`; never wait past `total_cap_seconds`.

Why a per-sensor stall budget and not "first N sensors" or a fixed deadline:
returning once `min_sensors_for_2d` sensors were covered let one tetra (4 mics)
win the race, silently excluding later-arriving nodes (single-node solves, no
cross-node TDOA). A fixed wall-clock deadline instead drops windows whenever
ingest runs slower than real time. Tracking progress per sensor means a dead
node costs at most one stall budget (and nothing afterwards when a persistent
`SensorStreamActivityTracker` is shared across waits), while a node that is
merely late but still streaming is waited for, up to the cap.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

NANOSECONDS_PER_SECOND = 1_000_000_000
STOP_REASON_ALL_COVERED = "all_covered"
STOP_REASON_STALLED = "stalled"
STOP_REASON_TOTAL_CAP = "total_cap"
STOP_REASON_REQUIRED_COVERAGE_DEADLINE = "required_coverage_deadline"
STOP_REASON_SPAN_OLDER_THAN_BUFFER = "span_older_than_buffer"


class BufferSnapshotSource(Protocol):
    async def snapshot_state(self, sensor_ids: list[str] | None = None) -> list[dict[str, Any]]: ...


@dataclass(slots=True)
class SensorStreamActivityTracker:
    """Remembers, across waits, when each sensor's buffer end last advanced.

    Only updated from coverage-wait polls (no hot-path cost). A sensor seen for
    the first time counts as having just advanced, so an unknown sensor gets one
    stall budget of benefit of the doubt; a dead node is then skipped
    immediately by every later wait because its end time never moves again.
    """

    last_end_time_ns_by_sensor: dict[str, int] = field(default_factory=dict)
    last_advance_monotonic_by_sensor: dict[str, float] = field(default_factory=dict)

    def observe(self, snapshot: list[dict[str, Any]], now_monotonic: float) -> None:
        for state in snapshot:
            if not state.get("present"):
                continue
            sensor_id = state["sensor_id"]
            end_time_ns = int(state.get("end_time_ns") or 0)
            previous_end_time_ns = self.last_end_time_ns_by_sensor.get(sensor_id)
            if previous_end_time_ns is None or end_time_ns > previous_end_time_ns:
                self.last_advance_monotonic_by_sensor[sensor_id] = now_monotonic
            self.last_end_time_ns_by_sensor[sensor_id] = end_time_ns

    def seconds_since_advance(self, sensor_id: str, now_monotonic: float) -> float:
        last_advance = self.last_advance_monotonic_by_sensor.get(sensor_id)
        return float("inf") if last_advance is None else now_monotonic - last_advance

    def forget_missing(self, present_sensor_ids: set[str]) -> None:
        """Drop sensors no longer registered so removed nodes do not accumulate."""
        for sensor_id in list(self.last_end_time_ns_by_sensor):
            if sensor_id not in present_sensor_ids:
                self.last_end_time_ns_by_sensor.pop(sensor_id, None)
                self.last_advance_monotonic_by_sensor.pop(sensor_id, None)


@dataclass(frozen=True, slots=True)
class SpanCoverageWaitResult:
    covered_sensor_ids: list[str]
    # Uncovered sensors that were still eligible when the cap/deadline fired,
    # or that were released as stalled (diagnostics only).
    released_sensor_ids: list[str]
    snapshot: list[dict[str, Any]]
    waited_seconds: float
    stop_reason: str


def _sensor_covers_span(state: dict[str, Any], span_start_ns: int, span_end_ns: int) -> bool:
    start_ns = state.get("start_time_ns")
    end_ns = state.get("end_time_ns")
    return bool(
        state.get("present")
        and start_ns is not None
        and end_ns is not None
        and start_ns <= span_start_ns
        and end_ns >= span_end_ns
    )


async def await_sensor_span_coverage(
    *,
    buffer: BufferSnapshotSource,
    sensor_ids: list[str],
    span_start_ns: int,
    span_end_ns: int,
    stall_budget_seconds: float,
    total_cap_seconds: float,
    poll_interval_seconds: float,
    activity_tracker: SensorStreamActivityTracker | None = None,
    required_covered_count: int = 0,
    required_coverage_deadline_seconds: float | None = None,
) -> SpanCoverageWaitResult:
    """Wait until every eligible sensor covers [span_start_ns, span_end_ns].

    Eligible = present, buffer start at or before the span start (a ring buffer
    start only moves forward, so a later start can never cover), and advanced
    within `stall_budget_seconds` and not trailing the span end by more audio
    than the remaining cap. Returns as soon as no uncovered sensor is
    eligible, or at `total_cap_seconds`.

    `required_covered_count` / `required_coverage_deadline_seconds` keep the
    legacy drop contract: until that many sensors are covered the wait does not
    give up on absent/stalled sensors, but it does stop at the deadline (the
    caller then drops the candidate as `buffer_lag_timeout`).
    """
    tracker = activity_tracker if activity_tracker is not None else SensorStreamActivityTracker()
    stall_budget_seconds = max(0.0, stall_budget_seconds)
    total_cap_seconds = max(0.0, total_cap_seconds)
    started = time.monotonic()
    while True:
        snapshot = await buffer.snapshot_state(sensor_ids=sensor_ids)
        now = time.monotonic()
        waited = now - started
        tracker.observe(snapshot, now)
        covered: list[str] = []
        eligible_uncovered: list[str] = []
        stalled_uncovered: list[str] = []
        oldest_present_start_ns: int | None = None
        for state in snapshot:
            start_ns = state.get("start_time_ns")
            if state.get("present") and start_ns is not None:
                if oldest_present_start_ns is None or start_ns < oldest_present_start_ns:
                    oldest_present_start_ns = start_ns
            if _sensor_covers_span(state, span_start_ns, span_end_ns):
                covered.append(state["sensor_id"])
            elif state.get("present") and start_ns is not None and start_ns <= span_start_ns:
                # Live audio arrives at most ~real time, so a buffer whose end
                # trails the span end by more audio than the remaining cap can
                # never catch up in time; skip it instead of burning the cap.
                audio_deficit_seconds = (span_end_ns - int(state.get("end_time_ns") or 0)) / NANOSECONDS_PER_SECOND
                can_catch_up = audio_deficit_seconds <= total_cap_seconds - waited
                if can_catch_up and tracker.seconds_since_advance(state["sensor_id"], now) < stall_budget_seconds:
                    eligible_uncovered.append(state["sensor_id"])
                else:
                    stalled_uncovered.append(state["sensor_id"])

        def _result(stop_reason: str, released: list[str]) -> SpanCoverageWaitResult:
            return SpanCoverageWaitResult(
                covered_sensor_ids=covered,
                released_sensor_ids=released,
                snapshot=snapshot,
                waited_seconds=waited,
                stop_reason=stop_reason,
            )

        if len(covered) == len(sensor_ids):
            return _result(STOP_REASON_ALL_COVERED, [])
        # The span's trailing edge already fell out of the oldest buffer: no
        # amount of waiting recovers it.
        if oldest_present_start_ns is not None and span_end_ns < oldest_present_start_ns:
            return _result(STOP_REASON_SPAN_OLDER_THAN_BUFFER, [])
        if waited >= total_cap_seconds:
            return _result(STOP_REASON_TOTAL_CAP, eligible_uncovered + stalled_uncovered)
        if len(covered) < required_covered_count:
            if required_coverage_deadline_seconds is not None and waited >= required_coverage_deadline_seconds:
                return _result(STOP_REASON_REQUIRED_COVERAGE_DEADLINE, eligible_uncovered + stalled_uncovered)
        elif not eligible_uncovered:
            return _result(STOP_REASON_STALLED, stalled_uncovered)
        await asyncio.sleep(poll_interval_seconds)
