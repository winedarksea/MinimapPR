"""Real-time backlog tracking for staged pipeline queues."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class _StageState:
    queued_event_times_ns: OrderedDict[str, int]
    inflight_event_times_ns: dict[str, int]


class PipelineRealtimeTracker:
    """Tracks queue age and end-to-end latency across pipeline stages.

    The key operator question is not just queue depth, but how old the oldest
    event still waiting in the pipeline is relative to wall-clock time.
    """

    def __init__(self, stage_names: tuple[str, ...]) -> None:
        self._stages = {
            stage_name: _StageState(queued_event_times_ns=OrderedDict(), inflight_event_times_ns={})
            for stage_name in stage_names
        }
        self._last_completed_event_time_ns: int | None = None
        self._last_completed_lag_ns: int | None = None
        self._max_completed_lag_ns: int = 0
        self._completed_count: int = 0

    def mark_enqueued(self, *, stage_name: str, item_id: str, event_time_ns: int) -> None:
        self._stages[stage_name].queued_event_times_ns[item_id] = int(event_time_ns)

    def mark_dropped(self, *, stage_name: str, item_id: str) -> None:
        """Retire an item that was enqueued but will never be processed.

        Without this, a shed item stays in ``queued_event_times_ns`` forever and
        pins ``oldest_queued_event_time_ns``, so the pipeline reports a lag that
        only grows — the exact symptom load shedding exists to prevent.
        """
        self._stages[stage_name].queued_event_times_ns.pop(item_id, None)

    def mark_started(self, *, stage_name: str, item_id: str) -> None:
        stage = self._stages[stage_name]
        event_time_ns = stage.queued_event_times_ns.pop(item_id, None)
        if event_time_ns is not None:
            stage.inflight_event_times_ns[item_id] = event_time_ns

    def mark_finished(self, *, stage_name: str, item_id: str) -> None:
        self._stages[stage_name].inflight_event_times_ns.pop(item_id, None)

    def record_completed(self, *, event_time_ns: int, completed_time_ns: int | None = None) -> None:
        finished_ns = completed_time_ns if completed_time_ns is not None else time.time_ns()
        lag_ns = max(0, int(finished_ns) - int(event_time_ns))
        self._last_completed_event_time_ns = int(event_time_ns)
        self._last_completed_lag_ns = lag_ns
        self._max_completed_lag_ns = max(self._max_completed_lag_ns, lag_ns)
        self._completed_count += 1

    def snapshot(self, now_ns: int | None = None) -> dict[str, Any]:
        now = now_ns if now_ns is not None else time.time_ns()
        stages: dict[str, Any] = {}
        oldest_pipeline_event_time_ns: int | None = None
        for stage_name, stage in self._stages.items():
            oldest_queued_event_time_ns = next(iter(stage.queued_event_times_ns.values()), None)
            oldest_inflight_event_time_ns = (
                min(stage.inflight_event_times_ns.values())
                if stage.inflight_event_times_ns
                else None
            )
            stage_event_times = [
                event_time_ns
                for event_time_ns in (oldest_queued_event_time_ns, oldest_inflight_event_time_ns)
                if event_time_ns is not None
            ]
            oldest_stage_event_time_ns = min(stage_event_times) if stage_event_times else None
            if oldest_stage_event_time_ns is not None:
                if oldest_pipeline_event_time_ns is None:
                    oldest_pipeline_event_time_ns = oldest_stage_event_time_ns
                else:
                    oldest_pipeline_event_time_ns = min(oldest_pipeline_event_time_ns, oldest_stage_event_time_ns)

            stages[stage_name] = {
                "queued_items": len(stage.queued_event_times_ns),
                "inflight_items": len(stage.inflight_event_times_ns),
                "oldest_queued_event_time_ns": oldest_queued_event_time_ns,
                "oldest_inflight_event_time_ns": oldest_inflight_event_time_ns,
                "seconds_behind_realtime": _lag_seconds(now, oldest_stage_event_time_ns),
            }

        return {
            "now_ns": now,
            "stages": stages,
            "pipeline_seconds_behind_realtime": _lag_seconds(now, oldest_pipeline_event_time_ns),
            "last_completed_event_time_ns": self._last_completed_event_time_ns,
            "last_completed_lag_seconds": _seconds_from_ns(self._last_completed_lag_ns),
            "max_completed_lag_seconds": _seconds_from_ns(self._max_completed_lag_ns),
            "completed_events": self._completed_count,
        }


def _lag_seconds(now_ns: int, event_time_ns: int | None) -> float | None:
    if event_time_ns is None:
        return None
    return max(0.0, (int(now_ns) - int(event_time_ns)) / 1_000_000_000.0)


def _seconds_from_ns(duration_ns: int | None) -> float | None:
    if duration_ns is None:
        return None
    return max(0.0, float(duration_ns) / 1_000_000_000.0)
