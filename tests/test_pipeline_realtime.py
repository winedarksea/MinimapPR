"""Unit tests for PipelineRealtimeTracker load-shedding bookkeeping."""

from __future__ import annotations

from minimappr.core.pipeline_realtime import PipelineRealtimeTracker


def _tracker() -> PipelineRealtimeTracker:
    return PipelineRealtimeTracker(stage_names=("localization", "classification", "rules"))


def test_mark_dropped_releases_the_reported_lag() -> None:
    """A shed item left in the tracker reports a lag that only grows.

    That is the exact symptom load shedding exists to prevent, so the drop must
    retire the item rather than merely remove it from the asyncio queue.
    """
    tracker = _tracker()
    tracker.mark_enqueued(stage_name="classification", item_id="stale", event_time_ns=1_000)
    tracker.mark_enqueued(
        stage_name="classification", item_id="fresh", event_time_ns=9_000_000_000
    )

    tracker.mark_dropped(stage_name="classification", item_id="stale")

    stage = tracker.snapshot(now_ns=10_000_000_000)["stages"]["classification"]
    assert stage["queued_items"] == 1
    assert stage["oldest_queued_event_time_ns"] == 9_000_000_000
    assert stage["seconds_behind_realtime"] == 1.0


def test_mark_dropped_is_idempotent_for_unknown_items() -> None:
    tracker = _tracker()
    tracker.mark_dropped(stage_name="localization", item_id="never-enqueued")
    assert tracker.snapshot()["stages"]["localization"]["queued_items"] == 0
