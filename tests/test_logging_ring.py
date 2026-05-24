"""Tests for the in-process log ring buffer.

These regression-test the silent-extras bug uncovered during the live
diagnosis of the localization race (plan: valiant-launching-whale):
`LogCaptureHandler` was dropping caller-supplied `extra=` fields, which made
the per-record context attached to "Silent pipeline drop" warnings invisible
to /api/v1/system/logs consumers.
"""

from __future__ import annotations

import logging

from minimappr.core.logging_ring import LogCaptureHandler


def _make_handler_and_logger(name: str) -> tuple[LogCaptureHandler, logging.Logger]:
    handler = LogCaptureHandler(capacity=10)
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # Detach any previous handlers from prior tests to keep snapshots clean.
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
    logger.addHandler(handler)
    logger.propagate = False
    return handler, logger


def test_handler_preserves_caller_extras() -> None:
    handler, logger = _make_handler_and_logger("test_logging_ring.preserve")
    logger.warning(
        "Silent pipeline drop",
        extra={"drop_reason": "buffer_lag_timeout", "candidate_id": "evt-1", "n": 5},
    )
    records = handler.snapshot()
    assert len(records) == 1
    entry = records[0]
    assert entry["message"] == "Silent pipeline drop"
    assert "extra" in entry
    assert entry["extra"]["drop_reason"] == "buffer_lag_timeout"
    assert entry["extra"]["candidate_id"] == "evt-1"
    assert entry["extra"]["n"] == 5


def test_handler_omits_extra_when_caller_passed_none() -> None:
    handler, logger = _make_handler_and_logger("test_logging_ring.empty")
    logger.info("plain message")
    records = handler.snapshot()
    assert len(records) == 1
    assert "extra" not in records[0]


def test_handler_coerces_non_json_safe_extras_to_repr() -> None:
    handler, logger = _make_handler_and_logger("test_logging_ring.coerce")

    class _Opaque:
        def __repr__(self) -> str:
            return "<Opaque>"

    logger.warning("with opaque", extra={"thing": _Opaque(), "items": [1, _Opaque()]})
    records = handler.snapshot()
    assert len(records) == 1
    extra = records[0]["extra"]
    assert extra["thing"] == "<Opaque>"
    # Nested coercion through lists.
    assert extra["items"] == [1, "<Opaque>"]


def test_handler_preserves_nested_dict_extras() -> None:
    """`buffer_snapshot` is the most important extra in production — it's a
    list of dicts. Make sure nested structures round-trip."""
    handler, logger = _make_handler_and_logger("test_logging_ring.nested")
    snapshot = [
        {"sensor_id": "a", "end_time_ns": 1_000_000_000, "present": True},
        {"sensor_id": "b", "present": False},
    ]
    logger.warning("Silent pipeline drop", extra={"buffer_snapshot": snapshot})
    records = handler.snapshot()
    assert records[0]["extra"]["buffer_snapshot"] == snapshot
