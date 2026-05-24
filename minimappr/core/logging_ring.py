"""In-process ring buffer for recent log records, exposed via /api/v1/logs."""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Any


# Standard LogRecord attribute names — anything *not* in this set is treated as
# a caller-supplied `extra=` field and surfaced in the snapshot payload. Built
# once from a synthetic record so the list stays in sync with the stdlib.
_STANDARD_LOG_RECORD_ATTRS: frozenset[str] = frozenset(
    logging.LogRecord(name="", level=0, pathname="", lineno=0, msg="", args=None, exc_info=None).__dict__
) | {"message", "asctime"}


def _coerce_extra_value(value: Any) -> Any:
    """Coerce an `extra=` field value to a JSON-safe representation.

    The ring buffer is consumed via /api/v1/system/logs (JSON), so anything
    that isn't natively serializable falls back to `repr()` rather than
    crashing the handler — log handlers must never raise.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce_extra_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _coerce_extra_value(v) for k, v in value.items()}
    return repr(value)


class LogCaptureHandler(logging.Handler):
    """Keep the last N log records so the UI can show a live tail without tailing files."""

    def __init__(self, capacity: int = 2000) -> None:
        super().__init__()
        self._buffer: collections.deque[dict[str, Any]] = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._seq = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            msg = record.msg if isinstance(record.msg, str) else repr(record.msg)
        entry = {
            "ts_ns": int(record.created * 1_000_000_000),
            "level": record.levelname,
            "level_no": record.levelno,
            "logger": record.name,
            "message": msg,
        }
        extras = {
            key: _coerce_extra_value(value)
            for key, value in record.__dict__.items()
            if key not in _STANDARD_LOG_RECORD_ATTRS and not key.startswith("_")
        }
        if extras:
            entry["extra"] = extras
        if record.exc_info:
            entry["exc"] = self.format(record)
        with self._lock:
            self._seq += 1
            entry["seq"] = self._seq
            self._buffer.append(entry)

    def snapshot(
        self,
        *,
        limit: int | None = None,
        min_level: int = logging.NOTSET,
        logger_prefix: str | None = None,
        since_seq: int | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._buffer)
        if since_seq is not None:
            items = [e for e in items if e["seq"] > since_seq]
        if min_level > logging.NOTSET:
            items = [e for e in items if e["level_no"] >= min_level]
        if logger_prefix:
            items = [e for e in items if e["logger"].startswith(logger_prefix)]
        if limit is not None and len(items) > limit:
            items = items[-limit:]
        return items

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()


_GLOBAL_HANDLER: LogCaptureHandler | None = None


def install_global(capacity: int = 2000, level: int = logging.INFO) -> LogCaptureHandler:
    """Attach the ring handler to the root logger idempotently."""
    global _GLOBAL_HANDLER
    if _GLOBAL_HANDLER is not None:
        return _GLOBAL_HANDLER
    handler = LogCaptureHandler(capacity=capacity)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    _GLOBAL_HANDLER = handler
    return handler


def global_handler() -> LogCaptureHandler | None:
    return _GLOBAL_HANDLER


_START_NS = time.time_ns()


def process_start_ns() -> int:
    return _START_NS
