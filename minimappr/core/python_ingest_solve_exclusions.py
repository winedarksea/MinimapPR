"""Per-solve sensor exclusions on the Python-ingest path (plan item P8).

The Rust solve contract lists every excluded sensor with a reason
(``solve_provenance.excluded_sensors``). The Python-ingest solve has no
provenance block, so its exclusions ride on the localized product's
classification features and reach the stored detection's feature summary
under ``localization_excluded_sensors``, with the same reason spelling.

``SolveSensorExclusionLedger`` counts each excluded sensor once per solve
(buffer reads would count it several times per solve, plus omni-scanner and
audio-debug reads) and keeps a recent window, so the pipeline view reports a
mismatch that is happening now rather than one that happened once.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import Any

from minimappr.core.synchronized_window_skips import SAMPLE_RATE_MISMATCH_SKIP_REASON

LOCALIZATION_EXCLUDED_SENSORS_FEATURE_KEY = "localization_excluded_sensors"
# Long enough to span several solves of a quiet site, short enough that a
# fixed mismatch stops showing within minutes.
DEFAULT_RECENT_EXCLUSION_WINDOW_SECONDS = 300.0


class SolveSensorExclusionLedger:
    """Cumulative per-sensor, per-reason exclusion counts (one per solve) plus
    the same counts over the last ``recent_window_seconds``.

    Cumulative counts are bounded by the number of distinct sensors; the
    recent entries by the solves inside the window.
    """

    def __init__(
        self,
        *,
        recent_window_seconds: float = DEFAULT_RECENT_EXCLUSION_WINDOW_SECONDS,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.recent_window_seconds = float(recent_window_seconds)
        self._monotonic_clock = monotonic_clock
        self._counts: dict[str, dict[str, int]] = {}
        self._recent: deque[tuple[float, str, str]] = deque()

    def record_solve(self, exclusions: list[dict[str, Any]]) -> None:
        now_seconds = self._monotonic_clock()
        for exclusion in exclusions:
            sensor_id, reason = str(exclusion["sensor_id"]), str(exclusion["reason"])
            reasons = self._counts.setdefault(sensor_id, {})
            reasons[reason] = reasons.get(reason, 0) + 1
            self._recent.append((now_seconds, sensor_id, reason))
        self._prune(now_seconds)

    def counts_by_sensor(self) -> dict[str, dict[str, int]]:
        return {sensor_id: dict(reasons) for sensor_id, reasons in self._counts.items()}

    def recent_counts_by_sensor(self) -> dict[str, dict[str, int]]:
        self._prune(self._monotonic_clock())
        recent: dict[str, dict[str, int]] = {}
        for _, sensor_id, reason in self._recent:
            reasons = recent.setdefault(sensor_id, {})
            reasons[reason] = reasons.get(reason, 0) + 1
        return recent

    def _prune(self, now_seconds: float) -> None:
        horizon_seconds = now_seconds - self.recent_window_seconds
        while self._recent and self._recent[0][0] < horizon_seconds:
            self._recent.popleft()


def counts_by_reason(counts_by_sensor: dict[str, dict[str, int]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for reasons in counts_by_sensor.values():
        for reason, count in reasons.items():
            totals[reason] = totals.get(reason, 0) + count
    return totals


async def python_ingest_excluded_sensor_features(
    *, buffer, registry, sensor_ids: list[str], solve_sample_rate_hz: int, ledger: SolveSensorExclusionLedger
) -> dict[str, Any]:
    """``{"localization_excluded_sensors": [...]}`` for sensors a solve at
    ``solve_sample_rate_hz`` left out for their rate; ``{}`` when none. Each
    exclusion is counted once in ``ledger``."""
    mismatched = await buffer.sample_rate_mismatched_sensors(sensor_ids, solve_sample_rate_hz)
    if not mismatched:
        return {}
    node_ids = await registry.sensor_node_ids(sorted(mismatched))
    exclusions = [
        {
            "sensor_id": sensor_id,
            "node_id": node_ids.get(sensor_id),
            "reason": SAMPLE_RATE_MISMATCH_SKIP_REASON,
            "sensor_sample_rate_hz": mismatch.buffered_sample_rate_hz,
            "solve_sample_rate_hz": mismatch.requested_sample_rate_hz,
        }
        for sensor_id, mismatch in sorted(mismatched.items())
    ]
    ledger.record_solve(exclusions)
    return {LOCALIZATION_EXCLUDED_SENSORS_FEATURE_KEY: exclusions}


__all__ = [
    "DEFAULT_RECENT_EXCLUSION_WINDOW_SECONDS",
    "LOCALIZATION_EXCLUDED_SENSORS_FEATURE_KEY",
    "SolveSensorExclusionLedger",
    "counts_by_reason",
    "python_ingest_excluded_sensor_features",
]
