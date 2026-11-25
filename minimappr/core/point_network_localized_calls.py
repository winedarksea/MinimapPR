"""Localized calls and their arrival timing at a sensor (plan item P7).

A ``LocalizedCall`` is one solve placed in the site frame: where the source
is, and the span of audio the solve heard on its alignment reference's
timeline (the reporting / triggering node, whose trigger the solve window is
centred on). Moving that span to another sensor's timeline only needs the
propagation-distance difference, so no emission time has to be guessed.

``LocalizedCallRegistry`` remembers recent calls (time- and count-bounded) so a
point-network clip can see whether another caller overlaps it.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True, slots=True)
class LocalizedCall:
    call_id: str
    source_position_m: np.ndarray
    alignment_reference_sensor_id: str
    alignment_reference_position_m: np.ndarray
    # The solve's event span on the alignment reference's timeline.
    event_span_start_ns: int
    event_span_end_ns: int
    sound_speed_mps: float

    def arrival_offset_seconds_at(self, sensor_position_m: np.ndarray) -> float:
        """Arrival at ``sensor_position_m`` minus arrival at the alignment reference."""
        source = self.source_position_m
        return float(
            (np.linalg.norm(source - sensor_position_m) - np.linalg.norm(source - self.alignment_reference_position_m))
            / self.sound_speed_mps
        )

    def event_span_at(self, sensor_position_m: np.ndarray) -> tuple[int, int]:
        """The event span moved onto ``sensor_position_m``'s timeline."""
        shift_ns = seconds_to_nanoseconds(self.arrival_offset_seconds_at(sensor_position_m))
        return self.event_span_start_ns + shift_ns, self.event_span_end_ns + shift_ns


def seconds_to_nanoseconds(seconds: float) -> int:
    return int(round(seconds * NANOSECONDS_PER_SECOND))


class LocalizedCallRegistry:
    """Recent localized calls, newest last; re-registering a call id replaces it
    (an event solve and the render covering it share one solve id)."""

    def __init__(self, *, retention_seconds: float, max_records: int) -> None:
        self._retention_ns = seconds_to_nanoseconds(retention_seconds)
        self._max_records = max(1, int(max_records))
        self._calls: OrderedDict[str, LocalizedCall] = OrderedDict()
        self._newest_span_end_ns: int | None = None

    def register(self, call: LocalizedCall) -> None:
        self._calls.pop(call.call_id, None)
        self._calls[call.call_id] = call
        if self._newest_span_end_ns is None or call.event_span_end_ns > self._newest_span_end_ns:
            self._newest_span_end_ns = call.event_span_end_ns
        self._prune()

    def calls_other_than(self, call_id: str) -> list[LocalizedCall]:
        return [call for other_id, call in self._calls.items() if other_id != call_id]

    def __len__(self) -> int:
        return len(self._calls)

    def _prune(self) -> None:
        horizon_ns = None if self._newest_span_end_ns is None else self._newest_span_end_ns - self._retention_ns
        if horizon_ns is not None:
            for stale_id in [cid for cid, call in self._calls.items() if call.event_span_end_ns < horizon_ns]:
                del self._calls[stale_id]
        while len(self._calls) > self._max_records:
            self._calls.popitem(last=False)


__all__ = ["LocalizedCall", "LocalizedCallRegistry", "seconds_to_nanoseconds"]
