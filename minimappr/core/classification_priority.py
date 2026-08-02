"""Significance scoring and priority ordering for the classification stage.

The classification lane is the pipeline's most expensive stage and is routinely
oversubscribed: on a live box it has run ~2.5 s per item against arrivals an
order of magnitude faster, pinning its queue at max depth. A FIFO queue under
that load decides what to classify purely by arrival order, so a faint,
unlocalized, one-off trigger displaces a strong event that lands on an existing
track.

This module scores each localized product once, at enqueue time, and orders the
queue by that score. It does not make the stage faster — it decides *which* work
survives a queue that cannot drain, and which item is shed when it overflows.

Score composition (all terms in [0, 1], weighted to a significance in [0, 1]):

  * **track affinity** — does this land where something is already being
    tracked? Continuing evidence for a live track is worth more than an isolated
    blip, and it is the single strongest "this is really something" signal
    available before inference runs.
  * **localization confidence** — a well-conditioned solve implies a real,
    coherent wavefront rather than a noise excursion.
  * **capability tier** — a full 3D fix outranks a 2D fix, which outranks a
    classification-only product.
  * **signal excess** — how far the classification window sits above the
    trigger floor, in dB. Deliberately capped and gently sloped so genuinely
    faint long-range events are de-prioritized, never excluded.
  * **corroboration** — how many sensors contributed; a multi-sensor (and
    especially cross-node) event is harder to explain as local noise.

Degraded classification audio (gaps in coverage) applies a multiplicative
penalty rather than a term, because coverage loss undermines every other signal
at once.

Per-classifier importance is deliberately absent: which classifier will claim a
window is not knowable until inference runs, so it cannot inform admission.
Routing already decides that downstream.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence


# Lower sorts first in a min-heap, so the most urgent work has the smallest key.
# The shutdown sentinel uses -inf rather than a fixed negative bucket so it wins
# against any real key regardless of how many buckets are configured.
_SENTINEL_KEY: tuple[float, int] = (float("-inf"), 0)


@dataclass(frozen=True, slots=True)
class PriorityWeights:
    """Relative contribution of each significance term. Normalized on use."""

    track_affinity: float = 0.35
    localization_confidence: float = 0.25
    capability_tier: float = 0.15
    signal_excess: float = 0.15
    corroboration: float = 0.10

    def normalized(self) -> "PriorityWeights":
        total = (
            self.track_affinity
            + self.localization_confidence
            + self.capability_tier
            + self.signal_excess
            + self.corroboration
        )
        if total <= 0.0:
            # Degenerate configuration: fall back to an even split rather than
            # dividing by zero and silently scoring everything identically.
            return PriorityWeights(0.2, 0.2, 0.2, 0.2, 0.2)
        return PriorityWeights(
            track_affinity=self.track_affinity / total,
            localization_confidence=self.localization_confidence / total,
            capability_tier=self.capability_tier / total,
            signal_excess=self.signal_excess / total,
            corroboration=self.corroboration / total,
        )


_CAPABILITY_TIER_SCORE = {
    "full_3d": 1.0,
    "2d": 0.6,
    "classification_only": 0.3,
    "alerting_only": 0.1,
}

# Sensor count at which corroboration saturates. Four is one full tetrahedral
# array; beyond that the extra evidence is cross-node and already reflected in
# the localization confidence and tier terms.
_CORROBORATION_SATURATION = 4.0

# Signal excess is scored across this many dB above the trigger floor. Kept
# shallow on purpose: a 6 dB event is not six times more worth classifying than
# a 1 dB one, and long-range work is faint by nature.
_SIGNAL_EXCESS_SPAN_DB = 18.0


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def track_affinity(
    position_m: Sequence[float] | None,
    track_positions_m: Iterable[Sequence[float]],
    *,
    radius_m: float,
) -> float:
    """1.0 on top of an active track, decaying linearly to 0.0 at ``radius_m``."""
    if position_m is None or radius_m <= 0.0:
        return 0.0
    best = 0.0
    for track_position in track_positions_m:
        if track_position is None or len(track_position) < 3:
            continue
        dx = float(position_m[0]) - float(track_position[0])
        dy = float(position_m[1]) - float(track_position[1])
        dz = float(position_m[2]) - float(track_position[2])
        distance_m = math.sqrt(dx * dx + dy * dy + dz * dz)
        if not math.isfinite(distance_m):
            continue
        affinity = 1.0 - (distance_m / radius_m)
        if affinity > best:
            best = affinity
            if best >= 1.0:
                break
    return _clamp01(best)


def signal_excess(signal_rms: float | None, trigger_rms: float) -> float:
    """Score the classification window's level above the trigger floor."""
    if signal_rms is None or signal_rms <= 0.0 or trigger_rms <= 0.0:
        return 0.0
    excess_db = 20.0 * math.log10(signal_rms / trigger_rms)
    if not math.isfinite(excess_db) or excess_db <= 0.0:
        return 0.0
    return _clamp01(excess_db / _SIGNAL_EXCESS_SPAN_DB)


def corroboration(sensor_count: int) -> float:
    if sensor_count <= 1:
        return 0.0
    return _clamp01((sensor_count - 1) / (_CORROBORATION_SATURATION - 1))


def score_significance(
    *,
    localization_confidence: float | None,
    capability_tier: str | None,
    sensor_count: int,
    signal_rms: float | None,
    trigger_rms: float,
    position_m: Sequence[float] | None,
    track_positions_m: Iterable[Sequence[float]] = (),
    track_radius_m: float = 50.0,
    audio_degraded: bool = False,
    weights: PriorityWeights | None = None,
) -> float:
    """Combine the admission signals into a single significance in [0, 1]."""
    w = (weights or PriorityWeights()).normalized()
    tier_score = _CAPABILITY_TIER_SCORE.get((capability_tier or "").strip().lower(), 0.3)
    significance = (
        w.track_affinity * track_affinity(position_m, track_positions_m, radius_m=track_radius_m)
        + w.localization_confidence * _clamp01(float(localization_confidence or 0.0))
        + w.capability_tier * tier_score
        + w.signal_excess * signal_excess(signal_rms, trigger_rms)
        + w.corroboration * corroboration(int(sensor_count))
    )
    if audio_degraded:
        # Coverage gaps corrupt every other term at once, so this scales the
        # whole score rather than subtracting a fixed amount.
        significance *= 0.5
    return _clamp01(significance)


def priority_key(
    *,
    significance: float,
    event_time_ns: int,
    buckets: int = 10,
) -> tuple[int, int]:
    """Coarse significance first, newest-first within a bucket.

    Bucketing is what makes this "recency *and* significance" rather than one
    or the other: items of comparable significance are ordered purely by how
    fresh they are, while a materially more significant item still jumps the
    queue. Both key elements are negated because the heap pops the smallest.
    """
    bucket_count = max(1, int(buckets))
    bucket = int(round(_clamp01(significance) * bucket_count))
    return (-bucket, -int(event_time_ns))


class PriorityStageQueue(asyncio.Queue):
    """An ``asyncio.Queue`` that pops the most significant item first.

    Drop-in for the plain stage queue: producers still ``put_nowait`` raw
    pipeline items and consumers still ``get`` them, so backpressure accounting,
    ``join()``, and ``task_done()`` are unchanged. Ordering is applied
    internally via the documented ``_init``/``_put``/``_get`` hooks that
    ``asyncio.PriorityQueue`` itself uses.

    ``None`` is the shutdown sentinel and always sorts ahead of real work so a
    stop request is never stuck behind a full queue.
    """

    def __init__(self, maxsize: int = 0, *, key: Callable[[Any], tuple[float, int]]) -> None:
        self._key = key
        # Ties on (bucket, -event_time_ns) fall back to insertion order, which
        # also keeps heapq from ever comparing two pipeline items directly.
        self._counter = itertools.count()
        super().__init__(maxsize)

    def _init(self, maxsize: int) -> None:
        self._queue: list[tuple[tuple[float, int], int, Any]] = []

    def _put(self, item: Any) -> None:
        key = _SENTINEL_KEY if item is None else self._key(item)
        heapq.heappush(self._queue, (key, next(self._counter), item))

    def _get(self) -> Any:
        return heapq.heappop(self._queue)[2]

    def peek_keys(self) -> list[tuple[float, int]]:
        """Queued sort keys in heap order — for tests and diagnostics."""
        return [entry[0] for entry in self._queue]

    def evict_least_significant(self) -> Any | None:
        """Remove and return the lowest-priority queued item.

        A priority queue must shed from the *tail*, not the head: evicting the
        head would discard exactly the work admission control just decided was
        most worth doing. Never evicts the shutdown sentinel.
        """
        if not self._queue:
            return None
        worst_index = max(range(len(self._queue)), key=lambda i: self._queue[i][:2])
        if self._queue[worst_index][2] is None:
            return None
        entry = self._queue[worst_index]
        self._queue[worst_index] = self._queue[-1]
        self._queue.pop()
        # Re-heapify rather than sift: the queue is bounded at a few dozen items,
        # so the O(n) cost is irrelevant and it avoids depending on heapq's
        # private sift helpers to restore the invariant.
        heapq.heapify(self._queue)
        # Mirror get_nowait()'s side effect so a blocked producer can proceed.
        self._wakeup_next(self._putters)
        return entry[2]
