"""Per-zone rolling sound-pressure-level window.

``spl_db`` exists on the system only per-``DetectionEvent`` (models.py). There is
no aggregate zone SPL anywhere, so this is a new derived concept, built from the
detection events tee'd off ``LiveEventHub``.

The window reports its **max**, not its mean: a mean over a 60 s window buries a
gunshot under 59 s of quiet, which is the opposite of what an operator wants a
"how loud was that zone" number for.

A zone with no sample in the window reports ``None``, which the state mapper
publishes as HA's unknown sentinel — never 0.0 dB, which would be a reading we
never took.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque


class ZoneSplAggregator:
    def __init__(self, window_seconds: float) -> None:
        if window_seconds <= 0.0:
            raise ValueError("window_seconds must be > 0")
        self._window_ns = int(window_seconds * 1_000_000_000)
        # zone_id -> samples ordered oldest-first, so expiry is a popleft loop.
        self._samples: dict[str, deque[tuple[int, float]]] = defaultdict(deque)

    def observe(self, *, zone_ids: list[str], spl_db: float | None, timestamp_ns: int) -> None:
        """Record one detection's SPL against every zone it fell inside.

        A detection with no zones contributes nothing: there is no "site SPL"
        entity, and attributing it to an arbitrary zone would be a fabrication.
        """
        if spl_db is None:
            return
        try:
            value = float(spl_db)
        except (TypeError, ValueError):
            return
        for zone_id in zone_ids:
            key = str(zone_id).strip()
            if key:
                self._samples[key].append((timestamp_ns, value))

    def max_for_zone(self, zone_id: str, *, now_ns: int | None = None) -> float | None:
        """Window max for one zone, or None when the window holds no sample."""
        key = str(zone_id).strip()
        samples = self._samples.get(key)
        if not samples:
            return None
        self._expire(samples, now_ns=now_ns)
        if not samples:
            # Fully expired: drop the deque so a long-quiet site does not retain
            # one empty entry per zone it ever heard.
            self._samples.pop(key, None)
            return None
        return max(value for _timestamp, value in samples)

    def prune(self, *, now_ns: int | None = None) -> None:
        """Expire every zone's window. Called once per publish cycle."""
        for key in list(self._samples):
            samples = self._samples[key]
            self._expire(samples, now_ns=now_ns)
            if not samples:
                del self._samples[key]

    def tracked_zone_count(self) -> int:
        return len(self._samples)

    def _expire(self, samples: deque[tuple[int, float]], *, now_ns: int | None) -> None:
        cutoff = (now_ns if now_ns is not None else time.time_ns()) - self._window_ns
        while samples and samples[0][0] < cutoff:
            samples.popleft()
