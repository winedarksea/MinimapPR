"""Sticky N-slot assignment of live tracks to fixed HA ``device_tracker`` entities.

Why a fixed pool at all: HA's entity registry persists every ``unique_id``
forever. Minting one entity per track id would grow the registry without bound
and leave an orphan per track that ever existed. A pool of N slots is bounded,
and the discovery payloads for all N are known ahead of time.

Two properties matter for the HA side to be usable:

* **Sticky** — once a track holds a slot it keeps it for its whole life, so an
  automation watching ``track_slot_03`` follows one target rather than watching
  targets shuffle underneath it every cycle.
* **Deterministic** — given the same track set, the same assignment results,
  regardless of dict iteration order. Ties break on the lowest track id.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrackSlotCandidate:
    """The subset of a track the allocator ranks and publishes."""
    track_id: str
    tqi: float
    label: str = ""
    lat: float | None = None
    lon: float | None = None
    altitude_m: float | None = None
    status: str = ""
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class TrackSlotAssignment:
    slot_index: int
    candidate: TrackSlotCandidate | None
    """None means the slot is vacant and should publish ``not_home``."""


class TrackSlotAllocator:
    def __init__(self, slot_count: int) -> None:
        if slot_count < 0:
            raise ValueError("slot_count must be >= 0")
        self._slot_count = slot_count
        self._assigned: dict[int, str] = {}

    @property
    def slot_count(self) -> int:
        return self._slot_count

    def assigned_track_ids(self) -> dict[int, str]:
        return dict(self._assigned)

    def assign(self, candidates: list[TrackSlotCandidate]) -> list[TrackSlotAssignment]:
        """Map the current track set onto the slot pool.

        Ranking is highest track-quality-index first so that when there are more
        tracks than slots, the pool shows the best-supported ones. A track that
        already holds a slot is never moved, even if a better track appears —
        eviction only happens when the incumbent disappears.
        """
        if self._slot_count == 0:
            self._assigned.clear()
            return []

        by_id = {candidate.track_id: candidate for candidate in candidates}
        # Release slots whose track is gone before ranking newcomers, so a
        # departing track frees its slot in the same cycle its replacement lands.
        for slot_index, track_id in list(self._assigned.items()):
            if track_id not in by_id:
                del self._assigned[slot_index]

        held = set(self._assigned.values())
        newcomers = sorted(
            (candidate for candidate in candidates if candidate.track_id not in held),
            key=lambda candidate: (-candidate.tqi, candidate.track_id),
        )
        free_slots = [index for index in range(self._slot_count) if index not in self._assigned]
        for slot_index, candidate in zip(free_slots, newcomers):
            self._assigned[slot_index] = candidate.track_id

        return [
            TrackSlotAssignment(
                slot_index=index,
                candidate=by_id.get(self._assigned[index]) if index in self._assigned else None,
            )
            for index in range(self._slot_count)
        ]
