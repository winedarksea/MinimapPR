"""Perched relocation gate with ghost-perch revert (Field Mode W2-tracker, S3).

Gate: a fix 10-30 m from a perched track is the same bird on a new perch when
its species matches the track's top species, no other track of that species is
nearer, and the track has been silent long enough that one bird could not have
sung at both perches (``min_silence``). The track keeps its id and its filter
restarts at the new perch.

Ghost perch: a flock of one species, singing in alternating bouts from perches
20-30 m apart, looks exactly like one bird hopping. So the old perch is kept as
a *ghost* for ``ghost_seconds``. If a same-species fix lands back on the ghost
before it expires, the hop was really a second bird: the relocation is
reverted (the track returns to its old perch, the new perch gets its own new
track) and both tracks are barred from relocating while the ghost window lasts.
One mistake per pair instead of ping-ponging identities across the flock.
A track may hop at most once per ghost window, so a chain of same-species
perches (A->B->C) cannot overwrite the ghost that tells the flock apart; the
cost is that a real second hop inside the window starts a new track.

Revert listeners (Field Mode S-fixes integration): layers that followed the
hop (the bird-entity registry moves with its bound track) are told when it is
undone, so the post-hop fixes they took are re-read as the other bird's.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from minimappr.models import TrackState, TrackStatus

_NANOSECONDS_PER_SECOND = 1_000_000_000
_logger = logging.getLogger(__name__)


def _normalised(label: str | None) -> str:
    return (label or "").strip().lower()


@dataclass(slots=True)
class RelocationGhost:
    track_id: str
    perch_position_m: tuple[float, float, float]
    left_at_ns: int
    species: str


@dataclass(frozen=True, slots=True)
class RelocationRevert:
    """A relocation undone: *track_id* went back to its old perch; whatever it
    picked up since *hop_time_ns* belongs to *split_track_id* (another bird)."""

    track_id: str
    split_track_id: str
    hop_time_ns: int
    reverted_at_ns: int


RelocationRevertListener = Callable[[RelocationRevert], None]


class PerchedRelocationGate:
    def __init__(
        self,
        *,
        association_distance_m: float,
        relocation_gate_max_m: float,
        relocation_min_silence_seconds: float,
        relocation_ghost_seconds: float,
    ) -> None:
        self._association_distance_m = float(association_distance_m)
        self._relocation_gate_max_m = float(relocation_gate_max_m)
        self._min_silence_ns = int(float(relocation_min_silence_seconds) * _NANOSECONDS_PER_SECOND)
        self._ghost_ns = int(float(relocation_ghost_seconds) * _NANOSECONDS_PER_SECOND)
        self._ghosts: dict[str, RelocationGhost] = {}
        self._relocation_blocked_until_ns: dict[str, int] = {}
        self.relocations_accepted = 0
        self.relocations_reverted = 0
        self._revert_listeners: list[RelocationRevertListener] = []

    def add_revert_listener(self, listener: RelocationRevertListener) -> None:
        self._revert_listeners.append(listener)

    def find_relocation_match(
        self,
        *,
        timestamp_ns: int,
        position_m: tuple[float, float, float],
        detection_species: str,
        tracks: Iterable[TrackState],
        top_species_for: Callable[[TrackState], list[str]],
        is_perched: Callable[[str], bool],
    ) -> str | None:
        self._expire(timestamp_ns)
        species = _normalised(detection_species)
        measurement = np.asarray(position_m, dtype=np.float64)
        same_species: list[tuple[float, TrackState]] = []
        for track in tracks:
            if track.status == TrackStatus.DROPPED.value:
                continue
            if species in {_normalised(label) for label in top_species_for(track)}:
                distance = float(np.linalg.norm(measurement - np.asarray(track.position_m)))
                same_species.append((distance, track))
        if not same_species:
            return None
        distance, nearest = min(same_species, key=lambda item: item[0])
        # Only the nearest same-species track may relocate ("no other track of
        # that species is nearer").
        if not is_perched(nearest.id):
            return None
        if not self._association_distance_m <= distance <= self._relocation_gate_max_m:
            return None
        # One bird cannot sing at both perches within min_silence.
        if timestamp_ns - nearest.last_seen_ns < self._min_silence_ns:
            return None
        if self._relocation_blocked_until_ns.get(nearest.id, 0) > timestamp_ns:
            return None
        # One hop per ghost window: chaining hops would erase the ghost that
        # lets a flock be told apart from one bird.
        if nearest.id in self._ghosts:
            return None
        self.relocations_accepted += 1
        if self._ghost_ns > 0:
            self._ghosts[nearest.id] = RelocationGhost(
                track_id=nearest.id,
                perch_position_m=tuple(float(v) for v in nearest.position_m),  # type: ignore[arg-type]
                left_at_ns=timestamp_ns,
                species=species,
            )
        return nearest.id

    def find_ghost_hit(
        self, *, timestamp_ns: int, position_m: tuple[float, float, float], detection_species: str
    ) -> RelocationGhost | None:
        """A same-species fix back on a fresh ghost perch, or ``None``."""
        self._expire(timestamp_ns)
        species = _normalised(detection_species)
        measurement = np.asarray(position_m, dtype=np.float64)
        for ghost in self._ghosts.values():
            if ghost.species != species:
                continue
            if np.linalg.norm(measurement - np.asarray(ghost.perch_position_m)) <= self._association_distance_m:
                return ghost
        return None

    def mark_reverted(self, ghost: RelocationGhost, new_track_id: str, timestamp_ns: int) -> None:
        self._ghosts.pop(ghost.track_id, None)
        blocked_until = timestamp_ns + self._ghost_ns
        self._relocation_blocked_until_ns[ghost.track_id] = blocked_until
        self._relocation_blocked_until_ns[new_track_id] = blocked_until
        self.relocations_reverted += 1
        revert = RelocationRevert(
            track_id=ghost.track_id,
            split_track_id=new_track_id,
            hop_time_ns=int(ghost.left_at_ns),
            reverted_at_ns=int(timestamp_ns),
        )
        for listener in self._revert_listeners:
            try:
                listener(revert)
            except Exception:  # noqa: BLE001 - a listener bug must not break the tracker update
                _logger.exception("relocation revert listener failed")

    def forget(self, track_id: str) -> None:
        self._ghosts.pop(track_id, None)
        self._relocation_blocked_until_ns.pop(track_id, None)

    def _expire(self, timestamp_ns: int) -> None:
        expired = [tid for tid, g in self._ghosts.items() if timestamp_ns - g.left_at_ns > self._ghost_ns]
        for track_id in expired:
            self._ghosts.pop(track_id, None)
        stale_blocks = [tid for tid, until in self._relocation_blocked_until_ns.items() if until <= timestamp_ns]
        for track_id in stale_blocks:
            self._relocation_blocked_until_ns.pop(track_id, None)
