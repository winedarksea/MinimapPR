"""Concurrent-perch guard for perched association (Field Mode S-fixes tracking).

The S gate's same-species flock (three chickadees on perches 10-11 m apart,
bouts rotating every 1.2 s) ended up on one track: association accepted the
far fix, the perched filter snapped to it (same identity), and the entity
layer followed. A snap to a far perch is only "one bird hopping" when no
other bird of that species could be the singer:

- the track was silent for less than ``min_silence_seconds``: one bird cannot
  sing at both perches that quickly (the relocation gate's own rule); or
- another track of the same species sang *after* this track's last bout
  (within the last ``concurrent_activity_seconds``): the silence is just a
  neighbour's turn, so a fix on yet another perch is another bird, not this
  one hopping. The most recent singer of the species may still hop.

A fix "on another perch" is one the perched filter would snap to: beyond
``snap_distance_m`` and statistically inconsistent with the predicted perch
(same chi-square threshold the filter uses), so a far bird whose fixes carry
tens of metres of error still averages into its one track.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from minimappr.core.track_filters_perched import DEFAULT_RELOCATION_SNAP_MIN_MAHALANOBIS_SQUARED
from minimappr.models import TrackState, TrackStatus

_NANOSECONDS_PER_SECOND = 1_000_000_000
# Variance floor (m^2) so a missing or degenerate covariance cannot make the
# innovation covariance singular.
_MINIMUM_VARIANCE_M2 = 1.0e-6


@dataclass(frozen=True, slots=True)
class ConcurrentPerchGuardConfig:
    enabled: bool
    snap_distance_m: float
    min_silence_seconds: float
    concurrent_activity_seconds: float


def _normalised(label: str | None) -> str:
    return (label or "").strip().lower()


def _covariance_or_none(value: list[list[float]] | None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        covariance = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        return None
    return 0.5 * (covariance + covariance.T)


def fix_lands_on_another_perch(
    *,
    position_m: tuple[float, float, float],
    measurement_covariance_m2: list[list[float]] | None,
    track: TrackState,
    snap_distance_m: float,
) -> bool:
    """Would the perched filter read this fix as a new perch for *track*?"""
    innovation = np.asarray(position_m, dtype=np.float64) - np.asarray(track.position_m, dtype=np.float64)
    if not 0.0 < snap_distance_m < float(np.linalg.norm(innovation)):
        return False
    combined = np.eye(3) * _MINIMUM_VARIANCE_M2
    for covariance in (
        _covariance_or_none(measurement_covariance_m2),
        _covariance_or_none(track.position_covariance_m2),
    ):
        if covariance is not None:
            combined = combined + covariance
    try:
        mahalanobis_squared = float(innovation @ np.linalg.solve(combined, innovation))
    except np.linalg.LinAlgError:
        return True
    return mahalanobis_squared > DEFAULT_RELOCATION_SNAP_MIN_MAHALANOBIS_SQUARED


def concurrently_active_perch_conflicts(
    *,
    timestamp_ns: int,
    position_m: tuple[float, float, float],
    measurement_covariance_m2: list[list[float]] | None,
    detection_species: str | None,
    tracks: Iterable[TrackState],
    top_species_for: Callable[[TrackState], list[str]],
    is_perched: Callable[[str], bool],
    config: ConcurrentPerchGuardConfig,
) -> set[str]:
    """Ids of perched same-species tracks this fix must not snap onto."""
    species = _normalised(detection_species)
    if not config.enabled or not species:
        return set()
    same_species = [
        track
        for track in tracks
        if track.status != TrackStatus.DROPPED.value
        and is_perched(track.id)
        and species in {_normalised(label) for label in top_species_for(track)}
    ]
    min_silence_ns = int(config.min_silence_seconds * _NANOSECONDS_PER_SECOND)
    concurrent_ns = int(config.concurrent_activity_seconds * _NANOSECONDS_PER_SECOND)
    conflicts: set[str] = set()
    for track in same_species:
        if not fix_lands_on_another_perch(
            position_m=position_m,
            measurement_covariance_m2=measurement_covariance_m2,
            track=track,
            snap_distance_m=config.snap_distance_m,
        ):
            continue
        sang_too_recently = timestamp_ns - track.last_seen_ns < min_silence_ns
        neighbour_sang_since = any(
            other.id != track.id
            and other.last_seen_ns > track.last_seen_ns
            and timestamp_ns - other.last_seen_ns <= concurrent_ns
            for other in same_species
        )
        if sang_too_recently or neighbour_sang_since:
            conflicts.add(track.id)
    return conflicts
