"""Partial-constraint binding of unpositioned detections to tracks (W2-tracker, S3).

A bout that fails a full fix (omni-only node, multipath-ruined TDOA, one node
heard it) used to become a separate unlocalized event, so a bird localized on
one bout with a weak ID and classified on another with a weak fix was never
both. Here such a detection binds to an existing recent track when

1. its species evidence overlaps the track's top species (top-k on each side),
2. every partial spatial constraint it carries is consistent with the track's
   position (normalized residual <= ``max_normalized_residual``), and
3. no other candidate track fits comparably well (an omni bout between two
   same-species birds is unattributable, so it stays unbound).

Constraints (all residuals are dimensionless, 1.0 = one sigma / one margin):

- ``BearingConstraint``  tetra-node DOA ray; residual = angle error / sigma.
- ``HyperbolaConstraint`` one cross-node TDOA; residual = range-difference error / sigma.
- ``SnrRankingConstraint`` the reporting node should be the nearest node, within a
  loose level margin (1/r spreading: margin_dB -> distance ratio 10^(dB/20)).
  Deliberately loose so faint, distant or shielded birds are never rejected on level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Below this length a vector has no usable direction.
_MINIMUM_DIRECTION_NORM = 1e-9
_DECIBELS_PER_AMPLITUDE_DECADE = 20.0


@dataclass(frozen=True)
class BearingConstraint:
    origin_m: np.ndarray
    unit_direction: np.ndarray
    sigma_rad: float
    kind: str = "bearing"
    maximum_normalized_residual: float | None = None

    def normalized_residual(self, position_m: np.ndarray) -> float:
        offset = np.asarray(position_m, dtype=np.float64) - self.origin_m
        distance = float(np.linalg.norm(offset))
        if distance < _MINIMUM_DIRECTION_NORM:
            return 0.0
        cosine = float(np.clip(np.dot(offset / distance, self.unit_direction), -1.0, 1.0))
        return math.acos(cosine) / self.sigma_rad

    def corrected_position(self, position_m: np.ndarray) -> np.ndarray | None:
        """Closest point on the ray (range unknown, so keep the track's range)."""
        offset = np.asarray(position_m, dtype=np.float64) - self.origin_m
        along = float(np.dot(offset, self.unit_direction))
        if along <= 0.0:
            return None
        return self.origin_m + along * self.unit_direction


@dataclass(frozen=True)
class HyperbolaConstraint:
    sensor_m: np.ndarray
    reference_sensor_m: np.ndarray
    # |x - sensor| - |x - reference| measured, i.e. sound_speed * tdoa.
    range_difference_m: float
    sigma_m: float
    kind: str = "hyperbola"
    maximum_normalized_residual: float | None = None

    def _range_difference_error(self, position_m: np.ndarray) -> float:
        position = np.asarray(position_m, dtype=np.float64)
        predicted = float(
            np.linalg.norm(position - self.sensor_m) - np.linalg.norm(position - self.reference_sensor_m)
        )
        return predicted - self.range_difference_m

    def normalized_residual(self, position_m: np.ndarray) -> float:
        return abs(self._range_difference_error(position_m)) / self.sigma_m

    def corrected_position(self, position_m: np.ndarray) -> np.ndarray | None:
        """One Gauss-Newton step onto the hyperboloid."""
        position = np.asarray(position_m, dtype=np.float64)
        to_sensor = position - self.sensor_m
        to_reference = position - self.reference_sensor_m
        sensor_norm = float(np.linalg.norm(to_sensor))
        reference_norm = float(np.linalg.norm(to_reference))
        if min(sensor_norm, reference_norm) < _MINIMUM_DIRECTION_NORM:
            return None
        gradient = to_sensor / sensor_norm - to_reference / reference_norm
        gradient_norm_squared = float(np.dot(gradient, gradient))
        if gradient_norm_squared < _MINIMUM_DIRECTION_NORM:
            return None
        return position - self._range_difference_error(position) * gradient / gradient_norm_squared


@dataclass(frozen=True)
class SnrRankingConstraint:
    reporting_node_m: np.ndarray
    other_node_positions_m: tuple[np.ndarray, ...]
    level_margin_db: float
    kind: str = "snr_ranking"
    # The margin is already the tolerance (not a sigma), so it is a hard bound.
    maximum_normalized_residual: float | None = 1.0

    def normalized_residual(self, position_m: np.ndarray) -> float:
        if not self.other_node_positions_m:
            return 0.0
        position = np.asarray(position_m, dtype=np.float64)
        reporting_distance = float(np.linalg.norm(position - self.reporting_node_m))
        nearest_other = min(float(np.linalg.norm(position - node)) for node in self.other_node_positions_m)
        if reporting_distance <= nearest_other:
            return 0.0
        excess_db = _DECIBELS_PER_AMPLITUDE_DECADE * math.log10(
            reporting_distance / max(nearest_other, _MINIMUM_DIRECTION_NORM)
        )
        return excess_db / self.level_margin_db

    def corrected_position(self, position_m: np.ndarray) -> np.ndarray | None:
        return None


PartialConstraint = BearingConstraint | HyperbolaConstraint | SnrRankingConstraint


@dataclass(frozen=True, slots=True)
class PartialBindingConfig:
    window_seconds: float = 25.0
    max_normalized_residual: float = 3.0
    species_top_k: int = 3
    soft_update_gain: float = 0.3
    ambiguity_score_ratio: float = 0.5


@dataclass(slots=True)
class PartialBindingCandidate:
    track_id: str
    position_m: np.ndarray
    last_seen_ns: int
    top_species_labels: list[str]


@dataclass(slots=True)
class PartialBindingDecision:
    track_id: str
    mean_squared_residual: float
    corrected_position_m: np.ndarray | None
    constraint_kinds: list[str] = field(default_factory=list)


def _species_overlap(detection_top_labels: list[str], track_top_labels: list[str]) -> bool:
    detection = {label.strip().lower() for label in detection_top_labels}
    return any(label.strip().lower() in detection for label in track_top_labels)


def choose_track_for_partial_constraints(
    *,
    timestamp_ns: int,
    detection_top_species_labels: list[str],
    constraints: list[PartialConstraint],
    candidates: list[PartialBindingCandidate],
    config: PartialBindingConfig,
) -> PartialBindingDecision | None:
    """Best uniquely-consistent candidate track, or ``None`` (stay unbound)."""
    if not constraints or not detection_top_species_labels:
        return None
    window_ns = int(config.window_seconds * 1_000_000_000)
    scored: list[tuple[float, PartialBindingCandidate]] = []
    for candidate in candidates:
        if not 0 <= timestamp_ns - candidate.last_seen_ns <= window_ns:
            continue
        if not _species_overlap(detection_top_species_labels, candidate.top_species_labels):
            continue
        residuals = [constraint.normalized_residual(candidate.position_m) for constraint in constraints]
        if any(
            residual > _residual_gate(constraint, config)
            for residual, constraint in zip(residuals, constraints)
        ):
            continue
        scored.append((float(np.mean(np.square(residuals))), candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0])
    best_score, best = scored[0]
    if len(scored) > 1 and not best_score < config.ambiguity_score_ratio * scored[1][0]:
        return None
    return PartialBindingDecision(
        track_id=best.track_id,
        mean_squared_residual=best_score,
        corrected_position_m=_soft_corrected_position(best.position_m, constraints, config.soft_update_gain),
        constraint_kinds=[constraint.kind for constraint in constraints],
    )


def _residual_gate(constraint: PartialConstraint, config: PartialBindingConfig) -> float:
    own_gate = constraint.maximum_normalized_residual
    if own_gate is None:
        return config.max_normalized_residual
    return min(own_gate, config.max_normalized_residual)


def _soft_corrected_position(
    position_m: np.ndarray, constraints: list[PartialConstraint], gain: float
) -> np.ndarray | None:
    """Move the track a fraction *gain* toward each constraint's nearest point."""
    if gain <= 0.0:
        return None
    position = np.asarray(position_m, dtype=np.float64).copy()
    moved = False
    for constraint in constraints:
        target = constraint.corrected_position(position)
        if target is None:
            continue
        position = position + gain * (target - position)
        moved = True
    return position if moved else None
