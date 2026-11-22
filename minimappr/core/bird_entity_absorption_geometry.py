"""Covariance-aware entity absorption (Field Mode S-fixes tracking).

The W3 absorption rule took any same-species fix within a fixed 12 m of a
pinpointed entity. The S gate showed both ways that fails:

- a same-species flock with perches 10-11 m apart collapsed into one entity
  (fixes with ~1 m cross-range error, well apart beyond their uncertainty);
- a far bird's fix solved 300 m out along the right bearing (range
  unobservable, covariance hundreds of metres along the ray) started a second
  entity for the same bird.

Both are a fixed radius standing in for the fix's own covariance. Here a fix
absorbs when its horizontal Mahalanobis distance to the entity (fix + entity
covariance, 2 dof) is within a chi-square gate, with a small distance floor so
overconfident covariances do not split one bird on jitter. A fix that is
itself too uncertain to pinpoint anything is *bearing evidence*: it may join a
better-located entity but must not move it. And two fixes inconsistent beyond
their covariances inside a short window are two birds, whatever the tracker
did with them.
"""

from __future__ import annotations

import math

import numpy as np

from minimappr.core.bird_entity_model import (
    BirdEntityRecord,
    EntityFixSample,
    Vec3,
    horizontal_distance_m,
    horizontal_major_sigma_m,
)
from minimappr.core.entity_reporting_config import EntityReportingConfig
from minimappr.core.range_projection import RANGE_BEARING_PROJECTED

NANOSECONDS_PER_SECOND = 1_000_000_000
# Variance floor (m^2) on the summed horizontal covariance so a degenerate
# (zero) covariance cannot make the solve singular.
_MINIMUM_HORIZONTAL_VARIANCE_M2 = 1.0e-6
# Recent fixes are kept for this many concurrency windows (late arrivals from
# the classification queue can be a little out of order).
_FIX_MEMORY_WINDOWS = 2


def _horizontal(covariance_m2: list[list[float]] | None) -> np.ndarray | None:
    if covariance_m2 is None:
        return None
    try:
        covariance = np.asarray(covariance_m2, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if covariance.ndim != 2 or covariance.shape[0] < 2 or covariance.shape[1] < 2:
        return None
    horizontal = covariance[:2, :2]
    if not np.all(np.isfinite(horizontal)):
        return None
    return 0.5 * (horizontal + horizontal.T)


def horizontal_mahalanobis_squared(
    first_m: Vec3,
    first_covariance_m2: list[list[float]] | None,
    second_m: Vec3,
    second_covariance_m2: list[list[float]] | None,
) -> float | None:
    """Horizontal Mahalanobis distance squared; None when neither side has a covariance."""
    covariances = [c for c in (_horizontal(first_covariance_m2), _horizontal(second_covariance_m2)) if c is not None]
    if not covariances:
        return None
    combined = sum(covariances) + np.eye(2) * _MINIMUM_HORIZONTAL_VARIANCE_M2
    residual = np.array([float(first_m[0]) - float(second_m[0]), float(first_m[1]) - float(second_m[1])])
    try:
        value = float(residual @ np.linalg.solve(combined, residual))
    except np.linalg.LinAlgError:
        return None
    return value if math.isfinite(value) else None


def fix_is_bearing_evidence(
    covariance_m2: list[list[float]] | None,
    range_projection_mode: str | None,
    config: EntityReportingConfig,
) -> bool:
    """A fix that could not pinpoint a bird on its own (range unobserved or too uncertain)."""
    if range_projection_mode == RANGE_BEARING_PROJECTED:
        return True
    sigma_m = horizontal_major_sigma_m(covariance_m2)
    return sigma_m is not None and sigma_m > config.pinpointed_max_horizontal_uncertainty_m


def should_absorb_as_bearing_evidence(
    entity: BirdEntityRecord,
    covariance_m2: list[list[float]] | None,
    range_projection_mode: str | None,
    config: EntityReportingConfig,
) -> bool:
    """Bearing evidence joining an entity located at least as well: absorb, don't move it."""
    if not config.bearing_evidence_absorption_enabled or not entity.is_spatial:
        return False
    if not fix_is_bearing_evidence(covariance_m2, range_projection_mode, config):
        return False
    fix_sigma_m = horizontal_major_sigma_m(covariance_m2)
    entity_sigma_m = entity.horizontal_uncertainty_m
    if fix_sigma_m is None or entity_sigma_m is None:
        return True
    return entity_sigma_m <= fix_sigma_m


def covariance_gated_fix_score(
    entity_position_m: Vec3,
    entity_covariance_m2: list[list[float]] | None,
    detection_position_m: Vec3,
    detection_covariance_m2: list[list[float]] | None,
    config: EntityReportingConfig,
) -> float | None:
    """Normalised mismatch in [0, 1] for a fix against a pinpointed entity, None = another bird.

    Falls back to the fixed W3 radius when the fix carries no covariance.
    """
    distance_m = horizontal_distance_m(detection_position_m, entity_position_m)
    if detection_covariance_m2 is None:
        if distance_m > config.absorption_max_fix_distance_m:
            return None
        return distance_m / config.absorption_max_fix_distance_m
    if distance_m <= config.absorption_min_fix_distance_m:
        return 0.0
    mahalanobis_squared = horizontal_mahalanobis_squared(
        detection_position_m, detection_covariance_m2, entity_position_m, entity_covariance_m2
    )
    if mahalanobis_squared is None or mahalanobis_squared > config.absorption_chi2_gate:
        return None
    return mahalanobis_squared / config.absorption_chi2_gate


def fixes_are_two_birds(first: EntityFixSample, second: EntityFixSample, config: EntityReportingConfig) -> bool:
    """Concurrent (within the window) and inconsistent beyond their covariances."""
    window_ns = int(config.concurrent_fix_window_seconds * NANOSECONDS_PER_SECOND)
    if abs(first.event_time_ns - second.event_time_ns) >= window_ns:
        return False
    distance_m = horizontal_distance_m(first.position_m, second.position_m)
    if distance_m <= config.absorption_min_fix_distance_m:
        return False
    mahalanobis_squared = horizontal_mahalanobis_squared(
        first.position_m, first.covariance_m2, second.position_m, second.covariance_m2
    )
    if mahalanobis_squared is None:
        return distance_m > config.absorption_max_fix_distance_m
    return mahalanobis_squared > config.absorption_chi2_gate


def concurrent_fix_conflict(
    entity: BirdEntityRecord, candidate: EntityFixSample | None, config: EntityReportingConfig
) -> bool:
    """Does *candidate* contradict a fix this entity took moments ago?"""
    if candidate is None or not config.absorption_covariance_gate_enabled:
        return False
    return any(fixes_are_two_birds(candidate, recent, config) for recent in entity.recent_fixes)


def remember_fix(entity: BirdEntityRecord, sample: EntityFixSample, config: EntityReportingConfig) -> None:
    """Keep the entity's recent raw fixes for the concurrency check (bounded by time)."""
    keep_ns = int(_FIX_MEMORY_WINDOWS * config.concurrent_fix_window_seconds * NANOSECONDS_PER_SECOND)
    newest_ns = max([sample.event_time_ns, *(fix.event_time_ns for fix in entity.recent_fixes)])
    entity.recent_fixes = [
        fix for fix in [*entity.recent_fixes, sample] if newest_ns - fix.event_time_ns <= keep_ns
    ]


def forget_fixes_since(entity: BirdEntityRecord, since_ns: int) -> int:
    """Drop the entity's recent fixes at or after *since_ns*; returns how many.

    Used when the tracker undoes a perched hop the entity followed (Field Mode
    S-fixes integration): the fixes taken at the new perch were another bird's,
    and keeping them would make this bird's next fix at its own perch look
    concurrent-inconsistent ("two birds") and split it into a duplicate entity.
    """
    kept = [fix for fix in entity.recent_fixes if fix.event_time_ns < since_ns]
    dropped = len(entity.recent_fixes) - len(kept)
    entity.recent_fixes = kept
    return dropped
