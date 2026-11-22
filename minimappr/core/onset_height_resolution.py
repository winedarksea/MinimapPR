"""Node-plane mirror (ground-image) height resolution for onset clusters (Field Mode S-fixes).

With coplanar node centroids, a fix and its mirror across the node plane give
identical cross-node TDOAs; only the compact arrays' elevation evidence differs,
and that is weak. The ground-reflection image arrives from the mirror side too,
so an unconstrained solve could land on it (countersinging: z -2.0 m for a bird
at 4.0 m). Resolution order:

1. the ground/canopy prior, when exactly one of the twins is admissible;
2. the cluster's SRP (which carries the compact arrays' elevation evidence),
   when one twin beats the other by ``min_score_ratio``;
3. otherwise the height is reported as uncertain: the fix moves to the node
   plane (midway between the twins) and the vertical variance grows to cover
   both, instead of picking one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from minimappr.core.onset_srp_geometry_prior import NodePlaneGroundPrior

HEIGHT_NOT_APPLICABLE = "not_applicable"
HEIGHT_RESOLVED_BY_PRIOR = "resolved_by_prior"
HEIGHT_RESOLVED_BY_SRP = "resolved_by_srp"
HEIGHT_AMBIGUOUS = "ambiguous"
HEIGHT_OUTSIDE_PRIOR = "outside_prior"
# Twins closer than this (m) are the same point: the fix already sits on the plane.
MIRROR_COINCIDENT_DISTANCE_M = 1.0e-3


@dataclass(frozen=True, slots=True)
class HeightResolution:
    position_m: np.ndarray
    added_vertical_variance_m2: float
    status: str


def resolve_node_plane_mirror(
    position_m: np.ndarray,
    *,
    prior: NodePlaneGroundPrior,
    score_points: Callable[[np.ndarray], np.ndarray],
    min_score_ratio: float,
) -> HeightResolution:
    position = np.asarray(position_m, dtype=np.float64)
    if not prior.is_planar:
        return HeightResolution(position, 0.0, HEIGHT_NOT_APPLICABLE)
    mirror = prior.mirror_across_node_plane(position)
    separation_m = float(np.linalg.norm(mirror - position))
    if separation_m < MIRROR_COINCIDENT_DISTANCE_M:
        return HeightResolution(position, 0.0, HEIGHT_NOT_APPLICABLE)
    twins = np.vstack([position, mirror])
    if prior.enabled:
        admissible = prior.admissible_mask(twins)
        if admissible[0] != admissible[1]:
            return HeightResolution(twins[int(admissible[1])], 0.0, HEIGHT_RESOLVED_BY_PRIOR)
        if not admissible.any():
            return HeightResolution(position, 0.0, HEIGHT_OUTSIDE_PRIOR)
    scores = np.asarray(score_points(twins), dtype=np.float64)
    better = int(np.argmax(scores))
    worse_score = float(scores[1 - better])
    if worse_score <= 0.0 or float(scores[better]) >= min_score_ratio * worse_score:
        return HeightResolution(twins[better], 0.0, HEIGHT_RESOLVED_BY_SRP)
    # Unresolvable: the node plane is the honest centre; +-half the separation is 1 sigma.
    return HeightResolution(0.5 * (position + mirror), (0.5 * separation_m) ** 2, HEIGHT_AMBIGUOUS)


def add_vertical_variance(covariance_m2: list[list[float]] | None, variance_m2: float) -> list[list[float]] | None:
    if covariance_m2 is None or variance_m2 <= 0.0:
        return covariance_m2
    covariance = np.asarray(covariance_m2, dtype=np.float64).copy()
    covariance[2, 2] += variance_m2
    return [[float(value) for value in row] for row in covariance]


@dataclass(frozen=True, slots=True)
class VerticalSupport:
    interval_z_m: tuple[float, float]
    peak_z_m: float


def vertical_support_at(
    position_m: np.ndarray,
    *,
    z_bounds_m: tuple[float, float],
    score_points: Callable[[np.ndarray], np.ndarray],
    sample_count: int,
    support_ratio: float,
) -> VerticalSupport | None:
    """SRP along the vertical through the fix, between the ground and the canopy ceiling.

    Planar layouts barely constrain height (only the compact arrays' elevation does), and
    the LM Jacobian understated it (a canopy bird at 9 m reported 5.4 +- 2.0 m). The heights
    scoring within ``support_ratio`` of the column peak bound where the bird can be.
    """
    low_z, high_z = z_bounds_m
    if not high_z > low_z:
        return None
    heights = np.linspace(low_z, high_z, sample_count)
    column = np.repeat(np.asarray(position_m, dtype=np.float64)[None, :], sample_count, axis=0)
    column[:, 2] = heights
    scores = np.asarray(score_points(column), dtype=np.float64)
    if not np.any(scores > 0.0):
        return None
    peak = int(np.argmax(scores))
    supported = np.flatnonzero(scores >= support_ratio * scores[peak])
    return VerticalSupport((float(heights[supported[0]]), float(heights[supported[-1]])), float(heights[peak]))


def apply_vertical_support(
    position_m: np.ndarray, covariance_m2: list[list[float]] | None, support: VerticalSupport | None
) -> tuple[np.ndarray, list[list[float]] | None]:
    """Move z to the column peak if the fix left the supported band; vertical std >= half the band."""
    if support is None:
        return position_m, covariance_m2
    position = np.asarray(position_m, dtype=np.float64).copy()
    low_z, high_z = support.interval_z_m
    if not low_z <= position[2] <= high_z:
        position[2] = support.peak_z_m
    if covariance_m2 is None:
        return position, covariance_m2
    covariance = np.asarray(covariance_m2, dtype=np.float64).copy()
    covariance[2, 2] = max(float(covariance[2, 2]), (0.5 * (high_z - low_z)) ** 2)
    return position, [[float(value) for value in row] for row in covariance]
