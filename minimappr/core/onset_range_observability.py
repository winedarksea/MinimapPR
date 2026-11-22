"""Range observability of a multi-node onset cluster from its SRP range profile (Field Mode S-fixes).

Why: a 20 m layout barely sees wavefront curvature from a bird 80 m out,
especially when narrowband frames correlate envelopes (ms-wide peaks). The LM
still converges somewhere along the bearing (one bout landed at 391 m for a
bird at ~80 m) and its Jacobian covariance, scaled by the residual, called that
range observable. Range is judged here from the data instead: the cluster's SRP
is sampled along its bearing ridge (max over a small cone at each range, so the
ridge's bend near the array is followed), and the ranges scoring within
``support_ratio`` of the profile peak form the supported range interval.

* Narrow interval (far / near <= ``max_interval_ratio``, or at most
  ``interval_floor_m`` wide (a fraction of the layout aperture: inside the layout any ratio
  is large)): range observed. The
  fix is kept (moved to the profile peak if LM left the interval) and the radial
  variance grows to at least the interval half-width squared.
* Wide or open-ended interval: ``range_bearing_projected``. The fix sits at the
  profile peak on the ridge (the SRP's best range, inside the interval), and the
  radial standard deviation is the interval half-width. The interval itself is
  exported so consumers can show "somewhere 50-1000 m along this bearing".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from minimappr.core.localization_uncertainty import (
    covariance_to_nested_list,
    range_observability_from_covariance,
)
from minimappr.core.range_projection import (
    BEARING_PROJECTED_RANGE_OBSERVABILITY_CAP,
    RANGE_BEARING_PROJECTED,
    range_mode_is_unobservable,
)

EPSILON = 1.0e-9


@dataclass(frozen=True, slots=True)
class RangeProfileSettings:
    min_range_m: float
    max_range_m: float
    sample_count: int
    cone_half_angle_deg: float
    cone_step_deg: float
    support_ratio: float
    max_interval_ratio: float
    interval_floor_m: float


@dataclass(frozen=True, slots=True)
class RangeSupport:
    interval_m: tuple[float, float]
    peak_range_m: float
    peak_position_m: np.ndarray
    bearing: np.ndarray
    observable: bool


@dataclass(frozen=True, slots=True)
class RangeJudgedPosition:
    position_m: np.ndarray
    covariance_m2: list[list[float]] | None
    range_projection_mode: str | None
    range_observability: float | None
    range_interval_m: tuple[float, float] | None


def _cone_directions(bearing: np.ndarray, half_angle_deg: float, step_deg: float) -> np.ndarray:
    azimuth = float(np.arctan2(bearing[1], bearing[0]))
    elevation = float(np.arcsin(np.clip(bearing[2], -1.0, 1.0)))
    offsets = np.radians(np.arange(-half_angle_deg, half_angle_deg + EPSILON, step_deg)) if half_angle_deg > 0 else np.zeros(1)
    azimuths, elevations = np.meshgrid(azimuth + offsets, elevation + offsets)
    elevations = np.clip(elevations, -np.pi / 2, np.pi / 2)
    return np.column_stack(
        [
            (np.cos(elevations) * np.cos(azimuths)).ravel(),
            (np.cos(elevations) * np.sin(azimuths)).ravel(),
            np.sin(elevations).ravel(),
        ]
    )


def range_support_along_bearing(
    position_m: np.ndarray,
    origin_m: np.ndarray,
    *,
    score_points: Callable[[np.ndarray], np.ndarray],
    settings: RangeProfileSettings,
    admissible_mask: Callable[[np.ndarray], np.ndarray] | None = None,
) -> RangeSupport | None:
    offset = np.asarray(position_m, dtype=np.float64) - origin_m
    range_m = float(np.linalg.norm(offset))
    if not np.isfinite(range_m) or range_m <= EPSILON:
        return None
    bearing = offset / range_m
    ranges = np.geomspace(settings.min_range_m, settings.max_range_m, settings.sample_count)
    directions = _cone_directions(bearing, settings.cone_half_angle_deg, settings.cone_step_deg)
    points = origin_m + (ranges[:, None, None] * directions[None, :, :]).reshape(-1, 3)
    scores = np.asarray(score_points(points), dtype=np.float64).reshape(ranges.size, directions.shape[0])
    if admissible_mask is not None:
        # Below-ground / above-canopy candidates cannot be the source.
        scores = np.where(admissible_mask(points).reshape(scores.shape), scores, -np.inf)
    best_direction_index = np.argmax(scores, axis=1)
    profile = scores[np.arange(ranges.size), best_direction_index]
    if not np.any(np.isfinite(profile)) or float(np.nanmax(profile)) <= 0.0:
        return None
    peak_index = int(np.argmax(profile))
    supported = np.flatnonzero(profile >= settings.support_ratio * profile[peak_index])
    near_m, far_m = float(ranges[supported[0]]), float(ranges[supported[-1]])
    return RangeSupport(
        interval_m=(near_m, far_m),
        peak_range_m=float(ranges[peak_index]),
        peak_position_m=points.reshape(ranges.size, -1, 3)[peak_index, best_direction_index[peak_index]],
        bearing=bearing,
        observable=far_m <= settings.max_interval_ratio * near_m or far_m - near_m <= settings.interval_floor_m,
    )


def _radial_inflated_covariance(covariance: np.ndarray, radial_axis: np.ndarray, radial_std_m: float) -> np.ndarray:
    radial_variance = float(radial_axis @ covariance @ radial_axis)
    deficit = radial_std_m**2 - radial_variance
    if deficit <= 0.0:
        return covariance
    return covariance + deficit * np.outer(radial_axis, radial_axis)


def judge_range_from_profile(
    *,
    position_m: np.ndarray,
    covariance_m2: list[list[float]] | None,
    range_projection_mode: str | None,
    range_observability: float | None,
    origin_m: np.ndarray,
    support: RangeSupport | None,
) -> RangeJudgedPosition:
    """Apply the profile verdict. Never upgrades an already-unobservable range."""
    position = np.asarray(position_m, dtype=np.float64)
    unchanged = RangeJudgedPosition(position, covariance_m2, range_projection_mode, range_observability, None)
    if support is None or covariance_m2 is None or range_mode_is_unobservable(range_projection_mode):
        return unchanged
    near_m, far_m = support.interval_m
    radial_std_m = 0.5 * (far_m - near_m)
    covariance = np.asarray(covariance_m2, dtype=np.float64)
    current_range_m = float(np.linalg.norm(position - origin_m))
    if support.observable:
        if not near_m <= current_range_m <= far_m:
            position = np.asarray(support.peak_position_m, dtype=np.float64)
        radial_axis = (position - origin_m) / max(float(np.linalg.norm(position - origin_m)), EPSILON)
        covariance = _radial_inflated_covariance(covariance, radial_axis, radial_std_m)
        return RangeJudgedPosition(
            position_m=position,
            covariance_m2=covariance_to_nested_list(covariance),
            range_projection_mode=range_projection_mode,
            range_observability=range_observability_from_covariance(covariance, position - origin_m),
            range_interval_m=support.interval_m,
        )
    # The profile peak is the SRP's own best range on the ridge; an arbitrary default range
    # could sit far from every supported range's evidence (flock scene: ghosts 30 m up).
    projected = np.asarray(support.peak_position_m, dtype=np.float64)
    report_range_m = float(np.linalg.norm(projected - origin_m))
    radial_axis = (projected - origin_m) / max(report_range_m, EPSILON)
    # Lateral spread scales with range along the cone; the radial axis is the interval.
    scale = report_range_m / max(current_range_m, EPSILON)
    lateral_projector = np.eye(3) - np.outer(radial_axis, radial_axis)
    lateral = lateral_projector @ covariance @ lateral_projector * scale**2
    projected_covariance = lateral + radial_std_m**2 * np.outer(radial_axis, radial_axis)
    return RangeJudgedPosition(
        position_m=projected,
        covariance_m2=covariance_to_nested_list(projected_covariance),
        range_projection_mode=RANGE_BEARING_PROJECTED,
        range_observability=BEARING_PROJECTED_RANGE_OBSERVABILITY_CAP,
        range_interval_m=support.interval_m,
    )
