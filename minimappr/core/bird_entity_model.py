"""Field Mode W3-entity: the bird-entity record and its spatial-state geometry.

One bird is one entity that moves through spatial states as evidence arrives:

    heard_nearby (no bearing/position) -> direction_known (bearing fan from a
    node) -> pinpointed (position + uncertainty)

States are ranked; an observation never lowers the state (one failed bout does
not demote a bird). Only maintenance steps an entity back (track expiry,
uncertainty growth, spatial staleness). Pure data + math: no IO, no clocks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

BirdEntitySpatialState = Literal["heard_nearby", "direction_known", "pinpointed"]
ENTITY_STATE_HEARD_NEARBY: BirdEntitySpatialState = "heard_nearby"
ENTITY_STATE_DIRECTION_KNOWN: BirdEntitySpatialState = "direction_known"
ENTITY_STATE_PINPOINTED: BirdEntitySpatialState = "pinpointed"
ENTITY_STATE_RANK: dict[str, int] = {
    ENTITY_STATE_HEARD_NEARBY: 0,
    ENTITY_STATE_DIRECTION_KNOWN: 1,
    ENTITY_STATE_PINPOINTED: 2,
}
ENTITY_STATE_ONE_STEP_BACK: dict[str, BirdEntitySpatialState] = {
    ENTITY_STATE_PINPOINTED: ENTITY_STATE_DIRECTION_KNOWN,
    ENTITY_STATE_DIRECTION_KNOWN: ENTITY_STATE_HEARD_NEARBY,
    ENTITY_STATE_HEARD_NEARBY: ENTITY_STATE_HEARD_NEARBY,
}
FULL_CIRCLE_DEG = 360.0
HALF_CIRCLE_DEG = 180.0

Vec3 = tuple[float, float, float]


@dataclass(slots=True)
class BirdEntityBearingEstimate:
    """A bearing fan from ``origin_node_id`` (compass degrees, clockwise from north)."""

    origin_node_id: str
    origin_position_m: Vec3
    bearing_deg: float
    half_width_deg: float
    # None when range is unobservable (bearing-only solve): draw a default-length fan.
    range_min_m: float | None = None
    range_max_m: float | None = None


@dataclass(slots=True)
class BirdEntityRecord:
    entity_id: str
    state: BirdEntitySpatialState
    label_category: str
    species_top_label: str
    species_top_probability: float
    species_belief: dict[str, float]
    first_heard_ns: int
    last_heard_ns: int
    reporting_node_id: str | None
    reporting_node_position_m: Vec3 | None
    reporting_node_ids: list[str] = field(default_factory=list)
    position_m: Vec3 | None = None
    position_covariance_m2: list[list[float]] | None = None
    horizontal_uncertainty_m: float | None = None
    bearing: BirdEntityBearingEstimate | None = None
    track_id: str | None = None
    canonical_detection_id: str | None = None
    last_spatial_update_ns: int | None = None
    # Restarts whenever the state changes so the hold timer measures time *in* a state.
    last_state_change_ns: int = 0
    detection_count: int = 0
    absorbed_detection_count: int = 0
    # Last "just sang" pulse (any absorbed or observed bout).
    last_pulse_ns: int = 0

    @property
    def is_spatial(self) -> bool:
        return self.state != ENTITY_STATE_HEARD_NEARBY

    def believes_label(self, label: str) -> bool:
        return label == self.species_top_label or label in self.species_belief

    def anchor_position_m(self) -> Vec3 | None:
        """Where the entity is for matching: its position, else its reporting node."""
        if self.position_m is not None:
            return self.position_m
        return self.reporting_node_position_m


def horizontal_major_sigma_m(covariance_m2: list[list[float]] | None) -> float | None:
    """1-sigma length of the horizontal (east/north) covariance major axis."""
    if covariance_m2 is None:
        return None
    try:
        covariance = np.asarray(covariance_m2, dtype=np.float64)
        if covariance.shape[0] < 2 or covariance.shape[1] < 2:
            return None
        horizontal = covariance[:2, :2]
        horizontal = 0.5 * (horizontal + horizontal.T)
        eigenvalues = np.linalg.eigvalsh(horizontal)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None
    if not np.all(np.isfinite(eigenvalues)):
        return None
    return math.sqrt(max(float(eigenvalues[-1]), 0.0))


def compass_bearing_deg(origin_m: Vec3, target_m: Vec3) -> float:
    """Clockwise-from-north bearing in the local ENU frame (x east, y north)."""
    east = float(target_m[0]) - float(origin_m[0])
    north = float(target_m[1]) - float(origin_m[1])
    return math.degrees(math.atan2(east, north)) % FULL_CIRCLE_DEG


def angular_difference_deg(first_deg: float, second_deg: float) -> float:
    difference = abs((first_deg - second_deg) % FULL_CIRCLE_DEG)
    return min(difference, FULL_CIRCLE_DEG - difference)


def horizontal_distance_m(first_m: Vec3, second_m: Vec3) -> float:
    return math.hypot(float(first_m[0]) - float(second_m[0]), float(first_m[1]) - float(second_m[1]))


def bearing_estimate_toward(
    *,
    origin_node_id: str,
    origin_position_m: Vec3,
    target_position_m: Vec3,
    covariance_m2: list[list[float]] | None,
    range_observable: bool,
    default_half_width_deg: float,
    min_half_width_deg: float,
    max_half_width_deg: float,
    sigma_multiplier: float,
) -> BirdEntityBearingEstimate:
    """Bearing fan from a node toward *target*; width from tangential spread.

    The half width is ``atan(k * sigma_tangential / range)``: the covariance
    projected perpendicular to the ray, so an elongated radial ellipse (good
    bearing, poor range — the common case outside a node layout) gives a narrow
    fan with a wide range band rather than a misleading blob.
    """
    bearing_deg = compass_bearing_deg(origin_position_m, target_position_m)
    range_m = horizontal_distance_m(origin_position_m, target_position_m)
    half_width_deg = default_half_width_deg
    range_min_m: float | None = None
    range_max_m: float | None = None
    radial_unit, tangential_unit = _ray_units(bearing_deg)
    horizontal = _horizontal_covariance(covariance_m2)
    if horizontal is not None and range_m > 0.0:
        tangential_sigma_m = math.sqrt(max(float(tangential_unit @ horizontal @ tangential_unit), 0.0))
        half_width_deg = math.degrees(math.atan2(sigma_multiplier * tangential_sigma_m, range_m))
        if range_observable:
            radial_sigma_m = math.sqrt(max(float(radial_unit @ horizontal @ radial_unit), 0.0))
            range_min_m = max(0.0, range_m - sigma_multiplier * radial_sigma_m)
            range_max_m = range_m + sigma_multiplier * radial_sigma_m
    elif range_observable:
        range_min_m = range_max_m = range_m
    half_width_deg = min(max(half_width_deg, min_half_width_deg), max_half_width_deg)
    return BirdEntityBearingEstimate(
        origin_node_id=origin_node_id,
        origin_position_m=tuple(float(v) for v in origin_position_m[:3]),  # type: ignore[arg-type]
        bearing_deg=bearing_deg,
        half_width_deg=half_width_deg,
        range_min_m=range_min_m,
        range_max_m=range_max_m,
    )


def uncertainty_ellipse(
    covariance_m2: list[list[float]] | None, sigma_multiplier: float
) -> tuple[float, float, float] | None:
    """(semi_major_m, semi_minor_m, major_axis_compass_deg) at *sigma_multiplier*."""
    horizontal = _horizontal_covariance(covariance_m2)
    if horizontal is None:
        return None
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(horizontal)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(eigenvalues)):
        return None
    major_axis = eigenvectors[:, 1]
    orientation_deg = math.degrees(math.atan2(float(major_axis[0]), float(major_axis[1]))) % HALF_CIRCLE_DEG
    return (
        sigma_multiplier * math.sqrt(max(float(eigenvalues[1]), 0.0)),
        sigma_multiplier * math.sqrt(max(float(eigenvalues[0]), 0.0)),
        orientation_deg,
    )


def _horizontal_covariance(covariance_m2: list[list[float]] | None) -> np.ndarray | None:
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


def _ray_units(bearing_deg: float) -> tuple[np.ndarray, np.ndarray]:
    bearing_rad = math.radians(bearing_deg)
    radial = np.array([math.sin(bearing_rad), math.cos(bearing_rad)])
    tangential = np.array([math.cos(bearing_rad), -math.sin(bearing_rad)])
    return radial, tangential
