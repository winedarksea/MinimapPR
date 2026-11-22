"""Candidate grids and per-frame SRP evaluation for onset-frame localization.

Two grid kinds share one evaluator:

* direction grid (unit vectors) for a single compact array, where only the
  bearing is observable; predicted lag for pair (a, b) is -(p_b - p_a).u / c;
* spatial polar grid (log-spaced rings x azimuth x elevation around the layout
  centroid, plus a far-field ring) for multi-node windows, where predicted lag
  is (|x - p_b| - |x - p_a|) / c.

SRP sums every pair's full correlation curve at each candidate's predicted lag
(never per-pair argmax), so a reflective pair whose direct path is its second
peak still contributes the right evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minimappr.core.onset_frame_correlation import PairCorrelationCurve

# The far-field ring sits this many times beyond the farthest finite ring.
FAR_FIELD_RING_MULTIPLIER = 10.0
# Elevation span (deg) of the spatial grid: birds sit from below a hillside
# node up into the canopy; straight down/up is not useful at range.
SPATIAL_GRID_ELEVATION_MIN_DEG = -30.0
SPATIAL_GRID_ELEVATION_MAX_DEG = 60.0
EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class CandidateGrid:
    points_m: np.ndarray  # (P, 3) candidate positions (direction grid: centroid + unit * far range)
    directions: np.ndarray  # (P, 3) unit vectors from the centroid
    ranges_m: np.ndarray  # (P,) distance from the centroid (inf-like for direction grid)
    centroid_m: np.ndarray
    is_direction_grid: bool


def _unit_directions(azimuth_step_deg: float, elevations_deg: np.ndarray) -> np.ndarray:
    azimuths = np.radians(np.arange(0.0, 360.0, azimuth_step_deg))
    rows = []
    for elevation in np.radians(elevations_deg):
        if abs(abs(elevation) - np.pi / 2) < EPSILON:
            rows.append([0.0, 0.0, np.sign(elevation)])
            continue
        rows.extend(
            np.column_stack(
                [
                    np.cos(elevation) * np.cos(azimuths),
                    np.cos(elevation) * np.sin(azimuths),
                    np.full(azimuths.size, np.sin(elevation)),
                ]
            ).tolist()
        )
    return np.asarray(rows, dtype=np.float64)


def build_direction_grid(
    centroid_m: np.ndarray, *, azimuth_step_deg: float, elevation_step_deg: float, far_range_m: float
) -> CandidateGrid:
    elevations = np.arange(-90.0, 90.0 + EPSILON, elevation_step_deg)
    directions = _unit_directions(azimuth_step_deg, elevations)
    return CandidateGrid(
        points_m=centroid_m + directions * far_range_m,
        directions=directions,
        ranges_m=np.full(directions.shape[0], far_range_m),
        centroid_m=np.asarray(centroid_m, dtype=np.float64),
        is_direction_grid=True,
    )


def build_spatial_grid(
    centroid_m: np.ndarray,
    *,
    azimuth_step_deg: float,
    elevation_step_deg: float,
    min_range_m: float,
    max_range_m: float,
    ring_count: int,
) -> CandidateGrid:
    elevations = np.arange(
        SPATIAL_GRID_ELEVATION_MIN_DEG, SPATIAL_GRID_ELEVATION_MAX_DEG + EPSILON, elevation_step_deg
    )
    directions = _unit_directions(azimuth_step_deg, elevations)
    rings = np.concatenate(
        [np.geomspace(min_range_m, max_range_m, max(2, ring_count)), [max_range_m * FAR_FIELD_RING_MULTIPLIER]]
    )
    ranges = np.repeat(rings, directions.shape[0])
    tiled = np.tile(directions, (rings.size, 1))
    # The centroid itself (a bird inside a tight layout).
    points = np.vstack([centroid_m[None, :], centroid_m + tiled * ranges[:, None]])
    return CandidateGrid(
        points_m=points,
        directions=np.vstack([[0.0, 0.0, 1.0], tiled]),
        ranges_m=np.concatenate([[0.0], ranges]),
        centroid_m=np.asarray(centroid_m, dtype=np.float64),
        is_direction_grid=False,
    )


def augment_spatial_grid(grid: CandidateGrid, *, extra_points_m: np.ndarray) -> CandidateGrid:
    """Field Mode S-fixes: add node near-field shells to a spatial grid.

    The ground/canopy prior deliberately does not trim the grid: in a planar
    layout a frame whose best match is a bird's below-ground mirror twin would
    then peak somewhere else entirely and split the bird into a second cluster
    (measured on two_species_overlap). Twins are folded after LM instead
    (``onset_height_resolution``)."""
    centroid = grid.centroid_m
    extra = np.asarray(extra_points_m, dtype=np.float64).reshape(-1, 3)
    offsets = extra - centroid
    extra_ranges = np.linalg.norm(offsets, axis=1)
    extra_directions = offsets / np.maximum(extra_ranges, EPSILON)[:, None]
    points = np.vstack([grid.points_m, extra])
    directions = np.vstack([grid.directions, extra_directions])
    ranges = np.concatenate([grid.ranges_m, extra_ranges])
    return CandidateGrid(
        points_m=points,
        directions=directions,
        ranges_m=ranges,
        centroid_m=centroid,
        is_direction_grid=False,
    )


def predicted_lags_s(
    grid: CandidateGrid,
    sensor_positions: dict[str, np.ndarray],
    sensor_a: str,
    sensor_b: str,
    sound_speed_mps: float,
) -> np.ndarray:
    """Lag of sensor_b behind sensor_a at every candidate."""
    position_a = np.asarray(sensor_positions[sensor_a], dtype=np.float64)
    position_b = np.asarray(sensor_positions[sensor_b], dtype=np.float64)
    if grid.is_direction_grid:
        return -(grid.directions @ (position_b - position_a)) / sound_speed_mps
    distance_a = np.linalg.norm(grid.points_m - position_a, axis=1)
    distance_b = np.linalg.norm(grid.points_m - position_b, axis=1)
    return (distance_b - distance_a) / sound_speed_mps


class LagCache:
    """Predicted lags per (sensor_a, sensor_b), shared by every frame of a window."""

    def __init__(self, grid: CandidateGrid, sensor_positions: dict[str, np.ndarray], sound_speed_mps: float):
        self._grid = grid
        self._positions = sensor_positions
        self._sound_speed_mps = sound_speed_mps
        self._lags: dict[tuple[str, str], np.ndarray] = {}

    def lags(self, sensor_a: str, sensor_b: str) -> np.ndarray:
        key = (sensor_a, sensor_b)
        if key not in self._lags:
            self._lags[key] = predicted_lags_s(
                self._grid, self._positions, sensor_a, sensor_b, self._sound_speed_mps
            )
        return self._lags[key]


def steered_response_power(curves: list[PairCorrelationCurve], lag_cache: LagCache) -> np.ndarray:
    """Sum of pair curves (each weighted by its soft-SNR weight share) at every candidate."""
    total_weight = sum(curve.weight_sum for curve in curves)
    if not curves or total_weight <= 0.0:
        return np.zeros(0)
    power = None
    for curve in curves:
        contribution = curve.value_at(lag_cache.lags(curve.sensor_a, curve.sensor_b)) * (
            curve.weight_sum / total_weight
        )
        power = contribution if power is None else power + contribution
    return power


@dataclass(frozen=True, slots=True)
class GridPeak:
    point_index: int
    value: float
    contrast: float  # (peak - median) / peak in [0, 1]: sharpness of the frame's map


def angular_separation_deg(direction_a: np.ndarray, direction_b: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(direction_a, direction_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def spatial_cluster_radius_m(range_m: float, radius_m: float, range_fraction: float) -> float:
    return radius_m + range_fraction * max(range_m, 0.0)


def pick_frame_peaks(
    power: np.ndarray,
    grid: CandidateGrid,
    *,
    max_peaks: int,
    min_relative_height: float,
    suppression_radius_m: float,
    suppression_range_fraction: float,
    suppression_angle_deg: float,
) -> list[GridPeak]:
    """Greedy non-maximum suppression; peaks below min_relative_height x best stop it."""
    if power.size == 0:
        return []
    remaining = power.astype(np.float64).copy()
    median = float(np.median(power))
    peaks: list[GridPeak] = []
    best_value = None
    for _ in range(max(1, max_peaks)):
        index = int(np.argmax(remaining))
        value = float(remaining[index])
        if not np.isfinite(value) or value <= 0.0:
            break
        if best_value is None:
            best_value = value
        elif value < min_relative_height * best_value:
            break
        peaks.append(GridPeak(point_index=index, value=value, contrast=max(0.0, (value - median) / value)))
        if grid.is_direction_grid:
            cosines = grid.directions @ grid.directions[index]
            suppressed = cosines >= np.cos(np.radians(suppression_angle_deg))
        else:
            radius = spatial_cluster_radius_m(
                float(grid.ranges_m[index]), suppression_radius_m, suppression_range_fraction
            )
            distances = np.linalg.norm(grid.points_m - grid.points_m[index], axis=1)
            angle_close = (grid.directions @ grid.directions[index]) >= np.cos(np.radians(suppression_angle_deg))
            same_far_ring = angle_close & (grid.ranges_m >= grid.ranges_m.max() - EPSILON) & (
                grid.ranges_m[index] >= grid.ranges_m.max() - EPSILON
            )
            suppressed = (distances <= radius) | same_far_ring
        remaining[suppressed] = -np.inf
    return peaks
