//! Candidate grids and per-frame SRP for onset-frame localization (port of
//! `minimappr/core/onset_srp_grid.py`).
//!
//! A direction grid (single compact array: only bearing is observable) and a
//! polar spatial grid (log rings x azimuth x elevation around the layout
//! centroid, a far-field ring, and per-node near-field shells) share one
//! evaluator. SRP sums every pair's full correlation curve at each candidate's
//! predicted lag (never per-pair argmax), so a reflective pair whose direct
//! path is its second peak still contributes the right evidence.
//!
//! Why not `srp_layout::polar_grid`: that grid is node-anchored with a
//! different ring/elevation layout; the Python candidate indices and NMS order
//! must match exactly for decision-level parity.

use std::collections::BTreeMap;

use super::frame_correlation::PairCorrelationCurve;
use super::numeric::{arange, geomspace, median};

pub type Point3 = [f64; 3];

/// The far-field ring sits this many times beyond the farthest finite ring.
pub const FAR_FIELD_RING_MULTIPLIER: f64 = 10.0;
/// Elevation span (deg) of the spatial grid.
pub const SPATIAL_GRID_ELEVATION_MIN_DEG: f64 = -30.0;
pub const SPATIAL_GRID_ELEVATION_MAX_DEG: f64 = 60.0;
pub const EPSILON: f64 = 1e-12;
const FULL_CIRCLE_DEG: f64 = 360.0;
const QUARTER_CIRCLE_DEG: f64 = 90.0;

pub fn subtract(a: &Point3, b: &Point3) -> Point3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn dot(a: &Point3, b: &Point3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn norm(a: &Point3) -> f64 {
    dot(a, a).sqrt()
}

pub fn distance(a: &Point3, b: &Point3) -> f64 {
    norm(&subtract(a, b))
}

pub fn add_scaled(origin: &Point3, direction: &Point3, scale: f64) -> Point3 {
    [
        origin[0] + direction[0] * scale,
        origin[1] + direction[1] * scale,
        origin[2] + direction[2] * scale,
    ]
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateGrid {
    pub points_m: Vec<Point3>,
    /// Unit vectors from the centroid.
    pub directions: Vec<Point3>,
    /// Distance from the centroid (the far range for a direction grid).
    pub ranges_m: Vec<f64>,
    pub centroid_m: Point3,
    pub is_direction_grid: bool,
    /// `max(ranges_m)`, cached (the far-field ring test).
    pub max_range_m: f64,
}

impl CandidateGrid {
    fn new(
        points_m: Vec<Point3>,
        directions: Vec<Point3>,
        ranges_m: Vec<f64>,
        centroid_m: Point3,
        is_direction_grid: bool,
    ) -> Self {
        let max_range_m = ranges_m.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self {
            points_m,
            directions,
            ranges_m,
            centroid_m,
            is_direction_grid,
            max_range_m,
        }
    }
}

fn unit_directions(azimuth_step_deg: f64, elevations_deg: &[f64]) -> Vec<Point3> {
    let azimuths: Vec<f64> = arange(0.0, FULL_CIRCLE_DEG, azimuth_step_deg)
        .iter()
        .map(|deg| deg.to_radians())
        .collect();
    let mut rows = Vec::new();
    for elevation in elevations_deg.iter().map(|deg| deg.to_radians()) {
        if (elevation.abs() - std::f64::consts::FRAC_PI_2).abs() < EPSILON {
            rows.push([0.0, 0.0, elevation.signum()]);
            continue;
        }
        for azimuth in &azimuths {
            rows.push([
                elevation.cos() * azimuth.cos(),
                elevation.cos() * azimuth.sin(),
                elevation.sin(),
            ]);
        }
    }
    rows
}

pub fn build_direction_grid(
    centroid_m: Point3,
    azimuth_step_deg: f64,
    elevation_step_deg: f64,
    far_range_m: f64,
) -> CandidateGrid {
    let elevations = arange(
        -QUARTER_CIRCLE_DEG,
        QUARTER_CIRCLE_DEG + EPSILON,
        elevation_step_deg,
    );
    let directions = unit_directions(azimuth_step_deg, &elevations);
    let points = directions
        .iter()
        .map(|direction| add_scaled(&centroid_m, direction, far_range_m))
        .collect();
    let ranges = vec![far_range_m; directions.len()];
    CandidateGrid::new(points, directions, ranges, centroid_m, true)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialGridSettings {
    pub azimuth_step_deg: f64,
    pub elevation_step_deg: f64,
    pub min_range_m: f64,
    pub max_range_m: f64,
    pub ring_count: usize,
}

pub fn build_spatial_grid(centroid_m: Point3, settings: &SpatialGridSettings) -> CandidateGrid {
    let elevations = arange(
        SPATIAL_GRID_ELEVATION_MIN_DEG,
        SPATIAL_GRID_ELEVATION_MAX_DEG + EPSILON,
        settings.elevation_step_deg,
    );
    let directions = unit_directions(settings.azimuth_step_deg, &elevations);
    let mut rings = geomspace(
        settings.min_range_m,
        settings.max_range_m,
        settings.ring_count.max(2),
    );
    rings.push(settings.max_range_m * FAR_FIELD_RING_MULTIPLIER);
    // The centroid itself (a bird inside a tight layout).
    let mut points = vec![centroid_m];
    let mut grid_directions = vec![[0.0, 0.0, 1.0]];
    let mut ranges = vec![0.0];
    for ring in &rings {
        for direction in &directions {
            points.push(add_scaled(&centroid_m, direction, *ring));
            grid_directions.push(*direction);
            ranges.push(*ring);
        }
    }
    CandidateGrid::new(points, grid_directions, ranges, centroid_m, false)
}

/// Add node near-field shells to a spatial grid (S-fixes). The ground prior
/// deliberately does not trim the grid: twins are folded after LM instead.
pub fn augment_spatial_grid(grid: CandidateGrid, extra_points_m: &[Point3]) -> CandidateGrid {
    let centroid = grid.centroid_m;
    let CandidateGrid {
        mut points_m,
        mut directions,
        mut ranges_m,
        ..
    } = grid;
    for point in extra_points_m {
        let offset = subtract(point, &centroid);
        let range = norm(&offset);
        let scale = range.max(EPSILON);
        points_m.push(*point);
        directions.push([offset[0] / scale, offset[1] / scale, offset[2] / scale]);
        ranges_m.push(range);
    }
    CandidateGrid::new(points_m, directions, ranges_m, centroid, false)
}

/// Predicted lags per sensor pair, shared by every frame of a window.
pub struct LagCache<'grid> {
    grid: &'grid CandidateGrid,
    positions: &'grid BTreeMap<String, Point3>,
    sound_speed_mps: f64,
    lags: BTreeMap<(String, String), Vec<f64>>,
}

impl<'grid> LagCache<'grid> {
    pub fn new(
        grid: &'grid CandidateGrid,
        positions: &'grid BTreeMap<String, Point3>,
        sound_speed_mps: f64,
    ) -> Self {
        Self {
            grid,
            positions,
            sound_speed_mps,
            lags: BTreeMap::new(),
        }
    }

    /// Lag of `sensor_b` behind `sensor_a` at every candidate.
    pub fn lags(&mut self, sensor_a: &str, sensor_b: &str) -> &[f64] {
        let key = (sensor_a.to_string(), sensor_b.to_string());
        if !self.lags.contains_key(&key) {
            let position_a = self.positions[sensor_a];
            let position_b = self.positions[sensor_b];
            let speed = self.sound_speed_mps;
            let values = if self.grid.is_direction_grid {
                let baseline = subtract(&position_b, &position_a);
                self.grid
                    .directions
                    .iter()
                    .map(|direction| -dot(direction, &baseline) / speed)
                    .collect()
            } else {
                self.grid
                    .points_m
                    .iter()
                    .map(|point| {
                        (distance(point, &position_b) - distance(point, &position_a)) / speed
                    })
                    .collect()
            };
            self.lags.insert(key.clone(), values);
        }
        &self.lags[&key]
    }
}

/// Sum of pair curves (each weighted by its soft-SNR weight share) at every candidate.
pub fn steered_response_power(
    curves: &[PairCorrelationCurve],
    lag_cache: &mut LagCache<'_>,
) -> Vec<f64> {
    let total_weight: f64 = curves.iter().map(|curve| curve.weight_sum).sum();
    if curves.is_empty() || total_weight <= 0.0 {
        return Vec::new();
    }
    let mut power: Option<Vec<f64>> = None;
    for curve in curves {
        let share = curve.weight_sum / total_weight;
        let lags = lag_cache.lags(&curve.sensor_a, &curve.sensor_b);
        match power.as_mut() {
            None => {
                power = Some(
                    lags.iter()
                        .map(|lag| curve.value_at(*lag) * share)
                        .collect(),
                )
            }
            Some(sum) => {
                for (value, lag) in sum.iter_mut().zip(lags) {
                    *value += curve.value_at(*lag) * share;
                }
            }
        }
    }
    power.unwrap_or_default()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridPeak {
    pub point_index: usize,
    pub value: f64,
    /// `(peak - median) / peak` in [0, 1]: sharpness of the frame's map.
    pub contrast: f64,
}

pub fn angular_separation_deg(direction_a: &Point3, direction_b: &Point3) -> f64 {
    dot(direction_a, direction_b)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

pub fn spatial_cluster_radius_m(range_m: f64, radius_m: f64, range_fraction: f64) -> f64 {
    radius_m + range_fraction * range_m.max(0.0)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeakPickingSettings {
    pub max_peaks: usize,
    pub min_relative_height: f64,
    pub suppression_radius_m: f64,
    pub suppression_range_fraction: f64,
    pub suppression_angle_deg: f64,
}

/// Greedy non-maximum suppression; peaks below `min_relative_height` x best stop it.
pub fn pick_frame_peaks(
    power: &[f64],
    grid: &CandidateGrid,
    settings: &PeakPickingSettings,
) -> Vec<GridPeak> {
    if power.is_empty() {
        return Vec::new();
    }
    let mut remaining = power.to_vec();
    let power_median = median(power);
    let cosine_limit = settings.suppression_angle_deg.to_radians().cos();
    let mut peaks = Vec::new();
    let mut best_value: Option<f64> = None;
    for _ in 0..settings.max_peaks.max(1) {
        let index = super::numeric::argmax(&remaining);
        let value = remaining[index];
        if !value.is_finite() || value <= 0.0 {
            break;
        }
        match best_value {
            None => best_value = Some(value),
            Some(best) if value < settings.min_relative_height * best => break,
            Some(_) => {}
        }
        peaks.push(GridPeak {
            point_index: index,
            value,
            contrast: ((value - power_median) / value).max(0.0),
        });
        let peak_direction = grid.directions[index];
        if grid.is_direction_grid {
            for (slot, direction) in remaining.iter_mut().zip(&grid.directions) {
                if dot(direction, &peak_direction) >= cosine_limit {
                    *slot = f64::NEG_INFINITY;
                }
            }
        } else {
            let radius = spatial_cluster_radius_m(
                grid.ranges_m[index],
                settings.suppression_radius_m,
                settings.suppression_range_fraction,
            );
            let peak_point = grid.points_m[index];
            let peak_on_far_ring = grid.ranges_m[index] >= grid.max_range_m - EPSILON;
            for (candidate, slot) in remaining.iter_mut().enumerate() {
                let within_radius = distance(&grid.points_m[candidate], &peak_point) <= radius;
                let same_far_ring = peak_on_far_ring
                    && grid.ranges_m[candidate] >= grid.max_range_m - EPSILON
                    && dot(&grid.directions[candidate], &peak_direction) >= cosine_limit;
                if within_radius || same_far_ring {
                    *slot = f64::NEG_INFINITY;
                }
            }
        }
    }
    peaks
}
