//! Node-plane mirror height resolution and SRP range observability (port of
//! `minimappr/core/onset_height_resolution.py` and
//! `minimappr/core/onset_range_observability.py`).
//!
//! Height: with coplanar node centroids a fix and its mirror give identical
//! cross-node TDOAs. Resolved by the ground/canopy prior (exactly one twin
//! admissible), else by the cluster's SRP (ratio), else reported ambiguous:
//! the fix sits on the plane and the vertical variance covers both twins.
//!
//! Range: a 20 m layout barely sees curvature from a bird 80 m out, yet LM's
//! residual-scaled covariance called such ranges observable. The cluster SRP
//! is sampled along its bearing ridge (max over a small cone per range); the
//! ranges within `support_ratio` of the peak form the supported interval. A
//! narrow interval keeps the fix (radial variance >= half-width²); a wide one
//! becomes `range_bearing_projected` at the profile peak.

use nalgebra::{Matrix3, Vector3};

use super::geometry_prior::NodePlaneGroundPrior;
use super::numeric::{arange, argmax, geomspace, linspace};
use super::refinement::BEARING_PROJECTED_RANGE_OBSERVABILITY;
use super::srp_grid::{distance, norm, subtract, Point3};
use super::uncertainty::{matrix_from_rows, range_observability_from_covariance, rows_from_matrix};
use crate::range_projection::{is_unobservable, RANGE_BEARING_PROJECTED};

pub const HEIGHT_NOT_APPLICABLE: &str = "not_applicable";
pub const HEIGHT_RESOLVED_BY_PRIOR: &str = "resolved_by_prior";
pub const HEIGHT_RESOLVED_BY_SRP: &str = "resolved_by_srp";
pub const HEIGHT_AMBIGUOUS: &str = "ambiguous";
pub const HEIGHT_OUTSIDE_PRIOR: &str = "outside_prior";
/// Twins closer than this (m) are the same point.
pub const MIRROR_COINCIDENT_DISTANCE_M: f64 = 1.0e-3;
const RANGE_EPSILON: f64 = 1.0e-9;
const CONE_ANGLE_EPSILON_DEG: f64 = 1.0e-9;

pub type Covariance = Option<[[f64; 3]; 3]>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeightResolution {
    pub position_m: Point3,
    pub added_vertical_variance_m2: f64,
    pub status: &'static str,
}

pub fn resolve_node_plane_mirror(
    position_m: &Point3,
    prior: &NodePlaneGroundPrior,
    score_points: &dyn Fn(&[Point3]) -> Vec<f64>,
    min_score_ratio: f64,
) -> HeightResolution {
    let unchanged = |status| HeightResolution {
        position_m: *position_m,
        added_vertical_variance_m2: 0.0,
        status,
    };
    if !prior.is_planar {
        return unchanged(HEIGHT_NOT_APPLICABLE);
    }
    let mirror = prior.mirror_across_node_plane(position_m);
    let separation_m = distance(&mirror, position_m);
    if separation_m < MIRROR_COINCIDENT_DISTANCE_M {
        return unchanged(HEIGHT_NOT_APPLICABLE);
    }
    let twins = [*position_m, mirror];
    if prior.enabled {
        let admissible = [
            prior.is_admissible(&twins[0]),
            prior.is_admissible(&twins[1]),
        ];
        if admissible[0] != admissible[1] {
            let chosen = if admissible[1] { twins[1] } else { twins[0] };
            return HeightResolution {
                position_m: chosen,
                added_vertical_variance_m2: 0.0,
                status: HEIGHT_RESOLVED_BY_PRIOR,
            };
        }
        if !admissible[0] && !admissible[1] {
            return unchanged(HEIGHT_OUTSIDE_PRIOR);
        }
    }
    let scores = score_points(&twins);
    let better = argmax(&scores);
    let worse_score = scores[1 - better];
    if worse_score <= 0.0 || scores[better] >= min_score_ratio * worse_score {
        return HeightResolution {
            position_m: twins[better],
            added_vertical_variance_m2: 0.0,
            status: HEIGHT_RESOLVED_BY_SRP,
        };
    }
    // Unresolvable: the node plane is the honest centre; +-half the separation is 1 sigma.
    let midpoint = [
        0.5 * (position_m[0] + mirror[0]),
        0.5 * (position_m[1] + mirror[1]),
        0.5 * (position_m[2] + mirror[2]),
    ];
    HeightResolution {
        position_m: midpoint,
        added_vertical_variance_m2: (0.5 * separation_m).powi(2),
        status: HEIGHT_AMBIGUOUS,
    }
}

pub fn add_vertical_variance(covariance: Covariance, variance_m2: f64) -> Covariance {
    let mut rows = covariance?;
    if variance_m2 > 0.0 {
        rows[2][2] += variance_m2;
    }
    Some(rows)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VerticalSupport {
    pub interval_z_m: (f64, f64),
    pub peak_z_m: f64,
}

/// SRP along the vertical through the fix between ground and canopy ceiling.
pub fn vertical_support_at(
    position_m: &Point3,
    z_bounds_m: (f64, f64),
    score_points: &dyn Fn(&[Point3]) -> Vec<f64>,
    sample_count: usize,
    support_ratio: f64,
) -> Option<VerticalSupport> {
    let (low_z, high_z) = z_bounds_m;
    if !(high_z > low_z) {
        return None;
    }
    let heights = linspace(low_z, high_z, sample_count);
    let column: Vec<Point3> = heights
        .iter()
        .map(|z| [position_m[0], position_m[1], *z])
        .collect();
    let scores = score_points(&column);
    if !scores.iter().any(|score| *score > 0.0) {
        return None;
    }
    let peak = argmax(&scores);
    let threshold = support_ratio * scores[peak];
    let first = scores.iter().position(|score| *score >= threshold)?;
    let last = scores.iter().rposition(|score| *score >= threshold)?;
    Some(VerticalSupport {
        interval_z_m: (heights[first], heights[last]),
        peak_z_m: heights[peak],
    })
}

/// Move z to the column peak if outside the band; vertical std >= half the band.
pub fn apply_vertical_support(
    position_m: Point3,
    covariance: Covariance,
    support: Option<VerticalSupport>,
) -> (Point3, Covariance) {
    let Some(support) = support else {
        return (position_m, covariance);
    };
    let mut position = position_m;
    let (low_z, high_z) = support.interval_z_m;
    if !(low_z <= position[2] && position[2] <= high_z) {
        position[2] = support.peak_z_m;
    }
    let covariance = covariance.map(|mut rows| {
        rows[2][2] = rows[2][2].max((0.5 * (high_z - low_z)).powi(2));
        rows
    });
    (position, covariance)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeProfileSettings {
    pub min_range_m: f64,
    pub max_range_m: f64,
    pub sample_count: usize,
    pub cone_half_angle_deg: f64,
    pub cone_step_deg: f64,
    pub support_ratio: f64,
    pub max_interval_ratio: f64,
    pub interval_floor_m: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeSupport {
    pub interval_m: (f64, f64),
    pub peak_range_m: f64,
    pub peak_position_m: Point3,
    pub observable: bool,
}

fn cone_directions(bearing: &Point3, half_angle_deg: f64, step_deg: f64) -> Vec<Point3> {
    let azimuth = bearing[1].atan2(bearing[0]);
    let elevation = bearing[2].clamp(-1.0, 1.0).asin();
    let offsets: Vec<f64> = if half_angle_deg > 0.0 {
        arange(
            -half_angle_deg,
            half_angle_deg + CONE_ANGLE_EPSILON_DEG,
            step_deg,
        )
        .iter()
        .map(|d| d.to_radians())
        .collect()
    } else {
        vec![0.0]
    };
    let mut directions = Vec::with_capacity(offsets.len() * offsets.len());
    for elevation_offset in &offsets {
        let cone_elevation = (elevation + elevation_offset)
            .clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
        for azimuth_offset in &offsets {
            let cone_azimuth = azimuth + azimuth_offset;
            directions.push([
                cone_elevation.cos() * cone_azimuth.cos(),
                cone_elevation.cos() * cone_azimuth.sin(),
                cone_elevation.sin(),
            ]);
        }
    }
    directions
}

pub fn range_support_along_bearing(
    position_m: &Point3,
    origin_m: &Point3,
    score_points: &dyn Fn(&[Point3]) -> Vec<f64>,
    settings: &RangeProfileSettings,
    admissible: Option<&dyn Fn(&Point3) -> bool>,
) -> Option<RangeSupport> {
    let offset = subtract(position_m, origin_m);
    let range_m = norm(&offset);
    if !range_m.is_finite() || range_m <= RANGE_EPSILON {
        return None;
    }
    let bearing = [
        offset[0] / range_m,
        offset[1] / range_m,
        offset[2] / range_m,
    ];
    let ranges = geomspace(
        settings.min_range_m,
        settings.max_range_m,
        settings.sample_count,
    );
    let directions = cone_directions(
        &bearing,
        settings.cone_half_angle_deg,
        settings.cone_step_deg,
    );
    let mut points = Vec::with_capacity(ranges.len() * directions.len());
    for range in &ranges {
        for direction in &directions {
            points.push([
                origin_m[0] + range * direction[0],
                origin_m[1] + range * direction[1],
                origin_m[2] + range * direction[2],
            ]);
        }
    }
    let mut scores = score_points(&points);
    if let Some(admissible) = admissible {
        // Below-ground / above-canopy candidates cannot be the source.
        for (score, point) in scores.iter_mut().zip(&points) {
            if !admissible(point) {
                *score = f64::NEG_INFINITY;
            }
        }
    }
    let width = directions.len();
    let best_direction: Vec<usize> = (0..ranges.len())
        .map(|row| argmax(&scores[row * width..(row + 1) * width]))
        .collect();
    let profile: Vec<f64> = (0..ranges.len())
        .map(|row| scores[row * width + best_direction[row]])
        .collect();
    let finite_max = profile
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !profile.iter().any(|v| v.is_finite()) || finite_max <= 0.0 {
        return None;
    }
    let peak = argmax(&profile);
    let threshold = settings.support_ratio * profile[peak];
    let first = profile.iter().position(|v| *v >= threshold)?;
    let last = profile.iter().rposition(|v| *v >= threshold)?;
    let (near_m, far_m) = (ranges[first], ranges[last]);
    Some(RangeSupport {
        interval_m: (near_m, far_m),
        peak_range_m: ranges[peak],
        peak_position_m: points[peak * width + best_direction[peak]],
        observable: far_m <= settings.max_interval_ratio * near_m
            || far_m - near_m <= settings.interval_floor_m,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct RangeJudgedPosition {
    pub position_m: Point3,
    pub covariance_m2: Covariance,
    pub range_projection_mode: Option<String>,
    pub range_observability: Option<f64>,
    pub range_interval_m: Option<(f64, f64)>,
}

fn radial_inflated(
    covariance: &Matrix3<f64>,
    radial_axis: &Vector3<f64>,
    radial_std_m: f64,
) -> Matrix3<f64> {
    let radial_variance = (radial_axis.transpose() * covariance * radial_axis)[(0, 0)];
    let deficit = radial_std_m * radial_std_m - radial_variance;
    if deficit <= 0.0 {
        return *covariance;
    }
    covariance + radial_axis * radial_axis.transpose() * deficit
}

/// Apply the profile verdict; never upgrades an already-unobservable range.
pub fn judge_range_from_profile(
    position_m: &Point3,
    covariance: Covariance,
    range_projection_mode: Option<&str>,
    range_observability: Option<f64>,
    origin_m: &Point3,
    support: Option<RangeSupport>,
) -> RangeJudgedPosition {
    let unchanged = RangeJudgedPosition {
        position_m: *position_m,
        covariance_m2: covariance,
        range_projection_mode: range_projection_mode.map(str::to_string),
        range_observability,
        range_interval_m: None,
    };
    let (Some(support), Some(rows)) = (support, covariance) else {
        return unchanged;
    };
    if is_unobservable(range_projection_mode) {
        return unchanged;
    }
    let (near_m, far_m) = support.interval_m;
    let radial_std_m = 0.5 * (far_m - near_m);
    let matrix = matrix_from_rows(&rows);
    let origin = Vector3::new(origin_m[0], origin_m[1], origin_m[2]);
    let current_range_m = distance(position_m, origin_m);
    if support.observable {
        let mut position = *position_m;
        if !(near_m <= current_range_m && current_range_m <= far_m) {
            position = support.peak_position_m;
        }
        let offset = Vector3::new(position[0], position[1], position[2]) - origin;
        let radial_axis = offset / offset.norm().max(RANGE_EPSILON);
        let inflated = radial_inflated(&matrix, &radial_axis, radial_std_m);
        return RangeJudgedPosition {
            position_m: position,
            covariance_m2: Some(rows_from_matrix(&inflated)),
            range_projection_mode: range_projection_mode.map(str::to_string),
            range_observability: range_observability_from_covariance(&inflated, Some(offset)),
            range_interval_m: Some(support.interval_m),
        };
    }
    // The profile peak is the SRP's own best range on the ridge.
    let projected = support.peak_position_m;
    let projected_offset = Vector3::new(projected[0], projected[1], projected[2]) - origin;
    let report_range_m = projected_offset.norm();
    let radial_axis = projected_offset / report_range_m.max(RANGE_EPSILON);
    // Lateral spread scales with range along the cone; the radial axis is the interval.
    let scale = report_range_m / current_range_m.max(RANGE_EPSILON);
    let lateral_projector = Matrix3::identity() - radial_axis * radial_axis.transpose();
    let lateral = lateral_projector * matrix * lateral_projector * (scale * scale);
    let projected_covariance =
        lateral + radial_axis * radial_axis.transpose() * (radial_std_m * radial_std_m);
    RangeJudgedPosition {
        position_m: projected,
        covariance_m2: Some(rows_from_matrix(&projected_covariance)),
        range_projection_mode: Some(RANGE_BEARING_PROJECTED.to_string()),
        range_observability: Some(BEARING_PROJECTED_RANGE_OBSERVABILITY),
        range_interval_m: Some(support.interval_m),
    }
}
