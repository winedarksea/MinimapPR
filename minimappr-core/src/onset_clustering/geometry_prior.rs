//! Ground / canopy source prior and node-plane geometry (port of
//! `minimappr/core/onset_srp_geometry_prior.py`).
//!
//! With (nearly) coplanar node centroids, cross-node TDOAs are symmetric about
//! the node plane, so a source and its mirror fit equally well; the ground
//! reflection sits on the mirror side too. The prior: ground lies
//! `node_height_above_ground_m` below the plane fitted through the node
//! centroids (a hillside layout tilts it) and a source sits between the ground
//! and `source_max_height_above_ground_m` above it. It decides between mirror
//! twins after LM, masks the range profile, and bounds LM in non-planar layouts.

use nalgebra::DMatrix;

use super::numeric::{arange, geomspace};
use super::srp_grid::{dot, Point3};
use crate::tdoa_solver::robust_linear_algebra::sorted_singular_value_decomposition;

/// Below this singular-value ratio the centroids are collinear and define no
/// plane; the node plane then falls back to horizontal.
pub const COLLINEAR_SINGULAR_VALUE_RATIO: f64 = 1.0e-3;
/// Nodes needed before a fitted plane can show out-of-plane spread.
pub const MIN_NODES_FOR_NON_PLANAR_LAYOUT: usize = 4;
const SINGULAR_VALUE_FLOOR: f64 = 1e-12;
const SHELL_ANGLE_EPSILON_DEG: f64 = 1e-9;
const FULL_CIRCLE_DEG: f64 = 360.0;
const QUARTER_CIRCLE_DEG: f64 = 90.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodePlaneGroundPrior {
    pub plane_point_m: Point3,
    /// Unit normal pointing up.
    pub plane_normal: Point3,
    pub node_height_above_ground_m: f64,
    pub source_max_height_above_ground_m: f64,
    pub is_planar: bool,
    pub enabled: bool,
}

impl NodePlaneGroundPrior {
    fn signed_plane_distance(&self, point: &Point3) -> f64 {
        let offset = [
            point[0] - self.plane_point_m[0],
            point[1] - self.plane_point_m[1],
            point[2] - self.plane_point_m[2],
        ];
        dot(&offset, &self.plane_normal)
    }

    pub fn height_above_ground_m(&self, point: &Point3) -> f64 {
        self.signed_plane_distance(point) + self.node_height_above_ground_m
    }

    /// True inside the ground..canopy slab (always True when disabled).
    pub fn is_admissible(&self, point: &Point3) -> bool {
        if !self.enabled {
            return true;
        }
        let height = self.height_above_ground_m(point);
        (0.0..=self.source_max_height_above_ground_m).contains(&height)
    }

    pub fn mirror_across_node_plane(&self, point: &Point3) -> Point3 {
        let distance = self.signed_plane_distance(point);
        let n = self.plane_normal;
        [
            point[0] - 2.0 * distance * n[0],
            point[1] - 2.0 * distance * n[1],
            point[2] - 2.0 * distance * n[2],
        ]
    }

    pub fn fold_above_plane(&self, point: &Point3) -> Point3 {
        if self.signed_plane_distance(point) < 0.0 {
            self.mirror_across_node_plane(point)
        } else {
            *point
        }
    }

    pub fn fold_direction_above_plane(&self, direction: &Point3) -> Point3 {
        let along_normal = dot(direction, &self.plane_normal);
        if along_normal >= 0.0 {
            return *direction;
        }
        let n = self.plane_normal;
        [
            direction[0] - 2.0 * along_normal * n[0],
            direction[1] - 2.0 * along_normal * n[1],
            direction[2] - 2.0 * along_normal * n[2],
        ]
    }

    /// `(z_min, z_max)` of the slab under the point's horizontal position (LM bounds).
    pub fn vertical_bounds_m(&self, point: &Point3) -> Option<(f64, f64)> {
        if !self.enabled || self.plane_normal[2].abs() <= COLLINEAR_SINGULAR_VALUE_RATIO {
            return None;
        }
        let horizontal = [
            point[0] - self.plane_point_m[0],
            point[1] - self.plane_point_m[1],
        ];
        let plane_z = self.plane_point_m[2]
            - (horizontal[0] * self.plane_normal[0] + horizontal[1] * self.plane_normal[1])
                / self.plane_normal[2];
        let vertical_per_normal_m = 1.0 / self.plane_normal[2];
        let ground_z = plane_z - self.node_height_above_ground_m * vertical_per_normal_m;
        Some((
            ground_z,
            ground_z + self.source_max_height_above_ground_m * vertical_per_normal_m,
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundPriorSettings {
    pub enabled: bool,
    pub node_height_above_ground_m: f64,
    pub source_max_height_above_ground_m: f64,
    pub planar_layout_max_flatness_ratio: f64,
}

pub fn build_node_plane_ground_prior(
    node_centroids_m: &[Point3],
    settings: &GroundPriorSettings,
) -> NodePlaneGroundPrior {
    let count = node_centroids_m.len();
    let mut plane_point = [0.0; 3];
    for centroid in node_centroids_m {
        for axis in 0..3 {
            plane_point[axis] += centroid[axis];
        }
    }
    for value in plane_point.iter_mut() {
        *value /= count as f64;
    }
    let mut normal = [0.0, 0.0, 1.0];
    let mut is_planar = true;
    if count >= 3 {
        let centred = DMatrix::from_fn(count, 3, |row, column| {
            node_centroids_m[row][column] - plane_point[column]
        });
        if let Ok(decomposition) = sorted_singular_value_decomposition(&centred) {
            let singular = &decomposition.singular_values;
            let largest = singular[0].max(SINGULAR_VALUE_FLOOR);
            if singular.len() > 2 && singular[1] > COLLINEAR_SINGULAR_VALUE_RATIO * largest {
                let row = decomposition.right_vectors_transposed.row(2);
                let candidate = [row[0], row[1], row[2]];
                normal = if candidate[2] >= 0.0 {
                    candidate
                } else {
                    [-candidate[0], -candidate[1], -candidate[2]]
                };
            }
            let flatness = if singular.len() > 2 {
                singular[2] / largest
            } else {
                0.0
            };
            is_planar = count < MIN_NODES_FOR_NON_PLANAR_LAYOUT
                || flatness < settings.planar_layout_max_flatness_ratio;
        }
    }
    let length = dot(&normal, &normal).sqrt();
    NodePlaneGroundPrior {
        plane_point_m: plane_point,
        plane_normal: [normal[0] / length, normal[1] / length, normal[2] / length],
        node_height_above_ground_m: settings.node_height_above_ground_m,
        source_max_height_above_ground_m: settings.source_max_height_above_ground_m,
        is_planar,
        enabled: settings.enabled,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeShellSettings {
    pub min_range_m: f64,
    pub max_range_m: f64,
    pub ring_count: usize,
    pub azimuth_step_deg: f64,
    pub elevation_step_deg: f64,
}

/// Polar shells around every node: the layout-centred grid samples a source
/// 2 m from one node at ~10 deg steps as seen from that node, so the node's
/// own (range-blind) compact pairs never line up there and the far end of its
/// bearing ray wins the frame. Order: node, ring, elevation, azimuth.
pub fn node_near_field_points(
    node_centres_m: &[Point3],
    settings: &NodeShellSettings,
) -> Vec<Point3> {
    if settings.ring_count < 1 || node_centres_m.is_empty() {
        return Vec::new();
    }
    let azimuths: Vec<f64> = arange(0.0, FULL_CIRCLE_DEG, settings.azimuth_step_deg)
        .iter()
        .map(|d| d.to_radians())
        .collect();
    let elevations: Vec<f64> = arange(
        -QUARTER_CIRCLE_DEG,
        QUARTER_CIRCLE_DEG + SHELL_ANGLE_EPSILON_DEG,
        settings.elevation_step_deg,
    )
    .iter()
    .map(|d| d.to_radians())
    .collect();
    let mut directions = Vec::with_capacity(azimuths.len() * elevations.len());
    for elevation in &elevations {
        for azimuth in &azimuths {
            directions.push([
                elevation.cos() * azimuth.cos(),
                elevation.cos() * azimuth.sin(),
                elevation.sin(),
            ]);
        }
    }
    let rings = if settings.ring_count > 1 {
        geomspace(
            settings.min_range_m,
            settings.max_range_m,
            settings.ring_count,
        )
    } else {
        vec![settings.min_range_m]
    };
    let mut points = Vec::with_capacity(node_centres_m.len() * rings.len() * directions.len());
    for centre in node_centres_m {
        for ring in &rings {
            for direction in &directions {
                points.push([
                    centre[0] + ring * direction[0],
                    centre[1] + ring * direction[1],
                    centre[2] + ring * direction[2],
                ]);
            }
        }
    }
    points
}
