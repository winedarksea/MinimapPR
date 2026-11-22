//! Seeded random layouts, source placement and rigid transforms.
//!
//! Every draw is a pure hash of (seed, purpose, index) through
//! `minimappr_mesh_sim::seeded_random`, so a trial never depends on the
//! order things are generated in, and transformed twins of a trial can
//! reuse exactly the same noise.

use minimappr_mesh_sim::seeded_random::{hashed_gaussian, hashed_uniform};

use crate::gate_config::TrialGeometryConfig;

pub fn uniform_in(seed: u64, purpose: &str, index: u64, range: [f64; 2]) -> f64 {
    range[0] + (range[1] - range[0]) * hashed_uniform(seed, purpose, index, 0)
}

pub fn gaussian(seed: u64, purpose: &str, index: u64, axis: u64) -> f64 {
    hashed_gaussian(seed, purpose, index, axis)
}

pub fn count_in(seed: u64, purpose: &str, range: [usize; 2]) -> usize {
    let span = (range[1] - range[0] + 1) as f64;
    (range[0] + (hashed_uniform(seed, purpose, 0, 0) * span) as usize).min(range[1])
}

pub fn horizontal_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

pub fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

pub fn horizontal_centroid(points: &[[f64; 3]]) -> [f64; 2] {
    let count = points.len().max(1) as f64;
    [points.iter().map(|p| p[0]).sum::<f64>() / count, points.iter().map(|p| p[1]).sum::<f64>() / count]
}

/// √(λmin/λmax) of the horizontal scatter (1 = round, 0 = collinear).
pub fn horizontal_aspect_ratio(points: &[[f64; 3]]) -> f64 {
    let [cx, cy] = horizontal_centroid(points);
    let (mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0);
    for p in points {
        let (dx, dy) = (p[0] - cx, p[1] - cy);
        xx += dx * dx;
        xy += dx * dy;
        yy += dy * dy;
    }
    let half_trace = 0.5 * (xx + yy);
    let spread = (0.25 * (xx - yy).powi(2) + xy * xy).sqrt();
    let (major, minor) = (half_trace + spread, (half_trace - spread).max(0.0));
    if major <= 0.0 { 0.0 } else { (minor / major).sqrt() }
}

/// Azimuth (radians, CCW from east) of `point` seen from `origin`.
pub fn azimuth_from(origin_xy_m: [f64; 2], point: [f64; 3]) -> f64 {
    (point[1] - origin_xy_m[1]).atan2(point[0] - origin_xy_m[0])
}

pub fn wrapped_angle_difference_rad(a: f64, b: f64) -> f64 {
    (a - b + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}

/// Irregular non-collinear phone layout centred near the origin.
pub fn random_phone_layout(seed: u64, node_count: usize, aperture_m: f64, config: &TrialGeometryConfig) -> Vec<[f64; 3]> {
    let radius = 0.5 * aperture_m;
    let minimum_separation = config.minimum_node_separation_fraction * aperture_m;
    let mut best: Vec<[f64; 3]> = Vec::new();
    for attempt in 0..config.layout_rejection_attempts {
        let mut nodes: Vec<[f64; 3]> = Vec::with_capacity(node_count);
        let mut draw = 0;
        while nodes.len() < node_count && draw < node_count as u64 * 20 {
            let key = attempt * 1_000 + draw;
            draw += 1;
            // Uniform over the disk (√u radius).
            let r = radius * hashed_uniform(seed, "layout_radius", key, 0).sqrt();
            let angle = std::f64::consts::TAU * hashed_uniform(seed, "layout_angle", key, 0);
            let height = uniform_in(seed, "layout_height", key, config.phone_height_range_m);
            let candidate = [r * angle.cos(), r * angle.sin(), height];
            if nodes.iter().all(|node| horizontal_distance(*node, candidate) >= minimum_separation) {
                nodes.push(candidate);
            }
        }
        if nodes.len() == node_count && horizontal_aspect_ratio(&nodes) >= config.minimum_layout_aspect_ratio {
            return nodes;
        }
        if nodes.len() > best.len() {
            best = nodes;
        }
    }
    best
}

/// Source inside the layout: uniform over a disk around the centroid.
pub fn inside_source(seed: u64, nodes: &[[f64; 3]], aperture_m: f64, config: &TrialGeometryConfig) -> [f64; 3] {
    let [cx, cy] = horizontal_centroid(nodes);
    let r = 0.5 * aperture_m * config.inside_source_radius_fraction * hashed_uniform(seed, "source_radius", 0, 0).sqrt();
    let angle = std::f64::consts::TAU * hashed_uniform(seed, "source_angle", 0, 0);
    [cx + r * angle.cos(), cy + r * angle.sin(), uniform_in(seed, "source_height", 0, config.source_height_range_m)]
}

/// Source at a horizontal range from the centroid, random azimuth.
pub fn source_at_range(seed: u64, nodes: &[[f64; 3]], range_m: f64, config: &TrialGeometryConfig) -> [f64; 3] {
    let [cx, cy] = horizontal_centroid(nodes);
    let angle = std::f64::consts::TAU * hashed_uniform(seed, "source_angle", range_m.to_bits(), 0);
    [cx + range_m * angle.cos(), cy + range_m * angle.sin(), uniform_in(seed, "source_height", 1, config.source_height_range_m)]
}

/// Proper or improper rigid motion of the horizontal plane (z untouched).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanarRigidMotion {
    pub rotation_rad: f64,
    pub translation_m: [f64; 2],
    /// Reflect y before rotating.
    pub mirrored: bool,
}

impl PlanarRigidMotion {
    pub fn random(seed: u64, maximum_translation_m: f64, mirrored: bool) -> Self {
        Self {
            rotation_rad: std::f64::consts::TAU * hashed_uniform(seed, "motion_rotation", 0, 0),
            translation_m: [
                maximum_translation_m * (2.0 * hashed_uniform(seed, "motion_east", 0, 0) - 1.0),
                maximum_translation_m * (2.0 * hashed_uniform(seed, "motion_north", 0, 0) - 1.0),
            ],
            mirrored,
        }
    }

    /// Linear part only (for error vectors and covariances).
    pub fn rotate_vector(&self, vector: [f64; 2]) -> [f64; 2] {
        let y = if self.mirrored { -vector[1] } else { vector[1] };
        let (sin, cos) = self.rotation_rad.sin_cos();
        [cos * vector[0] - sin * y, sin * vector[0] + cos * y]
    }

    pub fn apply(&self, point: [f64; 3]) -> [f64; 3] {
        let [x, y] = self.rotate_vector([point[0], point[1]]);
        [x + self.translation_m[0], y + self.translation_m[1], point[2]]
    }

    /// Horizontal 2×2 covariance block mapped through the linear part.
    pub fn apply_to_covariance(&self, covariance: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
        let column_x = self.rotate_vector([1.0, 0.0]);
        let column_y = self.rotate_vector([0.0, 1.0]);
        let linear = [[column_x[0], column_y[0]], [column_x[1], column_y[1]]];
        let mut result = [[0.0; 2]; 2];
        for row in 0..2 {
            for column in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        result[row][column] += linear[row][k] * covariance[k][l] * linear[column][l];
                    }
                }
            }
        }
        result
    }
}

/// Eigenvalues (major, minor) of a symmetric 2×2 matrix.
pub fn symmetric_eigenvalues_2x2(matrix: [[f64; 2]; 2]) -> [f64; 2] {
    let half_trace = 0.5 * (matrix[0][0] + matrix[1][1]);
    let spread = (0.25 * (matrix[0][0] - matrix[1][1]).powi(2) + matrix[0][1] * matrix[0][1]).sqrt();
    [half_trace + spread, half_trace - spread]
}

/// Δᵀ Σ⁻¹ Δ for a horizontal error; ∞ when Σ is singular.
pub fn mahalanobis_squared_2d(error: [f64; 2], covariance: [[f64; 2]; 2]) -> f64 {
    let [[a, b], [_, d]] = covariance;
    let determinant = a * d - b * b;
    if determinant.is_nan() || determinant <= 0.0 {
        return f64::INFINITY;
    }
    (d * error[0] * error[0] - 2.0 * b * error[0] * error[1] + a * error[1] * error[1]) / determinant
}

pub fn horizontal_block(covariance: &[[f64; 3]; 3]) -> [[f64; 2]; 2] {
    [[covariance[0][0], covariance[0][1]], [covariance[1][0], covariance[1][1]]]
}
