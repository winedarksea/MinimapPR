//! Closed-form multi-node bearing triangulation; port of
//! `minimappr/core/multi_node_bearing_fusion.py::fuse_bearings` (the async
//! TTL store stays server-side). The same per-node rays feed the joint solver
//! as DOA residuals; this closed form is the fast path and a start point.
//!
//! Inputs carry no solve ID, so "same call" is geometric: the fused point must
//! be in front of every array and within `max_miss_sigma` range-scaled lateral
//! standard deviations of every ray; confidence is discounted by agreement.

use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};

use super::robust_linear_algebra::ascending_symmetric_eigen;

const EPSILON: f64 = 1.0e-9;
const SPEED_OF_SOUND_FALLBACK_MPS: f64 = 343.0;
/// Lateral variance floor (m²) so a zero-width cone cannot dominate.
const LATERAL_VARIANCE_FLOOR_M2: f64 = 1.0e-4;
/// Confidence floor inside the information weight.
const MINIMUM_OBSERVATION_WEIGHT_CONFIDENCE: f64 = 0.05;
/// Per-observation confidence cap and combined-confidence cap.
const MAXIMUM_OBSERVATION_CONFIDENCE: f64 = 0.99;
const MAXIMUM_COMBINED_CONFIDENCE: f64 = 0.95;
/// Covariance eigenvalue floor (std, metres) after inversion.
const STABILIZED_MINIMUM_STD_M: f64 = 1.0;
/// `_stabilize_covariance` never floors below this std.
const STABILIZATION_ABSOLUTE_MINIMUM_STD_M: f64 = 1.0e-3;
const NANOSECONDS_PER_SECOND: f64 = 1.0e9;
const MINIMUM_DISTINCT_NODES: usize = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BearingObservation {
    pub node_id: String,
    /// Ray apex (contributing-sensor centroid).
    pub origin_m: [f64; 3],
    /// Unit bearing from origin toward the source.
    pub direction: [f64; 3],
    /// Perpendicular uncertainty of the cone, stated at `range_prior_m`.
    pub lateral_std_m: f64,
    pub confidence: f64,
    pub event_time_ns: i64,
    /// Solve range at which `lateral_std_m` is stated.
    pub range_prior_m: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
/// Missing keys deserialize to the Python keyword defaults.
#[serde(default)]
pub struct BearingFusionConfig {
    pub min_separation_deg: f64,
    pub max_condition: f64,
    pub max_range_m: f64,
    pub sound_speed_mps: f64,
    pub window_seconds: f64,
    /// Largest per-ray miss distance, in lateral std at the fused range.
    pub max_miss_sigma: f64,
    /// Floor on the range-scaled lateral std in the agreement test (metres).
    pub min_lateral_std_m: f64,
}

impl Default for BearingFusionConfig {
    /// Python keyword defaults.
    fn default() -> Self {
        Self {
            min_separation_deg: 5.0,
            max_condition: 1.0e4,
            max_range_m: 1200.0,
            sound_speed_mps: SPEED_OF_SOUND_FALLBACK_MPS,
            window_seconds: 1.5,
            max_miss_sigma: 3.0,
            min_lateral_std_m: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BearingFusionResult {
    pub position_m: Vector3<f64>,
    pub covariance_m2: Matrix3<f64>,
    pub contributor_node_ids: Vec<String>,
    pub range_m: f64,
    pub confidence: f64,
}

/// Python's reason strings, as an enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BearingFusionRejection {
    InsufficientNodes,
    NearParallel,
    Stale,
    Degenerate,
    BehindArray,
    RaysDisagree,
    OutOfRange,
}

impl BearingFusionRejection {
    pub fn reason(&self) -> &'static str {
        match self {
            Self::InsufficientNodes => "insufficient_nodes",
            Self::NearParallel => "near_parallel",
            Self::Stale => "stale",
            Self::Degenerate => "degenerate",
            Self::BehindArray => "behind_array",
            Self::RaysDisagree => "rays_disagree",
            Self::OutOfRange => "out_of_range",
        }
    }
}

fn vector(values: &[f64; 3]) -> Vector3<f64> {
    Vector3::new(values[0], values[1], values[2])
}

/// Smallest angle between bearing lines from the signed ray angle: parallel
/// and anti-parallel rays both leave range along the shared axis unobservable.
fn min_pairwise_line_separation_deg(directions: &[Vector3<f64>]) -> f64 {
    let mut smallest = 180.0_f64;
    for (index, first) in directions.iter().enumerate() {
        for second in &directions[index + 1..] {
            let signed_dot = first.dot(second).clamp(-1.0, 1.0);
            let ray_angle_deg = signed_dot.acos().to_degrees();
            smallest = smallest.min(ray_angle_deg).min(180.0 - ray_angle_deg);
        }
    }
    smallest
}

/// Per-ray miss distance over lateral std at the fused range.
fn normalized_ray_misses(
    selected: &[&BearingObservation],
    directions: &[Vector3<f64>],
    position_m: &Vector3<f64>,
    min_lateral_std_m: f64,
) -> Result<Vec<f64>, BearingFusionRejection> {
    let mut normalized_misses = Vec::with_capacity(selected.len());
    for (observation, direction) in selected.iter().zip(directions) {
        let offset = position_m - vector(&observation.origin_m);
        let ray_parameter_m = offset.dot(direction);
        if !(ray_parameter_m > 0.0) {
            return Err(BearingFusionRejection::BehindArray);
        }
        let range_prior_m = observation.range_prior_m;
        if !(range_prior_m.is_finite() && range_prior_m > EPSILON) {
            return Err(BearingFusionRejection::Degenerate);
        }
        let miss_distance_m = (offset - direction * ray_parameter_m).norm();
        let lateral_std_at_fused_range_m =
            (observation.lateral_std_m * ray_parameter_m / range_prior_m).max(min_lateral_std_m);
        normalized_misses.push(miss_distance_m / lateral_std_at_fused_range_m);
    }
    Ok(normalized_misses)
}

/// Clamp covariance eigenvalues up to `max(minimum_std, 1e-3)²`.
pub fn stabilize_covariance(
    covariance_m2: &Matrix3<f64>,
    minimum_std_m: f64,
) -> Option<Matrix3<f64>> {
    let (eigenvalues, eigenvectors) = ascending_symmetric_eigen(covariance_m2).ok()?;
    let floor_m2 = minimum_std_m
        .max(STABILIZATION_ABSOLUTE_MINIMUM_STD_M)
        .powi(2);
    let clamped = eigenvalues.map(|value| value.max(floor_m2));
    let stabilized = eigenvectors * Matrix3::from_diagonal(&clamped) * eigenvectors.transpose();
    Some(0.5 * (stabilized + stabilized.transpose()))
}

/// Keep the most confident observation per node, in first-seen node order.
fn best_observation_per_node(observations: &[BearingObservation]) -> Vec<&BearingObservation> {
    let mut selected: Vec<&BearingObservation> = Vec::new();
    for observation in observations {
        match selected
            .iter_mut()
            .find(|kept| kept.node_id == observation.node_id)
        {
            Some(kept) if observation.confidence > kept.confidence => *kept = observation,
            Some(_) => {}
            None => selected.push(observation),
        }
    }
    selected
}

pub fn fuse_bearings(
    observations: &[BearingObservation],
    config: &BearingFusionConfig,
) -> Result<BearingFusionResult, BearingFusionRejection> {
    if observations.len() < MINIMUM_DISTINCT_NODES {
        return Err(BearingFusionRejection::InsufficientNodes);
    }
    let selected = best_observation_per_node(observations);
    if selected.len() < MINIMUM_DISTINCT_NODES {
        return Err(BearingFusionRejection::InsufficientNodes);
    }
    let directions: Vec<Vector3<f64>> = selected.iter().map(|obs| vector(&obs.direction)).collect();
    if min_pairwise_line_separation_deg(&directions) < config.min_separation_deg {
        return Err(BearingFusionRejection::NearParallel);
    }
    let speed = if config.sound_speed_mps > EPSILON {
        config.sound_speed_mps
    } else {
        SPEED_OF_SOUND_FALLBACK_MPS
    };
    for (index, first) in selected.iter().enumerate() {
        for second in &selected[index + 1..] {
            let baseline_m = (vector(&first.origin_m) - vector(&second.origin_m)).norm();
            let allowed_s = config.window_seconds + baseline_m / speed;
            // abs_diff: `(a - b).abs()` overflows (panics in debug) on extreme stamps.
            let separation_s =
                first.event_time_ns.abs_diff(second.event_time_ns) as f64 / NANOSECONDS_PER_SECOND;
            if separation_s > allowed_s {
                return Err(BearingFusionRejection::Stale);
            }
        }
    }

    let mut information = Matrix3::zeros();
    let mut target = Vector3::zeros();
    for (observation, direction) in selected.iter().zip(&directions) {
        let lateral_variance_m2 = observation
            .lateral_std_m
            .powi(2)
            .max(LATERAL_VARIANCE_FLOOR_M2);
        let weight = observation
            .confidence
            .max(MINIMUM_OBSERVATION_WEIGHT_CONFIDENCE)
            / lateral_variance_m2;
        let projection = (Matrix3::identity() - direction * direction.transpose()) * weight;
        information += projection;
        target += projection * vector(&observation.origin_m);
    }
    let (eigenvalues, eigenvectors) =
        ascending_symmetric_eigen(&information).map_err(|_| BearingFusionRejection::Degenerate)?;
    let (smallest, largest) = (eigenvalues[0], eigenvalues[2]);
    if smallest <= EPSILON || largest / smallest.max(EPSILON) > config.max_condition {
        return Err(BearingFusionRejection::Degenerate);
    }
    let inverse_information = eigenvectors
        * Matrix3::from_diagonal(&eigenvalues.map(|value| 1.0 / value))
        * eigenvectors.transpose();
    let position_m = inverse_information * target;
    if !position_m
        .iter()
        .chain(inverse_information.iter())
        .all(|value| value.is_finite())
    {
        return Err(BearingFusionRejection::Degenerate);
    }
    let normalized_misses = normalized_ray_misses(
        &selected,
        &directions,
        &position_m,
        config.min_lateral_std_m,
    )?;
    if normalized_misses
        .iter()
        .any(|&normalized_miss| normalized_miss > config.max_miss_sigma)
    {
        return Err(BearingFusionRejection::RaysDisagree);
    }
    let covariance_m2 = stabilize_covariance(&inverse_information, STABILIZED_MINIMUM_STD_M)
        .ok_or(BearingFusionRejection::Degenerate)?;
    let origin_centroid = selected
        .iter()
        .fold(Vector3::zeros(), |sum, obs| sum + vector(&obs.origin_m))
        / selected.len() as f64;
    let range_m = (position_m - origin_centroid).norm();
    if !range_m.is_finite() || range_m > config.max_range_m {
        return Err(BearingFusionRejection::OutOfRange);
    }
    let miss_probability: f64 = selected
        .iter()
        .map(|obs| 1.0 - obs.confidence.min(MAXIMUM_OBSERVATION_CONFIDENCE))
        .product();
    let mean_squared_normalized_miss = normalized_misses
        .iter()
        .map(|normalized_miss| normalized_miss * normalized_miss)
        .sum::<f64>()
        / normalized_misses.len() as f64;
    let agreement_factor = (-0.5 * mean_squared_normalized_miss).exp();
    Ok(BearingFusionResult {
        position_m,
        covariance_m2,
        contributor_node_ids: selected.iter().map(|obs| obs.node_id.clone()).collect(),
        range_m,
        confidence: ((1.0 - miss_probability) * agreement_factor).min(MAXIMUM_COMBINED_CONFIDENCE),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(node_id: &str, origin: [f64; 3], toward: [f64; 3]) -> BearingObservation {
        let offset = vector(&toward) - vector(&origin);
        let direction = offset.normalize();
        BearingObservation {
            node_id: node_id.into(),
            origin_m: origin,
            direction: [direction.x, direction.y, direction.z],
            lateral_std_m: 0.5,
            confidence: 0.8,
            event_time_ns: 0,
            range_prior_m: offset.norm(),
        }
    }

    #[test]
    fn two_crossing_rays_intersect_at_source() {
        let source = [20.0, 30.0, 5.0];
        let fused = fuse_bearings(
            &[
                observation("a", [0.0, 0.0, 0.0], source),
                observation("b", [40.0, 0.0, 0.0], source),
            ],
            &BearingFusionConfig::default(),
        )
        .unwrap();
        assert!((fused.position_m - vector(&source)).norm() < 1e-9);
        // Exact intersection: agreement factor is 1, so only the cap applies.
        assert!((fused.confidence - 0.95).abs() < 1e-12);
    }

    #[test]
    fn partial_agreement_lowers_confidence_below_noisy_or() {
        let fused = fuse_bearings(
            &[
                observation("a", [0.0, 0.0, 0.0], [20.0, 30.0, 5.0]),
                observation("b", [40.0, 0.0, 0.0], [20.0, 30.0, 7.0]),
            ],
            &BearingFusionConfig::default(),
        )
        .unwrap();
        assert!(fused.confidence < 0.95 && fused.confidence > 0.0);
    }

    #[test]
    fn contradictory_rays_are_rejected() {
        let config = BearingFusionConfig {
            max_range_m: 6000.0,
            ..BearingFusionConfig::default()
        };
        let behind = [
            observation("a", [0.0, 0.0, 0.0], [-10.0, 0.0, 0.0]),
            observation("b", [10.0, 10.0, 0.0], [10.0, 20.0, 0.0]),
        ];
        assert_eq!(
            fuse_bearings(&behind, &config),
            Err(BearingFusionRejection::BehindArray)
        );
        let vertically_skewed = [
            observation("a", [0.0, 0.0, 0.0], [20.0, 0.0, 0.0]),
            observation("b", [20.0, -20.0, 30.0], [20.0, 0.0, 30.0]),
        ];
        assert_eq!(
            fuse_bearings(&vertically_skewed, &config),
            Err(BearingFusionRejection::RaysDisagree)
        );
    }

    #[test]
    fn parallel_and_single_node_inputs_are_rejected() {
        let far = [0.0, 1000.0, 0.0];
        let rays = [
            observation("a", [0.0, 0.0, 0.0], far),
            observation("b", [1.0, 0.0, 0.0], far),
        ];
        assert_eq!(
            fuse_bearings(&rays, &BearingFusionConfig::default()),
            Err(BearingFusionRejection::NearParallel)
        );
        let same_node = [
            observation("a", [0.0, 0.0, 0.0], far),
            observation("a", [9.0, 0.0, 0.0], [5.0, 5.0, 0.0]),
        ];
        assert_eq!(
            fuse_bearings(&same_node, &BearingFusionConfig::default()),
            Err(BearingFusionRejection::InsufficientNodes)
        );
    }
}
