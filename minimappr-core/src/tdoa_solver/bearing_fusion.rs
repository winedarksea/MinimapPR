//! Closed-form multi-node bearing triangulation; port of
//! `minimappr/core/multi_node_bearing_fusion.py::fuse_bearings` (the async
//! TTL store stays server-side). The same per-node rays feed the joint solver
//! as DOA residuals; this closed form is the fast path and a start point.

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
    pub lateral_std_m: f64,
    pub confidence: f64,
    pub event_time_ns: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BearingFusionConfig {
    pub min_separation_deg: f64,
    pub max_condition: f64,
    pub max_range_m: f64,
    pub sound_speed_mps: f64,
    pub window_seconds: f64,
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
    OutOfRange,
}

impl BearingFusionRejection {
    pub fn reason(&self) -> &'static str {
        match self {
            Self::InsufficientNodes => "insufficient_nodes",
            Self::NearParallel => "near_parallel",
            Self::Stale => "stale",
            Self::Degenerate => "degenerate",
            Self::OutOfRange => "out_of_range",
        }
    }
}

fn vector(values: &[f64; 3]) -> Vector3<f64> {
    Vector3::new(values[0], values[1], values[2])
}

fn min_pairwise_separation_deg(directions: &[Vector3<f64>]) -> f64 {
    let mut smallest = 180.0_f64;
    for (index, first) in directions.iter().enumerate() {
        for second in &directions[index + 1..] {
            let dot = first.dot(second).clamp(-1.0, 1.0);
            smallest = smallest.min(dot.abs().acos().to_degrees());
        }
    }
    smallest
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
    if min_pairwise_separation_deg(&directions) < config.min_separation_deg {
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
            let separation_s =
                (first.event_time_ns - second.event_time_ns).abs() as f64 / NANOSECONDS_PER_SECOND;
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
    Ok(BearingFusionResult {
        position_m,
        covariance_m2,
        contributor_node_ids: selected.iter().map(|obs| obs.node_id.clone()).collect(),
        range_m,
        confidence: (1.0 - miss_probability).min(MAXIMUM_COMBINED_CONFIDENCE),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(node_id: &str, origin: [f64; 3], toward: [f64; 3]) -> BearingObservation {
        let direction = (vector(&toward) - vector(&origin)).normalize();
        BearingObservation {
            node_id: node_id.into(),
            origin_m: origin,
            direction: [direction.x, direction.y, direction.z],
            lateral_std_m: 0.5,
            confidence: 0.8,
            event_time_ns: 0,
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
        assert!((fused.confidence - 0.95).abs() < 1e-12);
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
