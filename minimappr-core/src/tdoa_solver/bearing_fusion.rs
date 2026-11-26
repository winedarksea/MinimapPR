//! Closed-form multi-node bearing triangulation; port of
//! `minimappr/core/multi_node_bearing_fusion.py::fuse_bearings` (the async
//! TTL store stays server-side). The same per-node rays feed the joint solver
//! as DOA residuals; this closed form is the fast path and a start point.
//!
//! Lateral std at ray parameter t is the angular part (`lateral_std_m` stated
//! at `range_prior_m`, scaled by t) in quadrature with the range-independent
//! node-position part (`position_std_m`), floored at `min_lateral_std_m`. The
//! intersection is solved with each cone's std at its solve range, then
//! re-weighted once at the first fused range, so weights and gate agree.
//!
//! Inputs carry no solve ID, so "same call" is geometric: the fused point must
//! be in front of every array and within `max_miss_sigma` lateral std of every
//! ray; confidence is discounted by agreement. Per node, the cone agreeing best
//! with the others wins; if the set still fails with more than two nodes, the
//! node whose removal ranks best is dropped (bounded leave-one-out).

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
/// Rank tiers for comparing candidate selections (lower is better).
const RANK_TIER_FUSED: u8 = 0;
const RANK_TIER_RAYS_DISAGREE: u8 = 1;
const RANK_TIER_OTHER_REJECTION: u8 = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BearingObservation {
    pub node_id: String,
    /// Ray apex (contributing-sensor centroid).
    pub origin_m: [f64; 3],
    /// Unit bearing from origin toward the source.
    pub direction: [f64; 3],
    /// Angular part of the lateral std, stated at `range_prior_m`.
    pub lateral_std_m: f64,
    pub confidence: f64,
    pub event_time_ns: i64,
    /// Solve range at which `lateral_std_m` is stated.
    pub range_prior_m: f64,
    /// Range-independent lateral std (node-position error); 0 when unstated.
    #[serde(default)]
    pub position_std_m: f64,
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
    /// Numerical floor on the combined lateral std (metres).
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
    /// May be a subset of the input nodes (outlier rejection).
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

struct SelectionSolve {
    outcome: Result<BearingFusionResult, BearingFusionRejection>,
    max_normalized_miss: f64,
}

impl SelectionSolve {
    fn rejected(rejection: BearingFusionRejection) -> Self {
        Self {
            outcome: Err(rejection),
            max_normalized_miss: f64::INFINITY,
        }
    }

    fn rank(&self) -> (u8, f64) {
        match &self.outcome {
            Ok(_) => (RANK_TIER_FUSED, self.max_normalized_miss),
            Err(BearingFusionRejection::RaysDisagree) => {
                (RANK_TIER_RAYS_DISAGREE, self.max_normalized_miss)
            }
            Err(_) => (RANK_TIER_OTHER_REJECTION, f64::INFINITY),
        }
    }

    /// Python tuple `<` on `(tier, max_normalized_miss)`.
    fn ranks_better_than(&self, other: &Self) -> bool {
        let (tier, miss) = self.rank();
        let (other_tier, other_miss) = other.rank();
        tier < other_tier || (tier == other_tier && miss < other_miss)
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

/// Angular part scaled to `ray_parameter_m`, in quadrature with the position
/// part; `None` (degenerate) without a usable solve range or a positive std.
pub(crate) fn lateral_std_at_ray_parameter_m(
    observation: &BearingObservation,
    ray_parameter_m: f64,
    min_lateral_std_m: f64,
) -> Option<f64> {
    let range_prior_m = observation.range_prior_m;
    if !(range_prior_m.is_finite() && range_prior_m > EPSILON) {
        return None;
    }
    let angular_std_m = observation.lateral_std_m * ray_parameter_m / range_prior_m;
    let position_std_m = observation.position_std_m;
    let combined_std_m =
        (angular_std_m * angular_std_m + position_std_m * position_std_m).sqrt();
    let lateral_std_m = combined_std_m.max(min_lateral_std_m);
    (lateral_std_m.is_finite() && lateral_std_m > EPSILON).then_some(lateral_std_m)
}

fn lateral_stds_m(
    selected: &[&BearingObservation],
    ray_parameters_m: &[f64],
    min_lateral_std_m: f64,
) -> Option<Vec<f64>> {
    selected
        .iter()
        .zip(ray_parameters_m)
        .map(|(observation, &ray_parameter_m)| {
            lateral_std_at_ray_parameter_m(observation, ray_parameter_m, min_lateral_std_m)
        })
        .collect()
}

/// Distance along each ray to the foot of `position_m`; `None` if any is not in front.
fn ray_parameters_m(
    selected: &[&BearingObservation],
    directions: &[Vector3<f64>],
    position_m: &Vector3<f64>,
) -> Option<Vec<f64>> {
    selected
        .iter()
        .zip(directions)
        .map(|(observation, direction)| {
            let ray_parameter_m = (position_m - vector(&observation.origin_m)).dot(direction);
            (ray_parameter_m > 0.0).then_some(ray_parameter_m)
        })
        .collect()
}

/// Closed-form intersection with wᵢ = confidence / σ²_lat.
fn weighted_intersection(
    selected: &[&BearingObservation],
    directions: &[Vector3<f64>],
    lateral_stds_m: &[f64],
    max_condition: f64,
) -> Result<(Vector3<f64>, Matrix3<f64>), BearingFusionRejection> {
    let mut information = Matrix3::zeros();
    let mut target = Vector3::zeros();
    for ((observation, direction), lateral_std_m) in
        selected.iter().zip(directions).zip(lateral_stds_m)
    {
        let lateral_variance_m2 = (lateral_std_m * lateral_std_m).max(LATERAL_VARIANCE_FLOOR_M2);
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
    if smallest <= EPSILON || largest / smallest.max(EPSILON) > max_condition {
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
    Ok((position_m, inverse_information))
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

/// Fuse exactly one cone per node, or say why not.
fn solve_selection(selected: &[&BearingObservation], config: &BearingFusionConfig, speed: f64) -> SelectionSolve {
    use BearingFusionRejection::*;
    let directions: Vec<Vector3<f64>> = selected.iter().map(|obs| vector(&obs.direction)).collect();
    if min_pairwise_line_separation_deg(&directions) < config.min_separation_deg {
        return SelectionSolve::rejected(NearParallel);
    }
    for (index, first) in selected.iter().enumerate() {
        for second in &selected[index + 1..] {
            let baseline_m = (vector(&first.origin_m) - vector(&second.origin_m)).norm();
            let allowed_s = config.window_seconds + baseline_m / speed;
            // abs_diff: `(a - b).abs()` overflows (panics in debug) on extreme stamps.
            let separation_s =
                first.event_time_ns.abs_diff(second.event_time_ns) as f64 / NANOSECONDS_PER_SECOND;
            if separation_s > allowed_s {
                return SelectionSolve::rejected(Stale);
            }
        }
    }
    let solve = || -> Result<(Vector3<f64>, Matrix3<f64>, Vec<f64>, Vec<f64>), BearingFusionRejection> {
        let priors: Vec<f64> = selected.iter().map(|obs| obs.range_prior_m).collect();
        let prior_stds = lateral_stds_m(selected, &priors, config.min_lateral_std_m).ok_or(Degenerate)?;
        let (first_position, _) =
            weighted_intersection(selected, &directions, &prior_stds, config.max_condition)?;
        let first_parameters =
            ray_parameters_m(selected, &directions, &first_position).ok_or(BehindArray)?;
        let reweighted_stds =
            lateral_stds_m(selected, &first_parameters, config.min_lateral_std_m).ok_or(Degenerate)?;
        let (position_m, covariance_m2) =
            weighted_intersection(selected, &directions, &reweighted_stds, config.max_condition)?;
        let parameters = ray_parameters_m(selected, &directions, &position_m).ok_or(BehindArray)?;
        let gate_stds =
            lateral_stds_m(selected, &parameters, config.min_lateral_std_m).ok_or(Degenerate)?;
        Ok((position_m, covariance_m2, parameters, gate_stds))
    };
    let (position_m, covariance_m2, parameters, gate_stds) = match solve() {
        Ok(value) => value,
        Err(rejection) => return SelectionSolve::rejected(rejection),
    };
    let normalized_misses: Vec<f64> = selected
        .iter()
        .zip(&directions)
        .zip(parameters.iter().zip(&gate_stds))
        .map(|((observation, direction), (&ray_parameter_m, &lateral_std_m))| {
            let offset = position_m - vector(&observation.origin_m);
            (offset - direction * ray_parameter_m).norm() / lateral_std_m
        })
        .collect();
    let max_normalized_miss = normalized_misses.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max_normalized_miss > config.max_miss_sigma {
        return SelectionSolve {
            outcome: Err(RaysDisagree),
            max_normalized_miss,
        };
    }
    let Some(stabilized) = stabilize_covariance(&covariance_m2, STABILIZED_MINIMUM_STD_M) else {
        return SelectionSolve::rejected(Degenerate);
    };
    let origin_centroid = selected
        .iter()
        .fold(Vector3::zeros(), |sum, obs| sum + vector(&obs.origin_m))
        / selected.len() as f64;
    let range_m = (position_m - origin_centroid).norm();
    if !range_m.is_finite() || range_m > config.max_range_m {
        return SelectionSolve::rejected(OutOfRange);
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
    SelectionSolve {
        outcome: Ok(BearingFusionResult {
            position_m,
            covariance_m2: stabilized,
            contributor_node_ids: selected.iter().map(|obs| obs.node_id.clone()).collect(),
            range_m,
            confidence: ((1.0 - miss_probability) * agreement_factor)
                .min(MAXIMUM_COMBINED_CONFIDENCE),
        }),
        max_normalized_miss,
    }
}

/// Cones grouped by node, in first-seen node order.
type ConesByNode<'a> = Vec<(&'a str, Vec<&'a BearingObservation>)>;
/// One chosen cone per node: (index into `ConesByNode`, cone).
type Selection<'a> = Vec<(usize, &'a BearingObservation)>;

fn selected_cones<'a>(selection: &Selection<'a>) -> Vec<&'a BearingObservation> {
    selection.iter().map(|(_, cone)| *cone).collect()
}

/// Per node, swap in the cone that best agrees with the others' choices; only
/// strict rank improvements, at most one pass per node.
fn refine_cone_choice<'a>(
    cones_by_node: &ConesByNode<'a>,
    mut selection: Selection<'a>,
    config: &BearingFusionConfig,
    speed: f64,
) -> (Selection<'a>, SelectionSolve) {
    let mut solve = solve_selection(&selected_cones(&selection), config, speed);
    for _ in 0..selection.len() {
        let mut changed = false;
        for position in 0..selection.len() {
            let node_index = selection[position].0;
            for &cone in &cones_by_node[node_index].1 {
                if std::ptr::eq(cone, selection[position].1) {
                    continue;
                }
                let mut trial = selection.clone();
                trial[position].1 = cone;
                let trial_solve = solve_selection(&selected_cones(&trial), config, speed);
                if trial_solve.ranks_better_than(&solve) {
                    selection = trial;
                    solve = trial_solve;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    (selection, solve)
}

pub fn fuse_bearings(
    observations: &[BearingObservation],
    config: &BearingFusionConfig,
) -> Result<BearingFusionResult, BearingFusionRejection> {
    if observations.len() < MINIMUM_DISTINCT_NODES {
        return Err(BearingFusionRejection::InsufficientNodes);
    }
    let mut cones_by_node: ConesByNode = Vec::new();
    for observation in observations {
        match cones_by_node
            .iter_mut()
            .find(|(node_id, _)| *node_id == observation.node_id)
        {
            Some((_, cones)) => cones.push(observation),
            None => cones_by_node.push((observation.node_id.as_str(), vec![observation])),
        }
    }
    if cones_by_node.len() < MINIMUM_DISTINCT_NODES {
        return Err(BearingFusionRejection::InsufficientNodes);
    }
    let speed = if config.sound_speed_mps > EPSILON {
        config.sound_speed_mps
    } else {
        SPEED_OF_SOUND_FALLBACK_MPS
    };
    let initial: Selection = cones_by_node
        .iter()
        .enumerate()
        .map(|(node_index, (_, cones))| {
            let most_confident = cones[1..].iter().fold(cones[0], |best, &cone| {
                if cone.confidence > best.confidence {
                    cone
                } else {
                    best
                }
            });
            (node_index, most_confident)
        })
        .collect();
    let (mut selection, mut solve) = refine_cone_choice(&cones_by_node, initial, config, speed);
    let full_set_outcome = solve.outcome.clone();
    while solve.outcome.is_err() && selection.len() > MINIMUM_DISTINCT_NODES {
        let mut best_drop: Option<(usize, SelectionSolve)> = None;
        for drop_position in 0..selection.len() {
            let remaining: Vec<&BearingObservation> = selection
                .iter()
                .enumerate()
                .filter(|(position, _)| *position != drop_position)
                .map(|(_, (_, cone))| *cone)
                .collect();
            let candidate = solve_selection(&remaining, config, speed);
            if best_drop
                .as_ref()
                .map_or(true, |(_, best)| candidate.ranks_better_than(best))
            {
                best_drop = Some((drop_position, candidate));
            }
        }
        let drop_position = best_drop.map(|(position, _)| position).unwrap_or(0);
        selection.remove(drop_position);
        (selection, solve) = refine_cone_choice(&cones_by_node, selection, config, speed);
    }
    match solve.outcome {
        Ok(result) => Ok(result),
        Err(_) => full_set_outcome,
    }
}

#[cfg(test)]
#[path = "bearing_fusion_tests.rs"]
mod tests;
