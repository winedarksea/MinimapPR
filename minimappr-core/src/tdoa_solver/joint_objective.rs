//! Joint TDOA + DOA solve over mixed nodes (plan §Joint TDOA + DOA).
//!
//! Inter-node TDOAs, intra-node stereo cones, array DOAs and world bearing
//! rays all enter one whitened least-squares objective; the canopy prior is
//! optional. Multi-start LM finds the best basin and also reports a
//! near-equal-cost *alternate* (the stereo-cone or collinear-layout mirror)
//! instead of silently picking one.

use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};

use super::canopy_height_prior::CanopyHeightPrior;
use super::joint_residual_terms::{
    resolve_doa_term, resolve_range_difference_term, resolve_world_ray, JointObservations,
    JointResidualModel, ResolvedBearingTerm,
};
use super::levenberg_marquardt_driver::{
    refine_position_with_retry, PositionResidualModel, RefinementOutcome,
};
use super::node_descriptor::NodeDescriptor;
use super::robust_linear_algebra::invert_symmetric_positive_definite;
use super::TdoaSolverError;

pub const DEFAULT_SOUND_SPEED_MPS: f64 = 343.0;
/// Ranges (m) along each DOA / bearing ray used as LM start points.
pub const DEFAULT_RAY_START_RANGES_M: [f64; 3] = [5.0, 20.0, 80.0];
/// Information-matrix condition above which the fix is declared unobservable.
pub const DEFAULT_MAXIMUM_COVARIANCE_CONDITION_NUMBER: f64 = 1.0e12;
/// A second basin closer than this to the best is the same fix, not a mirror.
pub const DEFAULT_MIRROR_CANDIDATE_MINIMUM_SEPARATION_M: f64 = 1.0;
/// χ² margin (≈ 2σ in one dof) within which an alternate basin is "ambiguous".
pub const DEFAULT_AMBIGUITY_CHI_SQUARE_MARGIN: f64 = 4.0;
/// Layout-start offset as a multiple of the node aperture (min 1 m).
const LAYOUT_START_APERTURE_MULTIPLIER: f64 = 1.0;
const MINIMUM_LAYOUT_START_OFFSET_M: f64 = 1.0;
const POSITION_DIMENSIONS: usize = 3;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointSolverConfig {
    pub sound_speed_mps: f64,
    pub canopy_height_prior: Option<CanopyHeightPrior>,
    pub ray_start_ranges_m: Vec<f64>,
    pub maximum_covariance_condition_number: f64,
    pub mirror_candidate_minimum_separation_m: f64,
    pub ambiguity_chi_square_margin: f64,
}

impl Default for JointSolverConfig {
    fn default() -> Self {
        Self {
            sound_speed_mps: DEFAULT_SOUND_SPEED_MPS,
            canopy_height_prior: None,
            ray_start_ranges_m: DEFAULT_RAY_START_RANGES_M.to_vec(),
            maximum_covariance_condition_number: DEFAULT_MAXIMUM_COVARIANCE_CONDITION_NUMBER,
            mirror_candidate_minimum_separation_m: DEFAULT_MIRROR_CANDIDATE_MINIMUM_SEPARATION_M,
            ambiguity_chi_square_margin: DEFAULT_AMBIGUITY_CHI_SQUARE_MARGIN,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct JointSolveResult {
    pub position_m: Vector3<f64>,
    pub covariance_m2: Matrix3<f64>,
    /// Σ of squared whitened residuals at the fix.
    pub chi_square: f64,
    pub residual_count: usize,
    pub degrees_of_freedom: usize,
    pub converged_start_count: usize,
    /// Another basin within the χ² margin (mirror ambiguity), if any.
    pub ambiguous_alternate_position_m: Option<Vector3<f64>>,
}

/// Build the whitened model from descriptors + observations (validates both).
pub fn build_joint_residual_model(
    nodes: &[NodeDescriptor],
    observations: &JointObservations,
    config: &JointSolverConfig,
) -> Result<JointResidualModel, TdoaSolverError> {
    for node in nodes {
        node.validate()?;
    }
    let range_difference_terms = observations
        .mic_pair_tdoas
        .iter()
        .map(|observation| {
            resolve_range_difference_term(nodes, observation, config.sound_speed_mps)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut bearing_terms = observations
        .doas
        .iter()
        .map(|observation| resolve_doa_term(nodes, observation))
        .collect::<Result<Vec<_>, _>>()?;
    for ray in &observations.world_bearing_rays {
        bearing_terms.push(resolve_world_ray(ray)?);
    }
    Ok(JointResidualModel {
        range_difference_terms,
        bearing_terms,
        canopy_height_prior: config.canopy_height_prior,
        sound_speed_mps: config.sound_speed_mps,
    })
}

fn bearing_ray_intersection(bearing_terms: &[ResolvedBearingTerm]) -> Option<Vector3<f64>> {
    if bearing_terms.len() < 2 {
        return None;
    }
    let mut information = Matrix3::zeros();
    let mut target = Vector3::zeros();
    for term in bearing_terms {
        let weight = 1.0 / term.total_std_rad.powi(2);
        let projection =
            (Matrix3::identity() - term.direction * term.direction.transpose()) * weight;
        information += projection;
        target += projection * term.origin_m;
    }
    // Rays that are nearly parallel have no usable intersection; skip quietly
    // (the other starts still run).
    invert_symmetric_positive_definite(&information, DEFAULT_MAXIMUM_COVARIANCE_CONDITION_NUMBER)
        .ok()
        .map(|covariance| covariance * target)
}

fn start_positions(
    nodes: &[NodeDescriptor],
    model: &JointResidualModel,
    config: &JointSolverConfig,
) -> Vec<Vector3<f64>> {
    let node_positions: Vec<Vector3<f64>> =
        nodes.iter().map(NodeDescriptor::position_vector).collect();
    let centroid = node_positions
        .iter()
        .fold(Vector3::zeros(), |sum, position| sum + position)
        / node_positions.len().max(1) as f64;
    let aperture_m = node_positions
        .iter()
        .flat_map(|first| {
            node_positions
                .iter()
                .map(move |second| (first - second).norm())
        })
        .fold(0.0_f64, f64::max);
    let layout_offset_m =
        (aperture_m * LAYOUT_START_APERTURE_MULTIPLIER).max(MINIMUM_LAYOUT_START_OFFSET_M);
    let mut starts = Vec::new();
    if let Some(intersection) = bearing_ray_intersection(&model.bearing_terms) {
        starts.push(intersection);
    }
    for term in &model.bearing_terms {
        for range_m in &config.ray_start_ranges_m {
            starts.push(term.origin_m + term.direction * *range_m);
        }
    }
    starts.push(centroid);
    // Vertical starts matter: on a coplanar layout ∂r/∂z vanishes in the
    // layout plane, so starts inside it can never climb to a canopy source.
    for axis in [
        Vector3::x(),
        -Vector3::x(),
        Vector3::y(),
        -Vector3::y(),
        Vector3::z(),
        -Vector3::z(),
    ] {
        starts.push(centroid + axis * layout_offset_m);
        for range_m in &config.ray_start_ranges_m {
            starts.push(centroid + axis * *range_m);
        }
    }
    starts
}

/// Solve, then invert the whitened information matrix. An unobservable
/// geometry (e.g. one stereo phone alone, collinear mono phones without a
/// tie-breaker) is an explicit `Singular` / `IllConditioned` error.
pub fn solve_joint_position(
    nodes: &[NodeDescriptor],
    observations: &JointObservations,
    config: &JointSolverConfig,
) -> Result<JointSolveResult, TdoaSolverError> {
    if nodes.is_empty() {
        return Err(TdoaSolverError::EmptySystem("nodes"));
    }
    let model = build_joint_residual_model(nodes, observations, config)?;
    let residual_count = model.residual_count();
    if residual_count == 0 {
        return Err(TdoaSolverError::EmptySystem("observations"));
    }
    let mut converged: Vec<RefinementOutcome> = start_positions(nodes, &model, config)
        .into_iter()
        .map(|start| refine_position_with_retry(&model, start, 1.0))
        .filter(|outcome| outcome.converged)
        .collect();
    if let (Some(canopy), Some(best)) = (
        config.canopy_height_prior,
        converged
            .iter()
            .min_by(|left, right| left.cost.total_cmp(&right.cost))
            .cloned(),
    ) {
        let centroid_z =
            nodes.iter().map(|node| node.position_m[2]).sum::<f64>() / nodes.len() as f64;
        for probe in canopy.band_probe_starts(&best.position_m, centroid_z) {
            let outcome = refine_position_with_retry(&model, probe, 1.0);
            if outcome.converged {
                converged.push(outcome);
            }
        }
    }
    converged.sort_by(|left, right| left.cost.total_cmp(&right.cost));
    let best = converged
        .first()
        .cloned()
        .ok_or(TdoaSolverError::RefinementDidNotConverge)?;
    let ambiguous_alternate_position_m = converged
        .iter()
        .skip(1)
        .find(|candidate| {
            (candidate.position_m - best.position_m).norm()
                > config.mirror_candidate_minimum_separation_m
                && candidate.cost <= best.cost + config.ambiguity_chi_square_margin
        })
        .map(|candidate| candidate.position_m);

    let jacobian = model.jacobian(&best.position_m);
    let information_dynamic = jacobian.transpose() * &jacobian;
    let information = Matrix3::from_fn(|row, column| information_dynamic[(row, column)]);
    let covariance_m2 = invert_symmetric_positive_definite(
        &information,
        config.maximum_covariance_condition_number,
    )?;
    Ok(JointSolveResult {
        position_m: best.position_m,
        covariance_m2,
        chi_square: best.cost,
        residual_count,
        degrees_of_freedom: residual_count.saturating_sub(POSITION_DIMENSIONS),
        converged_start_count: converged.len(),
        ambiguous_alternate_position_m,
    })
}
