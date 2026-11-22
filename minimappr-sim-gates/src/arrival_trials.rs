//! Arrival-time Monte Carlo trials through the shipping core localizer
//! (`mesh::arrival_time_localizer`: Cauchy grid seed → `JointLmSeedRefiner`).
//!
//! The nodes handed to the localizer carry the residuals a converged field
//! session leaves: layout position error, clock-filter offset error and the
//! call's pick error, each declared to the solver at its true σ. Noise is
//! drawn per node index from the seed, so a node-dropped or rigidly moved
//! twin of a trial sees exactly the same noise.

use minimappr_core::mesh::arrival_time_localizer::{ArrivalFix, ArrivalLocalizationError, ArrivalTimeLocalizer, NodeArrival};

use crate::gate_config::RealisticPerturbationConfig;
use crate::trial_geometry::{
    azimuth_from, distance, gaussian, horizontal_block, horizontal_centroid, mahalanobis_squared_2d, wrapped_angle_difference_rad,
    PlanarRigidMotion,
};

/// Emission time on the island clock; arbitrary, keeps arrivals positive.
const EMISSION_ISLAND_TIME_S: f64 = 3.0;

/// One trial's world truth plus how its measurements are corrupted.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrivalTrial {
    pub seed: u64,
    pub true_node_positions_m: Vec<[f64; 3]>,
    pub source_position_m: [f64; 3],
    /// Extra arrival delay per node (pocketed phone, multipath lock), seconds.
    pub extra_arrival_delay_s: Vec<f64>,
}

impl ArrivalTrial {
    pub fn new(seed: u64, true_node_positions_m: Vec<[f64; 3]>, source_position_m: [f64; 3]) -> Self {
        let node_count = true_node_positions_m.len();
        Self { seed, true_node_positions_m, source_position_m, extra_arrival_delay_s: vec![0.0; node_count] }
    }

    /// Measurements as the localizer sees them, for the kept node indices,
    /// optionally in a rigidly moved world (noise vectors move with it).
    pub fn arrivals(
        &self,
        kept_node_indices: &[usize],
        motion: Option<&PlanarRigidMotion>,
        perturbation: &RealisticPerturbationConfig,
        sound_speed_mps: f64,
    ) -> Vec<NodeArrival> {
        let declared_timing_std = perturbation.arrival_pick_std_s.hypot(perturbation.clock_residual_std_s);
        let declared_position_std = perturbation.layout_position_std_m.max(perturbation.phone_height_std_m);
        kept_node_indices
            .iter()
            .map(|&node| {
                let truth = self.true_node_positions_m[node];
                let index = node as u64;
                let error_m = [
                    perturbation.layout_position_std_m * gaussian(self.seed, "layout_error", index, 0),
                    perturbation.layout_position_std_m * gaussian(self.seed, "layout_error", index, 1),
                    perturbation.phone_height_std_m * gaussian(self.seed, "layout_error", index, 2),
                ];
                let reported = [truth[0] + error_m[0], truth[1] + error_m[1], truth[2] + error_m[2]];
                let timing_error_s = perturbation.clock_residual_std_s * gaussian(self.seed, "clock_residual", index, 0)
                    + perturbation.arrival_pick_std_s * gaussian(self.seed, "arrival_pick", index, 0);
                NodeArrival {
                    position_m: motion.map_or(reported, |motion| motion.apply(reported)),
                    position_std_m: declared_position_std,
                    reference_arrival_s: EMISSION_ISLAND_TIME_S
                        + distance(truth, self.source_position_m) / sound_speed_mps
                        + timing_error_s
                        + self.extra_arrival_delay_s[node],
                    timing_std_s: declared_timing_std,
                }
            })
            .collect()
    }

    pub fn all_node_indices(&self) -> Vec<usize> {
        (0..self.true_node_positions_m.len()).collect()
    }
}

/// A solved trial scored against truth.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredArrivalFix {
    pub fix: ArrivalFix,
    pub horizontal_error_vector_m: [f64; 2],
    pub horizontal_error_m: f64,
    /// From the true layout centroid.
    pub bearing_error_deg: f64,
    pub horizontal_covariance_m2: [[f64; 2]; 2],
    pub mahalanobis_squared: f64,
}

pub fn score_fix(trial: &ArrivalTrial, fix: ArrivalFix) -> ScoredArrivalFix {
    let truth = trial.source_position_m;
    let error = [fix.fix.position_m[0] - truth[0], fix.fix.position_m[1] - truth[1]];
    let centroid = horizontal_centroid(&trial.true_node_positions_m);
    let bearing_error_deg =
        wrapped_angle_difference_rad(azimuth_from(centroid, fix.fix.position_m), azimuth_from(centroid, truth)).abs().to_degrees();
    let horizontal_covariance_m2 = horizontal_block(&fix.fix.covariance_m2);
    ScoredArrivalFix {
        horizontal_error_vector_m: error,
        horizontal_error_m: error[0].hypot(error[1]),
        bearing_error_deg,
        mahalanobis_squared: mahalanobis_squared_2d(error, horizontal_covariance_m2),
        horizontal_covariance_m2,
        fix,
    }
}

/// Localize a trial with the kept nodes (in the trial's own frame).
pub fn solve_trial(
    trial: &ArrivalTrial,
    kept_node_indices: &[usize],
    localizer: &ArrivalTimeLocalizer,
    perturbation: &RealisticPerturbationConfig,
) -> Result<ScoredArrivalFix, ArrivalLocalizationError> {
    let arrivals = trial.arrivals(kept_node_indices, None, perturbation, localizer.sound_speed_mps);
    let mut fix = localizer.localize(&arrivals)?;
    fix.used_arrival_indices = fix.used_arrival_indices.iter().map(|&local| kept_node_indices[local]).collect();
    Ok(score_fix(trial, fix))
}
