//! Interface between SRP (global search) and the LM solver (local polish).
//!
//! SRP finds the right basin by summing full curves (robust to a pair whose
//! strongest peak is a reflection); LM then fits per-pair TDOAs exactly.
//! Each pair's TDOA here is the local curve peak nearest the SRP seed's
//! predicted lag, not the pair's global argmax, so a reflection-dominated
//! pair contributes its direct-path peak. The solver (W2-core-solver)
//! implements `TdoaSeedRefiner`; this module never iterates LM itself.

use super::polar_grid::{norm, sub};
use super::robust_scoring::{RobustPairScorer, WeightedPair};

/// One pair measurement for the LM objective.
#[derive(Clone, Debug, PartialEq)]
pub struct PairTdoaObservation {
    pub sensor_a_position_m: [f64; 3],
    pub sensor_b_position_m: [f64; 3],
    /// Layout node indices; equal for intra-node (array) pairs.
    pub sensor_a_node: usize,
    pub sensor_b_node: usize,
    /// τ_ab = (|x − a| − |x − b|) / c at the local peak near the seed.
    pub measured_tdoa_s: f64,
    /// Coherence at that peak divided by the pair's global peak (≤ 1).
    /// < 1 means a stronger peak exists elsewhere (likely a reflection).
    pub relative_peak_coherence: f64,
    /// SRP pair weight (node weights × significance); use as LM weight.
    pub weight: f64,
    /// Lag resolution of the source curve.
    pub lag_resolution_s: f64,
}

/// Everything an LM refiner needs to polish an SRP fix.
#[derive(Clone, Debug, PartialEq)]
pub struct LmSeed {
    /// Converged SRP cell centre (far-field: centroid + direction × range).
    pub initial_position_m: [f64; 3],
    /// Set when the SRP maximum is on the far-field ring: fit a bearing.
    pub far_field_direction: Option<[f64; 3]>,
    pub observations: Vec<PairTdoaObservation>,
    pub sound_speed_mps: f64,
    /// Absolute z bounds from the canopy prior (ground + height band).
    pub source_height_bounds_m: [f64; 2],
    /// Huber knee used by SRP, in normalized-coherence units, so the solver
    /// can keep the same robust loss shape.
    pub huber_knee: f64,
    /// SRP exact score at the seed (−Σ w ρ), for comparing candidates.
    pub srp_score: f64,
}

impl LmSeed {
    /// Per-observation residual (predicted − measured), seconds.
    pub fn residuals_s(&self, position_m: [f64; 3]) -> Vec<f64> {
        self.observations
            .iter()
            .map(|observation| {
                let predicted = (norm(sub(position_m, observation.sensor_a_position_m))
                    - norm(sub(position_m, observation.sensor_b_position_m)))
                    / self.sound_speed_mps;
                predicted - observation.measured_tdoa_s
            })
            .collect()
    }
}

/// Implemented by the LM solver module.
pub trait TdoaSeedRefiner {
    type Fix;
    type Error;
    fn refine(&self, seed: &LmSeed) -> Result<Self::Fix, Self::Error>;
}

fn predicted_lag_s(pair: &WeightedPair, position_m: [f64; 3], far_field_direction: Option<[f64; 3]>, sound_speed_mps: f64) -> f64 {
    match far_field_direction {
        Some(direction) => {
            let baseline = sub(pair.sensor_b_position_m, pair.sensor_a_position_m);
            (direction[0] * baseline[0] + direction[1] * baseline[1] + direction[2] * baseline[2]) / sound_speed_mps
        }
        None => {
            (norm(sub(position_m, pair.sensor_a_position_m)) - norm(sub(position_m, pair.sensor_b_position_m)))
                / sound_speed_mps
        }
    }
}

/// Build the handoff from a converged SRP estimate.
pub fn build_lm_seed(
    scorer: &RobustPairScorer,
    initial_position_m: [f64; 3],
    far_field_direction: Option<[f64; 3]>,
    srp_score: f64,
    source_height_bounds_m: [f64; 2],
    peak_search_half_window_s: f64,
) -> LmSeed {
    let observations = scorer
        .pairs
        .iter()
        .filter(|pair| pair.weight > 0.0)
        .map(|pair| {
            let predicted = predicted_lag_s(pair, initial_position_m, far_field_direction, scorer.sound_speed_mps);
            let (measured_tdoa_s, coherence) = pair.curve.local_peak_near(predicted, peak_search_half_window_s);
            PairTdoaObservation {
                sensor_a_position_m: pair.sensor_a_position_m,
                sensor_b_position_m: pair.sensor_b_position_m,
                sensor_a_node: pair.sensor_a_node,
                sensor_b_node: pair.sensor_b_node,
                measured_tdoa_s,
                relative_peak_coherence: (coherence / pair.curve.peak_coherence.max(f32::MIN_POSITIVE)) as f64,
                weight: pair.weight,
                lag_resolution_s: pair.curve.lag_step_s,
            }
        })
        .collect();
    LmSeed {
        initial_position_m,
        far_field_direction,
        observations,
        sound_speed_mps: scorer.sound_speed_mps,
        source_height_bounds_m,
        huber_knee: scorer.huber_knee,
        srp_score,
    }
}
