//! `TdoaSeedRefiner` for `srp_layout::LmSeed`: the LM polish after a global
//! search (SRP over clips, or the arrival-time grid in
//! [`super::arrival_time_localizer`]).
//!
//! Residual per pair: (predicted − measured TDOA) / σ, with σ combining the
//! pair's timing σ (`lag_resolution_s / √weight`) and both nodes' layout
//! position σ (a node off by δ moves its range by up to δ). A weak Gaussian
//! height prior centred in the seed's height band keeps z observable on a
//! near-flat phone layout, where z barely changes any TDOA. Worst-first
//! outlier rejection stands in for a robust loss (graceful degradation when
//! one phone's clip is bad). The covariance is `(JᵀJ)⁻¹` and refuses an
//! unobservable geometry instead of returning a meaningless ellipse.

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

use crate::srp_layout::lm_handoff::{LmSeed, PairTdoaObservation, TdoaSeedRefiner};
use crate::tdoa_solver::levenberg_marquardt_driver::{
    refine_position_with_retry, PositionResidualModel,
};
use crate::tdoa_solver::robust_linear_algebra::invert_symmetric_positive_definite;
use crate::tdoa_solver::TdoaSolverError;

const POSITION_DIMENSIONS: usize = 3;
const DISTANCE_EPSILON_M: f64 = 1e-9;

#[derive(Debug, Clone, PartialEq)]
pub struct SeedRefinerConfig {
    /// Floor on any pair's timing σ (sub-sample interpolation limit).
    pub minimum_timing_std_s: f64,
    /// σ of the weak height prior; `None` disables it.
    pub height_prior_std_m: Option<f64>,
    pub outlier_rejection_sigma: f64,
    pub maximum_outlier_rejections: usize,
    /// Never reject below this many pair observations.
    pub minimum_observations_after_rejection: usize,
    pub maximum_covariance_condition_number: f64,
}

impl Default for SeedRefinerConfig {
    fn default() -> Self {
        Self {
            minimum_timing_std_s: 5e-6,
            height_prior_std_m: Some(15.0),
            outlier_rejection_sigma: 4.5,
            maximum_outlier_rejections: 2,
            minimum_observations_after_rejection: 3,
            maximum_covariance_condition_number: 1.0e14,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RefinedTdoaFix {
    pub position_m: [f64; 3],
    pub covariance_m2: [[f64; 3]; 3],
    pub chi_square: f64,
    pub degrees_of_freedom: usize,
    /// Indices into `LmSeed::observations` rejected as outliers.
    pub rejected_observation_indices: Vec<usize>,
}

impl RefinedTdoaFix {
    /// Square root of the larger horizontal covariance eigenvalue.
    pub fn horizontal_major_sigma_m(&self) -> f64 {
        let [[a, b, _], [_, d, _], _] = self.covariance_m2;
        let half_trace = 0.5 * (a + d);
        let spread = (0.25 * (a - d).powi(2) + b * b).sqrt();
        (half_trace + spread).max(0.0).sqrt()
    }
}

#[derive(Debug, Clone)]
struct PairTerm {
    sensor_a_m: Vector3<f64>,
    sensor_b_m: Vector3<f64>,
    measured_tdoa_s: f64,
    total_std_s: f64,
}

struct SeedResidualModel {
    terms: Vec<PairTerm>,
    sound_speed_mps: f64,
    height_prior: Option<(f64, f64)>,
}

impl SeedResidualModel {
    fn row_count(&self) -> usize {
        self.terms.len() + usize::from(self.height_prior.is_some())
    }

    fn normalized_pair_residuals(&self, position_m: &Vector3<f64>) -> Vec<f64> {
        self.residuals(position_m).iter().take(self.terms.len()).copied().collect()
    }
}

impl PositionResidualModel for SeedResidualModel {
    fn residuals(&self, position_m: &Vector3<f64>) -> DVector<f64> {
        let mut rows = Vec::with_capacity(self.row_count());
        for term in &self.terms {
            let predicted = ((position_m - term.sensor_a_m).norm()
                - (position_m - term.sensor_b_m).norm())
                / self.sound_speed_mps;
            rows.push((predicted - term.measured_tdoa_s) / term.total_std_s);
        }
        if let Some((centre_z, std_z)) = self.height_prior {
            rows.push((position_m[2] - centre_z) / std_z);
        }
        DVector::from_vec(rows)
    }

    fn jacobian(&self, position_m: &Vector3<f64>) -> DMatrix<f64> {
        let mut jacobian = DMatrix::zeros(self.row_count(), POSITION_DIMENSIONS);
        for (row, term) in self.terms.iter().enumerate() {
            let offset_a = position_m - term.sensor_a_m;
            let offset_b = position_m - term.sensor_b_m;
            let gradient = offset_a / offset_a.norm().max(DISTANCE_EPSILON_M)
                - offset_b / offset_b.norm().max(DISTANCE_EPSILON_M);
            let scaled = gradient / (self.sound_speed_mps * term.total_std_s);
            for column in 0..POSITION_DIMENSIONS {
                jacobian[(row, column)] = scaled[column];
            }
        }
        if let Some((_, std_z)) = self.height_prior {
            jacobian[(self.terms.len(), 2)] = 1.0 / std_z;
        }
        jacobian
    }
}

/// LM refiner over an `LmSeed`; `node_position_std_m[i]` is layout node i's σ.
#[derive(Debug, Clone)]
pub struct JointLmSeedRefiner {
    pub config: SeedRefinerConfig,
    pub node_position_std_m: Vec<f64>,
}

impl JointLmSeedRefiner {
    fn pair_term(&self, observation: &PairTdoaObservation, sound_speed_mps: f64) -> PairTerm {
        let timing_std = observation.lag_resolution_s / observation.weight.max(f64::MIN_POSITIVE).sqrt();
        let node_std = |node: usize| self.node_position_std_m.get(node).copied().unwrap_or(0.0);
        let position_std_s = (node_std(observation.sensor_a_node).powi(2)
            + node_std(observation.sensor_b_node).powi(2))
        .sqrt()
            / sound_speed_mps;
        let total_std_s = (timing_std.max(self.config.minimum_timing_std_s).powi(2)
            + position_std_s.powi(2))
        .sqrt();
        PairTerm {
            sensor_a_m: Vector3::from(observation.sensor_a_position_m),
            sensor_b_m: Vector3::from(observation.sensor_b_position_m),
            measured_tdoa_s: observation.measured_tdoa_s,
            total_std_s,
        }
    }
}

impl TdoaSeedRefiner for JointLmSeedRefiner {
    type Fix = RefinedTdoaFix;
    type Error = TdoaSolverError;

    fn refine(&self, seed: &LmSeed) -> Result<RefinedTdoaFix, TdoaSolverError> {
        let usable: Vec<usize> = (0..seed.observations.len())
            .filter(|&index| seed.observations[index].weight > 0.0)
            .collect();
        if usable.len() < 2 {
            return Err(TdoaSolverError::InsufficientMeasurements {
                needed: 2,
                available: usable.len(),
            });
        }
        let height_prior = self.config.height_prior_std_m.map(|std| {
            let [low, high] = seed.source_height_bounds_m;
            (0.5 * (low + high), std)
        });
        let mut active = usable;
        let mut rejected = Vec::new();
        let mut position = Vector3::from(seed.initial_position_m);
        loop {
            let model = SeedResidualModel {
                terms: active
                    .iter()
                    .map(|&index| self.pair_term(&seed.observations[index], seed.sound_speed_mps))
                    .collect(),
                sound_speed_mps: seed.sound_speed_mps,
                height_prior,
            };
            let outcome = refine_position_with_retry(&model, position, 1.0);
            if !outcome.position_m.iter().all(|value| value.is_finite()) {
                return Err(TdoaSolverError::RefinementDidNotConverge);
            }
            position = outcome.position_m;
            let normalized = model.normalized_pair_residuals(&position);
            let worst = normalized
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                .map(|(index, value)| (index, value.abs()));
            let may_reject = rejected.len() < self.config.maximum_outlier_rejections
                && active.len() > self.config.minimum_observations_after_rejection;
            if let Some((worst_index, worst_value)) = worst {
                if may_reject && worst_value > self.config.outlier_rejection_sigma {
                    rejected.push(active.remove(worst_index));
                    continue;
                }
            }
            let jacobian = model.jacobian(&position);
            let information_dynamic = jacobian.transpose() * &jacobian;
            let information = Matrix3::from_fn(|row, column| information_dynamic[(row, column)]);
            let covariance = invert_symmetric_positive_definite(
                &information,
                self.config.maximum_covariance_condition_number,
            )?;
            return Ok(RefinedTdoaFix {
                position_m: [position[0], position[1], position[2]],
                covariance_m2: std::array::from_fn(|row| {
                    std::array::from_fn(|column| covariance[(row, column)])
                }),
                chi_square: outcome.cost,
                degrees_of_freedom: model.row_count().saturating_sub(POSITION_DIMENSIONS),
                rejected_observation_indices: rejected,
            });
        }
    }
}
