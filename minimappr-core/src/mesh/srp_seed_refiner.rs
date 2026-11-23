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
/// Height σ used to pin a solve at a band edge.
const PINNED_HEIGHT_STD_M: f64 = 0.05;
/// Variance of a uniform distribution over a range r is r² / 12.
const UNIFORM_VARIANCE_DIVISOR: f64 = 12.0;

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
    /// Inflate the horizontal covariance by the drift of the fix across the
    /// seed's height band (height is barely observable from a flat layout).
    pub marginalize_height_band: bool,
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
            marginalize_height_band: true,
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

/// Largest believable height-band drift: max(sensor aperture, horizontal
/// range of the fix from the sensor centroid). A canopy-height change can
/// slide a fix along its range, but not past the scale of the geometry.
fn horizontal_drift_bound_m(terms: &[PairTerm], position_m: &Vector3<f64>) -> f64 {
    let sensors: Vec<&Vector3<f64>> = terms.iter().flat_map(|term| [&term.sensor_a_m, &term.sensor_b_m]).collect();
    let count = sensors.len().max(1) as f64;
    let centroid = sensors.iter().fold([0.0; 2], |acc, p| [acc[0] + p[0] / count, acc[1] + p[1] / count]);
    let aperture = sensors
        .iter()
        .flat_map(|a| sensors.iter().map(move |b| (a[0] - b[0]).hypot(a[1] - b[1])))
        .fold(0.0, f64::max);
    aperture.max((position_m[0] - centroid[0]).hypot(position_m[1] - centroid[1]))
}

/// Horizontal covariance with the source height marginalized over its
/// band: Σ_h = (I_hh)⁻¹ + g gᵀ·Var(z), g = drift / band the horizontal
/// shift per metre of height, Var(z) the height variance the *data* leave
/// once combined with the band taken as uniform (variance band²/12).
///
/// Why not (JᵀJ)⁻¹ + ΔΔᵀ/12 (the original term): that charged the whole
/// band's drift even when the TDOAs pin the height down, and added it to a
/// block that already spread the weak height prior into x, y. Under
/// correctly declared noise it overstated inside-the-layout σ ~1.5× (NEES
/// median 0.42 vs 1.39 expected; closeout C3). Where height is truly
/// unobservable (data information ≈ 0) this reduces to the original term
/// (apart from the conditional base block).
fn height_band_horizontal_covariance(
    information: &Matrix3<f64>,
    height_prior_information: f64,
    band_span_m: f64,
    drift_m: [f64; 2],
) -> Option<nalgebra::Matrix2<f64>> {
    let horizontal_information = information.fixed_view::<2, 2>(0, 0).into_owned();
    let conditional = horizontal_information.try_inverse()?;
    let cross = information.fixed_view::<2, 1>(0, 2).into_owned();
    // Schur complement: information on z after x, y are solved for, minus
    // the weak prior's own share.
    let height_information = information[(2, 2)] - (cross.transpose() * conditional * cross)[(0, 0)];
    let data_height_information = (height_information - height_prior_information).max(0.0);
    let band_variance = band_span_m.powi(2) / UNIFORM_VARIANCE_DIVISOR;
    let height_variance = 1.0 / (data_height_information + 1.0 / band_variance.max(f64::MIN_POSITIVE));
    let per_metre = nalgebra::Vector2::new(drift_m[0], drift_m[1]) / band_span_m.max(f64::MIN_POSITIVE);
    Some(conditional + per_metre * per_metre.transpose() * height_variance)
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
            // Like the core solvers: an unconverged LM point is not a fix,
            // and its (JᵀJ)⁻¹ would be an ellipse around the wrong place.
            if !outcome.converged {
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
            let mut covariance = invert_symmetric_positive_definite(
                &information,
                self.config.maximum_covariance_condition_number,
            )?;
            if self.config.marginalize_height_band {
                let pinned_at = |height_m: f64| {
                    let pinned = SeedResidualModel {
                        terms: model.terms.clone(),
                        sound_speed_mps: model.sound_speed_mps,
                        height_prior: Some((height_m, PINNED_HEIGHT_STD_M)),
                    };
                    let start = Vector3::new(position[0], position[1], height_m);
                    let outcome = refine_position_with_retry(&pinned, start, 1.0);
                    outcome.converged.then_some(outcome.position_m)
                };
                let [band_low, band_high] = seed.source_height_bounds_m;
                let drift_bound = horizontal_drift_bound_m(&model.terms, &position);
                let pinned_pair = (pinned_at(band_low), pinned_at(band_high));
                if let (Some(low), Some(high)) = pinned_pair {
                    // A pinned solve can run off along an unobservable range
                    // direction (seen: 3e8 m, W4-sim-gates), which is not a
                    // height effect. Clamp its length at the geometric bound
                    // rather than dropping it: dropping made σ jump down as
                    // the drift crossed the bound (σ non-monotone in the
                    // node set). The clamp is continuous and frame-invariant.
                    let raw_drift = [high[0] - low[0], high[1] - low[1]];
                    let drift_length = raw_drift[0].hypot(raw_drift[1]);
                    let scale = if drift_length > drift_bound { drift_bound / drift_length } else { 1.0 };
                    let drift = [raw_drift[0] * scale, raw_drift[1] * scale];
                    let prior_information = model.height_prior.map_or(0.0, |(_, std)| std.powi(-2));
                    // The data are credited with height information only when
                    // they can vouch for it: a solved height inside the band
                    // (outside, they contradict the model; seen: a misplaced
                    // node manufacturing a 22 m height, mesh offline_host_loss)
                    // and enough pairs to pin three coordinates (with two, a
                    // 3-phone solve, any height on the intersection curve fits
                    // exactly; such fixes landed 5–10σ out when credited,
                    // closeout C4). Otherwise keep the
                    // original conservative term: weak-prior marginal plus the
                    // full band's drift.
                    let height_in_band = (band_low..=band_high).contains(&position[2]);
                    let pins_every_coordinate = model.terms.len() >= POSITION_DIMENSIONS;
                    let horizontal = if height_in_band && pins_every_coordinate {
                        height_band_horizontal_covariance(&information, prior_information, band_high - band_low, drift)
                    } else {
                        let band_drift = nalgebra::Vector2::new(drift[0], drift[1]);
                        Some(covariance.fixed_view::<2, 2>(0, 0).into_owned() + band_drift * band_drift.transpose() / UNIFORM_VARIANCE_DIVISOR)
                    };
                    if let Some(horizontal) = horizontal {
                        for row in 0..2 {
                            for column in 0..2 {
                                covariance[(row, column)] = horizontal[(row, column)];
                            }
                        }
                    }
                }
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    const SOUND_SPEED_MPS: f64 = 343.0;
    const TIMING_STD_S: f64 = 2e-5;

    fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    /// Pairs against node 0, exact TDOAs plus `bias_s` on pair `biased_pair`.
    fn seed_for(source: [f64; 3], nodes: &[[f64; 3]], start: [f64; 3], biased_pair: Option<(usize, f64)>) -> LmSeed {
        let observations = (1..nodes.len())
            .map(|index| {
                let bias = biased_pair.filter(|(pair, _)| *pair == index - 1).map_or(0.0, |(_, bias)| bias);
                PairTdoaObservation {
                    sensor_a_position_m: nodes[0],
                    sensor_b_position_m: nodes[index],
                    sensor_a_node: 0,
                    sensor_b_node: index,
                    measured_tdoa_s: (distance(source, nodes[0]) - distance(source, nodes[index])) / SOUND_SPEED_MPS + bias,
                    relative_peak_coherence: 1.0,
                    weight: 1.0,
                    lag_resolution_s: TIMING_STD_S,
                }
            })
            .collect();
        LmSeed {
            initial_position_m: start,
            far_field_direction: None,
            observations,
            sound_speed_mps: SOUND_SPEED_MPS,
            source_height_bounds_m: [0.0, 10.0],
            huber_knee: 1.0,
            srp_score: 0.0,
        }
    }

    fn refiner(node_count: usize) -> JointLmSeedRefiner {
        JointLmSeedRefiner { config: SeedRefinerConfig::default(), node_position_std_m: vec![0.05; node_count] }
    }

    fn spread_layout() -> Vec<[f64; 3]> {
        vec![[0.0, 0.0, 1.0], [12.0, 0.0, 1.5], [0.0, 11.0, 0.5], [-10.0, -3.0, 2.0], [7.0, -9.0, 1.0], [-4.0, 9.0, 3.0]]
    }

    #[test]
    fn recovers_source_with_region_that_covers_it() {
        let nodes = spread_layout();
        let source = [3.0, 4.0, 5.0];
        let fix = refiner(nodes.len()).refine(&seed_for(source, &nodes, [4.0, 2.0, 4.0], None)).expect("fix");
        let horizontal_error = (fix.position_m[0] - source[0]).hypot(fix.position_m[1] - source[1]);
        assert!(horizontal_error < 0.5, "error {horizontal_error}");
        assert!(horizontal_error <= 4.0 * fix.horizontal_major_sigma_m(), "σ {}", fix.horizontal_major_sigma_m());
        assert!(fix.rejected_observation_indices.is_empty());
        assert_eq!(fix.degrees_of_freedom, nodes.len() - 1 + 1 - POSITION_DIMENSIONS);
    }

    /// Needs redundancy: with five pairs a 1.4 m bias can be absorbed by the
    /// fit and the wrong pair rejected (worst-first is not a robust loss).
    #[test]
    fn rejects_one_grossly_biased_pair() {
        let mut nodes = spread_layout();
        nodes.extend([[9.0, 8.0, 1.0], [-8.0, -10.0, 1.5], [14.0, -2.0, 0.5]]);
        let source = [3.0, 4.0, 5.0];
        let bias_s = 4.0e-3;
        let fix = refiner(nodes.len()).refine(&seed_for(source, &nodes, [4.0, 2.0, 4.0], Some((2, bias_s)))).expect("fix");
        assert_eq!(fix.rejected_observation_indices, vec![2]);
        let horizontal_error = (fix.position_m[0] - source[0]).hypot(fix.position_m[1] - source[1]);
        assert!(horizontal_error < 0.5, "error {horizontal_error}");
    }

    #[test]
    fn too_few_pairs_is_an_error() {
        let nodes = &spread_layout()[..2];
        let result = refiner(2).refine(&seed_for([3.0, 4.0, 5.0], nodes, [4.0, 2.0, 4.0], None));
        assert!(matches!(result, Err(TdoaSolverError::InsufficientMeasurements { .. })));
    }
}
