//! Per-cluster refinement of an SRP onset cluster (port of
//! `minimappr/core/onset_cluster_refinement.py`).
//!
//! Each cluster's frames vote for per-pair TDOAs (the curve maximum nearest
//! the lag predicted at the cluster's SRP seed, aggregated by weighted
//! median). A single compact array goes through the existing Cartesian solver
//! (`tdoa_solver::solve_cartesian_tdoa`, SciPy-TRF parity engine); multi-node
//! windows run a local soft-L1 TRF from the seed (SciPy-compatible iterates,
//! `scipy_trf_compatible`) so the answer stays in the basin the cluster voted
//! for. LM output is kept only if it holds most of the cluster's SRP score.

use std::collections::BTreeMap;

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

use super::bounded_trf::{bounded_trf_least_squares, VerticalBounds};
use super::python_set_order::small_int_set_iteration_order;
use super::srp_grid::Point3;
use super::tdoa_votes::{solve_reference_tdoas, vote_pair_tdoas, ClusterGeometry};
use super::types::OnsetCluster;
use super::uncertainty::{covariance_from_jacobian, range_observability_from_covariance};
use crate::range_projection::{is_unobservable, RANGE_BEARING_PROJECTED, RANGE_REFINED};
use crate::tdoa_solver::cartesian_tdoa_solve::{solve_cartesian_tdoa, CartesianTdoaSolveRequest};
use crate::tdoa_solver::levenberg_marquardt_driver::PositionResidualModel;
use crate::tdoa_solver::measurement_types::{PairTdoaMeasurement, SensorPositions};
use crate::tdoa_solver::robust_linear_algebra::sorted_singular_values;
use crate::tdoa_solver::scipy_trf_compatible::{scipy_trf_least_squares, TrfSettings};

/// LM may lose this fraction of the seed's SRP score and still be preferred.
pub const REFINEMENT_MIN_SCORE_RATIO: f64 = 0.9;

/// Fit quality = exp(-residual / (this many interpolated sample periods)).
pub const FIT_QUALITY_SAMPLE_PERIODS: f64 = 4.0;

/// A bounded LM needs a strictly interior start.
pub const BOUNDED_SEED_MARGIN_M: f64 = 1.0e-3;

/// Python's float `BEARING_PROJECTED_RANGE_OBSERVABILITY_CAP` (the core
/// constant is f32; 0.05f32 as f64 is not 0.05).
pub const BEARING_PROJECTED_RANGE_OBSERVABILITY: f64 = 0.05;

/// SciPy `least_squares` defaults (trf): ftol = xtol = gtol = 1e-8, max_nfev = 100 n.
const SCIPY_DEFAULT_TOLERANCE: f64 = 1e-8;

const SCIPY_DEFAULT_EVALUATIONS_PER_PARAMETER: usize = 100;

const POSITION_PARAMETER_COUNT: usize = 3;

/// Minimum TDOA measurements for an LM refine.
const MINIMUM_MEASUREMENTS_FOR_REFINE: usize = 3;

#[derive(Debug, Clone, PartialEq)]
pub struct RefinedClusterPosition {
    pub position_m: Point3,
    pub covariance_m2: Option<[[f64; 3]; 3]>,
    pub confidence: f64,
    pub gdop: f64,
    pub range_projection_mode: String,
    pub range_observability: Option<f64>,
    pub residual_rms_seconds: Option<f64>,
    pub reference_sensor: String,
    pub tdoa_s: BTreeMap<String, f64>,
    pub refined_by_least_squares: bool,
}

fn seed_only_result(
    seed_m: &Point3,
    cluster: &OnsetCluster,
    is_bearing_only: bool,
    reference_sensor: &str,
    minimum_std_m: f64,
) -> RefinedClusterPosition {
    let count = cluster.peaks.len();
    let mut covariance = Matrix3::identity() * minimum_std_m * minimum_std_m;
    if count > 1 {
        let mut mean = [0.0; 3];
        for peak in &cluster.peaks {
            for axis in 0..3 {
                mean[axis] += peak.position_m[axis];
            }
        }
        for value in mean.iter_mut() {
            *value /= count as f64;
        }
        let spread = Matrix3::from_fn(|row, column| {
            cluster
                .peaks
                .iter()
                .map(|peak| {
                    (peak.position_m[row] - mean[row]) * (peak.position_m[column] - mean[column])
                })
                .sum::<f64>()
                / (count - 1) as f64
        });
        covariance += spread;
    }
    let mean_value = cluster.peaks.iter().map(|peak| peak.value).sum::<f64>() / count as f64;
    RefinedClusterPosition {
        position_m: *seed_m,
        covariance_m2: Some(super::uncertainty::rows_from_matrix(&covariance)),
        confidence: mean_value.clamp(0.0, 1.0),
        gdop: f64::NAN,
        range_projection_mode: if is_bearing_only {
            RANGE_BEARING_PROJECTED
        } else {
            RANGE_REFINED
        }
        .to_string(),
        range_observability: is_bearing_only.then_some(BEARING_PROJECTED_RANGE_OBSERVABILITY),
        residual_rms_seconds: None,
        reference_sensor: reference_sensor.to_string(),
        tdoa_s: BTreeMap::new(),
        refined_by_least_squares: false,
    }
}

/// Seed and solver settings for one cluster.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefinementSettings {
    pub centroid_m: Point3,
    pub seed_is_bearing_only: bool,
    pub sample_rate_hz: u32,
    pub interpolation_factor: usize,
    pub far_field_default_range_m: f64,
    pub minimum_position_std_m: f64,
    /// Ground..canopy z slab (non-planar layouts only, where no mirror twin exists).
    pub vertical_bounds_m: Option<(f64, f64)>,
}

pub fn refine_cluster_position(
    cluster: &OnsetCluster,
    geometry: &ClusterGeometry<'_>,
    seed_m: &Point3,
    settings: &RefinementSettings,
) -> RefinedClusterPosition {
    let frame_order = small_int_set_iteration_order(
        &cluster
            .peaks
            .iter()
            .map(|peak| peak.frame_index)
            .collect::<Vec<_>>(),
    );
    // Counter.most_common(1): highest count, ties to the first seen (set iteration order).
    let mut reference_counts: Vec<(String, usize)> = Vec::new();
    for index in frame_order {
        let sensor = &geometry.frames[&index].reference_sensor;
        match reference_counts
            .iter_mut()
            .find(|(existing, _)| existing == sensor)
        {
            Some((_, count)) => *count += 1,
            None => reference_counts.push((sensor.clone(), 1)),
        }
    }
    let mut reference_sensor = reference_counts[0].0.clone();
    let mut best_count = reference_counts[0].1;
    for (sensor, count) in &reference_counts {
        if *count > best_count {
            best_count = *count;
            reference_sensor = sensor.clone();
        }
    }
    let measurements = vote_pair_tdoas(seed_m, cluster, geometry, settings.sample_rate_hz);
    let mut involved: Vec<String> = measurements
        .iter()
        .flat_map(|m| [m.sensor_a.clone(), m.sensor_b.clone()])
        .collect();
    involved.sort();
    involved.dedup();
    let seed_result = seed_only_result(
        seed_m,
        cluster,
        settings.seed_is_bearing_only,
        &reference_sensor,
        settings.minimum_position_std_m,
    );
    if measurements.len() < MINIMUM_MEASUREMENTS_FOR_REFINE || !involved.contains(&reference_sensor)
    {
        return seed_result;
    }
    let reference_tdoa_s = solve_reference_tdoas(&measurements, &involved, &reference_sensor);
    let seed_score = geometry.cluster_srp_score(seed_m, cluster);
    let refined = if settings.seed_is_bearing_only {
        refine_bearing_with_cartesian_solver(
            &measurements,
            &involved,
            geometry,
            &reference_sensor,
            &reference_tdoa_s,
            settings,
        )
    } else {
        let mean_peak_value =
            cluster.peaks.iter().map(|peak| peak.value).sum::<f64>() / cluster.peaks.len() as f64;
        refine_position_from_seed(
            &measurements,
            geometry,
            seed_m,
            &reference_sensor,
            &reference_tdoa_s,
            mean_peak_value,
            settings,
        )
    };
    let Some(refined) = refined else {
        return seed_result;
    };
    if geometry.cluster_srp_score(&refined.position_m, cluster)
        < REFINEMENT_MIN_SCORE_RATIO * seed_score
    {
        return seed_result;
    }
    refined
}

fn refine_bearing_with_cartesian_solver(
    measurements: &[PairTdoaMeasurement],
    involved: &[String],
    geometry: &ClusterGeometry<'_>,
    reference_sensor: &str,
    reference_tdoa_s: &BTreeMap<String, f64>,
    settings: &RefinementSettings,
) -> Option<RefinedClusterPosition> {
    let sensor_positions: SensorPositions = involved
        .iter()
        .map(|id| {
            let p = geometry.sensor_positions[id];
            (id.clone(), Vector3::new(p[0], p[1], p[2]))
        })
        .collect();
    let mut request = CartesianTdoaSolveRequest::with_defaults(
        sensor_positions,
        measurements.to_vec(),
        geometry.sound_speed_mps,
        settings.sample_rate_hz,
        settings.interpolation_factor as u32,
        reference_sensor.to_string(),
        reference_tdoa_s.clone(),
    );
    request.far_field_initial_range_m = settings.far_field_default_range_m;
    let solve = solve_cartesian_tdoa(&request).ok()?;
    let centroid = Vector3::new(
        settings.centroid_m[0],
        settings.centroid_m[1],
        settings.centroid_m[2],
    );
    let offset = solve.position_m - centroid;
    let length = offset.norm();
    if length <= 0.0 || !length.is_finite() {
        return None;
    }
    let (position, covariance) = if is_unobservable(Some(solve.range_projection_mode)) {
        // The radial search can publish km-to-Mm ranges on an asymptotic cost;
        // rescale the cone to the default far-field range along the same bearing.
        let scale = settings.far_field_default_range_m / length;
        (
            centroid + offset * scale,
            solve.covariance_m2 * (scale * scale),
        )
    } else {
        (solve.position_m, solve.covariance_m2)
    };
    Some(RefinedClusterPosition {
        position_m: [position[0], position[1], position[2]],
        covariance_m2: covariance
            .iter()
            .all(|v| v.is_finite())
            .then(|| super::uncertainty::rows_from_matrix(&covariance)),
        confidence: solve.confidence,
        gdop: solve.gdop,
        range_projection_mode: solve.range_projection_mode.to_string(),
        range_observability: Some(solve.radial_observability),
        residual_rms_seconds: Some(solve.residual_rms_seconds),
        reference_sensor: reference_sensor.to_string(),
        tdoa_s: reference_tdoa_s.clone(),
        refined_by_least_squares: true,
    })
}

/// Weighted TDOA residuals in units of `f_scale` (SciPy's soft-L1 `f_scale` is
/// folded in so the f_scale = 1 TRF port reproduces the scaled iterates).
pub struct ScaledTdoaResiduals {
    pub rows: Vec<(Vector3<f64>, Vector3<f64>, f64, f64)>,
    pub sound_speed_mps: f64,
    pub residual_scale_s: f64,
}

impl ScaledTdoaResiduals {
    pub fn unscaled(&self, position: &Vector3<f64>) -> DVector<f64> {
        DVector::from_iterator(
            self.rows.len(),
            self.rows.iter().map(|(a, b, tdoa, weight)| {
                weight
                    * (((position - a).norm() - (position - b).norm()) / self.sound_speed_mps
                        - tdoa)
            }),
        )
    }
}

impl PositionResidualModel for ScaledTdoaResiduals {
    fn residuals(&self, position: &Vector3<f64>) -> DVector<f64> {
        self.unscaled(position) / self.residual_scale_s
    }

    fn jacobian(&self, position: &Vector3<f64>) -> DMatrix<f64> {
        let mut jacobian = DMatrix::zeros(self.rows.len(), POSITION_PARAMETER_COUNT);
        for (row, (a, b, _, weight)) in self.rows.iter().enumerate() {
            let gradient = ((position - a).normalize() - (position - b).normalize())
                * (*weight / self.sound_speed_mps / self.residual_scale_s);
            for axis in 0..POSITION_PARAMETER_COUNT {
                jacobian[(row, axis)] = gradient[axis];
            }
        }
        jacobian
    }
}

fn refine_position_from_seed(
    measurements: &[PairTdoaMeasurement],
    geometry: &ClusterGeometry<'_>,
    seed_m: &Point3,
    reference_sensor: &str,
    reference_tdoa_s: &BTreeMap<String, f64>,
    mean_peak_value: f64,
    settings: &RefinementSettings,
) -> Option<RefinedClusterPosition> {
    let sample_time_s =
        1.0 / (settings.sample_rate_hz as f64 * settings.interpolation_factor.max(1) as f64);
    let vector = |id: &str| {
        let p = geometry.sensor_positions[id];
        Vector3::new(p[0], p[1], p[2])
    };
    let model = ScaledTdoaResiduals {
        rows: measurements
            .iter()
            .map(|m| {
                (
                    vector(&m.sensor_a),
                    vector(&m.sensor_b),
                    m.tdoa_seconds,
                    m.weight.sqrt(),
                )
            })
            .collect(),
        sound_speed_mps: geometry.sound_speed_mps,
        residual_scale_s: sample_time_s,
    };
    let trf_settings = TrfSettings {
        ftol: SCIPY_DEFAULT_TOLERANCE,
        xtol: SCIPY_DEFAULT_TOLERANCE,
        // SciPy's gradient test is absolute; residuals here are 1/f_scale larger
        // and so is the gradient squared.
        gtol: SCIPY_DEFAULT_TOLERANCE / (sample_time_s * sample_time_s),
        max_function_evaluations: SCIPY_DEFAULT_EVALUATIONS_PER_PARAMETER
            * POSITION_PARAMETER_COUNT,
    };
    let mut seed = Vector3::new(seed_m[0], seed_m[1], seed_m[2]);
    let outcome = match settings
        .vertical_bounds_m
        .filter(|(lower, upper)| upper > lower)
    {
        Some((lower, upper)) => {
            seed[2] = seed[2].clamp(lower + BOUNDED_SEED_MARGIN_M, upper - BOUNDED_SEED_MARGIN_M);
            bounded_trf_least_squares(
                &model,
                seed,
                &VerticalBounds {
                    lower_z_m: lower,
                    upper_z_m: upper,
                },
                &trf_settings,
                sample_time_s * sample_time_s,
            )?
        }
        None => scipy_trf_least_squares(&model, seed, &trf_settings)?,
    };
    if !outcome.success || !outcome.position_m.iter().all(|value| value.is_finite()) {
        return None;
    }
    let solution = outcome.position_m;
    let residuals = model.unscaled(&solution);
    let residual_rms_s = (residuals.dot(&residuals) / residuals.len() as f64).sqrt();
    // result.jac is in the model's (scaled) units; back to seconds per metre.
    let jacobian = &outcome.scaled_jacobian * sample_time_s;
    let covariance = covariance_from_jacobian(
        &jacobian,
        residual_rms_s,
        sample_time_s,
        settings.minimum_position_std_m,
    )?;
    let fit_quality = (-residual_rms_s / (FIT_QUALITY_SAMPLE_PERIODS * sample_time_s)).exp();
    let singular_values = sorted_singular_values(&jacobian).ok();
    let gdop = match singular_values {
        Some(values)
            if values.len() == POSITION_PARAMETER_COUNT && values.iter().all(|v| *v > 0.0) =>
        {
            values.iter().map(|v| 1.0 / (v * v)).sum::<f64>().sqrt()
        }
        _ => f64::INFINITY,
    };
    let centroid = Vector3::new(
        settings.centroid_m[0],
        settings.centroid_m[1],
        settings.centroid_m[2],
    );
    Some(RefinedClusterPosition {
        position_m: [solution[0], solution[1], solution[2]],
        covariance_m2: Some(super::uncertainty::rows_from_matrix(&covariance)),
        confidence: (mean_peak_value * fit_quality).clamp(0.0, 1.0),
        gdop,
        range_projection_mode: RANGE_REFINED.to_string(),
        range_observability: range_observability_from_covariance(
            &covariance,
            Some(solution - centroid),
        ),
        residual_rms_seconds: Some(residual_rms_s),
        reference_sensor: reference_sensor.to_string(),
        tdoa_s: reference_tdoa_s.clone(),
        refined_by_least_squares: true,
    })
}
