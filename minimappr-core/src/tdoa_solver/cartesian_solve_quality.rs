//! Covariance, range-projection mode, observability, confidence and GDOP for
//! a solved Cartesian position; the tail of Python `solve_cartesian_tdoa`.

use nalgebra::{DMatrix, Matrix3, Vector3};

use super::cartesian_tdoa_solve::CartesianTdoaSolveRequest;
use super::measurement_types::normalized_pair_quality;
use super::robust_linear_algebra::{
    ascending_symmetric_eigen, sorted_singular_value_decomposition,
};
use super::spatial_constraint_terms::EPSILON;
use super::TdoaSolverError;
use crate::range_projection::{
    BEARING_PROJECTED_CONFIDENCE_CAP, BEARING_PROJECTED_RANGE_OBSERVABILITY_CAP, RANGE_ASYMPTOTIC,
    RANGE_BEARING_PROJECTED, RANGE_BOUNDARY, RANGE_REFINED, UNOBSERVABLE_CONFIDENCE_CAP,
    UNOBSERVABLE_RANGE_OBSERVABILITY_CAP,
};
use crate::tdoa_node_position_covariance::{
    node_position_error_covariance_m2, SensorNodePositionStd, WeightedTdoaRow,
};

/// Full-Jacobian condition above which a REFINED range is demoted to ASYMPTOTIC.
pub const RANGE_PROJECTION_CONDITION_THRESHOLD: f64 = 1.0e6;
/// Bearing (preliminary Jacobian) condition below which the bearing counts as observed.
pub const BEARING_OBSERVED_CONDITION_THRESHOLD: f64 = 1.0e8;
/// Cone covariance: radial std is at least this multiple of the solved range…
pub const CONE_RADIAL_STD_RANGE_MULTIPLIER: f64 = 4.0;
/// …and at least this many metres (mirrors `srp_phat.rs`).
pub const CONE_RADIAL_STD_FLOOR_M: f64 = 200.0;
/// Peak quality used when neither TDOA pairs nor bearing constraints exist.
pub const NEUTRAL_PEAK_QUALITY: f64 = 0.5;
/// Fit-quality scale in sample periods: `exp(-rms / (4 · σ_sample))`.
pub const FIT_QUALITY_SAMPLE_PERIODS: f64 = 4.0;
/// Weight of radial observability in the observability factor.
pub const RADIAL_OBSERVABILITY_FACTOR_WEIGHT: f64 = 0.25;
const SPATIAL_DIMENSIONS: usize = 3;

#[derive(Debug, Clone, PartialEq)]
pub struct CartesianTdoaSolve {
    pub position_m: Vector3<f64>,
    pub covariance_m2: Matrix3<f64>,
    pub confidence: f64,
    pub gdop: f64,
    pub residual_rms_seconds: f64,
    pub radial_observability: f64,
    pub range_projection_mode: &'static str,
    pub condition_number: f64,
}

pub struct CartesianFinalizationInputs<'request> {
    pub request: &'request CartesianTdoaSolveRequest,
    pub best_position_m: Vector3<f64>,
    pub centroid_m: Vector3<f64>,
    pub aperture_m: f64,
    pub sample_time_std_s: f64,
    pub minimum_position_std_m: f64,
    pub residual_rms_s: f64,
    pub effective_time_std_s: f64,
    pub range_projection_mode: &'static str,
    pub radial_refinement_std_m: Option<f64>,
    pub preliminary_jacobian: DMatrix<f64>,
    pub final_jacobian: DMatrix<f64>,
}

/// Pad with zero rows up to 3 so the SVD always yields a full 3×3 `Vᵀ`
/// (zero rows add zero singular values and do not change the others).
fn padded_to_three_rows(jacobian: &DMatrix<f64>) -> DMatrix<f64> {
    if jacobian.nrows() >= SPATIAL_DIMENSIONS {
        return jacobian.clone();
    }
    let mut padded = DMatrix::zeros(SPATIAL_DIMENSIONS, jacobian.ncols());
    padded.rows_mut(0, jacobian.nrows()).copy_from(jacobian);
    padded
}

/// `numpy.linalg.cond` (2-norm) of an `m×3` matrix: `σ_max / σ_min` over `min(m, 3)` values.
fn two_norm_condition_number(jacobian: &DMatrix<f64>) -> f64 {
    if jacobian.nrows() == 0 {
        return f64::INFINITY;
    }
    match sorted_singular_value_decomposition(jacobian) {
        Ok(decomposition) => {
            let values = decomposition.singular_values;
            values[0] / values[values.len() - 1]
        }
        Err(_) => f64::INFINITY,
    }
}

/// `_covariance_from_svd`: per-axis `(σ_t / s_i)²` floored, unobservable axes at the cap.
fn covariance_from_singular_values(
    singular_values: &[f64],
    right_vectors_transposed: &DMatrix<f64>,
    time_std_s: f64,
    minimum_position_std_m: f64,
    unobservable_position_std_m: f64,
) -> Matrix3<f64> {
    let variance_floor_m2 = minimum_position_std_m.powi(2);
    let mut variances = Vector3::repeat(unobservable_position_std_m.powi(2));
    for (index, singular_value) in singular_values.iter().take(SPATIAL_DIMENSIONS).enumerate() {
        if *singular_value > EPSILON {
            variances[index] = (time_std_s / singular_value).powi(2).max(variance_floor_m2);
        }
    }
    let right = Matrix3::from_fn(|row, column| right_vectors_transposed[(row, column)]);
    let covariance = right.transpose() * Matrix3::from_diagonal(&variances) * right;
    0.5 * (covariance + covariance.transpose())
}

/// Python `pair_measurement_node_position_covariance_m2`: the stated node
/// position errors through this solve's weighted pair rows; the non-TDOA rows
/// of the final Jacobian (constraints, canopy prior) only shape the inverse.
fn node_position_covariance_m2(inputs: &CartesianFinalizationInputs<'_>) -> Option<Matrix3<f64>> {
    let request = inputs.request;
    let uncertainty = request.node_position_uncertainty.as_ref()?;
    if request.measurements.is_empty() {
        return None;
    }
    let sensor_ids: Vec<&String> = request.sensor_positions.keys().collect();
    let sensor_index = |sensor_id: &str| sensor_ids.iter().position(|id| id.as_str() == sensor_id);
    let node_ids: Vec<&String> = uncertainty.node_position_std_m.keys().collect();
    // A sensor of a node with no stated error gets an index past the stated
    // nodes, so it moves no row.
    let node_index_by_sensor: Vec<usize> = sensor_ids
        .iter()
        .map(|sensor_id| {
            uncertainty
                .sensor_node_ids
                .get(sensor_id.as_str())
                .and_then(|node_id| node_ids.iter().position(|id| *id == node_id))
                .unwrap_or(node_ids.len())
        })
        .collect();
    let rows: Vec<WeightedTdoaRow> = request
        .measurements
        .iter()
        .map(|measurement| {
            Some(WeightedTdoaRow {
                sensor_a: sensor_index(&measurement.sensor_a)?,
                sensor_b: sensor_index(&measurement.sensor_b)?,
                weight: measurement.weight,
            })
        })
        .collect::<Option<_>>()?;
    let mut additional_normal_matrix = [[0.0_f64; 3]; 3];
    let jacobian = &inputs.final_jacobian;
    for row in request.measurements.len()..jacobian.nrows() {
        for out_axis in 0..3 {
            for in_axis in 0..3 {
                additional_normal_matrix[out_axis][in_axis] +=
                    jacobian[(row, out_axis)] * jacobian[(row, in_axis)];
            }
        }
    }
    let sensor_positions_m: Vec<[f64; 3]> = request
        .sensor_positions
        .values()
        .map(|position| [position.x, position.y, position.z])
        .collect();
    let node_position_std_m: Vec<f64> = node_ids
        .iter()
        .map(|node_id| uncertainty.node_position_std_m[*node_id])
        .collect();
    let position = inputs.best_position_m;
    node_position_error_covariance_m2(
        [position.x, position.y, position.z],
        &sensor_positions_m,
        &rows,
        additional_normal_matrix,
        request.sound_speed_mps,
        SensorNodePositionStd {
            node_index_by_sensor: &node_index_by_sensor,
            node_position_std_m: &node_position_std_m,
        },
    )
    .map(|covariance| Matrix3::from_fn(|row, column| covariance[row][column]))
}

pub fn finalize_cartesian_solution(
    inputs: &CartesianFinalizationInputs<'_>,
) -> Result<CartesianTdoaSolve, TdoaSolverError> {
    let request = inputs.request;
    let has_tdoa_measurements = !request.measurements.is_empty();
    let best_position = inputs.best_position_m;
    let mut range_projection_mode = inputs.range_projection_mode;

    let (singular_values, right_vectors_transposed, main_condition_number) =
        match sorted_singular_value_decomposition(&padded_to_three_rows(&inputs.final_jacobian)) {
            Ok(decomposition) => {
                let values: Vec<f64> = decomposition.singular_values.iter().copied().collect();
                let condition = values[0] / values[values.len() - 1].max(EPSILON);
                (
                    values,
                    Some(decomposition.right_vectors_transposed),
                    condition,
                )
            }
            Err(_) => (vec![0.0; SPATIAL_DIMENSIONS], None, f64::INFINITY),
        };
    if range_projection_mode == RANGE_REFINED
        && has_tdoa_measurements
        && main_condition_number > RANGE_PROJECTION_CONDITION_THRESHOLD
    {
        range_projection_mode = RANGE_ASYMPTOTIC;
    }
    let solved_range_m = (best_position - inputs.centroid_m).norm();
    let unobservable_position_std_m = inputs
        .radial_refinement_std_m
        .unwrap_or(0.0)
        .max(solved_range_m)
        .max(inputs.aperture_m)
        .max(inputs.minimum_position_std_m);
    let mut covariance_m2 = match &right_vectors_transposed {
        Some(right) => covariance_from_singular_values(
            &singular_values,
            right,
            inputs.effective_time_std_s,
            inputs.minimum_position_std_m,
            unobservable_position_std_m,
        ),
        None => Matrix3::identity() * unobservable_position_std_m.powi(2),
    };
    let radial_axis = (best_position - inputs.centroid_m) / solved_range_m.max(EPSILON);
    let radial_outer = radial_axis * radial_axis.transpose();
    if let Some(radial_std_m) = inputs.radial_refinement_std_m {
        let radial_variance_m2 = radial_axis.dot(&(covariance_m2 * radial_axis));
        covariance_m2 += radial_outer * (radial_std_m.powi(2) - radial_variance_m2).max(0.0);
    }
    let (mut eigenvalues, _) = ascending_symmetric_eigen(&covariance_m2)?;
    let mut radial_variance_m2 = radial_axis.dot(&(covariance_m2 * radial_axis));
    let mut lateral_variance_m2 = ((covariance_m2.trace() - radial_variance_m2) / 2.0)
        .max(inputs.minimum_position_std_m.powi(2));

    let bearing_condition_number = two_norm_condition_number(&inputs.preliminary_jacobian);
    if range_projection_mode == RANGE_ASYMPTOTIC
        && has_tdoa_measurements
        && bearing_condition_number < BEARING_OBSERVED_CONDITION_THRESHOLD
    {
        range_projection_mode = RANGE_BEARING_PROJECTED;
        let prior_radial_std_m = request
            .far_field_prior_radial_std_m
            .unwrap_or(request.far_field_initial_range_m);
        let cone_radial_std_m = (solved_range_m * CONE_RADIAL_STD_RANGE_MULTIPLIER)
            .max(prior_radial_std_m)
            .max(CONE_RADIAL_STD_FLOOR_M);
        let cone_lateral_std_m = lateral_variance_m2.sqrt();
        covariance_m2 = (Matrix3::identity() - radial_outer) * cone_lateral_std_m.powi(2)
            + radial_outer * cone_radial_std_m.powi(2);
        eigenvalues = ascending_symmetric_eigen(&covariance_m2)?.0;
        radial_variance_m2 = cone_radial_std_m.powi(2);
        lateral_variance_m2 = cone_lateral_std_m.powi(2);
    }
    let mut radial_observability = (lateral_variance_m2
        / radial_variance_m2.max(lateral_variance_m2))
    .sqrt()
    .clamp(0.0, 1.0);
    let condition_observability = (eigenvalues[0].max(EPSILON) / eigenvalues[2].max(EPSILON))
        .sqrt()
        .clamp(0.0, 1.0);
    radial_observability = radial_observability.min(condition_observability);
    let unobservable_mode =
        range_projection_mode == RANGE_ASYMPTOTIC || range_projection_mode == RANGE_BOUNDARY;
    let bearing_projected_mode = range_projection_mode == RANGE_BEARING_PROJECTED;
    if unobservable_mode {
        radial_observability =
            radial_observability.min(UNOBSERVABLE_RANGE_OBSERVABILITY_CAP as f64);
    } else if bearing_projected_mode {
        radial_observability =
            radial_observability.min(BEARING_PROJECTED_RANGE_OBSERVABILITY_CAP as f64);
    }

    let peak_quality = if has_tdoa_measurements {
        request
            .measurements
            .iter()
            .map(|measurement| normalized_pair_quality(measurement.correlation_peak))
            .sum::<f64>()
            / request.measurements.len() as f64
    } else if !request.bearing_constraints.is_empty() {
        request
            .bearing_constraints
            .iter()
            .map(|constraint| constraint.quality)
            .sum::<f64>()
            / request.bearing_constraints.len() as f64
    } else {
        NEUTRAL_PEAK_QUALITY
    };
    let fit_scale_s = (inputs.sample_time_std_s * FIT_QUALITY_SAMPLE_PERIODS).max(EPSILON);
    let fit_quality = (-inputs.residual_rms_s / fit_scale_s).exp();
    let observability_factor = if bearing_projected_mode {
        // The cone covariance's huge radial eigenvalue would crush a valid bearing.
        1.0
    } else {
        let factor = condition_observability
            .max(radial_observability * RADIAL_OBSERVABILITY_FACTOR_WEIGHT)
            .sqrt();
        if has_tdoa_measurements && bearing_condition_number < BEARING_OBSERVED_CONDITION_THRESHOLD
        {
            factor.max(request.bearing_observed_obs_floor)
        } else {
            factor
        }
    };
    let mut confidence = (peak_quality * fit_quality * observability_factor).clamp(0.0, 1.0);
    if unobservable_mode {
        confidence = confidence.min(UNOBSERVABLE_CONFIDENCE_CAP as f64);
    } else if bearing_projected_mode {
        confidence = confidence.min(BEARING_PROJECTED_CONFIDENCE_CAP as f64);
    }
    let observable_singular_values: Vec<f64> = singular_values
        .iter()
        .copied()
        .filter(|value| *value > EPSILON)
        .collect();
    let gdop = if observable_singular_values.len() == SPATIAL_DIMENSIONS {
        observable_singular_values
            .iter()
            .map(|value| 1.0 / (value * value))
            .sum::<f64>()
            .sqrt()
    } else {
        f64::INFINITY
    };
    // Added last (Python `solve_cartesian_tdoa` does the same): a rigid
    // placement error carries no range information, so it must not raise the
    // eigenvalue-ratio observability above.
    if let Some(node_position_term_m2) = node_position_covariance_m2(inputs) {
        covariance_m2 += node_position_term_m2;
    }
    Ok(CartesianTdoaSolve {
        position_m: best_position,
        covariance_m2,
        confidence,
        gdop,
        residual_rms_seconds: inputs.residual_rms_s,
        radial_observability,
        range_projection_mode,
        condition_number: main_condition_number,
    })
}
