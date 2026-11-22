//! Residual rows and Jacobians for TDOA pairs, bearing rays and amplitude
//! ratios; port of the helpers in `cartesian_tdoa.py` / `spatial_constraints.py`.
//!
//! Units follow Python: every row is in seconds (bearing / amplitude rows are
//! scaled by `residual_scale_seconds`, normally one interpolated sample).

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

use super::measurement_types::{
    AmplitudeRatioConstraint, PairTdoaMeasurement, SensorPositions, SpatialBearingConstraint,
};
use super::robust_linear_algebra::minimum_norm_least_squares;
use super::TdoaSolverError;

pub const EPSILON: f64 = 1.0e-12;
/// Minimum pairs for the far-field plane-wave direction solve.
pub const MINIMUM_BEARING_SOLVE_PAIR_COUNT: usize = 3;
/// Floor on bearing-constraint strength in the ray-intersection start.
pub const MINIMUM_BEARING_INTERSECTION_STRENGTH: f64 = 0.01;

pub fn sensor_position<'positions>(
    sensor_positions: &'positions SensorPositions,
    sensor_id: &str,
) -> Result<&'positions Vector3<f64>, TdoaSolverError> {
    sensor_positions
        .get(sensor_id)
        .ok_or_else(|| TdoaSolverError::UnknownSensor(sensor_id.to_string()))
}

/// Check every measurement references a known sensor before any residual is built.
pub fn validate_measurement_sensors(
    sensor_positions: &SensorPositions,
    measurements: &[PairTdoaMeasurement],
) -> Result<(), TdoaSolverError> {
    for measurement in measurements {
        sensor_position(sensor_positions, &measurement.sensor_a)?;
        sensor_position(sensor_positions, &measurement.sensor_b)?;
    }
    Ok(())
}

/// Weighted TDOA residual rows `√w · (τ̂(x) − τ)`.
pub fn append_tdoa_residuals(
    rows: &mut Vec<f64>,
    position_m: &Vector3<f64>,
    sensor_positions: &SensorPositions,
    measurements: &[PairTdoaMeasurement],
    sound_speed_mps: f64,
) {
    for measurement in measurements {
        let distance_a_m = (position_m - sensor_positions[&measurement.sensor_a]).norm();
        let distance_b_m = (position_m - sensor_positions[&measurement.sensor_b]).norm();
        let predicted_tdoa_s = (distance_a_m - distance_b_m) / sound_speed_mps;
        rows.push(measurement.weight.sqrt() * (predicted_tdoa_s - measurement.tdoa_seconds));
    }
}

/// Weighted TDOA Jacobian rows. `distance_offset_m` is added to each distance:
/// 0 for the optimiser, `EPSILON` to reproduce Python's covariance Jacobian.
pub fn append_tdoa_jacobian(
    rows: &mut Vec<Vector3<f64>>,
    position_m: &Vector3<f64>,
    sensor_positions: &SensorPositions,
    measurements: &[PairTdoaMeasurement],
    sound_speed_mps: f64,
    distance_offset_m: f64,
) {
    for measurement in measurements {
        let offset_a = position_m - sensor_positions[&measurement.sensor_a];
        let offset_b = position_m - sensor_positions[&measurement.sensor_b];
        let distance_a_m = (offset_a.norm() + distance_offset_m).max(EPSILON);
        let distance_b_m = (offset_b.norm() + distance_offset_m).max(EPSILON);
        let gradient = (offset_a / distance_a_m - offset_b / distance_b_m) / sound_speed_mps;
        rows.push(gradient * measurement.weight.sqrt());
    }
}

/// Unit-vector bearing residual `(x̂ − d) · scale · √strength` about `origin_m`.
/// Returns three zeros when the position sits on the origin (as Python does).
pub fn append_bearing_residual(
    rows: &mut Vec<f64>,
    position_m: &Vector3<f64>,
    origin_m: &Vector3<f64>,
    direction: &Vector3<f64>,
    scale_seconds: f64,
    strength: f64,
) {
    let offset_m = position_m - origin_m;
    let offset_norm_m = offset_m.norm();
    if offset_norm_m < EPSILON {
        rows.extend([0.0, 0.0, 0.0]);
        return;
    }
    let row = (offset_m / offset_norm_m - direction) * scale_seconds * strength.sqrt();
    rows.extend(row.iter().copied());
}

/// Jacobian of [`append_bearing_residual`]: `(I − x̂x̂ᵀ) / d · scale · √strength`.
pub fn append_bearing_jacobian(
    rows: &mut Vec<Vector3<f64>>,
    position_m: &Vector3<f64>,
    origin_m: &Vector3<f64>,
    scale_seconds: f64,
    strength: f64,
) {
    let offset_m = position_m - origin_m;
    let distance_m = offset_m.norm().max(EPSILON);
    let unit = offset_m / distance_m;
    let block = (Matrix3::identity() - unit * unit.transpose()) / distance_m
        * scale_seconds
        * strength.sqrt();
    for row_index in 0..3 {
        rows.push(block.row(row_index).transpose());
    }
}

pub fn append_spatial_constraint_residuals(
    rows: &mut Vec<f64>,
    position_m: &Vector3<f64>,
    bearing_constraints: &[SpatialBearingConstraint],
    amplitude_constraints: &[AmplitudeRatioConstraint],
    residual_scale_seconds: f64,
) {
    for constraint in bearing_constraints {
        append_bearing_residual(
            rows,
            position_m,
            &constraint.origin_m,
            &constraint.direction,
            residual_scale_seconds,
            constraint.strength,
        );
    }
    for constraint in amplitude_constraints {
        let distance_a_m = (position_m - constraint.origin_a_m).norm().max(EPSILON);
        let distance_b_m = (position_m - constraint.origin_b_m).norm().max(EPSILON);
        let predicted_log_ratio = (distance_b_m / distance_a_m).ln();
        rows.push(
            (predicted_log_ratio - constraint.measured_log_amplitude_ratio)
                * residual_scale_seconds
                * constraint.strength.sqrt(),
        );
    }
}

pub fn append_spatial_constraint_jacobian(
    rows: &mut Vec<Vector3<f64>>,
    position_m: &Vector3<f64>,
    bearing_constraints: &[SpatialBearingConstraint],
    amplitude_constraints: &[AmplitudeRatioConstraint],
    residual_scale_seconds: f64,
) {
    for constraint in bearing_constraints {
        append_bearing_jacobian(
            rows,
            position_m,
            &constraint.origin_m,
            residual_scale_seconds,
            constraint.strength,
        );
    }
    for constraint in amplitude_constraints {
        let offset_a = position_m - constraint.origin_a_m;
        let offset_b = position_m - constraint.origin_b_m;
        let distance_a_squared = offset_a.dot(&offset_a).max(EPSILON);
        let distance_b_squared = offset_b.dot(&offset_b).max(EPSILON);
        rows.push(
            (offset_b / distance_b_squared - offset_a / distance_a_squared)
                * residual_scale_seconds
                * constraint.strength.sqrt(),
        );
    }
}

/// Stack 3-vectors into an `m×3` matrix (`m` may be zero).
pub fn stack_jacobian_rows(rows: &[Vector3<f64>]) -> DMatrix<f64> {
    let mut matrix = DMatrix::zeros(rows.len(), 3);
    for (index, row) in rows.iter().enumerate() {
        matrix.row_mut(index).copy_from(&row.transpose());
    }
    matrix
}

/// Far-field plane-wave direction: weighted least squares of
/// `(b − a) · d = c · τ_ab` over pairs, normalised.
pub fn broadband_direction_from_tdoas(
    sensor_positions: &SensorPositions,
    measurements: &[PairTdoaMeasurement],
    sound_speed_mps: f64,
) -> Result<Vector3<f64>, TdoaSolverError> {
    if measurements.len() < MINIMUM_BEARING_SOLVE_PAIR_COUNT {
        return Err(TdoaSolverError::InsufficientMeasurements {
            needed: MINIMUM_BEARING_SOLVE_PAIR_COUNT,
            available: measurements.len(),
        });
    }
    let mut weighted_rows = DMatrix::zeros(measurements.len(), 3);
    let mut weighted_targets = DVector::zeros(measurements.len());
    for (index, measurement) in measurements.iter().enumerate() {
        let weight_root = measurement.weight.sqrt();
        let baseline = sensor_position(sensor_positions, &measurement.sensor_b)?
            - sensor_position(sensor_positions, &measurement.sensor_a)?;
        weighted_rows
            .row_mut(index)
            .copy_from(&(baseline * weight_root).transpose());
        weighted_targets[index] = sound_speed_mps * measurement.tdoa_seconds * weight_root;
    }
    let solution = minimum_norm_least_squares(&weighted_rows, &weighted_targets)?;
    let direction = Vector3::new(solution[0], solution[1], solution[2]);
    let norm = direction.norm();
    if norm < EPSILON {
        return Err(TdoaSolverError::DegenerateDirection);
    }
    Ok(direction / norm)
}

/// Least-squares intersection of ≥ 2 bearing rays, or `None`.
pub fn bearing_intersection_initial_position(
    constraints: &[SpatialBearingConstraint],
) -> Option<Vector3<f64>> {
    if constraints.len() < 2 {
        return None;
    }
    let mut information = Matrix3::zeros();
    let mut target = Vector3::zeros();
    for constraint in constraints {
        let projection =
            Matrix3::identity() - constraint.direction * constraint.direction.transpose();
        let weighted_projection = projection
            * constraint
                .strength
                .max(MINIMUM_BEARING_INTERSECTION_STRENGTH);
        information += weighted_projection;
        target += weighted_projection * constraint.origin_m;
    }
    let information_dynamic = DMatrix::from_column_slice(3, 3, information.as_slice());
    let target_dynamic = DVector::from_column_slice(target.as_slice());
    let solution = minimum_norm_least_squares(&information_dynamic, &target_dynamic).ok()?;
    let position = Vector3::new(solution[0], solution[1], solution[2]);
    position
        .iter()
        .all(|value| value.is_finite())
        .then_some(position)
}
