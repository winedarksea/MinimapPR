//! Multi-start points for the Cartesian TDOA solve (`_algebraic_initial_position`
//! and the start list in `minimappr/core/cartesian_tdoa.py`). Start order is
//! parity-relevant: the best-cost candidate wins ties by first occurrence.

use nalgebra::{DMatrix, DVector, Vector3};

use super::cartesian_tdoa_solve::CartesianTdoaSolveRequest;
use super::measurement_types::SensorPositions;
use super::robust_linear_algebra::minimum_norm_least_squares;
use super::spatial_constraint_terms::{
    bearing_intersection_initial_position, broadband_direction_from_tdoas, sensor_position,
};
use super::TdoaSolverError;

pub const MAX_REFINEMENT_START_COUNT: usize = 4;
/// Third TDOA start: this many apertures out along the plane-wave bearing…
pub const FAR_START_APERTURE_MULTIPLIER: f64 = 10.0;
/// …but at least this far (metres).
pub const FAR_START_MINIMUM_RANGE_M: f64 = 1.0;

/// Largest pairwise sensor distance (metres).
pub fn array_aperture_m(sensor_positions: &SensorPositions) -> f64 {
    let positions: Vec<&Vector3<f64>> = sensor_positions.values().collect();
    let mut aperture_m = 0.0_f64;
    for (index, first) in positions.iter().enumerate() {
        for second in &positions[index + 1..] {
            aperture_m = aperture_m.max((*first - *second).norm());
        }
    }
    aperture_m
}

/// Linearised spherical-intersection start (`_algebraic_initial_position`).
fn algebraic_initial_position(
    request: &CartesianTdoaSolveRequest,
) -> Result<Option<Vector3<f64>>, TdoaSolverError> {
    let reference_position =
        *sensor_position(&request.sensor_positions, &request.reference_sensor)?;
    let row_count = request.reference_tdoa_s.len();
    const LINEARISED_UNKNOWN_COUNT: usize = 4;
    const MINIMUM_ALGEBRAIC_ROW_COUNT: usize = 3;
    if row_count < MINIMUM_ALGEBRAIC_ROW_COUNT {
        return Ok(None);
    }
    let mut rows = DMatrix::zeros(row_count, LINEARISED_UNKNOWN_COUNT);
    let mut targets = DVector::zeros(row_count);
    for (index, (sensor_id, tdoa_seconds)) in request.reference_tdoa_s.iter().enumerate() {
        let range_difference_m = request.sound_speed_mps * tdoa_seconds;
        let position = *sensor_position(&request.sensor_positions, sensor_id)?;
        let baseline = 2.0 * (position - reference_position);
        rows[(index, 0)] = baseline.x;
        rows[(index, 1)] = baseline.y;
        rows[(index, 2)] = baseline.z;
        rows[(index, 3)] = 2.0 * range_difference_m;
        targets[index] = position.dot(&position)
            - reference_position.dot(&reference_position)
            - range_difference_m * range_difference_m;
    }
    Ok(minimum_norm_least_squares(&rows, &targets)
        .ok()
        .map(|solution| Vector3::new(solution[0], solution[1], solution[2]))
        .filter(|position| position.iter().all(|value| value.is_finite())))
}

/// Multi-start points plus the plane-wave direction (`None` without TDOAs),
/// which the far-field guard reuses as a candidate bearing.
pub fn initial_positions(
    request: &CartesianTdoaSolveRequest,
    centroid_m: &Vector3<f64>,
    aperture_m: f64,
    minimum_position_std_m: f64,
) -> Result<(Vec<Vector3<f64>>, Option<Vector3<f64>>), TdoaSolverError> {
    let mut plane_wave_direction = None;
    let mut starts = if request.measurements.is_empty() {
        let search_scale_m = aperture_m.max(minimum_position_std_m);
        vec![
            *centroid_m,
            centroid_m + Vector3::x() * search_scale_m,
            centroid_m + Vector3::y() * search_scale_m,
            centroid_m + Vector3::z() * search_scale_m,
        ]
    } else {
        let direction = broadband_direction_from_tdoas(
            &request.sensor_positions,
            &request.measurements,
            request.sound_speed_mps,
        )?;
        plane_wave_direction = Some(direction);
        vec![
            *centroid_m,
            centroid_m + direction * aperture_m.max(minimum_position_std_m),
            centroid_m
                + direction
                    * (aperture_m * FAR_START_APERTURE_MULTIPLIER).max(FAR_START_MINIMUM_RANGE_M),
        ]
    };
    if let Some(intersection) = bearing_intersection_initial_position(&request.bearing_constraints)
    {
        starts.insert(0, intersection);
    }
    if let Some(prior) = &request.bearing_prior {
        starts.insert(
            0,
            centroid_m + prior.direction * aperture_m.max(minimum_position_std_m),
        );
    }
    if !request.measurements.is_empty() {
        if let Some(algebraic) = algebraic_initial_position(request)? {
            starts.insert(0, algebraic);
        }
    }
    starts.truncate(MAX_REFINEMENT_START_COUNT);
    Ok((starts, plane_wave_direction))
}
