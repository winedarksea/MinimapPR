//! Multi-start Cartesian TDOA solve; numerical port of
//! `minimappr/core/cartesian_tdoa.py::solve_cartesian_tdoa`.
//!
//! Deviations from Python, all numerically negligible for parity:
//! - SciPy TRF with `loss="soft_l1"` (f_scale 1) is replaced by plain LM. The
//!   residuals are in seconds (~1e-5), so soft-L1's weight `(1 + r²)^-¼`
//!   differs from 1 by ~1e-10 and the minimiser is the same.
//! - LM uses the analytic Jacobian (SciPy used 2-point finite differences).
//! Extension (off in parity mode): optional [`CanopyHeightPrior`] row plus a
//! z-mirrored restart of the best candidate.

use std::collections::BTreeMap;

use nalgebra::{DMatrix, DVector, Vector3};

use super::canopy_height_prior::CanopyHeightPrior;
use super::cartesian_solve_quality::{finalize_cartesian_solution, CartesianFinalizationInputs};
use super::levenberg_marquardt_driver::{
    refine_position_with_retry, residual_cost, PositionResidualModel,
};
use super::measurement_types::{
    AmplitudeRatioConstraint, BearingPrior, PairTdoaMeasurement, SensorPositions,
    SpatialBearingConstraint,
};
use super::radial_range_refinement::{
    adaptive_radial_refinement, radial_standard_deviation, RadialRefinementRequest,
};
use super::robust_linear_algebra::minimum_norm_least_squares;
use super::scipy_trf_compatible::{scipy_trf_least_squares, TrfSettings};
use super::spatial_constraint_terms::{
    append_bearing_jacobian, append_bearing_residual, append_spatial_constraint_jacobian,
    append_spatial_constraint_residuals, append_tdoa_jacobian, append_tdoa_residuals,
    bearing_intersection_initial_position, broadband_direction_from_tdoas, sensor_position,
    stack_jacobian_rows, validate_measurement_sensors, EPSILON,
};
use super::TdoaSolverError;
use crate::range_projection::RANGE_REFINED;

pub use super::cartesian_solve_quality::CartesianTdoaSolve;

/// Python `BEARING_OBSERVED_OBS_FACTOR_FLOOR`.
pub const BEARING_OBSERVED_OBS_FACTOR_FLOOR: f64 = 1.0;
pub const MAX_REFINEMENT_START_COUNT: usize = 4;
pub const COMPACT_ARRAY_RADIAL_REFINEMENT_APERTURE_MULTIPLIER: f64 = 2.0;
pub const PRIOR_DOMINATED_ASYMPTOTIC_RADIAL_STD_FRACTION: f64 = 0.9;
pub const DEFAULT_ASYMPTOTIC_RADIAL_STD_FRACTION: f64 = 5.0;
pub const DEFAULT_FAR_FIELD_INITIAL_RANGE_M: f64 = 50.0;
/// Position-std floor (metres) regardless of sample rate.
pub const MINIMUM_POSITION_STD_FLOOR_M: f64 = 0.05;
/// Third TDOA start: this many apertures out along the plane-wave bearing…
pub const FAR_START_APERTURE_MULTIPLIER: f64 = 10.0;
/// …but at least this far (metres).
pub const FAR_START_MINIMUM_RANGE_M: f64 = 1.0;
/// Minimum radius of the radial search as a fraction of the aperture.
pub const RADIAL_SEARCH_MINIMUM_APERTURE_FRACTION: f64 = 0.5;
/// Radial refinement must improve the RMS by at least this fraction to be kept.
pub const RADIAL_REFINEMENT_MINIMUM_RELATIVE_IMPROVEMENT: f64 = 1.0e-4;

/// Which nonlinear refinement drives the multi-start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CartesianRefinementEngine {
    /// Reproduces SciPy TRF iterates (incl. its early `gtol` stop on
    /// second-scale residuals): server-equivalent, the parity default.
    ScipyTrfParity,
    /// MINPACK LM to full convergence. Same answer on well-conditioned
    /// layouts; on compact arrays it finds the true (noise-driven) minimum,
    /// which the Python range-mode thresholds were not calibrated for.
    LevenbergMarquardt,
}

/// Inputs mirroring the Python keyword arguments.
#[derive(Debug, Clone)]
pub struct CartesianTdoaSolveRequest {
    pub sensor_positions: SensorPositions,
    pub measurements: Vec<PairTdoaMeasurement>,
    pub sound_speed_mps: f64,
    pub sample_rate_hz: u32,
    pub interpolation_factor: u32,
    pub reference_sensor: String,
    pub reference_tdoa_s: BTreeMap<String, f64>,
    pub bearing_prior: Option<BearingPrior>,
    pub bearing_constraints: Vec<SpatialBearingConstraint>,
    pub amplitude_constraints: Vec<AmplitudeRatioConstraint>,
    pub far_field_initial_range_m: f64,
    pub far_field_prior_radial_std_m: Option<f64>,
    pub radial_refinement_enabled: bool,
    pub bearing_observed_obs_floor: f64,
    /// Field Mode extension; `None` reproduces the Python solver exactly.
    pub canopy_height_prior: Option<CanopyHeightPrior>,
    pub refinement_engine: CartesianRefinementEngine,
}

impl CartesianTdoaSolveRequest {
    /// A request with the Python defaults for every optional argument.
    pub fn with_defaults(
        sensor_positions: SensorPositions,
        measurements: Vec<PairTdoaMeasurement>,
        sound_speed_mps: f64,
        sample_rate_hz: u32,
        interpolation_factor: u32,
        reference_sensor: String,
        reference_tdoa_s: BTreeMap<String, f64>,
    ) -> Self {
        Self {
            sensor_positions,
            measurements,
            sound_speed_mps,
            sample_rate_hz,
            interpolation_factor,
            reference_sensor,
            reference_tdoa_s,
            bearing_prior: None,
            bearing_constraints: Vec::new(),
            amplitude_constraints: Vec::new(),
            far_field_initial_range_m: DEFAULT_FAR_FIELD_INITIAL_RANGE_M,
            far_field_prior_radial_std_m: None,
            radial_refinement_enabled: true,
            bearing_observed_obs_floor: BEARING_OBSERVED_OBS_FACTOR_FLOOR,
            canopy_height_prior: None,
            refinement_engine: CartesianRefinementEngine::ScipyTrfParity,
        }
    }
}

/// `combined_residual_seconds` from Python, plus the optional canopy row.
pub struct CartesianResidualModel<'request> {
    pub request: &'request CartesianTdoaSolveRequest,
    pub centroid_m: Vector3<f64>,
    pub sample_time_std_s: f64,
}

impl PositionResidualModel for CartesianResidualModel<'_> {
    fn residuals(&self, position_m: &Vector3<f64>) -> DVector<f64> {
        let request = self.request;
        let mut rows = Vec::new();
        append_tdoa_residuals(
            &mut rows,
            position_m,
            &request.sensor_positions,
            &request.measurements,
            request.sound_speed_mps,
        );
        if let Some(prior) = &request.bearing_prior {
            append_bearing_residual(
                &mut rows,
                position_m,
                &self.centroid_m,
                &prior.direction,
                self.sample_time_std_s,
                prior.strength,
            );
        }
        append_spatial_constraint_residuals(
            &mut rows,
            position_m,
            &request.bearing_constraints,
            &request.amplitude_constraints,
            self.sample_time_std_s,
        );
        if let Some(canopy) = &request.canopy_height_prior {
            rows.push(canopy.residual(position_m, self.sample_time_std_s));
        }
        DVector::from_vec(rows)
    }

    fn jacobian(&self, position_m: &Vector3<f64>) -> DMatrix<f64> {
        let request = self.request;
        let mut rows = Vec::new();
        append_tdoa_jacobian(
            &mut rows,
            position_m,
            &request.sensor_positions,
            &request.measurements,
            request.sound_speed_mps,
            0.0,
        );
        if let Some(prior) = &request.bearing_prior {
            append_bearing_jacobian(
                &mut rows,
                position_m,
                &self.centroid_m,
                self.sample_time_std_s,
                prior.strength,
            );
        }
        append_spatial_constraint_jacobian(
            &mut rows,
            position_m,
            &request.bearing_constraints,
            &request.amplitude_constraints,
            self.sample_time_std_s,
        );
        if let Some(canopy) = &request.canopy_height_prior {
            rows.push(canopy.jacobian(position_m, self.sample_time_std_s));
        }
        stack_jacobian_rows(&rows)
    }
}

/// Jacobian used for observability/covariance: TDOA (with Python's `+ε`
/// distance offset) and spatial constraints, *without* the bearing prior rows.
pub fn observability_jacobian(
    request: &CartesianTdoaSolveRequest,
    position_m: &Vector3<f64>,
    sample_time_std_s: f64,
) -> DMatrix<f64> {
    let mut rows = Vec::new();
    append_tdoa_jacobian(
        &mut rows,
        position_m,
        &request.sensor_positions,
        &request.measurements,
        request.sound_speed_mps,
        EPSILON,
    );
    append_spatial_constraint_jacobian(
        &mut rows,
        position_m,
        &request.bearing_constraints,
        &request.amplitude_constraints,
        sample_time_std_s,
    );
    if let Some(canopy) = &request.canopy_height_prior {
        rows.push(canopy.jacobian(position_m, sample_time_std_s));
    }
    stack_jacobian_rows(&rows)
}

fn array_aperture_m(sensor_positions: &SensorPositions) -> f64 {
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

fn root_mean_square(values: &DVector<f64>) -> f64 {
    (values.dot(values) / values.len().max(1) as f64).sqrt()
}

fn initial_positions(
    request: &CartesianTdoaSolveRequest,
    centroid_m: &Vector3<f64>,
    aperture_m: f64,
    minimum_position_std_m: f64,
) -> Result<Vec<Vector3<f64>>, TdoaSolverError> {
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
    Ok(starts)
}

/// SciPy `least_squares` settings used by the Python solver: first pass…
pub const SCIPY_PRIMARY_TRF_SETTINGS: TrfSettings = TrfSettings {
    ftol: 1.0e-9,
    xtol: 1.0e-9,
    gtol: 1.0e-9,
    max_function_evaluations: 100,
};
/// …and the retry from a non-converged first pass.
pub const SCIPY_RETRY_TRF_SETTINGS: TrfSettings = TrfSettings {
    ftol: 1.0e-11,
    xtol: 1.0e-11,
    gtol: 1.0e-11,
    max_function_evaluations: 500,
};

/// One multi-start refinement; `Some((Σr², x))` for an accepted candidate.
fn refine_from_start(
    model: &CartesianResidualModel<'_>,
    start: Vector3<f64>,
    engine: CartesianRefinementEngine,
    residual_scale: f64,
) -> Option<(f64, Vector3<f64>)> {
    let position = match engine {
        CartesianRefinementEngine::LevenbergMarquardt => {
            let outcome = refine_position_with_retry(model, start, residual_scale);
            outcome.converged.then_some(outcome.position_m)?
        }
        CartesianRefinementEngine::ScipyTrfParity => {
            let mut outcome = scipy_trf_least_squares(model, start, &SCIPY_PRIMARY_TRF_SETTINGS)?;
            if !outcome.success && outcome.position_m.iter().all(|value| value.is_finite()) {
                outcome =
                    scipy_trf_least_squares(model, outcome.position_m, &SCIPY_RETRY_TRF_SETTINGS)?;
            }
            (outcome.success && outcome.jacobian_finite).then_some(outcome.position_m)?
        }
    };
    let cost = residual_cost(model, &position);
    (cost.is_finite() && position.iter().all(|value| value.is_finite())).then_some((cost, position))
}

pub fn solve_cartesian_tdoa(
    request: &CartesianTdoaSolveRequest,
) -> Result<CartesianTdoaSolve, TdoaSolverError> {
    if request.sensor_positions.is_empty() {
        return Err(TdoaSolverError::EmptySystem("sensor_positions"));
    }
    validate_measurement_sensors(&request.sensor_positions, &request.measurements)?;
    let sensor_count = request.sensor_positions.len() as f64;
    let centroid_m = request
        .sensor_positions
        .values()
        .fold(Vector3::zeros(), |sum, position| sum + position)
        / sensor_count;
    let aperture_m = array_aperture_m(&request.sensor_positions);
    let interpolated_rate =
        request.sample_rate_hz as f64 * request.interpolation_factor.max(1) as f64;
    let sample_time_std_s = 1.0 / interpolated_rate.max(1.0);
    let minimum_position_std_m =
        (request.sound_speed_mps * sample_time_std_s).max(MINIMUM_POSITION_STD_FLOOR_M);
    let has_tdoa_measurements = !request.measurements.is_empty();

    let model = CartesianResidualModel {
        request,
        centroid_m,
        sample_time_std_s,
    };
    let starts = initial_positions(request, &centroid_m, aperture_m, minimum_position_std_m)?;
    let residual_scale = 1.0 / sample_time_std_s;
    let mut best: Option<(f64, Vector3<f64>)> = None;
    let consider = |start: Vector3<f64>, best: &mut Option<(f64, Vector3<f64>)>| {
        if let Some((cost, position)) =
            refine_from_start(&model, start, request.refinement_engine, residual_scale)
        {
            if best.map_or(true, |(best_cost, _)| cost < best_cost) {
                *best = Some((cost, position));
            }
        }
    };
    for start in starts {
        consider(start, &mut best);
    }
    if let (Some(canopy), Some((_, position))) = (&request.canopy_height_prior, best) {
        for probe in canopy.band_probe_starts(&position, centroid_m.z) {
            consider(probe, &mut best);
        }
    }
    let (_, mut best_position) = best.ok_or(TdoaSolverError::RefinementDidNotConverge)?;

    let mut residual_rms_s = root_mean_square(&model.residuals(&best_position));
    let mut effective_time_std_s = residual_rms_s.max(sample_time_std_s);
    let mut range_projection_mode: &'static str = RANGE_REFINED;
    let mut radial_refinement_std_m: Option<f64> = None;
    let preliminary_jacobian = observability_jacobian(request, &best_position, sample_time_std_s);
    let best_range_before_radial_refinement_m = (best_position - centroid_m).norm();
    let preliminary_radial_std_m = radial_standard_deviation(
        &best_position,
        &centroid_m,
        &preliminary_jacobian,
        effective_time_std_s,
    );
    let weak_cartesian_range = preliminary_radial_std_m
        >= best_range_before_radial_refinement_m
            .max(aperture_m)
            .max(minimum_position_std_m);
    let compact_array_far_field_candidate = best_range_before_radial_refinement_m
        >= aperture_m * COMPACT_ARRAY_RADIAL_REFINEMENT_APERTURE_MULTIPLIER;
    if has_tdoa_measurements
        && request.radial_refinement_enabled
        && (weak_cartesian_range || compact_array_far_field_candidate)
    {
        let best_offset = best_position - centroid_m;
        let best_offset_norm = best_offset.norm();
        let refinement = adaptive_radial_refinement(&RadialRefinementRequest {
            centroid_m,
            direction: best_offset / best_offset_norm.max(EPSILON),
            initial_radius_m: best_offset_norm
                .max(request.far_field_initial_range_m)
                .max(aperture_m)
                .max(minimum_position_std_m),
            minimum_radius_m: (aperture_m * RADIAL_SEARCH_MINIMUM_APERTURE_FRACTION)
                .max(minimum_position_std_m),
            residual_model: &model,
            effective_time_std_s,
            asymptotic_radial_std_fraction: Some(if request.bearing_prior.is_some() {
                PRIOR_DOMINATED_ASYMPTOTIC_RADIAL_STD_FRACTION
            } else {
                DEFAULT_ASYMPTOTIC_RADIAL_STD_FRACTION
            }),
        })?;
        if refinement.fit_residual_rms_seconds
            < residual_rms_s * (1.0 - RADIAL_REFINEMENT_MINIMUM_RELATIVE_IMPROVEMENT)
        {
            best_position = refinement.position_m;
            radial_refinement_std_m = Some(refinement.radial_std_m);
            range_projection_mode = refinement.projection_mode;
            residual_rms_s = root_mean_square(&model.residuals(&best_position));
            effective_time_std_s = residual_rms_s.max(sample_time_std_s);
        }
    }
    let final_jacobian = observability_jacobian(request, &best_position, sample_time_std_s);
    finalize_cartesian_solution(&CartesianFinalizationInputs {
        request,
        best_position_m: best_position,
        centroid_m,
        aperture_m,
        sample_time_std_s,
        minimum_position_std_m,
        residual_rms_s,
        effective_time_std_s,
        range_projection_mode,
        radial_refinement_std_m,
        preliminary_jacobian,
        final_jacobian,
    })
}
