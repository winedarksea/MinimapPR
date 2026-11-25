//! One-dimensional log-radius range refinement along a fixed bearing; port of
//! `minimappr/core/radial_range_refinement.py`. `maximum_radius_m` (the
//! far-field envelope) stops a cost that keeps falling from running the radius
//! to float saturation: the result is then asymptotic at the envelope.

use nalgebra::{DMatrix, Vector3};

use super::bounded_scalar_minimizer::minimize_bounded_scalar;
use super::levenberg_marquardt_driver::PositionResidualModel;
use super::TdoaSolverError;
use crate::range_projection::{RANGE_ASYMPTOTIC, RANGE_REFINED};

const EPSILON: f64 = 1.0e-12;
const MAX_RADIAL_SEARCH_EXPANSIONS: usize = 40;
const RADIAL_SEARCH_EXPANSION_FACTOR: f64 = 2.0;
/// Brent tolerance in log-radius and its evaluation cap (Python `xatol`, `maxiter`).
const LOG_RADIUS_ABSOLUTE_TOLERANCE: f64 = 1.0e-10;
const LOG_RADIUS_MAXIMUM_EVALUATIONS: usize = 200;
/// Finite-difference step for the radial derivative: relative, minimum-radius
/// relative, and absolute floor (metres).
const RADIAL_DERIVATIVE_RELATIVE_STEP: f64 = 1.0e-4;
const RADIAL_DERIVATIVE_MINIMUM_RADIUS_FRACTION: f64 = 0.1;
const RADIAL_DERIVATIVE_ABSOLUTE_STEP_M: f64 = 1.0e-3;

#[derive(Debug, Clone, PartialEq)]
pub struct RadialRefinement {
    pub position_m: Vector3<f64>,
    pub radial_std_m: f64,
    pub projection_mode: &'static str,
    pub fit_residual_rms_seconds: f64,
}

/// `σ_t / ‖J · r̂‖`: the Fisher radial standard deviation at `position_m`.
pub fn radial_standard_deviation(
    position_m: &Vector3<f64>,
    centroid_m: &Vector3<f64>,
    jacobian: &DMatrix<f64>,
    time_std_s: f64,
) -> f64 {
    let radial_offset = position_m - centroid_m;
    let radial_norm = radial_offset.norm();
    if radial_norm < EPSILON {
        return f64::INFINITY;
    }
    let radial_axis = radial_offset / radial_norm;
    let radial_axis_column = DMatrix::from_column_slice(3, 1, radial_axis.as_slice());
    let radial_information = (jacobian * radial_axis_column).norm();
    if radial_information < EPSILON {
        return f64::INFINITY;
    }
    time_std_s / radial_information
}

fn root_mean_square(values: &nalgebra::DVector<f64>) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    (values.dot(values) / values.len() as f64).sqrt()
}

/// Inputs to [`adaptive_radial_refinement`].
pub struct RadialRefinementRequest<'model> {
    pub centroid_m: Vector3<f64>,
    pub direction: Vector3<f64>,
    pub initial_radius_m: f64,
    pub minimum_radius_m: f64,
    pub residual_model: &'model dyn PositionResidualModel,
    pub effective_time_std_s: f64,
    pub asymptotic_radial_std_fraction: Option<f64>,
    /// `None`: unbounded search.
    pub maximum_radius_m: Option<f64>,
}

pub fn adaptive_radial_refinement(
    request: &RadialRefinementRequest<'_>,
) -> Result<RadialRefinement, TdoaSolverError> {
    let direction = request.direction / request.direction.norm().max(EPSILON);
    let centroid_m = request.centroid_m;
    let model = request.residual_model;
    let squared_cost_for_log_radius = |log_radius: f64| -> f64 {
        let residual = model.residuals(&(centroid_m + direction * log_radius.exp()));
        if residual.iter().all(|value| value.is_finite()) {
            residual.dot(&residual)
        } else {
            f64::INFINITY
        }
    };

    let lower_radius_m = request.minimum_radius_m.max(EPSILON);
    let center_radius_m = request
        .initial_radius_m
        .max(lower_radius_m * RADIAL_SEARCH_EXPANSION_FACTOR);
    let mut lower_log_radius = lower_radius_m.ln();
    let mut center_log_radius = center_radius_m.ln();
    let mut lower_cost = squared_cost_for_log_radius(lower_log_radius);
    let mut center_cost = squared_cost_for_log_radius(center_log_radius);
    // At least one expansion above the centre, so the bracket is never empty.
    let ceiling_radius_m = request.maximum_radius_m.map_or(f64::INFINITY, |maximum| {
        maximum.max(center_radius_m * RADIAL_SEARCH_EXPANSION_FACTOR)
    });
    let mut upper_radius_m = (center_radius_m * RADIAL_SEARCH_EXPANSION_FACTOR).min(ceiling_radius_m);
    let mut upper_log_radius = upper_radius_m.ln();
    let mut upper_cost = squared_cost_for_log_radius(upper_log_radius);

    let mut expansion_count = 0_usize;
    while upper_cost.is_finite()
        && upper_cost < center_cost
        && expansion_count < MAX_RADIAL_SEARCH_EXPANSIONS
    {
        if upper_radius_m >= ceiling_radius_m {
            // Still falling at the envelope: asymptotic there.
            break;
        }
        lower_log_radius = center_log_radius;
        lower_cost = center_cost;
        center_log_radius = upper_log_radius;
        center_cost = upper_cost;
        upper_radius_m = (upper_radius_m * RADIAL_SEARCH_EXPANSION_FACTOR).min(ceiling_radius_m);
        if !upper_radius_m.is_finite() {
            break;
        }
        upper_log_radius = upper_radius_m.ln();
        upper_cost = squared_cost_for_log_radius(upper_log_radius);
        expansion_count += 1;
    }

    let has_finite_interior_minimum = lower_cost.is_finite()
        && center_cost.is_finite()
        && upper_cost.is_finite()
        && center_cost <= lower_cost
        && center_cost <= upper_cost;
    let (solved_radius_m, mut projection_mode) = if has_finite_interior_minimum {
        let mut objective = squared_cost_for_log_radius;
        let optimized = minimize_bounded_scalar(
            &mut objective,
            lower_log_radius,
            upper_log_radius,
            LOG_RADIUS_ABSOLUTE_TOLERANCE,
            LOG_RADIUS_MAXIMUM_EVALUATIONS,
        );
        if optimized.success && optimized.value.is_finite() {
            let solved = optimized.argument.exp();
            let expanded_boundary_radius_m =
                center_radius_m / RADIAL_SEARCH_EXPANSION_FACTOR.sqrt();
            let mode = if expansion_count > 0 && solved >= expanded_boundary_radius_m {
                RANGE_ASYMPTOTIC
            } else {
                RANGE_REFINED
            };
            (solved, mode)
        } else {
            (center_radius_m, RANGE_ASYMPTOTIC)
        }
    } else {
        let farthest_finite_log_radius = [
            (lower_cost, lower_log_radius),
            (center_cost, center_log_radius),
            (upper_cost, upper_log_radius),
        ]
        .iter()
        .filter(|(cost, _)| cost.is_finite())
        .map(|(_, log_radius)| *log_radius)
        .fold(None, |best: Option<f64>, candidate| {
            Some(best.map_or(candidate, |current| current.max(candidate)))
        })
        .ok_or(TdoaSolverError::RadialRefinementNoFiniteCandidate)?;
        (farthest_finite_log_radius.exp(), RANGE_ASYMPTOTIC)
    };

    let fitted_residual = model.residuals(&(centroid_m + direction * solved_radius_m));
    let fit_residual_rms_seconds = root_mean_square(&fitted_residual);

    let radial_standard_deviation_at = |radius_m: f64| -> f64 {
        let derivative_delta_m = (radius_m * RADIAL_DERIVATIVE_RELATIVE_STEP)
            .max(request.minimum_radius_m * RADIAL_DERIVATIVE_MINIMUM_RADIUS_FRACTION)
            .max(RADIAL_DERIVATIVE_ABSOLUTE_STEP_M);
        let left_radius_m = (radius_m - derivative_delta_m).max(lower_radius_m);
        let right_radius_m = radius_m + derivative_delta_m;
        let left_residual = model.residuals(&(centroid_m + direction * left_radius_m));
        let right_residual = model.residuals(&(centroid_m + direction * right_radius_m));
        let derivative =
            (right_residual - left_residual) / (right_radius_m - left_radius_m).max(EPSILON);
        let radial_information = derivative.norm();
        if radial_information > EPSILON {
            request.effective_time_std_s / radial_information
        } else {
            radius_m * RADIAL_SEARCH_EXPANSION_FACTOR
        }
    };

    let fitted_radial_std_m = radial_standard_deviation_at(solved_radius_m);
    if let Some(fraction) = request.asymptotic_radial_std_fraction {
        if fitted_radial_std_m >= solved_radius_m * fraction {
            projection_mode = RANGE_ASYMPTOTIC;
        }
    }
    let published_radius_m = if projection_mode == RANGE_ASYMPTOTIC {
        solved_radius_m.max(request.initial_radius_m)
    } else {
        solved_radius_m
    };
    let mut radial_std_m = radial_standard_deviation_at(published_radius_m);
    if projection_mode == RANGE_ASYMPTOTIC {
        radial_std_m = radial_std_m.max(published_radius_m);
    }
    Ok(RadialRefinement {
        position_m: centroid_m + direction * published_radius_m,
        radial_std_m,
        projection_mode,
        fit_residual_rms_seconds,
    })
}
