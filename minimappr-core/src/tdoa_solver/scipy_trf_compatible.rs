//! Port of SciPy's unbounded Trust Region Reflective least squares
//! (`least_squares(method="trf", loss="soft_l1", jac="2-point")`, i.e.
//! `trf_no_bounds` + `solve_lsq_trust_region` + `approx_derivative`).
//!
//! Why this exists next to LM: the server's Python Cartesian solver feeds
//! residuals in *seconds* (~1e-6) to SciPy, whose `gtol` test is absolute, so
//! on compact-array (bearing-only) geometries it stops after 0–3 steps, often
//! at the start point. The downstream range-projection classification
//! (radial refinement adoption, asymptotic / bearing-projected modes) is
//! calibrated on that behaviour. Reproducing the iterates is the only way to
//! get 1e-5 parity there; a converged LM legitimately lands elsewhere (see
//! `docs/field_mode_progress/core_solver.md`).

use nalgebra::{DMatrix, DVector, Vector3};

use super::levenberg_marquardt_driver::PositionResidualModel;
use super::robust_linear_algebra::sorted_singular_value_decomposition;

const PARAMETER_COUNT: usize = 3;
/// SciPy's machine epsilon (`np.finfo(float).eps`).
const MACHINE_EPSILON: f64 = f64::EPSILON;
/// Trust-region root-finding: relative tolerance on `‖p‖ − Δ`, iteration cap.
const TRUST_REGION_ROOT_RELATIVE_TOLERANCE: f64 = 0.01;
const TRUST_REGION_ROOT_MAX_ITERATIONS: usize = 10;
/// Initial-alpha heuristic factor (`0.001 · α_upper`).
const TRUST_REGION_ALPHA_UPPER_FRACTION: f64 = 0.001;
/// Radius update thresholds and factors (`update_tr_radius`).
const RADIUS_SHRINK_RATIO_THRESHOLD: f64 = 0.25;
const RADIUS_GROW_RATIO_THRESHOLD: f64 = 0.75;
const RADIUS_SHRINK_FACTOR: f64 = 0.25;
const RADIUS_GROW_FACTOR: f64 = 2.0;
/// A step longer than this fraction of Δ counts as hitting the boundary.
const BOUNDARY_HIT_FRACTION: f64 = 0.95;

#[derive(Debug, Clone, PartialEq)]
pub struct TrfOutcome {
    pub position_m: Vector3<f64>,
    /// SciPy status: 0 = max_nfev, 1 = gtol, 2 = ftol, 3 = xtol, 4 = ftol+xtol.
    pub status: u8,
    pub success: bool,
    /// SciPy's candidate filter also requires a finite final Jacobian.
    pub jacobian_finite: bool,
    pub function_evaluations: usize,
    /// SciPy's `result.jac`: the robust-loss-scaled Jacobian at `position_m`
    /// (in the model's residual units), the input to Jacobian covariances.
    pub scaled_jacobian: DMatrix<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrfSettings {
    pub ftol: f64,
    pub xtol: f64,
    pub gtol: f64,
    pub max_function_evaluations: usize,
}

/// soft_l1 with `f_scale = 1`: returns `(Σ ρ₀ / 2, ρ₁, ρ₂)`.
pub(crate) fn soft_l1(residuals: &DVector<f64>) -> (f64, DVector<f64>, DVector<f64>) {
    let mut half_cost = 0.0;
    let mut first = DVector::zeros(residuals.len());
    let mut second = DVector::zeros(residuals.len());
    for (index, residual) in residuals.iter().enumerate() {
        let t = 1.0 + residual * residual;
        half_cost += 2.0 * (t.sqrt() - 1.0);
        first[index] = 1.0 / t.sqrt();
        second[index] = -0.5 * t.powf(-1.5);
    }
    (0.5 * half_cost, first, second)
}

pub(crate) fn soft_l1_cost(residuals: &DVector<f64>) -> f64 {
    0.5 * residuals
        .iter()
        .map(|residual| 2.0 * ((1.0 + residual * residual).sqrt() - 1.0))
        .sum::<f64>()
}

/// `scale_for_robust_loss_function`: row-scale J and rescale f.
pub(crate) fn robust_scaled_system(
    jacobian: &DMatrix<f64>,
    residuals: &DVector<f64>,
) -> (DMatrix<f64>, DVector<f64>) {
    let (_, first, second) = soft_l1(residuals);
    let mut scaled_jacobian = jacobian.clone();
    let mut scaled_residuals = residuals.clone();
    for row in 0..residuals.len() {
        let mut row_scale = first[row] + 2.0 * second[row] * residuals[row] * residuals[row];
        if row_scale < MACHINE_EPSILON {
            row_scale = MACHINE_EPSILON;
        }
        row_scale = row_scale.sqrt();
        scaled_residuals[row] *= first[row] / row_scale;
        scaled_jacobian.row_mut(row).scale_mut(row_scale);
    }
    (scaled_jacobian, scaled_residuals)
}

/// SciPy `approx_derivative(method="2-point", rel_step=None)`.
fn forward_difference_jacobian(
    model: &dyn PositionResidualModel,
    position: &Vector3<f64>,
    residuals_at_position: &DVector<f64>,
) -> DMatrix<f64> {
    let relative_step = MACHINE_EPSILON.sqrt();
    let mut jacobian = DMatrix::zeros(residuals_at_position.len(), PARAMETER_COUNT);
    for axis in 0..PARAMETER_COUNT {
        let sign = if position[axis] >= 0.0 { 1.0 } else { -1.0 };
        let step = relative_step * sign * position[axis].abs().max(1.0);
        let mut stepped = *position;
        stepped[axis] = position[axis] + step;
        let actual_step = stepped[axis] - position[axis];
        let column = (model.residuals(&stepped) - residuals_at_position) / actual_step;
        jacobian.column_mut(axis).copy_from(&column);
    }
    jacobian
}

/// `solve_lsq_trust_region`: returns `(step, alpha)`.
pub(crate) fn solve_trust_region_step(
    residual_count: usize,
    projected_residuals: &DVector<f64>,
    singular_values: &DVector<f64>,
    right_vectors: &DMatrix<f64>,
    radius: f64,
    initial_alpha: f64,
) -> (DVector<f64>, f64) {
    let scaled = singular_values.component_mul(projected_residuals);
    let phi_and_derivative = |alpha: f64| -> (f64, f64) {
        let denominator = singular_values.map(|value| value * value + alpha);
        let p_norm = scaled.component_div(&denominator).norm();
        let derivative = -scaled
            .iter()
            .zip(denominator.iter())
            .map(|(value, denom)| value * value / denom.powi(3))
            .sum::<f64>()
            / p_norm;
        (p_norm - radius, derivative)
    };
    let full_rank =
        residual_count >= PARAMETER_COUNT && singular_values.len() == PARAMETER_COUNT && {
            let threshold = MACHINE_EPSILON * residual_count as f64 * singular_values[0];
            singular_values[singular_values.len() - 1] > threshold
        };
    if full_rank {
        let gauss_newton = -(right_vectors * projected_residuals.component_div(singular_values));
        if gauss_newton.norm() <= radius {
            return (gauss_newton, 0.0);
        }
    }
    let mut alpha_upper = scaled.norm() / radius;
    let mut alpha_lower = if full_rank {
        let (phi, derivative) = phi_and_derivative(0.0);
        -phi / derivative
    } else {
        0.0
    };
    let heuristic = |lower: f64, upper: f64| {
        (TRUST_REGION_ALPHA_UPPER_FRACTION * upper).max((lower * upper).sqrt())
    };
    // Python: `initial_alpha is None or not full_rank and initial_alpha == 0`.
    let mut alpha = if !full_rank && initial_alpha == 0.0 {
        heuristic(alpha_lower, alpha_upper)
    } else {
        initial_alpha
    };
    for _ in 0..TRUST_REGION_ROOT_MAX_ITERATIONS {
        if alpha < alpha_lower || alpha > alpha_upper {
            alpha = heuristic(alpha_lower, alpha_upper);
        }
        let (phi, derivative) = phi_and_derivative(alpha);
        if phi < 0.0 {
            alpha_upper = alpha;
        }
        let ratio = phi / derivative;
        alpha_lower = alpha_lower.max(alpha - ratio);
        alpha -= (phi + radius) * ratio / radius;
        if phi.abs() < TRUST_REGION_ROOT_RELATIVE_TOLERANCE * radius {
            break;
        }
    }
    let denominator = singular_values.map(|value| value * value + alpha);
    let mut step = -(right_vectors * scaled.component_div(&denominator));
    let step_norm = step.norm();
    step *= radius / step_norm;
    (step, alpha)
}

pub(crate) fn update_trust_radius(
    radius: f64,
    actual: f64,
    predicted: f64,
    step_norm: f64,
    bound_hit: bool,
) -> (f64, f64) {
    let ratio = if predicted > 0.0 {
        actual / predicted
    } else if predicted == 0.0 && actual == 0.0 {
        1.0
    } else {
        0.0
    };
    let new_radius = if ratio < RADIUS_SHRINK_RATIO_THRESHOLD {
        RADIUS_SHRINK_FACTOR * step_norm
    } else if ratio > RADIUS_GROW_RATIO_THRESHOLD && bound_hit {
        radius * RADIUS_GROW_FACTOR
    } else {
        radius
    };
    (new_radius, ratio)
}

pub(crate) fn check_termination(
    actual: f64,
    cost: f64,
    step_norm: f64,
    position_norm: f64,
    ratio: f64,
    settings: &TrfSettings,
) -> Option<u8> {
    let ftol_satisfied = actual < settings.ftol * cost && ratio > RADIUS_SHRINK_RATIO_THRESHOLD;
    let xtol_satisfied = step_norm < settings.xtol * (settings.xtol + position_norm);
    match (ftol_satisfied, xtol_satisfied) {
        (true, true) => Some(4),
        (true, false) => Some(2),
        (false, true) => Some(3),
        (false, false) => None,
    }
}

fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}

/// Run SciPy-compatible TRF from `initial_position`. `None` where SciPy would
/// raise (non-finite initial residuals, SVD failure).
pub fn scipy_trf_least_squares(
    model: &dyn PositionResidualModel,
    initial_position: Vector3<f64>,
    settings: &TrfSettings,
) -> Option<TrfOutcome> {
    let mut position = initial_position;
    let mut residuals = model.residuals(&position);
    if !all_finite(residuals.as_slice()) {
        return None;
    }
    let residual_count = residuals.len();
    let mut function_evaluations = 1_usize;
    let mut cost = soft_l1(&residuals).0;
    let (mut jacobian, mut scaled_residuals) = robust_scaled_system(
        &forward_difference_jacobian(model, &position, &residuals),
        &residuals,
    );
    let mut gradient = jacobian.transpose() * &scaled_residuals;
    let mut radius = position.norm();
    if radius == 0.0 {
        radius = 1.0;
    }
    let mut alpha = 0.0_f64;
    let mut status: Option<u8> = None;
    loop {
        if gradient.amax() < settings.gtol {
            status = Some(1);
        }
        // `>=` (not SciPy's `==`) so a zero budget cannot spin forever.
        if status.is_some() || function_evaluations >= settings.max_function_evaluations {
            break;
        }
        let decomposition = sorted_singular_value_decomposition(&jacobian).ok()?;
        let right_vectors = decomposition.right_vectors_transposed.transpose();
        let projected = decomposition.left_vectors.transpose() * &scaled_residuals;
        let mut actual_reduction = -1.0_f64;
        let mut candidate = position;
        let mut candidate_residuals = residuals.clone();
        let mut candidate_cost = cost;
        while actual_reduction <= 0.0 && function_evaluations < settings.max_function_evaluations {
            let (step, new_alpha) = solve_trust_region_step(
                residual_count,
                &projected,
                &decomposition.singular_values,
                &right_vectors,
                radius,
                alpha,
            );
            alpha = new_alpha;
            let jacobian_step = &jacobian * &step;
            let predicted_reduction =
                -(0.5 * jacobian_step.dot(&jacobian_step) + step.dot(&gradient));
            candidate = position + Vector3::new(step[0], step[1], step[2]);
            candidate_residuals = model.residuals(&candidate);
            function_evaluations += 1;
            let step_norm = step.norm();
            if !all_finite(candidate_residuals.as_slice()) {
                radius = RADIUS_SHRINK_FACTOR * step_norm;
                continue;
            }
            candidate_cost = soft_l1_cost(&candidate_residuals);
            actual_reduction = cost - candidate_cost;
            let (new_radius, ratio) = update_trust_radius(
                radius,
                actual_reduction,
                predicted_reduction,
                step_norm,
                step_norm > BOUNDARY_HIT_FRACTION * radius,
            );
            status = check_termination(
                actual_reduction,
                cost,
                step_norm,
                position.norm(),
                ratio,
                settings,
            );
            if status.is_some() {
                break;
            }
            alpha *= radius / new_radius;
            radius = new_radius;
        }
        if actual_reduction > 0.0 {
            position = candidate;
            residuals = candidate_residuals;
            cost = candidate_cost;
            let raw_jacobian = forward_difference_jacobian(model, &position, &residuals);
            (jacobian, scaled_residuals) = robust_scaled_system(&raw_jacobian, &residuals);
            gradient = jacobian.transpose() * &scaled_residuals;
        }
    }
    let status = status.unwrap_or(0);
    Some(TrfOutcome {
        position_m: position,
        status,
        success: status > 0,
        jacobian_finite: all_finite(jacobian.as_slice()),
        function_evaluations,
        scaled_jacobian: jacobian,
    })
}
