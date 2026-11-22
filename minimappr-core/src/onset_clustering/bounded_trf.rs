//! SciPy-compatible bounded Trust Region Reflective least squares
//! (`least_squares(method="trf", loss="soft_l1", bounds=...)`, i.e.
//! `trf_bounds` + `select_step` + Coleman-Li scaling + bound-aware 2-point
//! Jacobian), for the onset refine's ground..canopy z slab.
//!
//! Why: non-planar layouts bound the multi-node LM's z (S-fixes onset). A
//! clamp-after-unbounded-solve would land elsewhere than the reference; the
//! bounded iterates are reproduced instead so decisions keep parity.
//!
//! Residuals arrive pre-divided by SciPy's `f_scale` (see `refinement.rs`), so
//! every quantity is uniformly rescaled except `theta = max(0.995, 1 - |g|)`,
//! which reads the gradient in original units: `gradient_unit_scale` (=
//! `f_scale²`) converts it back.

use nalgebra::{DMatrix, DVector, Vector3};

use super::bounded_trf_step::{select_step, StepContext};
use crate::tdoa_solver::levenberg_marquardt_driver::PositionResidualModel;
use crate::tdoa_solver::robust_linear_algebra::sorted_singular_value_decomposition;
use crate::tdoa_solver::scipy_trf_compatible::{
    check_termination, robust_scaled_system, soft_l1_cost, solve_trust_region_step,
    update_trust_radius, TrfOutcome, TrfSettings,
};

pub(super) const PARAMETER_COUNT: usize = 3;
const MACHINE_EPSILON: f64 = f64::EPSILON;
/// `theta` floor: how far a step backs off from a bound.
const MINIMUM_THETA: f64 = 0.995;
const RADIUS_SHRINK_FACTOR: f64 = 0.25;
const BOUNDARY_HIT_FRACTION: f64 = 0.95;

/// Bounds on z only (x, y unbounded).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VerticalBounds {
    pub lower_z_m: f64,
    pub upper_z_m: f64,
}

impl VerticalBounds {
    fn lower(&self) -> [f64; PARAMETER_COUNT] {
        [f64::NEG_INFINITY, f64::NEG_INFINITY, self.lower_z_m]
    }

    fn upper(&self) -> [f64; PARAMETER_COUNT] {
        [f64::INFINITY, f64::INFINITY, self.upper_z_m]
    }
}

pub(super) struct Bounded<'a> {
    model: &'a dyn PositionResidualModel,
    lower: [f64; PARAMETER_COUNT],
    upper: [f64; PARAMETER_COUNT],
}

impl Bounded<'_> {
    /// `approx_derivative(method="2-point", bounds=...)`: a forward step that
    /// would leave the box flips sign (or shrinks to the farther bound).
    fn jacobian(&self, position: &Vector3<f64>, residuals: &DVector<f64>) -> DMatrix<f64> {
        let relative_step = MACHINE_EPSILON.sqrt();
        let mut jacobian = DMatrix::zeros(residuals.len(), PARAMETER_COUNT);
        for axis in 0..PARAMETER_COUNT {
            let sign = if position[axis] >= 0.0 { 1.0 } else { -1.0 };
            let mut step = relative_step * sign * position[axis].abs().max(1.0);
            let lower_distance = position[axis] - self.lower[axis];
            let upper_distance = self.upper[axis] - position[axis];
            let stepped_value = position[axis] + step;
            if self.lower[axis].is_finite() || self.upper[axis].is_finite() {
                let violated = stepped_value < self.lower[axis] || stepped_value > self.upper[axis];
                let fitting = step.abs() <= lower_distance.max(upper_distance);
                if violated && fitting {
                    step = -step;
                } else if !fitting {
                    step = if upper_distance >= lower_distance {
                        upper_distance
                    } else {
                        -lower_distance
                    };
                }
            }
            let mut stepped = *position;
            stepped[axis] = position[axis] + step;
            let actual_step = stepped[axis] - position[axis];
            let column = (self.model.residuals(&stepped) - residuals) / actual_step;
            jacobian.column_mut(axis).copy_from(&column);
        }
        jacobian
    }

    /// `CL_scaling_vector`: (v, dv).
    fn coleman_li_scaling(
        &self,
        position: &Vector3<f64>,
        gradient: &DVector<f64>,
    ) -> ([f64; 3], [f64; 3]) {
        let mut v = [1.0; PARAMETER_COUNT];
        let mut dv = [0.0; PARAMETER_COUNT];
        for axis in 0..PARAMETER_COUNT {
            if gradient[axis] < 0.0 && self.upper[axis].is_finite() {
                v[axis] = self.upper[axis] - position[axis];
                dv[axis] = -1.0;
            }
            if gradient[axis] > 0.0 && self.lower[axis].is_finite() {
                v[axis] = position[axis] - self.lower[axis];
                dv[axis] = 1.0;
            }
        }
        (v, dv)
    }

    pub(super) fn in_bounds(&self, point: &DVector<f64>) -> bool {
        (0..PARAMETER_COUNT)
            .all(|axis| point[axis] >= self.lower[axis] && point[axis] <= self.upper[axis])
    }

    /// `step_size_to_bound`: (step, hits).
    pub(super) fn step_size_to_bound(
        &self,
        point: &DVector<f64>,
        direction: &DVector<f64>,
    ) -> (f64, [f64; 3]) {
        let mut steps = [f64::INFINITY; PARAMETER_COUNT];
        for axis in 0..PARAMETER_COUNT {
            if direction[axis] != 0.0 {
                let to_lower = (self.lower[axis] - point[axis]) / direction[axis];
                let to_upper = (self.upper[axis] - point[axis]) / direction[axis];
                steps[axis] = to_lower.max(to_upper);
            }
        }
        let minimum = steps.iter().copied().fold(f64::INFINITY, f64::min);
        let mut hits = [0.0; PARAMETER_COUNT];
        for axis in 0..PARAMETER_COUNT {
            if steps[axis] == minimum {
                hits[axis] = direction[axis].signum() * f64::from(direction[axis] != 0.0);
            }
        }
        (minimum, hits)
    }

    /// `make_strictly_feasible(x, rstep=0)`.
    fn strictly_feasible(&self, point: &DVector<f64>) -> Vector3<f64> {
        let mut feasible = Vector3::new(point[0], point[1], point[2]);
        for axis in 0..PARAMETER_COUNT {
            if feasible[axis] <= self.lower[axis] {
                feasible[axis] = next_toward(self.lower[axis], self.upper[axis]);
            } else if feasible[axis] >= self.upper[axis] {
                feasible[axis] = next_toward(self.upper[axis], self.lower[axis]);
            }
            if feasible[axis] < self.lower[axis] || feasible[axis] > self.upper[axis] {
                feasible[axis] = 0.5 * (self.lower[axis] + self.upper[axis]);
            }
        }
        feasible
    }
}

/// `np.nextafter(from, toward)` for finite `from`.
fn next_toward(from: f64, toward: f64) -> f64 {
    if from == toward || !from.is_finite() {
        return from;
    }
    let bits = from.to_bits() as i64;
    let up = toward > from;
    let next = if from == 0.0 {
        if up {
            f64::from_bits(1)
        } else {
            -f64::from_bits(1)
        }
    } else if (from > 0.0) == up {
        f64::from_bits((bits + 1) as u64)
    } else {
        f64::from_bits((bits - 1) as u64)
    };
    next
}

/// Bounded SciPy TRF from a strictly interior `initial_position`.
/// `gradient_unit_scale` multiplies the gradient norm back to original units (theta only).
pub fn bounded_trf_least_squares(
    model: &dyn PositionResidualModel,
    initial_position: Vector3<f64>,
    bounds: &VerticalBounds,
    settings: &TrfSettings,
    gradient_unit_scale: f64,
) -> Option<TrfOutcome> {
    let bounded = Bounded {
        model,
        lower: bounds.lower(),
        upper: bounds.upper(),
    };
    let mut position = initial_position;
    let mut residuals = model.residuals(&position);
    if !residuals.iter().all(|value| value.is_finite()) {
        return None;
    }
    let residual_count = residuals.len();
    let mut function_evaluations = 1_usize;
    let mut cost = soft_l1_cost(&residuals);
    let (mut jacobian, mut scaled_residuals) =
        robust_scaled_system(&bounded.jacobian(&position, &residuals), &residuals);
    let mut gradient = jacobian.transpose() * &scaled_residuals;
    let (v0, _) = bounded.coleman_li_scaling(&position, &gradient);
    let mut radius = (0..PARAMETER_COUNT)
        .map(|axis| (position[axis] / v0[axis].sqrt()).powi(2))
        .sum::<f64>()
        .sqrt();
    if radius == 0.0 {
        radius = 1.0;
    }
    let mut alpha = 0.0_f64;
    let mut status: Option<u8> = None;
    loop {
        let (v, dv) = bounded.coleman_li_scaling(&position, &gradient);
        let gradient_norm = (0..PARAMETER_COUNT)
            .map(|axis| (gradient[axis] * v[axis]).abs())
            .fold(0.0, f64::max);
        if gradient_norm < settings.gtol {
            status = Some(1);
        }
        if status.is_some() || function_evaluations >= settings.max_function_evaluations {
            break;
        }
        let scaling = DVector::from_iterator(PARAMETER_COUNT, v.iter().map(|value| value.sqrt()));
        let diagonal_hat = DVector::from_iterator(
            PARAMETER_COUNT,
            (0..PARAMETER_COUNT).map(|axis| gradient[axis] * dv[axis]),
        );
        let gradient_hat = scaling.component_mul(&gradient);
        let mut jacobian_hat = jacobian.clone();
        for axis in 0..PARAMETER_COUNT {
            jacobian_hat.column_mut(axis).scale_mut(scaling[axis]);
        }
        let mut augmented = DMatrix::zeros(residual_count + PARAMETER_COUNT, PARAMETER_COUNT);
        augmented
            .rows_mut(0, residual_count)
            .copy_from(&jacobian_hat);
        for axis in 0..PARAMETER_COUNT {
            augmented[(residual_count + axis, axis)] = diagonal_hat[axis].sqrt();
        }
        let mut augmented_residuals = DVector::zeros(residual_count + PARAMETER_COUNT);
        augmented_residuals
            .rows_mut(0, residual_count)
            .copy_from(&scaled_residuals);
        let decomposition = sorted_singular_value_decomposition(&augmented).ok()?;
        let right_vectors = decomposition.right_vectors_transposed.transpose();
        let projected = decomposition.left_vectors.transpose() * &augmented_residuals;
        let theta = MINIMUM_THETA.max(1.0 - gradient_norm * gradient_unit_scale);
        let position_vector = DVector::from_column_slice(position.as_slice());
        let mut actual_reduction = -1.0_f64;
        let mut candidate = position;
        let mut candidate_residuals = residuals.clone();
        let mut candidate_cost = cost;
        while actual_reduction <= 0.0 && function_evaluations < settings.max_function_evaluations {
            let (step_hat, new_alpha) = solve_trust_region_step(
                residual_count,
                &projected,
                &decomposition.singular_values,
                &right_vectors,
                radius,
                alpha,
            );
            alpha = new_alpha;
            let step = scaling.component_mul(&step_hat);
            let context = StepContext {
                position: &position_vector,
                jacobian_hat: &jacobian_hat,
                diagonal_hat: &diagonal_hat,
                gradient_hat: &gradient_hat,
                scaling: &scaling,
                radius,
                theta,
            };
            let (step, step_hat, predicted_reduction) =
                select_step(&bounded, &context, step, step_hat)?;
            candidate = bounded.strictly_feasible(&(&position_vector + &step));
            candidate_residuals = model.residuals(&candidate);
            function_evaluations += 1;
            let step_hat_norm = step_hat.norm();
            if !candidate_residuals.iter().all(|value| value.is_finite()) {
                radius = RADIUS_SHRINK_FACTOR * step_hat_norm;
                continue;
            }
            candidate_cost = soft_l1_cost(&candidate_residuals);
            actual_reduction = cost - candidate_cost;
            let (new_radius, ratio) = update_trust_radius(
                radius,
                actual_reduction,
                predicted_reduction,
                step_hat_norm,
                step_hat_norm > BOUNDARY_HIT_FRACTION * radius,
            );
            status = check_termination(
                actual_reduction,
                cost,
                step.norm(),
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
            (jacobian, scaled_residuals) =
                robust_scaled_system(&bounded.jacobian(&position, &residuals), &residuals);
            gradient = jacobian.transpose() * &scaled_residuals;
        }
    }
    let status = status.unwrap_or(0);
    Some(TrfOutcome {
        position_m: position,
        status,
        success: status > 0,
        jacobian_finite: jacobian.iter().all(|value| value.is_finite()),
        function_evaluations,
        scaled_jacobian: jacobian,
    })
}
