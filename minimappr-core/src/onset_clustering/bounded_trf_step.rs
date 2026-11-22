//! Step selection for the bounded TRF port (SciPy `select_step` with its
//! quadratic-model helpers): the trust-region step, its reflection off the
//! first bound hit, or the scaled anti-gradient, whichever models lowest.
//! Split from `bounded_trf.rs`.

use nalgebra::{DMatrix, DVector};

use super::bounded_trf::{Bounded, PARAMETER_COUNT};

/// Quadratic model `0.5 (|J s|² + Σ diag s²) + g·s`.
fn evaluate_quadratic(
    jacobian: &DMatrix<f64>,
    gradient: &DVector<f64>,
    step: &DVector<f64>,
    diagonal: &DVector<f64>,
) -> f64 {
    let projected = jacobian * step;
    let quadratic = projected.dot(&projected) + step.component_mul(diagonal).dot(step);
    0.5 * quadratic + step.dot(gradient)
}

/// `build_quadratic_1d` along `direction` from `start` (a, b, c).
fn build_quadratic_1d(
    jacobian: &DMatrix<f64>,
    gradient: &DVector<f64>,
    direction: &DVector<f64>,
    start: Option<&DVector<f64>>,
    diagonal: &DVector<f64>,
) -> (f64, f64, f64) {
    let v = jacobian * direction;
    let a = 0.5 * (v.dot(&v) + direction.component_mul(diagonal).dot(direction));
    let mut b = gradient.dot(direction);
    let mut c = 0.0;
    if let Some(start) = start {
        let u = jacobian * start;
        b += u.dot(&v);
        c = 0.5 * u.dot(&u) + gradient.dot(start);
        b += start.component_mul(diagonal).dot(direction);
        c += 0.5 * start.component_mul(diagonal).dot(start);
    }
    (a, b, c)
}

/// `minimize_quadratic_1d` on [lower, upper]: (t, value).
fn minimize_quadratic_1d(a: f64, b: f64, lower: f64, upper: f64, c: f64) -> (f64, f64) {
    let mut candidates = vec![lower, upper];
    if a != 0.0 {
        let extremum = -0.5 * b / a;
        if lower < extremum && extremum < upper {
            candidates.push(extremum);
        }
    }
    let mut best = (candidates[0], candidates[0] * (a * candidates[0] + b) + c);
    for t in candidates.into_iter().skip(1) {
        let value = t * (a * t + b) + c;
        if value < best.1 {
            best = (t, value);
        }
    }
    best
}

/// `intersect_trust_region`: positive root of |x + s t| = Δ.
fn positive_trust_region_root(
    point: &DVector<f64>,
    direction: &DVector<f64>,
    radius: f64,
) -> Option<f64> {
    let a = direction.dot(direction);
    if a == 0.0 {
        return None;
    }
    let b = point.dot(direction);
    let c = point.dot(point) - radius * radius;
    if c > 0.0 {
        return None;
    }
    let d = (b * b - a * c).sqrt();
    let q = -(b + d.copysign(b));
    let (t1, t2) = (q / a, c / q);
    Some(t1.max(t2))
}

/// Inputs of one `select_step` call.
pub(super) struct StepContext<'a> {
    pub(super) position: &'a DVector<f64>,
    pub(super) jacobian_hat: &'a DMatrix<f64>,
    pub(super) diagonal_hat: &'a DVector<f64>,
    pub(super) gradient_hat: &'a DVector<f64>,
    pub(super) scaling: &'a DVector<f64>,
    pub(super) radius: f64,
    pub(super) theta: f64,
}

/// `select_step`: (step, step_hat, predicted_reduction).
pub(super) fn select_step(
    bounded: &Bounded<'_>,
    context: &StepContext<'_>,
    mut step: DVector<f64>,
    mut step_hat: DVector<f64>,
) -> Option<(DVector<f64>, DVector<f64>, f64)> {
    let StepContext {
        position,
        jacobian_hat,
        diagonal_hat,
        gradient_hat,
        scaling,
        radius,
        theta,
    } = *context;
    if bounded.in_bounds(&(position + &step)) {
        let value = evaluate_quadratic(jacobian_hat, gradient_hat, &step_hat, diagonal_hat);
        return Some((step, step_hat, -value));
    }
    let (step_stride, hits) = bounded.step_size_to_bound(position, &step);
    let mut reflected_hat = step_hat.clone();
    for axis in 0..PARAMETER_COUNT {
        if hits[axis] != 0.0 {
            reflected_hat[axis] *= -1.0;
        }
    }
    let reflected = scaling.component_mul(&reflected_hat);
    step *= step_stride;
    step_hat *= step_stride;
    let on_bound = position + &step;
    let to_trust_region = positive_trust_region_root(&step_hat, &reflected_hat, radius)?;
    let (to_bound, _) = bounded.step_size_to_bound(&on_bound, &reflected);
    let reflected_stride = to_bound.min(to_trust_region);
    let (stride_lower, stride_upper) = if reflected_stride > 0.0 {
        let lower = (1.0 - theta) * step_stride / reflected_stride;
        let upper = if reflected_stride == to_bound {
            theta * to_bound
        } else {
            to_trust_region
        };
        (lower, upper)
    } else {
        (0.0, -1.0)
    };
    let (reflected_step, reflected_hat_step, reflected_value) = if stride_lower <= stride_upper {
        let (a, b, c) = build_quadratic_1d(
            jacobian_hat,
            gradient_hat,
            &reflected_hat,
            Some(&step_hat),
            diagonal_hat,
        );
        let (stride, value) = minimize_quadratic_1d(a, b, stride_lower, stride_upper, c);
        let hat = &reflected_hat * stride + &step_hat;
        (scaling.component_mul(&hat), hat, value)
    } else {
        (reflected.clone(), reflected_hat.clone(), f64::INFINITY)
    };
    step *= theta;
    step_hat *= theta;
    let step_value = evaluate_quadratic(jacobian_hat, gradient_hat, &step_hat, diagonal_hat);
    let anti_gradient_hat = -gradient_hat;
    let anti_gradient = scaling.component_mul(&anti_gradient_hat);
    let to_trust_region = radius / anti_gradient_hat.norm();
    let (to_bound, _) = bounded.step_size_to_bound(position, &anti_gradient);
    let anti_stride = if to_bound < to_trust_region {
        theta * to_bound
    } else {
        to_trust_region
    };
    let (a, b, _) = build_quadratic_1d(
        jacobian_hat,
        gradient_hat,
        &anti_gradient_hat,
        None,
        diagonal_hat,
    );
    let (anti_stride, anti_value) = minimize_quadratic_1d(a, b, 0.0, anti_stride, 0.0);
    if step_value < reflected_value && step_value < anti_value {
        Some((step, step_hat, -step_value))
    } else if reflected_value < step_value && reflected_value < anti_value {
        Some((reflected_step, reflected_hat_step, -reflected_value))
    } else {
        Some((
            anti_gradient * anti_stride,
            anti_gradient_hat * anti_stride,
            -anti_value,
        ))
    }
}
