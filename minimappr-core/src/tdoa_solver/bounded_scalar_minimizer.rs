//! Brent's bounded scalar minimiser, a line-for-line port of SciPy's
//! `_minimize_scalar_bounded` (`minimize_scalar(method="bounded")`).
//!
//! Ported rather than replaced so the radial range refinement visits the same
//! points as the Python oracle; on flat far-field cost curves a different 1D
//! method could settle elsewhere inside the tolerance band.

/// SciPy hard-codes `sqrt(2.2e-16)` (not `sqrt(f64::EPSILON)`).
const SCIPY_SQRT_EPSILON_INPUT: f64 = 2.2e-16;
/// Golden-section fraction `(3 − √5) / 2`.
const GOLDEN_SECTION_FRACTION_NUMERATOR: f64 = 3.0;
const GOLDEN_SECTION_SQRT_ARGUMENT: f64 = 5.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundedMinimum {
    pub argument: f64,
    pub value: f64,
    pub evaluation_count: usize,
    /// False when the evaluation cap was hit or a NaN appeared.
    pub success: bool,
}

fn sign_with_zero_as_positive(value: f64) -> f64 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        1.0
    }
}

/// Minimise `function` on `[lower, upper]` to absolute tolerance `absolute_tolerance`.
pub fn minimize_bounded_scalar(
    function: &mut dyn FnMut(f64) -> f64,
    lower: f64,
    upper: f64,
    absolute_tolerance: f64,
    maximum_evaluations: usize,
) -> BoundedMinimum {
    let sqrt_epsilon = SCIPY_SQRT_EPSILON_INPUT.sqrt();
    let golden_mean =
        0.5 * (GOLDEN_SECTION_FRACTION_NUMERATOR - GOLDEN_SECTION_SQRT_ARGUMENT.sqrt());
    let (mut a, mut b) = (lower, upper);
    let mut fulc = a + golden_mean * (b - a);
    let mut nfc = fulc;
    let mut xf = fulc;
    let mut rat = 0.0_f64;
    let mut e = 0.0_f64;
    let mut fx = function(xf);
    let mut evaluation_count = 1_usize;
    let mut fu = f64::INFINITY;
    let mut ffulc = fx;
    let mut fnfc = fx;
    let mut xm = 0.5 * (a + b);
    let mut tol1 = sqrt_epsilon * xf.abs() + absolute_tolerance / 3.0;
    let mut tol2 = 2.0 * tol1;
    let mut success = true;

    while (xf - xm).abs() > (tol2 - 0.5 * (b - a)) {
        let mut golden = true;
        if e.abs() > tol1 {
            golden = false;
            let mut r = (xf - nfc) * (fx - ffulc);
            let mut q = (xf - fulc) * (fx - fnfc);
            let mut p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            r = e;
            e = rat;
            if p.abs() < (0.5 * q * r).abs() && p > q * (a - xf) && p < q * (b - xf) {
                rat = (p + 0.0) / q;
                let x = xf + rat;
                if (x - a) < tol2 || (b - x) < tol2 {
                    let si = sign_with_zero_as_positive(xm - xf);
                    rat = tol1 * si;
                }
            } else {
                golden = true;
            }
        }
        if golden {
            e = if xf >= xm { a - xf } else { b - xf };
            rat = golden_mean * e;
        }
        let si = sign_with_zero_as_positive(rat);
        let x = xf + si * rat.abs().max(tol1);
        fu = function(x);
        evaluation_count += 1;

        if fu <= fx {
            if x >= xf {
                a = xf;
            } else {
                b = xf;
            }
            fulc = nfc;
            ffulc = fnfc;
            nfc = xf;
            fnfc = fx;
            xf = x;
            fx = fu;
        } else {
            if x < xf {
                a = x;
            } else {
                b = x;
            }
            if fu <= fnfc || nfc == xf {
                fulc = nfc;
                ffulc = fnfc;
                nfc = x;
                fnfc = fu;
            } else if fu <= ffulc || fulc == xf || fulc == nfc {
                fulc = x;
                ffulc = fu;
            }
        }
        xm = 0.5 * (a + b);
        tol1 = sqrt_epsilon * xf.abs() + absolute_tolerance / 3.0;
        tol2 = 2.0 * tol1;
        if evaluation_count >= maximum_evaluations {
            success = false;
            break;
        }
    }
    if xf.is_nan() || fx.is_nan() || fu.is_nan() {
        success = false;
    }
    BoundedMinimum {
        argument: xf,
        value: fx,
        evaluation_count,
        success,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_parabola_minimum_inside_bounds() {
        let mut parabola = |x: f64| (x - 1.234).powi(2) + 0.5;
        let minimum = minimize_bounded_scalar(&mut parabola, -3.0, 4.0, 1e-10, 200);
        assert!(minimum.success);
        assert!((minimum.argument - 1.234).abs() < 1e-7);
    }

    #[test]
    fn monotone_function_settles_at_boundary() {
        let mut ramp = |x: f64| x;
        let minimum = minimize_bounded_scalar(&mut ramp, 2.0, 5.0, 1e-10, 200);
        assert!((minimum.argument - 2.0).abs() < 1e-6);
    }
}
