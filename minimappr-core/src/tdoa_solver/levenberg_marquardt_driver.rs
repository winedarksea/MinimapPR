//! Adapter from a position-only residual model to the MINPACK-derived
//! `levenberg-marquardt` crate.
//!
//! Why the crate rather than a hand-rolled LM: its trust-region / damping
//! update is a faithful port of MINPACK `lmder` (the same algorithm behind
//! `scipy.optimize.least_squares(method="lm")`), it is pure Rust on top of
//! nalgebra, and it builds for wasm32 and aarch64-linux-android unchanged.

use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt, TerminationReason};
use nalgebra::storage::Owned;
use nalgebra::{DMatrix, DVector, Dyn, Vector3, U3};

/// A residual vector `r(x)` over a 3D position, with its `m×3` Jacobian.
/// The residual length must not depend on `x`.
pub trait PositionResidualModel {
    fn residuals(&self, position_m: &Vector3<f64>) -> DVector<f64>;
    fn jacobian(&self, position_m: &Vector3<f64>) -> DMatrix<f64>;
}

/// LM stopping rules (see `LevenbergMarquardt::with_*`). `patience` is the
/// crate's evaluation budget multiplier: max evaluations = patience · (n + 1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LevenbergMarquardtSettings {
    pub relative_cost_tolerance: f64,
    pub relative_step_tolerance: f64,
    pub gradient_orthogonality_tolerance: f64,
    pub evaluation_patience: usize,
}

/// Tight first pass. Tighter than the Python solver's 1e-9 TRF tolerances so
/// the Rust optimum is at least as converged as the oracle's.
pub const PRIMARY_REFINEMENT_SETTINGS: LevenbergMarquardtSettings = LevenbergMarquardtSettings {
    relative_cost_tolerance: 1.0e-12,
    relative_step_tolerance: 1.0e-12,
    gradient_orthogonality_tolerance: 1.0e-14,
    evaluation_patience: 100,
};

/// Retry from the first pass's end point with a larger budget (mirrors the
/// Python retry with `max_nfev=500`).
pub const RETRY_REFINEMENT_SETTINGS: LevenbergMarquardtSettings = LevenbergMarquardtSettings {
    relative_cost_tolerance: 1.0e-13,
    relative_step_tolerance: 1.0e-13,
    gradient_orthogonality_tolerance: 1.0e-15,
    evaluation_patience: 500,
};

/// Result of one refinement run.
#[derive(Debug, Clone, PartialEq)]
pub struct RefinementOutcome {
    pub position_m: Vector3<f64>,
    /// `Σ r²` of the *unscaled* model residuals at `position_m`.
    pub cost: f64,
    pub converged: bool,
}

struct ScaledPositionProblem<'model> {
    model: &'model dyn PositionResidualModel,
    position_m: Vector3<f64>,
    residual_scale: f64,
}

impl LeastSquaresProblem<f64, Dyn, U3> for ScaledPositionProblem<'_> {
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, U3>;
    type ParameterStorage = Owned<f64, U3>;

    fn set_params(&mut self, parameters: &Vector3<f64>) {
        self.position_m = *parameters;
    }

    fn params(&self) -> Vector3<f64> {
        self.position_m
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        let residuals = self.model.residuals(&self.position_m) * self.residual_scale;
        residuals
            .iter()
            .all(|value| value.is_finite())
            .then_some(residuals)
    }

    fn jacobian(&self) -> Option<nalgebra::OMatrix<f64, Dyn, U3>> {
        let jacobian = self.model.jacobian(&self.position_m) * self.residual_scale;
        if !jacobian.iter().all(|value| value.is_finite()) {
            return None;
        }
        let rows = jacobian.nrows();
        Some(nalgebra::OMatrix::<f64, Dyn, U3>::from_iterator_generic(
            Dyn(rows),
            U3,
            jacobian.iter().copied(),
        ))
    }
}

/// Sum of squared model residuals; `inf` when any residual is non-finite.
pub fn residual_cost(model: &dyn PositionResidualModel, position_m: &Vector3<f64>) -> f64 {
    let residuals = model.residuals(position_m);
    if residuals.iter().all(|value| value.is_finite()) {
        residuals.dot(&residuals)
    } else {
        f64::INFINITY
    }
}

/// Run LM from `initial_position_m`.
///
/// `residual_scale` multiplies residuals and Jacobian inside the optimiser
/// only; it does not move the minimiser but keeps MINPACK's absolute
/// thresholds meaningful when residuals are in seconds (~1e-5).
pub fn refine_position(
    model: &dyn PositionResidualModel,
    initial_position_m: Vector3<f64>,
    residual_scale: f64,
    settings: LevenbergMarquardtSettings,
) -> RefinementOutcome {
    let problem = ScaledPositionProblem {
        model,
        position_m: initial_position_m,
        residual_scale,
    };
    let optimiser = LevenbergMarquardt::new()
        .with_ftol(settings.relative_cost_tolerance)
        .with_xtol(settings.relative_step_tolerance)
        .with_gtol(settings.gradient_orthogonality_tolerance)
        .with_patience(settings.evaluation_patience)
        // All three parameters are metres, so MINPACK's column-norm scaling
        // buys nothing and is harmful: on a symmetry axis (collinear layout)
        // a ~1e-17 Jacobian column becomes a ~1e17 m step.
        .with_scale_diag(false);
    let (solved, report) = optimiser.minimize(problem);
    let position_m = solved.position_m;
    let cost = residual_cost(model, &position_m);
    let finite = position_m.iter().all(|value| value.is_finite()) && cost.is_finite();
    RefinementOutcome {
        position_m,
        cost,
        // NoImprovementPossible = converged to machine precision, which is
        // a success for us (the tolerances are deliberately very tight).
        converged: finite
            && (report.termination.was_successful()
                || matches!(
                    report.termination,
                    TerminationReason::NoImprovementPossible(_)
                )),
    }
}

/// Primary pass, then (if it ran out of budget but stayed finite) a retry
/// from its end point, as the Python solver does.
pub fn refine_position_with_retry(
    model: &dyn PositionResidualModel,
    initial_position_m: Vector3<f64>,
    residual_scale: f64,
) -> RefinementOutcome {
    let first = refine_position(
        model,
        initial_position_m,
        residual_scale,
        PRIMARY_REFINEMENT_SETTINGS,
    );
    if first.converged || !first.position_m.iter().all(|value| value.is_finite()) {
        return first;
    }
    refine_position(
        model,
        first.position_m,
        residual_scale,
        RETRY_REFINEMENT_SETTINGS,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DistanceToPoints {
        anchors: Vec<Vector3<f64>>,
        measured_ranges: Vec<f64>,
    }

    impl PositionResidualModel for DistanceToPoints {
        fn residuals(&self, position_m: &Vector3<f64>) -> DVector<f64> {
            DVector::from_iterator(
                self.anchors.len(),
                self.anchors
                    .iter()
                    .zip(&self.measured_ranges)
                    .map(|(anchor, range)| (position_m - anchor).norm() - range),
            )
        }
        fn jacobian(&self, position_m: &Vector3<f64>) -> DMatrix<f64> {
            let mut jacobian = DMatrix::zeros(self.anchors.len(), 3);
            for (row, anchor) in self.anchors.iter().enumerate() {
                let offset = position_m - anchor;
                let unit = offset / offset.norm();
                jacobian.row_mut(row).copy_from(&unit.transpose());
            }
            jacobian
        }
    }

    #[test]
    fn trilateration_converges_to_truth() {
        let truth = Vector3::new(3.0, -2.0, 1.5);
        let anchors = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(10.0, 0.0, 0.0),
            Vector3::new(0.0, 10.0, 0.0),
            Vector3::new(0.0, 0.0, 10.0),
        ];
        let measured_ranges = anchors
            .iter()
            .map(|anchor| (truth - anchor).norm())
            .collect();
        let model = DistanceToPoints {
            anchors,
            measured_ranges,
        };
        let outcome = refine_position_with_retry(&model, Vector3::new(1.0, 1.0, 1.0), 1.0);
        assert!(outcome.converged);
        assert!((outcome.position_m - truth).norm() < 1e-9);
    }
}
