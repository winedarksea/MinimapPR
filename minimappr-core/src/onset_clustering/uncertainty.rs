//! Jacobian covariance and range observability (port of the pieces of
//! `minimappr/core/localization_uncertainty.py` the onset path uses).

use nalgebra::{DMatrix, Matrix3, Vector3};

use crate::tdoa_solver::bearing_fusion::stabilize_covariance;
use crate::tdoa_solver::robust_linear_algebra::ascending_symmetric_eigen;

/// `np.linalg.pinv` default relative cutoff (NumPy 2: `rcond = 1e-15`).
pub const PSEUDO_INVERSE_RELATIVE_CUTOFF: f64 = 1e-15;
/// Python floors the timing std at 1 us in `covariance_from_jacobian`.
pub const ABSOLUTE_MINIMUM_TIME_STD_S: f64 = 1.0e-6;
/// Variance floor in the observability ratio.
const OBSERVABILITY_EPSILON: f64 = 1e-9;

pub fn matrix_from_rows(rows: &[[f64; 3]; 3]) -> Matrix3<f64> {
    Matrix3::from_fn(|row, column| rows[row][column])
}

pub fn rows_from_matrix(matrix: &Matrix3<f64>) -> [[f64; 3]; 3] {
    let mut rows = [[0.0; 3]; 3];
    for (row, values) in rows.iter_mut().enumerate() {
        for (column, value) in values.iter_mut().enumerate() {
            *value = matrix[(row, column)];
        }
    }
    rows
}

fn symmetric_pseudo_inverse(matrix: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    let (eigenvalues, eigenvectors) = ascending_symmetric_eigen(matrix).ok()?;
    let largest = eigenvalues
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let cutoff = PSEUDO_INVERSE_RELATIVE_CUTOFF * largest;
    let inverted = eigenvalues.map(|value| {
        if value.abs() > cutoff {
            1.0 / value
        } else {
            0.0
        }
    });
    Some(eigenvectors * Matrix3::from_diagonal(&inverted) * eigenvectors.transpose())
}

/// `pinv(JᵀJ) · max(rms, min_time_std, 1 us)²`, eigenvalues floored at `minimum_std_m²`.
pub fn covariance_from_jacobian(
    jacobian: &DMatrix<f64>,
    residual_rms_seconds: f64,
    minimum_time_std_s: f64,
    minimum_std_m: f64,
) -> Option<Matrix3<f64>> {
    if jacobian.is_empty() || !jacobian.iter().all(|value| value.is_finite()) {
        return None;
    }
    let information = jacobian.transpose() * jacobian;
    let information = Matrix3::from_fn(|row, column| information[(row, column)]);
    let covariance = symmetric_pseudo_inverse(&information)?;
    if !covariance.iter().all(|value| value.is_finite()) {
        return None;
    }
    let time_std_s = residual_rms_seconds
        .max(minimum_time_std_s)
        .max(ABSOLUTE_MINIMUM_TIME_STD_S);
    stabilize_covariance(&(covariance * (time_std_s * time_std_s)), minimum_std_m)
}

/// sqrt(lateral / radial) variance ratio along `direction` (or min/max eigen ratio).
pub fn range_observability_from_covariance(
    covariance: &Matrix3<f64>,
    direction: Option<Vector3<f64>>,
) -> Option<f64> {
    if !covariance.iter().all(|value| value.is_finite()) {
        return None;
    }
    if let Some(direction) = direction {
        let length = direction.norm();
        if length >= OBSERVABILITY_EPSILON {
            let radial_axis = direction / length;
            let radial_variance = (radial_axis.transpose() * covariance * radial_axis)[(0, 0)];
            let lateral_variance =
                ((covariance.trace() - radial_variance) / 2.0).max(OBSERVABILITY_EPSILON);
            let ratio = lateral_variance
                / radial_variance
                    .max(lateral_variance)
                    .max(OBSERVABILITY_EPSILON);
            return Some(ratio.sqrt().clamp(0.0, 1.0));
        }
    }
    let (eigenvalues, _) = ascending_symmetric_eigen(covariance).ok()?;
    let minimum = eigenvalues[0].max(OBSERVABILITY_EPSILON);
    let maximum = eigenvalues[2].max(minimum);
    Some((minimum / maximum).sqrt().clamp(0.0, 1.0))
}
