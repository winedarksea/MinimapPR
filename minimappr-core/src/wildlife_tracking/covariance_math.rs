//! Small dense covariance helpers shared by the perched tracker and the bird
//! entity registry. They mirror the numpy calls of the Python reference
//! (`np.linalg.solve`, `np.linalg.eigh` with an eigenvalue floor) so the
//! decisions built on them agree with the server.

use nalgebra::{Matrix2, Matrix3, SymmetricEigen, Vector2, Vector3};

use super::{Covariance3, Position3};

pub fn matrix_from_covariance(covariance_m2: &Covariance3) -> Matrix3<f64> {
    Matrix3::from_fn(|row, column| covariance_m2[row][column])
}

pub fn covariance_from_matrix(matrix: &Matrix3<f64>) -> Covariance3 {
    let mut covariance = [[0.0; 3]; 3];
    for (row, values) in covariance.iter_mut().enumerate() {
        for (column, value) in values.iter_mut().enumerate() {
            *value = matrix[(row, column)];
        }
    }
    covariance
}

pub fn vector_from_position(position_m: &Position3) -> Vector3<f64> {
    Vector3::new(position_m[0], position_m[1], position_m[2])
}

pub fn position_from_vector(vector: &Vector3<f64>) -> Position3 {
    [vector[0], vector[1], vector[2]]
}

pub fn symmetrized(matrix: &Matrix3<f64>) -> Matrix3<f64> {
    (matrix + matrix.transpose()) * 0.5
}

pub fn all_finite(covariance_m2: &Covariance3) -> bool {
    covariance_m2.iter().flatten().all(|value| value.is_finite())
}

/// Symmetrized covariance, or None when any entry is non-finite (the
/// reference's `_covariance_or_none`).
pub fn finite_symmetrized(covariance_m2: Option<&Covariance3>) -> Option<Matrix3<f64>> {
    let covariance_m2 = covariance_m2?;
    all_finite(covariance_m2).then(|| symmetrized(&matrix_from_covariance(covariance_m2)))
}

/// Eigenvalues raised to at least `floor` (`eigh` + `np.maximum`); None when
/// the eigenvalues are not finite.
pub fn eigenvalue_floored(matrix: &Matrix3<f64>, floor: f64) -> Option<Matrix3<f64>> {
    let symmetric = symmetrized(matrix);
    let eigen = SymmetricEigen::new(symmetric);
    if !eigen.eigenvalues.iter().all(|value| value.is_finite()) {
        return None;
    }
    let floored = eigen.eigenvalues.map(|value| value.max(floor));
    let rebuilt = eigen.eigenvectors * Matrix3::from_diagonal(&floored) * eigen.eigenvectors.transpose();
    Some(symmetrized(&rebuilt))
}

pub fn largest_eigenvalue(matrix: &Matrix3<f64>) -> f64 {
    SymmetricEigen::new(symmetrized(matrix)).eigenvalues.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// `residual · A⁻¹ · residual`; None when `A` is singular or the result is not finite.
pub fn mahalanobis_squared(residual: &Vector3<f64>, covariance: &Matrix3<f64>) -> Option<f64> {
    let solved = covariance.lu().solve(residual)?;
    let value = residual.dot(&solved);
    value.is_finite().then_some(value)
}

/// Upper-left east/north block, symmetrized; None when it is not finite.
pub fn horizontal_block(covariance_m2: Option<&Covariance3>) -> Option<Matrix2<f64>> {
    let covariance_m2 = covariance_m2?;
    let block = Matrix2::new(covariance_m2[0][0], covariance_m2[0][1], covariance_m2[1][0], covariance_m2[1][1]);
    if !block.iter().all(|value| value.is_finite()) {
        return None;
    }
    Some((block + block.transpose()) * 0.5)
}

/// Eigenvalues of a symmetric 2×2 in ascending order (`eigvalsh`).
pub fn symmetric_eigenvalues_2x2(matrix: &Matrix2<f64>) -> [f64; 2] {
    let eigen = SymmetricEigen::new(*matrix);
    let (first, second) = (eigen.eigenvalues[0], eigen.eigenvalues[1]);
    if first <= second { [first, second] } else { [second, first] }
}

/// Ascending eigenvalues and the matching unit eigenvectors of a symmetric 2×2.
pub fn symmetric_eigen_2x2(matrix: &Matrix2<f64>) -> ([f64; 2], [Vector2<f64>; 2]) {
    let eigen = SymmetricEigen::new(*matrix);
    let first = (eigen.eigenvalues[0], eigen.eigenvectors.column(0).into_owned());
    let second = (eigen.eigenvalues[1], eigen.eigenvectors.column(1).into_owned());
    let (low, high) = if first.0 <= second.0 { (first, second) } else { (second, first) };
    ([low.0, high.0], [low.1, high.1])
}

pub fn horizontal_distance_m(first_m: &Position3, second_m: &Position3) -> f64 {
    (first_m[0] - second_m[0]).hypot(first_m[1] - second_m[1])
}

pub fn distance_m(first_m: &Position3, second_m: &Position3) -> f64 {
    (vector_from_position(first_m) - vector_from_position(second_m)).norm()
}
