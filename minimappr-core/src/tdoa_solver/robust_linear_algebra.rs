//! Small dense linear-algebra helpers that never panic.
//!
//! Every decomposition first checks its input is finite and uses nalgebra's
//! `try_*` variants, so a singular or NaN-poisoned system surfaces as a
//! [`TdoaSolverError`] instead of an abort deep inside a phone or WASM worker.

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

use super::TdoaSolverError;

/// Convergence epsilon handed to nalgebra's iterative SVD / eigen solvers.
const DECOMPOSITION_CONVERGENCE_EPSILON: f64 = f64::EPSILON;
/// Iteration cap for the iterative decompositions (0 would mean "forever").
const DECOMPOSITION_MAX_ITERATIONS: usize = 10_000;

/// Thin SVD with singular values sorted descending.
pub struct SortedSingularValueDecomposition {
    pub left_vectors: DMatrix<f64>,
    pub singular_values: DVector<f64>,
    /// Rows are right singular vectors (`Vᵀ`).
    pub right_vectors_transposed: DMatrix<f64>,
}

fn ensure_all_finite(values: &[f64], context: &'static str) -> Result<(), TdoaSolverError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(TdoaSolverError::NonFiniteInput(context))
    }
}

/// SVD of an arbitrary `m×n` matrix; errors on non-finite input or non-convergence.
pub fn sorted_singular_value_decomposition(
    matrix: &DMatrix<f64>,
) -> Result<SortedSingularValueDecomposition, TdoaSolverError> {
    ensure_all_finite(matrix.as_slice(), "svd input")?;
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return Err(TdoaSolverError::EmptySystem("svd input"));
    }
    let decomposition = matrix
        .clone()
        .try_svd(
            true,
            true,
            DECOMPOSITION_CONVERGENCE_EPSILON,
            DECOMPOSITION_MAX_ITERATIONS,
        )
        .ok_or(TdoaSolverError::DecompositionDidNotConverge("svd"))?;
    match (decomposition.u, decomposition.v_t) {
        (Some(left_vectors), Some(right_vectors_transposed)) => {
            Ok(SortedSingularValueDecomposition {
                left_vectors,
                singular_values: decomposition.singular_values,
                right_vectors_transposed,
            })
        }
        _ => Err(TdoaSolverError::DecompositionDidNotConverge("svd vectors")),
    }
}

/// Singular values only, sorted descending.
pub fn sorted_singular_values(matrix: &DMatrix<f64>) -> Result<DVector<f64>, TdoaSolverError> {
    Ok(sorted_singular_value_decomposition(matrix)?.singular_values)
}

/// Minimum-norm least-squares solution, matching `numpy.linalg.lstsq(rcond=None)`:
/// singular values below `max(m, n) · ε · σ_max` are treated as zero.
pub fn minimum_norm_least_squares(
    matrix: &DMatrix<f64>,
    target: &DVector<f64>,
) -> Result<DVector<f64>, TdoaSolverError> {
    ensure_all_finite(target.as_slice(), "lstsq target")?;
    let decomposition = sorted_singular_value_decomposition(matrix)?;
    let largest_singular_value = decomposition.singular_values.max();
    let cutoff = matrix.nrows().max(matrix.ncols()) as f64 * f64::EPSILON * largest_singular_value;
    let projected_target = decomposition.left_vectors.transpose() * target;
    let mut scaled = DVector::zeros(decomposition.singular_values.len());
    for (index, singular_value) in decomposition.singular_values.iter().enumerate() {
        if *singular_value > cutoff {
            scaled[index] = projected_target[index] / singular_value;
        }
    }
    let solution = decomposition.right_vectors_transposed.transpose() * scaled;
    ensure_all_finite(solution.as_slice(), "lstsq solution")?;
    Ok(solution)
}

/// Eigen-decomposition of a symmetric 3×3 matrix with eigenvalues ascending
/// (the `numpy.linalg.eigh` convention). Columns of the matrix are eigenvectors.
pub fn ascending_symmetric_eigen(
    matrix: &Matrix3<f64>,
) -> Result<(Vector3<f64>, Matrix3<f64>), TdoaSolverError> {
    ensure_all_finite(matrix.as_slice(), "eigh input")?;
    let symmetric = 0.5 * (matrix + matrix.transpose());
    let decomposition = symmetric
        .try_symmetric_eigen(
            DECOMPOSITION_CONVERGENCE_EPSILON,
            DECOMPOSITION_MAX_ITERATIONS,
        )
        .ok_or(TdoaSolverError::DecompositionDidNotConverge(
            "symmetric eigen",
        ))?;
    let mut order = [0_usize, 1, 2];
    order.sort_by(|left, right| {
        decomposition.eigenvalues[*left]
            .partial_cmp(&decomposition.eigenvalues[*right])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let eigenvalues = Vector3::new(
        decomposition.eigenvalues[order[0]],
        decomposition.eigenvalues[order[1]],
        decomposition.eigenvalues[order[2]],
    );
    let eigenvectors = Matrix3::from_columns(&[
        decomposition.eigenvectors.column(order[0]).into_owned(),
        decomposition.eigenvectors.column(order[1]).into_owned(),
        decomposition.eigenvectors.column(order[2]).into_owned(),
    ]);
    Ok((eigenvalues, eigenvectors))
}

/// Inverse of a symmetric positive-definite 3×3 information matrix.
///
/// Refuses (explicit error, never a panic or a silently huge matrix) when the
/// matrix is non-finite, not positive definite, or its condition number
/// exceeds `maximum_condition_number`.
pub fn invert_symmetric_positive_definite(
    information: &Matrix3<f64>,
    maximum_condition_number: f64,
) -> Result<Matrix3<f64>, TdoaSolverError> {
    let (eigenvalues, eigenvectors) = ascending_symmetric_eigen(information)?;
    let smallest = eigenvalues[0];
    let largest = eigenvalues[2];
    if smallest <= 0.0 || largest <= 0.0 {
        return Err(TdoaSolverError::Singular {
            smallest_eigenvalue: smallest,
        });
    }
    let condition_number = largest / smallest;
    if !condition_number.is_finite() || condition_number > maximum_condition_number {
        return Err(TdoaSolverError::IllConditioned { condition_number });
    }
    let inverse_eigenvalues = Matrix3::from_diagonal(&eigenvalues.map(|value| 1.0 / value));
    let inverse = eigenvectors * inverse_eigenvalues * eigenvectors.transpose();
    Ok(0.5 * (inverse + inverse.transpose()))
}

/// `‖v‖` helper for nalgebra column views built from slices.
pub fn vector3_from_slice(values: &[f64; 3]) -> Vector3<f64> {
    Vector3::new(values[0], values[1], values[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn least_squares_matches_minimum_norm_for_underdetermined_system() {
        // x + y = 2 has minimum-norm solution (1, 1).
        let matrix = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let target = DVector::from_vec(vec![2.0]);
        let solution = minimum_norm_least_squares(&matrix, &target).unwrap();
        assert!((solution[0] - 1.0).abs() < 1e-12);
        assert!((solution[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn singular_information_is_an_error_not_a_panic() {
        let rank_two = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        assert!(invert_symmetric_positive_definite(&rank_two, 1e12).is_err());
        let ill = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1e-14);
        assert!(matches!(
            invert_symmetric_positive_definite(&ill, 1e12),
            Err(TdoaSolverError::IllConditioned { .. })
        ));
        let poisoned = Matrix3::new(f64::NAN, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        assert!(invert_symmetric_positive_definite(&poisoned, 1e12).is_err());
    }

    #[test]
    fn eigenvalues_are_ascending() {
        let matrix = Matrix3::from_diagonal(&Vector3::new(3.0, 1.0, 2.0));
        let (eigenvalues, _) = ascending_symmetric_eigen(&matrix).unwrap();
        assert!(eigenvalues[0] <= eigenvalues[1] && eigenvalues[1] <= eigenvalues[2]);
    }
}
