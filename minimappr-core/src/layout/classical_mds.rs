//! Classical multidimensional scaling: pairwise distances → initial 2D
//! layout. Recovers positions only up to rotation, translation and
//! reflection (the mirror is resolved later by the GPS fit).
//!
//! Missing pairs (phones that did not hear each other) are filled with
//! shortest-path distances through the graph, an upper bound that is good
//! enough for an initial guess the robust refine then corrects.

use nalgebra::{DMatrix, SymmetricEigen};

/// Square distance matrix; `None` = not measured. Diagonal ignored.
pub type DistanceMatrix = Vec<Vec<Option<f64>>>;

#[derive(Clone, Debug, PartialEq)]
pub enum MdsError {
    NotSquare,
    TooFewNodes,
    /// Some node has no path to the rest (should be split into islands first).
    DisconnectedGraph,
    EigenDecompositionFailed,
}

/// Minimum nodes for a 2D layout.
pub const MINIMUM_NODES_FOR_LAYOUT: usize = 2;

/// Symmetrise (average both directions) and fill gaps by Floyd–Warshall.
pub fn complete_distance_matrix(distances: &DistanceMatrix) -> Result<Vec<Vec<f64>>, MdsError> {
    let n = distances.len();
    if distances.iter().any(|row| row.len() != n) {
        return Err(MdsError::NotSquare);
    }
    let mut complete = vec![vec![f64::INFINITY; n]; n];
    for i in 0..n {
        complete[i][i] = 0.0;
        for j in 0..n {
            if i == j {
                continue;
            }
            let measured: Vec<f64> = [distances[i][j], distances[j][i]]
                .into_iter()
                .flatten()
                .filter(|d| d.is_finite() && *d >= 0.0)
                .collect();
            if !measured.is_empty() {
                complete[i][j] = measured.iter().sum::<f64>() / measured.len() as f64;
            }
        }
    }
    for via in 0..n {
        for i in 0..n {
            for j in 0..n {
                let through = complete[i][via] + complete[via][j];
                if through < complete[i][j] {
                    complete[i][j] = through;
                }
            }
        }
    }
    if complete.iter().flatten().any(|d| !d.is_finite()) {
        return Err(MdsError::DisconnectedGraph);
    }
    Ok(complete)
}

/// Classical MDS into `dimensions` (2 for the horizontal layout).
/// Returns one point per node, centred on the origin.
pub fn classical_mds(
    distances: &DistanceMatrix,
    dimensions: usize,
) -> Result<Vec<Vec<f64>>, MdsError> {
    let n = distances.len();
    if n < MINIMUM_NODES_FOR_LAYOUT {
        return Err(MdsError::TooFewNodes);
    }
    let complete = complete_distance_matrix(distances)?;
    // B = −½ J D² J with J the centring matrix.
    let squared = DMatrix::from_fn(n, n, |i, j| complete[i][j].powi(2));
    let row_means: Vec<f64> = (0..n).map(|i| squared.row(i).mean()).collect();
    let grand_mean = squared.mean();
    let gram = DMatrix::from_fn(n, n, |i, j| {
        -0.5 * (squared[(i, j)] - row_means[i] - row_means[j] + grand_mean)
    });
    let eigen =
        SymmetricEigen::try_new(gram, f64::EPSILON, 0).ok_or(MdsError::EigenDecompositionFailed)?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| eigen.eigenvalues[*b].total_cmp(&eigen.eigenvalues[*a]));
    let coordinates = (0..n)
        .map(|node| {
            order
                .iter()
                .take(dimensions)
                .map(|&component| {
                    let scale = eigen.eigenvalues[component].max(0.0).sqrt();
                    eigen.eigenvectors[(node, component)] * scale
                })
                .collect()
        })
        .collect();
    Ok(coordinates)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn distances_of(points: &[[f64; 2]]) -> DistanceMatrix {
        points
            .iter()
            .map(|a| {
                points
                    .iter()
                    .map(|b| Some(((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()))
                    .collect()
            })
            .collect()
    }

    fn layout_distance(points: &[Vec<f64>], i: usize, j: usize) -> f64 {
        ((points[i][0] - points[j][0]).powi(2) + (points[i][1] - points[j][1]).powi(2)).sqrt()
    }

    const TRUE_POINTS: [[f64; 2]; 5] = [
        [0.0, 0.0],
        [12.0, 1.0],
        [5.0, 9.0],
        [-4.0, 6.0],
        [8.0, -7.0],
    ];

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn exact_distances_reproduce_geometry() {
        let distances = distances_of(&TRUE_POINTS);
        let layout = classical_mds(&distances, 2).unwrap();
        for i in 0..5 {
            for j in 0..5 {
                assert!((layout_distance(&layout, i, j) - distances[i][j].unwrap()).abs() < 1e-6);
            }
        }
        let centroid: f64 = layout.iter().map(|p| p[0]).sum::<f64>() / 5.0;
        assert!(centroid.abs() < 1e-9);
    }

    #[test]
    fn missing_pair_is_filled_and_layout_stays_close() {
        let mut distances = distances_of(&TRUE_POINTS);
        distances[1][3] = None;
        distances[3][1] = None;
        let layout = classical_mds(&distances, 2).unwrap();
        let worst = (0..5)
            .flat_map(|i| (0..5).map(move |j| (i, j)))
            .filter(|(i, j)| !((*i == 1 && *j == 3) || (*i == 3 && *j == 1)))
            .map(|(i, j)| {
                (layout_distance(&layout, i, j) - distances_of(&TRUE_POINTS)[i][j].unwrap()).abs()
            })
            .fold(0.0, f64::max);
        assert!(worst < 4.0, "initial guess only; worst {worst}");
    }

    #[test]
    fn disconnected_graph_is_an_error() {
        let distances = vec![
            vec![None, Some(3.0), None],
            vec![Some(3.0), None, None],
            vec![None, None, None],
        ];
        assert_eq!(
            classical_mds(&distances, 2),
            Err(MdsError::DisconnectedGraph)
        );
    }
}
