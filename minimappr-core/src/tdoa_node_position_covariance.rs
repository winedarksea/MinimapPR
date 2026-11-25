//! Node-position term of a TDOA fix's covariance.
//!
//! A node whose stated position is off by a rigid translation `d` (every
//! channel of the node moves together) shifts each arrival at its sensors by
//! `-u_s . d / c`, `u_s` the unit vector from sensor `s` toward the source.
//! So, like a clock offset, the error is shared by all of the node's TDOAs
//! (correlated per node, not independent per pair), and its effect on the fix
//! grows with geometric dilution. It is propagated through the solve's own
//! least-squares sensitivity: `G R G^T`, `G = (J^T W J)^-1 J^T W` at the solved
//! position, where `R` holds each node's `d ~ N(0, sigma^2 I)` (per-axis
//! 1-sigma `position_std_m`, isotropic) projected onto the per-sensor unit
//! vectors. This is the direction-dependent projection, the same arrival
//! covariance the E5 observability oracle uses (`sigma^2 u_i . u_j / c^2`
//! inside a node). For a single-sensor node it is exactly a clock offset of
//! `sigma / c`; for a compact array it also covers the near-field spread of
//! the unit vectors.
//!
//! One implementation for every solver: the sidecar SRP (unweighted
//! reference-channel TDOAs, `srp_phat.rs`), the Rust port of the Python
//! Cartesian solve (weighted pair TDOAs, `tdoa_solver`), and, mirrored line
//! for line, Python `minimappr/core/node_position_covariance.py`. The shared
//! fixture `tests/fixtures/node_position_covariance_parity_cases.json` pins
//! the two languages together.

/// One TDOA row, `arrival(sensor_a) - arrival(sensor_b)`, entering the solve
/// with least-squares `weight` (its residual is scaled by `sqrt(weight)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightedTdoaRow {
    pub sensor_a: usize,
    pub sensor_b: usize,
    pub weight: f64,
}

/// Which node owns each sensor and every node's per-axis 1-sigma position
/// error. `0.0` means the node states none: it contributes nothing, which is
/// the behaviour before positions carried an accuracy.
#[derive(Clone, Copy, Debug)]
pub struct SensorNodePositionStd<'a> {
    /// Indexed by sensor: index into `node_position_std_m`.
    pub node_index_by_sensor: &'a [usize],
    pub node_position_std_m: &'a [f64],
}

/// Ridge on the normal matrix, relative to its trace: keeps a direction the
/// measurements cannot see at all (a planar layout's normal) from blowing up
/// the inverse, while staying far below any direction they can see. Same
/// value as the clock term (`srp_phat_clock_covariance.rs`).
pub const RELATIVE_NORMAL_MATRIX_RIDGE: f64 = 1.0e-12;
/// Added to each sensor-to-source distance so a source on a sensor has a
/// finite (zero) unit vector; matches the Python solver's Jacobian.
const DISTANCE_EPSILON_M: f64 = 1.0e-12;

/// Position covariance (m^2) from the nodes' stated position errors, or
/// `None` when no node states one, no row involves such a node, or the
/// geometry cannot be inverted. `additional_normal_matrix` is `J^T J` of any
/// non-TDOA rows the solve's covariance includes (bearing or amplitude
/// constraints): they shape the sensitivity but do not depend on node
/// positions here.
pub fn node_position_error_covariance_m2(
    position_m: [f64; 3],
    sensor_positions_m: &[[f64; 3]],
    tdoa_rows: &[WeightedTdoaRow],
    additional_normal_matrix: [[f64; 3]; 3],
    sound_speed_mps: f64,
    nodes: SensorNodePositionStd<'_>,
) -> Option<[[f64; 3]; 3]> {
    let node_states_an_error = |node: usize| {
        nodes
            .node_position_std_m
            .get(node)
            .is_some_and(|std_m| *std_m > 0.0)
    };
    let sensor_node = |sensor: usize| nodes.node_index_by_sensor.get(sensor).copied();
    let mut any_row_moves = false;
    for row in tdoa_rows {
        let (node_a, node_b) = (sensor_node(row.sensor_a)?, sensor_node(row.sensor_b)?);
        any_row_moves |= node_states_an_error(node_a) || node_states_an_error(node_b);
    }
    if !any_row_moves || !(sound_speed_mps > 0.0) {
        return None;
    }

    let unit_toward_source = |sensor: usize| -> Option<[f64; 3]> {
        let sensor_position_m = sensor_positions_m.get(sensor)?;
        let offset_m = [0, 1, 2].map(|axis| position_m[axis] - sensor_position_m[axis]);
        let distance_m = norm(offset_m) + DISTANCE_EPSILON_M;
        Some(offset_m.map(|axis| axis / distance_m))
    };
    // Weighted Jacobian rows sqrt(w) (u_a - u_b) / c, and the unit vectors
    // the node shifts project onto.
    let mut weighted_rows = Vec::with_capacity(tdoa_rows.len());
    let mut normal_matrix = additional_normal_matrix;
    for row in tdoa_rows {
        let unit_a = unit_toward_source(row.sensor_a)?;
        let unit_b = unit_toward_source(row.sensor_b)?;
        let row_scale = row.weight.max(0.0).sqrt() / sound_speed_mps;
        let jacobian_row = [0, 1, 2].map(|axis| (unit_a[axis] - unit_b[axis]) * row_scale);
        for out_axis in 0..3 {
            for in_axis in 0..3 {
                normal_matrix[out_axis][in_axis] += jacobian_row[out_axis] * jacobian_row[in_axis];
            }
        }
        weighted_rows.push((jacobian_row, unit_a, unit_b, row_scale));
    }
    let trace = normal_matrix[0][0] + normal_matrix[1][1] + normal_matrix[2][2];
    if !(trace.is_finite() && trace > 0.0) {
        return None;
    }
    for (axis, matrix_row) in normal_matrix.iter_mut().enumerate() {
        matrix_row[axis] += trace * RELATIVE_NORMAL_MATRIX_RIDGE;
    }
    let normal_inverse = invert_symmetric_3x3(normal_matrix)?;

    // Node k moved by d shifts row r by sqrt(w_r) (-[node(a)=k] u_a + [node(b)=k] u_b) . d / c.
    // Its axes are independent with variance sigma_k^2, so each axis j adds
    // sigma_k^2 g g^T with g = N^-1 sum_r J_r shift_rj.
    let mut covariance_m2 = [[0.0_f64; 3]; 3];
    for (node, &position_std_m) in nodes.node_position_std_m.iter().enumerate() {
        if !(position_std_m > 0.0) {
            continue;
        }
        let variance_m2 = position_std_m * position_std_m;
        for shift_axis in 0..3 {
            let mut projected = [0.0_f64; 3];
            let mut node_moves_any_row = false;
            for (row, (jacobian_row, unit_a, unit_b, row_scale)) in
                tdoa_rows.iter().zip(&weighted_rows)
            {
                let a_on_node = sensor_node(row.sensor_a) == Some(node);
                let b_on_node = sensor_node(row.sensor_b) == Some(node);
                if !a_on_node && !b_on_node {
                    continue;
                }
                node_moves_any_row = true;
                let shift = (f64::from(u8::from(b_on_node)) * unit_b[shift_axis]
                    - f64::from(u8::from(a_on_node)) * unit_a[shift_axis])
                    * row_scale;
                for axis in 0..3 {
                    projected[axis] += jacobian_row[axis] * shift;
                }
            }
            if !node_moves_any_row {
                break;
            }
            let sensitivity = multiply_3x3_vector(normal_inverse, projected);
            for out_row in 0..3 {
                for out_column in 0..3 {
                    covariance_m2[out_row][out_column] +=
                        variance_m2 * sensitivity[out_row] * sensitivity[out_column];
                }
            }
        }
    }
    covariance_m2
        .iter()
        .flatten()
        .all(|value| value.is_finite())
        .then_some(covariance_m2)
}

fn norm(vector: [f64; 3]) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

pub(crate) fn invert_symmetric_3x3(matrix: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let cofactor = |row_a: usize, row_b: usize, col_a: usize, col_b: usize| {
        matrix[row_a][col_a] * matrix[row_b][col_b] - matrix[row_a][col_b] * matrix[row_b][col_a]
    };
    let adjugate = [
        [
            cofactor(1, 2, 1, 2),
            -cofactor(0, 2, 1, 2),
            cofactor(0, 1, 1, 2),
        ],
        [
            -cofactor(1, 2, 0, 2),
            cofactor(0, 2, 0, 2),
            -cofactor(0, 1, 0, 2),
        ],
        [
            cofactor(1, 2, 0, 1),
            -cofactor(0, 2, 0, 1),
            cofactor(0, 1, 0, 1),
        ],
    ];
    let determinant = matrix[0][0] * adjugate[0][0]
        + matrix[0][1] * adjugate[1][0]
        + matrix[0][2] * adjugate[2][0];
    if determinant == 0.0 || !determinant.is_finite() {
        return None;
    }
    Some(adjugate.map(|row| row.map(|value| value / determinant)))
}

pub(crate) fn multiply_3x3_vector(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    matrix.map(|row| row[0] * vector[0] + row[1] * vector[1] + row[2] * vector[2])
}

#[cfg(test)]
#[path = "tdoa_node_position_covariance_tests.rs"]
mod tdoa_node_position_covariance_tests;
