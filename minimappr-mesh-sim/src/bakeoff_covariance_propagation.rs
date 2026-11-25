//! Per-node timing offsets propagated into a TDOA fix's covariance.
//!
//! A node's clock offset shifts every TDOA between it and the reference node
//! by the same amount, and its effect on position grows with geometric
//! dilution, so a flat per-axis term would understate it at range. This is
//! the same least-squares sandwich `G R Gᵀ`, `G = (JᵀJ)⁻¹Jᵀ`, that the
//! sidecar applies to its SRP covariance (`srp_phat_clock_covariance.rs`,
//! which is private to the core crate), here for the onset-clustering fix.
//! A node's surveyed-position error enters the same way: it moves that
//! node's predicted arrival by about σ_position / c.

use minimappr_core::tdoa_solver::nalgebra::{Matrix3, Vector3};

/// Ridge on `JᵀJ` relative to its trace, so a direction no measurement sees
/// (a planar layout's normal) does not blow up the inverse.
const RELATIVE_NORMAL_MATRIX_RIDGE: f64 = 1.0e-12;
const DIRECTION_NORM_EPSILON: f64 = 1.0e-9;

/// One reference-relative TDOA row: `tdoa = arrival(node) − arrival(reference)`.
pub struct ReferenceTdoaGeometry<'a> {
    pub position_m: [f64; 3],
    pub reference_node: usize,
    pub measured_nodes: &'a [usize],
    pub node_positions_m: &'a [[f64; 3]],
    pub sound_speed_mps: f64,
}

fn unit_from_node(position: &Vector3<f64>, node: &[f64; 3]) -> Vector3<f64> {
    let offset = position - Vector3::from(*node);
    let norm = offset.norm();
    if norm > DIRECTION_NORM_EPSILON {
        offset / norm
    } else {
        Vector3::zeros()
    }
}

/// Covariance (m²) from independent per-node timing offsets with 1-sigma
/// `node_offset_std_s[node]`; `None` when the geometry cannot be inverted.
pub fn node_offset_position_covariance_m2(geometry: &ReferenceTdoaGeometry<'_>, node_offset_std_s: &[f64]) -> Option<[[f64; 3]; 3]> {
    let position = Vector3::from(geometry.position_m);
    let reference_unit = unit_from_node(&position, &geometry.node_positions_m[geometry.reference_node]);
    let rows: Vec<Vector3<f64>> = geometry
        .measured_nodes
        .iter()
        .map(|node| (unit_from_node(&position, &geometry.node_positions_m[*node]) - reference_unit) / geometry.sound_speed_mps)
        .collect();
    let mut normal = Matrix3::<f64>::zeros();
    for row in &rows {
        normal += row * row.transpose();
    }
    let trace = normal.trace();
    if !(trace.is_finite() && trace > 0.0) {
        return None;
    }
    normal += Matrix3::identity() * (trace * RELATIVE_NORMAL_MATRIX_RIDGE);
    let normal_inverse = normal.try_inverse()?;
    let mut covariance = Matrix3::<f64>::zeros();
    for (node, offset_std_s) in node_offset_std_s.iter().enumerate() {
        let reference_shift = f64::from(u8::from(node == geometry.reference_node));
        let mut projected = Vector3::zeros();
        for (row, measured) in rows.iter().zip(geometry.measured_nodes) {
            let shift = f64::from(u8::from(*measured == node)) - reference_shift;
            projected += row * shift;
        }
        let sensitivity_m_per_s = normal_inverse * projected;
        covariance += sensitivity_m_per_s * sensitivity_m_per_s.transpose() * offset_std_s.powi(2);
    }
    Some(std::array::from_fn(|row| std::array::from_fn(|column| covariance[(row, column)])))
}

pub fn add_covariances(left: &[[f64; 3]; 3], right: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| std::array::from_fn(|column| left[row][column] + right[row][column]))
}
