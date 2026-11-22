//! Robust least-squares layout refine: MDS initial guess → Levenberg–
//! Marquardt on 2D positions with Huber IRLS, then rejection of pairs that
//! stay far out (a reflection-locked range, a phone that moved mid-round).
//!
//! Heights are fixed inputs (relative barometric heights), so the solve is
//! horizontal but residuals use 3D distances: on a sloped trail, a 3 m
//! height difference over 10 m would otherwise shrink the layout by 0.45 m.
//!
//! The result is a rigid layout in a local frame with per-node horizontal
//! σ from the gauge-free (pseudo-inverse) covariance.

use nalgebra::{DMatrix, DVector};

use super::classical_mds::{classical_mds, DistanceMatrix, MdsError};

#[derive(Clone, Debug, PartialEq)]
pub struct RangeMeasurement {
    pub node_i: usize,
    pub node_j: usize,
    pub distance_m: f64,
    pub sigma_m: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayoutRefineConfig {
    pub huber_threshold_sigma: f64,
    pub rejection_threshold_sigma: f64,
    pub maximum_iterations: usize,
    pub convergence_step_m: f64,
    pub initial_damping: f64,
    /// Tiny ridge that pins the 3 gauge freedoms (translation, rotation).
    pub gauge_ridge: f64,
    /// Singular values below this fraction of the largest count as gauge/null.
    pub null_space_relative_tolerance: f64,
}

impl Default for LayoutRefineConfig {
    fn default() -> Self {
        Self {
            huber_threshold_sigma: 2.0,
            rejection_threshold_sigma: 5.0,
            maximum_iterations: 50,
            convergence_step_m: 1e-6,
            initial_damping: 1e-3,
            gauge_ridge: 1e-9,
            null_space_relative_tolerance: 1e-9,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayoutSolution {
    /// `[x, y, height]` per node in the local frame.
    pub positions_m: Vec<[f64; 3]>,
    pub horizontal_sigmas_m: Vec<f64>,
    pub rejected_measurements: Vec<usize>,
    pub normalized_residuals: Vec<f64>,
    pub rms_residual_m: f64,
    /// False when the active pairs leave a non-gauge freedom (flex or a dangling node).
    pub is_rigid: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LayoutError {
    Mds(MdsError),
    HeightCountMismatch,
    SolveFailed,
}

fn residual_and_gradient(
    positions: &[[f64; 2]],
    heights: &[f64],
    measurement: &RangeMeasurement,
) -> (f64, [f64; 2]) {
    let (i, j) = (measurement.node_i, measurement.node_j);
    let dx = positions[i][0] - positions[j][0];
    let dy = positions[i][1] - positions[j][1];
    let dh = heights[i] - heights[j];
    let length = (dx * dx + dy * dy + dh * dh).sqrt().max(f64::EPSILON);
    (length - measurement.distance_m, [dx / length, dy / length])
}

fn huber_weight(normalized: f64, threshold: f64) -> f64 {
    if normalized.abs() <= threshold {
        1.0
    } else {
        threshold / normalized.abs()
    }
}

/// Horizontal distance matrix for MDS (3D ranges with height differences removed).
fn horizontal_distance_matrix(
    node_count: usize,
    heights: &[f64],
    measurements: &[RangeMeasurement],
    active: &[bool],
) -> DistanceMatrix {
    let mut matrix = vec![vec![None; node_count]; node_count];
    for (m, _) in measurements.iter().zip(active).filter(|(_, a)| **a) {
        let dh = heights[m.node_i] - heights[m.node_j];
        let horizontal = (m.distance_m.powi(2) - dh * dh).max(0.0).sqrt();
        matrix[m.node_i][m.node_j] = Some(horizontal);
        matrix[m.node_j][m.node_i] = Some(horizontal);
    }
    matrix
}

/// Solve the layout of `node_count` nodes from ranges and fixed heights.
pub fn solve_layout(
    node_count: usize,
    heights_m: &[f64],
    measurements: &[RangeMeasurement],
    config: &LayoutRefineConfig,
) -> Result<LayoutSolution, LayoutError> {
    if heights_m.len() != node_count {
        return Err(LayoutError::HeightCountMismatch);
    }
    let mut active: Vec<bool> = measurements
        .iter()
        .map(|m| {
            m.node_i < node_count
                && m.node_j < node_count
                && m.node_i != m.node_j
                && m.sigma_m > 0.0
        })
        .collect();
    let mut rejected = Vec::new();
    let initial = classical_mds(
        &horizontal_distance_matrix(node_count, heights_m, measurements, &active),
        2,
    )
    .map_err(LayoutError::Mds)?;
    let mut positions: Vec<[f64; 2]> = initial.iter().map(|p| [p[0], p[1]]).collect();
    loop {
        // Re-solves after a rejection warm-start from the Huber solution: a
        // fresh MDS with the gap filled by a shortest path can start in a
        // folded basin.
        let normal = refine(&mut positions, heights_m, measurements, &active, config)?;
        let normalized: Vec<f64> = measurements
            .iter()
            .map(|m| residual_and_gradient(&positions, heights_m, m).0 / m.sigma_m)
            .collect();
        let worst = (0..measurements.len())
            .filter(|index| {
                active[*index] && normalized[*index].abs() > config.rejection_threshold_sigma
            })
            .max_by(|a, b| normalized[*a].abs().total_cmp(&normalized[*b].abs()));
        if let Some(index) = worst {
            active[index] = false;
            rejected.push(index);
            if classical_mds(
                &horizontal_distance_matrix(node_count, heights_m, measurements, &active),
                2,
            )
            .is_ok()
            {
                continue;
            }
            // Dropping it would disconnect a node: keep it (down-weighted by Huber).
            active[index] = true;
            rejected.pop();
        }
        return Ok(finish(
            node_count,
            positions,
            heights_m,
            measurements,
            &active,
            normalized,
            rejected,
            normal,
            config,
        ));
    }
}

#[allow(clippy::too_many_arguments)]
fn finish(
    node_count: usize,
    positions: Vec<[f64; 2]>,
    heights_m: &[f64],
    measurements: &[RangeMeasurement],
    active: &[bool],
    normalized: Vec<f64>,
    rejected: Vec<usize>,
    normal: DMatrix<f64>,
    config: &LayoutRefineConfig,
) -> LayoutSolution {
    const GAUGE_FREEDOMS_2D: usize = 3;
    let svd = normal.svd(true, true);
    let largest = svd.singular_values.max();
    let null_count = svd
        .singular_values
        .iter()
        .filter(|s| **s <= largest * config.null_space_relative_tolerance)
        .count();
    let pseudo_inverse = svd
        .pseudo_inverse(largest * config.null_space_relative_tolerance)
        .unwrap_or_else(|_| DMatrix::zeros(2 * node_count, 2 * node_count));
    let horizontal_sigmas_m = (0..node_count)
        .map(|n| {
            (pseudo_inverse[(2 * n, 2 * n)] + pseudo_inverse[(2 * n + 1, 2 * n + 1)])
                .max(0.0)
                .sqrt()
        })
        .collect();
    let active_residuals: Vec<f64> = measurements
        .iter()
        .zip(active)
        .filter(|(_, a)| **a)
        .map(|(m, _)| residual_and_gradient(&positions, heights_m, m).0)
        .collect();
    let rms_residual_m = if active_residuals.is_empty() {
        0.0
    } else {
        (active_residuals.iter().map(|r| r * r).sum::<f64>() / active_residuals.len() as f64).sqrt()
    };
    LayoutSolution {
        positions_m: positions
            .iter()
            .zip(heights_m)
            .map(|(p, h)| [p[0], p[1], *h])
            .collect(),
        horizontal_sigmas_m,
        rejected_measurements: rejected,
        normalized_residuals: normalized,
        rms_residual_m,
        is_rigid: node_count < 2 || null_count <= GAUGE_FREEDOMS_2D,
    }
}

/// LM with Huber IRLS. Returns the final (unweighted-by-Huber) normal matrix
/// JᵀΣ⁻¹J for the covariance.
fn refine(
    positions: &mut [[f64; 2]],
    heights: &[f64],
    measurements: &[RangeMeasurement],
    active: &[bool],
    config: &LayoutRefineConfig,
) -> Result<DMatrix<f64>, LayoutError> {
    let unknowns = 2 * positions.len();
    let mut damping = config.initial_damping;
    let cost = |positions: &[[f64; 2]], weights: &[f64]| -> f64 {
        measurements
            .iter()
            .zip(active)
            .zip(weights)
            .filter(|((_, a), _)| **a)
            .map(|((m, _), w)| {
                w * (residual_and_gradient(positions, heights, m).0 / m.sigma_m).powi(2)
            })
            .sum()
    };
    let assemble = |positions: &[[f64; 2]], weights: &[f64]| {
        let mut normal = DMatrix::<f64>::zeros(unknowns, unknowns);
        let mut gradient = DVector::<f64>::zeros(unknowns);
        for ((m, &is_active), &weight) in measurements.iter().zip(active).zip(weights) {
            if !is_active {
                continue;
            }
            let (residual, [gx, gy]) = residual_and_gradient(positions, heights, m);
            let w = weight / m.sigma_m.powi(2);
            let entries = [
                (2 * m.node_i, gx),
                (2 * m.node_i + 1, gy),
                (2 * m.node_j, -gx),
                (2 * m.node_j + 1, -gy),
            ];
            for &(row, row_value) in &entries {
                gradient[row] += w * row_value * residual;
                for &(column, column_value) in &entries {
                    normal[(row, column)] += w * row_value * column_value;
                }
            }
        }
        (normal, gradient)
    };
    let mut weights = vec![1.0; measurements.len()];
    for _iteration in 0..config.maximum_iterations {
        for (index, m) in measurements.iter().enumerate() {
            let normalized = residual_and_gradient(positions, heights, m).0 / m.sigma_m;
            weights[index] = huber_weight(normalized, config.huber_threshold_sigma);
        }
        let (normal, gradient) = assemble(positions, &weights);
        let current_cost = cost(positions, &weights);
        let mut improved = false;
        for _attempt in 0..10 {
            let mut system = normal.clone();
            for k in 0..unknowns {
                system[(k, k)] += damping * normal[(k, k)].max(1.0) + config.gauge_ridge;
            }
            let Some(step) = system.lu().solve(&(-&gradient)) else {
                damping *= 10.0;
                continue;
            };
            let candidate: Vec<[f64; 2]> = positions
                .iter()
                .enumerate()
                .map(|(n, p)| [p[0] + step[2 * n], p[1] + step[2 * n + 1]])
                .collect();
            if cost(&candidate, &weights) <= current_cost {
                positions.copy_from_slice(&candidate);
                damping = (damping / 10.0).max(1e-12);
                improved = true;
                if step.amax() < config.convergence_step_m {
                    let unit = vec![1.0; measurements.len()];
                    return Ok(assemble(positions, &unit).0);
                }
                break;
            }
            damping *= 10.0;
        }
        if !improved {
            break;
        }
    }
    let unit = vec![1.0; measurements.len()];
    let normal = assemble(positions, &unit).0;
    if normal.iter().any(|v| !v.is_finite()) {
        return Err(LayoutError::SolveFailed);
    }
    Ok(normal)
}

#[cfg(test)]
#[path = "robust_layout_refine_tests.rs"]
mod tests;
