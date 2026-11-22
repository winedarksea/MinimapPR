//! The core layout chain the chirality, anchor and invariance gates drive:
//! noisy two-way chirp ranges → `solve_layout` (MDS + robust refine, an
//! arbitrary frame and handedness) → `fit_layout_to_gps` with noisy GPS.

use minimappr_core::layout::gps_anchor_fit::{
    fit_layout_to_gps, AnchorFit, AnchorFitConfig, AnchorFitError, ChiralityCue, GpsFix, RigidTransform2d,
};
use minimappr_core::layout::robust_layout_refine::{solve_layout, LayoutError, LayoutRefineConfig, LayoutSolution, RangeMeasurement};

use crate::trial_geometry::{distance, gaussian};

/// All-pairs ranges with Gaussian noise (seed, purpose keyed by the pair).
pub fn noisy_ranges(seed: u64, purpose: &str, positions_m: &[[f64; 3]], range_std_m: f64) -> Vec<RangeMeasurement> {
    let mut ranges = Vec::new();
    for node_i in 0..positions_m.len() {
        for node_j in node_i + 1..positions_m.len() {
            let noise = range_std_m * gaussian(seed, purpose, node_i as u64, node_j as u64);
            ranges.push(RangeMeasurement {
                node_i,
                node_j,
                distance_m: distance(positions_m[node_i], positions_m[node_j]) + noise,
                sigma_m: range_std_m,
            });
        }
    }
    ranges
}

pub fn solve_layout_from_truth(
    seed: u64,
    purpose: &str,
    positions_m: &[[f64; 3]],
    range_std_m: f64,
) -> Result<LayoutSolution, LayoutError> {
    let heights: Vec<f64> = positions_m.iter().map(|p| p[2]).collect();
    solve_layout(positions_m.len(), &heights, &noisy_ranges(seed, purpose, positions_m, range_std_m), &LayoutRefineConfig::default())
}

pub fn horizontal(positions_m: &[[f64; 3]]) -> Vec<[f64; 2]> {
    positions_m.iter().map(|p| [p[0], p[1]]).collect()
}

/// GPS fixes: truth + per-phone error (σ per axis), keyed by `epoch`.
pub fn gps_fixes(seed: u64, purpose: &str, epoch: u64, truth_m: &[[f64; 3]], error_std_m: f64) -> Vec<Option<GpsFix>> {
    truth_m
        .iter()
        .enumerate()
        .map(|(index, p)| {
            let key = index as u64;
            Some(GpsFix {
                east_m: p[0] + error_std_m * gaussian(seed, purpose, key, 2 * epoch),
                north_m: p[1] + error_std_m * gaussian(seed, purpose, key, 2 * epoch + 1),
                sigma_m: error_std_m,
            })
        })
        .collect()
}

pub fn fit_to_gps(layout_xy_m: &[[f64; 2]], fixes: &[Option<GpsFix>], cue: ChiralityCue) -> Result<AnchorFit, AnchorFitError> {
    fit_layout_to_gps(layout_xy_m, fixes, cue, &AnchorFitConfig::default())
}

/// Sign of the cross-covariance determinant between two point sets:
/// true when `candidate` is a mirror image of `reference`.
pub fn is_reflection_of(candidate: &[[f64; 2]], reference: &[[f64; 2]]) -> bool {
    let count = candidate.len().max(1) as f64;
    let mean = |points: &[[f64; 2]]| {
        let sum = points.iter().fold([0.0; 2], |acc, p| [acc[0] + p[0], acc[1] + p[1]]);
        [sum[0] / count, sum[1] / count]
    };
    let (a_mean, b_mean) = (mean(candidate), mean(reference));
    let mut h = [[0.0; 2]; 2];
    for (a, b) in candidate.iter().zip(reference) {
        let (da, db) = ([a[0] - a_mean[0], a[1] - a_mean[1]], [b[0] - b_mean[0], b[1] - b_mean[1]]);
        for row in 0..2 {
            for column in 0..2 {
                h[row][column] += da[row] * db[column];
            }
        }
    }
    h[0][0] * h[1][1] - h[0][1] * h[1][0] < 0.0
}

pub fn apply_all(transform: &RigidTransform2d, points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    points.iter().map(|p| transform.apply(*p)).collect()
}
