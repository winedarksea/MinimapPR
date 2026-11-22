//! Relating two layout solves: each MDS solve has its own arbitrary frame
//! (rotation, translation and possibly a reflection), so positions and a
//! locked chirality can only move between solves through their common nodes.

use crate::layout::gps_anchor_fit::{weighted_kabsch_2d, RigidTransform2d};

use super::island_sync_engine::IslandLayout;

/// Common nodes needed to carry positions between two layout frames.
const MINIMUM_COMMON_NODES_FOR_FRAME_CHANGE: usize = 2;
/// Minimum common nodes for a reflection test between two layouts.
const MINIMUM_COMMON_NODES_FOR_REFLECTION_TEST: usize = 3;
/// Cross-covariance determinants below this (m⁴) mean a degenerate overlap.
const REFLECTION_TEST_DETERMINANT_FLOOR_M4: f64 = 1e-6;

/// The same node's position in two layouts.
type PositionPair = ([f64; 3], [f64; 3]);

/// Whether `new` is a mirror image of `old` over their common nodes
/// (sign of the 2×2 cross-covariance determinant); `None` if degenerate.
pub fn layouts_are_reflected(old: &IslandLayout, new: &IslandLayout) -> Option<bool> {
    let common: Vec<PositionPair> = old
        .local_positions_m
        .iter()
        .filter_map(|(node, a)| new.local_positions_m.get(node).map(|b| (*a, *b)))
        .collect();
    if common.len() < MINIMUM_COMMON_NODES_FOR_REFLECTION_TEST {
        return None;
    }
    let count = common.len() as f64;
    let mean = |pick: fn(&PositionPair) -> [f64; 3]| {
        let sum = common.iter().map(pick).fold([0.0; 2], |acc, p| [acc[0] + p[0], acc[1] + p[1]]);
        [sum[0] / count, sum[1] / count]
    };
    let (old_mean, new_mean) = (mean(|pair| pair.0), mean(|pair| pair.1));
    let mut h = [[0.0; 2]; 2];
    for (a, b) in &common {
        let (da, db) = ([a[0] - old_mean[0], a[1] - old_mean[1]], [b[0] - new_mean[0], b[1] - new_mean[1]]);
        for row in 0..2 {
            for column in 0..2 {
                h[row][column] += da[row] * db[column];
            }
        }
    }
    let determinant = h[0][0] * h[1][1] - h[0][1] * h[1][0];
    (determinant.abs() > REFLECTION_TEST_DETERMINANT_FLOOR_M4).then_some(determinant < 0.0)
}

/// Rigid map from the previous layout's frame into a new solve's frame over
/// their common nodes (reflection allowed); `None` with fewer than two.
pub fn previous_to_new_frame(previous: &IslandLayout, new: &IslandLayout) -> Option<RigidTransform2d> {
    let common: Vec<([f64; 2], [f64; 2])> = previous
        .local_positions_m
        .iter()
        .filter_map(|(node, a)| new.local_positions_m.get(node).map(|b| ([a[0], a[1]], [b[0], b[1]])))
        .collect();
    if common.len() < MINIMUM_COMMON_NODES_FOR_FRAME_CHANGE {
        return None;
    }
    let mirrored = layouts_are_reflected(previous, new) == Some(true);
    let (source, target): (Vec<[f64; 2]>, Vec<[f64; 2]>) = common.into_iter().unzip();
    Some(weighted_kabsch_2d(&source, &target, &vec![1.0; source.len()], mirrored))
}
