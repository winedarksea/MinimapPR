//! Coarse-to-fine refinement as best-first branch-and-bound over polar cells.
//!
//! A cell's interval-envelope score bounds the exact score of every point
//! inside it, and the exact score at its centre is achievable. So: pop the
//! cell with the highest bound, split it (each axis halved), keep children
//! whose bound beats the best exact score seen so far (the incumbent), and
//! stop when no open cell can beat the incumbent by more than the
//! tolerance. A fixed-width beam is not enough: big coarse cells all reach
//! the maximum bound together and a beam drops the true cell on ties.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::config::RefinementConfig;
use super::polar_grid::{LayoutFrame, SearchCell};
use super::robust_scoring::{CellScore, RobustPairScorer};

#[derive(Clone, Copy, Debug)]
pub struct ScoredCell {
    pub cell: SearchCell,
    pub envelope: CellScore,
}

/// Best cell found with its exact score at the representative point.
#[derive(Clone, Copy, Debug)]
pub struct RefinedEstimate {
    pub cell: SearchCell,
    pub exact_score: f64,
    pub cells_evaluated: usize,
    /// False when the evaluation budget ran out before the bound closed.
    pub is_certified_within_tolerance: bool,
}

struct OpenCell(ScoredCell);

impl PartialEq for OpenCell {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for OpenCell {}

impl PartialOrd for OpenCell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OpenCell {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.envelope.score.total_cmp(&other.0.envelope.score)
    }
}

fn is_converged(scored: &ScoredCell, frame: &LayoutFrame, finest_lag_step_s: f64, config: &RefinementConfig) -> bool {
    let below_lag_resolution = scored.envelope.widest_interval_half_width_s <= finest_lag_step_s;
    let below_size_target = !scored.cell.is_far_field() && scored.cell.bounding_radius_m(frame) <= config.final_cell_radius_m;
    below_lag_resolution || below_size_target
}

/// Refine from `seed_cells` (already envelope-scored). `None` only when
/// `seed_cells` is empty.
pub fn refine_branch_and_bound(
    seed_cells: &[ScoredCell],
    scorer: &RobustPairScorer,
    frame: &LayoutFrame,
    config: &RefinementConfig,
) -> Option<RefinedEstimate> {
    let tolerance = config.optimality_tolerance_temperatures * scorer.likelihood_scale();
    let finest_lag_step_s = scorer.finest_lag_step_s();
    let mut incumbent: Option<(SearchCell, f64)> = None;
    let consider = |cell: &SearchCell, incumbent: &mut Option<(SearchCell, f64)>| {
        let exact = scorer.exact_cell_score(cell, frame);
        if incumbent.is_none_or(|(_, best)| exact > best) {
            *incumbent = Some((*cell, exact));
        }
    };

    // Seed the incumbent from the best few coarse cells so pruning starts early.
    let mut by_bound: Vec<&ScoredCell> = seed_cells.iter().collect();
    by_bound.sort_by(|left, right| right.envelope.score.total_cmp(&left.envelope.score));
    for scored in by_bound.iter().take(config.incumbent_seed_cell_count.max(1)) {
        consider(&scored.cell, &mut incumbent);
    }
    let mut open: BinaryHeap<OpenCell> = seed_cells.iter().map(|scored| OpenCell(*scored)).collect();
    let mut cells_evaluated = seed_cells.len();
    let mut is_certified = true;

    while let Some(OpenCell(scored)) = open.pop() {
        let (_, best) = incumbent?;
        if scored.envelope.score <= best + tolerance {
            break;
        }
        if is_converged(&scored, frame, finest_lag_step_s, config) {
            consider(&scored.cell, &mut incumbent);
            continue;
        }
        if cells_evaluated >= config.maximum_cell_evaluations {
            is_certified = false;
            break;
        }
        for child in scored.cell.subdivide() {
            let envelope = scorer.envelope_score(&child, frame);
            consider(&child, &mut incumbent);
            cells_evaluated += 1;
            let (_, best) = incumbent?;
            if envelope.score > best + tolerance {
                open.push(OpenCell(ScoredCell { cell: child, envelope }));
            }
        }
    }
    incumbent.map(|(cell, exact_score)| RefinedEstimate {
        cell,
        exact_score,
        cells_evaluated,
        is_certified_within_tolerance: is_certified,
    })
}
