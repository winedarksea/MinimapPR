//! Adaptive refinement of horizontal columns for the likelihood region.
//!
//! Coarse cells (2 m × 4.4 m slabs near the layout, ~15 m far out) span
//! several metres of path difference, so their envelope likelihood is flat
//! over a large area. Columns that can still hold mass (upper-bound
//! likelihood ≥ prune) are split horizontally into four (live height slabs
//! halved while thicker than the footprint) until every live cell's TDOA
//! intervals are below `region_lag_resolution_s` (the envelope is then
//! tight) or the footprint reaches a range-relative floor (bounds cost for
//! long far-range wedges). Splitting whole columns keeps footprints disjoint, so the height
//! marginal Σ_z L·Δz stays exact per column. Far-field bearing columns are
//! refined the same way in azimuth and elevation (a coarse 5° × 10° patch
//! envelope would otherwise give the far ring spurious mass).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::config::UncertaintyConfig;
use super::polar_grid::{CoarsePolarGrid, HorizontalColumn, LayoutFrame, SearchCell};
use super::robust_scoring::{CellScore, RobustPairScorer};

/// Refined columns plus cost; `is_complete` is false when the evaluation
/// budget left some columns coarser than their resolution target.
#[derive(Clone, Debug)]
pub struct RegionRefinement {
    pub columns: Vec<RefinedColumn>,
    pub cells_evaluated: usize,
    pub is_complete: bool,
}

/// A final column with its height-marginalized likelihood Σ_z L · Δz.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefinedColumn {
    pub column: HorizontalColumn,
    pub likelihood_height_integral_m: f64,
}

#[derive(Clone, Copy, Debug)]
struct Slab {
    /// Height above ground (near) or elevation in radians (far).
    bounds: [f64; 2],
    /// Δz used in the height marginal; far patches count as a share of one
    /// coarse height slab proportional to their elevation span.
    marginal_weight_m: f64,
    likelihood: f64,
    /// Widest pair TDOA interval over this cell (envelope looseness).
    widest_interval_half_width_s: f64,
}

struct WorkColumn {
    column: HorizontalColumn,
    slabs: Vec<Slab>,
    peak_likelihood: f64,
    /// Refinement priority: an upper bound on the column's mass
    /// (peak likelihood × footprint area), so a loose stray coarse column is
    /// split before the winner is polished to centimetres.
    mass_bound: f64,
}

impl WorkColumn {
    fn new(column: HorizontalColumn, slabs: Vec<Slab>, far_field_area_per_radian_m2: f64) -> Self {
        let peak_likelihood = slabs.iter().map(|slab| slab.likelihood).fold(0.0, f64::max);
        let area_m2 = if column.is_far_field {
            far_field_area_per_radian_m2 * (column.layout_azimuth_rad[1] - column.layout_azimuth_rad[0])
        } else {
            column.area_m2
        };
        Self { column, slabs, peak_likelihood, mass_bound: peak_likelihood * area_m2 }
    }
}

impl PartialEq for WorkColumn {
    fn eq(&self, other: &Self) -> bool {
        self.mass_bound == other.mass_bound
    }
}

impl Eq for WorkColumn {}

impl PartialOrd for WorkColumn {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WorkColumn {
    fn cmp(&self, other: &Self) -> Ordering {
        self.mass_bound.total_cmp(&other.mass_bound)
    }
}

fn column_cell(column: &HorizontalColumn, slab_bounds: [f64; 2]) -> SearchCell {
    if column.is_far_field {
        SearchCell::FarField { azimuth_rad: column.layout_azimuth_rad, elevation_rad: slab_bounds }
    } else {
        SearchCell::Near { radial_m: column.radial_m, azimuth_rad: column.layout_azimuth_rad, height_m: slab_bounds }
    }
}

/// Pruned slabs carry no mass: their likelihood is only an upper bound
/// below the prune level, and on a large coarse column even that bound
/// times its area would swamp a centimetre-scale true region.
fn finish(work: WorkColumn, prune_likelihood: f64) -> RefinedColumn {
    RefinedColumn {
        column: work.column,
        likelihood_height_integral_m: work
            .slabs
            .iter()
            .filter(|slab| slab.likelihood >= prune_likelihood)
            .map(|slab| slab.likelihood * slab.marginal_weight_m)
            .sum(),
    }
}

/// Horizontal size of a column footprint (far: arc at the outer radius).
fn footprint_radius_m(column: &HorizontalColumn, frame: &LayoutFrame) -> (f64, f64) {
    if column.is_far_field {
        let arc_half_m = 0.5 * (column.layout_azimuth_rad[1] - column.layout_azimuth_rad[0]) * column.radial_m[0];
        (arc_half_m, column.radial_m[0])
    } else {
        let radius = column_cell(column, [0.0, 0.0]).bounding_radius_m(frame);
        (radius, 0.5 * (column.radial_m[0] + column.radial_m[1]))
    }
}

fn split_footprint(column: &HorizontalColumn) -> Vec<HorizontalColumn> {
    let [low, high] = column.layout_azimuth_rad;
    let azimuth_middle = 0.5 * (low + high);
    let azimuth_halves = [[low, azimuth_middle], [azimuth_middle, high]];
    if column.is_far_field {
        return azimuth_halves
            .into_iter()
            .map(|azimuth| HorizontalColumn { layout_azimuth_rad: azimuth, ..*column })
            .collect();
    }
    let [inner, outer] = column.radial_m;
    let radial_middle = 0.5 * (inner + outer);
    let mut children = Vec::with_capacity(4);
    for radial in [[inner, radial_middle], [radial_middle, outer]] {
        for azimuth in azimuth_halves {
            children.push(HorizontalColumn {
                radial_m: radial,
                layout_azimuth_rad: azimuth,
                area_m2: 0.5 * (azimuth[1] - azimuth[0]) * (radial[1].powi(2) - radial[0].powi(2)),
                is_far_field: false,
            });
        }
    }
    children
}

/// Refine coarse columns into the final region columns. `coarse_scores`
/// are envelope scores per grid cell; `reference_score` is the best exact
/// score (likelihood 1).
pub fn refine_region_columns(
    grid: &CoarsePolarGrid,
    coarse_scores: &[CellScore],
    reference_score: f64,
    scorer: &RobustPairScorer,
    frame: &LayoutFrame,
    config: &UncertaintyConfig,
) -> RegionRefinement {
    let far_field_area_per_radian_m2 = grid.far_field_area_per_radian_m2();
    let scale = scorer.likelihood_scale();
    let likelihood_of = |score: f64| ((score - reference_score) / scale).exp();
    let mut slabs_per_column: Vec<Vec<Slab>> = vec![Vec::new(); grid.columns.len()];
    for (cell_index, cell) in grid.cells.iter().enumerate() {
        let (bounds, marginal_weight_m) = match *cell {
            SearchCell::Near { height_m, .. } => (height_m, height_m[1] - height_m[0]),
            SearchCell::FarField { elevation_rad, .. } => (elevation_rad, grid.height_layer_thickness_m),
        };
        slabs_per_column[grid.cell_column_index[cell_index]].push(Slab {
            bounds,
            marginal_weight_m,
            likelihood: likelihood_of(coarse_scores[cell_index].score),
            widest_interval_half_width_s: coarse_scores[cell_index].widest_interval_half_width_s,
        });
    }

    let mut finished = Vec::new();
    let mut open = BinaryHeap::new();
    let route = |work: WorkColumn, open: &mut BinaryHeap<WorkColumn>| {
        if work.peak_likelihood >= config.region_prune_likelihood {
            open.push(work);
        }
    };
    for (column, slabs) in grid.columns.iter().zip(slabs_per_column) {
        route(WorkColumn::new(*column, slabs, far_field_area_per_radian_m2), &mut open);
    }

    let mut cells_evaluated = 0_usize;
    let mut is_complete = true;
    while let Some(work) = open.pop() {
        let (footprint_radius, range_m) = footprint_radius_m(&work.column, frame);
        let target_radius = config.region_minimum_cell_radius_m.max(config.region_relative_cell_radius * range_m);
        let widest_live_interval_s = work
            .slabs
            .iter()
            .filter(|slab| slab.likelihood >= config.region_prune_likelihood)
            .map(|slab| slab.widest_interval_half_width_s)
            .fold(0.0, f64::max);
        let is_resolved = footprint_radius <= target_radius || widest_live_interval_s <= config.region_lag_resolution_s;
        if is_resolved || cells_evaluated >= config.maximum_region_cell_evaluations {
            is_complete &= is_resolved;
            finished.push(finish(work, config.region_prune_likelihood));
            continue;
        }
        for column in split_footprint(&work.column) {
            let mut slabs = Vec::new();
            for parent in work.slabs.iter().filter(|slab| slab.likelihood >= config.region_prune_likelihood) {
                // Keep cells roughly isotropic: split height/elevation only
                // while the slab is thicker than the footprint.
                let slab_extent_m = if column.is_far_field {
                    (parent.bounds[1] - parent.bounds[0]) * column.radial_m[0]
                } else {
                    parent.bounds[1] - parent.bounds[0]
                };
                let middle = 0.5 * (parent.bounds[0] + parent.bounds[1]);
                let pieces: &[[f64; 2]] = if slab_extent_m > 2.0 * footprint_radius {
                    &[[parent.bounds[0], middle], [middle, parent.bounds[1]]]
                } else {
                    &[parent.bounds]
                };
                let weight_share = parent.marginal_weight_m / pieces.len() as f64;
                for bounds in pieces {
                    let envelope = scorer.envelope_score(&column_cell(&column, *bounds), frame);
                    cells_evaluated += 1;
                    slabs.push(Slab {
                        bounds: *bounds,
                        marginal_weight_m: weight_share,
                        likelihood: likelihood_of(envelope.score),
                        widest_interval_half_width_s: envelope.widest_interval_half_width_s,
                    });
                }
            }
            route(WorkColumn::new(column, slabs, far_field_area_per_radian_m2), &mut open);
        }
    }
    RegionRefinement { columns: finished, cells_evaluated, is_complete }
}
