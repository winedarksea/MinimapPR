//! `LayoutSrpSearcher`: geometry-only state built once per layout (grid,
//! per-cell TDOA intervals), reused across clips and onset frames.

use std::f64::consts::PI;

use super::config::LayoutSrpConfig;
use super::layout_geometry::{horizontal_dilution, horizontal_scatter, HorizontalScatter};
use super::layout_input::{
    align_clips_to_island_window, layout_sensors, plan_sensor_pairs, validate_layout_inputs, IslandAnalysisWindow,
    LayoutNode, LayoutSensor, LayoutSrpError, NodeClip, PlannedPair,
};
use super::likelihood_region::{build_likelihood_region, choose_uncertainty_display};
use super::lm_handoff::build_lm_seed;
use super::region_refinement::refine_region_columns;
use super::pair_correlation::compute_pair_coherence_curves;
use super::polar_grid::{CoarsePolarGrid, LayoutFrame, SearchCell};
use super::refinement::{refine_branch_and_bound, RefinedEstimate, ScoredCell};
use super::robust_scoring::{pair_significance, CellScore, PairSubset, RobustPairScorer, WeightedPair};
use super::{LayoutGeometryDiagnostics, LayoutSrpResult, MirrorCandidates, SourceEstimate};

/// The side chosen as primary plus its gaps over the other side.
struct OrderedMirrorWinner {
    estimate: RefinedEstimate,
    score_gap_temperatures: f64,
    array_score_gap_temperatures: f64,
}

/// Pick the primary side of a collinear layout.
///
/// Only rigid-array pairs may break the tie: on a near-line the inter-node
/// cross-trail aperture (tens of cm) is within the layout solve's own
/// position error, so its preference is not trustworthy even when the score
/// gap is large. So when the array pairs decisively favour one side, that
/// side is primary even if the all-pair score (driven by phone-position
/// jitter) prefers the other; otherwise the all-pair winner is primary and
/// the result stays unresolved. The array gap is measured in the array
/// pairs' own temperature (T · Σw_array) so the threshold does not depend on
/// how many phones are present.
fn order_mirror_candidates(
    left: RefinedEstimate,
    right: RefinedEstimate,
    scorer: &RobustPairScorer,
    frame: &LayoutFrame,
    config: &LayoutSrpConfig,
) -> (OrderedMirrorWinner, RefinedEstimate) {
    let scale = scorer.likelihood_scale();
    let array_weight: f64 =
        scorer.pairs.iter().filter(|pair| pair.sensor_a_node == pair.sensor_b_node).map(|pair| pair.weight).sum();
    let array_gap_of = |primary: &RefinedEstimate, other: &RefinedEstimate| {
        if array_weight > 0.0 {
            (scorer.exact_cell_score_for(&primary.cell, frame, PairSubset::IntraNodeOnly)
                - scorer.exact_cell_score_for(&other.cell, frame, PairSubset::IntraNodeOnly))
                / (scorer.likelihood_temperature * array_weight)
        } else {
            0.0
        }
    };
    let left_array_gap = array_gap_of(&left, &right);
    let resolution_gap = config.uncertainty.mirror_resolution_score_gap_temperatures;
    let left_is_primary = if left_array_gap.abs() > resolution_gap {
        left_array_gap > 0.0
    } else {
        left.exact_score >= right.exact_score
    };
    let (primary, other, array_gap) =
        if left_is_primary { (left, right, left_array_gap) } else { (right, left, -left_array_gap) };
    let winner = OrderedMirrorWinner {
        estimate: primary,
        score_gap_temperatures: (primary.exact_score - other.exact_score) / scale,
        array_score_gap_temperatures: array_gap,
    };
    (winner, other)
}

pub struct LayoutSrpSearcher {
    nodes: Vec<LayoutNode>,
    sensors: Vec<LayoutSensor>,
    pairs: Vec<PlannedPair>,
    scatter: HorizontalScatter,
    frame: LayoutFrame,
    grid: CoarsePolarGrid,
    /// (centre lag, half-width) per cell × pair, row-major by cell.
    cell_pair_intervals_s: Vec<(f32, f32)>,
    config: LayoutSrpConfig,
}

impl LayoutSrpSearcher {
    pub fn new(nodes: &[LayoutNode], config: &LayoutSrpConfig) -> Result<Self, LayoutSrpError> {
        validate_layout_inputs(nodes, config)?;
        let sensors = layout_sensors(nodes);
        let pairs = plan_sensor_pairs(nodes, &sensors, config);
        if sensors.len() < 2 || pairs.is_empty() {
            return Err(LayoutSrpError::TooFewSensors);
        }
        let node_positions: Vec<[f64; 3]> = nodes.iter().map(|node| node.position_m).collect();
        let scatter = horizontal_scatter(&node_positions);
        let ground_height_m = config.ground_height_m.unwrap_or_else(|| {
            sensors.iter().map(|sensor| sensor.position_m[2]).fold(f64::INFINITY, f64::min)
                - config.grid.assumed_sensor_height_above_ground_m
        });
        let frame = scatter.frame(ground_height_m);
        let grid = CoarsePolarGrid::build(&config.grid);
        let mut cell_pair_intervals_s = Vec::with_capacity(grid.cells.len() * pairs.len());
        for cell in &grid.cells {
            let geometry = cell.geometry(&frame);
            for pair in &pairs {
                let (center, half_width) = geometry.tdoa_interval_s(
                    sensors[pair.spec.sensor_a].position_m,
                    sensors[pair.spec.sensor_b].position_m,
                    config.sound_speed_mps,
                );
                cell_pair_intervals_s.push((center as f32, half_width as f32));
            }
        }
        Ok(Self {
            nodes: nodes.to_vec(),
            sensors,
            pairs,
            scatter,
            frame,
            grid,
            cell_pair_intervals_s,
            config: config.clone(),
        })
    }

    pub fn frame(&self) -> &LayoutFrame {
        &self.frame
    }

    pub fn coarse_cell_count(&self) -> usize {
        self.grid.cells.len()
    }

    pub fn pair_count(&self) -> usize {
        self.pairs.len()
    }

    fn build_scorer(&self, clips: &[NodeClip<'_>], window: IslandAnalysisWindow) -> Result<RobustPairScorer, LayoutSrpError> {
        let aligned = align_clips_to_island_window(&self.nodes, clips, window, &self.config)?;
        let specs: Vec<_> = self.pairs.iter().map(|pair| pair.spec).collect();
        let curves = compute_pair_coherence_curves(&aligned, &specs, self.config.sample_rate_hz, &self.config.correlation);
        let weighted: Vec<WeightedPair> = curves
            .into_iter()
            .zip(&self.pairs)
            .map(|(curve, planned)| {
                let significance = pair_significance(
                    &curve,
                    self.config.correlation.upsample_factor,
                    self.config.scoring.full_weight_significance_ratio,
                );
                let (node_a, node_b) = (self.sensors[planned.spec.sensor_a].node_index, self.sensors[planned.spec.sensor_b].node_index);
                let declared_lag_tolerance_s = if node_a == node_b {
                    0.0
                } else {
                    self.config.scoring.declared_sigma_pooling_multiplier
                        * self.nodes[node_a].pair_lag_std_s(&self.nodes[node_b], self.config.sound_speed_mps)
                };
                WeightedPair {
                    sensor_a_position_m: self.sensors[planned.spec.sensor_a].position_m,
                    sensor_b_position_m: self.sensors[planned.spec.sensor_b].position_m,
                    sensor_a_node: node_a,
                    sensor_b_node: node_b,
                    weight: planned.base_weight * significance,
                    declared_lag_tolerance_s,
                    curve,
                }
            })
            .collect();
        let scorer = RobustPairScorer::new(weighted, &self.config.scoring, self.config.sound_speed_mps);
        if scorer.total_weight <= 0.0 {
            return Err(LayoutSrpError::NoCoherentPairs);
        }
        Ok(scorer)
    }

    fn coarse_scores(&self, scorer: &RobustPairScorer) -> Vec<ScoredCell> {
        let pair_count = self.pairs.len();
        self.grid
            .cells
            .iter()
            .enumerate()
            .map(|(cell_index, cell)| {
                let row = &self.cell_pair_intervals_s[cell_index * pair_count..(cell_index + 1) * pair_count];
                ScoredCell {
                    cell: *cell,
                    envelope: scorer.envelope_score_from_intervals(
                        row.iter().map(|&(center, half_width)| (center as f64, half_width as f64)),
                    ),
                }
            })
            .collect()
    }

    fn source_estimate(&self, refined: &RefinedEstimate, scorer: &RobustPairScorer) -> SourceEstimate {
        let position_m = refined.cell.representative_position_m(&self.frame, self.config.grid.outer_radius_m);
        let dx = position_m[0] - self.frame.origin_xy_m[0];
        let dy = position_m[1] - self.frame.origin_xy_m[1];
        SourceEstimate {
            position_m,
            is_far_field: refined.cell.is_far_field(),
            world_azimuth_rad: dy.atan2(dx),
            horizontal_range_m: dx.hypot(dy),
            srp_score: refined.exact_score,
            normalized_srp_score: refined.exact_score / scorer.total_weight,
            cells_evaluated: refined.cells_evaluated,
            is_certified_within_tolerance: refined.is_certified_within_tolerance,
        }
    }

    fn far_field_direction(&self, cell: &SearchCell) -> Option<[f64; 3]> {
        match *cell {
            SearchCell::FarField { azimuth_rad, elevation_rad } => Some(self.frame.direction_world(
                0.5 * (azimuth_rad[0] + azimuth_rad[1]),
                0.5 * (elevation_rad[0] + elevation_rad[1]),
            )),
            SearchCell::Near { .. } => None,
        }
    }

    /// Exact robust score at each position (diagnostics and tests: compare
    /// truth against the estimate to tell search misses from model misses).
    pub fn exact_scores_at(
        &self,
        clips: &[NodeClip<'_>],
        window: IslandAnalysisWindow,
        positions_m: &[[f64; 3]],
    ) -> Result<Vec<f64>, LayoutSrpError> {
        let scorer = self.build_scorer(clips, window)?;
        Ok(positions_m.iter().map(|position| scorer.exact_score_at(*position)).collect())
    }

    /// Localize one analysis window.
    pub fn localize(&self, clips: &[NodeClip<'_>], window: IslandAnalysisWindow) -> Result<LayoutSrpResult, LayoutSrpError> {
        let scorer = self.build_scorer(clips, window)?;
        let coarse = self.coarse_scores(&scorer);
        let is_collinear = self.scatter.aspect_ratio() < self.config.uncertainty.collinear_aspect_ratio;

        let refine = |keep: &dyn Fn(&SearchCell) -> bool| -> Option<RefinedEstimate> {
            let seeds: Vec<ScoredCell> = coarse.iter().filter(|scored| keep(&scored.cell)).copied().collect();
            refine_branch_and_bound(&seeds, &scorer, &self.frame, &self.config.refinement)
        };
        let is_left_of_axis = |cell: &SearchCell| {
            let [low, high] = cell.layout_azimuth_bounds_rad();
            0.5 * (low + high) < PI
        };
        let (best, mirror) = if is_collinear {
            let left = refine(&|cell| is_left_of_axis(cell));
            let right = refine(&|cell| !is_left_of_axis(cell));
            match (left, right) {
                (Some(left), Some(right)) => {
                    let (winner, loser) = order_mirror_candidates(left, right, &scorer, &self.frame, &self.config);
                    let mirror = MirrorCandidates {
                        primary: self.source_estimate(&winner.estimate, &scorer),
                        mirror: self.source_estimate(&loser, &scorer),
                        score_gap_temperatures: winner.score_gap_temperatures,
                        array_score_gap_temperatures: winner.array_score_gap_temperatures,
                        is_resolved: winner.array_score_gap_temperatures
                            > self.config.uncertainty.mirror_resolution_score_gap_temperatures,
                    };
                    (winner.estimate, Some(mirror))
                }
                (Some(only), None) | (None, Some(only)) => (only, None),
                (None, None) => return Err(LayoutSrpError::NoCoherentPairs),
            }
        } else {
            (refine(&|_| true).ok_or(LayoutSrpError::NoCoherentPairs)?, None)
        };

        let estimate = self.source_estimate(&best, &scorer);
        let [lowest_height, highest_height] = self.config.grid.source_height_above_ground_range_m;
        let lm_seed = build_lm_seed(
            &scorer,
            estimate.position_m,
            self.far_field_direction(&best.cell),
            best.exact_score,
            [self.frame.ground_height_m + lowest_height, self.frame.ground_height_m + highest_height],
            self.config.refinement.handoff_peak_search_half_window_s,
        );

        let coarse_scores: Vec<CellScore> = coarse.iter().map(|scored| scored.envelope).collect();
        let reference_score = mirror
            .map(|candidates| candidates.primary.srp_score)
            .unwrap_or(best.exact_score);
        let region_refinement = refine_region_columns(
            &self.grid,
            &coarse_scores,
            reference_score,
            &scorer,
            &self.frame,
            &self.config.uncertainty,
        );
        let region = build_likelihood_region(&region_refinement, &self.grid, &self.frame, &self.config.uncertainty);
        let layout_aperture_m = 2.0
            * self
                .nodes
                .iter()
                .map(|node| (node.position_m[0] - self.frame.origin_xy_m[0]).hypot(node.position_m[1] - self.frame.origin_xy_m[1]))
                .fold(0.0_f64, f64::max);
        let display = choose_uncertainty_display(
            &region,
            estimate.is_far_field,
            estimate.world_azimuth_rad,
            estimate.horizontal_range_m,
            layout_aperture_m,
            &self.config.uncertainty,
        );
        let weighted_pair_positions: Vec<_> = scorer
            .pairs
            .iter()
            .filter(|pair| pair.weight > 0.0)
            .map(|pair| (pair.sensor_a_position_m, pair.sensor_b_position_m))
            .collect();
        let geometry = LayoutGeometryDiagnostics {
            horizontal_aspect_ratio: self.scatter.aspect_ratio(),
            is_collinear,
            layout_aperture_m,
            dilution_at_estimate: horizontal_dilution(estimate.position_m, self.frame.origin_xy_m, &weighted_pair_positions),
        };
        Ok(LayoutSrpResult {
            estimate,
            lm_seed,
            mirror,
            geometry,
            region,
            display,
            frame: self.frame,
            coherent_pair_count: weighted_pair_positions.len(),
            total_pair_weight: scorer.total_weight,
        })
    }
}
