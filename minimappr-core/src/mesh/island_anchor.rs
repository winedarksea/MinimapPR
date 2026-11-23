//! Each island's map anchor: every round's layout is fitted to the members'
//! GPS fixes and fused through the island's [`FilteredMapAnchor`] (clamped
//! while the group is still, a still phone's multipath jump subtracted from
//! its own fixes), so positions the host publishes no longer swing with each
//! round's raw fit. Chirality cues from chirp bearings live here too.

use std::collections::BTreeMap;

use crate::layout::filtered_map_anchor::AnchorRound;
use crate::layout::gps_anchor_fit::{ChiralityCue, GpsFix};
use crate::protocol::mesh_messages::ChirpDirectionReport;
use crate::protocol::NodeId;

use super::chirality_cue::chirality_from_bearing_pair;
use super::config::IslandSyncConfig;
use super::fix_contour::{AnchorUncertainty, RegionContext};
use super::island_sync_engine::{IslandLayout, IslandState, IslandSyncEngine, MILLISECONDS_PER_SECOND};
use super::MeshMilliseconds;

/// A bearing pair locks chirality only if the two chiralities' predicted
/// turn angles differ by more than this many σ.
const DECISIVE_CUE_SIGMA_MULTIPLIER: f64 = 3.0;
/// Bearings kept to pair with the next one from the same reporter.
const REMEMBERED_DIRECTION_REPORTS: usize = 32;

impl IslandSyncEngine {
    /// Fit the island's current layout to the GPS fixes (both chiralities
    /// tried) and fuse it into the island's filtered anchor. Call after
    /// `apply_round_solution`, which re-frames the filter to the new solve.
    pub fn refresh_anchor(
        &mut self,
        island_id: u32,
        fixes: &BTreeMap<NodeId, GpsFix>,
        now_ms: MeshMilliseconds,
        group_is_still: bool,
        config: &IslandSyncConfig,
    ) {
        let Some(island) = self.islands.get_mut(&island_id) else { return };
        let Some(layout) = &island.layout else { return };
        island.group_is_still = group_is_still;
        let nodes: Vec<NodeId> = layout.local_positions_m.keys().copied().collect();
        let layout_xy: Vec<[f64; 2]> = nodes
            .iter()
            .map(|node| {
                let p = layout.local_positions_m[node];
                [p[0], p[1]]
            })
            .collect();
        let node_fixes: Vec<Option<GpsFix>> = nodes.iter().map(|node| fixes.get(node).copied()).collect();
        let round = AnchorRound {
            time_s: now_ms as f64 / MILLISECONDS_PER_SECOND,
            nodes: &nodes,
            layout_xy: &layout_xy,
            fixes: &node_fixes,
            cue: island.chirality_cue,
            group_is_still,
        };
        // A failed fit (too few fixes) leaves the filtered anchor coasting.
        let _ = island.anchor_tracker.update(&round, &config.anchor_fit);
        island.anchor = island.anchor_tracker.filtered_fit();
        if let Some(layout) = island.layout.as_mut() {
            layout.anchor = island.anchor.clone();
        }
    }

    /// Chirp bearings from one stereo phone or array node lock chirality
    /// once two of them in one round form a decisive pair ([`chirality_cue`]).
    /// A lock that contradicts the filtered anchor restarts it on a fresh fit;
    /// one that agrees only marks it locked (re-fusing the same fixes would
    /// count them twice).
    ///
    /// [`chirality_cue`]: super::chirality_cue
    pub fn apply_direction_report(
        &mut self,
        report: &ChirpDirectionReport,
        fixes: &BTreeMap<NodeId, GpsFix>,
        config: &IslandSyncConfig,
    ) -> bool {
        let Some(island) = self.island_of(report.reporting_node_id) else { return false };
        let island_id = island.island_id;
        let Some(layout) = &island.layout else { return false };
        let noise = config.relative_bearing_noise_std_deg.to_radians();
        let cue = self
            .recent_direction_reports
            .iter()
            .rev()
            .find_map(|earlier| chirality_from_bearing_pair(&layout.local_positions_m, earlier, report, noise, DECISIVE_CUE_SIGMA_MULTIPLIER));
        self.recent_direction_reports.push(report.clone());
        if self.recent_direction_reports.len() > REMEMBERED_DIRECTION_REPORTS {
            self.recent_direction_reports.remove(0);
        }
        let Some(cue) = cue else { return false };
        let Some(island) = self.islands.get_mut(&island_id) else { return false };
        island.chirality_cue = cue;
        let ChiralityCue::Locked { mirrored } = cue else { return true };
        let agrees = island.anchor_tracker.estimate().is_some_and(|estimate| estimate.transform.mirrored == mirrored);
        if agrees {
            if let Some(anchor) = island.anchor.as_mut() {
                anchor.chirality_confidence = 1.0;
            }
            if let Some(layout) = island.layout.as_mut() {
                layout.anchor = island.anchor.clone();
            }
            return true;
        }
        island.anchor_tracker.reset();
        let (captured_ms, group_is_still) = (island.layout.as_ref().map_or(0, |layout| layout.captured_ms), island.group_is_still);
        self.refresh_anchor(island_id, fixes, captured_ms, group_is_still, config);
        true
    }

    pub(super) fn anchored(layout: &IslandLayout, local: [f64; 3]) -> [f64; 3] {
        match &layout.anchor {
            Some(anchor) => {
                let xy = anchor.transform.apply([local[0], local[1]]);
                [xy[0], xy[1], local[2]]
            }
            None => local,
        }
    }

    /// `node`'s island extent and filtered anchor error, for published
    /// regions (`fix_contour::published_covariance`).
    pub fn region_context(&self, node: NodeId) -> Option<RegionContext> {
        self.island_of(node).and_then(Self::island_region_context)
    }

    /// Centroid and circumradius of the island's layout in the frame fixes
    /// are solved in (anchored when an anchor exists), plus the anchor error.
    pub fn island_region_context(island: &IslandState) -> Option<RegionContext> {
        let layout = island.layout.as_ref()?;
        let positions: Vec<[f64; 3]> = layout.local_positions_m.values().map(|local| Self::anchored(layout, *local)).collect();
        let extent = RegionContext::from_layout_positions(&positions);
        let anchor = layout.anchor.as_ref().map(|anchor| AnchorUncertainty {
            translation_sigma_m: anchor.translation_sigma_m,
            rotation_sigma_rad: anchor.rotation_sigma_rad,
            rotation_centre_m: extent.layout_centre_m,
        });
        Some(RegionContext { anchor, ..extent })
    }

    pub fn chirality_confidence(&self, node: NodeId) -> Option<f64> {
        self.island_of(node)?.anchor.as_ref().map(|anchor| anchor.chirality_confidence)
    }

    pub fn chirality_locked(&self, node: NodeId) -> bool {
        self.island_of(node)
            .is_some_and(|island| matches!(island.chirality_cue, ChiralityCue::Locked { .. }))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use crate::layout::filtered_map_anchor::FilteredMapAnchor;

    const LAYOUT: [[f64; 2]; 5] = [[0.0, 0.0], [14.0, 2.0], [6.0, 11.0], [-5.0, 8.0], [9.0, -9.0]];
    const ROUND_INTERVAL_MS: u64 = 5_000;

    fn engine_with_layout(config: &IslandSyncConfig) -> IslandSyncEngine {
        let nodes: Vec<NodeId> = (1..=LAYOUT.len() as u32).map(NodeId).collect();
        let layout = IslandLayout {
            captured_ms: 0,
            local_positions_m: nodes.iter().zip(LAYOUT).map(|(node, p)| (*node, [p[0], p[1], 1.4])).collect(),
            sigmas_m: nodes.iter().map(|node| (*node, 0.05)).collect(),
            anchor: None,
        };
        let island = IslandState {
            island_id: 1,
            tracker_epoch: 1,
            members: nodes.iter().copied().collect::<BTreeSet<_>>(),
            reference_node: Some(nodes[0]),
            filters: BTreeMap::new(),
            last_reference_time_s: 0.0,
            layout: Some(layout),
            previous_layout: None,
            layout_epoch: 1,
            chirality_cue: ChiralityCue::Locked { mirrored: false },
            anchor: None,
            anchor_tracker: FilteredMapAnchor::new(config.anchor_tracking.clone()),
            group_is_still: true,
        };
        let mut engine = IslandSyncEngine::default();
        engine.islands.insert(1, island);
        engine
    }

    /// Alternating ±3 m fix noise keeps the history honest without an RNG.
    fn fixes(round: u64, jump_east_m: f64) -> BTreeMap<NodeId, GpsFix> {
        LAYOUT
            .iter()
            .enumerate()
            .map(|(index, p)| {
                let wobble = if (round + index as u64) % 2 == 0 { 3.0 } else { -3.0 };
                let jump = if index == 0 { jump_east_m } else { 0.0 };
                (NodeId(index as u32 + 1), GpsFix { east_m: 100.0 + p[0] + wobble + jump, north_m: 50.0 + p[1] - wobble, sigma_m: 6.0 })
            })
            .collect()
    }

    #[test]
    fn published_positions_use_the_filtered_anchor_not_the_round_fit() {
        let config = IslandSyncConfig::default();
        let mut engine = engine_with_layout(&config);
        for round in 0..40 {
            engine.refresh_anchor(1, &fixes(round, 0.0), round * ROUND_INTERVAL_MS, true, &config);
        }
        let settled = engine.node_position_for_solve(NodeId(3), 40 * ROUND_INTERVAL_MS, false, &config).unwrap();
        for round in 40..60 {
            engine.refresh_anchor(1, &fixes(round, 20.0), round * ROUND_INTERVAL_MS, true, &config);
        }
        let after_jump = engine.node_position_for_solve(NodeId(3), 60 * ROUND_INTERVAL_MS, false, &config).unwrap();
        let moved = (after_jump.position_m[0] - settled.position_m[0]).hypot(after_jump.position_m[1] - settled.position_m[1]);
        assert!(moved < 0.5, "a still phone's 20 m GPS jump moved the published map {moved:.2} m");
        assert_eq!(engine.islands[&1].anchor_tracker.held_last_update(), &[NodeId(1)]);
    }
}
