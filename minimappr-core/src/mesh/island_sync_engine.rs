//! Host-side acoustic islands, each with its own clock domain, clock
//! filters, layout and map anchor (plan §Group splits and acoustic islands).
//!
//! Hearing edges accumulate across rounds (islands chirp in their own
//! rounds, and closure needs both directions); a merge or split starts a
//! fresh clock epoch for every member, so no clock model from before it is
//! reused. Every node carries a clock epoch that also bumps on invalidation
//! or a detected silent gap, which is what the clip fetcher checks to never
//! use a stale clip against a new clock model.

use std::collections::{BTreeMap, BTreeSet};

use crate::layout::gps_anchor_fit::{
    chirality_from_bearing, fit_layout_to_gps, AnchorFit, ChiralityCue, GpsFix,
};
use crate::protocol::mesh_messages::ChirpDirectionReport;
use crate::protocol::session_messages::ChirpArrivalReport;
use crate::protocol::NodeId;
use crate::sync::acoustic_islands::{acoustic_components, AcousticIslandTracker, HearingObservation, IslandEvent};
use crate::sync::clock_filter::{ClockUpdateOutcome, PhoneClockFilter};

use super::clock_mapping::NodeClockMapping;
use super::config::IslandSyncConfig;
use super::island_round_solver::IslandRoundSolution;
use super::MeshMilliseconds;

const MILLISECONDS_PER_SECOND: f64 = 1_000.0;

#[derive(Debug, Clone, PartialEq)]
pub struct IslandLayout {
    pub captured_ms: MeshMilliseconds,
    pub local_positions_m: BTreeMap<NodeId, [f64; 3]>,
    pub sigmas_m: BTreeMap<NodeId, f64>,
}

/// Minimum common nodes for a reflection test between two layouts.
const MINIMUM_COMMON_NODES_FOR_REFLECTION_TEST: usize = 3;
/// Cross-covariance determinants below this (m⁴) mean a degenerate overlap.
const REFLECTION_TEST_DETERMINANT_FLOOR_M4: f64 = 1e-6;

/// Whether `new` is a mirror image of `old` over their common nodes
/// (sign of the 2×2 cross-covariance determinant); `None` if degenerate.
pub fn layouts_are_reflected(old: &IslandLayout, new: &IslandLayout) -> Option<bool> {
    let common: Vec<([f64; 3], [f64; 3])> = old
        .local_positions_m
        .iter()
        .filter_map(|(node, a)| new.local_positions_m.get(node).map(|b| (*a, *b)))
        .collect();
    if common.len() < MINIMUM_COMMON_NODES_FOR_REFLECTION_TEST {
        return None;
    }
    let count = common.len() as f64;
    let mean = |pick: fn(&([f64; 3], [f64; 3])) -> [f64; 3]| {
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

#[derive(Debug, Clone)]
pub struct IslandState {
    pub island_id: u32,
    pub tracker_epoch: u64,
    pub members: BTreeSet<NodeId>,
    pub reference_node: Option<NodeId>,
    pub filters: BTreeMap<NodeId, PhoneClockFilter>,
    pub last_reference_time_s: f64,
    pub layout: Option<IslandLayout>,
    pub previous_layout: Option<IslandLayout>,
    pub layout_epoch: u32,
    pub chirality_cue: ChiralityCue,
    pub anchor: Option<AnchorFit>,
}

/// A node's position for a solve, in the anchored (or layout) frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodePositionForSolve {
    pub position_m: [f64; 3],
    pub std_m: f64,
}

#[derive(Debug, Clone, Default)]
pub struct IslandSyncEngine {
    hearing: BTreeMap<(NodeId, NodeId), (f64, MeshMilliseconds)>,
    tracker: AcousticIslandTracker,
    pub islands: BTreeMap<u32, IslandState>,
    node_clock_epochs: BTreeMap<NodeId, u32>,
    next_clock_epoch: u32,
    pub listen_only_nodes: BTreeSet<NodeId>,
}

impl IslandSyncEngine {
    fn bump_epoch(&mut self, node: NodeId) {
        self.next_clock_epoch += 1;
        self.node_clock_epochs.insert(node, self.next_clock_epoch);
    }

    pub fn node_clock_epoch(&self, node: NodeId) -> u32 {
        self.node_clock_epochs.get(&node).copied().unwrap_or(0)
    }

    /// Edges heard this round are refreshed; an emitter a reporting
    /// listener did not hear loses its edge at once (a split shows up in
    /// one round, not after the memory expires).
    pub fn record_hearing(
        &mut self,
        reports: &BTreeMap<NodeId, ChirpArrivalReport>,
        emission_order: &[NodeId],
        now_ms: MeshMilliseconds,
    ) {
        for report in reports.values() {
            for emitter in emission_order {
                let heard = report.arrivals.iter().any(|arrival| arrival.emitter_node_id == *emitter);
                if !heard {
                    self.hearing.remove(&(report.reporting_node_id, *emitter));
                }
            }
            for arrival in &report.arrivals {
                if arrival.emitter_node_id != report.reporting_node_id {
                    self.hearing.insert(
                        (report.reporting_node_id, arrival.emitter_node_id),
                        (arrival.peak_to_noise_db as f64, now_ms),
                    );
                }
            }
        }
    }

    /// Recompute islands over `nodes`; listen-only nodes need only hear.
    pub fn recompute_islands(
        &mut self,
        nodes: &[NodeId],
        now_ms: MeshMilliseconds,
        config: &IslandSyncConfig,
    ) -> Vec<IslandEvent> {
        self.hearing
            .retain(|_, (_, heard_ms)| now_ms.saturating_sub(*heard_ms) <= config.hearing_memory_ms);
        let snr = |listener: NodeId, emitter: NodeId| self.hearing.get(&(listener, emitter)).map(|(s, _)| *s);
        let mut observations = Vec::new();
        for (i, &a) in nodes.iter().enumerate() {
            for &b in &nodes[i + 1..] {
                let mut a_hears_b = snr(a, b);
                let mut b_hears_a = snr(b, a);
                if self.listen_only_nodes.contains(&a) {
                    b_hears_a = b_hears_a.or(a_hears_b);
                }
                if self.listen_only_nodes.contains(&b) {
                    a_hears_b = a_hears_b.or(b_hears_a);
                }
                observations.push(HearingObservation {
                    node_a: a.0,
                    node_b: b.0,
                    snr_a_hears_b_db: a_hears_b,
                    snr_b_hears_a_db: b_hears_a,
                });
            }
        }
        let ids: Vec<u32> = nodes.iter().map(|node| node.0).collect();
        let components = acoustic_components(&ids, &observations, config.hearing_minimum_snr_db);
        let events = self.tracker.update(components);
        let mut refreshed = BTreeMap::new();
        let tracked: Vec<_> = self.tracker.islands().to_vec();
        for island in tracked {
            let island_id = island.island_id as u32;
            let members: BTreeSet<NodeId> = island.members.iter().map(|id| NodeId(*id)).collect();
            let state = match self.islands.remove(&island_id) {
                Some(state) if state.tracker_epoch == island.clock_domain_epoch => IslandState { members, ..state },
                _ => {
                    for member in &members {
                        self.bump_epoch(*member);
                    }
                    IslandState {
                        island_id,
                        tracker_epoch: island.clock_domain_epoch,
                        members,
                        reference_node: None,
                        filters: BTreeMap::new(),
                        last_reference_time_s: 0.0,
                        layout: None,
                        previous_layout: None,
                        layout_epoch: 0,
                        chirality_cue: ChiralityCue::Unknown,
                        anchor: None,
                    }
                }
            };
            refreshed.insert(island_id, state);
        }
        self.islands = refreshed;
        events
    }

    pub fn island_of(&self, node: NodeId) -> Option<&IslandState> {
        self.islands.values().find(|island| island.members.contains(&node))
    }

    fn island_of_mut(&mut self, node: NodeId) -> Option<&mut IslandState> {
        self.islands.values_mut().find(|island| island.members.contains(&node))
    }

    /// TDOA pairs only inside one clock domain.
    pub fn share_clock_domain(&self, a: NodeId, b: NodeId) -> bool {
        match (self.island_of(a), self.island_of(b)) {
            (Some(x), Some(y)) => x.island_id == y.island_id && x.tracker_epoch == y.tracker_epoch,
            _ => false,
        }
    }

    /// Reported gap, input route change, interruption.
    pub fn invalidate_node_clock(&mut self, node: NodeId) {
        if let Some(island) = self.island_of_mut(node) {
            if let Some(filter) = island.filters.get_mut(&node) {
                filter.invalidate();
            }
            if island.reference_node == Some(node) {
                island.reference_node = None;
                island.filters.clear();
                let members: Vec<NodeId> = island.members.iter().copied().collect();
                for member in members {
                    self.bump_epoch(member);
                }
                return;
            }
        }
        self.bump_epoch(node);
    }

    /// Feed one island's round solution into its filters and layout.
    pub fn apply_round_solution(
        &mut self,
        island_id: u32,
        solution: &IslandRoundSolution,
        now_ms: MeshMilliseconds,
        config: &IslandSyncConfig,
    ) {
        let Some(island) = self.islands.get(&island_id) else { return };
        let mut epoch_bumps = Vec::new();
        let reference_changed = island.reference_node.is_some_and(|r| r != solution.reference_node);
        let island = self.islands.get_mut(&island_id).expect("checked above");
        if reference_changed {
            island.filters.clear();
            epoch_bumps.extend(island.members.iter().copied());
        }
        island.reference_node = Some(solution.reference_node);
        island.last_reference_time_s = solution.reference_time_s;
        for (node, (offset_s, sigma_s)) in &solution.clock_offsets {
            if *node == solution.reference_node {
                continue;
            }
            let filter = island
                .filters
                .entry(*node)
                .or_insert_with(|| PhoneClockFilter::new(config.clock_filter.clone()));
            let was_valid = filter.is_valid();
            let outcome = filter.update(solution.reference_time_s, *offset_s, *sigma_s);
            if matches!(outcome, ClockUpdateOutcome::SilentGapDetected { .. }) || !was_valid {
                epoch_bumps.push(*node);
            }
        }
        if !solution.layout_positions_m.is_empty() {
            let mut local_positions_m = solution.layout_positions_m.clone();
            let mut sigmas_m = solution.layout_sigmas_m.clone();
            if let Some(previous) = &island.layout {
                // Members not ranged this round keep their last position with a
                // σ that grows with the time they could have moved.
                let elapsed_s = now_ms.saturating_sub(previous.captured_ms) as f64 / MILLISECONDS_PER_SECOND;
                for (node, position) in &previous.local_positions_m {
                    if island.members.contains(node) && !local_positions_m.contains_key(node) {
                        local_positions_m.insert(*node, *position);
                        let sigma = previous.sigmas_m.get(node).copied().unwrap_or(config.range_sigma_m);
                        sigmas_m.insert(*node, sigma + config.relative_position_growth_mps * elapsed_s);
                    }
                }
            }
            let new_layout = IslandLayout {
                captured_ms: now_ms,
                local_positions_m,
                sigmas_m,
            };
            // MDS picks a chirality per solve; a locked cue must follow it.
            if let (ChiralityCue::Locked { mirrored }, Some(previous)) = (island.chirality_cue, &island.layout) {
                if layouts_are_reflected(previous, &new_layout) == Some(true) {
                    island.chirality_cue = ChiralityCue::Locked { mirrored: !mirrored };
                }
            }
            island.previous_layout = island.layout.replace(new_layout);
            island.layout_epoch += 1;
        }
        for node in epoch_bumps {
            self.bump_epoch(node);
        }
    }

    /// Fit the island's layout to the GPS fixes (both chiralities tried).
    pub fn refresh_anchor(&mut self, island_id: u32, fixes: &BTreeMap<NodeId, GpsFix>, config: &IslandSyncConfig) {
        let Some(island) = self.islands.get_mut(&island_id) else { return };
        let Some(layout) = &island.layout else { return };
        let nodes: Vec<NodeId> = layout.local_positions_m.keys().copied().collect();
        let layout_xy: Vec<[f64; 2]> = nodes
            .iter()
            .map(|node| {
                let p = layout.local_positions_m[node];
                [p[0], p[1]]
            })
            .collect();
        let node_fixes: Vec<Option<GpsFix>> = nodes.iter().map(|node| fixes.get(node).copied()).collect();
        island.anchor = fit_layout_to_gps(&layout_xy, &node_fixes, island.chirality_cue, &config.anchor_fit).ok();
    }

    /// A chirp DOA locks chirality for the island holding both nodes.
    pub fn apply_direction_report(
        &mut self,
        report: &ChirpDirectionReport,
        fixes: &BTreeMap<NodeId, GpsFix>,
        config: &IslandSyncConfig,
    ) -> bool {
        let Some(island) = self.island_of(report.reporting_node_id) else { return false };
        let island_id = island.island_id;
        let Some(layout) = &island.layout else { return false };
        let nodes: Vec<NodeId> = layout.local_positions_m.keys().copied().collect();
        let (Some(from), Some(to)) = (
            nodes.iter().position(|n| *n == report.reporting_node_id),
            nodes.iter().position(|n| *n == report.emitter_node_id),
        ) else {
            return false;
        };
        let layout_xy: Vec<[f64; 2]> = nodes
            .iter()
            .map(|node| {
                let p = layout.local_positions_m[node];
                [p[0], p[1]]
            })
            .collect();
        let node_fixes: Vec<Option<GpsFix>> = nodes.iter().map(|node| fixes.get(node).copied()).collect();
        let fit = |mirrored| {
            fit_layout_to_gps(&layout_xy, &node_fixes, ChiralityCue::Locked { mirrored }, &config.anchor_fit).ok()
        };
        let (Some(proper), Some(mirror)) = (fit(false), fit(true)) else { return false };
        let cue = chirality_from_bearing(
            &layout_xy,
            from,
            to,
            (report.world_azimuth_deg as f64).to_radians(),
            &proper.transform,
            &mirror.transform,
        );
        if let Some(island) = self.islands.get_mut(&island_id) {
            island.chirality_cue = cue;
        }
        self.refresh_anchor(island_id, fixes, config);
        true
    }

    /// Device clock → island reference clock for `node`, if its model is valid.
    pub fn clock_mapping(&self, node: NodeId) -> Option<NodeClockMapping> {
        let island = self.island_of(node)?;
        let epoch = self.node_clock_epoch(node);
        if island.reference_node == Some(node) {
            return Some(NodeClockMapping::identity(epoch, island.last_reference_time_s));
        }
        island.reference_node?;
        let filter = island.filters.get(&node)?;
        let prediction = filter.predict(island.last_reference_time_s)?;
        Some(NodeClockMapping::from_prediction(epoch, island.last_reference_time_s, &prediction))
    }

    fn anchored(island: &IslandState, local: [f64; 3]) -> [f64; 3] {
        match &island.anchor {
            Some(anchor) => {
                let xy = anchor.transform.apply([local[0], local[1]]);
                [xy[0], xy[1], local[2]]
            }
            None => local,
        }
    }

    /// Position for a solve. A node that walked at call time gets the layout
    /// captured before the call, never a later one (plan: geometry goes
    /// backward only for phones that were still).
    pub fn node_position_for_solve(
        &self,
        node: NodeId,
        call_ms: MeshMilliseconds,
        walked_at_call: bool,
        config: &IslandSyncConfig,
    ) -> Option<NodePositionForSolve> {
        let island = self.island_of(node)?;
        let latest = island.layout.as_ref()?;
        let layout = if walked_at_call && latest.captured_ms > call_ms {
            island.previous_layout.as_ref().unwrap_or(latest)
        } else {
            latest
        };
        let local = *layout.local_positions_m.get(&node)?;
        let base_sigma = layout.sigmas_m.get(&node).copied().unwrap_or(config.range_sigma_m);
        let sigma = if walked_at_call {
            let elapsed_s = call_ms.abs_diff(layout.captured_ms) as f64 / MILLISECONDS_PER_SECOND;
            base_sigma + config.relative_position_growth_mps * elapsed_s
        } else {
            base_sigma
        };
        Some(NodePositionForSolve {
            position_m: Self::anchored(island, local),
            std_m: sigma.max(config.range_sigma_m),
        })
    }

    pub fn chirality_confidence(&self, node: NodeId) -> Option<f64> {
        self.island_of(node)?.anchor.as_ref().map(|anchor| anchor.chirality_confidence)
    }

    pub fn chirality_locked(&self, node: NodeId) -> bool {
        self.island_of(node)
            .is_some_and(|island| matches!(island.chirality_cue, ChiralityCue::Locked { .. }))
    }
}
