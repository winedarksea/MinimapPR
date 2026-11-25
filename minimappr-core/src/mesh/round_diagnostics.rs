//! Per-round sync diagnostics the host keeps after `finish_round`: what each
//! island's solve ranged, the raw per-round offsets, how each clock filter
//! took them and the filtered rate. The round solution itself is dropped
//! after it updates the engine; phones show this as their live status and
//! the two-phone e2e script compares it with the offline analyser.

use crate::protocol::NodeId;
use crate::sync::clock_filter::ClockUpdateOutcome;

use super::island_round_solver::IslandRoundSolution;
use super::island_sync_engine::IslandSyncEngine;
use super::MeshMilliseconds;

const PARTS_PER_MILLION: f64 = 1e6;

#[derive(Debug, Clone, PartialEq)]
pub struct RangedPairDiagnostics {
    pub first_node: NodeId,
    pub second_node: NodeId,
    pub distance_m: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PeerClockDiagnostics {
    pub node_id: NodeId,
    /// This round's raw θ = local − reference, with its solve σ.
    pub round_offset_s: f64,
    pub round_offset_sigma_s: f64,
    /// `None` for the reference node (it has no filter).
    pub filter_outcome: Option<ClockUpdateOutcome>,
    /// Filtered offset/rate after this round's update.
    pub filtered_offset_s: Option<f64>,
    pub filtered_offset_sigma_s: Option<f64>,
    pub rate_offset_ppm: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RoundDiagnostics {
    pub round_id: u32,
    pub island_id: u32,
    pub completed_ms: MeshMilliseconds,
    pub reference_node: NodeId,
    pub reference_time_s: f64,
    pub reports_received: usize,
    pub ranged_pairs: Vec<RangedPairDiagnostics>,
    pub peers: Vec<PeerClockDiagnostics>,
}

pub struct RoundDiagnosticsInput<'round> {
    pub round_id: u32,
    pub island_id: u32,
    pub completed_ms: MeshMilliseconds,
    pub reports_received: usize,
    pub solution: &'round IslandRoundSolution,
    pub filter_outcomes: &'round [(NodeId, ClockUpdateOutcome)],
}

/// Read after `apply_round_solution`, so filtered values include this round.
pub fn build_round_diagnostics(input: RoundDiagnosticsInput<'_>, engine: &IslandSyncEngine) -> RoundDiagnostics {
    let solution = input.solution;
    let island = engine.islands.get(&input.island_id);
    let peers = solution
        .clock_offsets
        .iter()
        .map(|(node, (offset_s, sigma_s))| {
            let prediction = island
                .and_then(|island| island.filters.get(node))
                .and_then(|filter| filter.predict(solution.reference_time_s));
            PeerClockDiagnostics {
                node_id: *node,
                round_offset_s: *offset_s,
                round_offset_sigma_s: *sigma_s,
                filter_outcome: input
                    .filter_outcomes
                    .iter()
                    .find(|(filtered_node, _)| filtered_node == node)
                    .map(|(_, outcome)| *outcome),
                filtered_offset_s: prediction.map(|p| p.offset_s),
                filtered_offset_sigma_s: prediction.map(|p| p.offset_sigma_s),
                rate_offset_ppm: prediction.map(|p| p.rate_offset * PARTS_PER_MILLION),
            }
        })
        .collect();
    RoundDiagnostics {
        round_id: input.round_id,
        island_id: input.island_id,
        completed_ms: input.completed_ms,
        reference_node: solution.reference_node,
        reference_time_s: solution.reference_time_s,
        reports_received: input.reports_received,
        ranged_pairs: solution
            .ranged_pairs
            .iter()
            .map(|(first_node, second_node, distance_m)| RangedPairDiagnostics {
                first_node: *first_node,
                second_node: *second_node,
                distance_m: *distance_m,
            })
            .collect(),
        peers,
    }
}
