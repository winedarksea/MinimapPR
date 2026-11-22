//! One island's chirp round → clock offsets and a rigid layout, using the
//! `sync` and `layout` building blocks unchanged:
//! arrival reports → `ChirpRoundObservation` → BeepBeep pairs (closing-chirp
//! rate compensation) → two-pass clock graph → `solve_layout`.
//!
//! Emitters flagged muted are removed from the emission list, so a silent
//! phone is excluded from ranging instead of being mis-ranged against noise.

use std::collections::{BTreeMap, BTreeSet};

use crate::layout::robust_layout_refine::{solve_layout, RangeMeasurement};
use crate::protocol::session_messages::ChirpArrivalReport;
use crate::protocol::wire_codes::MotionState;
use crate::protocol::NodeId;
use crate::sync::chirp_round::{clock_offset_constraints, ChirpRoundObservation, RoundRateCompensation};
use crate::sync::clock_graph::solve_clock_graph;

use super::config::{IslandSyncConfig, DEVICE_SAMPLE_RATE_HZ};

/// Linear-acceleration energy assigned to a phone reported walking. With
/// the clock graph's 0.1 m per unit this models ~1 m of displacement over a
/// round (1.2–1.5 m/s across a 1–1.5 s round), so its offsets carry ~1.5 ms σ.
const WALKING_MOTION_ENERGY: f64 = 10.0;
/// Ranges a node needs for a unique 2D position (also the triangle size).
const MINIMUM_RANGES_FOR_UNIQUE_PLACEMENT: usize = 3;

#[derive(Debug, Clone)]
pub struct IslandRoundInput<'round> {
    /// Island members in a stable order.
    pub members: Vec<NodeId>,
    pub reports: &'round BTreeMap<NodeId, ChirpArrivalReport>,
    /// Emitters in slot order; the opener repeats at the end (closing chirp).
    pub emission_order: &'round [NodeId],
    pub self_path_lengths_m: &'round BTreeMap<NodeId, f64>,
    pub muted_emitters: &'round BTreeSet<NodeId>,
    pub preferred_reference: Option<NodeId>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IslandRoundSolution {
    pub reference_node: NodeId,
    /// Reference node's own clock at the round (seconds).
    pub reference_time_s: f64,
    /// θ = local − reference at `reference_time_s`, with σ.
    pub clock_offsets: BTreeMap<NodeId, (f64, f64)>,
    pub ranged_pairs: Vec<(NodeId, NodeId, f64)>,
    pub layout_positions_m: BTreeMap<NodeId, [f64; 3]>,
    pub layout_sigmas_m: BTreeMap<NodeId, f64>,
    /// Participants reported walking: in the clock graph, not in the layout.
    pub walking_nodes: BTreeSet<NodeId>,
}

fn local_seconds(index: u64, fraction: f32) -> f64 {
    (index as f64 + fraction as f64) / DEVICE_SAMPLE_RATE_HZ
}

/// `arrival_local_s[receiver][emission]`: the k-th emission by an emitter
/// is matched to that receiver's k-th arrival from it, in time order.
fn arrival_matrix(
    participants: &[NodeId],
    emissions: &[NodeId],
    reports: &BTreeMap<NodeId, ChirpArrivalReport>,
) -> Vec<Vec<Option<f64>>> {
    participants
        .iter()
        .map(|receiver| {
            let report = &reports[receiver];
            let mut per_emitter: BTreeMap<NodeId, Vec<f64>> = BTreeMap::new();
            for arrival in &report.arrivals {
                per_emitter
                    .entry(arrival.emitter_node_id)
                    .or_default()
                    .push(local_seconds(arrival.arrival_device_sample_index, arrival.arrival_fractional_sample));
            }
            for times in per_emitter.values_mut() {
                times.sort_by(f64::total_cmp);
            }
            let mut used: BTreeMap<NodeId, usize> = BTreeMap::new();
            emissions
                .iter()
                .map(|emitter| {
                    let occurrence = used.entry(*emitter).or_insert(0);
                    let time = per_emitter.get(emitter).and_then(|times| times.get(*occurrence)).copied();
                    *occurrence += 1;
                    time
                })
                .collect()
        })
        .collect()
}

pub fn solve_island_round(
    input: &IslandRoundInput<'_>,
    config: &IslandSyncConfig,
) -> Option<IslandRoundSolution> {
    let participants: Vec<NodeId> = input
        .members
        .iter()
        .copied()
        .filter(|node| input.reports.contains_key(node))
        .collect();
    let emissions: Vec<NodeId> = input
        .emission_order
        .iter()
        .copied()
        .filter(|node| participants.contains(node) && !input.muted_emitters.contains(node))
        .collect();
    let distinct_emitters: BTreeSet<NodeId> = emissions.iter().copied().collect();
    if distinct_emitters.len() < 2 {
        return None;
    }
    let index_of = |node: NodeId| participants.iter().position(|p| *p == node);
    let observation = ChirpRoundObservation {
        emission_emitters: emissions.iter().filter_map(|node| index_of(*node)).collect(),
        arrival_local_s: arrival_matrix(&participants, &emissions, input.reports),
        self_path_lengths_m: participants
            .iter()
            .map(|node| {
                input
                    .self_path_lengths_m
                    .get(node)
                    .copied()
                    .unwrap_or(config.default_self_path_length_m)
            })
            .collect(),
    };
    let compensation = if observation.closing_emission_index().is_some() {
        RoundRateCompensation::ClosingChirp
    } else {
        RoundRateCompensation::None
    };
    let (measurements, _failures) = observation.pairwise_measurements(config.sound_speed_mps, &compensation);
    if measurements.is_empty() {
        return None;
    }
    // Walking phones (plan: "down-weighted or skipped") stay in the clock
    // graph as heavily down-weighted vertices, but neither their offsets nor
    // their positions are taken from a round they walked through: at
    // 1–3 m/s a round moves them metres, i.e. milliseconds of offset error.
    let walking: BTreeSet<usize> = (0..participants.len())
        .filter(|&index| matches!(input.reports[&participants[index]].motion_state, MotionState::Walking))
        .collect();
    let is_still = |node: &NodeId| index_of(*node).is_some_and(|index| !walking.contains(&index));
    let reference_node = input
        .preferred_reference
        .filter(|node| distinct_emitters.contains(node) && is_still(node))
        .or_else(|| distinct_emitters.iter().copied().find(is_still))?;
    let reference_index = index_of(reference_node)?;
    let reference_emission = observation.first_emission_of(reference_index)?;
    let reference_time_s = observation.arrival_local_s[reference_index][reference_emission]?;
    let rates: Option<Vec<f64>> = observation
        .closing_chirp_rate_ratios()
        .and_then(|ratios| ratios.into_iter().collect());
    let motion_energy: Vec<f64> = participants
        .iter()
        .map(|node| match input.reports[node].motion_state {
            MotionState::Walking | MotionState::Handling => WALKING_MOTION_ENERGY,
            _ => 0.0,
        })
        .collect();
    let first_pass = clock_offset_constraints(&measurements, reference_time_s, rates.as_deref(), &[], &motion_energy);
    let coarse = solve_clock_graph(participants.len(), reference_index, &first_pass, &config.clock_graph).ok()?;
    let coarse_offsets: Vec<f64> = coarse.offsets_s.iter().map(|offset| offset.unwrap_or(0.0)).collect();
    let constraints = clock_offset_constraints(
        &measurements,
        reference_time_s,
        rates.as_deref(),
        &coarse_offsets,
        &motion_energy,
    );
    let solution = solve_clock_graph(participants.len(), reference_index, &constraints, &config.clock_graph).ok()?;
    let clock_offsets = participants
        .iter()
        .enumerate()
        .filter(|(index, _)| !walking.contains(index))
        .filter_map(|(index, node)| {
            let offset = solution.offsets_s[index]?;
            let sigma = solution.offset_sigmas_s[index].unwrap_or(config.clock_graph.pick_offset_sigma_s);
            Some((*node, (offset, sigma.max(config.clock_graph.pick_offset_sigma_s))))
        })
        .collect();
    let ranges: Vec<RangeMeasurement> = measurements
        .iter()
        .map(|m| RangeMeasurement {
            node_i: m.first_participant,
            node_j: m.second_participant,
            distance_m: m.result.distance_m,
            sigma_m: config.range_sigma_m,
        })
        .collect();
    let ranged_pairs = measurements
        .iter()
        .map(|m| (participants[m.first_participant], participants[m.second_participant], m.result.distance_m))
        .collect();
    let mut layout_positions_m = BTreeMap::new();
    let mut layout_sigmas_m = BTreeMap::new();
    // Walking phones break the rigid-layout assumption (plan: skipped in a
    // round's geometry); their previous positions are carried forward.
    let ranged_nodes: BTreeSet<usize> = measurements
        .iter()
        .flat_map(|m| [m.first_participant, m.second_participant])
        .filter(|node| !walking.contains(node))
        .collect();
    // A node ranged to fewer than three others has a flip ambiguity (two
    // circle intersections) that refinement cannot resolve, so it keeps its
    // previous layout position this round (carried forward by the engine).
    let mut placeable = ranged_nodes.clone();
    loop {
        let degree = |node: usize| {
            measurements
                .iter()
                .filter(|m| {
                    (m.first_participant == node && placeable.contains(&m.second_participant))
                        || (m.second_participant == node && placeable.contains(&m.first_participant))
                })
                .count()
        };
        if placeable.len() <= MINIMUM_RANGES_FOR_UNIQUE_PLACEMENT {
            break;
        }
        let Some(weakest) = placeable.iter().copied().min_by_key(|node| degree(*node)) else { break };
        if degree(weakest) >= MINIMUM_RANGES_FOR_UNIQUE_PLACEMENT {
            break;
        }
        placeable.remove(&weakest);
    }
    if placeable.len() >= 2 {
        let heights = vec![config.assumed_phone_height_m; participants.len()];
        // Unranged participants would leave the MDS graph disconnected, so
        // the layout is solved over the placeable subset only.
        let subset: Vec<usize> = placeable.iter().copied().collect();
        let subset_index = |index: usize| subset.iter().position(|s| *s == index);
        let subset_ranges: Vec<RangeMeasurement> = ranges
            .iter()
            .filter_map(|r| {
                Some(RangeMeasurement {
                    node_i: subset_index(r.node_i)?,
                    node_j: subset_index(r.node_j)?,
                    ..r.clone()
                })
            })
            .collect();
        let solved = solve_layout(subset.len(), &heights[..subset.len()], &subset_ranges, &config.layout_refine);
        if let Some(layout) = solved.ok().filter(|layout| layout.is_rigid) {
            for (position_in_subset, &participant) in subset.iter().enumerate() {
                let node = participants[participant];
                layout_positions_m.insert(node, layout.positions_m[position_in_subset]);
                layout_sigmas_m.insert(node, layout.horizontal_sigmas_m[position_in_subset]);
            }
        }
    }
    Some(IslandRoundSolution {
        reference_node,
        reference_time_s,
        clock_offsets,
        ranged_pairs,
        layout_positions_m,
        layout_sigmas_m,
        walking_nodes: walking.iter().map(|&index| participants[index]).collect(),
    })
}
