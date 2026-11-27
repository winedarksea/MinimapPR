//! A capture stall inside a chirp round (closeout F2-H1): the stalled
//! phone's closing-chirp rate is implausible, so its biased intervals stay
//! out of that round's clocks and layout instead of dragging the others.

use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::layout::gps_anchor_fit::weighted_kabsch_2d;
use crate::protocol::session_messages::ChirpArrival;

const PHONE_POSITIONS_M: [[f64; 2]; 5] = [[10.0, 0.0], [3.1, 9.5], [-8.1, 5.9], [-8.1, -5.9], [3.1, -9.5]];
const CLOCK_OFFSETS_S: [f64; 5] = [0.0, 0.012, -0.004, 0.021, -0.017];
const CLOCK_RATES_PPM: [f64; 5] = [0.0, 35.0, -20.0, 50.0, -45.0];
const SLOT_S: f64 = 0.22;
const ROUND_START_S: f64 = 10.0;
const STALLED_PHONE: usize = 3;
/// Lost between slots 2 and 3, inside the round (simulated stalls: 50–300 ms).
const STALL_AFTER_SLOT: usize = 2;
const STALL_S: f64 = 0.06;
/// Healthy phones must keep their layout to this (range σ is a few cm).
const HEALTHY_LAYOUT_TOLERANCE_M: f64 = 0.1;

/// Arrival reports for one round, emission order 0..5 then the opener again.
fn round_reports(stall: Option<(usize, f64)>, config: &IslandSyncConfig) -> (Vec<NodeId>, BTreeMap<NodeId, ChirpArrivalReport>) {
    let emission_order: Vec<NodeId> = (0..PHONE_POSITIONS_M.len()).chain([0]).map(|i| NodeId(i as u32)).collect();
    let mut reports = BTreeMap::new();
    for receiver in 0..PHONE_POSITIONS_M.len() {
        let arrivals = emission_order
            .iter()
            .enumerate()
            .map(|(slot, emitter)| {
                let emitter_index = emitter.0 as usize;
                let path_m = if emitter_index == receiver {
                    config.default_self_path_length_m
                } else {
                    let (a, b) = (PHONE_POSITIONS_M[receiver], PHONE_POSITIONS_M[emitter_index]);
                    (a[0] - b[0]).hypot(a[1] - b[1])
                };
                let true_time_s = ROUND_START_S + slot as f64 * SLOT_S + path_m / config.sound_speed_mps;
                let mut local_s = CLOCK_OFFSETS_S[receiver] + (1.0 + CLOCK_RATES_PPM[receiver] * 1e-6) * true_time_s;
                if let Some((phone, lost_s)) = stall {
                    if phone == receiver && slot > STALL_AFTER_SLOT {
                        local_s -= lost_s;
                    }
                }
                let samples = local_s * DEVICE_SAMPLE_RATE_HZ;
                ChirpArrival {
                    emitter_node_id: *emitter,
                    arrival_device_sample_index: samples.floor() as u64,
                    arrival_fractional_sample: (samples - samples.floor()) as f32,
                    peak_to_noise_db: 30.0,
                }
            })
            .collect::<Vec<_>>();
        let own = arrivals.iter().find(|a| a.emitter_node_id.0 as usize == receiver).map(|a| a.arrival_device_sample_index);
        reports.insert(NodeId(receiver as u32), ChirpArrivalReport {
            round_id: 1,
            reporting_node_id: NodeId(receiver as u32),
            own_emission_device_sample_index: own,
            arrivals,
            motion_state: MotionState::Still,
            own_chirp_inaudible: false,
            reporter_attitude: None,
        });
    }
    (emission_order, reports)
}

fn solve(stall: Option<(usize, f64)>, config: &IslandSyncConfig) -> IslandRoundSolution {
    let (emission_order, reports) = round_reports(stall, config);
    let members: Vec<NodeId> = (0..PHONE_POSITIONS_M.len()).map(|i| NodeId(i as u32)).collect();
    solve_island_round(
        &IslandRoundInput {
            members,
            reports: &reports,
            emission_order: &emission_order,
            self_path_lengths_m: &BTreeMap::new(),
            muted_emitters: &BTreeSet::new(),
            preferred_reference: Some(NodeId(0)),
            speaker_geometry: &RoundSpeakerGeometry::default(),
        },
        config,
    )
    .expect("round solves")
}

/// Largest layout residual of the healthy phones after a rigid fit to truth.
fn healthy_layout_error_m(solution: &IslandRoundSolution) -> f64 {
    let healthy: Vec<usize> = (0..PHONE_POSITIONS_M.len()).filter(|i| *i != STALLED_PHONE).collect();
    let solved: Vec<[f64; 2]> = healthy
        .iter()
        .map(|i| {
            let p = solution.layout_positions_m[&NodeId(*i as u32)];
            [p[0], p[1]]
        })
        .collect();
    let truth: Vec<[f64; 2]> = healthy.iter().map(|i| PHONE_POSITIONS_M[*i]).collect();
    // MDS picks its own handedness: score the better of the two rigid fits.
    [false, true]
        .iter()
        .map(|&mirrored| {
            let transform = weighted_kabsch_2d(&solved, &truth, &vec![1.0; solved.len()], mirrored);
            solved
                .iter()
                .zip(&truth)
                .map(|(s, t)| {
                    let a = transform.apply(*s);
                    (a[0] - t[0]).hypot(a[1] - t[1])
                })
                .fold(0.0, f64::max)
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn a_clean_round_places_and_clocks_every_phone() {
    let config = IslandSyncConfig::default();
    let solution = solve(None, &config);
    assert_eq!(solution.layout_positions_m.len(), PHONE_POSITIONS_M.len());
    assert_eq!(solution.clock_offsets.len(), PHONE_POSITIONS_M.len());
    assert!(healthy_layout_error_m(&solution) < HEALTHY_LAYOUT_TOLERANCE_M);
}

#[test]
fn a_phone_that_stalls_inside_the_round_is_left_out_of_it() {
    let config = IslandSyncConfig::default();
    let solution = solve(Some((STALLED_PHONE, STALL_S)), &config);
    let stalled = NodeId(STALLED_PHONE as u32);
    assert!(!solution.clock_offsets.contains_key(&stalled), "stalled phone's offset was used");
    assert!(!solution.layout_positions_m.contains_key(&stalled), "stalled phone was placed");
    let error = healthy_layout_error_m(&solution);
    assert!(error < HEALTHY_LAYOUT_TOLERANCE_M, "healthy phones dragged {error:.2} m");
}
