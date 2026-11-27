//! Speaker–mic geometry in the live solve, on synthetic rounds whose
//! chirps leave from offset speakers: corrected where the pair bearing is
//! known (anchored island of four), σ-widened where it is not (two phones
//! a metre apart) or where attitude is missing.

use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::layout::gps_anchor_fit::{fit_layout_to_gps, AnchorFitConfig, ChiralityCue, GpsFix};
use crate::mesh::config::DEVICE_SAMPLE_RATE_HZ;
use crate::mesh::island_round_solver::{solve_island_round, IslandRoundInput, IslandRoundSolution};
use crate::mesh::node_split_brain_tests::{phone, SilentAudio};
use crate::mesh::node_state_machine::{DeviceStatusChange, FieldNodeStateMachine, LocalChirpRound, NodeAction, NodeInput};
use crate::protocol::session_messages::ChirpArrival;
use crate::protocol::wire_codes::{MotionState, SessionId};
use crate::protocol::FieldMessage;

const EARPIECE_BODY_OFFSET_M: [f64; 3] = [0.0, 0.13, 0.0];
const SLOT_S: f64 = 0.25;
const ROUND_START_S: f64 = 20.0;
const PHONE_HEIGHT_M: f64 = 1.4;
/// Plausible Pixel rotation-vector heading accuracy.
const REPORTED_HEADING_ACCURACY_RAD: f32 = 0.1;
/// Island of four, ~20 m across: its GPS anchor pins bearings to a few degrees.
const ISLAND_POSITIONS_M: [[f64; 2]; 4] = [[0.0, 0.0], [24.0, 3.0], [10.0, 20.0], [-8.0, 14.0]];
const ISLAND_CLOCK_OFFSETS_S: [f64; 4] = [0.0, 0.011, -0.006, 0.018];
/// Phones 0 and 2 chirp from an earpiece, lying flat at these headings.
const ISLAND_YAWS_RAD: [Option<f64>; 4] = [Some(0.3), None, Some(2.0), None];
const ISLAND_GPS_SIGMA_M: f64 = 2.0;
/// MDS frames are arbitrary: the local layout is the truth rotated by this.
const LAYOUT_FRAME_ROTATION_RAD: f64 = 0.7;
/// After correction: GPS-anchored bearings are good to a few degrees.
const CORRECTED_RANGE_TOLERANCE_M: f64 = 0.005;
const CORRECTED_OFFSET_TOLERANCE_S: f64 = 20e-6;

/// Lying flat, body +y pointing at `yaw` CCW from north: q = rotation about up.
fn flat_attitude(yaw_rad: f64, heading_is_true_north: bool) -> DeviceAttitude {
    let half = 0.5 * yaw_rad;
    DeviceAttitude {
        body_to_east_north_up_wxyz: [half.cos() as f32, 0.0, 0.0, half.sin() as f32],
        heading_accuracy_rad: Some(REPORTED_HEADING_ACCURACY_RAD),
        heading_is_true_north,
    }
}

struct SyntheticRound {
    members: Vec<NodeId>,
    emission_order: Vec<NodeId>,
    reports: BTreeMap<NodeId, ChirpArrivalReport>,
    self_paths: BTreeMap<NodeId, f64>,
    body_offsets: BTreeMap<NodeId, [f64; 3]>,
}

/// Emission order 0..n then the opener; each chirp leaves its speaker
/// (mic + R(yaw)·v_body) and is heard at each mic.
fn synthetic_round(positions: &[[f64; 2]], clock_offsets_s: &[f64], yaws: &[Option<f64>], with_attitude: bool, config: &IslandSyncConfig) -> SyntheticRound {
    let count = positions.len();
    let members: Vec<NodeId> = (0..count).map(|i| NodeId(i as u32)).collect();
    let emission_order: Vec<NodeId> = (0..count).chain([0]).map(|i| NodeId(i as u32)).collect();
    let attitudes: Vec<Option<DeviceAttitude>> = yaws.iter().map(|yaw| yaw.map(|yaw| flat_attitude(yaw, false))).collect();
    let speaker_world: Vec<[f64; 3]> = attitudes
        .iter()
        .map(|attitude| attitude.map_or([0.0; 3], |a| rotate_body_to_east_north_up(a.body_to_east_north_up_wxyz, EARPIECE_BODY_OFFSET_M).unwrap()))
        .collect();
    let mic = |i: usize| [positions[i][0], positions[i][1], PHONE_HEIGHT_M];
    let mut reports = BTreeMap::new();
    for receiver in 0..count {
        let arrivals: Vec<ChirpArrival> = emission_order
            .iter()
            .enumerate()
            .map(|(slot, emitter)| {
                let e = emitter.0 as usize;
                let speaker = [0, 1, 2].map(|axis| mic(e)[axis] + speaker_world[e][axis]);
                let path_m = norm([0, 1, 2].map(|axis| speaker[axis] - mic(receiver)[axis]));
                let local_s = clock_offsets_s[receiver] + ROUND_START_S + slot as f64 * SLOT_S + path_m / config.sound_speed_mps;
                let samples = local_s * DEVICE_SAMPLE_RATE_HZ;
                ChirpArrival {
                    emitter_node_id: *emitter,
                    arrival_device_sample_index: samples.floor() as u64,
                    arrival_fractional_sample: (samples - samples.floor()) as f32,
                    peak_to_noise_db: 30.0,
                }
            })
            .collect();
        let own = arrivals.iter().find(|a| a.emitter_node_id.0 as usize == receiver).map(|a| a.arrival_device_sample_index);
        reports.insert(NodeId(receiver as u32), ChirpArrivalReport {
            round_id: 1,
            reporting_node_id: NodeId(receiver as u32),
            own_emission_device_sample_index: own,
            arrivals,
            motion_state: MotionState::Still,
            own_chirp_inaudible: false,
            reporter_attitude: if with_attitude { attitudes[receiver] } else { None },
        });
    }
    let self_paths = (0..count).map(|i| (NodeId(i as u32), norm(speaker_world[i]))).collect();
    let body_offsets = (0..count).filter(|i| yaws[*i].is_some()).map(|i| (NodeId(i as u32), EARPIECE_BODY_OFFSET_M)).collect();
    SyntheticRound { members, emission_order, reports, self_paths, body_offsets }
}

/// The previous round's layout, in a rotated MDS frame, GPS-anchored.
fn anchored_layout(positions: &[[f64; 2]], gps_sigma_m: f64) -> IslandLayout {
    let (sin, cos) = LAYOUT_FRAME_ROTATION_RAD.sin_cos();
    let local: Vec<[f64; 2]> = positions.iter().map(|p| [cos * p[0] - sin * p[1], sin * p[0] + cos * p[1]]).collect();
    // Alternating ±half-σ fix errors keep the anchor honest without an RNG.
    let fixes: Vec<Option<GpsFix>> = positions
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let wobble = if i % 2 == 0 { 0.5 * gps_sigma_m } else { -0.5 * gps_sigma_m };
            Some(GpsFix { east_m: p[0] + wobble, north_m: p[1] - wobble, sigma_m: gps_sigma_m })
        })
        .collect();
    let anchor = fit_layout_to_gps(&local, &fixes, ChiralityCue::Locked { mirrored: false }, &AnchorFitConfig::default()).unwrap();
    IslandLayout {
        captured_ms: 0,
        local_positions_m: local.iter().enumerate().map(|(i, p)| (NodeId(i as u32), [p[0], p[1], PHONE_HEIGHT_M])).collect(),
        sigmas_m: (0..positions.len()).map(|i| (NodeId(i as u32), 0.05)).collect(),
        anchor: Some(anchor),
    }
}

fn solve(round: &SyntheticRound, geometry: &RoundSpeakerGeometry, config: &IslandSyncConfig) -> IslandRoundSolution {
    solve_island_round(
        &IslandRoundInput {
            members: round.members.clone(),
            reports: &round.reports,
            emission_order: &round.emission_order,
            self_path_lengths_m: &round.self_paths,
            muted_emitters: &BTreeSet::new(),
            preferred_reference: Some(NodeId(0)),
            speaker_geometry: geometry,
        },
        config,
    )
    .expect("round solves")
}

fn true_distance(positions: &[[f64; 2]], a: NodeId, b: NodeId) -> f64 {
    let (p, q) = (positions[a.0 as usize], positions[b.0 as usize]);
    (p[0] - q[0]).hypot(p[1] - q[1])
}

fn worst_range_error(solution: &IslandRoundSolution, positions: &[[f64; 2]]) -> f64 {
    solution.ranged_pairs.iter().map(|(a, b, d)| (d - true_distance(positions, *a, *b)).abs()).fold(0.0, f64::max)
}

fn worst_offset_error(solution: &IslandRoundSolution, offsets: &[f64]) -> f64 {
    solution
        .clock_offsets
        .iter()
        .map(|(node, (offset, _))| (offset - (offsets[node.0 as usize] - offsets[0])).abs())
        .fold(0.0, f64::max)
}

#[test]
fn rotation_follows_the_android_rotation_vector_convention() {
    let identity = rotate_body_to_east_north_up([1.0, 0.0, 0.0, 0.0], EARPIECE_BODY_OFFSET_M).unwrap();
    assert!((identity[1] - 0.13).abs() < 1e-9, "body y is north at identity: {identity:?}");
    // Flat, turned 90° CCW: the top of the phone points west.
    let west = rotate_body_to_east_north_up(flat_attitude(std::f64::consts::FRAC_PI_2, true).body_to_east_north_up_wxyz, EARPIECE_BODY_OFFSET_M).unwrap();
    assert!((west[0] + 0.13).abs() < 1e-6 && west[1].abs() < 1e-6, "{west:?}");
    assert!(rotate_body_to_east_north_up([0.0; 4], EARPIECE_BODY_OFFSET_M).is_none());
}

#[test]
fn unreported_or_poor_heading_leaves_the_world_vector_unknown() {
    let config = SpeakerGeometryConfig::default();
    let mut attitude = flat_attitude(0.0, false);
    assert!(NodeSpeakerGeometry::from_attitude(EARPIECE_BODY_OFFSET_M, Some(&attitude), &config).world_offset.is_some());
    attitude.heading_accuracy_rad = None;
    assert!(NodeSpeakerGeometry::from_attitude(EARPIECE_BODY_OFFSET_M, Some(&attitude), &config).world_offset.is_none());
    attitude.heading_accuracy_rad = Some(1.0);
    assert!(NodeSpeakerGeometry::from_attitude(EARPIECE_BODY_OFFSET_M, Some(&attitude), &config).world_offset.is_none());
}

/// (a) Anchored island of four: the bias is removed from ranges and offsets.
#[test]
fn known_bearings_remove_the_speaker_bias() {
    let config = IslandSyncConfig::default();
    let round = synthetic_round(&ISLAND_POSITIONS_M, &ISLAND_CLOCK_OFFSETS_S, &ISLAND_YAWS_RAD, true, &config);
    let raw = solve(&round, &RoundSpeakerGeometry::default(), &config);
    let raw_range_error = worst_range_error(&raw, &ISLAND_POSITIONS_M);
    assert!(raw_range_error > 0.03, "the synthetic bias must be visible uncorrected: {raw_range_error}");
    let layout = anchored_layout(&ISLAND_POSITIONS_M, ISLAND_GPS_SIGMA_M);
    let geometry = RoundSpeakerGeometry::from_body_offsets(&round.reports, &round.body_offsets, Some(&layout), &config);
    let corrected = solve(&round, &geometry, &config);
    for (a, b, outcome) in &corrected.speaker_geometry_outcomes {
        let expected = if round.body_offsets.contains_key(a) || round.body_offsets.contains_key(b) {
            SpeakerGeometryTreatment::Corrected
        } else {
            SpeakerGeometryTreatment::NoSpeakerOffset
        };
        assert_eq!(outcome.treatment, expected, "pair {a:?}-{b:?}");
        assert!(outcome.range_sigma_m < config.range_sigma_m.hypot(0.5 * 0.13));
    }
    let range_error = worst_range_error(&corrected, &ISLAND_POSITIONS_M);
    assert!(range_error < CORRECTED_RANGE_TOLERANCE_M, "corrected {range_error:.4} m (raw {raw_range_error:.4} m)");
    let offset_error = worst_offset_error(&corrected, &ISLAND_CLOCK_OFFSETS_S);
    let raw_offset_error = worst_offset_error(&raw, &ISLAND_CLOCK_OFFSETS_S);
    assert!(offset_error < CORRECTED_OFFSET_TOLERANCE_S, "corrected {offset_error:e} s (raw {raw_offset_error:e} s)");
    assert!(raw_offset_error > 3.0 * offset_error);
}

/// (b) Two phones bottoms-facing 1 m apart: GPS cannot give their bearing,
/// so the range stays as measured and both σ widen by the bound.
#[test]
fn two_close_phones_are_widened_not_corrected() {
    let config = IslandSyncConfig::default();
    let positions = [[0.0, 0.0], [1.0, 0.0]];
    // The earpiece phone's top points west, away from its partner.
    let round = synthetic_round(&positions, &[0.0, 0.004], &[Some(std::f64::consts::FRAC_PI_2), None], true, &config);
    let raw = solve(&round, &RoundSpeakerGeometry::default(), &config);
    let gps_layout = anchored_layout(&positions, 4.0);
    for layout in [None, Some(&gps_layout)] {
        let geometry = RoundSpeakerGeometry::from_body_offsets(&round.reports, &round.body_offsets, layout, &config);
        let solution = solve(&round, &geometry, &config);
        assert_eq!(solution.ranged_pairs, raw.ranged_pairs, "range must not be guessed");
        assert!((solution.ranged_pairs[0].2 - 1.065).abs() < 1e-3, "reads long by v/2: {:?}", solution.ranged_pairs);
        let (_, _, outcome) = solution.speaker_geometry_outcomes[0];
        assert_eq!(outcome.treatment, SpeakerGeometryTreatment::WidenedBearingUnknown);
        assert!((outcome.range_sigma_m - config.range_sigma_m.hypot(0.065)).abs() < 1e-12);
        let offset_bound_s = 0.13 / (2.0 * config.sound_speed_mps);
        assert!((outcome.additional_offset_sigma_s - offset_bound_s).abs() < 1e-12);
        assert!(solution.clock_offsets[&NodeId(1)].1 >= offset_bound_s);
    }
}

/// (c) No attitude: values as before, σ widened only where a speaker vector exists.
#[test]
fn missing_attitude_keeps_values_and_widens() {
    let config = IslandSyncConfig::default();
    let round = synthetic_round(&ISLAND_POSITIONS_M, &ISLAND_CLOCK_OFFSETS_S, &ISLAND_YAWS_RAD, false, &config);
    let raw = solve(&round, &RoundSpeakerGeometry::default(), &config);
    let layout = anchored_layout(&ISLAND_POSITIONS_M, ISLAND_GPS_SIGMA_M);
    let geometry = RoundSpeakerGeometry::from_body_offsets(&round.reports, &round.body_offsets, Some(&layout), &config);
    let solution = solve(&round, &geometry, &config);
    assert_eq!(solution.ranged_pairs, raw.ranged_pairs);
    for (a, b, outcome) in &solution.speaker_geometry_outcomes {
        if round.body_offsets.contains_key(a) || round.body_offsets.contains_key(b) {
            assert_eq!(outcome.treatment, SpeakerGeometryTreatment::WidenedAttitudeUnusable);
        } else {
            assert_eq!(outcome.treatment, SpeakerGeometryTreatment::NoSpeakerOffset);
            assert_eq!(outcome.range_sigma_m, config.range_sigma_m);
            assert_eq!(outcome.additional_offset_sigma_s, 0.0);
        }
    }
}

#[test]
fn node_reports_only_a_fresh_attitude() {
    let config = crate::mesh::config::MeshConfig::default();
    let maximum_age_ms = config.island_sync.speaker_geometry.attitude_maximum_age_ms;
    let mut node = FieldNodeStateMachine::new(NodeId(2), phone(), SessionId([0; 16]), NodeId(1), config, 0);
    let attitude = flat_attitude(0.4, false);
    node.handle(1_000, 0, NodeInput::DeviceStatus(DeviceStatusChange::Attitude(attitude)), &SilentAudio);
    let round = LocalChirpRound { round_id: 1, own_emission_device_sample_index: None, arrivals: Vec::new(), own_chirp_inaudible: false };
    let reported = |actions: Vec<NodeAction>| {
        actions.into_iter().find_map(|action| match action {
            NodeAction::Send(message) => match message.message {
                FieldMessage::ChirpArrivalReport(report) => Some(report.reporter_attitude),
                _ => None,
            },
            _ => None,
        })
    };
    let fresh = node.handle(1_000 + maximum_age_ms, 0, NodeInput::ChirpRoundHeard(round.clone()), &SilentAudio);
    assert_eq!(reported(fresh), Some(Some(attitude)));
    let stale = node.handle(1_001 + maximum_age_ms, 0, NodeInput::ChirpRoundHeard(round), &SilentAudio);
    assert_eq!(reported(stale), Some(None));
}
