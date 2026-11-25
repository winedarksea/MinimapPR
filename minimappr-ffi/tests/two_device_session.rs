//! Two `FieldMeshDevice`s, a host and a joiner, exchange LAN envelopes in
//! memory with network delay, unrelated monotonic epochs and ±60 ppm audio
//! crystals. Every chirp they ask to play is rendered into both capture
//! streams. The session must join, run chirp rounds and report the true
//! range and audio-clock offset; a skipped emission must never mute a phone.

mod support;

use minimappr_core::sync::chirp_waveform::ChirpBand;
use minimappr_ffi::mesh_device::{FieldDeviceSetup, FieldMeshDevice};
use minimappr_ffi::mesh_tuning::{default_field_mesh_tuning, FieldMeshTuning};
use minimappr_ffi::stage1_detector::FieldStage1Mode;
use minimappr_ffi::sync_status::{FieldDeviceRole, FieldSyncStatus};
use support::synthetic_acoustic_pair::{PairWorld, PhoneWorld};

const HOST_NODE_ID: u32 = 0x1111_0001;
const JOINER_NODE_ID: u32 = 0x2222_0002;
const SESSION_ID: [u8; 16] = [7; 16];
const DISTANCE_M: f64 = 3.0;
/// The FFI's default d_ii for an unknown model (`IslandSyncConfig`).
const SELF_PATH_M: f64 = 0.02;
const STEP_NS: i64 = 10_000_000;
const SECOND_NS: i64 = 1_000_000_000;

fn tuning() -> FieldMeshTuning {
    FieldMeshTuning { periodic_round_interval_ms: 3_000, ..default_field_mesh_tuning() }
}

fn setup(node_id: u32, role: FieldDeviceRole, session_id: [u8; 16]) -> FieldDeviceSetup {
    FieldDeviceSetup {
        node_id,
        session_id: session_id.to_vec(),
        role,
        platform_name: "synthetic".into(),
        app_version: "test".into(),
        can_classify: false,
        can_emit_chirps: true,
        stage1_mode: FieldStage1Mode::Birds,
        rolling_buffer_seconds: 20,
    }
}

fn phone(node_id: u32, role: FieldDeviceRole, session_id: [u8; 16], offset_ns: i64, ppm: f64, capture_start_s: f64) -> PhoneWorld {
    PhoneWorld {
        device: FieldMeshDevice::new(setup(node_id, role, session_id), tuning(), offset_ns).expect("device"),
        monotonic_offset_ns: offset_ns,
        capture_start_true_s: capture_start_s,
        audio_rate_ppm: ppm,
        next_sample_index: 0,
        skip_emission_ids: Vec::new(),
        skipped_emissions: 0,
    }
}

fn world(joiner_session_id: [u8; 16]) -> PairWorld {
    let tuning = tuning();
    let band = ChirpBand { low_hz: tuning.waveform.band_low_hz, high_hz: tuning.waveform.band_high_hz };
    let phones = vec![
        // Host booted ~3 h before the test's true zero, joiner ~41 min.
        phone(HOST_NODE_ID, FieldDeviceRole::Host, SESSION_ID, 10_800 * SECOND_NS, 60.0, 0.013),
        phone(JOINER_NODE_ID, FieldDeviceRole::Joiner, joiner_session_id, 2_460 * SECOND_NS + 123_456_789, -60.0, 0.37),
    ];
    let mut world = PairWorld::new(phones, DISTANCE_M, SELF_PATH_M, tuning.speed_of_sound_mps(), band, tuning.waveform.duration_s);
    for phone in 0..2 {
        let actions = world.phones[phone].device.set_transport_link_up(world.phones[phone].local_ns(0), true);
        world.handle_actions(phone, actions);
    }
    world
}

fn joiner_peer(status: &FieldSyncStatus) -> minimappr_ffi::sync_status::FieldPeerSyncStatus {
    status.peers.iter().find(|peer| peer.node_id == JOINER_NODE_ID).cloned().expect("joiner in the host's peer table")
}

#[test]
fn host_and_joiner_join_range_and_align_clocks() {
    let mut world = world([0; 16]);
    world.run_until(30 * SECOND_NS, STEP_NS);
    let host = world.phones[0].device.sync_status();
    let joiner = world.phones[1].device.sync_status();
    assert!(joiner.session_clock_aligned, "joiner never locked the session clock: {joiner:?}");
    assert!(joiner.joined, "joiner never joined: {joiner:?}");
    assert_eq!(joiner.session_id_hex, "07".repeat(16), "joiner adopts the host's session id");
    let true_session_offset_ms = (world.phones[0].monotonic_offset_ns - world.phones[1].monotonic_offset_ns) as f64 / 1e6;
    assert!((joiner.session_clock_offset_ms.unwrap() - true_session_offset_ms).abs() < 2.0, "{joiner:?}");
    assert!(host.rounds_completed >= 3, "only {} rounds: {host:?}", host.rounds_completed);
    assert!(host.muted_node_ids.is_empty(), "{host:?}");

    let peer = joiner_peer(&host);
    let range_m = peer.range_m.expect("range to the joiner");
    assert!((range_m - DISTANCE_M).abs() < 0.01, "range {range_m} m vs {DISTANCE_M} m: {host:?}");
    let reference_phone = if peer.reference_node_id == HOST_NODE_ID { 0 } else { 1 };
    let node_phone = 1 - reference_phone;
    let node_id = if node_phone == 1 { JOINER_NODE_ID } else { HOST_NODE_ID };
    let node_peer = host.peers.iter().find(|p| p.node_id == node_id).expect("non-reference node in the peer table");
    let truth_us = world.true_offset_s(node_phone, reference_phone, node_peer.round_reference_time_s) * 1e6;
    let measured_us = node_peer.round_clock_offset_us.expect("offset");
    assert!((measured_us - truth_us).abs() < 10.0, "offset {measured_us} µs vs truth {truth_us} µs");

    let last_round = joiner.last_local_round.expect("joiner heard a round");
    assert!(last_round.slots.iter().all(|slot| slot.detected), "{last_round:?}");
}

#[test]
fn skipped_emission_reports_not_emitted_and_never_mutes() {
    let mut world = world(SESSION_ID);
    // Skip every joiner emission for the first part of the session.
    world.phones[1].skip_emission_ids = (1..=3).collect();
    world.run_until(25 * SECOND_NS, STEP_NS);
    let host = world.phones[0].device.sync_status();
    let joiner = world.phones[1].device.sync_status();
    assert!(world.phones[1].skipped_emissions >= 1, "no emission was skipped");
    assert!(host.muted_node_ids.is_empty(), "skipped emission muted a node: {host:?}");
    assert!(joiner.emissions_skipped >= 1);
    assert!(joiner.emissions_played >= 1, "{joiner:?}");
    assert!(joiner_peer(&host).range_m.is_some(), "rounds after the skips still range");
}

#[test]
fn joiner_with_the_wrong_session_id_is_never_admitted() {
    let mut world = world([9; 16]);
    world.run_until(8 * SECOND_NS, STEP_NS);
    let host = world.phones[0].device.sync_status();
    let joiner = world.phones[1].device.sync_status();
    assert!(!joiner.joined);
    assert!(joiner.host_node_id.is_none(), "joiner followed a host from another session");
    assert!(host.peers.is_empty());
}

/// A silent capture stall on the joiner (10 ms of audio lost, frame counter
/// still contiguous) jumps its clock offset past the silent-gap threshold.
/// In a two-phone island the stalled node cannot be left out of a re-solve
/// (one node is no island), so the round must still re-seed its filter and
/// keep ranging, not freeze the session's diagnostics for good.
#[test]
fn silent_capture_stall_in_a_two_phone_island_recovers() {
    let mut world = world(SESSION_ID);
    world.run_until(14 * SECOND_NS, STEP_NS);
    let before = joiner_peer(&world.phones[0].device.sync_status()).last_round_id;
    world.phones[1].capture_start_true_s += 0.010;
    world.run_until(40 * SECOND_NS, STEP_NS);
    let host = world.phones[0].device.sync_status();
    let peer = joiner_peer(&host);
    assert!(peer.last_round_id >= before + 4, "diagnostics froze at round {} (before stall {before}, completed {})", peer.last_round_id, host.rounds_completed);
    assert!(host.rounds_completed >= peer.last_round_id + 1 && host.rounds_completed <= peer.last_round_id + 2, "{host:?}");
    let range_m = peer.range_m.expect("range");
    assert!((range_m - DISTANCE_M).abs() < 0.01, "range {range_m} after the stall");
}
