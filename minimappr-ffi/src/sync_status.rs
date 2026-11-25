//! Live sync status for the session screen and `dumpsys` JSON: role, join
//! and session-clock state, round counters, this phone's last chirp round
//! and, on the host, per-peer range and clock offset from the round
//! diagnostics. Raw (uncorrected for speaker/mic offsets) until the
//! correction is wired into the round solver.

use std::collections::BTreeMap;

use minimappr_core::mesh::round_diagnostics::RoundDiagnostics;
use minimappr_core::protocol::NodeId;
use minimappr_core::sync::clock_filter::ClockUpdateOutcome;

const MICROSECONDS_PER_SECOND: f64 = 1e6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum FieldDeviceRole {
    Host,
    Joiner,
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct FieldSlotArrivalStatus {
    pub emitter_node_id: u32,
    pub is_self: bool,
    pub detected: bool,
    pub snr_db: Option<f64>,
    pub doppler_skew_samples: Option<f64>,
    pub peak_fraction: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct FieldLocalRoundStatus {
    pub round_id: u32,
    pub own_emission_played: bool,
    pub own_chirp_inaudible: bool,
    pub unsearchable_slot_count: u32,
    pub slots: Vec<FieldSlotArrivalStatus>,
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct FieldPeerSyncStatus {
    pub node_id: u32,
    pub range_m: Option<f64>,
    /// θ = peer − reference from the last round, µs.
    pub round_clock_offset_us: Option<f64>,
    pub round_clock_offset_sigma_us: Option<f64>,
    pub filtered_clock_offset_us: Option<f64>,
    pub filtered_clock_offset_sigma_us: Option<f64>,
    pub rate_offset_ppm: Option<f64>,
    /// Normalised innovation of the last filter update (None: reference,
    /// initialised or re-seeded).
    pub last_innovation_sigma: Option<f64>,
    pub last_filter_outcome: String,
    pub last_round_id: u32,
    pub is_reference: bool,
    pub reference_node_id: u32,
    /// The round's time on the reference node's audio clock (sample / 48 kHz).
    pub round_reference_time_s: f64,
}

#[derive(Debug, Clone, PartialEq, uniffi::Record)]
pub struct FieldSyncStatus {
    pub role: FieldDeviceRole,
    pub node_id: u32,
    pub host_node_id: Option<u32>,
    pub session_id_hex: String,
    pub joined: bool,
    pub session_clock_aligned: bool,
    pub session_clock_offset_ms: Option<f64>,
    pub session_clock_best_round_trip_ms: Option<f64>,
    pub clock_probe_exchanges_accepted: u64,
    pub own_clock_in_sync: bool,
    /// Own model from the host's logged `ClockModel` (joiner view).
    pub own_clock_offset_us: Option<f64>,
    pub own_clock_rate_ppm: Option<f64>,
    pub capture_anchor_count: u32,
    pub capture_rate_offset_ppm: Option<f64>,
    pub capture_anchor_residual_us: Option<f64>,
    pub capture_gap_count: u64,
    pub rounds_started: u32,
    pub rounds_completed: u32,
    pub rounds_heard: u64,
    pub emissions_requested: u64,
    pub emissions_played: u64,
    pub emissions_skipped: u64,
    pub muted_node_ids: Vec<u32>,
    pub last_local_round: Option<FieldLocalRoundStatus>,
    pub peers: Vec<FieldPeerSyncStatus>,
    /// Always true in this pass: the LAN transport is unsealed.
    pub transport_unsealed: bool,
}

fn outcome_name_and_innovation(outcome: Option<ClockUpdateOutcome>) -> (String, Option<f64>) {
    match outcome {
        None => ("reference".into(), None),
        Some(ClockUpdateOutcome::Initialised) => ("initialised".into(), None),
        Some(ClockUpdateOutcome::Accepted { normalized_innovation }) => ("accepted".into(), Some(normalized_innovation)),
        Some(ClockUpdateOutcome::RejectedOutlier { normalized_innovation }) => ("rejected_outlier".into(), Some(normalized_innovation)),
        Some(ClockUpdateOutcome::SilentGapDetected { jump_s }) => (format!("silent_gap_{:.1}ms", jump_s * 1e3), None),
    }
}

/// Latest per-peer view, kept across rounds so a peer missing from one
/// round still shows its last values.
#[derive(Debug, Default)]
pub struct PeerStatusBoard {
    peers: BTreeMap<u32, FieldPeerSyncStatus>,
}

impl PeerStatusBoard {
    pub fn absorb(&mut self, diagnostics: &[RoundDiagnostics], own_node: NodeId) {
        for round in diagnostics {
            for peer in &round.peers {
                let (last_filter_outcome, last_innovation_sigma) = outcome_name_and_innovation(peer.filter_outcome);
                let range_m = round
                    .ranged_pairs
                    .iter()
                    .find(|pair| {
                        (pair.first_node == peer.node_id && pair.second_node == own_node)
                            || (pair.second_node == peer.node_id && pair.first_node == own_node)
                    })
                    .map(|pair| pair.distance_m);
                let entry = FieldPeerSyncStatus {
                    node_id: peer.node_id.0,
                    range_m: range_m.or_else(|| self.peers.get(&peer.node_id.0).and_then(|previous| previous.range_m)),
                    round_clock_offset_us: Some(peer.round_offset_s * MICROSECONDS_PER_SECOND),
                    round_clock_offset_sigma_us: Some(peer.round_offset_sigma_s * MICROSECONDS_PER_SECOND),
                    filtered_clock_offset_us: peer.filtered_offset_s.map(|v| v * MICROSECONDS_PER_SECOND),
                    filtered_clock_offset_sigma_us: peer.filtered_offset_sigma_s.map(|v| v * MICROSECONDS_PER_SECOND),
                    rate_offset_ppm: peer.rate_offset_ppm,
                    last_innovation_sigma,
                    last_filter_outcome,
                    last_round_id: round.round_id,
                    is_reference: peer.node_id == round.reference_node,
                    reference_node_id: round.reference_node.0,
                    round_reference_time_s: round.reference_time_s,
                };
                self.peers.insert(peer.node_id.0, entry);
            }
        }
    }

    pub fn snapshot(&self) -> Vec<FieldPeerSyncStatus> {
        self.peers.values().cloned().collect()
    }
}
