//! Session, sync and status message bodies: join/auth, key exchange, chirp
//! scheduling and arrivals, islands, layout/clock models, heartbeats,
//! people-nearby summaries and the reserved group status.
//!
//! Sample indices are always on the *named node's own* 48 kHz device clock;
//! the host's clock model maps between them. Never wall-clock time.

use serde::{Deserialize, Serialize};

use super::wire_codes::{
    GroupStatusCode, KeyExchangeFlight, MotionState, NodeId, NodeKind, PhonePlacement, SessionId,
    SessionRole,
};

/// Static description of a node, recorded by the host at join.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeDescriptor {
    pub node_kind: NodeKind,
    pub platform_name: String,
    pub app_version: String,
    pub capture_sample_rate_hz: u32,
    pub microphone_count: u8,
    /// Array geometry (body frame, metres) for array nodes; empty for phones.
    pub microphone_positions_body_frame_m: Vec<[f32; 3]>,
    pub can_classify: bool,
    pub can_host: bool,
    pub can_emit_chirps: bool,
    pub has_pps_clock: bool,
}

/// Joiner -> host: authenticate with the QR join token.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JoinRequest {
    pub session_id: SessionId,
    pub join_token: [u8; 16],
    pub requested_role: SessionRole,
    pub node_descriptor: NodeDescriptor,
}

/// Host -> joiner: node id, chirp slot and where the event log stands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JoinAccepted {
    pub session_id: SessionId,
    pub assigned_node_id: NodeId,
    pub host_node_id: NodeId,
    pub chirp_slot_index: u16,
    pub event_log_head_sequence: u64,
}

/// One flight of the short-join-code SPAKE2 exchange (see `session_crypto`).
/// Carried unsealed: it is what establishes the key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JoinCodeKeyExchange {
    pub session_id: SessionId,
    /// Random per-attempt id chosen by the joiner (no node id yet).
    pub joiner_attempt_id: u32,
    pub flight: KeyExchangeFlight,
    pub payload: Vec<u8>,
}

/// One node's emission slot in a chirp round.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChirpSlotAssignment {
    pub emitter_node_id: NodeId,
    /// Offset from the round start, in host-clock samples.
    pub slot_offset_samples: u32,
    pub chirp_low_hz: f32,
    pub chirp_high_hz: f32,
    /// Which pseudo-random chirp code to emit (separates overlapping slots).
    pub chirp_code_id: u16,
}

/// Host -> island: when and how to chirp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChirpSchedule {
    pub island_id: u32,
    pub round_id: u32,
    pub round_start_host_sample_index: u64,
    pub slots: Vec<ChirpSlotAssignment>,
}

/// One received chirp.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChirpArrival {
    pub emitter_node_id: NodeId,
    /// Integer arrival index on the reporting node's clock.
    pub arrival_device_sample_index: u64,
    /// Sub-sample refinement in [-0.5, 0.5).
    pub arrival_fractional_sample: f32,
    pub peak_to_noise_db: f32,
}

/// Node -> host: what it heard in a round, plus its own emission time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChirpArrivalReport {
    pub round_id: u32,
    pub reporting_node_id: NodeId,
    /// Loopback-measured emission index of its own chirp, if it emitted.
    pub own_emission_device_sample_index: Option<u64>,
    pub arrivals: Vec<ChirpArrival>,
    pub motion_state: MotionState,
    /// Raised when its own slot was silent (muted phone).
    pub own_chirp_inaudible: bool,
}

/// Host -> session: current membership of one acoustic island.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IslandMembership {
    pub island_id: u32,
    pub membership_epoch: u32,
    pub member_node_ids: Vec<NodeId>,
}

/// One node's position in the island's local east-north-up frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodePositionEstimate {
    pub node_id: NodeId,
    pub east_north_up_m: [f32; 3],
    pub position_standard_deviation_m: [f32; 3],
}

/// Host -> island: layout solve result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayoutModel {
    pub island_id: u32,
    pub layout_epoch: u32,
    /// WGS84 anchor of the ENU frame origin (from the GPS fit).
    pub origin_latitude_deg: f64,
    pub origin_longitude_deg: f64,
    pub origin_altitude_m: f32,
    pub node_positions: Vec<NodePositionEstimate>,
}

/// Maps a node's device clock onto the island reference clock:
/// `reference_index = node_index - offset - rate_ppm * 1e-6 * (node_index - anchor)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeClockEstimate {
    pub node_id: NodeId,
    pub anchor_node_sample_index: u64,
    pub offset_samples: f64,
    pub rate_offset_ppm: f64,
    pub offset_standard_deviation_samples: f32,
}

/// Host -> island: clock filter result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClockModel {
    pub island_id: u32,
    pub clock_epoch: u32,
    pub reference_node_id: NodeId,
    pub node_clocks: Vec<NodeClockEstimate>,
}

/// Coarse GNSS fix carried in heartbeats.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GnssFix {
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub altitude_m: f32,
    pub horizontal_accuracy_m: f32,
    pub fix_device_sample_index: u64,
}

/// Node -> host, periodic (batched over cell while still).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Heartbeat {
    pub node_id: NodeId,
    pub local_sequence: u64,
    pub device_sample_index_now: u64,
    /// 0 = throttling imminent, 1 = cool (OS thermal headroom forecast).
    pub thermal_headroom_fraction: f32,
    pub battery_fraction: f32,
    pub placement: PhonePlacement,
    pub motion_state: MotionState,
    pub mic_only_mode: bool,
    pub capture_overrun_count: u32,
    pub gnss_fix: Option<GnssFix>,
}

/// Anonymous Bluetooth summary (option, off by default). Never addresses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PeopleNearbySummary {
    pub node_id: NodeId,
    pub summary_window_seconds: u16,
    pub unknown_advertiser_count: u16,
    pub signal_strength_minimum_dbm: i8,
    pub signal_strength_median_dbm: i8,
    pub signal_strength_maximum_dbm: i8,
}

/// Reserved: "taking a break", "heading home", "request assistance".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupStatus {
    pub node_id: NodeId,
    pub status: GroupStatusCode,
    pub note: Option<String>,
}
