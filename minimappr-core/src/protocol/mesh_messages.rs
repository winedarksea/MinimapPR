//! Mesh coordination message bodies (W3-mesh-sim): the replicated session
//! event log and its `since=N` cursor, host announcements and handover,
//! node status changes, chirp direction cues and the classification rota.
//!
//! Event-log entries carry *encoded `FieldMessage` frames*, so a mirror on
//! an older build keeps entries of kinds it does not understand (they decode
//! to `DecodedFrame::UnknownKind`) instead of failing the whole append.

use serde::{Deserialize, Serialize};

use super::session_messages::NodeDescriptor;
use super::wire_codes::{
    NodeId, NodeStatusChangeCode, SessionChangeReasonCode, SessionTransportCode,
};

/// One entry of the host's append-only session event log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventLogEntry {
    /// Monotonic, gap-free, starting at 1.
    pub sequence: u64,
    /// Host term that appended the entry.
    pub host_term: u32,
    /// `FieldMessage::encode_frame` bytes of the logged record.
    pub field_message_frame: Vec<u8>,
}

/// Host -> session: newly appended entries (or a catch-up reply).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventLogAppend {
    pub host_node_id: NodeId,
    pub host_term: u32,
    /// Sequence of the last entry the host holds (may exceed this batch).
    pub host_head_sequence: u64,
    pub entries: Vec<EventLogEntry>,
}

/// Node -> host: send me every entry with `sequence > since_sequence`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventLogCatchUpRequest {
    pub node_id: NodeId,
    pub since_sequence: u64,
}

/// Host -> session, periodic: liveness, term and log head. Also logged once
/// per term so every mirror records who hosted when.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostAnnouncement {
    pub host_node_id: NodeId,
    pub host_term: u32,
    pub event_log_head_sequence: u64,
    pub transport: SessionTransportCode,
    pub reason: SessionChangeReasonCode,
}

/// Host -> successor: planned handover (thermal, leaving). The successor
/// takes over from its mirror at `host_term`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostHandover {
    pub from_host_node_id: NodeId,
    pub to_node_id: NodeId,
    pub host_term: u32,
    pub reason: SessionChangeReasonCode,
    pub event_log_head_sequence: u64,
}

/// Logged at join: membership survives handover through the mirror.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionMemberRecord {
    pub node_id: NodeId,
    /// Handover succession order (lower = earlier).
    pub join_order: u32,
    pub node_descriptor: NodeDescriptor,
    /// d_ii of the ranging mic, metres (per-model table).
    pub self_path_length_m: f32,
}

/// Node -> host (and logged): audio route, gap, mute or thermal change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeStatusChange {
    pub node_id: NodeId,
    pub local_sequence: u64,
    pub change: NodeStatusChangeCode,
    pub device_sample_index: u64,
}

/// Array node or stereo phone -> host: world bearing of another node's
/// chirp. Locks the layout's chirality (plan §Mirror images).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChirpDirectionReport {
    pub round_id: u32,
    pub reporting_node_id: NodeId,
    pub emitter_node_id: NodeId,
    /// Counter-clockwise from east, degrees.
    pub world_azimuth_deg: f32,
    pub azimuth_standard_deviation_deg: f32,
}

/// Phones within a few metres of each other share classification duty.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassificationClump {
    /// Duty order: slot k of the rota belongs to `member_node_ids[k % n]`.
    pub member_node_ids: Vec<NodeId>,
}

/// Host -> island: round-robin classification inside chirp-ranged clumps.
/// A node classifies a trigger only when `floor(reference_s / slot_s) % n`
/// is its rank; nodes not in any clump always classify.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassificationRota {
    pub island_id: u32,
    pub rota_epoch: u32,
    pub slot_duration_milliseconds: u32,
    pub clumps: Vec<ClassificationClump>,
}

/// Host -> session (and logged): the session changed transport, so no
/// detection is lost silently.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransportNotice {
    pub host_node_id: NodeId,
    pub transport: SessionTransportCode,
    pub reason: SessionChangeReasonCode,
    /// Clip fetching is off on this transport (constrained or detection-only).
    pub clip_fetch_enabled: bool,
}
