//! Sans-IO Field Mode node and host state machines (plan §System
//! architecture, §Networking and session model, §Group splits).
//!
//! Every state machine takes `(now_ms, input)` plus injected capabilities
//! (audio window access on a node, a clip-set localizer on the host) and
//! returns actions: messages to send, chirps to emit, transport switches.
//! No tokio, threads, wall clock or randomness live here, so the Android
//! shell, the WASM PWA, a Linux array host and the Layer 2 mesh simulator
//! (`minimappr-mesh-sim`) all drive identical logic.
//!
//! `now_ms` is the device's coarse monotonic session clock (OS clock): it
//! drives timers and coarse scheduling only. Every TDOA-relevant time is a
//! sample index on a node's own 48 kHz audio clock, mapped through the
//! island clock models in [`clock_mapping`].
//!
//! Host-to-node state (clock models, islands, rota, transport notices,
//! entity updates, detections) flows through the replicated event log
//! ([`event_log`]); a node's mirror doubles as the detection acknowledgement
//! and as the state a successor host restores from ([`host_restoration`]).

pub mod arrival_time_localizer;
pub mod call_pipeline;
pub mod call_pipeline_types;
pub mod chirality_cue;
pub mod chirp_scheduler;
pub mod classification_rota;
pub mod clip_budget;
pub mod clock_mapping;
pub mod config;
pub mod detection_outbox;
pub mod event_log;
pub mod event_matcher;
pub mod fix_contour;
pub mod host_membership;
pub mod host_message_handling;
pub mod host_restoration;
pub mod host_state_machine;
pub mod host_sync_pipeline;
pub mod host_transport_policy;
pub mod island_round_solver;
pub mod island_sync_engine;
pub mod layout_frames;
pub mod local_tangent_plane;
pub mod node_host_link;
pub mod node_message_handling;
pub mod node_state_machine;
pub mod readiness_selection;
pub mod srp_seed_refiner;

use crate::protocol::{FieldMessage, NodeId};

/// Coarse session clock in milliseconds (timers only, never TDOA).
pub type MeshMilliseconds = u64;

/// Who a message is for. The transport resolves `Node` through the relay
/// or the local hotspot; `AllNodes` is a host broadcast.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageRecipient {
    Node(NodeId),
    AllNodes,
}

/// Clips travel below control messages (chunked by the transport).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrafficClass {
    Control,
    Bulk,
}

/// One message a state machine wants sent.
#[derive(Debug, Clone, PartialEq)]
pub struct OutboundMessage {
    pub recipient: MessageRecipient,
    pub message: FieldMessage,
    pub traffic_class: TrafficClass,
}

impl OutboundMessage {
    pub fn control(recipient: MessageRecipient, message: FieldMessage) -> Self {
        Self {
            recipient,
            message,
            traffic_class: TrafficClass::Control,
        }
    }

    pub fn bulk(recipient: MessageRecipient, message: FieldMessage) -> Self {
        Self {
            recipient,
            message,
            traffic_class: TrafficClass::Bulk,
        }
    }
}

/// Idempotency key of a detection: `(node_id, local_sequence)`.
pub type DetectionKey = (NodeId, u64);
