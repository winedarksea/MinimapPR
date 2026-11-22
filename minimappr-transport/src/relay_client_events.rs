//! Outputs, errors and counters of `RelayClientSession`.

use minimappr_core::protocol::session_messages::JoinCodeKeyExchange;
use minimappr_core::protocol::{FieldMessage, NodeId};
use minimappr_relay_wire::{FrameRejectionReason, RelayRole, SessionRelayLimits, SessionUsageCounters};

use crate::clip_chunking::ClipChunkingError;
use crate::outbound_scheduler::OutboundQueueError;

/// Something the application should react to.
#[derive(Debug, Clone, PartialEq)]
pub enum RelayClientEvent {
    /// The relay accepted the socket (first connect or resumed).
    Connected {
        peer_id: u32,
        role: RelayRole,
        host_peer_id: Option<u32>,
        usage: SessionUsageCounters,
        limits: SessionRelayLimits,
        resumed: bool,
    },
    Disconnected {
        reconnect_at_milliseconds: u64,
    },
    /// An authenticated, de-duplicated session message. Reassembled clips
    /// arrive here as their `ClipPayload`/`TapToHearPayload` message.
    MessageReceived {
        sender: NodeId,
        message: FieldMessage,
    },
    /// Authenticated message of a kind this build does not know.
    UnknownMessageKindReceived {
        sender: NodeId,
        message_kind_id: u16,
    },
    /// An unsealed join-code flight (host: from a joiner; joiner: from host).
    KeyExchangeFlightReceived {
        sender_peer_id: u32,
        exchange: JoinCodeKeyExchange,
    },
    PeerJoined {
        peer_id: u32,
        role: RelayRole,
    },
    PeerLeft {
        peer_id: u32,
    },
    /// The relay dropped one of our frames.
    FrameRejectedByRelay {
        reason: FrameRejectionReason,
        recipient_peer_id: Option<u32>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelayClientError {
    /// Sealed traffic needs the session key (QR, or the join-code exchange).
    NoSessionKeyYet,
    EncodeFailed(String),
    SealFailed,
    Queue(OutboundQueueError),
    Chunking(ClipChunkingError),
    /// `send_clip_payload` takes only `ClipPayload` or `TapToHearPayload`.
    NotAClipPayloadMessage,
}

impl std::fmt::Display for RelayClientError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "relay client error: {self:?}")
    }
}

impl std::error::Error for RelayClientError {}

impl From<OutboundQueueError> for RelayClientError {
    fn from(error: OutboundQueueError) -> Self {
        Self::Queue(error)
    }
}

impl From<ClipChunkingError> for RelayClientError {
    fn from(error: ClipChunkingError) -> Self {
        Self::Chunking(error)
    }
}

/// Inbound drop reasons, counted rather than surfaced one by one: they are
/// either an attacker's noise or a relay misbehaving, and the session goes on.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RelayClientInboundStatistics {
    pub frames_received: u64,
    pub malformed_relay_frames: u64,
    pub sealed_frames_before_session_key: u64,
    pub authentication_failures: u64,
    /// Envelope sender differs from the relay header sender.
    pub sender_mismatches: u64,
    /// Envelope addressed to another node (relay misrouting).
    pub misaddressed_envelopes: u64,
    /// Authentic envelope claiming this node as sender (relay reflection).
    pub reflected_own_envelopes: u64,
    pub replays_dropped: u64,
    pub undecodable_field_frames: u64,
    pub clip_chunks_rejected: u64,
    pub unparseable_control_notices: u64,
}
