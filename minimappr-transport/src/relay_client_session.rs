//! Sans-IO relay client: one peer's view of a relayed session.
//!
//! The adapter (native tokio-tungstenite, or a browser WebSocket) owns the
//! socket and the clock and calls in:
//! - `connection_parameters()` -> peer id + resume token for the URL
//! - `on_socket_opened`, `on_text_frame`, `on_binary_frame`, `on_socket_closed`
//! - `poll_transmit(now, rng)` until `None`, one frame per socket write
//! - `poll_event()` for application events
//! - `reconnect_due(now)` while disconnected
//!
//! Delivery is at-most-once per connection for control frames: the relay
//! does not buffer for absent peers, and durable state rides the host's event
//! log (plan §Host handover). Clips are resent after a reconnect while inside
//! their deadline.

use std::collections::VecDeque;

use minimappr_core::protocol::session_messages::JoinCodeKeyExchange;
use minimappr_core::protocol::{
    decode_field_message_frame, DecodedFrame, FieldMessage, NodeId, SessionId,
};
use minimappr_core::session_crypto::{EnvelopeRouting, MessageSealer, ReplayWindow, SessionKey};
use minimappr_relay_wire::{
    decode_relay_frame, encode_relay_frame, RelayControlNotice, RelayFrameHeader, RelayFrameKind,
    BROADCAST_PEER_ID,
};
use rand_core::{CryptoRng, RngCore};

use crate::clip_chunking::{split_payload_frame_into_chunks, ClipReassembler, ClipReassemblyProgress};
use crate::outbound_scheduler::{
    OutboundScheduler, OutboundSchedulerStatistics, QueuedControlFrame, ScheduledOutbound,
};
use crate::reconnect_backoff::ReconnectBackoff;
use crate::relay_client_events::{RelayClientError, RelayClientEvent, RelayClientInboundStatistics};
use crate::transport_config::RelayTransportConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelayConnectionPhase {
    Disconnected { reconnect_at_milliseconds: u64 },
    AwaitingWelcome,
    Connected,
}

/// Values the adapter puts in the connect URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelayConnectParameters {
    pub peer_id: u32,
    pub resume_token: Option<String>,
}

pub struct RelayClientSession {
    config: RelayTransportConfig,
    local_peer_id: u32,
    host_peer_id: Option<u32>,
    sealer: Option<MessageSealer>,
    next_sender_sequence: u64,
    replay_window: ReplayWindow,
    scheduler: OutboundScheduler,
    reassembler: ClipReassembler,
    pending_events: VecDeque<RelayClientEvent>,
    phase: RelayConnectionPhase,
    resume_token: Option<String>,
    has_connected_before: bool,
    backoff: ReconnectBackoff,
    inbound_statistics: RelayClientInboundStatistics,
}

impl RelayClientSession {
    /// `initial_sender_sequence` must exceed any sequence this peer id used
    /// earlier in the session (receivers' replay windows remember them); a
    /// shell restarting mid-session should persist it or derive it from time.
    pub fn new(
        config: RelayTransportConfig,
        local_peer_id: u32,
        session_key_and_id: Option<(&SessionKey, &SessionId)>,
        initial_sender_sequence: u64,
    ) -> Self {
        Self {
            scheduler: OutboundScheduler::new(&config),
            reassembler: ClipReassembler::new(&config),
            backoff: ReconnectBackoff::new(&config),
            config,
            local_peer_id,
            host_peer_id: None,
            sealer: session_key_and_id.map(|(key, id)| MessageSealer::new(key, id)),
            next_sender_sequence: initial_sender_sequence,
            replay_window: ReplayWindow::default(),
            pending_events: VecDeque::new(),
            phase: RelayConnectionPhase::Disconnected {
                reconnect_at_milliseconds: 0,
            },
            resume_token: None,
            has_connected_before: false,
            inbound_statistics: RelayClientInboundStatistics::default(),
        }
    }

    /// Installs the key a code joiner obtained through the exchange.
    pub fn install_session_key(&mut self, session_key: &SessionKey, session_id: &SessionId) {
        self.sealer = Some(MessageSealer::new(session_key, session_id));
    }

    pub fn local_peer_id(&self) -> u32 {
        self.local_peer_id
    }

    pub fn host_peer_id(&self) -> Option<u32> {
        self.host_peer_id
    }

    pub fn phase(&self) -> RelayConnectionPhase {
        self.phase
    }

    pub fn inbound_statistics(&self) -> RelayClientInboundStatistics {
        self.inbound_statistics
    }

    pub fn outbound_statistics(&self) -> OutboundSchedulerStatistics {
        self.scheduler.statistics()
    }

    pub fn abandoned_partial_clip_count(&self) -> u64 {
        self.reassembler.abandoned_partial_clip_count()
    }

    pub fn connection_parameters(&self) -> RelayConnectParameters {
        RelayConnectParameters {
            peer_id: self.local_peer_id,
            resume_token: self.resume_token.clone(),
        }
    }

    pub fn reconnect_due(&self, now_milliseconds: u64) -> bool {
        matches!(self.phase, RelayConnectionPhase::Disconnected { reconnect_at_milliseconds }
            if now_milliseconds >= reconnect_at_milliseconds)
    }

    pub fn on_socket_opened(&mut self) {
        self.phase = RelayConnectionPhase::AwaitingWelcome;
    }

    /// Socket closed or failed to open; schedules the next attempt.
    pub fn on_socket_closed(&mut self, now_milliseconds: u64) {
        let reconnect_at_milliseconds = now_milliseconds + self.backoff.next_delay_milliseconds();
        self.phase = RelayConnectionPhase::Disconnected {
            reconnect_at_milliseconds,
        };
        self.pending_events.push_back(RelayClientEvent::Disconnected {
            reconnect_at_milliseconds,
        });
    }

    pub fn poll_event(&mut self) -> Option<RelayClientEvent> {
        self.pending_events.pop_front()
    }

    /// Queues a sealed message at control priority.
    pub fn send_message(
        &mut self,
        recipient: NodeId,
        message: &FieldMessage,
    ) -> Result<(), RelayClientError> {
        if self.sealer.is_none() {
            return Err(RelayClientError::NoSessionKeyYet);
        }
        let plaintext_field_frame = message
            .encode_frame()
            .map_err(|error| RelayClientError::EncodeFailed(error.0))?;
        self.scheduler.enqueue_control(QueuedControlFrame {
            recipient_peer_id: recipient.0,
            relay_frame_kind: RelayFrameKind::SealedEnvelope,
            plaintext_field_frame,
        })?;
        Ok(())
    }

    /// Queues an unsealed join-code flight at control priority.
    pub fn send_key_exchange_flight(
        &mut self,
        recipient_peer_id: u32,
        exchange: JoinCodeKeyExchange,
    ) -> Result<(), RelayClientError> {
        let plaintext_field_frame = FieldMessage::JoinCodeKeyExchange(exchange)
            .encode_frame()
            .map_err(|error| RelayClientError::EncodeFailed(error.0))?;
        self.scheduler.enqueue_control(QueuedControlFrame {
            recipient_peer_id,
            relay_frame_kind: RelayFrameKind::JoinCodeKeyExchange,
            plaintext_field_frame,
        })?;
        Ok(())
    }

    /// Chunks a `ClipPayload`/`TapToHearPayload` below control traffic; it is
    /// abandoned if not fully sent within `abandon_after_milliseconds`
    /// (`None`: the config default; pass `ClipRequest::abandon_after_milliseconds`).
    pub fn send_clip_payload(
        &mut self,
        recipient: NodeId,
        clip_message: &FieldMessage,
        abandon_after_milliseconds: Option<u64>,
        now_milliseconds: u64,
    ) -> Result<(), RelayClientError> {
        if self.sealer.is_none() {
            return Err(RelayClientError::NoSessionKeyYet);
        }
        let request_id = match clip_message {
            FieldMessage::ClipPayload(payload) => payload.request_id,
            FieldMessage::TapToHearPayload(payload) => payload.request_id,
            _ => return Err(RelayClientError::NotAClipPayloadMessage),
        };
        let payload_frame = clip_message
            .encode_frame()
            .map_err(|error| RelayClientError::EncodeFailed(error.0))?;
        let chunks = split_payload_frame_into_chunks(
            request_id,
            NodeId(self.local_peer_id),
            &payload_frame,
            self.config.clip_chunk_body_bytes,
        )?;
        let lifetime = abandon_after_milliseconds
            .unwrap_or(self.config.default_clip_abandon_after_milliseconds);
        self.scheduler
            .enqueue_clip(recipient.0, chunks, now_milliseconds + lifetime)?;
        Ok(())
    }

    /// Next relay frame to write, or `None` (nothing queued / not connected).
    pub fn poll_transmit(
        &mut self,
        now_milliseconds: u64,
        random_source: &mut (impl RngCore + CryptoRng),
    ) -> Result<Option<Vec<u8>>, RelayClientError> {
        if self.phase != RelayConnectionPhase::Connected {
            self.scheduler.abandon_late_clips(now_milliseconds);
            return Ok(None);
        }
        let Some(scheduled) = self.scheduler.next_outbound(now_milliseconds) else {
            return Ok(None);
        };
        let (recipient_peer_id, kind, plaintext) = match scheduled {
            ScheduledOutbound::Control(frame) => (
                frame.recipient_peer_id,
                frame.relay_frame_kind,
                frame.plaintext_field_frame,
            ),
            ScheduledOutbound::ClipChunk {
                recipient_peer_id,
                chunk,
            } => (
                recipient_peer_id,
                RelayFrameKind::SealedEnvelope,
                FieldMessage::ClipPayloadChunk(chunk)
                    .encode_frame()
                    .map_err(|error| RelayClientError::EncodeFailed(error.0))?,
            ),
        };
        let body = match kind {
            RelayFrameKind::JoinCodeKeyExchange => plaintext,
            RelayFrameKind::SealedEnvelope => {
                let sealer = self.sealer.as_ref().ok_or(RelayClientError::NoSessionKeyYet)?;
                let routing = EnvelopeRouting {
                    sender_node_id: NodeId(self.local_peer_id),
                    recipient_node_id: NodeId(recipient_peer_id),
                    sender_sequence: self.next_sender_sequence,
                };
                self.next_sender_sequence += 1;
                sealer
                    .seal(routing, &plaintext, random_source)
                    .map_err(|_| RelayClientError::SealFailed)?
            }
        };
        let header = RelayFrameHeader {
            kind,
            sender_peer_id: self.local_peer_id,
            recipient_peer_id,
        };
        Ok(Some(encode_relay_frame(&header, &body)))
    }

    pub fn on_text_frame(&mut self, text: &str) {
        let Ok(notice) = RelayControlNotice::from_json(text) else {
            self.inbound_statistics.unparseable_control_notices += 1;
            return;
        };
        let event = match notice {
            RelayControlNotice::Welcome {
                peer_id,
                role,
                resume_token,
                host_peer_id,
                usage,
                limits,
            } => {
                let resumed = self.has_connected_before;
                self.has_connected_before = true;
                self.resume_token = Some(resume_token);
                self.host_peer_id = host_peer_id;
                self.phase = RelayConnectionPhase::Connected;
                self.backoff.reset();
                self.scheduler.rewind_clips_after_reconnect();
                RelayClientEvent::Connected {
                    peer_id,
                    role,
                    host_peer_id,
                    usage,
                    limits,
                    resumed,
                }
            }
            RelayControlNotice::PeerJoined { peer_id, role } => {
                if role == minimappr_relay_wire::RelayRole::Host {
                    self.host_peer_id = Some(peer_id);
                }
                RelayClientEvent::PeerJoined { peer_id, role }
            }
            RelayControlNotice::PeerLeft { peer_id } => RelayClientEvent::PeerLeft { peer_id },
            RelayControlNotice::FrameRejected {
                reason,
                recipient_peer_id,
            } => RelayClientEvent::FrameRejectedByRelay {
                reason,
                recipient_peer_id,
            },
        };
        self.pending_events.push_back(event);
    }

    pub fn on_binary_frame(&mut self, frame: &[u8], now_milliseconds: u64) {
        self.inbound_statistics.frames_received += 1;
        let Ok((header, body)) = decode_relay_frame(frame) else {
            self.inbound_statistics.malformed_relay_frames += 1;
            return;
        };
        match header.kind {
            RelayFrameKind::JoinCodeKeyExchange => self.on_key_exchange_body(&header, body),
            RelayFrameKind::SealedEnvelope => self.on_sealed_body(&header, body, now_milliseconds),
        }
    }

    fn on_key_exchange_body(&mut self, header: &RelayFrameHeader, body: &[u8]) {
        match decode_field_message_frame(body) {
            Ok(DecodedFrame::Known {
                message: FieldMessage::JoinCodeKeyExchange(exchange),
                ..
            }) => self
                .pending_events
                .push_back(RelayClientEvent::KeyExchangeFlightReceived {
                    sender_peer_id: header.sender_peer_id,
                    exchange,
                }),
            // Only key-exchange flights may travel unsealed; anything else
            // in this lane is a relay or attacker injection.
            _ => self.inbound_statistics.undecodable_field_frames += 1,
        }
    }

    fn on_sealed_body(&mut self, header: &RelayFrameHeader, body: &[u8], now_milliseconds: u64) {
        let Some(sealer) = self.sealer.as_ref() else {
            self.inbound_statistics.sealed_frames_before_session_key += 1;
            return;
        };
        let Ok((routing, plaintext)) = sealer.open(body) else {
            self.inbound_statistics.authentication_failures += 1;
            return;
        };
        if routing.sender_node_id.0 != header.sender_peer_id {
            self.inbound_statistics.sender_mismatches += 1;
            return;
        }
        let recipient = routing.recipient_node_id.0;
        if recipient != self.local_peer_id && recipient != BROADCAST_PEER_ID {
            self.inbound_statistics.misaddressed_envelopes += 1;
            return;
        }
        if !self
            .replay_window
            .accept(routing.sender_node_id, routing.sender_sequence)
        {
            self.inbound_statistics.replays_dropped += 1;
            return;
        }
        self.deliver_plaintext_frame(routing.sender_node_id, &plaintext, now_milliseconds, true);
    }

    fn deliver_plaintext_frame(
        &mut self,
        sender: NodeId,
        plaintext_field_frame: &[u8],
        now_milliseconds: u64,
        chunks_allowed: bool,
    ) {
        match decode_field_message_frame(plaintext_field_frame) {
            Ok(DecodedFrame::Known {
                message: FieldMessage::ClipPayloadChunk(chunk),
                ..
            }) if chunks_allowed => {
                match self.reassembler.accept_chunk(sender, chunk, now_milliseconds) {
                    Ok(ClipReassemblyProgress::Complete(payload_frame)) => {
                        // A reassembled frame may not itself be a chunk.
                        self.deliver_plaintext_frame(sender, &payload_frame, now_milliseconds, false);
                    }
                    Ok(_) => {}
                    Err(_) => self.inbound_statistics.clip_chunks_rejected += 1,
                }
            }
            Ok(DecodedFrame::Known { message, .. }) => self
                .pending_events
                .push_back(RelayClientEvent::MessageReceived { sender, message }),
            Ok(DecodedFrame::UnknownKind {
                message_kind_id, ..
            }) => self
                .pending_events
                .push_back(RelayClientEvent::UnknownMessageKindReceived {
                    sender,
                    message_kind_id,
                }),
            Err(_) => self.inbound_statistics.undecodable_field_frames += 1,
        }
    }
}
