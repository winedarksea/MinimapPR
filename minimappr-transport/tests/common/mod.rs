//! Shared helpers for transport tests: sample messages, a seeded CSPRNG, and
//! an in-memory relay that applies the same `minimappr-relay-wire` routing
//! and cap policy as the Durable Object.
#![allow(dead_code)]

use std::collections::BTreeMap;

use minimappr_core::protocol::detection_messages::{ClipOnset, ClipPayload, ReadinessQuery};
use minimappr_core::protocol::wire_codes::FrequencyBandHz;
use minimappr_core::protocol::{FieldMessage, NodeId, SessionId};
use minimappr_relay_wire::{
    decide_relay_delivery, RelayControlNotice, RelayDelivery, RelayFrameHeader, RelayRole,
    SessionRelayLimits, SessionUsageCounters,
};
use minimappr_transport::{RelayClientEvent, RelayClientSession};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;

pub const TEST_SESSION_ID: SessionId = SessionId([0x5A; 16]);

pub fn seeded_random_source(seed: u64) -> ChaCha20Rng {
    ChaCha20Rng::seed_from_u64(seed)
}

pub fn readiness_query(query_id: u32) -> FieldMessage {
    FieldMessage::ReadinessQuery(ReadinessQuery {
        query_id,
        target_node_id: NodeId(3),
        centre_device_sample_index: 123_456_789,
        half_window_samples: 36_000,
        band: FrequencyBandHz {
            low_hz: 2_000.0,
            high_hz: 8_000.0,
        },
    })
}

/// A clip whose container is `container_bytes` of a deterministic pattern.
pub fn clip_payload(request_id: u32, node_id: NodeId, container_bytes: usize) -> FieldMessage {
    FieldMessage::ClipPayload(ClipPayload {
        request_id,
        node_id,
        clip_container: (0..container_bytes).map(|index| (index * 31 % 251) as u8).collect(),
        onsets: vec![ClipOnset {
            device_sample_index: 48_000,
            positive_flux_db: 9.5,
        }],
        sensor_frames: Vec::new(),
    })
}

pub const GENEROUS_LIMITS: SessionRelayLimits = SessionRelayLimits {
    maximum_inbound_messages_per_session: 10_000,
    maximum_inbound_bytes_per_session: 64 * 1024 * 1024,
    maximum_frame_bytes: 64 * 1024,
    maximum_connections_per_session: 16,
};

struct LoopbackPeer {
    role: RelayRole,
    session: RelayClientSession,
    connected: bool,
}

/// In-memory stand-in for the relay Durable Object.
pub struct LoopbackRelay {
    peers: BTreeMap<u32, LoopbackPeer>,
    pub usage: SessionUsageCounters,
    pub limits: SessionRelayLimits,
    resume_serial: u64,
    pub now_milliseconds: u64,
    random_source: ChaCha20Rng,
}

impl LoopbackRelay {
    pub fn new(limits: SessionRelayLimits) -> Self {
        Self {
            peers: BTreeMap::new(),
            usage: SessionUsageCounters::default(),
            limits,
            resume_serial: 0,
            now_milliseconds: 0,
            random_source: seeded_random_source(99),
        }
    }

    fn host_peer_id(&self) -> Option<u32> {
        self.peers
            .iter()
            .find(|(_, peer)| peer.connected && peer.role == RelayRole::Host)
            .map(|(peer_id, _)| *peer_id)
    }

    pub fn attach(&mut self, role: RelayRole, session: RelayClientSession) {
        let peer_id = session.local_peer_id();
        self.peers.insert(peer_id, LoopbackPeer { role, session, connected: false });
        self.connect(peer_id);
    }

    pub fn connect(&mut self, peer_id: u32) {
        self.resume_serial += 1;
        let host_peer_id = self.host_peer_id();
        let peer = self.peers.get_mut(&peer_id).expect("attached");
        peer.connected = true;
        peer.session.on_socket_opened();
        let welcome = RelayControlNotice::Welcome {
            peer_id,
            role: peer.role,
            resume_token: format!("resume-{}", self.resume_serial),
            host_peer_id: if peer.role == RelayRole::Host { Some(peer_id) } else { host_peer_id },
            usage: self.usage,
            limits: self.limits,
        };
        peer.session.on_text_frame(&welcome.to_json());
        let role = peer.role;
        let joined = RelayControlNotice::PeerJoined { peer_id, role }.to_json();
        for (other_id, other) in self.peers.iter_mut() {
            if *other_id != peer_id && other.connected {
                other.session.on_text_frame(&joined);
            }
        }
    }

    pub fn disconnect(&mut self, peer_id: u32) {
        let now = self.now_milliseconds;
        let peer = self.peers.get_mut(&peer_id).expect("attached");
        peer.connected = false;
        peer.session.on_socket_closed(now);
    }

    pub fn session(&mut self, peer_id: u32) -> &mut RelayClientSession {
        &mut self.peers.get_mut(&peer_id).expect("attached").session
    }

    /// Moves frames until no connected peer has anything to send.
    pub fn run_until_quiet(&mut self) -> usize {
        let mut forwarded = 0;
        loop {
            let mut outbound = Vec::new();
            let now = self.now_milliseconds;
            for (peer_id, peer) in self.peers.iter_mut() {
                if !peer.connected {
                    continue;
                }
                while let Some(frame) = peer.session.poll_transmit(now, &mut self.random_source).unwrap() {
                    outbound.push((*peer_id, peer.role, frame));
                }
            }
            if outbound.is_empty() {
                return forwarded;
            }
            for (sender_peer_id, sender_role, frame) in outbound {
                forwarded += self.route(sender_peer_id, sender_role, &frame);
            }
        }
    }

    fn reject(&mut self, sender_peer_id: u32, notice: RelayControlNotice) {
        self.session(sender_peer_id).on_text_frame(&notice.to_json());
    }

    fn route(&mut self, sender_peer_id: u32, sender_role: RelayRole, frame: &[u8]) -> usize {
        if let Err(reason) = self.usage.try_admit_frame(frame.len() as u64, &self.limits) {
            self.reject(sender_peer_id, RelayControlNotice::FrameRejected { reason, recipient_peer_id: None });
            return 0;
        }
        let header = RelayFrameHeader::decode(frame).expect("clients send valid headers");
        let delivery = match decide_relay_delivery(sender_role, sender_peer_id, self.host_peer_id(), &header) {
            Ok(delivery) => delivery,
            Err(reason) => {
                let notice = RelayControlNotice::FrameRejected {
                    reason,
                    recipient_peer_id: Some(header.recipient_peer_id),
                };
                self.reject(sender_peer_id, notice);
                return 0;
            }
        };
        let now = self.now_milliseconds;
        let recipients: Vec<u32> = match delivery {
            RelayDelivery::SinglePeer(peer_id) => vec![peer_id],
            RelayDelivery::EveryOtherPeer => {
                self.peers.keys().copied().filter(|peer_id| *peer_id != sender_peer_id).collect()
            }
        };
        let mut delivered = 0;
        for recipient in recipients {
            if let Some(peer) = self.peers.get_mut(&recipient).filter(|peer| peer.connected) {
                peer.session.on_binary_frame(frame, now);
                delivered += 1;
            }
        }
        delivered
    }
}

pub fn drain_events(session: &mut RelayClientSession) -> Vec<RelayClientEvent> {
    std::iter::from_fn(|| session.poll_event()).collect()
}
