//! Role tokens and the relay's forwarding policy, as a pure function so the
//! Worker's Durable Object is only I/O glue and the policy is unit-tested.

use serde::{Deserialize, Serialize};

use crate::relay_frame::{RelayFrameHeader, RelayFrameKind, BROADCAST_PEER_ID};

/// Capability a role token grants on the relay. Independent of the in-session
/// role the host later assigns (sealed `JoinAccepted`), which the relay
/// cannot see.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelayRole {
    /// Session creator: may send anything, receives key-exchange flights.
    Host,
    /// Contributing phone or array node: may send sealed frames to anyone.
    Node,
    /// View-only map: receives everything, may send only key-exchange
    /// flights (to get the session key) to the host.
    Viewer,
}

impl RelayRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::Node => "node",
            Self::Viewer => "viewer",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "host" => Some(Self::Host),
            "node" => Some(Self::Node),
            "viewer" => Some(Self::Viewer),
            _ => None,
        }
    }
}

/// Where the relay delivers an accepted frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelayDelivery {
    /// The one socket bound to this peer id.
    SinglePeer(u32),
    /// Every socket except the sender's.
    EveryOtherPeer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameRejectionReason {
    MalformedRelayFrame,
    FrameLargerThanLimit,
    /// Header sender differs from the peer id this socket bound at connect.
    SenderPeerIdMismatch,
    /// Viewers may only send key-exchange flights.
    RoleMayNotSendSealedFrames,
    /// Non-host key-exchange flights must be addressed to the host.
    KeyExchangeMustTargetHost,
    /// A key-exchange flight was sent before any host connected.
    NoHostConnected,
    /// Host key-exchange replies go to one joiner, never broadcast.
    KeyExchangeMayNotBroadcast,
    RecipientNotConnected,
    SessionMessageCapReached,
    SessionByteCapReached,
    TextFramesNotAccepted,
}

/// Decides how to forward one frame. `host_peer_id` is the peer currently
/// bound by a host-role socket, if any.
pub fn decide_relay_delivery(
    sender_role: RelayRole,
    bound_sender_peer_id: u32,
    host_peer_id: Option<u32>,
    header: &RelayFrameHeader,
) -> Result<RelayDelivery, FrameRejectionReason> {
    if header.sender_peer_id != bound_sender_peer_id {
        return Err(FrameRejectionReason::SenderPeerIdMismatch);
    }
    match (header.kind, sender_role) {
        (RelayFrameKind::SealedEnvelope, RelayRole::Viewer) => {
            Err(FrameRejectionReason::RoleMayNotSendSealedFrames)
        }
        (RelayFrameKind::SealedEnvelope, _) => Ok(delivery_for(header.recipient_peer_id)),
        (RelayFrameKind::JoinCodeKeyExchange, RelayRole::Host) => {
            if header.recipient_peer_id == BROADCAST_PEER_ID {
                Err(FrameRejectionReason::KeyExchangeMayNotBroadcast)
            } else {
                Ok(RelayDelivery::SinglePeer(header.recipient_peer_id))
            }
        }
        (RelayFrameKind::JoinCodeKeyExchange, RelayRole::Node | RelayRole::Viewer) => {
            let host_peer_id = host_peer_id.ok_or(FrameRejectionReason::NoHostConnected)?;
            if header.recipient_peer_id != host_peer_id {
                return Err(FrameRejectionReason::KeyExchangeMustTargetHost);
            }
            Ok(RelayDelivery::SinglePeer(host_peer_id))
        }
    }
}

fn delivery_for(recipient_peer_id: u32) -> RelayDelivery {
    if recipient_peer_id == BROADCAST_PEER_ID {
        RelayDelivery::EveryOtherPeer
    } else {
        RelayDelivery::SinglePeer(recipient_peer_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOST: u32 = 1;
    const JOINER: u32 = 0x8000_0042;

    fn header(kind: RelayFrameKind, sender: u32, recipient: u32) -> RelayFrameHeader {
        RelayFrameHeader {
            kind,
            sender_peer_id: sender,
            recipient_peer_id: recipient,
        }
    }

    #[test]
    fn sealed_frames_route_by_recipient_or_broadcast() {
        let unicast = header(RelayFrameKind::SealedEnvelope, 5, HOST);
        assert_eq!(
            decide_relay_delivery(RelayRole::Node, 5, Some(HOST), &unicast),
            Ok(RelayDelivery::SinglePeer(HOST))
        );
        let broadcast = header(RelayFrameKind::SealedEnvelope, HOST, BROADCAST_PEER_ID);
        assert_eq!(
            decide_relay_delivery(RelayRole::Host, HOST, Some(HOST), &broadcast),
            Ok(RelayDelivery::EveryOtherPeer)
        );
    }

    #[test]
    fn spoofed_sender_and_viewer_sealed_frames_are_rejected() {
        let spoofed = header(RelayFrameKind::SealedEnvelope, HOST, 5);
        assert_eq!(
            decide_relay_delivery(RelayRole::Node, 5, Some(HOST), &spoofed),
            Err(FrameRejectionReason::SenderPeerIdMismatch)
        );
        let viewer_sealed = header(RelayFrameKind::SealedEnvelope, 9, HOST);
        assert_eq!(
            decide_relay_delivery(RelayRole::Viewer, 9, Some(HOST), &viewer_sealed),
            Err(FrameRejectionReason::RoleMayNotSendSealedFrames)
        );
    }

    #[test]
    fn key_exchange_flights_only_flow_between_host_and_one_joiner() {
        let to_host = header(RelayFrameKind::JoinCodeKeyExchange, JOINER, HOST);
        assert_eq!(
            decide_relay_delivery(RelayRole::Viewer, JOINER, Some(HOST), &to_host),
            Ok(RelayDelivery::SinglePeer(HOST))
        );
        assert_eq!(
            decide_relay_delivery(RelayRole::Node, JOINER, None, &to_host),
            Err(FrameRejectionReason::NoHostConnected)
        );
        let to_peer = header(RelayFrameKind::JoinCodeKeyExchange, JOINER, 7);
        assert_eq!(
            decide_relay_delivery(RelayRole::Node, JOINER, Some(HOST), &to_peer),
            Err(FrameRejectionReason::KeyExchangeMustTargetHost)
        );
        let host_reply = header(RelayFrameKind::JoinCodeKeyExchange, HOST, JOINER);
        assert_eq!(
            decide_relay_delivery(RelayRole::Host, HOST, Some(HOST), &host_reply),
            Ok(RelayDelivery::SinglePeer(JOINER))
        );
        let host_broadcast = header(RelayFrameKind::JoinCodeKeyExchange, HOST, BROADCAST_PEER_ID);
        assert_eq!(
            decide_relay_delivery(RelayRole::Host, HOST, Some(HOST), &host_broadcast),
            Err(FrameRejectionReason::KeyExchangeMayNotBroadcast)
        );
    }

    #[test]
    fn role_names_round_trip() {
        for role in [RelayRole::Host, RelayRole::Node, RelayRole::Viewer] {
            assert_eq!(RelayRole::parse(role.as_str()), Some(role));
        }
        assert_eq!(RelayRole::parse("admin"), None);
    }
}
