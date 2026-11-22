//! Relay-to-client control notices, sent as WebSocket *text* frames (JSON) so
//! they can never be confused with binary relay frames. They carry only
//! relay metadata (peer ids, roles, counters), never session content.

use serde::{Deserialize, Serialize};

use crate::relay_routing_policy::{FrameRejectionReason, RelayRole};
use crate::session_usage_caps::{SessionRelayLimits, SessionUsageCounters};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "notice", rename_all = "snake_case")]
pub enum RelayControlNotice {
    /// First frame on every accepted socket.
    Welcome {
        peer_id: u32,
        role: RelayRole,
        /// Present the same token to rebind this peer id after a drop; the
        /// relay refuses a second claim of a bound peer id without it.
        resume_token: String,
        host_peer_id: Option<u32>,
        usage: SessionUsageCounters,
        limits: SessionRelayLimits,
    },
    PeerJoined {
        peer_id: u32,
        role: RelayRole,
    },
    PeerLeft {
        peer_id: u32,
    },
    /// The sender's frame was dropped (not forwarded).
    FrameRejected {
        reason: FrameRejectionReason,
        recipient_peer_id: Option<u32>,
    },
}

impl RelayControlNotice {
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("notice serializes")
    }

    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notices_round_trip_through_json() {
        let notice = RelayControlNotice::FrameRejected {
            reason: FrameRejectionReason::SessionByteCapReached,
            recipient_peer_id: Some(4),
        };
        let json = notice.to_json();
        assert!(json.contains("\"notice\":\"frame_rejected\""));
        assert_eq!(RelayControlNotice::from_json(&json).unwrap(), notice);
    }
}
