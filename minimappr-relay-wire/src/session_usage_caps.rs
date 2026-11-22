//! Per-session relay budget. Every inbound WebSocket message is a billed
//! Durable Object request, so the relay caps messages and bytes per session
//! (plan §Online relay, Cost) instead of letting one runaway client or a
//! code-holding stranger spend the month's allowance.

use serde::{Deserialize, Serialize};

use crate::relay_routing_policy::FrameRejectionReason;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionRelayLimits {
    pub maximum_inbound_messages_per_session: u64,
    pub maximum_inbound_bytes_per_session: u64,
    pub maximum_frame_bytes: u64,
    pub maximum_connections_per_session: u32,
}

/// What a session has consumed so far (persisted by the Durable Object).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionUsageCounters {
    pub inbound_messages: u64,
    pub inbound_bytes: u64,
}

impl SessionUsageCounters {
    /// Admits and counts one inbound frame, or says which cap it breaks.
    /// Rejected frames are not counted, so the counters never pass the caps.
    pub fn try_admit_frame(
        &mut self,
        frame_length_bytes: u64,
        limits: &SessionRelayLimits,
    ) -> Result<(), FrameRejectionReason> {
        if frame_length_bytes > limits.maximum_frame_bytes {
            return Err(FrameRejectionReason::FrameLargerThanLimit);
        }
        if self.inbound_messages >= limits.maximum_inbound_messages_per_session {
            return Err(FrameRejectionReason::SessionMessageCapReached);
        }
        if self.inbound_bytes.saturating_add(frame_length_bytes)
            > limits.maximum_inbound_bytes_per_session
        {
            return Err(FrameRejectionReason::SessionByteCapReached);
        }
        self.inbound_messages += 1;
        self.inbound_bytes += frame_length_bytes;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: SessionRelayLimits = SessionRelayLimits {
        maximum_inbound_messages_per_session: 3,
        maximum_inbound_bytes_per_session: 100,
        maximum_frame_bytes: 60,
        maximum_connections_per_session: 4,
    };

    #[test]
    fn message_cap_stops_the_fourth_frame() {
        let mut counters = SessionUsageCounters::default();
        for _ in 0..3 {
            counters.try_admit_frame(10, &LIMITS).unwrap();
        }
        assert_eq!(
            counters.try_admit_frame(10, &LIMITS),
            Err(FrameRejectionReason::SessionMessageCapReached)
        );
        assert_eq!(counters.inbound_messages, 3);
    }

    #[test]
    fn byte_cap_and_frame_size_are_enforced_without_counting_rejects() {
        let mut counters = SessionUsageCounters::default();
        counters.try_admit_frame(60, &LIMITS).unwrap();
        assert_eq!(
            counters.try_admit_frame(61, &LIMITS),
            Err(FrameRejectionReason::FrameLargerThanLimit)
        );
        assert_eq!(
            counters.try_admit_frame(41, &LIMITS),
            Err(FrameRejectionReason::SessionByteCapReached)
        );
        counters.try_admit_frame(40, &LIMITS).unwrap();
        assert_eq!(counters.inbound_bytes, 100);
        assert_eq!(counters.inbound_messages, 2);
    }
}
