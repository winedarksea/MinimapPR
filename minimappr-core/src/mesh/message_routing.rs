//! Which role on a device consumes an inbound message. A device running the
//! host also runs its own node, so every shell (mesh simulator, phone FFI)
//! splits deliveries the same way: node-to-host traffic goes to the host,
//! everything else to the node, and host announcements to both (a node must
//! follow terms even on the host's own device).

use crate::protocol::FieldMessage;

/// Node → host traffic.
pub fn is_host_bound(message: &FieldMessage) -> bool {
    matches!(
        message,
        FieldMessage::JoinRequest(_)
            | FieldMessage::Heartbeat(_)
            | FieldMessage::DetectionEvent(_)
            | FieldMessage::QueuedDetectionUpload(_)
            | FieldMessage::ChirpArrivalReport(_)
            | FieldMessage::ReadinessAnswer(_)
            | FieldMessage::ClipPayload(_)
            | FieldMessage::NodeStatusChange(_)
            | FieldMessage::EventLogCatchUpRequest(_)
            | FieldMessage::ChirpDirectionReport(_)
    )
}

/// Where one inbound message goes on a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MessageDelivery {
    pub to_host: bool,
    pub to_node: bool,
}

pub fn delivery_for(message: &FieldMessage) -> MessageDelivery {
    let host_bound = is_host_bound(message);
    MessageDelivery {
        to_host: host_bound || matches!(message, FieldMessage::HostAnnouncement(_)),
        to_node: !host_bound,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::mesh_messages::HostAnnouncement;
    use crate::protocol::wire_codes::{SessionChangeReasonCode, SessionTransportCode};
    use crate::protocol::NodeId;

    #[test]
    fn announcements_reach_both_roles() {
        let announcement = FieldMessage::HostAnnouncement(HostAnnouncement {
            host_node_id: NodeId(1),
            host_term: 1,
            event_log_head_sequence: 0,
            transport: SessionTransportCode::LocalHotspot,
            reason: SessionChangeReasonCode::SessionStarted,
        });
        assert_eq!(delivery_for(&announcement), MessageDelivery { to_host: true, to_node: true });
    }
}
