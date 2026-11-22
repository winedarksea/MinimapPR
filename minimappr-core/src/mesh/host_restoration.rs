//! Successor host: rebuild host state from a node's mirrored event log
//! (plan §Host handover: "takes over from its mirror").
//!
//! Restored: the log itself (sequence numbers continue, detection keys
//! stay idempotent), membership and join order, the entity id counter, the
//! active transport and the previous term. Not restored: clock filters and
//! layouts. The successor runs a chirp round at once (its scheduler starts
//! fresh), which re-surveys every island within one round; reusing a dead
//! host's clock models would risk a stale model on the first new fix.

use crate::protocol::mesh_messages::EventLogEntry;
use crate::protocol::wire_codes::SessionTransportCode;
use crate::protocol::FieldMessage;

use super::event_log::{decode_entry, SessionEventLog};
use super::host_membership::SessionMembership;

#[derive(Debug, Clone, Default)]
pub struct RestoredHostState {
    pub event_log: SessionEventLog,
    pub membership: SessionMembership,
    pub highest_bird_entity_id: u64,
    pub highest_host_term: u32,
    pub active_transport: Option<SessionTransportCode>,
}

pub fn restore_from_mirror(entries: &[EventLogEntry]) -> RestoredHostState {
    let mut restored = RestoredHostState {
        event_log: SessionEventLog::from_mirror(entries),
        ..RestoredHostState::default()
    };
    for entry in entries {
        restored.highest_host_term = restored.highest_host_term.max(entry.host_term);
        match decode_entry(entry) {
            Some(FieldMessage::SessionMemberRecord(record)) => restored.membership.restore(&record),
            Some(FieldMessage::BirdEntityUpdate(update)) => {
                restored.highest_bird_entity_id = restored.highest_bird_entity_id.max(update.bird_entity_id);
            }
            Some(FieldMessage::TransportNotice(notice)) => restored.active_transport = Some(notice.transport),
            Some(FieldMessage::HostAnnouncement(announcement)) => {
                restored.highest_host_term = restored.highest_host_term.max(announcement.host_term);
                restored.active_transport = Some(announcement.transport);
            }
            _ => {}
        }
    }
    restored
}
