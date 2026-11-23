//! Successor host: rebuild host state from a node's mirrored event log
//! (plan §Host handover: "takes over from its mirror").
//!
//! Restored: the log itself (sequence numbers continue, detection keys
//! stay idempotent), membership and join order, the entity id counter, the
//! active transport, the previous term, the session's ENU origin, and every
//! live bird entity's latest update (with each island's last clock model, to
//! carry entity times onto the successor's reference clock), plus each
//! island's last published layout, only to tell which calibration scale a
//! restored entity's contour was drawn with. Not restored as live state:
//! clock filters and layouts. The successor runs a chirp round at once (its scheduler starts
//! fresh), which re-surveys every island within one round; reusing a dead
//! host's clock models would risk a stale model on the first new fix.

use std::collections::BTreeMap;

use crate::protocol::detection_messages::BirdEntityUpdate;
use crate::protocol::mesh_messages::EventLogEntry;
use crate::protocol::session_messages::{ClockModel, LayoutModel};
use crate::protocol::wire_codes::{BirdEntityState, SessionTransportCode};
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
    /// Latest non-expired update per (island, wire entity id).
    pub entities: BTreeMap<(u32, u64), BirdEntityUpdate>,
    /// Latest clock model per island (entity times are on its reference clock).
    pub clock_models: BTreeMap<u32, ClockModel>,
    /// (latitude, longitude) of the ENU origin the log's positions use.
    pub local_plane_origin_deg: Option<(f64, f64)>,
    /// Latest published layout per island (node positions in the fix frame).
    pub layouts: BTreeMap<u32, LayoutModel>,
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
                let key = (update.island_id, update.bird_entity_id);
                if update.state == BirdEntityState::Expired {
                    restored.entities.remove(&key);
                } else {
                    restored.entities.insert(key, update);
                }
            }
            Some(FieldMessage::ClockModel(model)) => {
                restored.clock_models.insert(model.island_id, model);
            }
            // An unset plane is published as (0, 0) (`island_model_records`).
            Some(FieldMessage::LayoutModel(layout)) => {
                if layout.origin_latitude_deg != 0.0 || layout.origin_longitude_deg != 0.0 {
                    restored.local_plane_origin_deg = Some((layout.origin_latitude_deg, layout.origin_longitude_deg));
                }
                restored.layouts.insert(layout.island_id, layout);
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
