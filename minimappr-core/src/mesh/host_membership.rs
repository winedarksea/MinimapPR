//! Who is in the session: join order (handover succession), descriptors,
//! presence, and the latest heartbeat facts the host schedules around
//! (thermal headroom, motion, GNSS fix, Mic-only, listen-only, muted).

use std::collections::BTreeMap;

use crate::protocol::mesh_messages::SessionMemberRecord;
use crate::protocol::session_messages::{GnssFix, Heartbeat, NodeDescriptor};
use crate::protocol::wire_codes::{MotionState, NodeKind};
use crate::protocol::NodeId;

use super::MeshMilliseconds;

#[derive(Debug, Clone)]
pub struct MemberRecord {
    pub node_id: NodeId,
    pub join_order: u32,
    pub descriptor: NodeDescriptor,
    pub self_path_length_m: f64,
    pub last_heard_ms: Option<MeshMilliseconds>,
    pub thermal_headroom: f32,
    pub motion_state: MotionState,
    pub mic_only: bool,
    /// Chirps moved off the speaker (e.g. iOS with earbuds): no slot.
    pub listen_only: bool,
    /// Own chirp inaudible in its self-recording: excluded from ranging.
    pub muted: bool,
    pub latest_gnss_fix: Option<GnssFix>,
    pub last_overrun_count: u32,
}

/// A PWA can never host (plan §Host handover).
pub fn descriptor_is_host_eligible(descriptor: &NodeDescriptor) -> bool {
    descriptor.can_host && descriptor.node_kind != NodeKind::BrowserPwa
}

#[derive(Debug, Clone, Default)]
pub struct SessionMembership {
    pub members: BTreeMap<NodeId, MemberRecord>,
    next_join_order: u32,
}

/// What a heartbeat changed that the host must act on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HeartbeatChanges {
    pub capture_overrun_increased: bool,
}

impl SessionMembership {
    /// Admit a joiner (idempotent: a rejoin keeps its join order).
    pub fn admit(&mut self, node_id: NodeId, descriptor: NodeDescriptor, self_path_length_m: f64) -> Option<SessionMemberRecord> {
        if self.members.contains_key(&node_id) {
            return None;
        }
        let join_order = self.next_join_order;
        self.next_join_order += 1;
        let record = SessionMemberRecord {
            node_id,
            join_order,
            node_descriptor: descriptor,
            self_path_length_m: self_path_length_m as f32,
        };
        self.restore(&record);
        Some(record)
    }

    /// Rebuild from a logged member record (successor host).
    pub fn restore(&mut self, record: &SessionMemberRecord) {
        self.next_join_order = self.next_join_order.max(record.join_order + 1);
        self.members.entry(record.node_id).or_insert_with(|| MemberRecord {
            node_id: record.node_id,
            join_order: record.join_order,
            mic_only: !record.node_descriptor.can_classify,
            listen_only: !record.node_descriptor.can_emit_chirps,
            descriptor: record.node_descriptor.clone(),
            self_path_length_m: record.self_path_length_m as f64,
            last_heard_ms: None,
            thermal_headroom: 1.0,
            motion_state: MotionState::Unknown,
            muted: false,
            latest_gnss_fix: None,
            last_overrun_count: 0,
        });
    }

    pub fn note_heard(&mut self, node: NodeId, now_ms: MeshMilliseconds) {
        if let Some(member) = self.members.get_mut(&node) {
            member.last_heard_ms = Some(now_ms);
        }
    }

    pub fn apply_heartbeat(&mut self, heartbeat: &Heartbeat) -> HeartbeatChanges {
        let Some(member) = self.members.get_mut(&heartbeat.node_id) else {
            return HeartbeatChanges::default();
        };
        let changes = HeartbeatChanges {
            capture_overrun_increased: heartbeat.capture_overrun_count > member.last_overrun_count,
        };
        member.thermal_headroom = heartbeat.thermal_headroom_fraction;
        member.motion_state = heartbeat.motion_state;
        member.mic_only = heartbeat.mic_only_mode || !member.descriptor.can_classify;
        member.last_overrun_count = heartbeat.capture_overrun_count;
        if heartbeat.gnss_fix.is_some() {
            member.latest_gnss_fix = heartbeat.gnss_fix.clone();
        }
        changes
    }

    pub fn is_present(&self, node: NodeId, now_ms: MeshMilliseconds, timeout_ms: MeshMilliseconds) -> bool {
        self.members
            .get(&node)
            .and_then(|member| member.last_heard_ms)
            .is_some_and(|heard| now_ms.saturating_sub(heard) <= timeout_ms)
    }

    pub fn present_nodes(&self, now_ms: MeshMilliseconds, timeout_ms: MeshMilliseconds) -> Vec<NodeId> {
        self.members
            .keys()
            .copied()
            .filter(|node| self.is_present(*node, now_ms, timeout_ms))
            .collect()
    }

    /// Host-eligible members in succession order.
    pub fn succession_order(&self) -> Vec<NodeId> {
        let mut eligible: Vec<&MemberRecord> = self
            .members
            .values()
            .filter(|member| descriptor_is_host_eligible(&member.descriptor))
            .collect();
        eligible.sort_by_key(|member| member.join_order);
        eligible.into_iter().map(|member| member.node_id).collect()
    }

    pub fn self_path_lengths(&self) -> BTreeMap<NodeId, f64> {
        self.members
            .values()
            .map(|member| (member.node_id, member.self_path_length_m))
            .collect()
    }
}
