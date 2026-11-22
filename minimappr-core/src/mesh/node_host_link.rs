//! A node's view of its host: which host and term it follows, when it last
//! heard from it, and the two ways that view breaks.
//!
//! - **Host gone, link alive** (relay up, host silent): the next eligible
//!   node by join order takes over from its mirror; later ranks wait one
//!   stagger each, so normally exactly one takes over.
//! - **Host unreachable** (walked out of the hotspot, relay dropped us):
//!   after the solo timeout the node runs solo, queueing detections, and
//!   rejoins on the next host announcement it hears.
//!
//! A node never takes over while its own link is down: it cannot tell a
//! dead host from its own isolation, and an automatic second hotspot is
//! exactly the offline federation the plan rules out.

use crate::protocol::mesh_messages::HostAnnouncement;
use crate::protocol::NodeId;

use super::config::LivenessConfig;
use super::MeshMilliseconds;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostLinkEvent {
    EnteredSolo,
    TakeOver { host_term: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnnouncementOutcome {
    pub accepted: bool,
    pub host_changed: bool,
    pub rejoined_from_solo: bool,
}

#[derive(Debug, Clone)]
pub struct HostLink {
    pub host_node_id: NodeId,
    pub host_term: u32,
    pub last_heard_from_host_ms: MeshMilliseconds,
    pub link_up: bool,
    pub solo: bool,
    pub host_head_sequence: u64,
    takeover_claimed_term: Option<u32>,
}

impl HostLink {
    pub fn new(initial_host: NodeId, now_ms: MeshMilliseconds) -> Self {
        Self {
            host_node_id: initial_host,
            host_term: 0,
            last_heard_from_host_ms: now_ms,
            link_up: true,
            solo: false,
            host_head_sequence: 0,
            takeover_claimed_term: None,
        }
    }

    pub fn connected(&self) -> bool {
        self.link_up && !self.solo
    }

    pub fn note_message_from(&mut self, from: NodeId, now_ms: MeshMilliseconds) {
        if from == self.host_node_id {
            self.last_heard_from_host_ms = now_ms;
        }
    }

    /// Follow the newest term; an equal term from the current host refreshes.
    pub fn on_announcement(&mut self, announcement: &HostAnnouncement, now_ms: MeshMilliseconds) -> AnnouncementOutcome {
        let newer = announcement.host_term > self.host_term;
        let same_host = announcement.host_term == self.host_term && announcement.host_node_id == self.host_node_id;
        if !(newer || same_host) {
            return AnnouncementOutcome { accepted: false, host_changed: false, rejoined_from_solo: false };
        }
        let host_changed = announcement.host_node_id != self.host_node_id;
        self.host_node_id = announcement.host_node_id;
        self.host_term = announcement.host_term;
        self.host_head_sequence = self.host_head_sequence.max(announcement.event_log_head_sequence);
        self.last_heard_from_host_ms = now_ms;
        let rejoined_from_solo = self.solo && self.link_up;
        if rejoined_from_solo {
            self.solo = false;
        }
        AnnouncementOutcome { accepted: true, host_changed, rejoined_from_solo }
    }

    /// This node became host itself.
    pub fn adopt_self_as_host(&mut self, own_node: NodeId, host_term: u32, now_ms: MeshMilliseconds) {
        self.host_node_id = own_node;
        self.host_term = host_term;
        self.last_heard_from_host_ms = now_ms;
        self.solo = false;
    }

    /// `succession_rank`: this node's position among eligible successors
    /// (current host excluded), `None` if not eligible.
    pub fn tick(&mut self, now_ms: MeshMilliseconds, succession_rank: Option<usize>, config: &LivenessConfig) -> Option<HostLinkEvent> {
        let silent_ms = now_ms.saturating_sub(self.last_heard_from_host_ms);
        if self.link_up {
            if let Some(rank) = succession_rank {
                let wait = config.host_silence_takeover_timeout_ms + rank as u64 * config.takeover_stagger_per_rank_ms;
                let next_term = self.host_term + 1;
                if silent_ms >= wait && self.takeover_claimed_term.is_none_or(|claimed| claimed < next_term) {
                    self.takeover_claimed_term = Some(next_term);
                    return Some(HostLinkEvent::TakeOver { host_term: next_term });
                }
            }
        }
        if !self.solo && silent_ms >= config.solo_fallback_timeout_ms {
            self.solo = true;
            return Some(HostLinkEvent::EnteredSolo);
        }
        None
    }
}
