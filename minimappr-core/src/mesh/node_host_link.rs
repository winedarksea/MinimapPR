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
use crate::protocol::wire_codes::SessionTransportCode;
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
    /// Transport the host last announced. Only on the relay can a live link
    /// coexist with a dead host; on the hotspot the link dies with its host,
    /// so a silent host on a live hotspot link means loss, not death.
    pub session_transport: SessionTransportCode,
    /// When the link last came up: a node that just reconnected has not
    /// had a chance to hear the host yet, so silence counts from here.
    link_up_since_ms: MeshMilliseconds,
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
            session_transport: SessionTransportCode::CloudRelay,
            link_up_since_ms: now_ms,
        }
    }

    pub fn set_link_up(&mut self, link_up: bool, now_ms: MeshMilliseconds) {
        if link_up && !self.link_up {
            self.link_up_since_ms = now_ms;
        }
        self.link_up = link_up;
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
    /// Two hosts claiming one term (simultaneous takeover) are resolved by
    /// join order: `announcer_wins_tie` says the announcer joined earlier.
    pub fn on_announcement(&mut self, announcement: &HostAnnouncement, announcer_wins_tie: bool, now_ms: MeshMilliseconds) -> AnnouncementOutcome {
        let newer = announcement.host_term > self.host_term;
        let same_term = announcement.host_term == self.host_term;
        let same_host = same_term && announcement.host_node_id == self.host_node_id;
        let wins_tie = same_term && announcement.host_node_id != self.host_node_id && announcer_wins_tie;
        if !(newer || same_host || wins_tie) {
            return AnnouncementOutcome { accepted: false, host_changed: false, rejoined_from_solo: false };
        }
        let host_changed = announcement.host_node_id != self.host_node_id;
        self.host_node_id = announcement.host_node_id;
        self.host_term = announcement.host_term;
        self.session_transport = announcement.transport;
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
        let silent_while_linked_ms = now_ms.saturating_sub(self.last_heard_from_host_ms.max(self.link_up_since_ms));
        if self.link_up && self.session_transport == SessionTransportCode::CloudRelay {
            if let Some(rank) = succession_rank {
                let wait = config.host_silence_takeover_timeout_ms + rank as u64 * config.takeover_stagger_per_rank_ms;
                let next_term = self.host_term + 1;
                if silent_while_linked_ms >= wait && self.takeover_claimed_term.is_none_or(|claimed| claimed < next_term) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::wire_codes::{SessionChangeReasonCode, SessionTransportCode};

    fn announcement(host: u32, term: u32) -> HostAnnouncement {
        HostAnnouncement {
            host_node_id: NodeId(host),
            host_term: term,
            event_log_head_sequence: 0,
            transport: SessionTransportCode::CloudRelay,
            reason: SessionChangeReasonCode::SessionStarted,
        }
    }

    #[test]
    fn rank_zero_takes_over_only_after_silence_on_a_live_link() {
        let config = LivenessConfig::default();
        let mut link = HostLink::new(NodeId(0), 0);
        link.on_announcement(&announcement(0, 1), false, 0);
        assert_eq!(link.tick(config.host_silence_takeover_timeout_ms - 1, Some(0), &config), None);
        assert_eq!(
            link.tick(config.host_silence_takeover_timeout_ms, Some(0), &config),
            Some(HostLinkEvent::TakeOver { host_term: 2 })
        );
        // Claimed once per term.
        assert_eq!(link.tick(config.host_silence_takeover_timeout_ms + 1, Some(0), &config), None);
    }

    /// Regression (found by the mesh sim): a phone walking back into hotspot
    /// range saw a long host silence and hijacked the session.
    #[test]
    fn reconnecting_node_waits_to_hear_the_host_before_any_takeover() {
        let config = LivenessConfig::default();
        let mut link = HostLink::new(NodeId(0), 0);
        link.on_announcement(&announcement(0, 1), false, 0);
        link.set_link_up(false, 1_000);
        assert_eq!(link.tick(config.solo_fallback_timeout_ms, Some(0), &config), Some(HostLinkEvent::EnteredSolo));
        let back_ms = 60_000;
        link.set_link_up(true, back_ms);
        assert_eq!(link.tick(back_ms + 100, Some(0), &config), None);
        let outcome = link.on_announcement(&announcement(0, 1), false, back_ms + 500);
        assert!(outcome.rejoined_from_solo && !link.solo);
    }

    #[test]
    fn equal_term_rival_wins_only_with_earlier_join_order() {
        let mut link = HostLink::new(NodeId(4), 0);
        link.on_announcement(&announcement(4, 2), false, 0);
        assert!(!link.on_announcement(&announcement(3, 2), false, 10).accepted);
        let outcome = link.on_announcement(&announcement(3, 2), true, 20);
        assert!(outcome.accepted && outcome.host_changed && link.host_node_id == NodeId(3));
    }
}
