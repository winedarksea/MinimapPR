//! Node-side regressions for two hosts claiming one term (simultaneous
//! takeover when the rank-0 announcement is lost): the mirror must end up
//! an exact prefix of the tie winner's log, never interleaved with the
//! loser's entries, which carry identical `(sequence, term)` pairs.

use crate::protocol::mesh_messages::{EventLogAppend, HostAnnouncement, SessionMemberRecord};
use crate::protocol::session_messages::NodeDescriptor;
use crate::protocol::wire_codes::{
    FrequencyBandHz, NodeKind, SessionChangeReasonCode, SessionId, SessionTransportCode,
};
use crate::protocol::{FieldMessage, NodeId};

use super::config::MeshConfig;
use super::event_log::SessionEventLog;
use super::node_state_machine::{
    EncodedLocalClip, FieldNodeStateMachine, LocalAudioReadiness, NodeAudioWindowSource, NodeInput,
};

const FIRST_HOST: NodeId = NodeId(1);
const TIE_WINNER: NodeId = NodeId(2);
const TIE_LOSER: NodeId = NodeId(3);
const FOLLOWER: NodeId = NodeId(5);
const CONTESTED_TERM: u32 = 2;

struct SilentAudio;

impl NodeAudioWindowSource for SilentAudio {
    fn readiness(&self, _centre: u64, _half_window: u32, _band: FrequencyBandHz) -> LocalAudioReadiness {
        LocalAudioReadiness {
            audio_still_buffered: false,
            speech_free: true,
            handling_noise_free: true,
            in_band_level_over_floor_db: 0.0,
        }
    }

    fn encode_clip(&self, _first: u64, _count: u32, _filter: u8) -> Option<EncodedLocalClip> {
        None
    }
}

fn phone() -> NodeDescriptor {
    NodeDescriptor {
        node_kind: NodeKind::Phone,
        platform_name: "test".into(),
        app_version: "0".into(),
        capture_sample_rate_hz: 48_000,
        microphone_count: 1,
        microphone_positions_body_frame_m: vec![],
        can_classify: true,
        can_host: true,
        can_emit_chirps: true,
        has_pps_clock: false,
    }
}

fn member(node: NodeId, join_order: u32) -> FieldMessage {
    FieldMessage::SessionMemberRecord(SessionMemberRecord {
        node_id: node,
        join_order,
        node_descriptor: phone(),
        self_path_length_m: 0.0,
    })
}

fn announcement(host: NodeId, term: u32, head: u64) -> FieldMessage {
    FieldMessage::HostAnnouncement(HostAnnouncement {
        host_node_id: host,
        host_term: term,
        event_log_head_sequence: head,
        transport: SessionTransportCode::CloudRelay,
        reason: SessionChangeReasonCode::HostSilent,
    })
}

fn append_since(log: &SessionEventLog, host: NodeId, term: u32, since: u64) -> FieldMessage {
    FieldMessage::EventLogAppend(EventLogAppend {
        host_node_id: host,
        host_term: term,
        host_head_sequence: log.head_sequence(),
        entries: log.entries_since(since, usize::MAX).to_vec(),
    })
}

fn deliver(node: &mut FieldNodeStateMachine, from: NodeId, message: FieldMessage, now_ms: u64) {
    node.handle(now_ms, 0, NodeInput::MessageReceived { from, message }, &SilentAudio);
}

#[test]
fn mirror_follows_only_the_tie_winner_of_a_contested_term() {
    let mut first_log = SessionEventLog::default();
    for (order, node) in [FIRST_HOST, TIE_WINNER, TIE_LOSER, FOLLOWER].into_iter().enumerate() {
        first_log.append(1, &member(node, order as u32));
    }
    let inherited = first_log.head_sequence();
    let mut winner_log = SessionEventLog::from_mirror(first_log.entries());
    winner_log.append(CONTESTED_TERM, &announcement(TIE_WINNER, CONTESTED_TERM, inherited));
    let mut loser_log = SessionEventLog::from_mirror(first_log.entries());
    loser_log.append(CONTESTED_TERM, &announcement(TIE_LOSER, CONTESTED_TERM, inherited));
    loser_log.append(CONTESTED_TERM, &member(NodeId(9), 9));

    let mut node = FieldNodeStateMachine::new(FOLLOWER, phone(), SessionId([0; 16]), FIRST_HOST, MeshConfig::default(), 0);
    deliver(&mut node, FIRST_HOST, announcement(FIRST_HOST, 1, inherited), 0);
    deliver(&mut node, FIRST_HOST, append_since(&first_log, FIRST_HOST, 1, 0), 10);
    assert_eq!(node.mirror.entries(), first_log.entries());

    // The loser is heard first and followed (newer term) ...
    deliver(&mut node, TIE_LOSER, announcement(TIE_LOSER, CONTESTED_TERM, loser_log.head_sequence()), 20);
    deliver(&mut node, TIE_LOSER, append_since(&loser_log, TIE_LOSER, CONTESTED_TERM, inherited), 30);
    assert_eq!(node.mirror.entries(), loser_log.entries());
    // ... then the earlier-joined rival of the same term wins the tie.
    deliver(&mut node, TIE_WINNER, announcement(TIE_WINNER, CONTESTED_TERM, winner_log.head_sequence()), 40);
    assert_eq!(node.link.host_node_id, TIE_WINNER);
    assert_eq!(node.mirror.head_sequence(), inherited, "contested-term entries must be dropped");
    // A late loser append must not slip back in under the shared term.
    deliver(&mut node, TIE_LOSER, append_since(&loser_log, TIE_LOSER, CONTESTED_TERM, inherited), 50);
    assert_eq!(node.mirror.head_sequence(), inherited);
    deliver(&mut node, TIE_WINNER, append_since(&winner_log, TIE_WINNER, CONTESTED_TERM, inherited), 60);
    assert_eq!(node.mirror.entries(), winner_log.entries());
}
