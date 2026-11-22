//! `RelayClientSession` end to end through an in-memory relay that applies
//! the Durable Object's routing and cap policy (no network).

mod common;

use common::*;
use minimappr_core::protocol::{FieldMessage, NodeId};
use minimappr_core::session_crypto::{SessionKey, DEFAULT_MAXIMUM_FAILED_CONFIRMATIONS};
use minimappr_relay_wire::{
    encode_relay_frame, FrameRejectionReason, RelayFrameHeader, RelayFrameKind, RelayRole,
    SessionRelayLimits, BROADCAST_PEER_ID,
};
use minimappr_transport::join_code_exchange_over_relay::{
    random_provisional_joiner_peer_id, HostRelayJoinCodeResponder, JoinerExchangeStep,
    JoinerRelayKeyExchange, RelayJoinExchangeError,
};
use minimappr_transport::{FieldJoinCode, RelayClientEvent, RelayClientSession, RelayTransportConfig};

const HOST_PEER: u32 = 1;
const NODE_PEER: u32 = 2;

fn keyed_session(peer_id: u32, key: &SessionKey) -> RelayClientSession {
    RelayClientSession::new(RelayTransportConfig::default(), peer_id, Some((key, &TEST_SESSION_ID)), 1)
}

fn received_messages(session: &mut RelayClientSession) -> Vec<(NodeId, FieldMessage)> {
    drain_events(session)
        .into_iter()
        .filter_map(|event| match event {
            RelayClientEvent::MessageReceived { sender, message } => Some((sender, message)),
            _ => None,
        })
        .collect()
}

fn host_and_node(limits: SessionRelayLimits) -> (LoopbackRelay, SessionKey) {
    let session_key = SessionKey::generate(&mut seeded_random_source(1));
    let mut relay = LoopbackRelay::new(limits);
    relay.attach(RelayRole::Host, keyed_session(HOST_PEER, &session_key));
    relay.attach(RelayRole::Node, keyed_session(NODE_PEER, &session_key));
    drain_events(relay.session(HOST_PEER));
    drain_events(relay.session(NODE_PEER));
    (relay, session_key)
}

/// Runs the join-code exchange through the relay; returns the joiner's outcome.
fn join_by_code(
    relay: &mut LoopbackRelay,
    host_responder: &mut HostRelayJoinCodeResponder,
    session_key: &SessionKey,
    joiner_peer: u32,
    typed_code: &str,
) -> Result<SessionKey, RelayJoinExchangeError> {
    let mut random_source = seeded_random_source(joiner_peer as u64);
    let code = FieldJoinCode::parse(typed_code).unwrap();
    let (mut joiner, first_flight) =
        JoinerRelayKeyExchange::start(code.secret_password_bytes(), TEST_SESSION_ID, &mut random_source);
    relay.session(joiner_peer).send_key_exchange_flight(HOST_PEER, first_flight).unwrap();
    for _round in 0..4 {
        relay.run_until_quiet();
        for event in drain_events(relay.session(HOST_PEER)) {
            if let RelayClientEvent::KeyExchangeFlightReceived { sender_peer_id, exchange } = event {
                let reply = host_responder.on_joiner_flight(sender_peer_id, &exchange, session_key, &mut random_source);
                relay.session(HOST_PEER).send_key_exchange_flight(sender_peer_id, reply.reply_flight).unwrap();
            }
        }
        relay.run_until_quiet();
        for event in drain_events(relay.session(joiner_peer)) {
            if let RelayClientEvent::KeyExchangeFlightReceived { exchange, .. } = event {
                match joiner.on_host_flight(&exchange)? {
                    JoinerExchangeStep::SendToHost(flight) => {
                        relay.session(joiner_peer).send_key_exchange_flight(HOST_PEER, flight).unwrap()
                    }
                    JoinerExchangeStep::SessionKeyObtained(key) => return Ok(key),
                }
            }
        }
    }
    panic!("exchange did not finish");
}

#[test]
fn sealed_unicast_and_broadcast_reach_only_intended_peers() {
    let (mut relay, _key) = host_and_node(GENEROUS_LIMITS);
    relay.session(NODE_PEER).send_message(NodeId(HOST_PEER), &readiness_query(1)).unwrap();
    relay.session(HOST_PEER).send_message(NodeId(BROADCAST_PEER_ID), &readiness_query(2)).unwrap();
    relay.run_until_quiet();
    assert_eq!(received_messages(relay.session(HOST_PEER)), vec![(NodeId(NODE_PEER), readiness_query(1))]);
    assert_eq!(received_messages(relay.session(NODE_PEER)), vec![(NodeId(HOST_PEER), readiness_query(2))]);
}

#[test]
fn code_joiner_gets_the_session_key_over_the_relay_then_talks_sealed() {
    let (mut relay, session_key) = host_and_node(GENEROUS_LIMITS);
    let host_code = FieldJoinCode::with_generated_secret("K7QX".into(), &mut seeded_random_source(3));
    let mut responder = HostRelayJoinCodeResponder::new(
        host_code.secret_password_bytes(),
        TEST_SESSION_ID,
        DEFAULT_MAXIMUM_FAILED_CONFIRMATIONS,
        RelayTransportConfig::default().maximum_pending_join_code_exchanges,
    );
    let joiner_peer = random_provisional_joiner_peer_id(&mut seeded_random_source(4));
    relay.attach(
        RelayRole::Node,
        RelayClientSession::new(RelayTransportConfig::default(), joiner_peer, None, 1),
    );
    // Before the key, sealed sends are refused locally.
    assert!(relay.session(joiner_peer).send_message(NodeId(HOST_PEER), &readiness_query(0)).is_err());
    let obtained = join_by_code(&mut relay, &mut responder, &session_key, joiner_peer, &host_code.display_text())
        .expect("right code joins");
    assert_eq!(obtained, session_key);
    relay.session(joiner_peer).install_session_key(&obtained, &TEST_SESSION_ID);
    relay.session(joiner_peer).send_message(NodeId(HOST_PEER), &readiness_query(9)).unwrap();
    relay.run_until_quiet();
    assert_eq!(received_messages(relay.session(HOST_PEER)), vec![(NodeId(joiner_peer), readiness_query(9))]);
}

#[test]
fn wrong_codes_are_rejected_and_the_code_retires_after_its_guess_budget() {
    let (mut relay, session_key) = host_and_node(GENEROUS_LIMITS);
    let mut responder = HostRelayJoinCodeResponder::new(b"731904", TEST_SESSION_ID, 2, 4);
    for attempt in 0..3_u32 {
        let joiner_peer = 0x8000_0100 + attempt;
        relay.attach(RelayRole::Viewer, RelayClientSession::new(RelayTransportConfig::default(), joiner_peer, None, 1));
        let outcome = join_by_code(&mut relay, &mut responder, &session_key, joiner_peer, "K7QX-000000");
        assert_eq!(outcome, Err(RelayJoinExchangeError::RejectedByHost));
    }
    assert!(responder.is_exhausted());
    // Even the right code is refused once retired.
    relay.attach(RelayRole::Node, RelayClientSession::new(RelayTransportConfig::default(), 0x8000_0200, None, 1));
    let outcome = join_by_code(&mut relay, &mut responder, &session_key, 0x8000_0200, "K7QX-731904");
    assert_eq!(outcome, Err(RelayJoinExchangeError::RejectedByHost));
}

#[test]
fn clip_arrives_whole_while_control_messages_interleave() {
    let (mut relay, _key) = host_and_node(GENEROUS_LIMITS);
    let clip = clip_payload(77, NodeId(NODE_PEER), 30_000);
    relay.session(NODE_PEER).send_clip_payload(NodeId(HOST_PEER), &clip, Some(5_000), 0).unwrap();
    relay.session(NODE_PEER).send_message(NodeId(HOST_PEER), &readiness_query(5)).unwrap();
    relay.run_until_quiet();
    let received = received_messages(relay.session(HOST_PEER));
    assert_eq!(received.len(), 2);
    assert_eq!(received[0].1, readiness_query(5), "control frame queued later still arrives first");
    assert_eq!(received[1], (NodeId(NODE_PEER), clip));
    assert!(relay.session(NODE_PEER).outbound_statistics().clip_chunks_sent >= 8);
}

#[test]
fn reconnect_resumes_and_resends_an_unexpired_clip_exactly_once() {
    let (mut relay, _key) = host_and_node(GENEROUS_LIMITS);
    let clip = clip_payload(12, NodeId(NODE_PEER), 12_000);
    relay.session(NODE_PEER).send_clip_payload(NodeId(HOST_PEER), &clip, Some(5_000), 0).unwrap();
    relay.run_until_quiet();
    assert_eq!(received_messages(relay.session(HOST_PEER)).len(), 1);
    relay.now_milliseconds = 100;
    relay.disconnect(NODE_PEER);
    assert!(!relay.session(NODE_PEER).reconnect_due(100));
    let parameters = relay.session(NODE_PEER).connection_parameters();
    assert_eq!(parameters.resume_token.as_deref(), Some("resume-2"));
    let reconnect_time = 100 + RelayTransportConfig::default().reconnect_initial_backoff_milliseconds;
    relay.now_milliseconds = reconnect_time;
    assert!(relay.session(NODE_PEER).reconnect_due(reconnect_time));
    relay.connect(NODE_PEER);
    let reconnect_events = drain_events(relay.session(NODE_PEER));
    assert!(reconnect_events
        .iter()
        .any(|event| matches!(event, RelayClientEvent::Connected { resumed: true, .. })));
    relay.run_until_quiet();
    // The clip was resent (rewind) but the receiver delivers it only once.
    assert!(received_messages(relay.session(HOST_PEER)).is_empty());
    assert_eq!(relay.session(NODE_PEER).outbound_statistics().clips_fully_sent, 2);
}

#[test]
fn relay_caps_surface_as_rejection_events() {
    let limits = SessionRelayLimits { maximum_inbound_messages_per_session: 3, ..GENEROUS_LIMITS };
    let (mut relay, _key) = host_and_node(limits);
    for query_id in 0..5 {
        relay.session(NODE_PEER).send_message(NodeId(HOST_PEER), &readiness_query(query_id)).unwrap();
    }
    relay.run_until_quiet();
    assert_eq!(received_messages(relay.session(HOST_PEER)).len(), 3);
    let rejections: Vec<_> = drain_events(relay.session(NODE_PEER))
        .into_iter()
        .filter(|event| {
            matches!(event, RelayClientEvent::FrameRejectedByRelay {
                reason: FrameRejectionReason::SessionMessageCapReached, ..
            })
        })
        .collect();
    assert_eq!(rejections.len(), 2);
}

#[test]
fn viewer_cannot_send_sealed_frames() {
    let (mut relay, key) = host_and_node(GENEROUS_LIMITS);
    relay.attach(RelayRole::Viewer, keyed_session(9, &key));
    drain_events(relay.session(9));
    relay.session(9).send_message(NodeId(HOST_PEER), &readiness_query(1)).unwrap();
    relay.run_until_quiet();
    assert!(received_messages(relay.session(HOST_PEER)).is_empty());
    assert!(drain_events(relay.session(9)).iter().any(|event| matches!(
        event,
        RelayClientEvent::FrameRejectedByRelay { reason: FrameRejectionReason::RoleMayNotSendSealedFrames, .. }
    )));
}

#[test]
fn tampered_replayed_and_misaddressed_frames_are_dropped() {
    let session_key = SessionKey::generate(&mut seeded_random_source(1));
    let mut sender = keyed_session(NODE_PEER, &session_key);
    let mut receiver = keyed_session(HOST_PEER, &session_key);
    sender.on_socket_opened();
    sender.on_text_frame(
        &minimappr_relay_wire::RelayControlNotice::Welcome {
            peer_id: NODE_PEER,
            role: RelayRole::Node,
            resume_token: "r".into(),
            host_peer_id: Some(HOST_PEER),
            usage: Default::default(),
            limits: GENEROUS_LIMITS,
        }
        .to_json(),
    );
    let mut random_source = seeded_random_source(5);
    sender.send_message(NodeId(HOST_PEER), &readiness_query(1)).unwrap();
    sender.send_message(NodeId(7), &readiness_query(2)).unwrap();
    let genuine = sender.poll_transmit(0, &mut random_source).unwrap().unwrap();
    let for_someone_else = sender.poll_transmit(0, &mut random_source).unwrap().unwrap();

    let mut tampered = genuine.clone();
    *tampered.last_mut().unwrap() ^= 1;
    receiver.on_binary_frame(&tampered, 0);
    receiver.on_binary_frame(&genuine, 0);
    receiver.on_binary_frame(&genuine, 0);
    receiver.on_binary_frame(&for_someone_else, 0);
    // A relay rewriting the header sender cannot re-attribute the envelope.
    let (header, body) = minimappr_relay_wire::decode_relay_frame(&genuine).unwrap();
    let respoofed = encode_relay_frame(&RelayFrameHeader { sender_peer_id: 3, ..header }, body);
    receiver.on_binary_frame(&respoofed, 0);
    // Unsealed lane carrying a non-key-exchange message is refused.
    let injected = encode_relay_frame(
        &RelayFrameHeader { kind: RelayFrameKind::JoinCodeKeyExchange, sender_peer_id: 3, recipient_peer_id: HOST_PEER },
        &readiness_query(3).encode_frame().unwrap(),
    );
    receiver.on_binary_frame(&injected, 0);

    assert_eq!(received_messages(&mut receiver).len(), 1);
    let statistics = receiver.inbound_statistics();
    assert_eq!(statistics.authentication_failures, 1);
    assert_eq!(statistics.replays_dropped, 1);
    assert_eq!(statistics.misaddressed_envelopes, 1);
    assert_eq!(statistics.sender_mismatches, 1);
    assert_eq!(statistics.undecodable_field_frames, 1);
}
