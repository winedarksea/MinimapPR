//! Sealing (XChaCha20-Poly1305) and join-code SPAKE2 handshake tests.
#![cfg(feature = "session-crypto")]

use minimappr_core::protocol::detection_messages::ReadinessQuery;
use minimappr_core::protocol::wire_codes::FrequencyBandHz;
use minimappr_core::protocol::{
    decode_field_message_frame, DecodedFrame, FieldMessage, NodeId, SessionId,
};
use minimappr_core::session_crypto::{
    read_envelope_routing, EnvelopeRouting, HostJoinCodeResponder, JoinCodeExchangeError,
    JoinerAwaitingHostMessage, MessageSealer, ReplayWindow, SealingError, SessionKey,
    DEFAULT_MAXIMUM_FAILED_CONFIRMATIONS,
};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;

const SESSION: SessionId = SessionId([0x5A; 16]);
const JOIN_CODE_SECRET: &[u8] = b"731904";

fn routing(sequence: u64) -> EnvelopeRouting {
    EnvelopeRouting {
        sender_node_id: NodeId(3),
        recipient_node_id: NodeId(1),
        sender_sequence: sequence,
    }
}

fn sample_frame() -> Vec<u8> {
    FieldMessage::ReadinessQuery(ReadinessQuery {
        query_id: 1,
        target_node_id: NodeId(3),
        centre_device_sample_index: 123_456_789,
        half_window_samples: 36_000,
        band: FrequencyBandHz {
            low_hz: 2_000.0,
            high_hz: 8_000.0,
        },
    })
    .encode_frame()
    .unwrap()
}

#[test]
fn seal_open_round_trip_and_relay_sees_only_routing() {
    let mut random_source = ChaCha20Rng::seed_from_u64(1);
    let session_key = SessionKey::generate(&mut random_source);
    let sealer = MessageSealer::new(&session_key, &SESSION);
    let frame = sample_frame();
    let envelope = sealer
        .seal(routing(42), &frame, &mut random_source)
        .unwrap();
    assert_eq!(read_envelope_routing(&envelope).unwrap(), routing(42));
    assert!(
        !envelope
            .windows(frame.len())
            .any(|window| window == frame.as_slice()),
        "plaintext leaked"
    );
    let (opened_routing, plaintext) = sealer.open(&envelope).unwrap();
    assert_eq!(opened_routing, routing(42));
    assert!(matches!(
        decode_field_message_frame(&plaintext).unwrap(),
        DecodedFrame::Known { .. }
    ));
    // Fresh random nonce per message: same plaintext, different ciphertext.
    let second = sealer
        .seal(routing(42), &frame, &mut random_source)
        .unwrap();
    assert_ne!(envelope, second);
}

#[test]
fn any_tampered_byte_fails_authentication() {
    let mut random_source = ChaCha20Rng::seed_from_u64(2);
    let sealer = MessageSealer::new(&SessionKey::generate(&mut random_source), &SESSION);
    let envelope = sealer
        .seal(routing(7), &sample_frame(), &mut random_source)
        .unwrap();
    for byte_index in 1..envelope.len() {
        let mut tampered = envelope.clone();
        tampered[byte_index] ^= 0x01;
        assert_eq!(
            sealer.open(&tampered),
            Err(SealingError::AuthenticationFailed),
            "byte {byte_index}"
        );
    }
    let mut wrong_version = envelope.clone();
    wrong_version[0] = 9;
    assert_eq!(
        sealer.open(&wrong_version),
        Err(SealingError::UnsupportedEnvelopeVersion(9))
    );
    assert_eq!(
        sealer.open(&envelope[..30]),
        Err(SealingError::EnvelopeTooShort)
    );
}

#[test]
fn wrong_key_or_wrong_session_cannot_open() {
    let mut random_source = ChaCha20Rng::seed_from_u64(3);
    let session_key = SessionKey::generate(&mut random_source);
    let envelope = MessageSealer::new(&session_key, &SESSION)
        .seal(routing(1), b"hello", &mut random_source)
        .unwrap();
    let other_key = SessionKey::generate(&mut random_source);
    assert_eq!(
        MessageSealer::new(&other_key, &SESSION).open(&envelope),
        Err(SealingError::AuthenticationFailed)
    );
    assert_eq!(
        MessageSealer::new(&session_key, &SessionId([0; 16])).open(&envelope),
        Err(SealingError::AuthenticationFailed)
    );
}

#[test]
fn relay_misdelivery_and_reflection_are_rejected_by_the_receiver() {
    let mut random_source = ChaCha20Rng::seed_from_u64(9);
    let sealer = MessageSealer::new(&SessionKey::generate(&mut random_source), &SESSION);
    let to_node_one = sealer
        .seal(routing(5), b"for node 1", &mut random_source)
        .unwrap();
    assert!(sealer
        .open_addressed_to_local_node(&to_node_one, NodeId(1))
        .is_ok());
    // The relay hands node 1's message to node 2: authentic, but not for it.
    assert_eq!(
        sealer.open_addressed_to_local_node(&to_node_one, NodeId(2)),
        Err(SealingError::NotAddressedToLocalNode)
    );
    // The relay reflects node 3's own message back to node 3 as a broadcast.
    let broadcast_from_three = sealer
        .seal(
            EnvelopeRouting {
                recipient_node_id: NodeId::BROADCAST,
                ..routing(6)
            },
            b"broadcast",
            &mut random_source,
        )
        .unwrap();
    assert!(sealer
        .open_addressed_to_local_node(&broadcast_from_three, NodeId(2))
        .is_ok());
    assert_eq!(
        sealer.open_addressed_to_local_node(&broadcast_from_three, NodeId(3)),
        Err(SealingError::ReflectedOwnMessage)
    );
}

#[test]
fn joiner_rejects_its_own_spake_message_reflected_as_the_host_flight() {
    let mut random_source = ChaCha20Rng::seed_from_u64(10);
    let (joiner, flight_one) =
        JoinerAwaitingHostMessage::start(JOIN_CODE_SECRET, SESSION, &mut random_source);
    assert!(matches!(
        joiner.on_host_spake_message(&flight_one),
        Err(JoinCodeExchangeError::MalformedSpakeMessage)
    ));
}

#[test]
fn replay_window_rejects_duplicates_and_stale_but_allows_reordering() {
    let mut window = ReplayWindow::default();
    let sender = NodeId(3);
    assert!(window.accept(sender, 10));
    assert!(!window.accept(sender, 10), "duplicate");
    assert!(window.accept(sender, 12));
    assert!(window.accept(sender, 11), "reordered within window");
    assert!(!window.accept(sender, 11));
    assert!(window.accept(sender, 200));
    assert!(
        !window.accept(sender, 12),
        "older than the 64-message window"
    );
    assert!(window.accept(NodeId(4), 12), "windows are per sender");
}

#[test]
fn join_code_handshake_delivers_the_session_key() {
    let mut random_source = ChaCha20Rng::seed_from_u64(4);
    let session_key = SessionKey::generate(&mut random_source);
    let mut host = HostJoinCodeResponder::new(
        JOIN_CODE_SECRET,
        SESSION,
        DEFAULT_MAXIMUM_FAILED_CONFIRMATIONS,
    );
    let (joiner, flight_one) =
        JoinerAwaitingHostMessage::start(JOIN_CODE_SECRET, SESSION, &mut random_source);
    let (pending, flight_two) = host
        .respond_to_joiner(&flight_one, &mut random_source)
        .unwrap();
    let (joiner, flight_three) = joiner.on_host_spake_message(&flight_two).unwrap();
    let flight_four = host
        .on_joiner_confirmation(pending, &flight_three, &session_key, &mut random_source)
        .unwrap();
    let received_key = joiner.on_sealed_session_key(&flight_four).unwrap();
    assert_eq!(received_key, session_key);
    assert_eq!(host.failed_confirmations(), 0);

    // The joiner can now read host traffic.
    let envelope = MessageSealer::new(&session_key, &SESSION)
        .seal(routing(1), b"welcome", &mut random_source)
        .unwrap();
    assert_eq!(
        MessageSealer::new(&received_key, &SESSION)
            .open(&envelope)
            .unwrap()
            .1,
        b"welcome"
    );
    // The short code itself is never the key.
    assert_ne!(&session_key.0[..JOIN_CODE_SECRET.len()], JOIN_CODE_SECRET);
}

#[test]
fn wrong_join_code_is_counted_and_the_code_retires_after_its_budget() {
    let mut random_source = ChaCha20Rng::seed_from_u64(5);
    let session_key = SessionKey::generate(&mut random_source);
    let mut host = HostJoinCodeResponder::new(
        JOIN_CODE_SECRET,
        SESSION,
        DEFAULT_MAXIMUM_FAILED_CONFIRMATIONS,
    );
    for attempt in 0..DEFAULT_MAXIMUM_FAILED_CONFIRMATIONS {
        let guess = format!("{:06}", 100_000 + attempt);
        let (joiner, flight_one) =
            JoinerAwaitingHostMessage::start(guess.as_bytes(), SESSION, &mut random_source);
        let (pending, flight_two) = host
            .respond_to_joiner(&flight_one, &mut random_source)
            .unwrap();
        let (_joiner, flight_three) = joiner.on_host_spake_message(&flight_two).unwrap();
        assert_eq!(
            host.on_joiner_confirmation(pending, &flight_three, &session_key, &mut random_source),
            Err(JoinCodeExchangeError::ConfirmationFailed)
        );
    }
    assert!(host.is_exhausted());
    let (_joiner, flight_one) =
        JoinerAwaitingHostMessage::start(JOIN_CODE_SECRET, SESSION, &mut random_source);
    assert!(matches!(
        host.respond_to_joiner(&flight_one, &mut random_source),
        Err(JoinCodeExchangeError::JoinCodeAttemptsExhausted)
    ));
}

#[test]
fn impostor_host_without_the_code_cannot_deliver_a_key() {
    let mut random_source = ChaCha20Rng::seed_from_u64(6);
    // A relay posing as host guesses the code.
    let mut impostor =
        HostJoinCodeResponder::new(b"000000", SESSION, DEFAULT_MAXIMUM_FAILED_CONFIRMATIONS);
    let (joiner, flight_one) =
        JoinerAwaitingHostMessage::start(JOIN_CODE_SECRET, SESSION, &mut random_source);
    let (pending, flight_two) = impostor
        .respond_to_joiner(&flight_one, &mut random_source)
        .unwrap();
    let (joiner, flight_three) = joiner.on_host_spake_message(&flight_two).unwrap();
    // The impostor's verification of the real joiner fails, so it learns the
    // guess was wrong but never gets a confirmed key...
    let attacker_key = SessionKey([0xEE; 32]);
    assert!(impostor
        .on_joiner_confirmation(pending, &flight_three, &attacker_key, &mut random_source)
        .is_err());
    // ...and a forged flight four does not open on the joiner.
    let forged = vec![0x11; 24 + 32 + 16];
    assert_eq!(
        joiner.on_sealed_session_key(&forged),
        Err(JoinCodeExchangeError::SealedSessionKeyInvalid)
    );
}

#[test]
fn transcripts_from_another_session_are_useless() {
    let mut random_source = ChaCha20Rng::seed_from_u64(7);
    let session_key = SessionKey::generate(&mut random_source);
    let mut host = HostJoinCodeResponder::new(
        JOIN_CODE_SECRET,
        SESSION,
        DEFAULT_MAXIMUM_FAILED_CONFIRMATIONS,
    );
    let (joiner, flight_one) =
        JoinerAwaitingHostMessage::start(JOIN_CODE_SECRET, SessionId([1; 16]), &mut random_source);
    let (pending, flight_two) = host
        .respond_to_joiner(&flight_one, &mut random_source)
        .unwrap();
    let (_joiner, flight_three) = joiner.on_host_spake_message(&flight_two).unwrap();
    assert_eq!(
        host.on_joiner_confirmation(pending, &flight_three, &session_key, &mut random_source),
        Err(JoinCodeExchangeError::ConfirmationFailed)
    );
}

#[test]
fn malformed_spake_messages_are_rejected() {
    let mut random_source = ChaCha20Rng::seed_from_u64(8);
    let host = HostJoinCodeResponder::new(
        JOIN_CODE_SECRET,
        SESSION,
        DEFAULT_MAXIMUM_FAILED_CONFIRMATIONS,
    );
    assert!(matches!(
        host.respond_to_joiner(&[0x41; 5], &mut random_source),
        Err(JoinCodeExchangeError::MalformedSpakeMessage)
    ));
    let (joiner, _) =
        JoinerAwaitingHostMessage::start(JOIN_CODE_SECRET, SESSION, &mut random_source);
    assert!(matches!(
        joiner.on_host_spake_message(&[0x42; 3]),
        Err(JoinCodeExchangeError::MalformedSpakeMessage)
    ));
}
