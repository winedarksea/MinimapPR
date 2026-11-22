//! Two native clients through the real relay Worker under
//! `wrangler dev --local`: join by code, SPAKE2 over the relay, sealed
//! exchange, clip chunks, resume, role checks, caps and join-code TTL.
#![cfg(feature = "native-websocket")]

mod common;
mod wrangler_local_harness;

use std::time::Duration;

use common::{clip_payload, readiness_query, TEST_SESSION_ID};
use minimappr_core::protocol::{FieldMessage, NodeId};
use minimappr_core::session_crypto::{SessionKey, DEFAULT_MAXIMUM_FAILED_CONFIRMATIONS};
use futures_util::{SinkExt, StreamExt};
use minimappr_relay_wire::relay_http_api::{
    connect_path, encode_hex, CreateRelaySessionRequest, CreateRelaySessionResponse, CONNECT_QUERY_PEER_ID,
    CONNECT_QUERY_ROLE_TOKEN,
};
use minimappr_relay_wire::{FrameRejectionReason, RelayRole, BROADCAST_PEER_ID};
use minimappr_transport::join_code_exchange_over_relay::{
    random_provisional_joiner_peer_id, HostRelayJoinCodeResponder, JoinerExchangeStep, JoinerRelayKeyExchange,
};
use minimappr_transport::native_websocket::{NativeRelayDriver, NativeRelayError, RelayHttpClient};
use minimappr_transport::{FieldJoinCode, RelayClientEvent, RelayClientSession, RelayTransportConfig};
use rand_core::OsRng;
use tokio_tungstenite::tungstenite::Message;
use wrangler_local_harness::{start_local_relay_worker, trigger_scheduled_cleanup};

const HOST_PEER: u32 = 1;
const JOIN_CODE_TTL_SECONDS: u64 = 4;
const MESSAGE_CAP: u64 = 120;
const BYTE_CAP: u64 = 256 * 1024;
const FRAME_CAP: u64 = 32 * 1024;
/// Main scenario binds host, joiner and viewer; the cap step fills the rest.
const PEER_BINDING_CAP: u32 = 6;
const EVENT_TIMEOUT: Duration = Duration::from_secs(15);
const PUMP_SLICE: Duration = Duration::from_millis(20);

fn create_request() -> CreateRelaySessionRequest {
    CreateRelaySessionRequest { field_session_id_hex: encode_hex(&TEST_SESSION_ID.0), issue_join_code: true }
}

async fn connected_driver(
    base_url: &str,
    relay_session_name: &str,
    token: &str,
    peer_id: u32,
    session_key: Option<&SessionKey>,
) -> NativeRelayDriver {
    let session = RelayClientSession::new(
        RelayTransportConfig::default(),
        peer_id,
        session_key.map(|key| (key, &TEST_SESSION_ID)),
        1,
    );
    let mut driver = NativeRelayDriver::new(session, base_url, relay_session_name, token).unwrap();
    driver.connect().await.expect("connect");
    wait_for(&mut driver, |event| matches!(event, RelayClientEvent::Connected { .. })).await;
    driver
}

/// Pumps until an event matches; other events are discarded.
async fn wait_for(driver: &mut NativeRelayDriver, wanted: impl Fn(&RelayClientEvent) -> bool) -> RelayClientEvent {
    let deadline = tokio::time::Instant::now() + EVENT_TIMEOUT;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        match driver.next_event(remaining, PUMP_SLICE).await {
            Some(event) if wanted(&event) => return event,
            Some(_) => continue,
            None => panic!("timed out waiting for relay event"),
        }
    }
}

async fn wait_for_message(driver: &mut NativeRelayDriver) -> (NodeId, FieldMessage) {
    match wait_for(driver, |event| matches!(event, RelayClientEvent::MessageReceived { .. })).await {
        RelayClientEvent::MessageReceived { sender, message } => (sender, message),
        _ => unreachable!(),
    }
}

async fn wait_for_rejection(driver: &mut NativeRelayDriver, reason: FrameRejectionReason) {
    wait_for(driver, |event| {
        matches!(event, RelayClientEvent::FrameRejectedByRelay { reason: got, .. } if *got == reason)
    })
    .await;
}

/// Host answers joiner flights until the joiner holds the session key.
async fn run_join_code_exchange(
    host: &mut NativeRelayDriver,
    joiner: &mut NativeRelayDriver,
    host_code: &FieldJoinCode,
    typed_code: &str,
    session_key: &SessionKey,
) -> SessionKey {
    let mut responder = HostRelayJoinCodeResponder::new(
        host_code.secret_password_bytes(),
        TEST_SESSION_ID,
        DEFAULT_MAXIMUM_FAILED_CONFIRMATIONS,
        RelayTransportConfig::default().maximum_pending_join_code_exchanges,
    );
    let joiner_code = FieldJoinCode::parse(typed_code).unwrap();
    let (mut exchange, first_flight) =
        JoinerRelayKeyExchange::start(joiner_code.secret_password_bytes(), TEST_SESSION_ID, &mut OsRng);
    joiner.session.send_key_exchange_flight(HOST_PEER, first_flight).unwrap();
    joiner.flush_outbound().await.unwrap();
    loop {
        let event = wait_for(host, |event| matches!(event, RelayClientEvent::KeyExchangeFlightReceived { .. })).await;
        let RelayClientEvent::KeyExchangeFlightReceived { sender_peer_id, exchange: flight } = event else {
            unreachable!()
        };
        let reply = responder.on_joiner_flight(sender_peer_id, &flight, session_key, &mut OsRng);
        assert!(reply.failure.is_none(), "host refused: {:?}", reply.failure);
        host.session.send_key_exchange_flight(sender_peer_id, reply.reply_flight).unwrap();
        host.flush_outbound().await.unwrap();
        let event = wait_for(joiner, |event| matches!(event, RelayClientEvent::KeyExchangeFlightReceived { .. })).await;
        let RelayClientEvent::KeyExchangeFlightReceived { exchange: flight, .. } = event else { unreachable!() };
        match exchange.on_host_flight(&flight).expect("joiner step") {
            JoinerExchangeStep::SendToHost(next) => {
                joiner.session.send_key_exchange_flight(HOST_PEER, next).unwrap();
                joiner.flush_outbound().await.unwrap();
            }
            JoinerExchangeStep::SessionKeyObtained(key) => return key,
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_clients_join_exchange_and_hit_caps_through_local_relay_worker() {
    let message_cap = MESSAGE_CAP.to_string();
    let byte_cap = BYTE_CAP.to_string();
    let frame_cap = FRAME_CAP.to_string();
    let ttl = JOIN_CODE_TTL_SECONDS.to_string();
    let peer_binding_cap = PEER_BINDING_CAP.to_string();
    let Some(worker) = start_local_relay_worker(&[
        ("JOIN_CODE_TTL_SECONDS", &ttl),
        ("SESSION_MAXIMUM_INBOUND_MESSAGES", &message_cap),
        ("SESSION_MAXIMUM_INBOUND_BYTES", &byte_cap),
        ("MAXIMUM_FRAME_BYTES", &frame_cap),
        ("USAGE_COUNTER_PERSIST_EVERY_MESSAGES", "1"),
        ("SESSION_MAXIMUM_PEER_BINDINGS", &peer_binding_cap),
    ])
    .await
    else {
        return;
    };
    let base_url = worker.http_base_url.clone();
    let api = RelayHttpClient::new(&base_url);

    // --- host registers; the Worker returns only the lookup half ---
    eprintln!("relay integration step: host registers; the Worker returns only the lookup half");
    let created: CreateRelaySessionResponse = api.create_session(&create_request()).await.unwrap();
    let issued = created.join_code.clone().expect("join code issued");
    let host_code = FieldJoinCode::with_generated_secret(issued.lookup_code.clone(), &mut OsRng);
    let session_key = SessionKey::generate(&mut OsRng);
    let mut host =
        connected_driver(&base_url, &created.relay_session_name, &created.host_token, HOST_PEER, Some(&session_key))
            .await;

    // --- joiner resolves the lookup half only, then runs SPAKE2 via the relay ---
    eprintln!("relay integration step: joiner resolves the lookup half only, then runs SPAKE2 via the relay");
    let typed_code = host_code.display_text();
    let parsed = FieldJoinCode::parse(&typed_code).unwrap();
    let resolved = api.resolve_join_code(&parsed.lookup_code, RelayRole::Node).await.unwrap();
    assert_eq!(resolved.relay_session_name, created.relay_session_name);
    assert_eq!(resolved.field_session_id_hex, encode_hex(&TEST_SESSION_ID.0));
    let joiner_peer = random_provisional_joiner_peer_id(&mut OsRng);
    let mut joiner =
        connected_driver(&base_url, &resolved.relay_session_name, &resolved.role_token, joiner_peer, None).await;
    let obtained = run_join_code_exchange(&mut host, &mut joiner, &host_code, &typed_code, &session_key).await;
    assert_eq!(obtained, session_key);
    joiner.session.install_session_key(&obtained, &TEST_SESSION_ID);

    // --- sealed unicast and broadcast ---
    eprintln!("relay integration step: sealed unicast and broadcast");
    joiner.session.send_message(NodeId(HOST_PEER), &readiness_query(1)).unwrap();
    joiner.flush_outbound().await.unwrap();
    assert_eq!(wait_for_message(&mut host).await, (NodeId(joiner_peer), readiness_query(1)));
    host.session.send_message(NodeId(BROADCAST_PEER_ID), &readiness_query(2)).unwrap();
    host.flush_outbound().await.unwrap();
    assert_eq!(wait_for_message(&mut joiner).await, (NodeId(HOST_PEER), readiness_query(2)));

    // --- a chunked clip plus a control message queued after it ---
    eprintln!("relay integration step: a chunked clip plus a control message queued after it");
    let clip = clip_payload(41, NodeId(joiner_peer), 20_000);
    joiner.session.send_clip_payload(NodeId(HOST_PEER), &clip, Some(10_000), joiner.now_milliseconds()).unwrap();
    joiner.session.send_message(NodeId(HOST_PEER), &readiness_query(3)).unwrap();
    joiner.flush_outbound().await.unwrap();
    assert_eq!(wait_for_message(&mut host).await.1, readiness_query(3));
    assert_eq!(wait_for_message(&mut host).await, (NodeId(joiner_peer), clip));

    // --- resume: a thief with a valid token cannot take the peer id ---
    eprintln!("relay integration step: resume: a thief with a valid token cannot take the peer id");
    joiner.drop_socket_abruptly();
    let mut thief = NativeRelayDriver::new(
        RelayClientSession::new(RelayTransportConfig::default(), joiner_peer, None, 1),
        &base_url,
        &created.relay_session_name,
        &created.node_token,
    )
    .unwrap();
    assert!(matches!(thief.connect().await, Err(NativeRelayError::Status { status_code: 409, .. })));
    tokio::time::sleep(Duration::from_millis(
        RelayTransportConfig::default().reconnect_initial_backoff_milliseconds,
    ))
    .await;
    let resumed = wait_for(&mut joiner, |event| matches!(event, RelayClientEvent::Connected { .. })).await;
    assert!(matches!(resumed, RelayClientEvent::Connected { resumed: true, .. }));
    joiner.session.send_message(NodeId(HOST_PEER), &readiness_query(4)).unwrap();
    joiner.flush_outbound().await.unwrap();
    assert_eq!(wait_for_message(&mut host).await.1, readiness_query(4));

    // --- role and token checks ---
    eprintln!("relay integration step: role and token checks");
    let host_by_code = api.resolve_join_code(&parsed.lookup_code, RelayRole::Host).await;
    assert!(matches!(host_by_code, Err(NativeRelayError::Status { status_code: 403, .. })));
    let mut forger = NativeRelayDriver::new(
        RelayClientSession::new(RelayTransportConfig::default(), 77, None, 1),
        &base_url,
        &created.relay_session_name,
        "not-a-token",
    )
    .unwrap();
    assert!(matches!(forger.connect().await, Err(NativeRelayError::Status { status_code: 403, .. })));
    let viewer =
        connected_driver(&base_url, &created.relay_session_name, &created.viewer_token, 90, Some(&session_key)).await;
    let mut viewer = viewer;
    viewer.session.send_message(NodeId(HOST_PEER), &readiness_query(5)).unwrap();
    viewer.flush_outbound().await.unwrap();
    wait_for_rejection(&mut viewer, FrameRejectionReason::RoleMayNotSendSealedFrames).await;

    // --- join-code TTL: expired codes 410 until cron removes them (404) ---
    eprintln!("relay integration step: join-code TTL: expired codes 410 until cron removes them (404)");
    let wait_until = issued.expires_at_unix_ms + 500;
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64;
    tokio::time::sleep(Duration::from_millis(wait_until.saturating_sub(now))).await;
    let expired = api.resolve_join_code(&parsed.lookup_code, RelayRole::Node).await;
    assert!(matches!(expired, Err(NativeRelayError::Status { status_code: 410, .. })), "{expired:?}");
    trigger_scheduled_cleanup(&worker).await;
    let cleaned = api.resolve_join_code(&parsed.lookup_code, RelayRole::Node).await;
    assert!(matches!(cleaned, Err(NativeRelayError::Status { status_code: 404, .. })), "{cleaned:?}");
    let reissued = api.issue_join_code(&created.relay_session_name, &created.host_token).await.unwrap();
    assert_ne!(reissued.lookup_code, issued.lookup_code);
    api.revoke_join_code(&reissued.lookup_code, &created.host_token).await.unwrap();
    let revoked = api.resolve_join_code(&reissued.lookup_code, RelayRole::Node).await;
    assert!(matches!(revoked, Err(NativeRelayError::Status { status_code: 404, .. })));

    // --- caps: peer bindings outlive sockets, so fresh peer ids run out ---
    eprintln!("relay integration step: caps: distinct peer bindings per session");
    let churned = api.create_session(&create_request()).await.unwrap();
    let mut churn_host =
        connected_driver(&base_url, &churned.relay_session_name, &churned.host_token, HOST_PEER, None).await;
    for churn_peer in 200..(200 + PEER_BINDING_CAP - 1) {
        let mut churner = connected_driver(&base_url, &churned.relay_session_name, &churned.node_token, churn_peer, None).await;
        churner.close_gracefully().await;
    }
    let mut over_cap = NativeRelayDriver::new(
        RelayClientSession::new(RelayTransportConfig::default(), 300, None, 1),
        &base_url,
        &churned.relay_session_name,
        &churned.node_token,
    )
    .unwrap();
    // tungstenite drops the refusal body; the 429 can only be the binding
    // cap here (one live socket, a handful of messages spent).
    let refused = over_cap.connect().await;
    assert!(matches!(&refused, Err(NativeRelayError::Status { status_code: 429, .. })), "{refused:?}");
    churn_host.close_gracefully().await;

    // --- caps: message cap on a fresh session ---
    eprintln!("relay integration step: caps: message cap on a fresh session");
    let flooded = api.create_session(&create_request()).await.unwrap();
    let mut flooder =
        connected_driver(&base_url, &flooded.relay_session_name, &flooded.host_token, HOST_PEER, Some(&session_key))
            .await;
    for query_id in 0..(MESSAGE_CAP as u32 + 3) {
        flooder.session.send_message(NodeId(BROADCAST_PEER_ID), &readiness_query(query_id)).unwrap();
    }
    flooder.flush_outbound().await.unwrap();
    wait_for_rejection(&mut flooder, FrameRejectionReason::SessionMessageCapReached).await;

    // --- caps: text frames are rejected but still spend the message cap ---
    eprintln!("relay integration step: caps: text frames spend the message cap");
    let texted = api.create_session(&create_request()).await.unwrap();
    let raw_url = format!(
        "{}{}?{CONNECT_QUERY_PEER_ID}={HOST_PEER}&{CONNECT_QUERY_ROLE_TOKEN}={}",
        base_url.replacen("http://", "ws://", 1),
        connect_path(&texted.relay_session_name),
        texted.host_token
    );
    let (mut raw_socket, _) = tokio_tungstenite::connect_async(raw_url.as_str()).await.expect("raw connect");
    for _ in 0..(MESSAGE_CAP + 1) {
        raw_socket.send(Message::Text("not a relay frame".into())).await.unwrap();
    }
    let cap_notice = tokio::time::timeout(EVENT_TIMEOUT, async {
        while let Some(Ok(message)) = raw_socket.next().await {
            if let Message::Text(text) = message {
                if text.contains("session_message_cap_reached") {
                    return true;
                }
            }
        }
        false
    })
    .await;
    assert_eq!(cap_notice, Ok(true), "text frames must count against the message cap");
    let _ = raw_socket.close(None).await;

    // --- caps: frame size and byte cap on another fresh session ---
    eprintln!("relay integration step: caps: frame size and byte cap on another fresh session");
    let heavy = api.create_session(&create_request()).await.unwrap();
    let mut heavy_sender =
        connected_driver(&base_url, &heavy.relay_session_name, &heavy.host_token, HOST_PEER, Some(&session_key)).await;
    let oversized = clip_payload(1, NodeId(HOST_PEER), FRAME_CAP as usize + 1_000);
    heavy_sender.session.send_message(NodeId(BROADCAST_PEER_ID), &oversized).unwrap();
    heavy_sender.flush_outbound().await.unwrap();
    wait_for_rejection(&mut heavy_sender, FrameRejectionReason::FrameLargerThanLimit).await;
    let near_frame_cap = clip_payload(2, NodeId(HOST_PEER), FRAME_CAP as usize - 2_000);
    for _ in 0..(BYTE_CAP / (FRAME_CAP - 2_000) + 1) {
        heavy_sender.session.send_message(NodeId(BROADCAST_PEER_ID), &near_frame_cap).unwrap();
    }
    heavy_sender.flush_outbound().await.unwrap();
    wait_for_rejection(&mut heavy_sender, FrameRejectionReason::SessionByteCapReached).await;

    host.close_gracefully().await;
    joiner.close_gracefully().await;
}
