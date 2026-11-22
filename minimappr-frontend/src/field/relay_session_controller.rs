//! Host / join-by-code flows over the relay, driven from the browser.
//!
//! Host: register a relayed session, pair the Worker's lookup code with a
//! locally minted secret (never sent to the relay), answer SPAKE2 flights.
//! Joiner: resolve the lookup half, run SPAKE2 through the relay, install
//! the session key, then exchange sealed `FieldMessage`s.
//! All of it is `minimappr-transport` + `minimappr-core` compiled to wasm.

use std::cell::RefCell;
use std::rc::Rc;

use gloo_timers::callback::Interval;
use leptos::prelude::{GetUntracked, Set, Update};
use minimappr_core::protocol::{FieldMessage, NodeId, SessionId};
use minimappr_core::session_crypto::{SessionKey, DEFAULT_MAXIMUM_FAILED_CONFIRMATIONS};
use minimappr_relay_wire::relay_http_api::{decode_hex, encode_hex, CreateRelaySessionRequest};
use minimappr_relay_wire::{RelayRole, BROADCAST_PEER_ID};
use minimappr_transport::join_code_exchange_over_relay::{
    random_provisional_joiner_peer_id, HostRelayJoinCodeResponder, JoinerExchangeStep, JoinerRelayKeyExchange,
};
use minimappr_transport::web_websocket::{BrowserCryptoRandomSource, RelayFetchClient, WebRelayDriver, WebRelayDriverOptions};
use minimappr_transport::{FieldJoinCode, RelayClientEvent, RelayClientSession, RelayTransportConfig};

use super::field_configuration::{HOST_RELAY_PEER_ID, RELAY_HTTP_BASE_URL, RELAY_TICK_INTERVAL_MILLISECONDS};
use super::field_signals::FieldSignals;
use super::relay_message_presenter::describe_field_message;
use crate::mock::field_shim;
use crate::state::AppState;

const SESSION_ID_LENGTH_BYTES: usize = 16;

struct ActiveRelay {
    driver: WebRelayDriver,
    _tick_timer: Interval,
    is_host: bool,
}

thread_local! {
    static ACTIVE_RELAY: RefCell<Option<ActiveRelay>> = const { RefCell::new(None) };
}

fn random_source() -> Result<BrowserCryptoRandomSource, String> {
    BrowserCryptoRandomSource::from_global().ok_or_else(|| "WebCrypto unavailable".to_string())
}

/// Sender sequences must exceed anything this peer id sent before in the
/// session (receivers' replay windows); wall-clock milliseconds survive a
/// page reload, the case relay.md flags as open for shells.
fn initial_sender_sequence() -> u64 {
    js_sys::Date::now() as u64
}

fn install_active_relay(driver: WebRelayDriver, is_host: bool) {
    let ticking_driver = driver.clone();
    let tick_timer = Interval::new(RELAY_TICK_INTERVAL_MILLISECONDS, move || ticking_driver.tick());
    ACTIVE_RELAY.with(|slot| {
        if let Some(previous) = slot.borrow_mut().take() {
            previous.driver.shut_down();
        }
        *slot.borrow_mut() = Some(ActiveRelay {
            driver,
            _tick_timer: tick_timer,
            is_host,
        });
    });
}

/// Events every role handles the same way.
fn handle_common_event(event: &RelayClientEvent, signals: FieldSignals, app_state: &AppState) {
    match event {
        RelayClientEvent::Connected { peer_id, role, host_peer_id, resumed, .. } => {
            signals.relay_status.set(format!(
                "Connected as peer {peer_id} ({role:?}){}; host {}",
                if *resumed { ", resumed" } else { "" },
                host_peer_id.map_or("absent".to_string(), |id| id.to_string())
            ));
            field_shim::publish_relay_connection(app_state, true, false);
        }
        RelayClientEvent::Disconnected { .. } => {
            signals.relay_status.set("Disconnected; reconnecting".to_string());
            field_shim::publish_relay_connection(app_state, false, true);
        }
        RelayClientEvent::PeerJoined { peer_id, role } => signals.log_relay(format!("peer {peer_id} joined ({role:?})")),
        RelayClientEvent::PeerLeft { peer_id } => signals.log_relay(format!("peer {peer_id} left")),
        RelayClientEvent::MessageReceived { sender, message } => {
            signals.relay_sealed_messages_received.update(|count| *count += 1);
            signals.log_relay(format!("sealed from {}: {}", sender.0, describe_field_message(message)));
        }
        RelayClientEvent::UnknownMessageKindReceived { sender, message_kind_id } => {
            signals.log_relay(format!("sealed from {}: unknown kind {message_kind_id}", sender.0));
        }
        RelayClientEvent::FrameRejectedByRelay { reason, .. } => signals.log_relay(format!("relay rejected a frame: {reason:?}")),
        RelayClientEvent::KeyExchangeFlightReceived { .. } => {}
    }
}

pub async fn host_new_session(signals: FieldSignals, app_state: AppState) -> Result<(), String> {
    let mut rng = random_source()?;
    let mut session_id_bytes = [0_u8; SESSION_ID_LENGTH_BYTES];
    rand_core::RngCore::fill_bytes(&mut rng, &mut session_id_bytes);
    let session_id = SessionId(session_id_bytes);
    let session_key = SessionKey::generate(&mut rng);
    signals.relay_status.set("Registering session with relay".to_string());
    let created = RelayFetchClient::new(RELAY_HTTP_BASE_URL)
        .create_session(&CreateRelaySessionRequest {
            field_session_id_hex: encode_hex(&session_id.0),
            issue_join_code: true,
        })
        .await
        .map_err(|error| error.to_string())?;
    let lookup = created.join_code.ok_or("relay issued no join code")?;
    let join_code = FieldJoinCode::with_generated_secret(lookup.lookup_code, &mut rng);
    signals.join_code_display.set(Some(join_code.display_text()));
    signals.relay_role_description.set(Some("Host".to_string()));
    signals.relay_session_key_installed.set(true);

    let config = RelayTransportConfig::default();
    let responder = Rc::new(RefCell::new(HostRelayJoinCodeResponder::new(
        join_code.secret_password_bytes(),
        session_id,
        DEFAULT_MAXIMUM_FAILED_CONFIRMATIONS,
        config.maximum_pending_join_code_exchanges,
    )));
    let session = RelayClientSession::new(
        config,
        HOST_RELAY_PEER_ID,
        Some((&session_key, &session_id)),
        initial_sender_sequence(),
    );
    let driver = WebRelayDriver::new(
        session,
        RELAY_HTTP_BASE_URL,
        &created.relay_session_name,
        &created.host_token,
        WebRelayDriverOptions::default(),
    )
    .map_err(|error| error.to_string())?;
    let sink_driver = driver.clone();
    driver.set_event_sink(move |event| {
        if let RelayClientEvent::KeyExchangeFlightReceived { sender_peer_id, exchange } = &event {
            let reply = responder
                .borrow_mut()
                .on_joiner_flight(*sender_peer_id, exchange, &session_key, &mut rng);
            if let Some(failure) = &reply.failure {
                signals.log_relay(format!("join attempt from {sender_peer_id} refused: {failure:?}"));
            }
            if reply.joiner_received_session_key {
                signals.log_relay(format!("peer {sender_peer_id} joined by code (session key sent sealed)"));
            }
            let flight = reply.reply_flight;
            let _ = sink_driver.with_session(|session, _now| session.send_key_exchange_flight(*sender_peer_id, flight));
        }
        handle_common_event(&event, signals, &app_state);
    });
    driver.connect().map_err(|error| error.to_string())?;
    install_active_relay(driver, true);
    Ok(())
}

pub async fn join_session_by_code(typed_code: String, signals: FieldSignals, app_state: AppState) -> Result<(), String> {
    let join_code = FieldJoinCode::parse(&typed_code).map_err(|error| error.to_string())?;
    let mut rng = random_source()?;
    signals.relay_status.set("Resolving join code".to_string());
    let resolved = RelayFetchClient::new(RELAY_HTTP_BASE_URL)
        .resolve_join_code(&join_code.lookup_code, RelayRole::Node)
        .await
        .map_err(|error| error.to_string())?;
    let session_id_bytes: [u8; SESSION_ID_LENGTH_BYTES] = decode_hex(&resolved.field_session_id_hex)
        .and_then(|bytes| bytes.try_into().ok())
        .ok_or("relay returned a malformed session id")?;
    let session_id = SessionId(session_id_bytes);
    let provisional_peer_id = random_provisional_joiner_peer_id(&mut rng);
    signals.relay_role_description.set(Some(format!("Joiner (provisional peer {provisional_peer_id})")));

    let session = RelayClientSession::new(RelayTransportConfig::default(), provisional_peer_id, None, initial_sender_sequence());
    let driver = WebRelayDriver::new(
        session,
        RELAY_HTTP_BASE_URL,
        &resolved.relay_session_name,
        &resolved.role_token,
        WebRelayDriverOptions::default(),
    )
    .map_err(|error| error.to_string())?;
    let exchange: Rc<RefCell<Option<JoinerRelayKeyExchange>>> = Rc::new(RefCell::new(None));
    let sink_driver = driver.clone();
    driver.set_event_sink(move |event| {
        match &event {
            RelayClientEvent::Connected { host_peer_id: Some(host), .. }
            | RelayClientEvent::PeerJoined { peer_id: host, role: RelayRole::Host } => {
                let already_joined_or_joining =
                    signals.relay_session_key_installed.get_untracked() || exchange.borrow().is_some();
                if !already_joined_or_joining {
                    let (started, first_flight) =
                        JoinerRelayKeyExchange::start(join_code.secret_password_bytes(), session_id, &mut rng);
                    *exchange.borrow_mut() = Some(started);
                    let host = *host;
                    let _ = sink_driver.with_session(|session, _now| session.send_key_exchange_flight(host, first_flight));
                    signals.log_relay("sent SPAKE2 flight 1 to host".to_string());
                }
            }
            RelayClientEvent::KeyExchangeFlightReceived { sender_peer_id, exchange: flight } => {
                let mut exchange_slot = exchange.borrow_mut();
                let Some(in_progress) = exchange_slot.as_mut() else { return };
                match in_progress.on_host_flight(flight) {
                    Ok(JoinerExchangeStep::SendToHost(next)) => {
                        let host = *sender_peer_id;
                        let _ = sink_driver.with_session(|session, _now| session.send_key_exchange_flight(host, next));
                    }
                    Ok(JoinerExchangeStep::SessionKeyObtained(key)) => {
                        *exchange_slot = None;
                        sink_driver.with_session(|session, _now| session.install_session_key(&key, &session_id));
                        signals.relay_session_key_installed.set(true);
                        signals.log_relay("join code accepted; session key installed".to_string());
                    }
                    Err(error) => {
                        *exchange_slot = None;
                        signals.log_relay(format!("join failed: {error:?}"));
                    }
                }
            }
            _ => {}
        }
        handle_common_event(&event, signals, &app_state);
    });
    driver.connect().map_err(|error| error.to_string())?;
    install_active_relay(driver, false);
    Ok(())
}

/// Sends a sealed message: joiners to the host, the host to everyone.
pub fn send_sealed_message(message: &FieldMessage) -> Result<(), String> {
    ACTIVE_RELAY.with(|slot| {
        let slot = slot.borrow();
        let active = slot.as_ref().ok_or("no relay session")?;
        let recipient = if active.is_host { NodeId(BROADCAST_PEER_ID) } else { NodeId(HOST_RELAY_PEER_ID) };
        active
            .driver
            .with_session(|session, _now| session.send_message(recipient, message))
            .map_err(|error| error.to_string())
    })
}

/// This phone's relay peer id, used as the sender `NodeId` in messages.
pub fn local_peer_id() -> Option<u32> {
    ACTIVE_RELAY.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|active| active.driver.with_session(|session, _now| session.local_peer_id()))
    })
}
