//! One Durable Object per relayed session: a hibernating WebSocket relay.
//!
//! It uses the WebSocket Hibernation API (`accept_websocket_with_tags`) and
//! sets no timers except one alarm at session expiry, so an idle session
//! bills no duration. Frames are forwarded byte-for-byte after checking the
//! cleartext relay header against `decide_relay_delivery`; the object holds
//! no key and cannot read sealed bodies. Everything it decides is in
//! `minimappr-relay-wire` and unit-tested natively.

use std::cell::{Cell, RefCell};

use minimappr_relay_wire::relay_http_api::{
    CONNECT_QUERY_PEER_ID, CONNECT_QUERY_RESUME_TOKEN, CONNECT_QUERY_ROLE_TOKEN,
};
use minimappr_relay_wire::{
    decide_relay_delivery, ConnectionRefusalReason, FrameRejectionReason, RelayControlNotice, RelayDelivery,
    RelayFrameHeader, RelayRole, SessionUsageCounters, BROADCAST_PEER_ID,
};
use worker::*;

use crate::random_tokens::{digests_match, mint_secret_token, sha256_hex};
use crate::relay_object_state::*;
use crate::worker_config::RelayWorkerConfig;

#[durable_object]
pub struct SessionRelayObject {
    state: State,
    env: Env,
    /// Lazily loaded from storage after each wake from hibernation.
    cached_usage: RefCell<Option<SessionUsageCounters>>,
    admitted_since_last_persist: Cell<u64>,
}

fn now_unix_milliseconds() -> u64 {
    js_sys::Date::now() as u64
}

fn send_notice(socket: &WebSocket, notice: &RelayControlNotice) {
    // A failed send means that socket is closing; its close event cleans up.
    let _ = socket.send_with_str(notice.to_json());
}

impl DurableObject for SessionRelayObject {
    fn new(state: State, env: Env) -> Self {
        Self {
            state,
            env,
            cached_usage: RefCell::new(None),
            admitted_since_last_persist: Cell::new(0),
        }
    }

    async fn fetch(&self, mut request: Request) -> Result<Response> {
        let path = request.path();
        if path == INTERNAL_INITIALIZE_PATH {
            let body: InitializeRelayObjectRequest = request.json().await?;
            return self.initialize(body).await;
        }
        if path == INTERNAL_MINT_ROLE_TOKEN_PATH {
            let body: MintRoleTokenRequest = request.json().await?;
            return self.mint_role_token(body.role).await;
        }
        if path.ends_with(CONNECT_PATH_SUFFIX) {
            return self.accept_connection(&request).await;
        }
        Response::error("not found", 404)
    }

    async fn websocket_message(&self, socket: WebSocket, message: WebSocketIncomingMessage) -> Result<()> {
        let Some(attachment) = socket.deserialize_attachment::<SocketAttachment>()? else {
            let _ = socket.close(Some(CLOSE_CODE_NORMAL), Some("unbound socket"));
            return Ok(());
        };
        match message {
            WebSocketIncomingMessage::String(text) => {
                // Text frames are billed like binary ones, so they spend the
                // caps too; otherwise they would be an uncapped flood channel.
                if self.admit_inbound_frame(&socket, text.len() as u64).await? {
                    self.reject(&socket, FrameRejectionReason::TextFramesNotAccepted, None);
                }
                Ok(())
            }
            WebSocketIncomingMessage::Binary(frame) => self.relay_frame(&socket, &attachment, &frame).await,
        }
    }

    async fn websocket_close(&self, socket: WebSocket, _code: usize, _reason: String, _was_clean: bool) -> Result<()> {
        self.on_socket_gone(&socket).await
    }

    async fn websocket_error(&self, socket: WebSocket, _error: Error) -> Result<()> {
        self.on_socket_gone(&socket).await
    }

    /// Session expiry: drop every socket and wipe all storage.
    async fn alarm(&self) -> Result<Response> {
        for socket in self.state.get_websockets() {
            let _ = socket.close(Some(CLOSE_CODE_SESSION_EXPIRED), Some("session expired"));
        }
        self.state.storage().delete_all().await?;
        *self.cached_usage.borrow_mut() = None;
        Response::ok("expired")
    }
}

impl SessionRelayObject {
    async fn initialize(&self, request: InitializeRelayObjectRequest) -> Result<Response> {
        let storage = self.state.storage();
        storage
            .put(META_STORAGE_KEY, RelayObjectMeta { expires_at_unix_ms: request.expires_at_unix_ms })
            .await?;
        for (token_digest, role) in &request.role_token_digests {
            storage.put(&token_storage_key(token_digest), role).await?;
        }
        let offset_milliseconds = request.expires_at_unix_ms.saturating_sub(now_unix_milliseconds());
        storage.set_alarm(offset_milliseconds as i64).await?;
        Response::ok("initialized")
    }

    async fn live_meta(&self) -> Result<std::result::Result<RelayObjectMeta, Response>> {
        match self.state.storage().get::<RelayObjectMeta>(META_STORAGE_KEY).await? {
            None => Ok(Err(Response::error("unknown relay session", 404)?)),
            Some(meta) if meta.expires_at_unix_ms <= now_unix_milliseconds() => {
                Ok(Err(Response::error("relay session expired", 410)?))
            }
            Some(meta) => Ok(Ok(meta)),
        }
    }

    async fn mint_role_token(&self, role: RelayRole) -> Result<Response> {
        if let Err(refusal) = self.live_meta().await? {
            return Ok(refusal);
        }
        if role == RelayRole::Host {
            return Response::error("host tokens are only issued at session creation", 403);
        }
        let role_token = mint_secret_token()?;
        self.state
            .storage()
            .put(&token_storage_key(&sha256_hex(&role_token)), role)
            .await?;
        Response::from_json(&MintRoleTokenResponse { role_token })
    }

    fn host_peer_id(&self) -> Option<u32> {
        self.state
            .get_websockets_with_tag(HOST_ROLE_SOCKET_TAG)
            .iter()
            .find_map(|socket| socket.deserialize_attachment::<SocketAttachment>().ok().flatten())
            .map(|attachment| attachment.peer_id)
    }

    async fn accept_connection(&self, request: &Request) -> Result<Response> {
        let upgrade = request.headers().get("Upgrade")?.unwrap_or_default();
        if !upgrade.eq_ignore_ascii_case("websocket") {
            return Response::error("expected a WebSocket upgrade", 426);
        }
        if let Err(refusal) = self.live_meta().await? {
            return Ok(refusal);
        }
        let url = request.url()?;
        let query_value = |name: &str| {
            url.query_pairs()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.into_owned())
        };
        let Some(peer_id) = query_value(CONNECT_QUERY_PEER_ID)
            .and_then(|text| text.parse::<u32>().ok())
            .filter(|peer_id| *peer_id != BROADCAST_PEER_ID)
        else {
            return Response::error("missing or invalid peer id", 400);
        };
        let role_token = query_value(CONNECT_QUERY_ROLE_TOKEN).unwrap_or_default();
        let storage = self.state.storage();
        let Some(role) = storage
            .get::<RelayRole>(&token_storage_key(&sha256_hex(&role_token)))
            .await?
        else {
            return Response::error("invalid role token", 403);
        };
        let config = RelayWorkerConfig::from_env(&self.env)?;
        if role == RelayRole::Host {
            if let Some(existing_host_peer_id) = self.host_peer_id() {
                if existing_host_peer_id != peer_id {
                    return Response::error("another host is connected", 409);
                }
            }
        }
        let presented_resume_token = query_value(CONNECT_QUERY_RESUME_TOKEN);
        let peer_key = peer_storage_key(peer_id);
        let rebinding = match storage.get::<PeerBinding>(&peer_key).await? {
            None => false,
            Some(binding) => {
                let resume_matches = presented_resume_token.as_deref().is_some_and(|token| {
                    digests_match(&sha256_hex(token), &binding.resume_token_sha256_hex)
                });
                if binding.role != role || !resume_matches {
                    return Response::error("peer id already bound; resume token required", 409);
                }
                true
            }
        };
        let peer_tag = peer_socket_tag(peer_id);
        // A resume replaces its own stale socket, so that one does not count.
        let live_socket_count = self
            .state
            .get_websockets()
            .iter()
            .filter(|socket| !(rebinding && self.state.get_tags(socket).contains(&peer_tag)))
            .count();
        if live_socket_count as u32 >= config.limits.maximum_connections_per_session {
            return Response::error("session connection cap reached", 429);
        }
        let mut usage = self.load_usage().await?;
        match usage.try_admit_connection(!rebinding, &config.limits) {
            Ok(()) => {}
            Err(ConnectionRefusalReason::PeerBindingCapReached) => {
                return Response::error("session peer binding cap reached", 429);
            }
            Err(ConnectionRefusalReason::SessionMessageCapReached) => {
                return Response::error("session message cap reached", 429);
            }
        }
        *self.cached_usage.borrow_mut() = Some(usage);
        self.persist_usage().await?;
        // Rotate the resume token on every accepted connect.
        let resume_token = mint_secret_token()?;
        storage
            .put(&peer_key, PeerBinding { role, resume_token_sha256_hex: sha256_hex(&resume_token) })
            .await?;
        if rebinding {
            for stale_socket in self.state.get_websockets_with_tag(&peer_tag) {
                let _ = stale_socket.close(
                    Some(CLOSE_CODE_REPLACED_BY_RESUMED_CONNECTION),
                    Some("replaced by resumed connection"),
                );
            }
        }

        let pair = WebSocketPair::new()?;
        let server_socket = pair.server;
        let role_tag = role_socket_tag(role);
        self.state
            .accept_websocket_with_tags(&server_socket, &[peer_tag.as_str(), role_tag.as_str()]);
        server_socket.serialize_attachment(SocketAttachment {
            peer_id,
            role,
            connection_serial: mint_secret_token()?,
        })?;
        send_notice(
            &server_socket,
            &RelayControlNotice::Welcome {
                peer_id,
                role,
                resume_token,
                host_peer_id: if role == RelayRole::Host { Some(peer_id) } else { self.host_peer_id() },
                usage,
                limits: config.limits,
            },
        );
        let joined = RelayControlNotice::PeerJoined { peer_id, role };
        for other_socket in self.state.get_websockets() {
            if !self.state.get_tags(&other_socket).contains(&peer_tag) {
                send_notice(&other_socket, &joined);
            }
        }
        Response::from_websocket(pair.client)
    }

    fn reject(&self, socket: &WebSocket, reason: FrameRejectionReason, recipient_peer_id: Option<u32>) {
        send_notice(socket, &RelayControlNotice::FrameRejected { reason, recipient_peer_id });
    }

    async fn load_usage(&self) -> Result<SessionUsageCounters> {
        if let Some(usage) = *self.cached_usage.borrow() {
            return Ok(usage);
        }
        let usage = self
            .state
            .storage()
            .get::<SessionUsageCounters>(USAGE_STORAGE_KEY)
            .await?
            .unwrap_or_default();
        *self.cached_usage.borrow_mut() = Some(usage);
        Ok(usage)
    }

    async fn persist_usage(&self) -> Result<()> {
        let usage = *self.cached_usage.borrow();
        if let Some(usage) = usage {
            self.state.storage().put(USAGE_STORAGE_KEY, usage).await?;
            self.admitted_since_last_persist.set(0);
        }
        Ok(())
    }

    /// Counts one inbound message against the session caps (every inbound
    /// message is billed, so this runs before any parsing). `false`: the
    /// frame broke a cap and the sender was told.
    async fn admit_inbound_frame(&self, socket: &WebSocket, frame_length_bytes: u64) -> Result<bool> {
        let config = RelayWorkerConfig::from_env(&self.env)?;
        let mut usage = self.load_usage().await?;
        if let Err(reason) = usage.try_admit_frame(frame_length_bytes, &config.limits) {
            self.reject(socket, reason, None);
            return Ok(false);
        }
        *self.cached_usage.borrow_mut() = Some(usage);
        let admitted = self.admitted_since_last_persist.get() + 1;
        self.admitted_since_last_persist.set(admitted);
        if admitted >= config.usage_counter_persist_every_messages {
            self.persist_usage().await?;
        }
        Ok(true)
    }

    async fn relay_frame(&self, socket: &WebSocket, attachment: &SocketAttachment, frame: &[u8]) -> Result<()> {
        if !self.admit_inbound_frame(socket, frame.len() as u64).await? {
            return Ok(());
        }

        let Ok(header) = RelayFrameHeader::decode(frame) else {
            self.reject(socket, FrameRejectionReason::MalformedRelayFrame, None);
            return Ok(());
        };
        let delivery =
            match decide_relay_delivery(attachment.role, attachment.peer_id, self.host_peer_id(), &header) {
                Ok(delivery) => delivery,
                Err(reason) => {
                    self.reject(socket, reason, Some(header.recipient_peer_id));
                    return Ok(());
                }
            };
        match delivery {
            RelayDelivery::SinglePeer(recipient_peer_id) => {
                let recipients = self.state.get_websockets_with_tag(&peer_socket_tag(recipient_peer_id));
                if recipients.is_empty() {
                    self.reject(socket, FrameRejectionReason::RecipientNotConnected, Some(recipient_peer_id));
                }
                for recipient in recipients {
                    let _ = recipient.send_with_bytes(frame);
                }
            }
            RelayDelivery::EveryOtherPeer => {
                let sender_tag = peer_socket_tag(attachment.peer_id);
                for recipient in self.state.get_websockets() {
                    if !self.state.get_tags(&recipient).contains(&sender_tag) {
                        let _ = recipient.send_with_bytes(frame);
                    }
                }
            }
        }
        Ok(())
    }

    async fn on_socket_gone(&self, socket: &WebSocket) -> Result<()> {
        let _ = socket.close(Some(CLOSE_CODE_NORMAL), Some("closing"));
        self.persist_usage().await?;
        let Some(closing) = socket.deserialize_attachment::<SocketAttachment>()? else {
            return Ok(());
        };
        let peer_tag = peer_socket_tag(closing.peer_id);
        let still_connected_elsewhere = self.state.get_websockets_with_tag(&peer_tag).iter().any(|other| {
            other
                .deserialize_attachment::<SocketAttachment>()
                .ok()
                .flatten()
                .is_some_and(|attachment| attachment.connection_serial != closing.connection_serial)
        });
        if still_connected_elsewhere {
            return Ok(());
        }
        let left = RelayControlNotice::PeerLeft { peer_id: closing.peer_id };
        for other_socket in self.state.get_websockets() {
            if !self.state.get_tags(&other_socket).contains(&peer_tag) {
                send_notice(&other_socket, &left);
            }
        }
        Ok(())
    }
}
