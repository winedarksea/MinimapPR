//! Pumps a `RelayClientSession` over a tokio-tungstenite WebSocket.
//!
//! Frames are written and flushed one at a time, and the session is asked
//! for the next only afterwards, so control frames enqueued mid-clip
//! overtake the clip's remaining chunks (kernel send buffers still add some
//! head-of-line delay on a slow uplink; see relay.md open issues).

use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use minimappr_relay_wire::relay_http_api::{
    connect_path, CONNECT_QUERY_PEER_ID, CONNECT_QUERY_RESUME_TOKEN, CONNECT_QUERY_ROLE_TOKEN,
};
use rand_core::OsRng;
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};
use url::Url;

use super::NativeRelayError;
use crate::relay_client_session::RelayClientSession;
use crate::RelayClientEvent;

type RelaySocket = WebSocketStream<MaybeTlsStream<TcpStream>>;

pub struct NativeRelayDriver {
    pub session: RelayClientSession,
    websocket_connect_url_base: Url,
    role_token: String,
    socket: Option<RelaySocket>,
    clock_origin: Instant,
    random_source: OsRng,
}

impl NativeRelayDriver {
    /// `http_base_url` is the Worker origin (`http://` maps to `ws://`,
    /// `https://` to `wss://`).
    pub fn new(
        session: RelayClientSession,
        http_base_url: &str,
        relay_session_name: &str,
        role_token: &str,
    ) -> Result<Self, NativeRelayError> {
        let websocket_base = http_base_url
            .trim_end_matches('/')
            .replacen("https://", "wss://", 1)
            .replacen("http://", "ws://", 1);
        let websocket_connect_url_base =
            Url::parse(&format!("{websocket_base}{}", connect_path(relay_session_name)))
                .map_err(|error| NativeRelayError::InvalidUrl(error.to_string()))?;
        Ok(Self {
            session,
            websocket_connect_url_base,
            role_token: role_token.to_string(),
            socket: None,
            clock_origin: Instant::now(),
            random_source: OsRng,
        })
    }

    pub fn now_milliseconds(&self) -> u64 {
        self.clock_origin.elapsed().as_millis() as u64
    }

    pub fn is_socket_open(&self) -> bool {
        self.socket.is_some()
    }

    fn connect_url(&self) -> Url {
        let parameters = self.session.connection_parameters();
        let mut url = self.websocket_connect_url_base.clone();
        {
            let mut query = url.query_pairs_mut();
            query.append_pair(CONNECT_QUERY_PEER_ID, &parameters.peer_id.to_string());
            query.append_pair(CONNECT_QUERY_ROLE_TOKEN, &self.role_token);
            if let Some(resume_token) = &parameters.resume_token {
                query.append_pair(CONNECT_QUERY_RESUME_TOKEN, resume_token);
            }
        }
        url
    }

    /// Opens the socket. A refused upgrade (bad token, peer id already bound
    /// without a matching resume token, expired session) returns `Status`.
    pub async fn connect(&mut self) -> Result<(), NativeRelayError> {
        match tokio_tungstenite::connect_async(self.connect_url().as_str()).await {
            Ok((socket, _response)) => {
                self.socket = Some(socket);
                self.session.on_socket_opened();
                Ok(())
            }
            Err(error) => {
                let now = self.now_milliseconds();
                self.session.on_socket_closed(now);
                Err(map_connect_error(error))
            }
        }
    }

    /// Simulates a network drop: the socket vanishes without a close frame.
    pub fn drop_socket_abruptly(&mut self) {
        if self.socket.take().is_some() {
            let now = self.now_milliseconds();
            self.session.on_socket_closed(now);
        }
    }

    pub async fn close_gracefully(&mut self) {
        if let Some(mut socket) = self.socket.take() {
            let _ = socket.close(None).await;
            let now = self.now_milliseconds();
            self.session.on_socket_closed(now);
        }
    }

    /// Writes every frame the session has ready; returns how many.
    pub async fn flush_outbound(&mut self) -> Result<usize, NativeRelayError> {
        let mut written = 0;
        loop {
            if self.socket.is_none() {
                return Ok(written);
            }
            let now = self.now_milliseconds();
            let Some(frame) = self.session.poll_transmit(now, &mut self.random_source)? else {
                return Ok(written);
            };
            let socket = self.socket.as_mut().expect("checked above");
            if let Err(error) = socket.send(Message::Binary(frame)).await {
                self.drop_socket_abruptly();
                return Err(NativeRelayError::WebSocket(error.to_string()));
            }
            written += 1;
        }
    }

    /// One turn: reconnect if due, flush, then wait up to `wait` for one
    /// inbound frame and feed it to the session.
    pub async fn pump(&mut self, wait: Duration) -> Result<(), NativeRelayError> {
        if self.socket.is_none() {
            if self.session.reconnect_due(self.now_milliseconds()) {
                self.connect().await?;
            } else {
                tokio::time::sleep(wait).await;
                return Ok(());
            }
        }
        self.flush_outbound().await?;
        let Some(socket) = self.socket.as_mut() else {
            return Ok(());
        };
        let inbound = match tokio::time::timeout(wait, socket.next()).await {
            Err(_elapsed) => return Ok(()),
            Ok(inbound) => inbound,
        };
        let now = self.now_milliseconds();
        match inbound {
            Some(Ok(Message::Binary(frame))) => self.session.on_binary_frame(&frame, now),
            Some(Ok(Message::Text(text))) => self.session.on_text_frame(&text),
            Some(Ok(Message::Close(_))) | None => self.drop_socket_abruptly(),
            Some(Ok(_ping_pong_or_raw)) => {}
            Some(Err(error)) => {
                self.drop_socket_abruptly();
                return Err(NativeRelayError::WebSocket(error.to_string()));
            }
        }
        self.flush_outbound().await?;
        Ok(())
    }

    /// Pumps until the session emits an event or `timeout` passes.
    pub async fn next_event(&mut self, timeout: Duration, pump_slice: Duration) -> Option<RelayClientEvent> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(event) = self.session.poll_event() {
                return Some(event);
            }
            if Instant::now() >= deadline {
                return None;
            }
            // Transport errors already moved the session to Disconnected and
            // queued an event; keep pumping so reconnects happen.
            let _ = self.pump(pump_slice).await;
        }
    }
}

fn map_connect_error(error: tokio_tungstenite::tungstenite::Error) -> NativeRelayError {
    match error {
        tokio_tungstenite::tungstenite::Error::Http(response) => NativeRelayError::Status {
            status_code: response.status().as_u16(),
            body: response
                .body()
                .as_ref()
                .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
                .unwrap_or_default(),
        },
        other => NativeRelayError::WebSocket(other.to_string()),
    }
}
