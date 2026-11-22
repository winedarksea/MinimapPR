//! Pumps a `RelayClientSession` over the browser `WebSocket`.
//!
//! Callback driven: socket events feed the session, then the driver flushes
//! outbound frames and hands queued `RelayClientEvent`s to the application
//! sink. The application calls `tick()` on a timer so reconnects (with the
//! session's backoff) and gated flushes happen without inbound traffic.
//!
//! Head-of-line control: frames are written one at a time and writing stops
//! while `bufferedAmount` exceeds a ceiling, so a control frame enqueued
//! mid-clip waits behind at most the ceiling, not a whole clip sitting in the
//! browser's send buffer (relay.md open issue for the browser adapter).

use std::cell::RefCell;
use std::rc::Rc;

use minimappr_relay_wire::relay_http_api::{
    connect_path, CONNECT_QUERY_PEER_ID, CONNECT_QUERY_RESUME_TOKEN, CONNECT_QUERY_ROLE_TOKEN,
};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;

use super::{describe_javascript_value, BrowserCryptoRandomSource, WebRelayError};
use crate::relay_client_session::RelayClientSession;
use crate::RelayClientEvent;

/// Default `bufferedAmount` ceiling: two default 4 KiB clip chunks plus
/// sealing overhead, so control latency stays near one chunk's airtime.
pub const DEFAULT_MAXIMUM_BUFFERED_AMOUNT_BYTES: u32 = 12 * 1024;

#[derive(Debug, Clone)]
pub struct WebRelayDriverOptions {
    /// Writing pauses while the socket's `bufferedAmount` exceeds this.
    pub maximum_buffered_amount_bytes: u32,
}

impl Default for WebRelayDriverOptions {
    fn default() -> Self {
        Self {
            maximum_buffered_amount_bytes: DEFAULT_MAXIMUM_BUFFERED_AMOUNT_BYTES,
        }
    }
}

type RelayEventSink = Box<dyn FnMut(RelayClientEvent)>;

/// Closures must outlive the socket; dropping them detaches the handlers.
struct SocketCallbacks {
    _on_open: Closure<dyn FnMut(web_sys::Event)>,
    _on_message: Closure<dyn FnMut(web_sys::MessageEvent)>,
    _on_close: Closure<dyn FnMut(web_sys::CloseEvent)>,
    _on_error: Closure<dyn FnMut(web_sys::Event)>,
}

struct DriverState {
    session: RelayClientSession,
    websocket_connect_url_base: String,
    role_token: String,
    socket: Option<web_sys::WebSocket>,
    socket_callbacks: Option<SocketCallbacks>,
    /// Callbacks of closed sockets. A closure may be the one running when its
    /// socket dies, so it is parked here and freed on the next timer tick.
    retired_socket_callbacks: Vec<SocketCallbacks>,
    /// Bumped per socket so late callbacks from a replaced socket are ignored.
    socket_generation: u64,
    random_source: BrowserCryptoRandomSource,
    clock_origin_milliseconds: f64,
    options: WebRelayDriverOptions,
    shut_down: bool,
    last_error: Option<String>,
}

#[derive(Clone)]
pub struct WebRelayDriver {
    state: Rc<RefCell<DriverState>>,
    event_sink: Rc<RefCell<Option<RelayEventSink>>>,
    dispatching_events: Rc<RefCell<bool>>,
}

impl WebRelayDriver {
    /// `http_base_url` is the relay origin (`http` maps to `ws`, `https` to
    /// `wss`); `""` uses the page's own origin.
    pub fn new(
        session: RelayClientSession,
        http_base_url: &str,
        relay_session_name: &str,
        role_token: &str,
        options: WebRelayDriverOptions,
    ) -> Result<Self, WebRelayError> {
        let origin = if http_base_url.is_empty() {
            page_origin()?
        } else {
            http_base_url.trim_end_matches('/').to_string()
        };
        let websocket_base = origin.replacen("https://", "wss://", 1).replacen("http://", "ws://", 1);
        let random_source = BrowserCryptoRandomSource::from_global()
            .ok_or_else(|| WebRelayError::WebSocket("WebCrypto unavailable".to_string()))?;
        Ok(Self {
            state: Rc::new(RefCell::new(DriverState {
                session,
                websocket_connect_url_base: format!("{websocket_base}{}", connect_path(relay_session_name)),
                role_token: role_token.to_string(),
                socket: None,
                socket_callbacks: None,
                retired_socket_callbacks: Vec::new(),
                socket_generation: 0,
                random_source,
                clock_origin_milliseconds: js_sys::Date::now(),
                options,
                shut_down: false,
                last_error: None,
            })),
            event_sink: Rc::new(RefCell::new(None)),
            dispatching_events: Rc::new(RefCell::new(false)),
        })
    }

    /// Receives every session event, in order. It may call back into the
    /// driver (`with_session`, `flush_outbound`).
    pub fn set_event_sink(&self, sink: impl FnMut(RelayClientEvent) + 'static) {
        *self.event_sink.borrow_mut() = Some(Box::new(sink));
    }

    pub fn now_milliseconds(&self) -> u64 {
        elapsed_milliseconds(self.state.borrow().clock_origin_milliseconds)
    }

    pub fn is_socket_open(&self) -> bool {
        self.state.borrow().socket.is_some()
    }

    pub fn last_error(&self) -> Option<String> {
        self.state.borrow().last_error.clone()
    }

    /// Runs `operation` on the session (queue a message, install a key),
    /// then flushes whatever it queued.
    pub fn with_session<T>(&self, operation: impl FnOnce(&mut RelayClientSession, u64) -> T) -> T {
        let result = {
            let mut state = self.state.borrow_mut();
            let now = elapsed_milliseconds(state.clock_origin_milliseconds);
            operation(&mut state.session, now)
        };
        self.flush_outbound();
        self.dispatch_pending_events();
        result
    }

    /// Opens the socket; the outcome arrives as `Connected` or `Disconnected`.
    pub fn connect(&self) -> Result<(), WebRelayError> {
        let url = {
            let state = self.state.borrow();
            connect_url(&state)
        };
        let socket = web_sys::WebSocket::new(&url)
            .map_err(|error| WebRelayError::WebSocket(describe_javascript_value(&error)))?;
        socket.set_binary_type(web_sys::BinaryType::Arraybuffer);
        let generation = {
            let mut state = self.state.borrow_mut();
            state.socket_generation += 1;
            state.socket_generation
        };
        let callbacks = self.attach_socket_callbacks(&socket, generation);
        let mut state = self.state.borrow_mut();
        state.socket = Some(socket);
        state.socket_callbacks = Some(callbacks);
        Ok(())
    }

    /// Timer turn: reconnect when the backoff allows, flush, dispatch.
    pub fn tick(&self) {
        self.state.borrow_mut().retired_socket_callbacks.clear();
        let reconnect_now = {
            let state = self.state.borrow();
            let now = elapsed_milliseconds(state.clock_origin_milliseconds);
            !state.shut_down && state.socket.is_none() && state.session.reconnect_due(now)
        };
        if reconnect_now {
            if let Err(error) = self.connect() {
                self.record_socket_loss(Some(error.to_string()));
            }
        }
        self.flush_outbound();
        self.dispatch_pending_events();
    }

    /// Closes the socket and stops reconnecting.
    pub fn shut_down(&self) {
        self.state.borrow_mut().shut_down = true;
        self.drop_socket(None);
        // The socket closures and the application sink hold driver clones
        // (Rc cycles), and the app stops calling `tick()` once shut down, so
        // free them here. Deferred to a later task because shut_down may be
        // running inside one of those closures.
        let driver = self.clone();
        wasm_bindgen_futures::spawn_local(async move {
            driver.state.borrow_mut().retired_socket_callbacks.clear();
            driver.event_sink.borrow_mut().take();
        });
    }

    /// Writes ready frames until the queue empties or `bufferedAmount`
    /// passes the ceiling; returns how many were written.
    pub fn flush_outbound(&self) -> usize {
        let mut written = 0;
        loop {
            let mut state = self.state.borrow_mut();
            let Some(socket) = state.socket.clone() else {
                return written;
            };
            if socket.ready_state() != web_sys::WebSocket::OPEN
                || socket.buffered_amount() > state.options.maximum_buffered_amount_bytes
            {
                return written;
            }
            let now = elapsed_milliseconds(state.clock_origin_milliseconds);
            let DriverState { session, random_source, .. } = &mut *state;
            let frame = match session.poll_transmit(now, random_source) {
                Ok(Some(frame)) => frame,
                Ok(None) => return written,
                Err(error) => {
                    state.last_error = Some(error.to_string());
                    return written;
                }
            };
            drop(state);
            if let Err(error) = socket.send_with_u8_array(&frame) {
                self.drop_socket(Some(describe_javascript_value(&error)));
                return written;
            }
            written += 1;
        }
    }

    fn attach_socket_callbacks(&self, socket: &web_sys::WebSocket, generation: u64) -> SocketCallbacks {
        let on_open = {
            let driver = self.clone();
            Closure::<dyn FnMut(web_sys::Event)>::new(move |_event: web_sys::Event| {
                if !driver.is_current_generation(generation) {
                    return;
                }
                driver.state.borrow_mut().session.on_socket_opened();
                driver.flush_outbound();
                driver.dispatch_pending_events();
            })
        };
        let on_message = {
            let driver = self.clone();
            Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |event: web_sys::MessageEvent| {
                if !driver.is_current_generation(generation) {
                    return;
                }
                driver.on_socket_message(event.data());
            })
        };
        let on_close = {
            let driver = self.clone();
            Closure::<dyn FnMut(web_sys::CloseEvent)>::new(move |event: web_sys::CloseEvent| {
                if driver.is_current_generation(generation) {
                    driver.drop_socket(Some(format!("socket closed (code {})", event.code())));
                }
            })
        };
        let on_error = {
            let driver = self.clone();
            Closure::<dyn FnMut(web_sys::Event)>::new(move |_event: web_sys::Event| {
                // The browser hides the HTTP status of a refused upgrade.
                if driver.is_current_generation(generation) {
                    driver.drop_socket(Some("socket error (refused upgrade or network)".to_string()));
                }
            })
        };
        socket.set_onopen(Some(on_open.as_ref().unchecked_ref()));
        socket.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
        socket.set_onclose(Some(on_close.as_ref().unchecked_ref()));
        socket.set_onerror(Some(on_error.as_ref().unchecked_ref()));
        SocketCallbacks {
            _on_open: on_open,
            _on_message: on_message,
            _on_close: on_close,
            _on_error: on_error,
        }
    }

    fn on_socket_message(&self, data: wasm_bindgen::JsValue) {
        {
            let mut state = self.state.borrow_mut();
            let now = elapsed_milliseconds(state.clock_origin_milliseconds);
            if let Some(text) = data.as_string() {
                state.session.on_text_frame(&text);
            } else if let Ok(buffer) = data.dyn_into::<js_sys::ArrayBuffer>() {
                let bytes = js_sys::Uint8Array::new(&buffer).to_vec();
                state.session.on_binary_frame(&bytes, now);
            }
        }
        self.flush_outbound();
        self.dispatch_pending_events();
    }

    fn is_current_generation(&self, generation: u64) -> bool {
        self.state.borrow().socket_generation == generation
    }

    fn drop_socket(&self, reason: Option<String>) {
        let socket = {
            let mut state = self.state.borrow_mut();
            state.socket_generation += 1;
            if let Some(callbacks) = state.socket_callbacks.take() {
                state.retired_socket_callbacks.push(callbacks);
            }
            state.socket.take()
        };
        if let Some(socket) = socket {
            socket.set_onopen(None);
            socket.set_onmessage(None);
            socket.set_onclose(None);
            socket.set_onerror(None);
            let _ = socket.close();
            self.record_socket_loss(reason);
        }
    }

    fn record_socket_loss(&self, reason: Option<String>) {
        {
            let mut state = self.state.borrow_mut();
            let now = elapsed_milliseconds(state.clock_origin_milliseconds);
            state.session.on_socket_closed(now);
            if reason.is_some() {
                state.last_error = reason;
            }
        }
        self.dispatch_pending_events();
    }

    /// Drains session events into the sink. Re-entrant calls (the sink
    /// queuing more work) are folded into the outer loop.
    fn dispatch_pending_events(&self) {
        if *self.dispatching_events.borrow() {
            return;
        }
        *self.dispatching_events.borrow_mut() = true;
        loop {
            let next_event = self.state.borrow_mut().session.poll_event();
            let Some(event) = next_event else { break };
            let mut sink = self.event_sink.borrow_mut().take();
            if let Some(sink_function) = sink.as_mut() {
                sink_function(event);
            }
            let mut slot = self.event_sink.borrow_mut();
            if slot.is_none() {
                *slot = sink;
            }
        }
        *self.dispatching_events.borrow_mut() = false;
    }
}

fn connect_url(state: &DriverState) -> String {
    let parameters = state.session.connection_parameters();
    let mut url = format!(
        "{}?{CONNECT_QUERY_PEER_ID}={}&{CONNECT_QUERY_ROLE_TOKEN}={}",
        state.websocket_connect_url_base,
        parameters.peer_id,
        encode_query_component(&state.role_token)
    );
    if let Some(resume_token) = &parameters.resume_token {
        url.push_str(&format!("&{CONNECT_QUERY_RESUME_TOKEN}={}", encode_query_component(resume_token)));
    }
    url
}

fn encode_query_component(value: &str) -> String {
    String::from(js_sys::encode_uri_component(value))
}

fn page_origin() -> Result<String, WebRelayError> {
    let window = web_sys::window().ok_or_else(|| WebRelayError::WebSocket("no window for page origin".to_string()))?;
    window
        .location()
        .origin()
        .map_err(|error| WebRelayError::WebSocket(describe_javascript_value(&error)))
}

fn elapsed_milliseconds(origin_milliseconds: f64) -> u64 {
    (js_sys::Date::now() - origin_milliseconds).max(0.0) as u64
}
