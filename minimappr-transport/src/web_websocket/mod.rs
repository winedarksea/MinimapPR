//! Browser (wasm32) adapter: the relay Worker's HTTP API over `fetch`, a
//! driver that pumps a `RelayClientSession` over the browser `WebSocket`,
//! and a WebCrypto-backed CSPRNG for sealing and SPAKE2.
//!
//! Browser constraints that shape it:
//! - A `WebSocket` cannot set headers, so the role token rides the query
//!   string (the Worker's contract already).
//! - A refused upgrade surfaces only as `error` + close code 1006; the HTTP
//!   status (401/409/410) is not observable, unlike the native adapter.
//! - The socket is callback driven, so the driver owns its state behind
//!   `Rc<RefCell<..>>` and hands events to an application sink.

pub mod browser_crypto_random_source;
pub mod relay_fetch_client;
pub mod web_relay_driver;

pub use browser_crypto_random_source::BrowserCryptoRandomSource;
pub use relay_fetch_client::RelayFetchClient;
pub use web_relay_driver::{WebRelayDriver, WebRelayDriverOptions};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebRelayError {
    /// `fetch`/JSON failure (network down, CORS, malformed body).
    Http(String),
    /// The Worker answered with a non-success status.
    Status { status_code: u16, body: String },
    WebSocket(String),
    Transport(crate::RelayClientError),
}

impl std::fmt::Display for WebRelayError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "web relay error: {self:?}")
    }
}

impl std::error::Error for WebRelayError {}

impl From<crate::RelayClientError> for WebRelayError {
    fn from(error: crate::RelayClientError) -> Self {
        Self::Transport(error)
    }
}

/// Renders a thrown JS value for error messages.
pub(crate) fn describe_javascript_value(value: &wasm_bindgen::JsValue) -> String {
    value
        .as_string()
        .or_else(|| {
            js_sys::Reflect::get(value, &wasm_bindgen::JsValue::from_str("message"))
                .ok()
                .and_then(|message| message.as_string())
        })
        .unwrap_or_else(|| format!("{value:?}"))
}
