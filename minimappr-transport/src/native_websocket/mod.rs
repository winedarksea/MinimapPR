//! Native (tokio) adapter: the relay Worker's HTTP API over reqwest, and a
//! driver that pumps a `RelayClientSession` over tokio-tungstenite.

pub mod native_relay_driver;
pub mod relay_http_client;

pub use native_relay_driver::NativeRelayDriver;
pub use relay_http_client::RelayHttpClient;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeRelayError {
    InvalidUrl(String),
    Http(String),
    /// The Worker answered with a non-success status.
    Status { status_code: u16, body: String },
    WebSocket(String),
    Transport(crate::RelayClientError),
}

impl std::fmt::Display for NativeRelayError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "native relay error: {self:?}")
    }
}

impl std::error::Error for NativeRelayError {}

impl From<crate::RelayClientError> for NativeRelayError {
    fn from(error: crate::RelayClientError) -> Self {
        Self::Transport(error)
    }
}
