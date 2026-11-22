//! Field Mode session transport.
//!
//! Sans-IO core (no tokio, clock or OS randomness; all injected):
//! - `relay_client_session`: sealed messaging through the untrusted relay,
//!   replay filtering, reconnect with resume.
//! - `outbound_scheduler` + `clip_chunking`: clips chunked into
//!   `ClipPayloadChunk`s strictly below control frames, abandoned when late.
//! - `join_code` + `join_code_exchange_over_relay`: `LOOKUP-SECRET` codes
//!   and the SPAKE2 exchange carried as relay flights.
//!
//! `native_websocket` (feature `native-websocket`) adapts it to
//! tokio-tungstenite and the relay's HTTP API.

pub mod clip_chunking;
pub mod join_code;
pub mod join_code_exchange_over_relay;
pub mod outbound_scheduler;
pub mod reconnect_backoff;
pub mod relay_client_events;
pub mod relay_client_session;
pub mod transport_config;

#[cfg(feature = "native-websocket")]
pub mod native_websocket;

pub use join_code::FieldJoinCode;
pub use relay_client_events::{RelayClientError, RelayClientEvent, RelayClientInboundStatistics};
pub use relay_client_session::{RelayClientSession, RelayConnectParameters, RelayConnectionPhase};
pub use transport_config::RelayTransportConfig;
