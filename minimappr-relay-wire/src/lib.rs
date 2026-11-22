//! Field Mode relay wire contract, shared verbatim by the Cloudflare relay
//! Worker (`field-relay-worker/`, wasm) and the `minimappr-transport` client
//! so both ends agree on framing, routing policy and caps by construction.
//!
//! The relay is untrusted: it routes on a cleartext header and never holds a
//! session key or a join-code secret.

pub mod join_code_lookup;
pub mod relay_control_notice;
pub mod relay_frame;
pub mod relay_http_api;
pub mod relay_routing_policy;
pub mod session_usage_caps;

pub use relay_control_notice::RelayControlNotice;
pub use relay_frame::{
    decode_relay_frame, encode_relay_frame, RelayFrameDecodeError, RelayFrameHeader,
    RelayFrameKind, BROADCAST_PEER_ID, RELAY_FRAME_HEADER_LENGTH_BYTES,
};
pub use relay_routing_policy::{
    decide_relay_delivery, FrameRejectionReason, RelayDelivery, RelayRole,
};
pub use session_usage_caps::{ConnectionRefusalReason, SessionRelayLimits, SessionUsageCounters};
