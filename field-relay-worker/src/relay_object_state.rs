//! Durable Object storage layout and the Worker -> object internal API.
//!
//! Storage keys (all wiped by the expiry alarm):
//! - `meta`             -> `RelayObjectMeta`
//! - `token:<sha256>`   -> `RelayRole` (session-wide and per-joiner tokens)
//! - `peer:<peer id>`   -> `PeerBinding` (who may rebind that peer id)
//! - `usage`            -> `SessionUsageCounters`

use minimappr_relay_wire::RelayRole;
use serde::{Deserialize, Serialize};

pub const META_STORAGE_KEY: &str = "meta";
pub const USAGE_STORAGE_KEY: &str = "usage";
pub const INTERNAL_INITIALIZE_PATH: &str = "/internal/initialize";
pub const INTERNAL_MINT_ROLE_TOKEN_PATH: &str = "/internal/mint-role-token";
/// The Worker forwards `/v1/relay/{name}/connect` unchanged; the object
/// matches on this suffix. `/internal/*` paths are never forwarded from
/// outside, so only the Worker itself can reach them.
pub const CONNECT_PATH_SUFFIX: &str = "/connect";
/// Placeholder origin for Worker -> object stub requests.
pub const INTERNAL_REQUEST_ORIGIN: &str = "https://session-relay-object";
pub const HOST_ROLE_SOCKET_TAG: &str = "role:host";

/// WebSocket close codes the relay uses (4000-4999 is the application range).
pub const CLOSE_CODE_REPLACED_BY_RESUMED_CONNECTION: u16 = 4000;
pub const CLOSE_CODE_SESSION_EXPIRED: u16 = 4001;
pub const CLOSE_CODE_NORMAL: u16 = 1000;

pub fn token_storage_key(token_sha256_hex: &str) -> String {
    format!("token:{token_sha256_hex}")
}

pub fn peer_storage_key(peer_id: u32) -> String {
    format!("peer:{peer_id}")
}

pub fn peer_socket_tag(peer_id: u32) -> String {
    format!("peer:{peer_id}")
}

pub fn role_socket_tag(role: RelayRole) -> String {
    format!("role:{}", role.as_str())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelayObjectMeta {
    pub expires_at_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerBinding {
    pub role: RelayRole,
    pub resume_token_sha256_hex: String,
}

/// Survives hibernation with the socket (`serialize_attachment`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocketAttachment {
    pub peer_id: u32,
    pub role: RelayRole,
    /// Distinguishes a resumed socket from the one it replaced.
    pub connection_serial: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeRelayObjectRequest {
    pub expires_at_unix_ms: u64,
    pub role_token_digests: Vec<(String, RelayRole)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MintRoleTokenRequest {
    pub role: RelayRole,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MintRoleTokenResponse {
    pub role_token: String,
}
