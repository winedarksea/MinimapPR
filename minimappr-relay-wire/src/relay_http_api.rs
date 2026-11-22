//! JSON bodies and paths of the relay Worker's small HTTP API.
//!
//! - `POST /v1/sessions` (`CreateRelaySessionRequest`) -> `CreateRelaySessionResponse`
//! - `POST /v1/sessions/{relay_session_name}/join-codes` (Bearer host token) -> `IssuedJoinCodeLookup`
//! - `DELETE /v1/join-codes/{lookup_code}` (Bearer host token) -> 204
//! - `POST /v1/join` (`ResolveJoinCodeRequest`) -> `ResolveJoinCodeResponse`
//! - `GET /v1/relay/{relay_session_name}/connect?peer=&token=[&resume=]` -> WebSocket
//!
//! The Worker only ever sees the *lookup* half of a join code. The secret
//! half (the SPAKE2 password) stays on the phones; see `join_code_lookup`.

use serde::{Deserialize, Serialize};

use crate::relay_routing_policy::RelayRole;

pub const HEALTH_PATH: &str = "/v1/health";
pub const CREATE_SESSION_PATH: &str = "/v1/sessions";
pub const RESOLVE_JOIN_CODE_PATH: &str = "/v1/join";
pub const CONNECT_QUERY_PEER_ID: &str = "peer";
pub const CONNECT_QUERY_ROLE_TOKEN: &str = "token";
pub const CONNECT_QUERY_RESUME_TOKEN: &str = "resume";

pub fn issue_join_code_path(relay_session_name: &str) -> String {
    format!("/v1/sessions/{relay_session_name}/join-codes")
}

pub fn revoke_join_code_path(lookup_code: &str) -> String {
    format!("/v1/join-codes/{lookup_code}")
}

pub fn connect_path(relay_session_name: &str) -> String {
    format!("/v1/relay/{relay_session_name}/connect")
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateRelaySessionRequest {
    /// The Field Mode session id (hex). Not secret: it is the HKDF salt and
    /// appears in cleartext key-exchange flights; joiners need it for the
    /// SPAKE2 identities.
    pub field_session_id_hex: String,
    pub issue_join_code: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IssuedJoinCodeLookup {
    pub lookup_code: String,
    pub expires_at_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateRelaySessionResponse {
    pub relay_session_name: String,
    pub host_token: String,
    /// Session-wide node token for the QR code (QR joiners skip the PAKE).
    pub node_token: String,
    /// Session-wide view-only token for view links.
    pub viewer_token: String,
    pub session_expires_at_unix_ms: u64,
    pub join_code: Option<IssuedJoinCodeLookup>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolveJoinCodeRequest {
    pub lookup_code: String,
    /// `Node` or `Viewer`; `Host` is refused.
    pub requested_role: RelayRole,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolveJoinCodeResponse {
    pub relay_session_name: String,
    pub field_session_id_hex: String,
    /// Fresh per-joiner token minted by the session's Durable Object.
    pub role_token: String,
    pub role: RelayRole,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelayApiErrorBody {
    pub error: String,
}

pub fn encode_hex(bytes: &[u8]) -> String {
    const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        text.push(HEX_DIGITS[(byte >> 4) as usize] as char);
        text.push(HEX_DIGITS[(byte & 0x0F) as usize] as char);
    }
    text
}

pub fn decode_hex(text: &str) -> Option<Vec<u8>> {
    // `from_str_radix` alone accepts a leading `+` ("+f" -> 15), which would
    // let non-canonical ids through validation and into D1.
    if !text.len().is_multiple_of(2) || !text.bytes().all(|symbol| symbol.is_ascii_hexdigit()) {
        return None;
    }
    (0..text.len())
        .step_by(2)
        .map(|start| u8::from_str_radix(text.get(start..start + 2)?, 16).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trips_and_rejects_garbage() {
        let bytes = [0x00, 0x5A, 0xFF, 0x10];
        assert_eq!(encode_hex(&bytes), "005aff10");
        assert_eq!(decode_hex("005aff10").unwrap(), bytes);
        assert_eq!(decode_hex("abc"), None);
        assert_eq!(decode_hex("zz"), None);
        assert_eq!(decode_hex("+f"), None);
    }
}
