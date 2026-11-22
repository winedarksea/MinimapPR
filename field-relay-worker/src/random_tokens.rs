//! Token minting and hashing. Only SHA-256 digests of tokens are stored (D1
//! and Durable Object storage), so a storage dump grants no relay access.

use minimappr_relay_wire::join_code_lookup::lookup_code_from_random_bytes;
use minimappr_relay_wire::relay_http_api::encode_hex;
use sha2::{Digest, Sha256};
use worker::{Error, Result};

use crate::worker_config::{JOIN_CODE_LOOKUP_LENGTH, RANDOM_TOKEN_BYTES, RELAY_SESSION_NAME_BYTES};

fn random_bytes(length: usize) -> Result<Vec<u8>> {
    let mut bytes = vec![0_u8; length];
    // crypto.getRandomValues via getrandom's `js` feature.
    getrandom::getrandom(&mut bytes)
        .map_err(|error| Error::RustError(format!("randomness unavailable: {error}")))?;
    Ok(bytes)
}

pub fn mint_secret_token() -> Result<String> {
    Ok(encode_hex(&random_bytes(RANDOM_TOKEN_BYTES)?))
}

pub fn mint_relay_session_name() -> Result<String> {
    Ok(encode_hex(&random_bytes(RELAY_SESSION_NAME_BYTES)?))
}

pub fn mint_join_code_lookup() -> Result<String> {
    Ok(lookup_code_from_random_bytes(&random_bytes(JOIN_CODE_LOOKUP_LENGTH)?))
}

pub fn sha256_hex(token: &str) -> String {
    encode_hex(&Sha256::digest(token.as_bytes()))
}

/// Constant-time comparison of two equal-length digests.
pub fn digests_match(left: &str, right: &str) -> bool {
    left.len() == right.len()
        && left
            .bytes()
            .zip(right.bytes())
            .fold(0_u8, |difference, (a, b)| difference | (a ^ b))
            == 0
}

pub fn is_relay_session_name(text: &str) -> bool {
    text.len() == RELAY_SESSION_NAME_BYTES * 2
        && text.bytes().all(|symbol| symbol.is_ascii_hexdigit() && !symbol.is_ascii_uppercase())
}
