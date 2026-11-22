//! D1 access: relay session registrations and join-code lookups.
//!
//! D1 holds only routing metadata: lookup codes, relay session names, the
//! (public) Field session id and a SHA-256 of the host token. No role token,
//! no join-code secret, no session key.

use serde::Deserialize;
use worker::wasm_bindgen::JsValue;
use worker::{D1Database, Result};

use crate::random_tokens::mint_join_code_lookup;
use crate::worker_config::JOIN_CODE_ALLOCATION_ATTEMPTS;

/// D1 rejects BigInt binds, so integers travel as JS numbers (exact below 2^53).
fn d1_number(value: u64) -> JsValue {
    JsValue::from_f64(value as f64)
}

pub async fn insert_relay_session(
    database: &D1Database,
    relay_session_name: &str,
    host_token_sha256_hex: &str,
    field_session_id_hex: &str,
    now_unix_milliseconds: u64,
    expires_at_unix_milliseconds: u64,
) -> Result<()> {
    database
        .prepare(
            "INSERT INTO relay_sessions (relay_session_name, host_token_sha256_hex, \
             field_session_id_hex, created_at_unix_ms, expires_at_unix_ms) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
        )
        .bind(&[
            relay_session_name.into(),
            host_token_sha256_hex.into(),
            field_session_id_hex.into(),
            d1_number(now_unix_milliseconds),
            d1_number(expires_at_unix_milliseconds),
        ])?
        .run()
        .await?;
    Ok(())
}

#[derive(Debug, Deserialize)]
pub struct RelaySessionRow {
    pub host_token_sha256_hex: String,
    pub field_session_id_hex: String,
    pub expires_at_unix_ms: f64,
}

pub async fn find_relay_session(
    database: &D1Database,
    relay_session_name: &str,
) -> Result<Option<RelaySessionRow>> {
    database
        .prepare(
            "SELECT host_token_sha256_hex, field_session_id_hex, expires_at_unix_ms FROM relay_sessions \
             WHERE relay_session_name = ?1",
        )
        .bind(&[relay_session_name.into()])?
        .first::<RelaySessionRow>(None)
        .await
}

/// Allocates a fresh lookup code (retrying collisions); `None` if every
/// attempt collided.
pub async fn allocate_join_code(
    database: &D1Database,
    relay_session_name: &str,
    field_session_id_hex: &str,
    now_unix_milliseconds: u64,
    expires_at_unix_milliseconds: u64,
) -> Result<Option<String>> {
    for _ in 0..JOIN_CODE_ALLOCATION_ATTEMPTS {
        let lookup_code = mint_join_code_lookup()?;
        let result = database
            .prepare(
                "INSERT OR IGNORE INTO join_codes (lookup_code, relay_session_name, \
                 field_session_id_hex, created_at_unix_ms, expires_at_unix_ms, resolve_count) \
                 VALUES (?1, ?2, ?3, ?4, ?5, 0)",
            )
            .bind(&[
                lookup_code.as_str().into(),
                relay_session_name.into(),
                field_session_id_hex.into(),
                d1_number(now_unix_milliseconds),
                d1_number(expires_at_unix_milliseconds),
            ])?
            .run()
            .await?;
        let inserted = result.meta()?.and_then(|meta| meta.changes).unwrap_or(0) == 1;
        if inserted {
            return Ok(Some(lookup_code));
        }
    }
    Ok(None)
}

#[derive(Debug, Deserialize)]
pub struct JoinCodeRow {
    pub relay_session_name: String,
    pub field_session_id_hex: String,
    pub expires_at_unix_ms: f64,
    pub resolve_count: f64,
}

pub async fn find_join_code(database: &D1Database, lookup_code: &str) -> Result<Option<JoinCodeRow>> {
    database
        .prepare(
            "SELECT relay_session_name, field_session_id_hex, expires_at_unix_ms, resolve_count \
             FROM join_codes WHERE lookup_code = ?1",
        )
        .bind(&[lookup_code.into()])?
        .first::<JoinCodeRow>(None)
        .await
}

/// Counts one resolve; false if the code hit its resolve budget meanwhile.
pub async fn claim_join_code_resolve(
    database: &D1Database,
    lookup_code: &str,
    maximum_resolves: u64,
) -> Result<bool> {
    let result = database
        .prepare(
            "UPDATE join_codes SET resolve_count = resolve_count + 1 \
             WHERE lookup_code = ?1 AND resolve_count < ?2",
        )
        .bind(&[lookup_code.into(), d1_number(maximum_resolves)])?
        .run()
        .await?;
    Ok(result.meta()?.and_then(|meta| meta.changes).unwrap_or(0) == 1)
}

pub async fn delete_join_code(database: &D1Database, lookup_code: &str) -> Result<()> {
    database
        .prepare("DELETE FROM join_codes WHERE lookup_code = ?1")
        .bind(&[lookup_code.into()])?
        .run()
        .await?;
    Ok(())
}

/// Cron cleanup: expired codes, codes of expired sessions, expired sessions.
/// (Each session's Durable Object wipes its own storage with an alarm at
/// session expiry, so this covers all relay data.)
pub async fn delete_expired_rows(database: &D1Database, now_unix_milliseconds: u64) -> Result<()> {
    let cutoff = d1_number(now_unix_milliseconds);
    database
        .batch(vec![
            database
                .prepare("DELETE FROM join_codes WHERE expires_at_unix_ms <= ?1")
                .bind(std::slice::from_ref(&cutoff))?,
            database
                .prepare(
                    "DELETE FROM join_codes WHERE relay_session_name IN \
                     (SELECT relay_session_name FROM relay_sessions WHERE expires_at_unix_ms <= ?1)",
                )
                .bind(std::slice::from_ref(&cutoff))?,
            database
                .prepare("DELETE FROM relay_sessions WHERE expires_at_unix_ms <= ?1")
                .bind(&[cutoff])?,
        ])
        .await?;
    Ok(())
}
