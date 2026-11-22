//! Relay knobs, read from wrangler `[vars]` (overridable per run with
//! `wrangler dev --var NAME:value`, which the integration test uses to get
//! a short join-code TTL and small caps).

use minimappr_relay_wire::SessionRelayLimits;
use worker::{Env, Error, Result};

pub const JOIN_CODES_DATABASE_BINDING: &str = "JOIN_CODES_DB";
pub const SESSION_RELAY_OBJECT_BINDING: &str = "SESSION_RELAY";
/// Lookup-code collisions are retried this many times before failing (at
/// ~1M codes and a few hundred live sessions a retry is already rare).
pub const JOIN_CODE_ALLOCATION_ATTEMPTS: usize = 8;
/// Symbols in a lookup code (must match minimappr-transport's
/// `JOIN_CODE_LOOKUP_LENGTH`).
pub const JOIN_CODE_LOOKUP_LENGTH: usize = 4;
/// Bytes of randomness in every role / resume token and session name.
pub const RANDOM_TOKEN_BYTES: usize = 32;
pub const RELAY_SESSION_NAME_BYTES: usize = 16;
pub const MILLISECONDS_PER_SECOND: u64 = 1_000;

#[derive(Debug, Clone, Copy)]
pub struct RelayWorkerConfig {
    pub join_code_time_to_live_seconds: u64,
    pub session_time_to_live_seconds: u64,
    /// Resolves (token mints) one lookup code allows: bounds how many relay
    /// tokens a leaked or brute-forced lookup code can hand out.
    pub join_code_maximum_resolves: u64,
    pub limits: SessionRelayLimits,
    /// Usage counters are written to Durable Object storage every N
    /// admitted frames (and on every close) rather than per frame: storage
    /// writes are billed, and an eviction loses at most N-1 counts.
    pub usage_counter_persist_every_messages: u64,
}

fn read_number_var(env: &Env, name: &str) -> Result<u64> {
    let text = env.var(name)?.to_string();
    text.trim()
        .parse::<u64>()
        .map_err(|_| Error::RustError(format!("wrangler var {name} is not an unsigned integer: {text}")))
}

impl RelayWorkerConfig {
    pub fn from_env(env: &Env) -> Result<Self> {
        Ok(Self {
            join_code_time_to_live_seconds: read_number_var(env, "JOIN_CODE_TTL_SECONDS")?,
            session_time_to_live_seconds: read_number_var(env, "SESSION_TTL_SECONDS")?,
            join_code_maximum_resolves: read_number_var(env, "JOIN_CODE_MAXIMUM_RESOLVES")?,
            limits: SessionRelayLimits {
                maximum_inbound_messages_per_session: read_number_var(
                    env,
                    "SESSION_MAXIMUM_INBOUND_MESSAGES",
                )?,
                maximum_inbound_bytes_per_session: read_number_var(
                    env,
                    "SESSION_MAXIMUM_INBOUND_BYTES",
                )?,
                maximum_frame_bytes: read_number_var(env, "MAXIMUM_FRAME_BYTES")?,
                maximum_connections_per_session: read_number_var(env, "SESSION_MAXIMUM_CONNECTIONS")?
                    as u32,
            },
            usage_counter_persist_every_messages: read_number_var(
                env,
                "USAGE_COUNTER_PERSIST_EVERY_MESSAGES",
            )?
            .max(1),
        })
    }
}
