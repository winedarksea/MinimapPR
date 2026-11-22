-- Field Mode relay: routing metadata only (no tokens in clear, no secrets,
-- no session content). Rows are deleted by the cron cleanup at expiry.

CREATE TABLE IF NOT EXISTS relay_sessions (
    relay_session_name    TEXT PRIMARY KEY,
    host_token_sha256_hex TEXT NOT NULL,
    field_session_id_hex  TEXT NOT NULL,
    created_at_unix_ms    INTEGER NOT NULL,
    expires_at_unix_ms    INTEGER NOT NULL
);

-- lookup_code is only the relay-visible half of a spoken join code; the
-- secret half (the SPAKE2 password) never reaches the Worker.
CREATE TABLE IF NOT EXISTS join_codes (
    lookup_code          TEXT PRIMARY KEY,
    relay_session_name   TEXT NOT NULL,
    field_session_id_hex TEXT NOT NULL,
    created_at_unix_ms   INTEGER NOT NULL,
    expires_at_unix_ms   INTEGER NOT NULL,
    resolve_count        INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS join_codes_by_expiry ON join_codes (expires_at_unix_ms);
CREATE INDEX IF NOT EXISTS join_codes_by_session ON join_codes (relay_session_name);
CREATE INDEX IF NOT EXISTS relay_sessions_by_expiry ON relay_sessions (expires_at_unix_ms);
