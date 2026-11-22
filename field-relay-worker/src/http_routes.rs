//! The Worker's HTTP API (bodies in `minimappr_relay_wire::relay_http_api`).

use minimappr_relay_wire::join_code_lookup::normalize_lookup_code;
use minimappr_relay_wire::relay_http_api::{
    decode_hex, CreateRelaySessionRequest, CreateRelaySessionResponse, IssuedJoinCodeLookup,
    RelayApiErrorBody, ResolveJoinCodeRequest, ResolveJoinCodeResponse,
};
use minimappr_relay_wire::RelayRole;
use worker::*;

use crate::join_code_store::*;
use crate::random_tokens::*;
use crate::relay_object_state::*;
use crate::worker_config::*;

/// Field session ids are 128-bit (`minimappr_core::protocol::SessionId`).
const FIELD_SESSION_ID_BYTES: usize = 16;

pub fn json_error(message: &str, status: u16) -> Result<Response> {
    Ok(Response::from_json(&RelayApiErrorBody { error: message.to_string() })?.with_status(status))
}

fn now_unix_milliseconds() -> u64 {
    js_sys::Date::now() as u64
}

fn relay_object_stub(env: &Env, relay_session_name: &str) -> Result<Stub> {
    env.durable_object(SESSION_RELAY_OBJECT_BINDING)?
        .id_from_name(relay_session_name)?
        .get_stub()
}

async fn post_to_relay_object<B: serde::Serialize>(stub: &Stub, path: &str, body: &B) -> Result<Response> {
    let headers = Headers::new();
    headers.set("Content-Type", "application/json")?;
    let mut init = RequestInit::new();
    init.with_method(Method::Post)
        .with_headers(headers)
        .with_body(Some(serde_json::to_string(body)?.into()));
    stub.fetch_with_request(Request::new_with_init(&format!("{INTERNAL_REQUEST_ORIGIN}{path}"), &init)?)
        .await
}

fn bearer_token(request: &Request) -> Result<Option<String>> {
    Ok(request
        .headers()
        .get("Authorization")?
        .and_then(|value| value.strip_prefix("Bearer ").map(str::to_string)))
}

/// `POST /v1/sessions`
pub async fn create_session(mut request: Request, env: &Env) -> Result<Response> {
    let body: CreateRelaySessionRequest = match request.json().await {
        Ok(body) => body,
        Err(_) => return json_error("malformed body", 400),
    };
    let field_session_id_hex = body.field_session_id_hex.to_ascii_lowercase();
    if decode_hex(&field_session_id_hex).map(|bytes| bytes.len()) != Some(FIELD_SESSION_ID_BYTES) {
        return json_error("field_session_id_hex must be 32 hex digits", 400);
    }
    let config = RelayWorkerConfig::from_env(env)?;
    let now = now_unix_milliseconds();
    let session_expires_at = now + config.session_time_to_live_seconds * MILLISECONDS_PER_SECOND;
    let relay_session_name = mint_relay_session_name()?;
    let host_token = mint_secret_token()?;
    let node_token = mint_secret_token()?;
    let viewer_token = mint_secret_token()?;

    let stub = relay_object_stub(env, &relay_session_name)?;
    let initialize = InitializeRelayObjectRequest {
        expires_at_unix_ms: session_expires_at,
        role_token_digests: vec![
            (sha256_hex(&host_token), RelayRole::Host),
            (sha256_hex(&node_token), RelayRole::Node),
            (sha256_hex(&viewer_token), RelayRole::Viewer),
        ],
    };
    let initialized = post_to_relay_object(&stub, INTERNAL_INITIALIZE_PATH, &initialize).await?;
    if initialized.status_code() != 200 {
        return json_error("relay object initialization failed", 500);
    }
    let database = env.d1(JOIN_CODES_DATABASE_BINDING)?;
    insert_relay_session(
        &database,
        &relay_session_name,
        &sha256_hex(&host_token),
        &field_session_id_hex,
        now,
        session_expires_at,
    )
    .await?;
    let join_code = if body.issue_join_code {
        Some(issue_code(&database, &config, &relay_session_name, &field_session_id_hex, session_expires_at).await?)
    } else {
        None
    };
    Response::from_json(&CreateRelaySessionResponse {
        relay_session_name,
        host_token,
        node_token,
        viewer_token,
        session_expires_at_unix_ms: session_expires_at,
        join_code,
    })
}

async fn issue_code(
    database: &D1Database,
    config: &RelayWorkerConfig,
    relay_session_name: &str,
    field_session_id_hex: &str,
    session_expires_at: u64,
) -> Result<IssuedJoinCodeLookup> {
    let now = now_unix_milliseconds();
    let expires_at_unix_ms =
        (now + config.join_code_time_to_live_seconds * MILLISECONDS_PER_SECOND).min(session_expires_at);
    let lookup_code =
        allocate_join_code(database, relay_session_name, field_session_id_hex, now, expires_at_unix_ms)
            .await?
            .ok_or_else(|| Error::RustError("join code space congested".into()))?;
    Ok(IssuedJoinCodeLookup { lookup_code, expires_at_unix_ms })
}

async fn authorize_host(database: &D1Database, relay_session_name: &str, request: &Request) -> Result<Option<Response>> {
    let Some(session) = find_relay_session(database, relay_session_name).await? else {
        return Ok(Some(json_error("unknown relay session", 404)?));
    };
    let presented = bearer_token(request)?.unwrap_or_default();
    if !digests_match(&sha256_hex(&presented), &session.host_token_sha256_hex) {
        return Ok(Some(json_error("host token required", 403)?));
    }
    if session.expires_at_unix_ms as u64 <= now_unix_milliseconds() {
        return Ok(Some(json_error("relay session expired", 410)?));
    }
    Ok(None)
}

/// `POST /v1/sessions/{relay_session_name}/join-codes` (host): a new lookup code.
pub async fn issue_join_code(request: Request, env: &Env, relay_session_name: &str) -> Result<Response> {
    let database = env.d1(JOIN_CODES_DATABASE_BINDING)?;
    if let Some(refusal) = authorize_host(&database, relay_session_name, &request).await? {
        return Ok(refusal);
    }
    let config = RelayWorkerConfig::from_env(env)?;
    let session = find_relay_session(&database, relay_session_name).await?.expect("authorized above");
    let issued = issue_code(
        &database,
        &config,
        relay_session_name,
        &session.field_session_id_hex,
        session.expires_at_unix_ms as u64,
    )
    .await?;
    Response::from_json(&issued)
}

/// `DELETE /v1/join-codes/{lookup_code}` (host of that code's session).
pub async fn revoke_join_code(request: Request, env: &Env, lookup_code: &str) -> Result<Response> {
    let database = env.d1(JOIN_CODES_DATABASE_BINDING)?;
    let Some(lookup_code) = normalize_lookup_code(lookup_code, JOIN_CODE_LOOKUP_LENGTH) else {
        return json_error("unknown join code", 404);
    };
    let Some(code) = find_join_code(&database, &lookup_code).await? else {
        return json_error("unknown join code", 404);
    };
    if let Some(refusal) = authorize_host(&database, &code.relay_session_name, &request).await? {
        return Ok(refusal);
    }
    delete_join_code(&database, &lookup_code).await?;
    Ok(Response::empty()?.with_status(204))
}

/// `POST /v1/join`: lookup code -> fresh per-joiner role token. 404 for an
/// unknown (or already cleaned-up) code, 410 for an expired one still
/// awaiting cron cleanup, 429 when the code's resolve budget is spent.
pub async fn resolve_join_code(mut request: Request, env: &Env) -> Result<Response> {
    let body: ResolveJoinCodeRequest = match request.json().await {
        Ok(body) => body,
        Err(_) => return json_error("malformed body", 400),
    };
    if body.requested_role == RelayRole::Host {
        return json_error("join codes never grant the host role", 403);
    }
    let Some(lookup_code) = normalize_lookup_code(&body.lookup_code, JOIN_CODE_LOOKUP_LENGTH) else {
        return json_error("unknown join code", 404);
    };
    let database = env.d1(JOIN_CODES_DATABASE_BINDING)?;
    let Some(code) = find_join_code(&database, &lookup_code).await? else {
        return json_error("unknown join code", 404);
    };
    if code.expires_at_unix_ms as u64 <= now_unix_milliseconds() {
        return json_error("join code expired", 410);
    }
    let config = RelayWorkerConfig::from_env(env)?;
    if code.resolve_count as u64 >= config.join_code_maximum_resolves
        || !claim_join_code_resolve(&database, &lookup_code, config.join_code_maximum_resolves).await?
    {
        return json_error("join code resolve budget spent", 429);
    }
    let stub = relay_object_stub(env, &code.relay_session_name)?;
    let mut minted = post_to_relay_object(
        &stub,
        INTERNAL_MINT_ROLE_TOKEN_PATH,
        &MintRoleTokenRequest { role: body.requested_role },
    )
    .await?;
    if minted.status_code() != 200 {
        return json_error("relay session unavailable", minted.status_code());
    }
    let minted: MintRoleTokenResponse = minted.json().await?;
    Response::from_json(&ResolveJoinCodeResponse {
        relay_session_name: code.relay_session_name,
        field_session_id_hex: code.field_session_id_hex,
        role_token: minted.role_token,
        role: body.requested_role,
    })
}

/// `GET /v1/relay/{relay_session_name}/connect`: hand the upgrade to the object.
pub async fn forward_connect(request: Request, env: &Env, relay_session_name: &str) -> Result<Response> {
    if !is_relay_session_name(relay_session_name) {
        return json_error("unknown relay session", 404);
    }
    relay_object_stub(env, relay_session_name)?.fetch_with_request(request).await
}
