//! Field Mode relay Worker.
//!
//! - HTTP API (`http_routes`): register a relayed session, mint/resolve/revoke
//!   join-code lookups (D1), hand WebSocket upgrades to the session's object.
//! - `SessionRelayObject`: one hibernating Durable Object per session that
//!   forwards opaque sealed frames under per-session caps.
//! - Cron (`scheduled`): deletes expired join codes and sessions from D1.
//!
//! The relay is untrusted by design: it never receives a session key or the
//! secret half of a join code. See docs/field_mode_progress/relay.md.

mod http_routes;
mod join_code_store;
mod random_tokens;
mod relay_object_state;
mod session_relay_object;
mod worker_config;

use minimappr_relay_wire::relay_http_api::{CREATE_SESSION_PATH, HEALTH_PATH, RESOLVE_JOIN_CODE_PATH};
use worker::*;

pub use session_relay_object::SessionRelayObject;

#[event(fetch)]
pub async fn fetch(request: Request, env: Env, _context: Context) -> Result<Response> {
    console_error_panic_hook::set_once();
    let path = request.path();
    let segments: Vec<&str> = path.trim_matches('/').split('/').collect();
    let method = request.method();
    match (method, segments.as_slice()) {
        (Method::Get, _) if path == HEALTH_PATH => Response::ok("ok"),
        (Method::Post, _) if path == CREATE_SESSION_PATH => http_routes::create_session(request, &env).await,
        (Method::Post, _) if path == RESOLVE_JOIN_CODE_PATH => {
            http_routes::resolve_join_code(request, &env).await
        }
        (Method::Post, ["v1", "sessions", relay_session_name, "join-codes"]) => {
            let relay_session_name = relay_session_name.to_string();
            http_routes::issue_join_code(request, &env, &relay_session_name).await
        }
        (Method::Delete, ["v1", "join-codes", lookup_code]) => {
            let lookup_code = lookup_code.to_string();
            http_routes::revoke_join_code(request, &env, &lookup_code).await
        }
        (Method::Get, ["v1", "relay", relay_session_name, "connect"]) => {
            let relay_session_name = relay_session_name.to_string();
            http_routes::forward_connect(request, &env, &relay_session_name).await
        }
        _ => http_routes::json_error("not found", 404),
    }
}

#[event(scheduled)]
pub async fn scheduled(_event: ScheduledEvent, env: Env, _context: ScheduleContext) {
    console_error_panic_hook::set_once();
    let cleanup = async {
        let database = env.d1(worker_config::JOIN_CODES_DATABASE_BINDING)?;
        join_code_store::delete_expired_rows(&database, js_sys::Date::now() as u64).await
    };
    if let Err(error) = cleanup.await {
        console_error!("join code cleanup failed: {error}");
    }
}
