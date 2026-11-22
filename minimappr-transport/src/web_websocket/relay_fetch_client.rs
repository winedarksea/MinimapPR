//! Typed client for the relay Worker's HTTP API over the browser `fetch`.
//!
//! Mirrors `native_websocket::RelayHttpClient`. The Worker sends no CORS
//! headers, so the page must be same-origin with the relay (production
//! serves the PWA from the same Worker; local dev proxies `/v1/*`).

use minimappr_relay_wire::relay_http_api::{
    issue_join_code_path, CreateRelaySessionRequest, CreateRelaySessionResponse,
    IssuedJoinCodeLookup, ResolveJoinCodeRequest, ResolveJoinCodeResponse, CREATE_SESSION_PATH,
    HEALTH_PATH, RESOLVE_JOIN_CODE_PATH,
};
use minimappr_relay_wire::RelayRole;
use serde::de::DeserializeOwned;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

use super::{describe_javascript_value, WebRelayError};

const JSON_CONTENT_TYPE: &str = "application/json";

pub struct RelayFetchClient {
    http_base_url: String,
}

impl RelayFetchClient {
    /// `http_base_url` is the relay origin; `""` means the page's own origin.
    pub fn new(http_base_url: &str) -> Self {
        Self {
            http_base_url: http_base_url.trim_end_matches('/').to_string(),
        }
    }

    pub fn http_base_url(&self) -> &str {
        &self.http_base_url
    }

    pub async fn health(&self) -> Result<(), WebRelayError> {
        self.send("GET", HEALTH_PATH, None, None).await.map(|_| ())
    }

    /// Host: registers a relayed session (and optionally a join code).
    pub async fn create_session(
        &self,
        request: &CreateRelaySessionRequest,
    ) -> Result<CreateRelaySessionResponse, WebRelayError> {
        let body = serde_json::to_string(request).map_err(|error| WebRelayError::Http(error.to_string()))?;
        decode_json(&self.send("POST", CREATE_SESSION_PATH, Some(&body), None).await?)
    }

    /// Host: a fresh lookup code (e.g. after the old one used up its guesses).
    pub async fn issue_join_code(
        &self,
        relay_session_name: &str,
        host_token: &str,
    ) -> Result<IssuedJoinCodeLookup, WebRelayError> {
        let path = issue_join_code_path(relay_session_name);
        decode_json(&self.send("POST", &path, None, Some(host_token)).await?)
    }

    /// Joiner: trades the *lookup* half of a join code for a role token.
    /// The secret half is never an argument here.
    pub async fn resolve_join_code(
        &self,
        lookup_code: &str,
        requested_role: RelayRole,
    ) -> Result<ResolveJoinCodeResponse, WebRelayError> {
        let request = ResolveJoinCodeRequest {
            lookup_code: lookup_code.to_string(),
            requested_role,
        };
        let body = serde_json::to_string(&request).map_err(|error| WebRelayError::Http(error.to_string()))?;
        decode_json(&self.send("POST", RESOLVE_JOIN_CODE_PATH, Some(&body), None).await?)
    }

    /// One request; returns the body text of a 2xx response.
    async fn send(
        &self,
        method: &str,
        path: &str,
        json_body: Option<&str>,
        bearer_token: Option<&str>,
    ) -> Result<String, WebRelayError> {
        let http_error = |value: JsValue| WebRelayError::Http(describe_javascript_value(&value));
        let headers = web_sys::Headers::new().map_err(http_error)?;
        let init = web_sys::RequestInit::new();
        init.set_method(method);
        if let Some(body) = json_body {
            headers.set("content-type", JSON_CONTENT_TYPE).map_err(http_error)?;
            init.set_body(&JsValue::from_str(body));
        }
        if let Some(token) = bearer_token {
            headers.set("authorization", &format!("Bearer {token}")).map_err(http_error)?;
        }
        init.set_headers(&headers);
        let url = format!("{}{path}", self.http_base_url);
        let request = web_sys::Request::new_with_str_and_init(&url, &init).map_err(http_error)?;
        let response_value = JsFuture::from(global_fetch(&request)?).await.map_err(http_error)?;
        let response: web_sys::Response = response_value.dyn_into().map_err(http_error)?;
        let text_promise = response.text().map_err(http_error)?;
        let body = JsFuture::from(text_promise)
            .await
            .map_err(http_error)?
            .as_string()
            .unwrap_or_default();
        if !response.ok() {
            return Err(WebRelayError::Status {
                status_code: response.status(),
                body,
            });
        }
        Ok(body)
    }
}

/// `fetch` from a page (`Window`) or a worker (`WorkerGlobalScope`).
fn global_fetch(request: &web_sys::Request) -> Result<js_sys::Promise, WebRelayError> {
    let global = js_sys::global();
    if let Some(window) = global.dyn_ref::<web_sys::Window>() {
        return Ok(window.fetch_with_request(request));
    }
    if let Some(worker) = global.dyn_ref::<web_sys::WorkerGlobalScope>() {
        return Ok(worker.fetch_with_request(request));
    }
    Err(WebRelayError::Http("no fetch on this global".to_string()))
}

fn decode_json<T: DeserializeOwned>(body: &str) -> Result<T, WebRelayError> {
    serde_json::from_str(body).map_err(|error| WebRelayError::Http(format!("bad JSON from relay: {error}")))
}
