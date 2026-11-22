//! Typed client for the relay Worker's HTTP API (`minimappr_relay_wire::relay_http_api`).

use minimappr_relay_wire::relay_http_api::{
    issue_join_code_path, revoke_join_code_path, CreateRelaySessionRequest,
    CreateRelaySessionResponse, IssuedJoinCodeLookup, ResolveJoinCodeRequest,
    ResolveJoinCodeResponse, CREATE_SESSION_PATH, HEALTH_PATH, RESOLVE_JOIN_CODE_PATH,
};
use minimappr_relay_wire::RelayRole;
use serde::de::DeserializeOwned;

use super::NativeRelayError;

pub struct RelayHttpClient {
    http_base_url: String,
    http: reqwest::Client,
}

impl RelayHttpClient {
    /// `http_base_url` like `https://relay.example` (no trailing slash needed).
    pub fn new(http_base_url: &str) -> Self {
        Self {
            http_base_url: http_base_url.trim_end_matches('/').to_string(),
            http: reqwest::Client::new(),
        }
    }

    pub fn http_base_url(&self) -> &str {
        &self.http_base_url
    }

    pub async fn health(&self) -> Result<(), NativeRelayError> {
        let response = self
            .http
            .get(format!("{}{HEALTH_PATH}", self.http_base_url))
            .send()
            .await
            .map_err(|error| NativeRelayError::Http(error.to_string()))?;
        expect_success(response).await.map(|_| ())
    }

    /// Host: registers a relayed session (and optionally a join code).
    pub async fn create_session(
        &self,
        request: &CreateRelaySessionRequest,
    ) -> Result<CreateRelaySessionResponse, NativeRelayError> {
        let response = self
            .http
            .post(format!("{}{CREATE_SESSION_PATH}", self.http_base_url))
            .json(request)
            .send()
            .await
            .map_err(|error| NativeRelayError::Http(error.to_string()))?;
        decode_json(response).await
    }

    /// Host: a fresh lookup code (e.g. after the old one used up its guesses).
    pub async fn issue_join_code(
        &self,
        relay_session_name: &str,
        host_token: &str,
    ) -> Result<IssuedJoinCodeLookup, NativeRelayError> {
        let response = self
            .http
            .post(format!(
                "{}{}",
                self.http_base_url,
                issue_join_code_path(relay_session_name)
            ))
            .bearer_auth(host_token)
            .send()
            .await
            .map_err(|error| NativeRelayError::Http(error.to_string()))?;
        decode_json(response).await
    }

    pub async fn revoke_join_code(
        &self,
        lookup_code: &str,
        host_token: &str,
    ) -> Result<(), NativeRelayError> {
        let response = self
            .http
            .delete(format!(
                "{}{}",
                self.http_base_url,
                revoke_join_code_path(lookup_code)
            ))
            .bearer_auth(host_token)
            .send()
            .await
            .map_err(|error| NativeRelayError::Http(error.to_string()))?;
        expect_success(response).await.map(|_| ())
    }

    /// Joiner: trades the *lookup* half of a join code for a role token.
    /// The secret half is never an argument here.
    pub async fn resolve_join_code(
        &self,
        lookup_code: &str,
        requested_role: RelayRole,
    ) -> Result<ResolveJoinCodeResponse, NativeRelayError> {
        let response = self
            .http
            .post(format!("{}{RESOLVE_JOIN_CODE_PATH}", self.http_base_url))
            .json(&ResolveJoinCodeRequest {
                lookup_code: lookup_code.to_string(),
                requested_role,
            })
            .send()
            .await
            .map_err(|error| NativeRelayError::Http(error.to_string()))?;
        decode_json(response).await
    }
}

async fn expect_success(response: reqwest::Response) -> Result<reqwest::Response, NativeRelayError> {
    if response.status().is_success() {
        return Ok(response);
    }
    let status_code = response.status().as_u16();
    let body = response.text().await.unwrap_or_default();
    Err(NativeRelayError::Status { status_code, body })
}

async fn decode_json<T: DeserializeOwned>(response: reqwest::Response) -> Result<T, NativeRelayError> {
    expect_success(response)
        .await?
        .json::<T>()
        .await
        .map_err(|error| NativeRelayError::Http(error.to_string()))
}
