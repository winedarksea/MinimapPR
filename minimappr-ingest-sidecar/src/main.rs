use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use http_body_util::BodyExt;
use serde::{Deserialize, Serialize};
use tokio::{fs, io::AsyncWriteExt};
use tower_http::trace::TraceLayer;
use tracing::{info, warn};
use uuid::Uuid;

const BINARY_ENDPOINT: &str = "/api/v1/ingest/binary";
const STORE_FORWARD_ENDPOINT: &str = "/api/v1/ingest/store-forward";

#[derive(Parser, Debug)]
#[command(author, version, about = "MinimapPR firmware ingest spool sidecar")]
struct Args {
    #[arg(long, env = "MINIMAPPR_SIDECAR_HOST", default_value = "0.0.0.0")]
    host: String,

    #[arg(long, env = "MINIMAPPR_SIDECAR_PORT", default_value_t = 8081)]
    port: u16,

    #[arg(long, env = "MINIMAPPR_INGEST_SPOOL_DIR", default_value = "data/spool")]
    spool_dir: PathBuf,

    #[arg(long, env = "MINIMAPPR_SIDECAR_MAX_BODY_BYTES", default_value_t = 33_554_432)]
    max_body_bytes: usize,
}

#[derive(Clone, Debug)]
struct AppState {
    spool_dir: Arc<PathBuf>,
    max_body_bytes: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct SpoolManifest {
    spool_id: String,
    endpoint: String,
    content_type: Option<String>,
    received_ns: u128,
    body_filename: String,
}

#[derive(Debug, Serialize)]
struct EnqueueResponse {
    accepted: bool,
    queued: bool,
    spool_id: String,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    accepted: bool,
    queued: bool,
    detail: String,
}

#[derive(Debug)]
struct BodyTooLargeError {
    limit_bytes: usize,
}

impl std::fmt::Display for BodyTooLargeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "request body exceeds maximum of {} bytes", self.limit_bytes)
    }
}

impl std::error::Error for BodyTooLargeError {}

#[derive(Debug)]
struct ClientDisconnectError;

impl std::fmt::Display for ClientDisconnectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "client disconnected before request body was fully received")
    }
}

impl std::error::Error for ClientDisconnectError {}

/// Heuristically detect whether an error represents a client-side disconnect
/// (connection reset, broken pipe, etc.) by walking the error message chain.
fn is_client_disconnect(err: &dyn std::error::Error) -> bool {
    let mut cursor: Option<&dyn std::error::Error> = Some(err);
    while let Some(e) = cursor {
        let desc = e.to_string().to_lowercase();
        if desc.contains("connection reset")
            || desc.contains("broken pipe")
            || desc.contains("connection aborted")
            || desc.contains("incomplete message")
        {
            return true;
        }
        cursor = e.source();
    }
    false
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let state = AppState {
        spool_dir: Arc::new(args.spool_dir),
        max_body_bytes: args.max_body_bytes,
    };
    ensure_spool_dirs(&state.spool_dir).await?;

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    info!(%addr, spool_dir = %state.spool_dir.display(), max_body_bytes = args.max_body_bytes, "starting ingest sidecar");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app(state))
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

fn app(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route(BINARY_ENDPOINT, post(ingest_binary))
        .route(STORE_FORWARD_ENDPOINT, post(ingest_store_forward))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut sigterm = signal(SignalKind::terminate()).expect("failed to register SIGTERM handler");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {},
        _ = sigterm.recv() => {},
    }
    info!("shutdown signal received; draining in-flight requests");
}

async fn healthz() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

async fn ingest_binary(State(state): State<AppState>, headers: HeaderMap, body: Body) -> Response {
    enqueue_request(state, BINARY_ENDPOINT, headers, body).await
}

async fn ingest_store_forward(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    enqueue_request(state, STORE_FORWARD_ENDPOINT, headers, body).await
}

async fn enqueue_request(
    state: AppState,
    endpoint: &'static str,
    headers: HeaderMap,
    body: Body,
) -> Response {
    match spool_body(&state.spool_dir, state.max_body_bytes, endpoint, headers, body).await {
        Ok(spool_id) => (
            StatusCode::ACCEPTED,
            Json(EnqueueResponse {
                accepted: true,
                queued: true,
                spool_id,
            }),
        )
            .into_response(),
        Err(err) => {
            if err.downcast_ref::<BodyTooLargeError>().is_some() {
                return (
                    StatusCode::PAYLOAD_TOO_LARGE,
                    Json(ErrorResponse {
                        accepted: false,
                        queued: false,
                        detail: err.to_string(),
                    }),
                )
                    .into_response();
            }
            if err.downcast_ref::<ClientDisconnectError>().is_some() {
                // Client closed the connection before the body was fully received;
                // the payload was not committed to the spool.  This is expected
                // when firmware uses a short HTTP timeout, so don't log a warning.
                return (
                    StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_REQUEST),
                    Json(ErrorResponse {
                        accepted: false,
                        queued: false,
                        detail: err.to_string(),
                    }),
                )
                    .into_response();
            }
            warn!(endpoint, error = %err, "failed to enqueue ingest request");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    accepted: false,
                    queued: false,
                    detail: "failed to durably enqueue ingest payload".to_string(),
                }),
            )
                .into_response()
        }
    }
}

async fn spool_body(
    spool_dir: &Path,
    max_body_bytes: usize,
    endpoint: &'static str,
    headers: HeaderMap,
    body: Body,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    // Reject early if Content-Length already declares an oversized body.
    if let Some(cl_value) = headers.get(axum::http::header::CONTENT_LENGTH) {
        if let Ok(cl_str) = cl_value.to_str() {
            if let Ok(cl_len) = cl_str.parse::<usize>() {
                if cl_len > max_body_bytes {
                    return Err(Box::new(BodyTooLargeError { limit_bytes: max_body_bytes }));
                }
            }
        }
    }

    let received_ns = now_ns()?;
    let spool_id = format!("{}-{}", received_ns, Uuid::new_v4());
    let body_filename = format!("{}.body", spool_id);
    let manifest_filename = format!("{}.json", spool_id);

    let tmp_dir = spool_dir.join("tmp");
    let ready_dir = spool_dir.join("ready");
    let tmp_body_path = tmp_dir.join(format!("{}.upload", body_filename));
    let tmp_manifest_path = tmp_dir.join(format!("{}.upload", manifest_filename));
    let ready_body_path = ready_dir.join(&body_filename);
    let ready_manifest_path = ready_dir.join(&manifest_filename);

    let result = spool_body_inner(
        &tmp_body_path,
        &tmp_manifest_path,
        &ready_body_path,
        &ready_manifest_path,
        &ready_dir,
        max_body_bytes,
        endpoint,
        headers,
        body,
        spool_id,
        received_ns,
    )
    .await;

    if result.is_err() {
        // Best-effort cleanup of any tmp-phase files left behind by the failed
        // attempt.  Files already promoted into ready/ are intentionally kept.
        let _ = fs::remove_file(&tmp_body_path).await;
        let _ = fs::remove_file(&tmp_manifest_path).await;
    }
    result
}

async fn spool_body_inner(
    tmp_body_path: &Path,
    tmp_manifest_path: &Path,
    ready_body_path: &Path,
    ready_manifest_path: &Path,
    ready_dir: &Path,
    max_body_bytes: usize,
    endpoint: &str,
    headers: HeaderMap,
    mut body: Body,
    spool_id: String,
    received_ns: u128,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let mut bytes_written: usize = 0;
    let mut file = fs::File::create(tmp_body_path).await?;
    while let Some(frame_result) = body.frame().await {
        let frame = match frame_result {
            Ok(f) => f,
            Err(e) => {
                let boxed: Box<dyn std::error::Error + Send + Sync> = Box::new(e);
                if is_client_disconnect(boxed.as_ref()) {
                    return Err(Box::new(ClientDisconnectError));
                }
                return Err(boxed);
            }
        };
        if let Some(chunk) = frame.data_ref() {
            bytes_written += chunk.len();
            if bytes_written > max_body_bytes {
                return Err(Box::new(BodyTooLargeError { limit_bytes: max_body_bytes }));
            }
            file.write_all(chunk).await?;
        }
    }
    file.flush().await?;
    file.sync_all().await?;
    drop(file);

    let content_type = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let body_filename = format!("{}.body", spool_id);
    let manifest = SpoolManifest {
        spool_id: spool_id.clone(),
        endpoint: endpoint.to_string(),
        content_type,
        received_ns,
        body_filename,
    };
    let manifest_bytes = serde_json::to_vec(&manifest)?;
    let mut manifest_file = fs::File::create(tmp_manifest_path).await?;
    manifest_file.write_all(&manifest_bytes).await?;
    manifest_file.write_all(b"\n").await?;
    manifest_file.flush().await?;
    manifest_file.sync_all().await?;
    drop(manifest_file);

    fs::rename(tmp_body_path, ready_body_path).await?;
    // Publish the manifest last. Python only scans manifests, so this is the queue commit point.
    // fsync the body into ready/ before touching the manifest; roll back on failure.
    if let Err(err) = fsync_dir(ready_dir).await {
        let _ = fs::remove_file(ready_body_path).await;
        return Err(Box::new(err));
    }
    if let Err(err) = fs::rename(tmp_manifest_path, ready_manifest_path).await {
        let _ = fs::remove_file(ready_body_path).await;
        return Err(Box::new(err));
    }
    fsync_dir(ready_dir).await?;

    Ok(spool_id)
}

async fn ensure_spool_dirs(spool_dir: &Path) -> std::io::Result<()> {
    fs::create_dir_all(spool_dir.join("tmp")).await?;
    fs::create_dir_all(spool_dir.join("ready")).await?;
    fs::create_dir_all(spool_dir.join("processing")).await?;
    fs::create_dir_all(spool_dir.join("failed")).await?;
    Ok(())
}

async fn fsync_dir(path: &Path) -> std::io::Result<()> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || std::fs::File::open(path).and_then(|file| file.sync_all()))
        .await
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?
}

fn now_ns() -> Result<u128, std::time::SystemTimeError> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    #[tokio::test]
    async fn returns_accepted_after_manifest_is_ready() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState {
            spool_dir: Arc::new(tmp.path().join("spool")),
            max_body_bytes: usize::MAX,
        };
        ensure_spool_dirs(&state.spool_dir).await.unwrap();
        let response = app(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(BINARY_ENDPOINT)
                    .header("content-type", "application/octet-stream")
                    .body(Body::from("payload"))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::ACCEPTED);
        assert_eq!(state.spool_dir.join("ready").read_dir().unwrap().count(), 2);
        assert_eq!(state.spool_dir.join("tmp").read_dir().unwrap().count(), 0);
    }

    #[tokio::test]
    async fn concurrent_uploads_create_unique_items() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState {
            spool_dir: Arc::new(tmp.path().join("spool")),
            max_body_bytes: usize::MAX,
        };
        ensure_spool_dirs(&state.spool_dir).await.unwrap();
        let router = app(state.clone());
        let first = router.clone().oneshot(
            Request::builder()
                .method("POST")
                .uri(STORE_FORWARD_ENDPOINT)
                .body(Body::from("{}"))
                .unwrap(),
        );
        let second = router.oneshot(
            Request::builder()
                .method("POST")
                .uri(STORE_FORWARD_ENDPOINT)
                .body(Body::from("{}"))
                .unwrap(),
        );

        let (first_response, second_response) = tokio::join!(first, second);
        assert_eq!(first_response.unwrap().status(), StatusCode::ACCEPTED);
        assert_eq!(second_response.unwrap().status(), StatusCode::ACCEPTED);
        assert_eq!(state.spool_dir.join("ready").read_dir().unwrap().count(), 4);
    }

    #[tokio::test]
    async fn oversized_body_returns_413() {
        let tmp = tempfile::tempdir().unwrap();
        let state = AppState {
            spool_dir: Arc::new(tmp.path().join("spool")),
            max_body_bytes: 8,
        };
        ensure_spool_dirs(&state.spool_dir).await.unwrap();
        // Limit to 8 bytes; send 16 bytes. Content-Length triggers early rejection.
        let response = app(state)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(BINARY_ENDPOINT)
                    .header("content-type", "application/octet-stream")
                    .header("content-length", "16")
                    .body(Body::from("0123456789abcdef"))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }
}
