mod audio_payload;
mod birdnet_render;
mod derived_cache;
mod dsp;
mod dsp_render_output;
mod dsp_worker;
mod envelope;
mod gcc_phat;
mod ingest_backend;
mod journal_reader;
mod leases;
mod manifests;
mod srp_phat;
mod storage_class;

use std::net::SocketAddr;
use std::path::PathBuf;

use axum::{
    body::Body,
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use derived_cache::DerivedCache;
use dsp_worker::{DspWorker, DspWorkerConfig, SharedDspState};
use ingest_backend::{
    BodyTooLargeError, ClientDisconnectError, IngestBackend, IngestStorageMode,
    InvalidIngestEnvelopeError, JournalCapacityExceededError, JournalRuntimeConfig,
};
use leases::PinLeaseRequest;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::trace::TraceLayer;
use tracing::{info, warn};

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

    #[arg(
        long,
        env = "MINIMAPPR_SIDECAR_MAX_BODY_BYTES",
        default_value_t = 33_554_432
    )]
    max_body_bytes: usize,

    #[arg(long, env = "MINIMAPPR_SIDECAR_STORAGE_MODE", value_enum, default_value_t = IngestStorageMode::Journal)]
    storage_mode: IngestStorageMode,

    #[arg(
        long,
        env = "MINIMAPPR_SIDECAR_SEGMENT_MAX_BYTES",
        default_value_t = 8_388_608
    )]
    segment_max_bytes: u64,

    #[arg(
        long,
        env = "MINIMAPPR_INGEST_CONSUMER_NAME",
        default_value = "python-ingest"
    )]
    consumer_name: String,

    #[arg(
        long,
        env = "MINIMAPPR_SIDECAR_TOTAL_JOURNAL_BUDGET_BYTES",
        default_value_t = 268_435_456
    )]
    total_journal_budget_bytes: u64,

    #[arg(
        long,
        env = "MINIMAPPR_SIDECAR_ADMISSION_RESERVE_BYTES",
        default_value_t = 16_777_216
    )]
    admission_reserve_bytes: u64,

    #[arg(
        long,
        env = "MINIMAPPR_SIDECAR_ALLOW_NON_TMPFS_JOURNAL",
        default_value_t = false
    )]
    allow_non_tmpfs_journal: bool,

    #[arg(
        long,
        env = "MINIMAPPR_SIDECAR_DERIVED_CACHE_BUDGET_BYTES",
        default_value_t = 67_108_864
    )]
    derived_cache_budget_bytes: u64,

    #[arg(
        long,
        env = "MINIMAPPR_SIDECAR_DERIVED_CACHE_ADMISSION_RESERVE_BYTES",
        default_value_t = 8_388_608
    )]
    derived_cache_admission_reserve_bytes: u64,

    #[arg(long, env = "MINIMAPPR_RUNTIME_PROFILE", default_value = "default")]
    runtime_profile: String,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_BAND_MIN_HZ",
        default_value_t = 300.0
    )]
    localization_band_min_hz: f32,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_BAND_MAX_HZ",
        default_value_t = 3500.0
    )]
    localization_band_max_hz: f32,

    #[arg(
        long,
        env = "MINIMAPPR_BIRDNET_SPATIAL_BLEND_MIN_HZ",
        default_value_t = 1000.0
    )]
    birdnet_spatial_blend_min_hz: f32,

    #[arg(
        long,
        env = "MINIMAPPR_BIRDNET_SPATIAL_BLEND_MAX_HZ",
        default_value_t = 3400.0
    )]
    birdnet_spatial_blend_max_hz: f32,

    #[arg(
        long,
        env = "MINIMAPPR_BIRDNET_PRE_BLEND_HIGHPASS_HZ",
        default_value_t = 100.0
    )]
    birdnet_pre_blend_highpass_hz: f32,
}

#[derive(Clone, Debug)]
struct AppState {
    backend: IngestBackend,
    max_body_bytes: usize,
    dsp_state: SharedDspState,
    derived_cache: Option<DerivedCache>,
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let storage_dir = args.spool_dir.clone();
    let storage_mode = args.storage_mode;
    let journal_runtime_config = JournalRuntimeConfig {
        consumer_name: args.consumer_name.clone(),
        total_journal_budget_bytes: args.total_journal_budget_bytes,
        admission_reserve_bytes: args.admission_reserve_bytes,
        enforce_tmpfs: args.storage_mode == IngestStorageMode::Journal
            && !args.allow_non_tmpfs_journal,
        derived_cache_budget_bytes: args.derived_cache_budget_bytes,
        derived_cache_admission_reserve_bytes: args.derived_cache_admission_reserve_bytes,
    };
    let backend = IngestBackend::open(
        args.spool_dir,
        args.storage_mode,
        args.segment_max_bytes,
        journal_runtime_config,
    )
    .await?;

    let dsp_state: SharedDspState = Arc::new(Mutex::new(Default::default()));

    // Spawn DSP worker if journal mode has a manifest store and derived cache.
    if let (Some(manifest_store), Some(derived_cache)) =
        (backend.manifest_store(), backend.derived_cache())
    {
        let worker = DspWorker::new(
            manifest_store,
            derived_cache,
            DspWorkerConfig {
                birdnet_hybrid_render_enabled: args.runtime_profile == "birdnet_hybrid_production",
                localization_band_hz: [
                    args.localization_band_min_hz,
                    args.localization_band_max_hz,
                ],
                spatial_blend_band_hz: [
                    args.birdnet_spatial_blend_min_hz,
                    args.birdnet_spatial_blend_max_hz,
                ],
                pre_blend_highpass_hz: args.birdnet_pre_blend_highpass_hz,
                ..DspWorkerConfig::default()
            },
            dsp_state.clone(),
        );
        tokio::spawn(worker.run_loop());
        info!("DSP worker spawned");
    }

    let state = AppState {
        derived_cache: backend.derived_cache(),
        backend,
        max_body_bytes: args.max_body_bytes,
        dsp_state,
    };

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    info!(
        %addr,
        storage_dir = %storage_dir.display(),
        storage_mode = ?storage_mode,
        max_body_bytes = args.max_body_bytes,
        total_journal_budget_bytes = args.total_journal_budget_bytes,
        admission_reserve_bytes = args.admission_reserve_bytes,
        derived_cache_budget_bytes = args.derived_cache_budget_bytes,
        "starting ingest sidecar"
    );

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
        .route("/api/v1/journal/pins", post(create_pin_lease))
        .route(
            "/api/v1/journal/pins/:lease_id",
            axum::routing::delete(release_pin_lease),
        )
        .route("/api/v1/dsp/status", get(dsp_status))
        .route("/api/v1/dsp/results", get(dsp_results))
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

async fn healthz(State(state): State<AppState>) -> Response {
    match state.backend.health().await {
        Ok(health) => (
            StatusCode::OK,
            Json(serde_json::json!({"status": "ok", "backend": health})),
        )
            .into_response(),
        Err(error) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"status": "degraded", "detail": error.to_string()})),
        )
            .into_response(),
    }
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
    match state
        .backend
        .enqueue(state.max_body_bytes, endpoint, headers, body)
        .await
    {
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
            if err.downcast_ref::<InvalidIngestEnvelopeError>().is_some() {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse {
                        accepted: false,
                        queued: false,
                        detail: err.to_string(),
                    }),
                )
                    .into_response();
            }
            if err.downcast_ref::<JournalCapacityExceededError>().is_some() {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
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

async fn create_pin_lease(
    State(state): State<AppState>,
    Json(request): Json<PinLeaseRequest>,
) -> Response {
    match state.backend.create_pin_lease(request).await {
        Ok(response) => (StatusCode::CREATED, Json(response)).into_response(),
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                accepted: false,
                queued: false,
                detail: error.to_string(),
            }),
        )
            .into_response(),
    }
}

async fn release_pin_lease(
    State(state): State<AppState>,
    axum::extract::Path(lease_id): axum::extract::Path<String>,
) -> Response {
    match state.backend.release_pin_lease(&lease_id).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                accepted: false,
                queued: false,
                detail: error.to_string(),
            }),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
struct DspResultsQuery {
    limit: Option<usize>,
}

#[derive(Serialize)]
struct DspStatusResponse {
    worker_running: bool,
    last_heartbeat_ns: Option<u128>,
    last_processed_ns: Option<u128>,
    pending_manifest_count: usize,
    total_tdoa_results: u64,
    total_localization_results: u64,
    total_classifier_renders: u64,
    total_failures: u64,
    dsp_worker_running: bool,
}

async fn dsp_status(State(state): State<AppState>) -> Response {
    let st = state.dsp_state.lock().await;
    Json(DspStatusResponse {
        worker_running: st.worker_running,
        last_heartbeat_ns: st.last_heartbeat_ns,
        last_processed_ns: st.last_processed_ns,
        pending_manifest_count: st.pending_count,
        total_tdoa_results: st.total_tdoa_results,
        total_localization_results: st.total_localization_results,
        total_classifier_renders: st.total_classifier_renders,
        total_failures: st.total_failures,
        dsp_worker_running: st.worker_running,
    })
    .into_response()
}

async fn dsp_results(
    State(state): State<AppState>,
    Query(params): Query<DspResultsQuery>,
) -> Response {
    let limit = params.limit.unwrap_or(50).min(200);
    let results: Vec<_> = {
        let st = state.dsp_state.lock().await;
        st.recent_results
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    };
    if let Some(derived_cache) = state.derived_cache.as_ref() {
        let now_ns = current_unix_ns();
        for result in &results {
            if let Some(handle) = result.derived_handle.as_ref() {
                let _ = derived_cache.touch(&handle.segment_id, now_ns).await;
            }
        }
    }
    Json(results).into_response()
}

fn current_unix_ns() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use tokio::fs;
    use tower::ServiceExt;

    fn test_app_state(backend: IngestBackend, max_body_bytes: usize) -> AppState {
        AppState {
            backend,
            max_body_bytes,
            dsp_state: Arc::new(Mutex::new(Default::default())),
            derived_cache: None,
        }
    }

    fn test_journal_runtime_config() -> JournalRuntimeConfig {
        JournalRuntimeConfig {
            consumer_name: "python-ingest".to_string(),
            total_journal_budget_bytes: 268_435_456,
            admission_reserve_bytes: 16_777_216,
            enforce_tmpfs: false,
            derived_cache_budget_bytes: 67_108_864,
            derived_cache_admission_reserve_bytes: 8_388_608,
        }
    }

    fn journal_test_payload() -> String {
        serde_json::json!({
            "node": {
                "id": "journal-node-1",
                "node_type": "point",
                "position_m": [0.0, 0.0, 0.0],
                "sensor_offsets_m": [[0.0, 0.0, 0.0]],
                "capabilities": ["audio"],
                "metadata": {},
                "properties": {}
            },
            "buffered_frames": [
                {
                    "frame": {
                        "start_time_ns": 1000,
                        "utc_start_ns": 1000,
                        "utc_end_ns": 2000,
                        "start_sample_index": 0,
                        "end_sample_index": 4,
                        "sample_rate_hz": 16000,
                        "channels": 1,
                        "encoding": "pcm16le",
                        "samples_per_channel": 4,
                        "samples_b64": "AA==",
                        "sequence": 1,
                        "time_quality": "gps_locked",
                        "toa_ns": 1000,
                        "tor_ns": 2000,
                        "source_type": "raw_sensor"
                    }
                }
            ],
            "sort_by_toa": true
        })
        .to_string()
    }

    #[tokio::test]
    async fn returns_accepted_after_manifest_is_ready() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_app_state(
            IngestBackend::open(
                tmp.path().join("spool"),
                IngestStorageMode::Spool,
                8_388_608,
                test_journal_runtime_config(),
            )
            .await
            .unwrap(),
            usize::MAX,
        );
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
        assert_eq!(
            tmp.path()
                .join("spool")
                .join("ready")
                .read_dir()
                .unwrap()
                .count(),
            2
        );
        assert_eq!(
            tmp.path()
                .join("spool")
                .join("tmp")
                .read_dir()
                .unwrap()
                .count(),
            0
        );
    }

    #[tokio::test]
    async fn concurrent_uploads_create_unique_items() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_app_state(
            IngestBackend::open(
                tmp.path().join("spool"),
                IngestStorageMode::Spool,
                8_388_608,
                test_journal_runtime_config(),
            )
            .await
            .unwrap(),
            usize::MAX,
        );
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
        assert_eq!(
            tmp.path()
                .join("spool")
                .join("ready")
                .read_dir()
                .unwrap()
                .count(),
            4
        );
    }

    #[tokio::test]
    async fn oversized_body_returns_413() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_app_state(
            IngestBackend::open(
                tmp.path().join("spool"),
                IngestStorageMode::Spool,
                8_388_608,
                test_journal_runtime_config(),
            )
            .await
            .unwrap(),
            8,
        );
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

    #[tokio::test]
    async fn journal_mode_appends_segment_and_index_files() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_app_state(
            IngestBackend::open(
                tmp.path().join("store"),
                IngestStorageMode::Journal,
                8_388_608,
                test_journal_runtime_config(),
            )
            .await
            .unwrap(),
            usize::MAX,
        );

        let response = app(state)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(STORE_FORWARD_ENDPOINT)
                    .header("content-type", "application/json")
                    .body(Body::from(journal_test_payload()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::ACCEPTED);

        let streams_dir = tmp.path().join("store").join("journal").join("streams");
        let stream_dir = streams_dir
            .read_dir()
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let segments_dir = stream_dir.join("segments");
        let mut segment_files = 0;
        let mut index_files = 0;
        let mut metadata_files = 0;
        let mut entries = fs::read_dir(&segments_dir).await.unwrap();
        while let Some(entry) = entries.next_entry().await.unwrap() {
            match entry.path().extension().and_then(std::ffi::OsStr::to_str) {
                Some("bin") => segment_files += 1,
                Some("jsonl") => index_files += 1,
                Some("json") => metadata_files += 1,
                _ => {}
            }
        }

        assert_eq!(segment_files, 1);
        assert_eq!(index_files, 1);
        assert_eq!(metadata_files, 1);
    }

    #[tokio::test]
    async fn journal_overload_returns_503() {
        let payload = journal_test_payload();
        let reserve = 32_u64;
        let tmp = tempfile::tempdir().unwrap();
        let state = test_app_state(
            IngestBackend::open(
                tmp.path().join("store"),
                IngestStorageMode::Journal,
                8_388_608,
                JournalRuntimeConfig {
                    consumer_name: "python-ingest".to_string(),
                    total_journal_budget_bytes: u64::try_from(payload.len()).unwrap() + reserve,
                    admission_reserve_bytes: reserve,
                    enforce_tmpfs: false,
                    derived_cache_budget_bytes: 67_108_864,
                    derived_cache_admission_reserve_bytes: 8_388_608,
                },
            )
            .await
            .unwrap(),
            usize::MAX,
        );

        let first = app(state.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(STORE_FORWARD_ENDPOINT)
                    .header("content-type", "application/json")
                    .body(Body::from(payload.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let second = app(state)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(STORE_FORWARD_ENDPOINT)
                    .header("content-type", "application/json")
                    .body(Body::from(payload))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(first.status(), StatusCode::ACCEPTED);
        assert_eq!(second.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
