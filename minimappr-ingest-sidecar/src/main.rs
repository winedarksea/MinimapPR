mod actors;
mod audio_payload;
mod birdnet_render;
mod classifier_helper;
mod derived_cache;
mod dsp;
mod dsp_render_output;
mod dsp_worker;
mod env_payload;
mod envelope;
mod gcc_phat;
mod iamf_writer;
mod ingest_backend;
mod journal_reader;
mod leases;
mod manifests;
mod render_mvdr;
mod runtime;
mod srp_phat;
mod storage_class;
mod stream_range_lease;

use std::net::SocketAddr;
use std::path::PathBuf;

use actors::environment::EnvironmentCache;
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
use env_payload::{EnvIngestPayload, EnvIngestResponse};
use iamf_writer::{
    split_bed_into_frames, split_object_into_frames, IamfScene, IamfWriter, LoudnessInfo,
    ObjectPosition, ObjectPositions,
};
use ingest_backend::{
    BodyTooLargeError, ClientDisconnectError, IngestBackend, IngestStorageMode,
    InvalidIngestEnvelopeError, JournalCapacityExceededError, JournalRuntimeConfig,
};
use leases::PinLeaseRequest;
use render_mvdr::{render_mvdr, MvdrRenderRequest, TrajectoryWaypoint};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use stream_range_lease::{StreamRangeLeaseRequest, StreamRangeLeaseStore};
use tokio::sync::{mpsc, RwLock};
use tower_http::trace::TraceLayer;
use tracing::{info, warn};

const BINARY_ENDPOINT: &str = "/api/v1/ingest/binary";
const STORE_FORWARD_ENDPOINT: &str = "/api/v1/ingest/store-forward";
const ENV_ENDPOINT: &str = "/api/v1/ingest/env";
const MIN_BIRDNET_CLASSIFICATION_WINDOW_SECONDS: f64 = 15.0;
const DEFAULT_BIRDNET_CLASSIFICATION_WINDOW_SECONDS: f64 = 30.0;
const DEFAULT_CLASSIFIER_RENDER_OVERLAP_SECONDS: f64 = 2.0;

#[derive(Parser, Debug)]
#[command(author, version, about = "MinimapPR firmware ingest spool sidecar")]
struct Args {
    #[arg(long, env = "MINIMAPPR_SIDECAR_HOST", default_value = "0.0.0.0")]
    host: String,

    #[arg(long, env = "MINIMAPPR_INGEST_PORT", default_value_t = 8081)]
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
        default_value_t = 33_554_432
    )]
    derived_cache_admission_reserve_bytes: u64,

    #[arg(long, env = "MINIMAPPR_RUNTIME_PROFILE", default_value = "default")]
    runtime_profile: String,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_WINDOW_SECONDS",
        default_value_t = 512.0 / 16_000.0
    )]
    localization_window_seconds: f64,

    #[arg(
        long,
        env = "MINIMAPPR_CLASSIFICATION_WINDOW_SECONDS",
        default_value_t = 0.0
    )]
    classification_window_seconds: f64,

    #[arg(
        long,
        env = "MINIMAPPR_CLASSIFIER_RENDER_MIN_INTERVAL_SECONDS",
        default_value_t = -1.0
    )]
    classifier_render_min_interval_seconds: f64,

    #[arg(
        long,
        env = "MINIMAPPR_MAX_SENSOR_BUFFER_SECONDS",
        default_value_t = 8.0
    )]
    max_sensor_buffer_seconds: f64,

    #[arg(long, env = "MINIMAPPR_DSP_PENDING_BATCH_SIZE", default_value_t = 128)]
    dsp_pending_batch_size: usize,

    #[arg(
        long,
        env = "MINIMAPPR_DSP_SKIP_STALE_MANIFESTS",
        default_value_t = true
    )]
    dsp_skip_stale_manifests: bool,

    #[arg(
        long,
        env = "MINIMAPPR_DSP_CONSUMED_MANIFEST_RETENTION_MAX_FILES",
        default_value_t = 20_000
    )]
    dsp_consumed_manifest_retention_max_files: usize,

    #[arg(
        long,
        env = "MINIMAPPR_DSP_CONSUMED_MANIFEST_PRUNE_INTERVAL",
        default_value_t = 256
    )]
    dsp_consumed_manifest_prune_interval: u64,

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
        env = "MINIMAPPR_LOCALIZATION_SRP_GRID_RESOLUTION_M",
        default_value_t = 0.5
    )]
    localization_srp_grid_resolution_m: f32,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_SEARCH_PADDING_M",
        default_value_t = 2.0
    )]
    localization_search_padding_m: f32,

    #[arg(
        long,
        env = "MINIMAPPR_DSP_LOCALIZATION_CADENCE_MS",
        default_value_t = 250
    )]
    dsp_localization_cadence_ms: u64,

    #[arg(
        long,
        env = "MINIMAPPR_DSP_LOCALIZATION_RMS_GATE",
        default_value_t = 0.0
    )]
    dsp_localization_rms_gate: f32,

    #[arg(
        long,
        env = "MINIMAPPR_TRIGGER_COOLDOWN_SECONDS",
        default_value_t = -1.0
    )]
    trigger_cooldown_seconds: f64,

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

    #[arg(long, env = "MINIMAPPR_DEFAULT_TEMPERATURE_C", default_value_t = 20.0)]
    default_temperature_c: f32,

    #[arg(long, env = "MINIMAPPR_DEFAULT_HUMIDITY", default_value_t = 0.5)]
    default_humidity_fraction: f32,

    #[arg(long, env = "MINIMAPPR_SIDECAR_CLASSIFIER_COMMAND_JSON")]
    classifier_command_json: Option<String>,
}

#[derive(Clone, Debug)]
struct AppState {
    backend: IngestBackend,
    max_body_bytes: usize,
    dsp_state: SharedDspState,
    #[allow(dead_code)]
    derived_cache: Option<DerivedCache>,
    range_lease_store: Option<Arc<StreamRangeLeaseStore>>,
    env_cache: EnvironmentCache,
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

    if std::env::var_os("MINIMAPPR_INGEST_PORT").is_none() {
        if let Some(legacy_port) = std::env::var_os("MINIMAPPR_SIDECAR_PORT") {
            std::env::set_var("MINIMAPPR_INGEST_PORT", legacy_port);
        }
    }
    let args = Args::parse();
    let storage_dir = args.spool_dir.clone();
    let storage_mode = args.storage_mode;
    // Create the in-process channel before opening the backend so we can inject
    // the sender into JournalRuntimeConfig. Capacity 512 = ~16s at 32ms frames.
    let (raw_manifest_tx, raw_manifest_rx) = mpsc::channel::<manifests::DspManifest>(512);
    let journal_runtime_config = JournalRuntimeConfig {
        consumer_name: args.consumer_name.clone(),
        total_journal_budget_bytes: args.total_journal_budget_bytes,
        admission_reserve_bytes: args.admission_reserve_bytes,
        enforce_tmpfs: args.storage_mode == IngestStorageMode::Journal
            && !args.allow_non_tmpfs_journal,
        derived_cache_budget_bytes: args.derived_cache_budget_bytes,
        derived_cache_admission_reserve_bytes: args.derived_cache_admission_reserve_bytes,
        raw_manifest_tx: Some(raw_manifest_tx),
    };
    let backend = IngestBackend::open(
        args.spool_dir,
        args.storage_mode,
        args.segment_max_bytes,
        journal_runtime_config,
    )
    .await?;

    let dsp_state: SharedDspState = Arc::new(RwLock::new(Default::default()));
    let env_cache = EnvironmentCache::new();

    // Spawn DSP worker if journal mode has a manifest store and derived cache.
    if let (Some(manifest_store), Some(derived_cache)) =
        (backend.manifest_store(), backend.derived_cache())
    {
        let localization_window_seconds = args.localization_window_seconds.max(512.0 / 16_000.0);
        let birdnet_hybrid_render_enabled = args.runtime_profile == "birdnet_hybrid_production";
        let classification_window_seconds = if birdnet_hybrid_render_enabled {
            if args.classification_window_seconds <= 0.0 {
                DEFAULT_BIRDNET_CLASSIFICATION_WINDOW_SECONDS
            } else {
                args.classification_window_seconds
                    .max(MIN_BIRDNET_CLASSIFICATION_WINDOW_SECONDS)
            }
            .max(localization_window_seconds)
        } else if args.classification_window_seconds <= 0.0 {
            localization_window_seconds
        } else {
            args.classification_window_seconds
                .max(localization_window_seconds)
        };
        let max_sensor_buffer_seconds = args
            .max_sensor_buffer_seconds
            .max(classification_window_seconds)
            .max(localization_window_seconds);
        let classifier_render_min_interval_seconds =
            if args.classifier_render_min_interval_seconds >= 0.0 {
                args.classifier_render_min_interval_seconds
            } else if birdnet_hybrid_render_enabled {
                (classification_window_seconds - DEFAULT_CLASSIFIER_RENDER_OVERLAP_SECONDS).max(0.0)
            } else {
                0.0
            };
        let trigger_cooldown_seconds = if args.trigger_cooldown_seconds >= 0.0 {
            args.trigger_cooldown_seconds
        } else {
            DspWorkerConfig::default().trigger_cooldown_seconds
        };
        let worker = DspWorker::new(
            manifest_store,
            derived_cache,
            DspWorkerConfig {
                window_seconds: localization_window_seconds,
                classification_window_seconds,
                classifier_render_min_interval_seconds,
                max_buffer_seconds: max_sensor_buffer_seconds,
                pending_manifest_batch_size: args.dsp_pending_batch_size,
                birdnet_hybrid_render_enabled,
                skip_stale_manifests_for_live_buffer: args.dsp_skip_stale_manifests,
                consumed_manifest_retention_max_files: args
                    .dsp_consumed_manifest_retention_max_files,
                consumed_manifest_prune_interval: args.dsp_consumed_manifest_prune_interval,
                localization_band_hz: [
                    args.localization_band_min_hz,
                    args.localization_band_max_hz,
                ],
                localization_srp_grid_resolution_m: args.localization_srp_grid_resolution_m,
                localization_search_padding_m: args.localization_search_padding_m,
                spatial_blend_band_hz: [
                    args.birdnet_spatial_blend_min_hz,
                    args.birdnet_spatial_blend_max_hz,
                ],
                pre_blend_highpass_hz: args.birdnet_pre_blend_highpass_hz,
                default_temperature_c: args.default_temperature_c,
                default_humidity_fraction: args.default_humidity_fraction,
                classifier_command_json: args.classifier_command_json.clone(),
                localization_cadence_ms: args.dsp_localization_cadence_ms,
                localization_rms_gate: args.dsp_localization_rms_gate,
                trigger_cooldown_seconds,
                ..DspWorkerConfig::default()
            },
            dsp_state.clone(),
        );
        let worker = worker.with_raw_manifest_receiver(raw_manifest_rx);
        let worker = worker.with_env_cache(env_cache.clone());
        let (worker, classification_worker) = worker.with_classification_worker(64);
        if let Some(cw) = classification_worker {
            tokio::spawn(cw.run_loop());
            info!("ClassificationWorker spawned");
        }
        tokio::spawn(worker.run_loop());
        info!("DSP worker spawned");
        // Eagerly init the rayon pool so first-frame latency is not inflated.
        let _ = runtime::dsp_pool();
    }

    let range_lease_store = backend
        .journal_root()
        .map(|root| Arc::new(StreamRangeLeaseStore::new(root)));

    let state = AppState {
        derived_cache: backend.derived_cache(),
        backend,
        max_body_bytes: args.max_body_bytes,
        dsp_state,
        range_lease_store,
        env_cache,
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
        .route(ENV_ENDPOINT, post(ingest_env))
        .route("/api/v1/journal/pins", post(create_pin_lease))
        .route(
            "/api/v1/journal/pins/:lease_id",
            axum::routing::delete(release_pin_lease),
        )
        .route("/api/v1/dsp/status", get(dsp_status))
        .route("/api/v1/dsp/results", get(dsp_results))
        // Capture pipeline: stream-range leases
        .route(
            "/api/v1/capture/range-lease",
            post(create_stream_range_lease),
        )
        .route(
            "/api/v1/capture/range-lease/:lease_id",
            axum::routing::delete(release_stream_range_lease),
        )
        .route(
            "/api/v1/capture/range-lease/:lease_id/heartbeat",
            post(heartbeat_stream_range_lease),
        )
        // Capture pipeline: journal range extraction
        .route("/api/v1/journal/range", get(journal_range))
        // Capture pipeline: MVDR beamforming
        .route("/api/v1/capture/render/mvdr", post(render_mvdr_endpoint))
        // Capture pipeline: IAMF encoding
        .route("/api/v1/capture/encode/iamf", post(encode_iamf))
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

async fn ingest_env(
    State(state): State<AppState>,
    Json(payload): Json<EnvIngestPayload>,
) -> Response {
    use actors::environment::EnvSample;
    use manifests::DspManifest;

    if payload.samples.len() > 64 {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(ErrorResponse {
                accepted: false,
                queued: false,
                detail: "env batch exceeds 64 samples".to_string(),
            }),
        )
            .into_response();
    }

    let now_ns = dsp_worker::system_now_ns();
    let mut accepted = 0usize;
    for dto in &payload.samples {
        let sample = EnvSample {
            t_ns: dto.t_ns,
            temp_c: dto.temp_c,
            rh_pct: dto.rh_pct,
        };
        state.env_cache.update(&dto.node_id, sample).await;
        accepted += 1;
    }

    // Journal env readings so the Python spool consumer can process them.
    if let Some(manifest_store) = state.backend.manifest_store() {
        if let Ok(samples_value) = serde_json::to_value(&payload.samples) {
            let manifest = DspManifest {
                manifest_id: String::new(),
                manifest_type: "env_sample_append".to_string(),
                created_ns: now_ns,
                source_handles: vec![],
                derived_handle: None,
                localization: None,
                classifier_render: None,
                birdnet: None,
                coverage_stats: None,
                promotion_ready: false,
                env_samples: Some(samples_value),
                raw_payload: None,
            };
            if let Err(err) = manifest_store.publish(manifest).await {
                warn!(error = %err, "failed to publish env_sample_append manifest");
            }
        }
    }

    (StatusCode::ACCEPTED, Json(EnvIngestResponse { accepted })).into_response()
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
    total_stale_manifest_skips: u64,
    total_classification_drops: u64,
    dsp_worker_running: bool,
}

async fn dsp_status(State(state): State<AppState>) -> Response {
    let st = state.dsp_state.read().await;
    Json(DspStatusResponse {
        worker_running: st.worker_running,
        last_heartbeat_ns: st.last_heartbeat_ns,
        last_processed_ns: st.last_processed_ns,
        pending_manifest_count: st.pending_count,
        total_tdoa_results: st.total_tdoa_results,
        total_localization_results: st.total_localization_results,
        total_classifier_renders: st.total_classifier_renders,
        total_failures: st.total_failures,
        total_stale_manifest_skips: st.total_stale_manifest_skips,
        total_classification_drops: st.total_classification_drops,
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
        let st = state.dsp_state.read().await;
        st.recent_results
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    };
    Json(results).into_response()
}

fn current_unix_ns() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

// ── Stream-Range-Lease handlers ───────────────────────────────────────────────

async fn create_stream_range_lease(
    State(state): State<AppState>,
    Json(request): Json<StreamRangeLeaseRequest>,
) -> Response {
    let Some(store) = state.range_lease_store.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                accepted: false,
                queued: false,
                detail: "stream range leases require journal storage mode".to_string(),
            }),
        )
            .into_response();
    };
    match store.create(request, current_unix_ns()).await {
        Ok((response, _)) => (StatusCode::CREATED, Json(response)).into_response(),
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

async fn release_stream_range_lease(
    State(state): State<AppState>,
    axum::extract::Path(lease_id): axum::extract::Path<String>,
) -> Response {
    let Some(store) = state.range_lease_store.as_ref() else {
        return StatusCode::NO_CONTENT.into_response();
    };
    match store.release(&lease_id, current_unix_ns()).await {
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

async fn heartbeat_stream_range_lease(
    State(state): State<AppState>,
    axum::extract::Path(lease_id): axum::extract::Path<String>,
) -> Response {
    let Some(store) = state.range_lease_store.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                accepted: false,
                queued: false,
                detail: "stream range leases require journal storage mode".to_string(),
            }),
        )
            .into_response();
    };
    match store.heartbeat(&lease_id, current_unix_ns()).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                accepted: false,
                queued: false,
                detail: error.to_string(),
            }),
        )
            .into_response(),
    }
}

// ── Journal range extraction ──────────────────────────────────────────────────

#[derive(Deserialize)]
struct JournalRangeQuery {
    stream_key: String,
    start_ns: u64,
    end_ns: u64,
}

#[derive(Serialize)]
struct JournalRangeEntry {
    segment_id: String,
    first_toa_ns: Option<u64>,
    last_toa_ns: Option<u64>,
    sample_rate_hz: Option<u32>,
    payload_bytes: u64,
    segment_path: std::path::PathBuf,
}

async fn journal_range(
    State(state): State<AppState>,
    Query(params): Query<JournalRangeQuery>,
) -> Response {
    let Some(journal_root) = state.backend.journal_root() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                accepted: false,
                queued: false,
                detail: "journal range requires journal storage mode".to_string(),
            }),
        )
            .into_response();
    };

    let segments_dir = journal_root
        .join("streams")
        .join(&params.stream_key)
        .join("segments");

    let mut entries = match tokio::fs::read_dir(&segments_dir).await {
        Ok(e) => e,
        Err(_) => {
            return Json(Vec::<JournalRangeEntry>::new()).into_response();
        }
    };

    let mut matching: Vec<JournalRangeEntry> = Vec::new();
    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Ok(content) = tokio::fs::read_to_string(&path).await else {
            continue;
        };
        let Ok(header) = serde_json::from_str::<serde_json::Value>(&content) else {
            continue;
        };

        let first_toa = header["first_toa_ns"].as_u64();
        let last_toa = header["last_toa_ns"].as_u64();
        let first_tor = header["first_tor_ns"].as_u64();
        let last_tor = header["last_tor_ns"].as_u64();
        let segment_id = header["segment_id"].as_str().unwrap_or("").to_string();
        let payload_bytes = header["payload_bytes"].as_u64().unwrap_or(0);
        let sample_rate = header["sample_rate_hz"].as_u64().map(|v| v as u32);

        // Overlap test: segment range overlaps [start_ns, end_ns].
        // Fall back to tor_ns when toa_ns is missing so non-GPS nodes are not silently excluded.
        let seg_start = first_toa.or(first_tor).unwrap_or(0);
        let seg_end = last_toa.or(last_tor).unwrap_or(0);
        if seg_start <= params.end_ns && seg_end >= params.start_ns {
            let bin_path = path.with_extension("bin");
            matching.push(JournalRangeEntry {
                segment_id,
                first_toa_ns: first_toa.or(first_tor),
                last_toa_ns: last_toa.or(last_tor),
                sample_rate_hz: sample_rate,
                payload_bytes,
                segment_path: bin_path,
            });
        }
    }

    // Sort by first_toa_ns ascending for ordered extraction.
    matching.sort_by_key(|e| e.first_toa_ns.unwrap_or(0));
    Json(matching).into_response()
}

// ── MVDR beamform endpoint ────────────────────────────────────────────────────

#[derive(Deserialize)]
struct MvdrRequest {
    /// Four channels, each a Vec<f32> of equal length (float32, ±1.0).
    channels: [Vec<f32>; 4],
    sample_rate_hz: u32,
    trajectory: Vec<MvdrWaypointDto>,
    fade_samples: Option<usize>,
}

#[derive(Deserialize)]
struct MvdrWaypointDto {
    sample_offset: usize,
    position_m: [f32; 3],
}

#[derive(Serialize)]
struct MvdrResponse {
    samples: Vec<f32>,
    sample_rate_hz: u32,
}

async fn render_mvdr_endpoint(Json(request): Json<MvdrRequest>) -> Response {
    let waypoints = request
        .trajectory
        .into_iter()
        .map(|wp| TrajectoryWaypoint {
            sample_offset: wp.sample_offset,
            position_m: wp.position_m,
        })
        .collect();

    let output = render_mvdr(MvdrRenderRequest {
        channels: request.channels,
        sample_rate_hz: request.sample_rate_hz,
        trajectory: waypoints,
        fade_samples: request.fade_samples,
    });

    Json(MvdrResponse {
        samples: output.samples,
        sample_rate_hz: output.sample_rate_hz,
    })
    .into_response()
}

// ── IAMF encode endpoint ──────────────────────────────────────────────────────

#[derive(Deserialize)]
struct IamfEncodeRequest {
    sample_rate_hz: u32,
    samples_per_frame: u32,
    bed_loudness: LoudnessInfoDto,
    object_loudness: Vec<LoudnessInfoDto>,
    /// 4-channel FOA bed, channels-first, float32.
    bed_channels: [Vec<f32>; 4],
    /// Mono object tracks, each float32.
    object_tracks: Vec<Vec<f32>>,
    /// Per-object positions per temporal unit: outer index = TU, inner = object_id → pos.
    positions_per_unit: Vec<std::collections::HashMap<u32, ObjectPositionDto>>,
}

#[derive(Deserialize, Clone)]
struct LoudnessInfoDto {
    integrated_loudness_lufs: f32,
    true_peak_dbfs: f32,
}

#[derive(Deserialize, Clone)]
struct ObjectPositionDto {
    azimuth_deg: f32,
    elevation_deg: f32,
    distance_norm: Option<f32>,
    distance_m: Option<f32>,
    end_azimuth_deg: Option<f32>,
    end_elevation_deg: Option<f32>,
    end_distance_norm: Option<f32>,
}

async fn encode_iamf(Json(request): Json<IamfEncodeRequest>) -> Response {
    let spf = request.samples_per_frame as usize;
    let scene = IamfScene {
        sample_rate_hz: request.sample_rate_hz,
        samples_per_frame: request.samples_per_frame,
        bed_loudness: LoudnessInfo {
            integrated_loudness_lufs: request.bed_loudness.integrated_loudness_lufs,
            true_peak_dbfs: request.bed_loudness.true_peak_dbfs,
        },
        object_loudness: request
            .object_loudness
            .iter()
            .map(|l| LoudnessInfo {
                integrated_loudness_lufs: l.integrated_loudness_lufs,
                true_peak_dbfs: l.true_peak_dbfs,
            })
            .collect(),
    };

    let n_objects = request.object_tracks.len();
    let writer = IamfWriter::new(scene, n_objects);

    let mut bitstream = writer.write_descriptor_obus();

    let bed_frame_chunks = split_bed_into_frames(&request.bed_channels, spf);
    let object_frame_chunks: Vec<Vec<Vec<u8>>> = request
        .object_tracks
        .iter()
        .map(|track| split_object_into_frames(track, spf))
        .collect();

    let n_frames = bed_frame_chunks[0].len();
    #[allow(clippy::needless_range_loop)]
    for fi in 0..n_frames {
        let bed_frames: [Vec<u8>; 4] = std::array::from_fn(|ch| bed_frame_chunks[ch][fi].clone());
        let object_frames: Vec<Vec<u8>> = object_frame_chunks
            .iter()
            .map(|chunks| chunks.get(fi).cloned().unwrap_or_default())
            .collect();

        let positions: ObjectPositions = request
            .positions_per_unit
            .get(fi)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|(id, p)| {
                (
                    id,
                    ObjectPosition {
                        azimuth_deg: p.azimuth_deg,
                        elevation_deg: p.elevation_deg,
                        distance_norm: p.distance_norm.or(p.distance_m).unwrap_or(0.0),
                        end_azimuth_deg: p.end_azimuth_deg,
                        end_elevation_deg: p.end_elevation_deg,
                        end_distance_norm: p.end_distance_norm,
                    },
                )
            })
            .collect();

        bitstream.extend(writer.write_temporal_unit(&bed_frames, &object_frames, &positions));
    }

    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/octet-stream")],
        bitstream,
    )
        .into_response()
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
            dsp_state: Arc::new(RwLock::new(Default::default())),
            derived_cache: None,
            range_lease_store: None,
            env_cache: EnvironmentCache::new(),
        }
    }

    fn test_journal_runtime_config() -> JournalRuntimeConfig {
        JournalRuntimeConfig {
            consumer_name: "python-ingest".to_string(),
            total_journal_budget_bytes: 268_435_456,
            admission_reserve_bytes: 16_777_216,
            enforce_tmpfs: false,
            derived_cache_budget_bytes: 67_108_864,
            derived_cache_admission_reserve_bytes: 33_554_432,
            raw_manifest_tx: None,
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
                    derived_cache_admission_reserve_bytes: 33_554_432,
                    raw_manifest_tx: None,
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
