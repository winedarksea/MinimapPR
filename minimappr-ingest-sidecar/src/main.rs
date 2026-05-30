mod actors;
mod audio_payload;
mod birdnet_render;
mod classifier_helper;
mod derived_cache;
mod dsp;
mod dsp_events;
mod dsp_render_output;
mod dsp_worker;
mod env_payload;
mod envelope;
mod gcc_phat;
#[allow(dead_code)]
mod iamf_writer;
mod ingest_backend;
mod journal_reader;
mod leases;
mod manifests;
mod render_mvdr;
mod runtime;
mod srp_phat;
mod storage_class;

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::collections::HashMap;

use actors::environment::EnvironmentCache;
use axum::{
    body::Body,
    extract::{Query, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use derived_cache::DerivedCache;
use dsp_events::{DspEventPublisher, ReplayableDspEvent};
use dsp_worker::{DspWorker, DspWorkerConfig, NodeAudioConfig, PreprocessStage, SharedDspState};
use env_payload::{EnvIngestPayload, EnvIngestResponse};
use ingest_backend::{
    BodyTooLargeError, ClientDisconnectError, IngestBackend, IngestStorageMode,
    InvalidIngestEnvelopeError, JournalCapacityExceededError, JournalRuntimeConfig,
    RawManifestChannelFullError, RawManifestQueueBackpressure, RawManifestQueueBytesExceededError,
};
use journal_reader::JournalPayloadHandle;
use leases::{PinKind, PinLeaseRequest, PinTargetKind};
use render_mvdr::{render_mvdr, MvdrRenderRequest, TrajectoryWaypoint};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, RwLock};
use tokio_stream::StreamExt;
use tower_http::trace::TraceLayer;
use tracing::{info, warn};
use uuid::Uuid;

use iamf_writer::{
    split_bed_into_frames, split_object_into_frames, IamfScene, IamfWriter, LoudnessInfo,
    ObjectPosition, ObjectPositions,
};

const BINARY_ENDPOINT: &str = "/api/v1/ingest/binary";
const STORE_FORWARD_ENDPOINT: &str = "/api/v1/ingest/store-forward";
const ENV_ENDPOINT: &str = "/api/v1/ingest/env";
const MIN_BIRDNET_CLASSIFICATION_WINDOW_SECONDS: f64 = 15.0;
const DEFAULT_BIRDNET_CLASSIFICATION_WINDOW_SECONDS: f64 = 30.0;
const DEFAULT_CLASSIFIER_RENDER_OVERLAP_SECONDS: f64 = 2.0;
const OVERLOAD_RETRY_AFTER_SECONDS: &str = "1";
const DSP_EVENT_CHANNEL_CAPACITY: usize = 4096;
const DSP_EVENT_REPLAY_CAPACITY: usize = 512;
const DSP_RESULTS_RECENT_CAPACITY: usize = 50;

#[derive(Parser, Debug)]
#[command(author, version, about = "MinimapPR firmware ingest sidecar")]
struct Args {
    #[arg(long, env = "MINIMAPPR_SIDECAR_HOST", default_value = "0.0.0.0")]
    host: String,

    #[arg(long, env = "MINIMAPPR_INGEST_PORT", default_value_t = 8081)]
    port: u16,

    #[arg(
        long,
        env = "MINIMAPPR_INGEST_STORAGE_DIR",
        default_value = "data/ingest"
    )]
    storage_dir: PathBuf,

    #[arg(
        long,
        env = "MINIMAPPR_SIDECAR_MAX_BODY_BYTES",
        default_value_t = 33_554_432
    )]
    max_body_bytes: usize,

    #[arg(
        long,
        env = "MINIMAPPR_SIDECAR_SEGMENT_MAX_BYTES",
        default_value_t = 8_388_608
    )]
    segment_max_bytes: u64,

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
        env = "MINIMAPPR_SIDECAR_MEMORY_ONLY_LIVE_PATH",
        default_value_t = false
    )]
    memory_only_live_path: bool,

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
        env = "MINIMAPPR_DSP_RAW_MANIFEST_CHANNEL_CAPACITY",
        default_value_t = 2048
    )]
    dsp_raw_manifest_channel_capacity: usize,

    #[arg(
        long,
        env = "MINIMAPPR_DSP_RAW_MANIFEST_QUEUE_BYTE_LIMIT",
        default_value_t = 67_108_864
    )]
    dsp_raw_manifest_queue_byte_limit: usize,

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
        env = "MINIMAPPR_DSP_CONSUMED_MANIFEST_PRUNE_MAX_AGE_SECONDS",
        default_value_t = 300
    )]
    dsp_consumed_manifest_prune_max_age_seconds: u64,

    #[arg(
        long,
        env = "MINIMAPPR_DSP_STREAM_INACTIVITY_EVICT_SECONDS",
        default_value_t = 3600
    )]
    dsp_stream_inactivity_evict_seconds: u64,

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
        env = "MINIMAPPR_GCC_PHAT_INTERP_FACTOR",
        default_value_t = 4
    )]
    gcc_phat_interp_factor: usize,

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
        env = "MINIMAPPR_LOCALIZATION_FAR_FIELD_DEFAULT_RANGE_M",
        default_value_t = 50.0
    )]
    localization_far_field_default_range_m: f32,

    #[arg(
        long,
        env = "MINIMAPPR_LOCALIZATION_FAR_FIELD_MAX_RANGE_M",
        default_value_t = 250.0
    )]
    localization_far_field_max_range_m: f32,

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
    accepting_ingest: Arc<AtomicBool>,
    #[allow(dead_code)]
    derived_cache: Option<DerivedCache>,
    env_cache: EnvironmentCache,
    raw_manifest_queue_backpressure: Arc<RawManifestQueueBackpressure>,
    dsp_event_publisher: DspEventPublisher,
}

#[derive(Debug, Serialize)]
struct EnqueueResponse {
    accepted: bool,
    queued: bool,
    journal_id: String,
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
    let storage_dir = args.storage_dir.clone();
    let raw_manifest_channel_capacity = args.dsp_raw_manifest_channel_capacity.max(1);
    let raw_manifest_queue_byte_limit = args.dsp_raw_manifest_queue_byte_limit.max(1);
    let raw_manifest_queue_backpressure = Arc::new(RawManifestQueueBackpressure::new(
        raw_manifest_queue_byte_limit,
    ));
    let (raw_manifest_tx, raw_manifest_rx) =
        mpsc::channel::<ingest_backend::QueuedRawManifest>(raw_manifest_channel_capacity);
    let accepting_ingest = Arc::new(AtomicBool::new(true));
    let journal_runtime_config = JournalRuntimeConfig {
        total_journal_budget_bytes: args.total_journal_budget_bytes,
        admission_reserve_bytes: args.admission_reserve_bytes,
        enforce_tmpfs: !args.allow_non_tmpfs_journal,
        memory_only_live_path: args.memory_only_live_path,
        derived_cache_budget_bytes: args.derived_cache_budget_bytes,
        derived_cache_admission_reserve_bytes: args.derived_cache_admission_reserve_bytes,
        raw_manifest_channel_capacity,
        raw_manifest_queue_backpressure: Some(raw_manifest_queue_backpressure.clone()),
        raw_manifest_tx: Some(raw_manifest_tx),
    };
    let backend = IngestBackend::open(
        args.storage_dir,
        IngestStorageMode::Journal,
        args.segment_max_bytes,
        journal_runtime_config,
    )
    .await?;

    let dsp_state: SharedDspState = Arc::new(RwLock::new(Default::default()));
    let env_cache = EnvironmentCache::new();

    let dsp_event_publisher = DspEventPublisher::new(
        dsp_state.clone(),
        DSP_EVENT_CHANNEL_CAPACITY,
        DSP_EVENT_REPLAY_CAPACITY,
        DSP_RESULTS_RECENT_CAPACITY,
    );
    let mut dsp_worker_handle = None;
    let mut classification_worker_handle = None;

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
                skip_stale_manifests_for_live_buffer: args.dsp_skip_stale_manifests,
                consumed_manifest_retention_max_files: args
                    .dsp_consumed_manifest_retention_max_files,
                consumed_manifest_prune_interval: args.dsp_consumed_manifest_prune_interval,
                consumed_manifest_prune_max_age_seconds: args
                    .dsp_consumed_manifest_prune_max_age_seconds,
                stream_inactivity_evict_seconds: args.dsp_stream_inactivity_evict_seconds,
                localization_band_hz: [
                    args.localization_band_min_hz,
                    args.localization_band_max_hz,
                ],
                gcc_phat_interp_factor: args.gcc_phat_interp_factor,
                localization_srp_grid_resolution_m: args.localization_srp_grid_resolution_m,
                localization_search_padding_m: args.localization_search_padding_m,
                localization_far_field_default_range_m: args.localization_far_field_default_range_m,
                localization_far_field_max_range_m: args.localization_far_field_max_range_m,
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
                query_persisted_raw_manifests: false,
                ..DspWorkerConfig::default()
            },
            dsp_state.clone(),
        );
        let worker = worker.with_raw_manifest_receiver(raw_manifest_rx);
        let worker = worker.with_env_cache(env_cache.clone());
        let worker = worker.with_dsp_event_publisher(dsp_event_publisher.clone());
        let worker = worker.with_shutdown_signal(accepting_ingest.clone());
        let (worker, classification_worker) = worker.with_classification_worker(64);
        if let Some(cw) = classification_worker {
            classification_worker_handle = Some(tokio::spawn(cw.run_loop()));
            info!("ClassificationWorker spawned");
        }
        dsp_worker_handle = Some(tokio::spawn(worker.run_loop()));
        info!("DSP worker spawned");
        // Eagerly init the rayon pool so first-frame latency is not inflated.
        let _ = runtime::dsp_pool();
    }

    let state = AppState {
        accepting_ingest: accepting_ingest.clone(),
        derived_cache: backend.derived_cache(),
        backend,
        max_body_bytes: args.max_body_bytes,
        dsp_state,
        env_cache,
        raw_manifest_queue_backpressure,
        dsp_event_publisher,
    };

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    info!(
        %addr,
        storage_dir = %storage_dir.display(),
        storage_mode = if args.memory_only_live_path {
            "memory_only_live_path"
        } else {
            "journal"
        },
        max_body_bytes = args.max_body_bytes,
        total_journal_budget_bytes = args.total_journal_budget_bytes,
        admission_reserve_bytes = args.admission_reserve_bytes,
        derived_cache_budget_bytes = args.derived_cache_budget_bytes,
        "starting ingest sidecar"
    );

    let listener = tokio::net::TcpListener::bind(addr).await?;
    let server_result = axum::serve(listener, app(state))
        .with_graceful_shutdown(shutdown_signal(accepting_ingest.clone()))
        .await;
    if let Some(handle) = dsp_worker_handle {
        if let Err(error) = handle.await {
            warn!(error = %error, "DSP worker task join failed during shutdown");
        }
    }
    if let Some(handle) = classification_worker_handle {
        if let Err(error) = handle.await {
            warn!(error = %error, "Classification worker task join failed during shutdown");
        }
    }
    server_result?;
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
        .route(
            "/api/v1/capture/range-lease",
            post(create_capture_range_lease),
        )
        .route(
            "/api/v1/capture/range-lease/:lease_id",
            axum::routing::delete(release_pin_lease),
        )
        .route("/api/v1/dsp/status", get(dsp_status))
        .route("/api/v1/dsp/results", get(dsp_results))
        .route("/api/v1/dsp/stream", get(dsp_stream))
        .route("/api/v1/dsp/config", post(post_dsp_config))
        // Capture pipeline: MVDR beamforming
        .route("/api/v1/capture/render/mvdr", post(render_mvdr_endpoint))
        // Capture pipeline: IAMF encoding
        .route("/api/v1/capture/encode/iamf", post(encode_iamf))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn shutdown_signal(accepting_ingest: Arc<AtomicBool>) {
    use tokio::signal::unix::{signal, SignalKind};
    let mut sigterm = signal(SignalKind::terminate()).expect("failed to register SIGTERM handler");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {},
        _ = sigterm.recv() => {},
    }
    accepting_ingest.store(false, Ordering::Release);
    info!("shutdown signal received; rejecting new ingest and draining in-flight work");
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
    if !state.accepting_ingest.load(Ordering::Acquire) {
        return overload_response("ingest sidecar is draining for shutdown".to_string());
    }
    use actors::environment::EnvSample;

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

    let mut accepted = 0usize;
    let mut sse_samples: Vec<serde_json::Value> = Vec::new();
    for dto in &payload.samples {
        let sample = EnvSample {
            t_ns: dto.t_ns,
            temp_c: dto.temp_c,
            rh_pct: dto.rh_pct,
        };
        state.env_cache.update(&dto.node_id, sample).await;
        sse_samples.push(serde_json::json!({
            "node_id": dto.node_id,
            "sample": {
                "timestamp_ns": dto.t_ns,
                "temperature_c": dto.temp_c,
                "humidity_fraction": dto.rh_pct / 100.0,
                "source": "rust_sidecar_env",
            }
        }));
        accepted += 1;
    }

    if !sse_samples.is_empty() {
        let created_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        let env_manifest = manifests::DspManifest {
            manifest_id: format!("manifest-{}", Uuid::new_v4()),
            manifest_type: "env_sample_append".to_string(),
            created_ns,
            source_handles: vec![],
            derived_handle: None,
            localization: None,
            classifier_render: None,
            birdnet: None,
            coverage_stats: None,
            promotion_ready: false,
            env_samples: Some(serde_json::json!({ "samples": sse_samples })),
            node_context: None,
            cluster_id: None,
            cluster_sensor_positions: None,
            raw_payload: None,
            raw_render_bytes: None,
            raw_audio_frame: None,
            raw_audio_bytes: None,
        };
        let _ = state.dsp_event_publisher.publish(env_manifest).await;
    }

    (StatusCode::ACCEPTED, Json(EnvIngestResponse { accepted })).into_response()
}

async fn enqueue_request(
    state: AppState,
    endpoint: &'static str,
    headers: HeaderMap,
    body: Body,
) -> Response {
    if !state.accepting_ingest.load(Ordering::Acquire) {
        return overload_response("ingest sidecar is draining for shutdown".to_string());
    }
    match state
        .backend
        .enqueue(state.max_body_bytes, endpoint, headers, body)
        .await
    {
        Ok(journal_id) => (
            StatusCode::ACCEPTED,
            Json(EnqueueResponse {
                accepted: true,
                queued: true,
                journal_id,
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
                // the payload was not committed. This is expected
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
                return overload_response(err.to_string());
            }
            if err.downcast_ref::<RawManifestChannelFullError>().is_some() {
                return overload_response(err.to_string());
            }
            if err
                .downcast_ref::<RawManifestQueueBytesExceededError>()
                .is_some()
            {
                return overload_response(err.to_string());
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

fn overload_response(detail: String) -> Response {
    let mut response = (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorResponse {
            accepted: false,
            queued: false,
            detail,
        }),
    )
        .into_response();
    response.headers_mut().insert(
        header::RETRY_AFTER,
        HeaderValue::from_static(OVERLOAD_RETRY_AFTER_SECONDS),
    );
    response
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

#[derive(Debug, Deserialize)]
struct CaptureRangeLeaseRequest {
    owner: String,
    stream_key: String,
    start_ns: u64,
    requested_end_ns: u64,
}

async fn create_capture_range_lease(
    State(state): State<AppState>,
    Json(request): Json<CaptureRangeLeaseRequest>,
) -> Response {
    if request.requested_end_ns <= request.start_ns {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                accepted: false,
                queued: false,
                detail: "requested_end_ns must be greater than start_ns".to_string(),
            }),
        )
            .into_response();
    }
    let ttl_seconds = ((request.requested_end_ns - request.start_ns) / 1_000_000_000).max(1);
    let pin_request = PinLeaseRequest {
        owner: request.owner,
        pin_kind: PinKind::Hard,
        target_kind: PinTargetKind::RawJournal,
        lease_id: None,
        ttl_seconds,
        handle: JournalPayloadHandle {
            journal_epoch: 0,
            segment_id: format!(
                "capture-range-{}-{}-{}",
                request.stream_key, request.start_ns, request.requested_end_ns
            ),
            stream_key: request.stream_key,
            payload_offset_bytes: 0,
            payload_length_bytes: 0,
            toa_ns: Some(request.start_ns),
            tor_ns: None,
            received_ns: None,
            sample_index_start: None,
            sample_count: None,
            integrity_hash: String::new(),
            segment_path: PathBuf::new(),
        },
        promotion_id: None,
    };
    create_pin_lease(State(state), Json(pin_request)).await
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
    raw_manifest_queue_depth: usize,
    raw_manifest_queue_bytes: usize,
    raw_manifest_queue_byte_limit: usize,
    total_localization_attempts: u64,
    total_classification_attempts: u64,
    total_tdoa_results: u64,
    total_localization_results: u64,
    total_classifier_renders: u64,
    total_failures: u64,
    total_stale_manifest_skips: u64,
    total_classification_drops: u64,
    // Mirrors of Python silent-drop counters. Names are kept stable so an
    // alerting rule (e.g. `total_window_underrun_drops` climbing while no
    // classifier renders emit) works against either backend.
    total_buffer_reanchors: u64,
    total_window_underrun_drops: u64,
    total_stale_streams_evicted: u64,
    dsp_worker_running: bool,
}

async fn dsp_status(State(state): State<AppState>) -> Response {
    let st = state.dsp_state.read().await;
    Json(DspStatusResponse {
        worker_running: st.worker_running,
        last_heartbeat_ns: st.last_heartbeat_ns,
        last_processed_ns: st.last_processed_ns,
        pending_manifest_count: st.pending_count,
        raw_manifest_queue_depth: state.raw_manifest_queue_backpressure.queued_messages(),
        raw_manifest_queue_bytes: state.raw_manifest_queue_backpressure.queued_bytes(),
        raw_manifest_queue_byte_limit: state.raw_manifest_queue_backpressure.byte_limit(),
        total_localization_attempts: st.total_localization_attempts,
        total_classification_attempts: st.total_classification_attempts,
        total_tdoa_results: st.total_tdoa_results,
        total_localization_results: st.total_localization_results,
        total_classifier_renders: st.total_classifier_renders,
        total_failures: st.total_failures,
        total_stale_manifest_skips: st.total_stale_manifest_skips,
        total_classification_drops: st.total_classification_drops,
        total_buffer_reanchors: st.total_buffer_reanchors,
        total_window_underrun_drops: st.total_window_underrun_drops,
        total_stale_streams_evicted: st.total_stale_streams_evicted,
        dsp_worker_running: st.worker_running,
    })
    .into_response()
}

/// Request body for POST /api/v1/dsp/config — sets per-node audio overrides.
#[derive(Debug, Deserialize)]
struct DspNodeConfigRequest {
    node_id: String,
    #[serde(flatten)]
    config: NodeAudioConfig,
}

/// Validate the new `stages` chain — reject obviously-malformed entries before
/// they can poison the per-stream filter cache. The validation must accept
/// every shape the matching Python `NodeAudioOverride.stages` accepts.
fn validate_preprocess_stages(stages: &[PreprocessStage]) -> Result<(), String> {
    for (idx, stage) in stages.iter().enumerate() {
        match stage {
            PreprocessStage::Gain { db } => {
                if !(-60.0..=60.0).contains(db) {
                    return Err(format!(
                        "stages[{idx}].db must be in [-60, 60] (got {db})"
                    ));
                }
            }
            PreprocessStage::Highpass { cutoff_hz, order }
            | PreprocessStage::Lowpass { cutoff_hz, order } => {
                if !cutoff_hz.is_finite() || *cutoff_hz <= 0.0 {
                    return Err(format!(
                        "stages[{idx}].cutoff_hz must be > 0 (got {cutoff_hz})"
                    ));
                }
                if *order < 2 || *order > 8 {
                    return Err(format!(
                        "stages[{idx}].order must be in [2, 8] (got {order})"
                    ));
                }
            }
            PreprocessStage::Bandpass {
                low_hz,
                high_hz,
                order,
            } => {
                if !low_hz.is_finite() || *low_hz <= 0.0 {
                    return Err(format!(
                        "stages[{idx}].low_hz must be > 0 (got {low_hz})"
                    ));
                }
                if !high_hz.is_finite() || *high_hz <= *low_hz {
                    return Err(format!(
                        "stages[{idx}].high_hz must be > low_hz (got high={high_hz} low={low_hz})"
                    ));
                }
                if *order < 2 || *order > 8 {
                    return Err(format!(
                        "stages[{idx}].order must be in [2, 8] (got {order})"
                    ));
                }
            }
            PreprocessStage::DcBlock | PreprocessStage::Passthrough => {}
        }
    }
    Ok(())
}

/// POST /api/v1/dsp/config — store per-node gain/highpass overrides applied
/// to audio during ingest (takes effect on the next frame for that node).
async fn post_dsp_config(
    State(state): State<AppState>,
    Json(body): Json<DspNodeConfigRequest>,
) -> Response {
    if body.node_id.is_empty() {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({"error": "node_id must not be empty"})),
        )
            .into_response();
    }
    if let Some(db) = body.config.gain_db {
        if !(-60.0..=60.0).contains(&db) {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({"error": "gain_db must be in [-60, 60]"})),
            )
                .into_response();
        }
    }
    if let Some(hz) = body.config.hp_hz {
        if hz < 0.0 {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({"error": "hp_hz must be >= 0"})),
            )
                .into_response();
        }
    }
    if let Err(error) = validate_preprocess_stages(&body.config.stages) {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({"error": error})),
        )
            .into_response();
    }
    {
        let mut st = state.dsp_state.write().await;
        st.node_audio_overrides
            .insert(body.node_id.clone(), body.config.clone());
    }
    Json(serde_json::json!({
        "node_id": body.node_id,
        "applied": true,
    }))
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

/// SSE endpoint that streams DSP result manifests
/// (localization_result, classifier_render, env_sample_append)
/// to Python consumers in real-time, bypassing the filesystem entirely.
async fn dsp_stream(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Sse<
    impl tokio_stream::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>,
> {
    let last_event_id = headers
        .get("last-event-id")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let rx = state.dsp_event_publisher.subscribe();
    let replay_window = state
        .dsp_event_publisher
        .replay_after(last_event_id, DSP_EVENT_REPLAY_CAPACITY)
        .await;
    let replayed_through_event_id = replay_window
        .events
        .last()
        .map(|event| event.event_id)
        .or(last_event_id)
        .unwrap_or(0);
    let mut replay_events = Vec::new();
    if replay_window.gap_detected {
        let data = serde_json::json!({
            "requested_after": last_event_id,
            "oldest_available": replay_window.oldest_available_event_id,
        });
        replay_events.push(Ok(axum::response::sse::Event::default()
            .event("replay_gap")
            .data(data.to_string())));
    }
    replay_events.extend(
        replay_window
            .events
            .into_iter()
            .filter_map(replayable_dsp_event_to_sse)
            .map(Ok),
    );
    let replay_stream = tokio_stream::iter(replay_events);
    let live_stream = futures::stream::unfold(
        LiveDspEventStreamState {
            rx,
            last_delivered_event_id: replayed_through_event_id,
            close_after_yield: false,
        },
        |mut stream_state| async move {
            if stream_state.close_after_yield {
                return None;
            }
            loop {
                match stream_state.rx.recv().await {
                    Ok(event) if event.event_id <= stream_state.last_delivered_event_id => continue,
                    Ok(event) => {
                        stream_state.last_delivered_event_id = event.event_id;
                        if let Some(sse_event) = replayable_dsp_event_to_sse(event) {
                            return Some((Ok(sse_event), stream_state));
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped_messages)) => {
                        stream_state.close_after_yield = true;
                        let gap_event = replay_gap_sse_event(
                            Some(stream_state.last_delivered_event_id),
                            Some(skipped_messages),
                        );
                        return Some((Ok(gap_event), stream_state));
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => return None,
                }
            }
        },
    );
    let stream = replay_stream.chain(live_stream);
    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new().interval(std::time::Duration::from_secs(15)),
    )
}

struct LiveDspEventStreamState {
    rx: tokio::sync::broadcast::Receiver<ReplayableDspEvent>,
    last_delivered_event_id: u64,
    close_after_yield: bool,
}

fn replayable_dsp_event_to_sse(event: ReplayableDspEvent) -> Option<axum::response::sse::Event> {
    let json = serde_json::to_string(&event.manifest).ok()?;
    Some(
        axum::response::sse::Event::default()
            .id(event.event_id.to_string())
            .data(json),
    )
}

fn replay_gap_sse_event(
    requested_after: Option<u64>,
    skipped_messages: Option<u64>,
) -> axum::response::sse::Event {
    let data = serde_json::json!({
        "requested_after": requested_after,
        "skipped_messages": skipped_messages,
    });
    axum::response::sse::Event::default()
        .event("replay_gap")
        .data(data.to_string())
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
    bed_wav_path: PathBuf,
    object_wav_path: Option<PathBuf>,
    positions_json_path: PathBuf,
    output_iamf_path: PathBuf,
    bitrate_per_channel_bps: Option<u32>,
}

#[derive(Debug, Deserialize, Default)]
struct IamfPositionsPayload {
    #[serde(default)]
    positions_per_unit: Vec<HashMap<String, IamfPositionJson>>,
}

#[derive(Debug, Deserialize)]
struct IamfPositionJson {
    azimuth_deg: f32,
    elevation_deg: f32,
    distance_norm: f32,
    #[serde(default)]
    end_azimuth_deg: Option<f32>,
    #[serde(default)]
    end_elevation_deg: Option<f32>,
    #[serde(default)]
    end_distance_norm: Option<f32>,
}

#[derive(Deserialize, Serialize, Clone)]
struct LoudnessInfoDto {
    integrated_loudness_lufs: f32,
    true_peak_dbfs: f32,
}

async fn encode_iamf(Json(request): Json<IamfEncodeRequest>) -> Response {
    if let Err(error) = validate_iamf_encode_request(&request).await {
        return (StatusCode::BAD_REQUEST, error).into_response();
    }
    match encode_iamf_with_rust_writer(&request).await {
        Ok(bytes) => (
            StatusCode::OK,
            [(axum::http::header::CONTENT_TYPE, "application/octet-stream")],
            bytes,
        )
            .into_response(),
        Err(error) => (StatusCode::BAD_GATEWAY, error).into_response(),
    }
}

async fn encode_iamf_with_rust_writer(request: &IamfEncodeRequest) -> Result<Vec<u8>, String> {
    let _bitrate_per_channel_bps = request.bitrate_per_channel_bps.unwrap_or(128_000);
    let bed = read_wav_channels_f32(&request.bed_wav_path, 4)?;
    let object_track = match &request.object_wav_path {
        Some(path) => Some(read_wav_mono_f32(path)?),
        None => None,
    };

    let positions_payload = tokio::fs::read_to_string(&request.positions_json_path)
        .await
        .map_err(|error| {
            format!(
                "failed to read positions_json_path {}: {error}",
                request.positions_json_path.display()
            )
        })?;
    let positions_payload: IamfPositionsPayload = serde_json::from_str(&positions_payload)
        .map_err(|error| format!("failed to parse IAMF positions JSON: {error}"))?;

    let object_loudness: Vec<LoudnessInfo> = request
        .object_loudness
        .iter()
        .map(|loudness| LoudnessInfo {
            integrated_loudness_lufs: loudness.integrated_loudness_lufs,
            true_peak_dbfs: loudness.true_peak_dbfs,
        })
        .collect();
    let object_position_defaults = first_object_position_defaults(&positions_payload);

    let scene = IamfScene {
        sample_rate_hz: request.sample_rate_hz,
        samples_per_frame: request.samples_per_frame,
        n_bed_channels: 4,
        bed_loudness: LoudnessInfo {
            integrated_loudness_lufs: request.bed_loudness.integrated_loudness_lufs,
            true_peak_dbfs: request.bed_loudness.true_peak_dbfs,
        },
        object_loudness,
        object_position_defaults,
    };

    let writer = IamfWriter::new(scene, object_track.as_ref().map_or(0, |_| 1));
    let bed_frames = split_bed_into_frames(&bed, request.samples_per_frame as usize);
    let object_frames = object_track
        .as_ref()
        .map(|samples| split_object_into_frames(samples, request.samples_per_frame as usize))
        .unwrap_or_default();
    let frame_count = bed_frames[0].len();
    let positions_per_unit = convert_positions_per_unit(&positions_payload, frame_count);

    let mut bitstream = writer.write_descriptor_obus();
    for frame_index in 0..frame_count {
        let positions = positions_per_unit
            .get(frame_index)
            .cloned()
            .unwrap_or_default();
        let bed_unit = [
            bed_frames[0][frame_index].clone(),
            bed_frames[1][frame_index].clone(),
            bed_frames[2][frame_index].clone(),
            bed_frames[3][frame_index].clone(),
        ];
        let object_unit = object_frames
            .get(frame_index)
            .map(|frame| vec![frame.clone()])
            .unwrap_or_default();
        bitstream.extend(writer.write_temporal_unit(&bed_unit, &object_unit, &positions));
    }

    tokio::fs::write(&request.output_iamf_path, &bitstream)
        .await
        .map_err(|error| {
            format!(
                "failed to write IAMF output {}: {error}",
                request.output_iamf_path.display()
            )
        })?;
    Ok(bitstream)
}

fn first_object_position_defaults(payload: &IamfPositionsPayload) -> Vec<ObjectPosition> {
    let mut defaults = Vec::new();
    for object_id in 0..1u32 {
        let default_position = payload
            .positions_per_unit
            .iter()
            .find_map(|unit| unit.get(&object_id.to_string()))
            .map(|position| ObjectPosition {
                azimuth_deg: position.azimuth_deg,
                elevation_deg: position.elevation_deg,
                distance_norm: position.distance_norm,
                end_azimuth_deg: None,
                end_elevation_deg: None,
                end_distance_norm: None,
            })
            .unwrap_or(ObjectPosition {
                azimuth_deg: 0.0,
                elevation_deg: 0.0,
                distance_norm: 0.0,
                end_azimuth_deg: None,
                end_elevation_deg: None,
                end_distance_norm: None,
            });
        defaults.push(default_position);
    }
    defaults
}

fn convert_positions_per_unit(
    payload: &IamfPositionsPayload,
    frame_count: usize,
) -> Vec<ObjectPositions> {
    let mut positions_per_unit = Vec::with_capacity(frame_count);
    for unit in payload.positions_per_unit.iter().take(frame_count) {
        let mut positions = ObjectPositions::new();
        for (object_id, position) in unit {
            if let Ok(parsed_id) = object_id.parse::<u32>() {
                positions.insert(
                    parsed_id,
                    ObjectPosition {
                        azimuth_deg: position.azimuth_deg,
                        elevation_deg: position.elevation_deg,
                        distance_norm: position.distance_norm,
                        end_azimuth_deg: position.end_azimuth_deg,
                        end_elevation_deg: position.end_elevation_deg,
                        end_distance_norm: position.end_distance_norm,
                    },
                );
            }
        }
        positions_per_unit.push(positions);
    }
    while positions_per_unit.len() < frame_count {
        positions_per_unit.push(ObjectPositions::new());
    }
    positions_per_unit
}

fn read_wav_channels_f32(path: &Path, expected_channels: usize) -> Result<[Vec<f32>; 4], String> {
    let reader = hound::WavReader::open(path)
        .map_err(|error| format!("failed to open WAV {}: {error}", path.display()))?;
    let spec = reader.spec();
    if spec.channels as usize != expected_channels {
        return Err(format!(
            "{} has {} channels, expected {}",
            path.display(),
            spec.channels,
            expected_channels,
        ));
    }
    if spec.sample_format != hound::SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err(format!(
            "{} must be 16-bit PCM for IAMF export (got {:?}/{} bits)",
            path.display(),
            spec.sample_format,
            spec.bits_per_sample,
        ));
    }

    let mut channels: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::new());
    let mut sample_index = 0usize;
    for sample in reader.into_samples::<i16>() {
        let value = sample
            .map_err(|error| format!("failed reading {}: {error}", path.display()))?
            as f32
            / 32767.0;
        channels[sample_index % expected_channels].push(value);
        sample_index += 1;
    }
    Ok(channels)
}

fn read_wav_mono_f32(path: &Path) -> Result<Vec<f32>, String> {
    let reader = hound::WavReader::open(path)
        .map_err(|error| format!("failed to open WAV {}: {error}", path.display()))?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err(format!(
            "{} has {} channels, expected mono",
            path.display(),
            spec.channels,
        ));
    }
    if spec.sample_format != hound::SampleFormat::Int || spec.bits_per_sample != 16 {
        return Err(format!(
            "{} must be 16-bit PCM for IAMF export (got {:?}/{} bits)",
            path.display(),
            spec.sample_format,
            spec.bits_per_sample,
        ));
    }

    let mut samples = Vec::new();
    for sample in reader.into_samples::<i16>() {
        samples.push(
            sample
                .map_err(|error| format!("failed reading {}: {error}", path.display()))?
                as f32
                / 32767.0,
        );
    }
    Ok(samples)
}

async fn validate_iamf_encode_request(request: &IamfEncodeRequest) -> Result<(), String> {
    if request.sample_rate_hz != 48_000 {
        return Err("IAMF YouTube export expects 48 kHz audio".to_string());
    }
    if request.samples_per_frame == 0 {
        return Err("samples_per_frame must be non-zero".to_string());
    }
    if !request.bed_loudness.integrated_loudness_lufs.is_finite()
        || !request.bed_loudness.true_peak_dbfs.is_finite()
    {
        return Err("bed_loudness must contain finite values".to_string());
    }
    if request.object_loudness.len() > 1 {
        return Err("IAMF base export supports at most one object audio element".to_string());
    }
    if request.object_loudness.iter().any(|loudness| {
        !loudness.integrated_loudness_lufs.is_finite() || !loudness.true_peak_dbfs.is_finite()
    }) {
        return Err("object_loudness must contain finite values".to_string());
    }
    ensure_file_exists(&request.bed_wav_path, "bed_wav_path").await?;
    ensure_file_exists(&request.positions_json_path, "positions_json_path").await?;
    if let Some(path) = &request.object_wav_path {
        ensure_file_exists(path, "object_wav_path").await?;
        if request.object_loudness.len() != 1 {
            return Err(
                "object_loudness must contain exactly one entry when object_wav_path is set"
                    .to_string(),
            );
        }
    } else if !request.object_loudness.is_empty() {
        return Err("object_loudness must be empty when object_wav_path is absent".to_string());
    }
    Ok(())
}

async fn ensure_file_exists(path: &Path, field_name: &str) -> Result<(), String> {
    match tokio::fs::metadata(path).await {
        Ok(metadata) if metadata.is_file() => Ok(()),
        Ok(_) => Err(format!(
            "{field_name} is not a regular file: {}",
            path.display()
        )),
        Err(error) => Err(format!(
            "{field_name} does not exist or is unreadable: {} ({error})",
            path.display()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iamf_test_request(tmp: &tempfile::TempDir, with_object: bool) -> IamfEncodeRequest {
        let bed = tmp.path().join("bed.wav");
        let positions = tmp.path().join("positions.json");
        let output = tmp.path().join("audio.iamf");
        std::fs::write(&bed, b"wav").unwrap();
        std::fs::write(&positions, b"{\"positions_per_unit\":[]}").unwrap();
        let object = if with_object {
            let path = tmp.path().join("object.wav");
            std::fs::write(&path, b"wav").unwrap();
            Some(path)
        } else {
            None
        };
        IamfEncodeRequest {
            sample_rate_hz: 48_000,
            samples_per_frame: 512,
            bed_loudness: LoudnessInfoDto {
                integrated_loudness_lufs: -20.0,
                true_peak_dbfs: -3.0,
            },
            object_loudness: if with_object {
                vec![LoudnessInfoDto {
                    integrated_loudness_lufs: -18.0,
                    true_peak_dbfs: -4.0,
                }]
            } else {
                vec![]
            },
            bed_wav_path: bed,
            object_wav_path: object,
            positions_json_path: positions,
            output_iamf_path: output,
            bitrate_per_channel_bps: Some(128_000),
        }
    }

    fn write_test_wav(path: &Path, channels: u16, sample_rate: u32, frames: &[i16]) {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for sample in frames {
            writer.write_sample(*sample).unwrap();
        }
        writer.finalize().unwrap();
    }

    #[tokio::test]
    async fn iamf_path_request_rejects_more_than_one_object() {
        let tmp = tempfile::tempdir().unwrap();
        let mut request = iamf_test_request(&tmp, true);
        request.object_loudness.push(LoudnessInfoDto {
            integrated_loudness_lufs: -18.0,
            true_peak_dbfs: -4.0,
        });

        let error = validate_iamf_encode_request(&request).await.unwrap_err();

        assert!(error.contains("at most one object"));
    }

    #[tokio::test]
    async fn iamf_encode_uses_positions_json_with_rust_writer() {
        let tmp = tempfile::tempdir().unwrap();
        let bed = tmp.path().join("bed.wav");
        let object = tmp.path().join("object.wav");
        let positions = tmp.path().join("positions.json");
        let output = tmp.path().join("audio.iamf");

        let mut bed_frames = Vec::new();
        for _ in 0..512 {
            bed_frames.extend_from_slice(&[0i16, 1024, 2048, 3072]);
        }
        write_test_wav(&bed, 4, 48_000, &bed_frames);
        write_test_wav(&object, 1, 48_000, &[0i16; 512]);

        std::fs::write(
            &positions,
            b"{\"sample_rate_hz\":48000,\"samples_per_frame\":512,\"positions_per_unit\":[{\"0\":{\"azimuth_deg\":45.0,\"elevation_deg\":10.0,\"distance_norm\":0.5,\"end_azimuth_deg\":60.0,\"end_elevation_deg\":12.0,\"end_distance_norm\":0.6}}]}",
        )
        .unwrap();

        let request = IamfEncodeRequest {
            sample_rate_hz: 48_000,
            samples_per_frame: 512,
            bed_loudness: LoudnessInfoDto {
                integrated_loudness_lufs: -20.0,
                true_peak_dbfs: -3.0,
            },
            object_loudness: vec![LoudnessInfoDto {
                integrated_loudness_lufs: -18.0,
                true_peak_dbfs: -4.0,
            }],
            bed_wav_path: bed,
            object_wav_path: Some(object),
            positions_json_path: positions,
            output_iamf_path: output.clone(),
            bitrate_per_channel_bps: Some(128_000),
        };

        let bytes = encode_iamf_with_rust_writer(&request).await.unwrap();

        assert!(!bytes.is_empty());
        assert_eq!(bytes, std::fs::read(output).unwrap());
    }
}
