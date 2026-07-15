use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{
    actors::{classification::ClassificationWorker, environment::EnvironmentCache},
    audio_payload::{decode_audio_payload_segments, DecodedAudioPayload},
    classifier_helper::ManifestClassificationAnnotator,
    derived_cache::DerivedCache,
    diagnostics::IngestDiagnostics,
    dsp::{coverage_stats, AudioCoverageStats, SensorStreamBuffer},
    dsp_events::DspEventPublisher,
    gcc_phat::TdoaResult,
    ingest_backend::QueuedRawManifest,
    journal_reader::JournalPayloadHandle,
    manifests::{DspManifest, ManifestStore},
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde::{Deserialize, Serialize};
use tokio::{sync::mpsc, sync::RwLock, time};
use tracing::{debug, error, info, warn};

const MIN_BIRDNET_CLASSIFICATION_WINDOW_SECONDS: f64 = 15.0;
const DEFAULT_BIRDNET_CLASSIFICATION_WINDOW_SECONDS: f64 = 30.0;
const DEFAULT_CLASSIFIER_RENDER_OVERLAP_SECONDS: f64 = 2.0;
const AUTOMATIC_GPS_CLUSTER_ID: &str = "auto-gps";
const EARTH_RADIUS_M: f64 = 6_371_000.0;
/// Largest mic array the spatial pipeline (SRP-PHAT / GCC-PHAT / render_mvdr_n)
/// will run on. Sirith Planar (5 mics) fits; anything larger falls back to omni.
const MAX_SPATIAL_CHANNELS: usize = 8;
/// Minimum out-of-collinearity span (in meters) required for an array's
/// geometry to be usable by the spatial pipeline.
const SPATIAL_GEOMETRY_MIN_SPAN_M: f32 = 0.005;

/// Centroid-relative Sirith tetrahedral mic positions [MK1, MK2, MK3, MK4].
pub(crate) const SIRITH_MIC_POSITIONS_M: [[f32; 3]; 4] = [
    [0.0, 0.050, 0.0],
    [0.0433, 0.025, 0.0],
    [0.0, 0.0, 0.0],
    [0.02165, 0.025, 0.04082],
];

#[derive(Clone, Debug)]
pub struct DspWorkerConfig {
    /// Global fixed/causal ingest chain used when a node has no replacement override.
    pub default_audio_config: NodeAudioConfig,
    pub poll_interval_ms: u64,
    pub pending_manifest_batch_size: usize,
    pub window_seconds: f64,
    pub classification_window_seconds: f64,
    pub classifier_render_min_interval_seconds: f64,
    pub max_buffer_seconds: f64,
    pub min_coverage_ratio: f64,
    pub localization_band_hz: [f32; 2],
    pub gcc_phat_interp_factor: usize,
    pub localization_srp_grid_resolution_m: f32,
    pub localization_search_padding_m: f32,
    pub localization_far_field_default_range_m: f32,
    pub localization_far_field_max_range_m: f32,
    /// Amplitude/SNR range prior (Phase 1c). When enabled, the projection distance
    /// for unobservable-range solves is derived from the reference-channel received
    /// level via inverse-square spreading instead of the fixed default. Ships off.
    pub localization_amplitude_range_prior_enabled: bool,
    pub localization_amplitude_reference_level_db: f32,
    pub localization_amplitude_prior_min_range_m: f32,
    pub localization_amplitude_prior_max_range_m: f32,
    pub localization_amplitude_prior_std_factor: f32,
    /// Band-split render crossover parameters (BEAMFORMED_RENDER_CONTRACT.md).
    /// The steered-band top is the geometry-derived alias cutoff, optionally
    /// clamped by `band_split_max_clamp_hz` (legacy env override).
    pub band_split_highpass_hz: f32,
    pub band_split_low_crossover_width_hz: f32,
    pub band_split_high_crossover_width_min_hz: f32,
    pub band_split_high_crossover_width_fraction: f32,
    pub band_split_max_clamp_hz: Option<f32>,
    pub min_localization_confidence: f32,
    /// Classification audio source: "beamformed" (default) | "omni" | "nearest_node_omni".
    /// Selects the render path in `dsp_render_output::compute_render_bytes`.
    pub classification_audio_source: String,
    pub skip_stale_manifests_for_live_buffer: bool,
    pub consumed_manifest_retention_max_files: usize,
    pub consumed_manifest_prune_interval: u64,
    /// Also prune consumed manifests if this many seconds have elapsed since the
    /// last prune, even if the count threshold has not been reached (e.g. quiet
    /// periods where ingest pauses and prune would never fire).
    pub consumed_manifest_prune_max_age_seconds: u64,
    /// Evict per-stream buffer and state entries (buffers, node_audio_state, etc.)
    /// after this many seconds of inactivity. Prevents RSS growth from node-ID
    /// churn across sensor reflashes or hostname changes.
    pub stream_inactivity_evict_seconds: u64,
    pub default_temperature_c: f32,
    pub default_humidity_fraction: f32,
    pub sound_speed_mps: f32,
    pub classifier_command_json: Option<String>,
    pub localization_cadence_ms: u64,
    pub localization_rms_gate: f32,
    pub trigger_cooldown_seconds: f64,
    /// When false, the worker stays fully memory-path for raw audio and does not
    /// poll ManifestStore for raw_journal_append items.
    pub query_persisted_raw_manifests: bool,
    /// Maximum trusted node clock skew in seconds before falling back to
    /// receipt-time alignment. Matching Python's _MAX_TRUSTED_NODE_CLOCK_SKEW_NS.
    pub max_trusted_node_clock_skew_seconds: f64,
}

impl Default for DspWorkerConfig {
    fn default() -> Self {
        let default_temperature_c = 20.0;
        let default_humidity_fraction = 0.5;
        Self {
            default_audio_config: NodeAudioConfig::default(),
            poll_interval_ms: 20,
            pending_manifest_batch_size: 128,
            window_seconds: 512.0 / 16_000.0,
            classification_window_seconds: DEFAULT_BIRDNET_CLASSIFICATION_WINDOW_SECONDS,
            classifier_render_min_interval_seconds: DEFAULT_BIRDNET_CLASSIFICATION_WINDOW_SECONDS
                - DEFAULT_CLASSIFIER_RENDER_OVERLAP_SECONDS,
            max_buffer_seconds: DEFAULT_BIRDNET_CLASSIFICATION_WINDOW_SECONDS
                + DEFAULT_CLASSIFIER_RENDER_OVERLAP_SECONDS,
            min_coverage_ratio: 0.85,
            localization_band_hz: [300.0, 3500.0],
            gcc_phat_interp_factor: 4,
            localization_srp_grid_resolution_m: 0.5,
            localization_search_padding_m: 2.0,
            localization_far_field_default_range_m: 50.0,
            localization_far_field_max_range_m: 1000.0,
            localization_amplitude_range_prior_enabled: false,
            localization_amplitude_reference_level_db: 100.0,
            localization_amplitude_prior_min_range_m: 5.0,
            localization_amplitude_prior_max_range_m: 1000.0,
            localization_amplitude_prior_std_factor: 2.0,
            band_split_highpass_hz: 100.0,
            band_split_low_crossover_width_hz: 100.0,
            band_split_high_crossover_width_min_hz: 400.0,
            band_split_high_crossover_width_fraction: 0.15,
            band_split_max_clamp_hz: None,
            min_localization_confidence: 0.20,
            classification_audio_source: "beamformed".to_string(),
            skip_stale_manifests_for_live_buffer: false,
            consumed_manifest_retention_max_files: 20_000,
            consumed_manifest_prune_interval: 256,
            consumed_manifest_prune_max_age_seconds: 300,
            stream_inactivity_evict_seconds: 3600,
            default_temperature_c,
            default_humidity_fraction,
            sound_speed_mps: 343.2,
            classifier_command_json: None,
            localization_cadence_ms: 250,
            localization_rms_gate: 0.015,
            trigger_cooldown_seconds: 0.8,
            query_persisted_raw_manifests: true,
            max_trusted_node_clock_skew_seconds: 300.0,
        }
    }
}

/// Per-pair TDOA result with mic pair indices.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PairTdoa {
    pub ch_a: usize,
    pub ch_b: usize,
    pub tdoa: TdoaResult,
}

/// One stage in the per-node preprocessing chain. The chain is an ordered
/// `Vec<PreprocessStage>` applied in sequence; each stage owns its own
/// per-stream filter memory inside [`NodeAudioState`]. The JSON shape
/// (tagged `type`, snake_case variants) is intentionally identical on both
/// sides of the wire so Python's `NodeAudioOverride.stages` and Rust's
/// `NodeAudioConfig.stages` are bit-for-bit interchangeable.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PreprocessStage {
    /// Apply a linear gain expressed in dB. `db == 0.0` is a no-op.
    Gain { db: f64 },
    /// Per-channel calibration trim. Missing entries leave later channels unchanged.
    ChannelGain { db_by_channel: Vec<f64> },
    /// Butterworth highpass — even order ≥ 2. Default order = 4.
    Highpass {
        cutoff_hz: f64,
        #[serde(default = "default_filter_order")]
        order: u8,
    },
    /// Butterworth lowpass — even order ≥ 2. Default order = 4.
    Lowpass {
        cutoff_hz: f64,
        #[serde(default = "default_filter_order")]
        order: u8,
    },
    /// Butterworth bandpass — even order ≥ 2 applied to both highpass and lowpass
    /// halves. Default order = 4 per half.
    Bandpass {
        low_hz: f64,
        high_hz: f64,
        #[serde(default = "default_filter_order")]
        order: u8,
    },
    /// First-order DC blocker — removes mean and very-low-frequency drift.
    DcBlock,
    /// Explicit no-op. Useful as a placeholder when toggling stages without
    /// reordering the chain.
    Passthrough,
}

fn default_filter_order() -> u8 {
    4
}

/// Per-node audio DSP override (gain and filters applied before buffer insertion).
///
/// Two shapes coexist for backward compatibility:
/// * **New**: `stages` — ordered chain of [`PreprocessStage`] variants. Preferred.
/// * **Legacy**: `gain_db` / `hp_hz` — flat scalars. Only honored when `stages` is empty.
///
/// An empty `stages` array with the legacy fields unset means **passthrough** —
/// the per-stream samples are written to the buffer unmodified.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct NodeAudioConfig {
    /// Legacy gain in dB applied to all channels for this node (0.0 = no change).
    /// Ignored when `stages` is non-empty.
    pub gain_db: Option<f64>,
    /// Legacy 1st-order IIR highpass cutoff in Hz (0 or None = disabled).
    /// Ignored when `stages` is non-empty.
    pub hp_hz: Option<f64>,
    /// Ordered chain of preprocessing stages. Empty + legacy unset = passthrough.
    #[serde(default)]
    pub stages: Vec<PreprocessStage>,
}

impl NodeAudioConfig {
    /// Materialize the canonical stage list — preferring `stages` when present,
    /// otherwise synthesizing from the legacy `gain_db` / `hp_hz` fields so old
    /// clients keep working without behavior change. The legacy `hp_hz` was a
    /// first-order IIR; we map it to an order-2 Butterworth highpass which is
    /// the closest standard Butterworth section (order=1 is not realizable as a
    /// biquad cascade).
    pub fn effective_stages(&self) -> Vec<PreprocessStage> {
        if !self.stages.is_empty() {
            return self.stages.clone();
        }
        let mut synthesized = Vec::new();
        // `db != 0.0` is true for NaN (NaN != anything), so the explicit
        // is_finite() check guards against a NaN gain_db propagating into
        // CompiledStage::from_stage and corrupting samples with `*= NaN`.
        // API validation rejects NaN at the boundary but legacy fields can
        // arrive through internal callers that bypass the API layer.
        if let Some(db) = self.gain_db.filter(|&db| db != 0.0 && db.is_finite()) {
            synthesized.push(PreprocessStage::Gain { db });
        }
        // `hz > 0.0` already rejects NaN (NaN > 0.0 is false).
        if let Some(hz) = self.hp_hz.filter(|&hz| hz > 0.0) {
            synthesized.push(PreprocessStage::Highpass {
                cutoff_hz: hz,
                order: 2,
            });
        }
        synthesized
    }
}

/// Shared state exposed via the /api/v1/dsp/* endpoints.
#[derive(Debug, Default)]
pub struct DspWorkerState {
    pub worker_running: bool,
    pub last_heartbeat_ns: Option<u128>,
    pub last_processed_ns: Option<u128>,
    pub total_localization_attempts: u64,
    pub total_classification_attempts: u64,
    pub total_tdoa_results: u64,
    pub total_localization_results: u64,
    pub total_classifier_renders: u64,
    pub total_failures: u64,
    pub total_stale_manifest_skips: u64,
    pub total_classification_drops: u64,
    /// Per-sensor SensorStreamBuffer re-anchor events (NTP/GPS correction or
    /// large frame jumps). Mirrors `SensorStreamBuffer.reanchor_count` on the
    /// Python side. Visible degradation signal for long-uptime timeline drift.
    pub total_buffer_reanchors: u64,
    /// `compute_manifest_frame` returned None because the requested audio
    /// window fell outside the buffer's coverage. Mirrors the Python
    /// `localization_drops_by_reason["no_window"]` counter — both are the
    /// canonical "pipeline is silently dropping" signal.
    pub total_window_underrun_drops: u64,
    pub total_stale_streams_evicted: u64,
    pub pending_count: usize,
    pub recent_results: Vec<DspManifest>,
    /// Per-node audio overrides applied during ingest (set via POST /api/v1/dsp/config).
    pub node_audio_overrides: HashMap<String, NodeAudioConfig>,
}

pub type SharedDspState = Arc<RwLock<DspWorkerState>>;

#[derive(Clone, Debug)]
pub(crate) struct LocalizationChannelState {
    pub(crate) coverage: Option<AudioCoverageStats>,
    pub(crate) window: Vec<f32>,
}

/// Carries a pre-built classifier_render manifest + raw PCM bytes to the
/// ClassificationWorker. The worker keeps the PCM in memory and hands the
/// annotated manifest back to the shared DSP event publisher (no disk persist).
pub struct ClassificationRequest {
    pub pcm_bytes: Vec<u8>,
    pub sample_rate_hz: u32,
    pub pending_manifest: DspManifest,
    pub raw_render_bytes: Option<String>,
}

/// All owned data needed to run the compute phase (SRP-PHAT + render + publish)
/// for a single manifest after the ingest (buffer-append) phase completes.
pub(crate) struct ComputePayload {
    pub(crate) manifest: DspManifest,
    pub(crate) stream_key: String,
    pub(crate) channel_states: Vec<LocalizationChannelState>,
    pub(crate) active_channels: Vec<usize>,
    pub(crate) classification_windows: Vec<Vec<f32>>,
    pub(crate) listenable_classification_windows: Vec<Vec<f32>>,
    pub(crate) classification_coverage: Vec<Option<AudioCoverageStats>>,
    pub(crate) classifier_render_start_ns: Option<u128>,
    pub(crate) classifier_render_end_ns: Option<u128>,
    /// Per-channel mic positions extracted from node_context.sensor_offsets_m.
    pub(crate) mic_positions_m: Vec<[f32; 3]>,
    /// Coplanar array up/down constraint extracted from node_context.node.half_space.
    pub(crate) half_space: crate::srp_phat::HalfSpace,
    pub(crate) source_ids: Vec<String>,
    pub(crate) sr: u32,
    pub(crate) now_ns: u128,
    pub(crate) effective_sound_speed_mps: f32,
    pub(crate) reported_temperature_c: Option<f32>,
    pub(crate) reported_humidity_fraction: Option<f32>,
    pub(crate) reported_environment_source: Option<String>,
    pub(crate) run_srp: bool,
    pub(crate) run_classifier_render: bool,
    /// Suppress the localization_result manifest for this frame (e.g. omni-only single-point node).
    pub(crate) skip_localization_result: bool,
    /// Pre-computed reason to force omni fallback render (e.g. "single_point_node").
    pub(crate) omni_fallback_reason: Option<String>,
    /// Canonical render channels used for localization/BirdNET-facing render bytes.
    pub(crate) omni_channels_override: Option<Vec<Vec<f32>>>,
    /// Listener-facing render channels; may conceal missing spans on copied windows only.
    pub(crate) listenable_omni_channels_override: Option<Vec<Vec<f32>>>,
    pub(crate) manifest_store: ManifestStore,
    pub(crate) derived_cache: DerivedCache,
    pub(crate) state: SharedDspState,
    pub(crate) classification_tx: Option<flume::Sender<ClassificationRequest>>,
    pub(crate) dsp_event_publisher: Option<DspEventPublisher>,
    pub(crate) config: DspWorkerConfig,
    pub(crate) consumed_since_prune: Arc<AtomicU64>,
    pub(crate) diagnostics: Option<Arc<IngestDiagnostics>>,
}

struct OwnedManifestAudio {
    manifest: DspManifest,
    stream_key: String,
    buffer_key: String,
    decoded: DecodedAudioPayload,
    source_ids: Vec<String>,
    sr: u32,
    now_ns: u128,
    channel_count: usize,
    buffer_channel_count: usize,
    buffer_channel_indices: Vec<usize>,
    classification_window_sec: f64,
    mic_positions_m: Vec<[f32; 3]>,
    half_space: crate::srp_phat::HalfSpace,
    effective_sound_speed_mps: f32,
    buffer_start_time_ns: i128,
    buffer_end_time_ns: i128,
    start_sample_index: Option<i64>,
    end_sample_index: Option<i64>,
    node_timestamp_is_available: bool,
    buffer_uses_receipt_time: bool,
    arrived_via_channel: bool,
}

struct BufferedManifestAudio {
    audio_end_ns: u128,
    end_ns: i128,
    channel_states: Vec<LocalizationChannelState>,
}

struct BufferedManifestTimingGates {
    on_heartbeat_cadence: bool,
    run_classifier_render: bool,
    run_srp: bool,
}

struct ManifestBufferRouting {
    buffer_key: String,
    buffer_channel_count: usize,
    buffer_channel_indices: Vec<usize>,
}

/// Incrementally-built geometry for the automatic, GPS-defined array. Sensor
/// order never changes after insertion because buffer indices are long-lived.
/// Re-sorting after a new node joins would silently associate prior audio with
/// the wrong position.
#[derive(Default)]
struct AutomaticGpsGeometry {
    origin: Option<ReportedGpsPosition>,
    sensor_positions: Vec<(String, [f32; 3])>,
    sensor_indices_by_id: HashMap<String, usize>,
}

#[derive(Clone, Copy)]
struct ReportedGpsPosition {
    lat_deg: f64,
    lon_deg: f64,
    alt_m: f64,
}

impl AutomaticGpsGeometry {
    fn update_from_manifest(
        &mut self,
        manifest: &DspManifest,
        node_id: &str,
        channel_count: usize,
    ) -> Option<Vec<(String, [f32; 3])>> {
        let node = manifest.node_context.as_ref()?.get("node")?;
        let position = reported_gps_position(node)?;
        let origin = *self.origin.get_or_insert(position);
        let node_origin_m = gps_position_to_local_m(position, origin);
        let offsets = sensor_offsets_for_channel_count(node, channel_count);
        if offsets.len() != channel_count {
            return None;
        }

        for (channel_index, offset_m) in offsets.into_iter().enumerate() {
            let sensor_id = format!("{node_id}:ch{channel_index}");
            let sensor_position = [
                node_origin_m[0] + offset_m[0],
                node_origin_m[1] + offset_m[1],
                node_origin_m[2] + offset_m[2],
            ];
            if let Some(index) = self.sensor_indices_by_id.get(&sensor_id).copied() {
                self.sensor_positions[index].1 = sensor_position;
            } else {
                let index = self.sensor_positions.len();
                self.sensor_indices_by_id.insert(sensor_id.clone(), index);
                self.sensor_positions.push((sensor_id, sensor_position));
            }
        }
        Some(self.sensor_positions.clone())
    }
}

struct RawAudioFramePublishRequest<'a> {
    publisher: &'a Option<DspEventPublisher>,
    source_manifest: &'a DspManifest,
    decoded: &'a DecodedAudioPayload,
    stream_key: &'a str,
    sample_rate_hz: u32,
    start_time_ns: i128,
    end_time_ns: i128,
    created_ns: u128,
    segment_index: usize,
    segment_count: usize,
}

pub struct DspWorker {
    manifest_store: ManifestStore,
    derived_cache: DerivedCache,
    config: DspWorkerConfig,
    state: SharedDspState,
    classifier_annotator: Option<ManifestClassificationAnnotator>,
    classification_tx: Option<flume::Sender<ClassificationRequest>>,
    raw_manifest_rx: Option<mpsc::Receiver<QueuedRawManifest>>,
    shutdown_requested: Option<Arc<AtomicBool>>,
    buffers: HashMap<String, Vec<SensorStreamBuffer>>,
    /// Geometry synthesized from live node GPS reports when no explicit cluster
    /// topology accompanies a manifest. This is the Rust counterpart of the
    /// Python server's coordinate-frame-backed NodeRegistry geometry.
    automatic_gps_geometry: AutomaticGpsGeometry,
    /// Per-node preprocessing state (biquad memory etc.). Keyed by `node_id`
    /// extracted from the stream_key prefix, matching how `node_audio_overrides`
    /// is keyed in [`DspWorkerState`]. Lives outside the shared RwLock so frame
    /// preprocessing only contends on the single owning worker's `&mut self`.
    node_audio_state: HashMap<String, NodeAudioState>,
    last_classifier_render_ns_by_stream: HashMap<String, u128>,
    last_localization_ns_by_stream: HashMap<String, u128>,
    last_trigger_ns_by_stream: HashMap<String, u128>,
    /// Time-only cadence gate used for node liveness heartbeats.
    /// Unlike `last_localization_ns_by_stream`, this is never blocked by the RMS
    /// energy gate or trigger cooldown — it fires unconditionally every
    /// `localization_cadence_ms` so nodes appear alive even during quiet periods.
    last_heartbeat_ns_by_stream: HashMap<String, u128>,
    consumed_manifests_since_prune: Arc<AtomicU64>,
    last_consumed_prune_ns: u128,
    last_stream_purge_ns: u128,
    deferred_source_manifest_ids: Vec<String>,
    env_cache: Option<EnvironmentCache>,
    /// Shared publisher for DSP result manifests streamed to Python via SSE.
    dsp_event_publisher: Option<DspEventPublisher>,
    /// Lock-free queue-wait/processing-latency + overload counters, mirrored
    /// against the Python `FusionMetrics` equivalents. See `diagnostics.rs`.
    diagnostics: Option<Arc<IngestDiagnostics>>,
}

impl DspWorker {
    pub fn new(
        manifest_store: ManifestStore,
        derived_cache: DerivedCache,
        config: DspWorkerConfig,
        state: SharedDspState,
    ) -> Self {
        let classifier_annotator = match ManifestClassificationAnnotator::from_command_json(
            config.classifier_command_json.as_deref(),
        ) {
            Ok(a) => a,
            Err(error) => {
                warn!(error = %error, "DSP worker classifier helper disabled");
                None
            }
        };
        Self {
            manifest_store,
            derived_cache,
            config,
            state,
            classifier_annotator,
            classification_tx: None,
            raw_manifest_rx: None,
            shutdown_requested: None,
            buffers: HashMap::new(),
            automatic_gps_geometry: AutomaticGpsGeometry::default(),
            node_audio_state: HashMap::new(),
            last_classifier_render_ns_by_stream: HashMap::new(),
            last_localization_ns_by_stream: HashMap::new(),
            last_trigger_ns_by_stream: HashMap::new(),
            last_heartbeat_ns_by_stream: HashMap::new(),
            consumed_manifests_since_prune: Arc::new(AtomicU64::new(0)),
            last_consumed_prune_ns: system_now_ns(),
            last_stream_purge_ns: system_now_ns(),
            deferred_source_manifest_ids: Vec::new(),
            env_cache: None,
            dsp_event_publisher: None,
            diagnostics: None,
        }
    }

    /// Wires the ManifestClassificationAnnotator to a dedicated `ClassificationWorker`
    /// task via a flume bounded channel (capacity 64, raised from legacy 16).
    /// Returns `(self, Some(worker))` when classification is configured,
    /// `(self, None)` when no classifier command is set.
    pub fn with_classification_worker(
        mut self,
        channel_capacity: usize,
    ) -> (Self, Option<ClassificationWorker>) {
        let Some(annotator) = self.classifier_annotator.take() else {
            return (self, None);
        };
        let (tx, rx) = flume::bounded(channel_capacity);
        self.classification_tx = Some(tx);
        let worker = ClassificationWorker {
            annotator,
            rx,
            dsp_event_publisher: self.dsp_event_publisher.clone(),
        };
        (self, Some(worker))
    }

    /// Injects an in-process receiver for raw_journal_append manifests from the
    /// co-located ingest backend, bypassing the 20ms filesystem poll cycle.
    pub fn with_raw_manifest_receiver(mut self, rx: mpsc::Receiver<QueuedRawManifest>) -> Self {
        self.raw_manifest_rx = Some(rx);
        self
    }

    /// Injects a shared EnvironmentCache for sound-speed interpolation (Phase 6).
    pub fn with_env_cache(mut self, cache: EnvironmentCache) -> Self {
        self.env_cache = Some(cache);
        self
    }

    /// Injects the shared publisher used for SSE delivery and replay.
    pub fn with_dsp_event_publisher(mut self, publisher: DspEventPublisher) -> Self {
        self.dsp_event_publisher = Some(publisher);
        self
    }

    /// Signals the worker to stop after the HTTP server has finished draining
    /// accepted requests and this worker has emptied its manifest queue.
    pub fn with_shutdown_signal(mut self, shutdown_requested: Arc<AtomicBool>) -> Self {
        self.shutdown_requested = Some(shutdown_requested);
        self
    }

    /// Injects the shared diagnostics counters used by `/api/v1/diagnostics/summary`.
    pub fn with_diagnostics(mut self, diagnostics: Arc<IngestDiagnostics>) -> Self {
        self.diagnostics = Some(diagnostics);
        self
    }

    pub async fn run_loop(mut self) {
        info!("DSP worker started");
        let interval = time::Duration::from_millis(self.config.poll_interval_ms);
        const STREAM_PURGE_INTERVAL_NS: u128 = 60 * 1_000_000_000; // once per minute
        loop {
            {
                let mut st = self.state.write().await;
                st.worker_running = true;
                st.last_heartbeat_ns = Some(system_now_ns());
            }
            let processed_manifest_count = self.process_pending().await;
            if self.should_exit_after_drain(processed_manifest_count) {
                break;
            }
            let now_ns = system_now_ns();
            if now_ns.saturating_sub(self.last_stream_purge_ns) >= STREAM_PURGE_INTERVAL_NS {
                self.last_stream_purge_ns = now_ns;
                self.purge_stale_streams().await;
            }
            if should_sleep_after_poll_cycle(processed_manifest_count) {
                time::sleep(interval).await;
            }
        }
        {
            let mut st = self.state.write().await;
            st.worker_running = false;
            st.last_heartbeat_ns = Some(system_now_ns());
        }
        info!("DSP worker drained and exited");
    }

    fn should_exit_after_drain(&self, processed_manifest_count: usize) -> bool {
        let shutdown_requested = self
            .shutdown_requested
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Acquire));
        if !shutdown_requested || processed_manifest_count > 0 {
            return false;
        }
        self.raw_manifest_rx.as_ref().is_none_or(|rx| rx.is_empty())
    }

    async fn process_pending(&mut self) -> usize {
        let batch_limit = self.config.pending_manifest_batch_size.max(1);
        let channel_limit = (batch_limit / 2).max(1);

        // Drain a bounded amount from the in-process channel, then also query disk
        // so fresh spillover manifests are not starved behind channel backlog.
        let channel_manifests: Vec<DspManifest> = if let Some(rx) = &mut self.raw_manifest_rx {
            let mut v = Vec::new();
            while let Ok(m) = rx.try_recv() {
                v.push(m.into_manifest());
                if v.len() >= channel_limit {
                    break;
                }
            }
            v
        } else {
            Vec::new()
        };

        // Only hit the filesystem when the in-process channel did not already
        // fill the batch budget. This keeps steady-state ingest on memory paths.
        let should_query_disk =
            self.config.query_persisted_raw_manifests && channel_manifests.len() < channel_limit;
        let disk_manifests = if should_query_disk {
            match self
                .manifest_store
                .query_pending_limited("raw_journal_append", batch_limit)
                .await
            {
                Ok(m) => m,
                Err(err) => {
                    warn!(error = %err, "DSP worker failed to query pending manifests");
                    return 0;
                }
            }
        } else {
            Vec::new()
        };
        let pending =
            merge_pending_manifests_for_batch(channel_manifests, disk_manifests, batch_limit);

        {
            let mut st = self.state.write().await;
            st.pending_count = pending.len();
        }

        let pending_backlog_depth = pending.len();

        // Capture the Tokio handle once so each rayon closure can call
        // `handle.spawn(run_io(...))` after the sync math phase completes.
        let handle = tokio::runtime::Handle::current();

        for manifest in pending {
            if let Some(payload) = self.ingest_one(manifest, pending_backlog_depth).await {
                self.note_stage_attempts(&payload).await;
                let h = handle.clone();
                // Dispatch to the dedicated DSP rayon pool. run_compute is sync:
                // it runs CPU math on the Rayon thread, then calls handle.spawn(run_io(...))
                // which queues async I/O onto Tokio and returns immediately. The Rayon
                // thread is free for the next frame without waiting on disk or DB.
                crate::runtime::dsp_pool().spawn_fifo(move || {
                    crate::actors::dsp_compute::run_compute(payload, h);
                });
            }
        }

        self.flush_deferred_source_manifest_consumptions().await;
        pending_backlog_depth
    }

    async fn ingest_one(
        &mut self,
        manifest: DspManifest,
        pending_backlog_depth: usize,
    ) -> Option<ComputePayload> {
        let owned_segments = self.prepare_owned_manifest_audio(manifest).await?;

        for (segment_index, owned) in owned_segments.iter().enumerate() {
            self.publish_raw_audio_frame_for_owned_manifest(
                owned,
                segment_index,
                owned_segments.len(),
            )
            .await;
        }

        if self
            .should_skip_stale_manifest_for_live_buffer(owned_segments.first()?)
            .await
        {
            return None;
        }

        // Arrays with up to MAX_SPATIAL_CHANNELS mics run the full spatial
        // pipeline (SRP-PHAT / GCC-PHAT / render_mvdr_n are all N-mic). This is
        // what lets the 5-mic planar array localize rather than fall straight to
        // omni. Arrays beyond that cap, or whose geometry is missing/degenerate
        // (wrong offset count, single sensor, or collinear mics), take the
        // large-array omni fallback with a specific reason. Behaviour for the
        // 1-4 channel tetra/point path is unchanged (never enters this block).
        {
            let last = owned_segments.last()?;
            if last.channel_count > 4 {
                let geometry_usable = last.channel_count <= MAX_SPATIAL_CHANNELS
                    && last.mic_positions_m.len() == last.channel_count
                    && crate::dsp_math::array_spans_at_least_2d(
                        &last.mic_positions_m,
                        SPATIAL_GEOMETRY_MIN_SPAN_M,
                    );
                if !geometry_usable {
                    let reason = if last.channel_count > MAX_SPATIAL_CHANNELS {
                        "array_exceeds_max_spatial_channels"
                    } else {
                        "non_tetrahedral_array_geometry_unusable"
                    };
                    // Large-array fallback rendering bypasses the rolling buffer.
                    // Use only the newest contiguous segment rather than falsely
                    // joining audio across an explicit capture gap.
                    return self.dispatch_large_array_manifest(
                        owned_segments.into_iter().last()?,
                        pending_backlog_depth,
                        reason,
                    );
                }
                // else: fall through to the normal spatial buffered path.
            }
        }

        let mut final_owned_and_buffered = None;
        for owned in owned_segments {
            let buffered = self.buffer_owned_manifest_audio(&owned).await?;
            final_owned_and_buffered = Some((owned, buffered));
        }
        let (final_owned, final_buffered) = final_owned_and_buffered?;
        self.dispatch_buffered_manifest(final_owned, final_buffered, pending_backlog_depth)
            .await
    }

    async fn prepare_owned_manifest_audio(
        &mut self,
        mut manifest: DspManifest,
    ) -> Option<Vec<OwnedManifestAudio>> {
        let now_ns = system_now_ns();
        let Some(first_handle) = manifest.source_handles.first().cloned() else {
            self.mark_source_manifest_consumed_if_persisted(&manifest)
                .await;
            return None;
        };

        if let (Some(diagnostics), Some(received_ns)) =
            (&self.diagnostics, first_handle.received_ns)
        {
            let wait_ms = now_ns.saturating_sub(received_ns) / 1_000_000;
            diagnostics.record_queue_wait_ms(wait_ms as u64);
        }

        let stream_key = first_handle.stream_key.clone();
        let arrived_via_channel = manifest.raw_payload.is_some();
        let raw_payload = if let Some(bytes) = manifest.raw_payload.clone() {
            bytes
        } else {
            warn!(
                manifest_id = %manifest.manifest_id,
                "DSP worker: raw manifest arrived without in-memory payload; frame dropped"
            );
            self.defer_source_manifest_consumption(&manifest);
            return None;
        };

        let decoded_segments = match decode_audio_payload_segments(&raw_payload) {
            Ok(decoded_segments) => decoded_segments,
            Err(err) => {
                self.note_failure().await;
                warn!(
                    manifest_id = %manifest.manifest_id,
                    error = %err,
                    "DSP worker failed to decode ingest audio; consuming manifest"
                );
                self.mark_source_manifest_consumed_if_persisted(&manifest)
                    .await;
                return None;
            }
        };
        if decoded_segments.is_empty()
            || decoded_segments
                .iter()
                .any(|decoded| decoded.channels.is_empty())
        {
            self.mark_source_manifest_consumed_if_persisted(&manifest)
                .await;
            return None;
        }

        let channel_count = decoded_segments[0].channels.len();
        self.apply_automatic_gps_geometry(&mut manifest, &stream_key, channel_count);
        let source_ids = manifest
            .source_handles
            .iter()
            .map(|handle| handle.segment_id.clone())
            .collect::<Vec<_>>();
        let classification_window_sec = self
            .config
            .classification_window_seconds
            .max(MIN_BIRDNET_CLASSIFICATION_WINDOW_SECONDS)
            .max(self.config.window_seconds);
        let mic_positions_m = mic_positions_from_manifest(&manifest);
        let half_space = half_space_from_node_context(&manifest.node_context);
        let buffer_routing = resolve_cluster_buffer_routing(&manifest, &stream_key, channel_count);
        let (buffer_key, buffer_channel_count, buffer_channel_indices) = buffer_routing
            .map(|routing| {
                (
                    routing.buffer_key,
                    routing.buffer_channel_count,
                    routing.buffer_channel_indices,
                )
            })
            .unwrap_or_else(|| {
                (
                    stream_key.clone(),
                    channel_count,
                    (0..channel_count).collect(),
                )
            });
        let segment_count = decoded_segments.len();
        let mut manifest_for_final_segment = Some(manifest);
        let mut owned_segments = Vec::with_capacity(segment_count);
        for (segment_index, mut decoded) in decoded_segments.into_iter().enumerate() {
            {
                let node_id_key = stream_key.split("__").next().unwrap_or(&stream_key);
                let st = self.state.read().await;
                let cfg = st.node_audio_overrides.get(node_id_key).cloned();
                drop(st);
                let cfg = cfg.unwrap_or_else(|| self.config.default_audio_config.clone());
                if !cfg.effective_stages().is_empty() {
                    let node_state = self
                        .node_audio_state
                        .entry(node_id_key.to_string())
                        .or_default();
                    node_state.apply(&mut decoded.channels, &cfg, decoded.sample_rate_hz);
                }
            }

            let sr = decoded.sample_rate_hz.max(1);
            let (env_temp_c, env_humidity_fraction) =
                if decoded.temperature_c.is_none() && decoded.humidity_fraction.is_none() {
                    if let Some(cache) = &self.env_cache {
                        let query_ns = first_handle.toa_ns.unwrap_or(now_ns as u64);
                        let node_id = stream_key.split("__").next().unwrap_or(&stream_key);
                        cache
                            .interpolate(node_id, query_ns)
                            .await
                            .map(|(temperature_c, humidity_fraction)| {
                                (Some(temperature_c), Some(humidity_fraction))
                            })
                            .unwrap_or((None, None))
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                };

            let effective_temperature_c = decoded.temperature_c.or(env_temp_c);
            let effective_humidity_fraction = decoded.humidity_fraction.or(env_humidity_fraction);
            let effective_sound_speed_mps = resolve_effective_sound_speed_mps(
                &self.config,
                effective_temperature_c,
                effective_humidity_fraction,
            );

            let frames_all = decoded.channels.iter().map(Vec::len).min().unwrap_or(0);
            let render_duration_ns =
                (frames_all as i128).saturating_mul(1_000_000_000) / i128::from(sr.max(1));
            let start_time_ns = resolve_buffer_start_time_ns(&decoded, &first_handle, sr, now_ns);
            let node_timestamp_is_available =
                decoded.start_time_ns.is_some_and(|start_ns| start_ns > 0)
                    || first_handle.toa_ns.is_some();
            let skew_ns = (start_time_ns - now_ns as i128).unsigned_abs();
            let max_skew_ns =
                (self.config.max_trusted_node_clock_skew_seconds * 1_000_000_000.0).round() as u128;
            let buffer_uses_receipt_time = should_use_receipt_time_alignment(
                node_timestamp_is_available,
                skew_ns,
                max_skew_ns,
            );
            let (buffer_start_time_ns, buffer_end_time_ns) = if buffer_uses_receipt_time {
                let anchor_ns = first_handle.tor_ns.unwrap_or(now_ns as u64) as i128;
                let start_ns = anchor_ns.saturating_sub(render_duration_ns).max(1);
                (start_ns, anchor_ns)
            } else {
                (start_time_ns, start_time_ns + render_duration_ns)
            };

            let start_sample_index = decoded.start_sample_index;
            let end_sample_index = decoded.end_sample_index;
            let segment_manifest = if segment_index + 1 == segment_count {
                manifest_for_final_segment
                    .take()
                    .expect("final segment owns source manifest")
            } else {
                manifest_for_final_segment
                    .as_ref()
                    .expect("source manifest remains available")
                    .clone()
            };

            owned_segments.push(OwnedManifestAudio {
                manifest: segment_manifest,
                stream_key: stream_key.clone(),
                buffer_key: buffer_key.clone(),
                decoded,
                source_ids: source_ids.clone(),
                sr,
                now_ns,
                channel_count,
                buffer_channel_count,
                buffer_channel_indices: buffer_channel_indices.clone(),
                classification_window_sec,
                mic_positions_m: mic_positions_m.clone(),
                half_space,
                effective_sound_speed_mps,
                buffer_start_time_ns,
                buffer_end_time_ns,
                start_sample_index,
                end_sample_index,
                node_timestamp_is_available,
                buffer_uses_receipt_time,
                arrived_via_channel,
            });
        }
        Some(owned_segments)
    }

    fn apply_automatic_gps_geometry(
        &mut self,
        manifest: &mut DspManifest,
        stream_key: &str,
        channel_count: usize,
    ) {
        // Explicit cluster geometry is still an operator-directed boundary.
        // Otherwise every GPS-reporting node on the same audio stream joins the
        // automatic array, exactly as the Python global sensor pool does.
        if manifest.cluster_id.is_some() || manifest.cluster_sensor_positions.is_some() {
            return;
        }
        let node_id = stream_key_node_id(stream_key);
        let Some(sensor_positions) =
            self.automatic_gps_geometry
                .update_from_manifest(manifest, node_id, channel_count)
        else {
            return;
        };
        manifest.cluster_id = Some(AUTOMATIC_GPS_CLUSTER_ID.to_string());
        manifest.cluster_sensor_positions = Some(sensor_positions);
    }

    async fn publish_raw_audio_frame_for_owned_manifest(
        &self,
        owned: &OwnedManifestAudio,
        segment_index: usize,
        segment_count: usize,
    ) {
        publish_raw_audio_frame_event(RawAudioFramePublishRequest {
            publisher: &self.dsp_event_publisher,
            source_manifest: &owned.manifest,
            decoded: &owned.decoded,
            stream_key: &owned.stream_key,
            sample_rate_hz: owned.sr,
            start_time_ns: owned.buffer_start_time_ns,
            end_time_ns: owned.buffer_end_time_ns,
            created_ns: owned.now_ns,
            segment_index,
            segment_count,
        })
        .await;
    }

    async fn should_skip_stale_manifest_for_live_buffer(
        &mut self,
        owned: &OwnedManifestAudio,
    ) -> bool {
        let source_manifest_is_stale = manifest_is_older_than_buffer_horizon(
            &owned.manifest,
            owned.now_ns,
            self.config.max_buffer_seconds,
        );
        if source_manifest_is_stale
            && self.config.skip_stale_manifests_for_live_buffer
            && !owned.arrived_via_channel
        {
            debug!(
                manifest_id = %owned.manifest.manifest_id,
                "DSP worker skipped stale disk manifest to protect live buffer continuity"
            );
            self.defer_source_manifest_consumption(&owned.manifest);
            let mut st = self.state.write().await;
            st.total_stale_manifest_skips += 1;
            st.last_processed_ns = Some(owned.now_ns);
            return true;
        }
        false
    }

    async fn buffer_owned_manifest_audio(
        &mut self,
        owned: &OwnedManifestAudio,
    ) -> Option<BufferedManifestAudio> {
        let buffers = self
            .buffers
            .entry(owned.buffer_key.clone())
            .or_insert_with(|| {
                (0..owned.buffer_channel_count)
                    .map(|_| SensorStreamBuffer::new(owned.sr, self.config.max_buffer_seconds))
                    .collect()
            });
        while buffers.len() < owned.buffer_channel_count {
            buffers.push(SensorStreamBuffer::new(
                owned.sr,
                self.config.max_buffer_seconds,
            ));
        }
        let reference_buffer_index = owned.buffer_channel_indices.first().copied().unwrap_or(0);

        let existing_sample_timeline_start_time_ns =
            if !owned.node_timestamp_is_available && !owned.buffer_uses_receipt_time {
                owned.start_sample_index.and_then(|sample_index| {
                    buffers
                        .get(reference_buffer_index)
                        .and_then(|buffer| buffer.time_for_sample_index(sample_index))
                })
            } else {
                None
            };
        let (buffer_start_time_ns, buffer_end_time_ns) = if let Some(existing_start_time_ns) =
            existing_sample_timeline_start_time_ns
        {
            (
                existing_start_time_ns,
                existing_start_time_ns + (owned.buffer_end_time_ns - owned.buffer_start_time_ns),
            )
        } else {
            (owned.buffer_start_time_ns, owned.buffer_end_time_ns)
        };

        let mut reanchor_delta: u64 = 0;
        for (channel_index, &buffer_channel_index) in
            owned.buffer_channel_indices.iter().enumerate()
        {
            let Some(channel_samples) = owned.decoded.channels.get(channel_index) else {
                break;
            };
            let Some(buffer) = buffers.get_mut(buffer_channel_index) else {
                self.note_failure().await;
                warn!(
                    manifest_id = %owned.manifest.manifest_id,
                    stream_key = %owned.stream_key,
                    buffer_key = %owned.buffer_key,
                    buffer_channel_index,
                    "DSP worker resolved an invalid cluster buffer channel index"
                );
                self.defer_source_manifest_consumption(&owned.manifest);
                return None;
            };
            let pre_reanchor = buffer.reanchor_count();
            if let Err(err) = buffer.append(
                buffer_start_time_ns,
                channel_samples,
                owned.start_sample_index,
                owned.end_sample_index,
            ) {
                self.note_failure().await;
                warn!(
                    manifest_id = %owned.manifest.manifest_id,
                    channel = channel_index,
                    error = %err,
                    "DSP worker failed to append decoded audio; consuming malformed source manifest"
                );
                self.defer_source_manifest_consumption(&owned.manifest);
                return None;
            }
            reanchor_delta =
                reanchor_delta.saturating_add(buffer.reanchor_count().saturating_sub(pre_reanchor));
        }

        let end_ns = resolve_buffer_end_time_ns(
            buffers,
            owned.end_sample_index,
            buffer_end_time_ns,
            owned.sr,
            reference_buffer_index,
        );
        let center_time_ns = resolve_localization_center_time_ns(
            buffers,
            buffer_start_time_ns,
            end_ns,
            self.config.window_seconds,
        );
        let channel_states = localization_channel_states_centered(
            buffers,
            center_time_ns,
            self.config.window_seconds,
        );
        // `buffers` borrow ends here; safe to take &self again for the
        // shared-state write.
        if reanchor_delta > 0 {
            self.note_buffer_reanchors(reanchor_delta).await;
        }
        Some(BufferedManifestAudio {
            audio_end_ns: end_ns.max(0) as u128,
            end_ns,
            channel_states,
        })
    }

    fn dispatch_large_array_manifest(
        &mut self,
        owned: OwnedManifestAudio,
        pending_backlog_depth: usize,
        reason: &str,
    ) -> Option<ComputePayload> {
        if !self.should_publish_classifier_render(
            &owned.buffer_key,
            owned.now_ns,
            pending_backlog_depth,
        ) {
            self.defer_source_manifest_consumption(&owned.manifest);
            return None;
        }

        let omni_channels = owned.decoded.channels.clone();
        Some(ComputePayload {
            manifest: owned.manifest,
            stream_key: owned.buffer_key,
            channel_states: (0..owned.buffer_channel_count)
                .map(|_| LocalizationChannelState {
                    coverage: None,
                    window: Vec::new(),
                })
                .collect(),
            active_channels: Vec::new(),
            classification_windows: (0..owned.buffer_channel_count)
                .map(|_| Vec::new())
                .collect(),
            listenable_classification_windows: (0..owned.buffer_channel_count)
                .map(|_| Vec::new())
                .collect(),
            classification_coverage: (0..owned.buffer_channel_count).map(|_| None).collect(),
            classifier_render_start_ns: Some(owned.buffer_start_time_ns.max(0) as u128),
            classifier_render_end_ns: Some(owned.buffer_end_time_ns.max(0) as u128),
            mic_positions_m: owned.mic_positions_m,
            half_space: owned.half_space,
            source_ids: owned.source_ids,
            sr: owned.sr,
            now_ns: owned.now_ns,
            effective_sound_speed_mps: owned.effective_sound_speed_mps,
            reported_temperature_c: owned.decoded.temperature_c,
            reported_humidity_fraction: owned.decoded.humidity_fraction,
            reported_environment_source: owned.decoded.environment_source,
            run_srp: false,
            run_classifier_render: true,
            skip_localization_result: true,
            omni_fallback_reason: Some(reason.to_string()),
            omni_channels_override: Some(omni_channels.clone()),
            listenable_omni_channels_override: Some(omni_channels),
            manifest_store: self.manifest_store.clone(),
            derived_cache: self.derived_cache.clone(),
            state: self.state.clone(),
            classification_tx: self.classification_tx.clone(),
            dsp_event_publisher: self.dsp_event_publisher.clone(),
            config: self.config.clone(),
            consumed_since_prune: self.consumed_manifests_since_prune.clone(),
            diagnostics: self.diagnostics.clone(),
        })
    }

    fn resolve_buffered_manifest_timing_gates(
        &mut self,
        stream_key: &str,
        channel_count: usize,
        audio_end_ns: u128,
        pending_backlog_depth: usize,
        windows: &[Vec<f32>],
    ) -> BufferedManifestTimingGates {
        let on_localization_cadence =
            self.should_run_localization(stream_key, audio_end_ns, windows);
        let run_classifier_render =
            self.should_publish_classifier_render(stream_key, audio_end_ns, pending_backlog_depth);
        let run_srp = channel_count >= 4 && (on_localization_cadence || run_classifier_render);
        let on_heartbeat_cadence =
            on_localization_cadence || self.should_emit_heartbeat(stream_key, audio_end_ns);
        BufferedManifestTimingGates {
            on_heartbeat_cadence,
            run_classifier_render,
            run_srp,
        }
    }

    async fn dispatch_buffered_manifest(
        &mut self,
        owned: OwnedManifestAudio,
        buffered: BufferedManifestAudio,
        pending_backlog_depth: usize,
    ) -> Option<ComputePayload> {
        if buffered
            .channel_states
            .iter()
            .all(|state| state.coverage.is_none())
        {
            let run_classifier_render = self.should_publish_classifier_render(
                &owned.buffer_key,
                buffered.audio_end_ns,
                pending_backlog_depth,
            );
            if run_classifier_render {
                debug!(
                    manifest_id = %owned.manifest.manifest_id,
                    "DSP worker: no localization coverage window after buffering; publishing omni fallback render"
                );
                let Some(buffers) = self.buffers.get(&owned.buffer_key) else {
                    error!(stream_key = %owned.stream_key, buffer_key = %owned.buffer_key, "stream buffers missing after append in omni fallback path — skipping");
                    return None;
                };
                let fallback_render_windows = channel_windows_ending_at(
                    buffers,
                    buffered.end_ns,
                    owned.classification_window_sec,
                );
                let listenable_fallback_render_windows =
                    channel_windows_ending_at_with_gap_concealment(
                        buffers,
                        buffered.end_ns,
                        owned.classification_window_sec,
                    );
                let fallback_render_coverage = channel_coverage_ending_at(
                    buffers,
                    buffered.end_ns,
                    owned.classification_window_sec,
                );
                let fallback_render_channels = if fallback_render_windows
                    .iter()
                    .any(|window| !window.is_empty())
                {
                    fallback_render_windows.clone()
                } else {
                    owned.decoded.channels.clone()
                };
                let listenable_fallback_render_channels = if listenable_fallback_render_windows
                    .iter()
                    .any(|window| !window.is_empty())
                {
                    listenable_fallback_render_windows.clone()
                } else {
                    fallback_render_channels.clone()
                };
                let (classifier_render_start_ns, classifier_render_end_ns) =
                    classifier_render_bounds_from_windows(
                        buffered.end_ns,
                        &fallback_render_channels,
                        owned.sr,
                    );
                return Some(ComputePayload {
                    manifest: owned.manifest,
                    stream_key: owned.buffer_key,
                    channel_states: buffered.channel_states,
                    active_channels: eligible_coverage_channels(
                        &fallback_render_coverage,
                        self.config.min_coverage_ratio,
                    ),
                    classification_windows: (0..owned.buffer_channel_count)
                        .map(|_| Vec::new())
                        .collect(),
                    listenable_classification_windows: (0..owned.buffer_channel_count)
                        .map(|_| Vec::new())
                        .collect(),
                    classification_coverage: fallback_render_coverage,
                    classifier_render_start_ns,
                    classifier_render_end_ns,
                    mic_positions_m: owned.mic_positions_m,
                    half_space: owned.half_space,
                    source_ids: owned.source_ids,
                    sr: owned.sr,
                    now_ns: owned.now_ns,
                    effective_sound_speed_mps: owned.effective_sound_speed_mps,
                    reported_temperature_c: owned.decoded.temperature_c,
                    reported_humidity_fraction: owned.decoded.humidity_fraction,
                    reported_environment_source: owned.decoded.environment_source,
                    run_srp: false,
                    run_classifier_render: true,
                    skip_localization_result: true,
                    omni_fallback_reason: Some("localization_coverage_unavailable".to_string()),
                    omni_channels_override: Some(fallback_render_channels),
                    listenable_omni_channels_override: Some(listenable_fallback_render_channels),
                    manifest_store: self.manifest_store.clone(),
                    derived_cache: self.derived_cache.clone(),
                    state: self.state.clone(),
                    classification_tx: self.classification_tx.clone(),
                    dsp_event_publisher: self.dsp_event_publisher.clone(),
                    config: self.config.clone(),
                    consumed_since_prune: self.consumed_manifests_since_prune.clone(),
                    diagnostics: self.diagnostics.clone(),
                });
            }
        }

        let active_channels = eligible_localization_channels(
            &buffered.channel_states,
            self.config.min_coverage_ratio,
        );
        let windows: Vec<Vec<f32>> = buffered
            .channel_states
            .iter()
            .map(|state| state.window.clone())
            .collect();
        let timing_gates = self.resolve_buffered_manifest_timing_gates(
            &owned.buffer_key,
            owned.buffer_channel_count,
            buffered.audio_end_ns,
            pending_backlog_depth,
            &windows,
        );

        if !timing_gates.on_heartbeat_cadence && !timing_gates.run_classifier_render {
            // Parity with Python's localization "no_window" silent-drop reason:
            // when every per-channel window is empty AND no cadence wants a
            // result, the frame is dropped invisibly. The Python pipeline now
            // counts the corresponding state via
            // `localization_drops_by_reason["no_window"]`; bump the equivalent
            // here so a Rust-backed deployment exposes the same signal via
            // /api/v1/dsp/status.total_window_underrun_drops. Heartbeat /
            // classifier-render dispatches are intentional emissions and not
            // counted as a drop.
            let all_windows_empty = windows.iter().all(|window| window.is_empty());
            if all_windows_empty {
                self.note_window_underrun_drop().await;
            }
            self.defer_source_manifest_consumption(&owned.manifest);
            return None;
        }

        let (classification_windows, listenable_classification_windows, classification_coverage) =
            if timing_gates.run_classifier_render {
                let Some(buffers) = self.buffers.get(&owned.buffer_key) else {
                    error!(stream_key = %owned.stream_key, buffer_key = %owned.buffer_key, "stream buffers missing after append in classifier render path — skipping");
                    return None;
                };
                (
                    channel_windows_ending_at(
                        buffers,
                        buffered.end_ns,
                        owned.classification_window_sec,
                    ),
                    channel_windows_ending_at_with_gap_concealment(
                        buffers,
                        buffered.end_ns,
                        owned.classification_window_sec,
                    ),
                    channel_coverage_ending_at(
                        buffers,
                        buffered.end_ns,
                        owned.classification_window_sec,
                    ),
                )
            } else {
                (
                    buffered.channel_states.iter().map(|_| Vec::new()).collect(),
                    buffered.channel_states.iter().map(|_| Vec::new()).collect(),
                    buffered.channel_states.iter().map(|_| None).collect(),
                )
            };
        let (classifier_render_start_ns, classifier_render_end_ns) =
            if timing_gates.run_classifier_render {
                classifier_render_bounds_from_windows(
                    buffered.end_ns,
                    &classification_windows,
                    owned.sr,
                )
            } else {
                (None, None)
            };

        Some(ComputePayload {
            manifest: owned.manifest,
            stream_key: owned.buffer_key,
            channel_states: buffered.channel_states,
            active_channels,
            classification_windows,
            listenable_classification_windows,
            classification_coverage,
            classifier_render_start_ns,
            classifier_render_end_ns,
            mic_positions_m: owned.mic_positions_m,
            half_space: owned.half_space,
            source_ids: owned.source_ids,
            sr: owned.sr,
            now_ns: owned.now_ns,
            effective_sound_speed_mps: owned.effective_sound_speed_mps,
            reported_temperature_c: owned.decoded.temperature_c,
            reported_humidity_fraction: owned.decoded.humidity_fraction,
            reported_environment_source: owned.decoded.environment_source,
            run_srp: timing_gates.run_srp,
            run_classifier_render: timing_gates.run_classifier_render,
            skip_localization_result: false,
            omni_fallback_reason: (owned.buffer_channel_count < 4)
                .then(|| "single_point_node".to_string()),
            omni_channels_override: None,
            listenable_omni_channels_override: None,
            manifest_store: self.manifest_store.clone(),
            derived_cache: self.derived_cache.clone(),
            state: self.state.clone(),
            classification_tx: self.classification_tx.clone(),
            dsp_event_publisher: self.dsp_event_publisher.clone(),
            config: self.config.clone(),
            consumed_since_prune: self.consumed_manifests_since_prune.clone(),
            diagnostics: self.diagnostics.clone(),
        })
    }

    #[cfg(test)]
    pub(crate) async fn process_one(
        &mut self,
        manifest: DspManifest,
        pending_backlog_depth: usize,
    ) {
        if let Some(payload) = self.ingest_one(manifest, pending_backlog_depth).await {
            self.note_stage_attempts(&payload).await;
            let result = crate::actors::dsp_compute::run_math(payload);
            crate::actors::dsp_compute::run_io(result).await;
        }
        self.flush_deferred_source_manifest_consumptions().await;
    }

    fn defer_source_manifest_consumption(&mut self, manifest: &DspManifest) {
        if !source_manifest_was_persisted(manifest) {
            return;
        }
        self.deferred_source_manifest_ids
            .push(manifest.manifest_id.clone());
    }

    async fn mark_source_manifest_consumed_if_persisted(&self, manifest: &DspManifest) {
        if !source_manifest_was_persisted(manifest) {
            return;
        }
        let _ = self
            .manifest_store
            .mark_consumed(&manifest.manifest_id)
            .await;
    }

    async fn flush_deferred_source_manifest_consumptions(&mut self) {
        if self.deferred_source_manifest_ids.is_empty() {
            return;
        }
        let manifest_ids = std::mem::take(&mut self.deferred_source_manifest_ids);
        let mut unique_manifest_ids = HashSet::with_capacity(manifest_ids.len());
        for id in manifest_ids {
            unique_manifest_ids.insert(id);
        }

        // Run all renames concurrently — on tmpfs these are in-memory directory
        // entry updates and complete in microseconds even at batch depth 128.
        let mut set = tokio::task::JoinSet::new();
        for manifest_id in unique_manifest_ids {
            let store = self.manifest_store.clone();
            set.spawn(async move {
                let result = store.mark_consumed(&manifest_id).await;
                (manifest_id, result)
            });
        }
        let mut successful_consumes = 0_u64;
        while let Some(result) = set.join_next().await {
            match result {
                Ok((_, Ok(()))) => successful_consumes += 1,
                Ok((manifest_id, Err(err))) => {
                    warn!(
                        manifest_id = %manifest_id,
                        error = %err,
                        "DSP worker failed to mark manifest consumed"
                    );
                }
                Err(join_err) => {
                    warn!(error = %join_err, "mark_consumed task panicked");
                }
            }
        }

        if successful_consumes == 0 {
            return;
        }

        let prune_interval = self.config.consumed_manifest_prune_interval.max(1);
        let previous = self
            .consumed_manifests_since_prune
            .fetch_add(successful_consumes, Ordering::Relaxed);
        let current = previous.saturating_add(successful_consumes);
        let count_due = previous / prune_interval < current / prune_interval;
        let prune_max_age_ns =
            (self.config.consumed_manifest_prune_max_age_seconds as u128) * 1_000_000_000;
        let time_due =
            system_now_ns().saturating_sub(self.last_consumed_prune_ns) >= prune_max_age_ns;
        if count_due || time_due {
            self.last_consumed_prune_ns = system_now_ns();
            let store = self.manifest_store.clone();
            let retention_max_files = self.config.consumed_manifest_retention_max_files;
            tokio::spawn(async move {
                if let Err(error) = store.prune_consumed_manifests(retention_max_files).await {
                    warn!(error = %error, "DSP worker failed to prune consumed manifests");
                }
            });
        }
    }

    async fn note_failures(&self, count: u64) {
        if count == 0 {
            return;
        }
        let mut st = self.state.write().await;
        st.total_failures += count;
    }

    async fn note_failure(&self) {
        self.note_failures(1).await;
    }

    /// Cross-process diagnostic counter mirroring the Python
    /// `SensorStreamBuffer.reanchor_count`. Surfaced via `/api/v1/dsp/status`
    /// so an operator alerting on long-uptime timeline drift sees the same
    /// signal whether the deployment is using the Python or the Rust ingest.
    async fn note_buffer_reanchors(&self, count: u64) {
        if count == 0 {
            return;
        }
        let mut st = self.state.write().await;
        st.total_buffer_reanchors = st.total_buffer_reanchors.saturating_add(count);
    }

    /// Evict per-stream entries (buffers, audio state, cadence timestamps) for
    /// streams that have not produced a heartbeat in `stream_inactivity_evict_seconds`.
    /// Mirrors the leases.rs `purge_expired_*` idiom. Called once per minute from
    /// `run_loop` to prevent RSS growth from node-ID churn (reflash, hostname change).
    async fn purge_stale_streams(&mut self) {
        let ttl_ns = (self.config.stream_inactivity_evict_seconds as u128) * 1_000_000_000;
        let now_ns = system_now_ns();
        let cutoff = now_ns.saturating_sub(ttl_ns);
        let stale: Vec<String> = self
            .last_heartbeat_ns_by_stream
            .iter()
            .filter(|(_, v)| **v < cutoff)
            .map(|(k, _)| k.clone())
            .collect();
        if stale.is_empty() {
            return;
        }
        for key in &stale {
            self.buffers.remove(key);
            self.last_classifier_render_ns_by_stream.remove(key);
            self.last_localization_ns_by_stream.remove(key);
            self.last_trigger_ns_by_stream.remove(key);
            self.last_heartbeat_ns_by_stream.remove(key);
            // node_audio_state is keyed by node_id (prefix before "__"), not stream_key.
            // Only remove if no remaining stream for that node still active.
            let node_id = key.split("__").next().unwrap_or(key.as_str());
            let node_still_active = self
                .last_heartbeat_ns_by_stream
                .keys()
                .any(|k| k.starts_with(node_id));
            if !node_still_active {
                self.node_audio_state.remove(node_id);
            }
        }
        let evicted = stale.len() as u64;
        let mut st = self.state.write().await;
        st.total_stale_streams_evicted = st.total_stale_streams_evicted.saturating_add(evicted);
        info!(count = evicted, "Evicted stale per-stream entries");
    }

    /// Mirror of the Python localization `"no_window"` drop reason. Incremented
    /// when `compute_manifest_frame` (or its callers) cannot extract an audio
    /// window for a manifest because the requested time is outside the buffer
    /// coverage. This is the single canonical "silent stall" signal — paired
    /// with `total_buffer_reanchors` it explains *why* the stall happened.
    async fn note_window_underrun_drop(&self) {
        let mut st = self.state.write().await;
        st.total_window_underrun_drops = st.total_window_underrun_drops.saturating_add(1);
    }

    async fn note_stage_attempts(&self, payload: &ComputePayload) {
        if !payload.run_srp && !payload.run_classifier_render {
            return;
        }
        let mut st = self.state.write().await;
        if payload.run_srp {
            st.total_localization_attempts += 1;
        }
        if payload.run_classifier_render {
            st.total_classification_attempts += 1;
        }
    }

    fn should_run_localization(
        &mut self,
        stream_key: &str,
        audio_ns: u128,
        windows: &[Vec<f32>],
    ) -> bool {
        if !self.should_trigger(stream_key, audio_ns, windows) {
            return false;
        }
        let cadence_ns = (self.config.localization_cadence_ms as u128) * 1_000_000;
        if cadence_ns == 0 {
            return true;
        }
        match self.last_localization_ns_by_stream.get(stream_key).copied() {
            Some(last_ns) if audio_ns.saturating_sub(last_ns) < cadence_ns => false,
            _ => {
                self.last_localization_ns_by_stream
                    .insert(stream_key.to_string(), audio_ns);
                true
            }
        }
    }

    /// Combined RMS energy gate + trigger cooldown, matching Python's IngestProcessor
    /// trigger evaluation. Returns true when the audio is loud enough and the cooldown
    /// period has elapsed since the last trigger for this stream.
    fn should_trigger(&mut self, stream_key: &str, audio_ns: u128, windows: &[Vec<f32>]) -> bool {
        if self.config.localization_rms_gate > 0.0 {
            let max_rms = windows
                .iter()
                .map(|w| {
                    if w.is_empty() {
                        0.0_f32
                    } else {
                        (w.iter().map(|s| s * s).sum::<f32>() / w.len() as f32).sqrt()
                    }
                })
                .fold(0.0_f32, f32::max);
            if max_rms < self.config.localization_rms_gate {
                return false;
            }
        }
        let cooldown_ns = (self.config.trigger_cooldown_seconds * 1_000_000_000.0).round() as u128;
        if cooldown_ns == 0 {
            return true;
        }
        match self.last_trigger_ns_by_stream.get(stream_key).copied() {
            Some(last_ns) if audio_ns.saturating_sub(last_ns) < cooldown_ns => false,
            _ => {
                self.last_trigger_ns_by_stream
                    .insert(stream_key.to_string(), audio_ns);
                true
            }
        }
    }

    /// Time-only cadence check for node liveness heartbeats.
    /// Never blocked by the RMS energy gate or trigger cooldown.
    fn should_emit_heartbeat(&mut self, stream_key: &str, audio_ns: u128) -> bool {
        let cadence_ns = (self.config.localization_cadence_ms as u128) * 1_000_000;
        if cadence_ns == 0 {
            return true;
        }
        match self.last_heartbeat_ns_by_stream.get(stream_key).copied() {
            Some(last_ns) if audio_ns.saturating_sub(last_ns) < cadence_ns => false,
            _ => {
                self.last_heartbeat_ns_by_stream
                    .insert(stream_key.to_string(), audio_ns);
                true
            }
        }
    }

    fn should_publish_classifier_render(
        &mut self,
        stream_key: &str,
        audio_ns: u128,
        pending_backlog_depth: usize,
    ) -> bool {
        let min_interval_ns = classifier_render_min_interval_ns(
            self.config.classifier_render_min_interval_seconds,
            pending_backlog_depth,
        );
        if min_interval_ns == 0 {
            return true;
        }
        match self
            .last_classifier_render_ns_by_stream
            .get(stream_key)
            .copied()
        {
            Some(last_ns) if audio_ns.saturating_sub(last_ns) < min_interval_ns => false,
            _ => {
                self.last_classifier_render_ns_by_stream
                    .insert(stream_key.to_string(), audio_ns);
                true
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Free functions shared between the ingest path and actors/dsp_compute
// ---------------------------------------------------------------------------

pub(crate) async fn dispatch_classification_result_standalone(
    result: crate::dsp_render_output::RenderPublishResult,
    classification_tx: &Option<flume::Sender<ClassificationRequest>>,
    state: &SharedDspState,
    raw_render_bytes: Option<String>,
) -> Option<DspManifest> {
    let pending = result.pending_manifest?;
    if let Some(tx) = classification_tx {
        let pcm_bytes = result.pcm_bytes.unwrap_or_default();
        let mut pending_with_bytes = pending.clone();
        pending_with_bytes.raw_render_bytes = raw_render_bytes.clone();
        let req = ClassificationRequest {
            pcm_bytes,
            sample_rate_hz: result.sample_rate_hz,
            pending_manifest: pending_with_bytes,
            raw_render_bytes: raw_render_bytes.clone(),
        };
        match tx.try_send(req) {
            Ok(()) => {
                // ClassificationWorker will publish the single classifier_render event.
                return None;
            }
            Err(flume::TrySendError::Full(_)) => {
                let mut st = state.write().await;
                st.total_classification_drops += 1;
                warn!(
                    drops = st.total_classification_drops,
                    "ClassificationWorker channel full; returning manifest without BirdNET labels"
                );
                // No disk write — caller broadcasts via SSE without labels.
            }
            Err(flume::TrySendError::Disconnected(_)) => {
                warn!(
                    "ClassificationWorker channel closed; returning manifest without BirdNET labels"
                );
                // No disk write — caller broadcasts via SSE without labels.
            }
        }
    }
    let mut fallback = pending;
    fallback.raw_render_bytes = raw_render_bytes;
    // No ClassificationWorker configured — return manifest unlabeled; caller broadcasts via SSE.
    Some(fallback)
}

async fn publish_raw_audio_frame_event(request: RawAudioFramePublishRequest<'_>) {
    let RawAudioFramePublishRequest {
        publisher,
        source_manifest,
        decoded,
        stream_key,
        sample_rate_hz,
        start_time_ns,
        end_time_ns,
        created_ns,
        segment_index,
        segment_count,
    } = request;
    let Some(publisher) = publisher else {
        return;
    };
    let Some(raw_pcm_bytes) = encode_interleaved_pcm16le(&decoded.channels) else {
        return;
    };
    let channel_count = decoded.channels.len();
    let sample_count = decoded.channels.iter().map(Vec::len).min().unwrap_or(0);
    // raw_audio_frame is a freshly decoded contiguous payload — every sample
    // is present at this point, so coverage is all-true. Matches Python's
    // AudioCoverageStats.to_json() 9-field shape at audio_buffer.py:30.
    let frame_coverage = vec![true; sample_count];
    let coverage_stats_json =
        serde_json::to_value(coverage_stats(&frame_coverage, sample_rate_hz.max(1))).ok();
    let raw_audio_manifest_id = if segment_count == 1 {
        format!("raw-audio-{}", source_manifest.manifest_id)
    } else {
        format!(
            "raw-audio-{}-segment-{}",
            source_manifest.manifest_id, segment_index
        )
    };
    let raw_manifest = DspManifest {
        manifest_id: raw_audio_manifest_id,
        manifest_type: "raw_audio_frame".to_string(),
        created_ns,
        source_handles: source_manifest.source_handles.clone(),
        derived_handle: None,
        localization: None,
        classifier_render: None,
        birdnet: None,
        coverage_stats: coverage_stats_json,
        promotion_ready: false,
        env_samples: None,
        node_context: source_manifest.node_context.clone(),
        cluster_id: source_manifest.cluster_id.clone(),
        cluster_sensor_positions: source_manifest.cluster_sensor_positions.clone(),
        raw_payload: None,
        raw_render_bytes: None,
        raw_audio_frame: Some(serde_json::json!({
            "stream_key": stream_key,
            "sample_rate_hz": sample_rate_hz,
            "channel_count": channel_count,
            "channels": channel_count,
            "sample_count": sample_count,
            "sample_format": "pcm16le",
            "start_time_ns": start_time_ns.max(0) as u128,
            "end_time_ns": end_time_ns.max(0) as u128,
            "start_sample_index": decoded.start_sample_index,
            "end_sample_index": decoded.end_sample_index,
            "source_manifest_id": source_manifest.manifest_id,
        })),
        raw_audio_bytes: Some(BASE64.encode(raw_pcm_bytes)),
    };
    let _ = publisher.publish(raw_manifest).await;
}

fn encode_interleaved_pcm16le(channels: &[Vec<f32>]) -> Option<Vec<u8>> {
    let sample_count = channels.iter().map(Vec::len).min().unwrap_or(0);
    if channels.is_empty() || sample_count == 0 {
        return None;
    }
    let mut raw = Vec::with_capacity(sample_count * channels.len() * std::mem::size_of::<i16>());
    for sample_index in 0..sample_count {
        for channel in channels {
            let scaled = (channel[sample_index].clamp(-1.0, 1.0) * 32767.0).round() as i16;
            raw.extend_from_slice(&scaled.to_le_bytes());
        }
    }
    Some(raw)
}

fn merge_pending_manifests_for_batch(
    channel_manifests: Vec<DspManifest>,
    disk_manifests: Vec<DspManifest>,
    batch_limit: usize,
) -> Vec<DspManifest> {
    let mut dedupe_manifest_ids = HashSet::new();
    let mut merged = Vec::with_capacity(channel_manifests.len() + disk_manifests.len());
    for manifest in channel_manifests.into_iter().chain(disk_manifests) {
        if dedupe_manifest_ids.insert(manifest.manifest_id.clone()) {
            merged.push(manifest);
        }
    }

    // Prioritize freshest manifests so live audio stays current even while a
    // backlog exists in either source. This keeps the 30-second BirdNET window
    // populated with recent audio rather than draining an old backlog first.
    merged.sort_unstable_by(|left, right| right.created_ns.cmp(&left.created_ns));
    merged.truncate(batch_limit);
    merged
}

fn should_sleep_after_poll_cycle(processed_manifest_count: usize) -> bool {
    processed_manifest_count == 0
}

pub(crate) async fn consume_manifest_standalone(
    manifest: &DspManifest,
    manifest_store: &ManifestStore,
    consumed_since_prune: &Arc<AtomicU64>,
    prune_interval: u64,
    retention_max_files: usize,
) {
    if !source_manifest_was_persisted(manifest) {
        return;
    }
    if let Err(err) = manifest_store.mark_consumed(&manifest.manifest_id).await {
        warn!(
            manifest_id = %manifest.manifest_id,
            error = %err,
            "DSP worker failed to mark manifest consumed"
        );
        return;
    }
    let count = consumed_since_prune.fetch_add(1, Ordering::Relaxed) + 1;
    if count.is_multiple_of(prune_interval) {
        let store = manifest_store.clone();
        tokio::spawn(async move {
            if let Err(error) = store.prune_consumed_manifests(retention_max_files).await {
                warn!(error = %error, "DSP worker failed to prune consumed manifests");
            }
        });
    }
}

pub(crate) fn source_manifest_was_persisted(manifest: &DspManifest) -> bool {
    // Manifests read from disk never include raw_payload because the field is
    // serde-skipped. Channel-delivered manifests include raw_payload and do not
    // require mark_consumed filesystem operations.
    manifest.raw_payload.is_none()
}

pub(crate) fn render_coverage_json(
    channel_coverage: &[Option<AudioCoverageStats>],
    active_channels: &[usize],
    min_coverage_ratio: f64,
    window_type: &str,
) -> Option<serde_json::Value> {
    serde_json::to_value(serde_json::json!({
        "window_type": window_type,
        "per_channel": channel_coverage.to_vec(),
        "active_channels": active_channels,
        "threshold": min_coverage_ratio,
    }))
    .ok()
}

// ---------------------------------------------------------------------------
// Pure helper functions
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn localization_channel_states(
    buffers: &[SensorStreamBuffer],
    end_ns: i128,
    window_seconds: f64,
) -> Vec<LocalizationChannelState> {
    buffers
        .iter()
        .map(|buf| LocalizationChannelState {
            coverage: buf.coverage_ending_at(end_ns, window_seconds),
            window: buf
                .window_ending_at(end_ns, window_seconds)
                .unwrap_or_default(),
        })
        .collect()
}

/// Build localization channel states using centered windows (matching Python's
/// `get_window(center_time_ns, window_seconds)`). TDOA is computed relative to
/// the window center, so centering on the frame midpoint gives symmetric coverage.
fn localization_channel_states_centered(
    buffers: &[SensorStreamBuffer],
    center_time_ns: i128,
    window_seconds: f64,
) -> Vec<LocalizationChannelState> {
    buffers
        .iter()
        .map(|buf| LocalizationChannelState {
            coverage: buf.coverage_centered_at(center_time_ns, window_seconds),
            window: buf
                .window_centered_at(center_time_ns, window_seconds)
                .unwrap_or_default(),
        })
        .collect()
}

fn resolve_localization_center_time_ns(
    buffers: &[SensorStreamBuffer],
    buffer_start_time_ns: i128,
    end_ns: i128,
    window_seconds: f64,
) -> i128 {
    let window_duration_ns = (window_seconds * 1_000_000_000.0).round() as i128;
    let half_window_duration_ns = window_duration_ns / 2;
    let buffered_range_start_ns = buffers
        .first()
        .and_then(|buffer| buffer.start_time_ns())
        .unwrap_or(buffer_start_time_ns);
    let buffered_range_end_ns = buffers
        .first()
        .and_then(|buffer| buffer.end_time_ns())
        .unwrap_or(end_ns);
    let unclamped_center_time_ns = end_ns.saturating_sub(half_window_duration_ns);
    let centered_window_start_limit_ns =
        buffered_range_start_ns.saturating_add(half_window_duration_ns);
    let centered_window_end_limit_ns =
        buffered_range_end_ns.saturating_sub(half_window_duration_ns);

    if centered_window_start_limit_ns <= centered_window_end_limit_ns {
        unclamped_center_time_ns.clamp(centered_window_start_limit_ns, centered_window_end_limit_ns)
    } else {
        unclamped_center_time_ns
    }
}

fn channel_windows_ending_at(
    buffers: &[SensorStreamBuffer],
    end_ns: i128,
    window_seconds: f64,
) -> Vec<Vec<f32>> {
    buffers
        .iter()
        .map(|buf| {
            buf.window_ending_at(end_ns, window_seconds)
                .unwrap_or_default()
        })
        .collect()
}

fn channel_windows_ending_at_with_gap_concealment(
    buffers: &[SensorStreamBuffer],
    end_ns: i128,
    window_seconds: f64,
) -> Vec<Vec<f32>> {
    buffers
        .iter()
        .map(|buf| {
            buf.window_ending_at_with_gap_concealment(end_ns, window_seconds)
                .unwrap_or_default()
        })
        .collect()
}

fn channel_coverage_ending_at(
    buffers: &[SensorStreamBuffer],
    end_ns: i128,
    window_seconds: f64,
) -> Vec<Option<AudioCoverageStats>> {
    buffers
        .iter()
        .map(|buf| buf.coverage_ending_at(end_ns, window_seconds))
        .collect()
}

fn classifier_render_bounds_from_windows(
    end_ns: i128,
    windows: &[Vec<f32>],
    sample_rate_hz: u32,
) -> (Option<u128>, Option<u128>) {
    let sample_count = windows.iter().map(|window| window.len()).max().unwrap_or(0);
    if sample_count == 0 || sample_rate_hz == 0 {
        return (None, None);
    }
    let duration_ns =
        ((sample_count as f64 / sample_rate_hz as f64) * 1_000_000_000.0).round() as i128;
    let render_end_ns = end_ns.max(0);
    let render_start_ns = render_end_ns.saturating_sub(duration_ns).max(0);
    (Some(render_start_ns as u128), Some(render_end_ns as u128))
}

fn eligible_localization_channels(
    channel_states: &[LocalizationChannelState],
    min_coverage_ratio: f64,
) -> Vec<usize> {
    channel_states
        .iter()
        .enumerate()
        .filter_map(|(channel_index, state)| {
            let coverage = state.coverage.as_ref()?;
            (coverage.coverage_ratio >= min_coverage_ratio && !state.window.is_empty())
                .then_some(channel_index)
        })
        .collect()
}

fn eligible_coverage_channels(
    channel_coverage: &[Option<AudioCoverageStats>],
    min_coverage_ratio: f64,
) -> Vec<usize> {
    channel_coverage
        .iter()
        .enumerate()
        .filter_map(|(channel_index, coverage)| {
            let coverage = coverage.as_ref()?;
            (coverage.coverage_ratio >= min_coverage_ratio).then_some(channel_index)
        })
        .collect()
}

/// Extract per-channel mic positions from a manifest, checking cluster positions
/// first, then falling back to node_context.node.sensor_offsets_m, and finally
/// to the hardcoded Sirith tetrahedral positions for legacy payloads.
pub(crate) fn mic_positions_from_manifest(
    manifest: &crate::manifests::DspManifest,
) -> Vec<[f32; 3]> {
    // Cluster-resolved positions are authoritative when present.
    if let Some(ref cluster_pos) = manifest.cluster_sensor_positions {
        if !cluster_pos.is_empty() {
            return cluster_pos.iter().map(|(_, pos)| *pos).collect();
        }
    }
    mic_positions_from_node_context(&manifest.node_context)
}

/// Extract per-channel mic positions from node_context.node.sensor_offsets_m.
/// Falls back to the hardcoded Sirith tetrahedral positions if absent.
pub(crate) fn mic_positions_from_node_context(
    node_context: &Option<serde_json::Value>,
) -> Vec<[f32; 3]> {
    let positions = node_context
        .as_ref()
        .and_then(|ctx| ctx.get("node"))
        .and_then(|node| node.get("sensor_offsets_m"))
        .and_then(|v| v.as_array())
        .and_then(|arr| {
            arr.iter()
                .map(|v| {
                    let arr = v.as_array()?;
                    if arr.len() < 3 {
                        return None;
                    }
                    Some([
                        arr[0].as_f64()? as f32,
                        arr[1].as_f64()? as f32,
                        arr[2].as_f64()? as f32,
                    ])
                })
                .collect::<Option<Vec<[f32; 3]>>>()
        })
        .filter(|v| !v.is_empty());
    positions.unwrap_or_else(|| SIRITH_MIC_POSITIONS_M.to_vec())
}

/// Extract the coplanar half-space constraint from node_context.node.half_space
/// ("upper" | "lower" | "none"). Absent or unrecognized values default to
/// `HalfSpace::None` (no constraint), matching non-planar arrays.
pub(crate) fn half_space_from_node_context(
    node_context: &Option<serde_json::Value>,
) -> crate::srp_phat::HalfSpace {
    node_context
        .as_ref()
        .and_then(|ctx| ctx.get("node"))
        .and_then(|node| node.get("half_space"))
        .and_then(|v| v.as_str())
        .map(crate::srp_phat::HalfSpace::from_wire_str)
        .unwrap_or(crate::srp_phat::HalfSpace::None)
}

fn resolve_effective_sound_speed_mps(
    config: &DspWorkerConfig,
    temperature_c: Option<f32>,
    humidity_fraction: Option<f32>,
) -> f32 {
    if temperature_c.is_none() && humidity_fraction.is_none() {
        return config.sound_speed_mps;
    }
    speed_of_sound_mps(
        temperature_c.unwrap_or(config.default_temperature_c),
        humidity_fraction.unwrap_or(config.default_humidity_fraction),
    )
}

fn resolve_buffer_start_time_ns(
    decoded: &DecodedAudioPayload,
    first_handle: &JournalPayloadHandle,
    sample_rate_hz: u32,
    now_ns: u128,
) -> i128 {
    decoded
        .start_time_ns
        .filter(|start_time_ns| *start_time_ns > 0)
        .or_else(|| first_handle.toa_ns.map(i128::from))
        .or_else(|| {
            let anchor_ns = first_handle
                .toa_ns
                .unwrap_or_else(|| first_handle.tor_ns.unwrap_or(now_ns as u64));
            decoded.start_sample_index.map(|sample_index| {
                sample_index_to_absolute_time_from_now_ns(
                    sample_index,
                    sample_rate_hz,
                    anchor_ns.into(),
                )
            })
        })
        .or_else(|| {
            let anchor_ns = first_handle
                .toa_ns
                .unwrap_or_else(|| first_handle.tor_ns.unwrap_or(now_ns as u64));
            first_handle.sample_index_start.map(|sample_index| {
                sample_index_to_absolute_time_from_now_ns(
                    sample_index as i64,
                    sample_rate_hz,
                    anchor_ns.into(),
                )
            })
        })
        .or_else(|| first_handle.tor_ns.map(i128::from))
        .unwrap_or(now_ns as i128)
}

fn resolve_buffer_end_time_ns(
    buffers: &[SensorStreamBuffer],
    end_sample_index: Option<i64>,
    fallback_end_time_ns: i128,
    _sample_rate_hz: u32,
    reference_buffer_index: usize,
) -> i128 {
    // The buffer is anchored to the sample-index timeline, so windowing must
    // ride that timeline regardless of how far the publish-time TOA has drifted.
    // GPS-disciplined nodes recompute utcStart from a freshly-sampled clock
    // correction on every publish (NodeRunner::buildFrameForPacket), which can
    // shift consecutive packets' TOAs by tens of milliseconds while the sample
    // counter advances at a fixed rate. Trusting TOA here makes
    // coverage_centered_at / window_ending_at overshoot the buffer tail and
    // return None, which silently drops the entire classifier_render.
    end_sample_index
        .and_then(|sample_index| {
            buffers
                .get(reference_buffer_index)
                .or_else(|| buffers.first())
                .and_then(|buffer| buffer.time_for_sample_index(sample_index))
        })
        .unwrap_or(fallback_end_time_ns)
}

fn resolve_cluster_buffer_routing(
    manifest: &crate::manifests::DspManifest,
    stream_key: &str,
    decoded_channel_count: usize,
) -> Option<ManifestBufferRouting> {
    // Route any cluster-aware manifest into the shared sensor buffer when the
    // cluster exposes enough geometry for SRP. This lets a tetrahedral node
    // contribute its four channels alongside separate point nodes instead of
    // forcing Rust to localize each manifest in isolation.
    if decoded_channel_count == 0 {
        return None;
    }
    let cluster_id = manifest.cluster_id.as_deref()?;
    let cluster_sensor_positions = manifest.cluster_sensor_positions.as_ref()?;
    // An automatic GPS array must begin sharing audio as soon as its first
    // node arrives. Waiting for four sensors routes those early frames into
    // isolated buffers, so the fourth node sees only itself. Explicit legacy
    // clusters retain their historical four-sensor minimum.
    if cluster_sensor_positions.len() < 4 && cluster_id != AUTOMATIC_GPS_CLUSTER_ID {
        return None;
    }
    let node_id = stream_key_node_id(stream_key);
    let stream_id = stream_key_stream_id(stream_key).unwrap_or("audio_main");
    let buffer_channel_indices = (0..decoded_channel_count)
        .map(|channel_index| {
            let sensor_id = format!("{node_id}:ch{channel_index}");
            cluster_sensor_positions
                .iter()
                .position(|(candidate_sensor_id, _)| candidate_sensor_id == &sensor_id)
        })
        .collect::<Option<Vec<_>>>()?;
    Some(ManifestBufferRouting {
        buffer_key: format!("cluster::{cluster_id}::{stream_id}"),
        buffer_channel_count: cluster_sensor_positions.len(),
        buffer_channel_indices,
    })
}

fn reported_gps_position(node: &serde_json::Value) -> Option<ReportedGpsPosition> {
    let position = node.get("position_geo")?;
    let lat_deg = position.get("lat")?.as_f64()?;
    let lon_deg = position.get("lon")?.as_f64()?;
    let alt_m = position
        .get("alt_m")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    if !lat_deg.is_finite()
        || !lon_deg.is_finite()
        || !alt_m.is_finite()
        || !(-90.0..=90.0).contains(&lat_deg)
        || !(-180.0..=180.0).contains(&lon_deg)
    {
        return None;
    }
    Some(ReportedGpsPosition {
        lat_deg,
        lon_deg,
        alt_m,
    })
}

fn sensor_offsets_for_channel_count(
    node: &serde_json::Value,
    channel_count: usize,
) -> Vec<[f32; 3]> {
    let offsets = node
        .get("sensor_offsets_m")
        .and_then(serde_json::Value::as_array)
        .and_then(|values| {
            values
                .iter()
                .take(channel_count)
                .map(|value| {
                    let values = value.as_array()?;
                    if values.len() < 3 {
                        return None;
                    }
                    Some([
                        values[0].as_f64()? as f32,
                        values[1].as_f64()? as f32,
                        values[2].as_f64()? as f32,
                    ])
                })
                .collect::<Option<Vec<_>>>()
        });
    match offsets {
        Some(offsets) if offsets.len() == channel_count => offsets,
        _ if channel_count == SIRITH_MIC_POSITIONS_M.len() => SIRITH_MIC_POSITIONS_M.to_vec(),
        _ if channel_count == 1 => vec![[0.0, 0.0, 0.0]],
        _ => Vec::new(),
    }
}

fn gps_position_to_local_m(position: ReportedGpsPosition, origin: ReportedGpsPosition) -> [f32; 3] {
    let radians_per_degree = std::f64::consts::PI / 180.0;
    let east_m = (position.lon_deg - origin.lon_deg)
        * radians_per_degree
        * EARTH_RADIUS_M
        * origin.lat_deg.to_radians().cos();
    let north_m = (position.lat_deg - origin.lat_deg) * radians_per_degree * EARTH_RADIUS_M;
    [
        east_m as f32,
        north_m as f32,
        (position.alt_m - origin.alt_m) as f32,
    ]
}

fn stream_key_node_id(stream_key: &str) -> &str {
    stream_key.split("__").next().unwrap_or(stream_key)
}

fn stream_key_stream_id(stream_key: &str) -> Option<&str> {
    stream_key.split("__").nth(1)
}

fn should_use_receipt_time_alignment(
    node_timestamp_is_available: bool,
    skew_ns: u128,
    max_trusted_node_clock_skew_ns: u128,
) -> bool {
    // Preserve packet/node timing for TDOA whenever firmware timing exists.
    // Receipt-time alignment is a last-resort fallback only when node timing
    // is absent and skew is beyond the trusted horizon.
    //
    // Receipt-time-fallback policy — intentional asymmetry with the Python ingest
    // path. Python's `_buffer_timestamps_for_frame` (ingest.py:688) *overrides*
    // firmware timestamps when time_quality=FREE_RUNNING + gps_optional capability
    // + node clock skew > max_trusted, so debug audio stays playable on dev nodes
    // that lack GPS/NTP lock. The Rust sidecar deliberately does *not* override:
    // TDOA correctness depends on consistent packet-time alignment across nodes,
    // and silently rewriting timestamps masks clock issues that operators must
    // see to act on. Do not align the two implementations — this divergence is
    // covered by parity tests as an expected difference, not a bug.
    !node_timestamp_is_available && skew_ns > max_trusted_node_clock_skew_ns
}

fn sample_index_to_absolute_time_from_now_ns(
    sample_index: i64,
    sample_rate_hz: u32,
    now_ns: u128,
) -> i128 {
    let relative_duration_ns = sample_index_to_relative_time_ns(sample_index, sample_rate_hz);
    if relative_duration_ns <= 0 {
        return now_ns as i128;
    }
    now_ns.saturating_sub(relative_duration_ns as u128) as i128
}

fn manifest_is_older_than_buffer_horizon(
    manifest: &DspManifest,
    now_ns: u128,
    max_buffer_seconds: f64,
) -> bool {
    let horizon_ns = (max_buffer_seconds.max(0.0) * 1_000_000_000.0).round() as u128;
    if horizon_ns == 0 {
        return false;
    }
    let newest_source_ns = manifest
        .source_handles
        .iter()
        .filter_map(|handle| handle.tor_ns.or(handle.toa_ns))
        .map(u128::from)
        .max()
        .unwrap_or(manifest.created_ns);
    let newest_received_ns = manifest
        .source_handles
        .iter()
        .filter_map(|handle| handle.received_ns)
        .max();

    // Use explicit ingest receipt time for freshness when available.
    // Source timing fields come from node/packet clocks and can legitimately
    // be far in the past (replay/tests/GPS-anchored captures). Staleness
    // gating is meant to protect live buffering from old queue backlog, so
    // a manifest that just arrived should not be dropped solely because its
    // embedded capture epoch is old.
    let freshness_anchor_ns =
        newest_received_ns.unwrap_or_else(|| newest_source_ns.max(manifest.created_ns));
    now_ns.saturating_sub(freshness_anchor_ns) > horizon_ns
}

fn classifier_render_min_interval_ns(
    classifier_render_min_interval_seconds: f64,
    _pending_backlog_depth: usize,
) -> u128 {
    if classifier_render_min_interval_seconds <= 0.0 {
        return 0;
    }
    (classifier_render_min_interval_seconds * 1_000_000_000.0).round() as u128
}

fn sample_index_to_relative_time_ns(sample_index: i64, sample_rate_hz: u32) -> i128 {
    if sample_rate_hz == 0 {
        return 0;
    }
    i128::from(sample_index).saturating_mul(1_000_000_000) / i128::from(sample_rate_hz)
}

fn speed_of_sound_mps(temperature_c: f32, humidity_fraction: f32) -> f32 {
    let humidity_percent = humidity_fraction.clamp(0.0, 1.0) * 100.0;
    331.3 + (0.606 * temperature_c) + (0.0124 * humidity_percent)
}

// ---------------------------------------------------------------------------
// Biquad cascade — Butterworth design via direct bilinear transform with
// per-section Q. Direct Form II Transposed for numerical stability.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub(crate) struct BiquadCoefficients {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct BiquadState {
    z1: f32,
    z2: f32,
}

impl BiquadCoefficients {
    #[inline]
    fn process(&self, x: f32, state: &mut BiquadState) -> f32 {
        let y = self.b0 * x + state.z1;
        state.z1 = self.b1 * x - self.a1 * y + state.z2;
        state.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// Even-order Butterworth lowpass as a cascade of `order/2` biquads.
/// Odd orders are rounded up to the next even by callers — the biquad
/// cascade can't realize an odd-order pole structure on its own.
pub(crate) fn butter_lowpass_sos(
    cutoff_hz: f64,
    sample_rate_hz: u32,
    order: u8,
) -> Vec<BiquadCoefficients> {
    let order = effective_filter_order(order);
    let n_sections = (order / 2) as usize;
    let fs = f64::from(sample_rate_hz.max(1));
    let nyquist = fs * 0.5;
    let fc = cutoff_hz.clamp(1.0, nyquist * 0.999);
    let k = (std::f64::consts::PI * fc / fs).tan();
    let k2 = k * k;
    let n = f64::from(order);
    (0..n_sections)
        .map(|s| {
            let theta = std::f64::consts::PI * (2.0 * s as f64 + 1.0) / (2.0 * n);
            let q = 1.0 / (2.0 * theta.cos());
            let norm = 1.0 / (1.0 + k / q + k2);
            BiquadCoefficients {
                b0: (k2 * norm) as f32,
                b1: (2.0 * k2 * norm) as f32,
                b2: (k2 * norm) as f32,
                a1: (2.0 * (k2 - 1.0) * norm) as f32,
                a2: ((1.0 - k / q + k2) * norm) as f32,
            }
        })
        .collect()
}

/// Even-order Butterworth highpass as a cascade of `order/2` biquads.
pub(crate) fn butter_highpass_sos(
    cutoff_hz: f64,
    sample_rate_hz: u32,
    order: u8,
) -> Vec<BiquadCoefficients> {
    let order = effective_filter_order(order);
    let n_sections = (order / 2) as usize;
    let fs = f64::from(sample_rate_hz.max(1));
    let nyquist = fs * 0.5;
    let fc = cutoff_hz.clamp(1.0, nyquist * 0.999);
    let k = (std::f64::consts::PI * fc / fs).tan();
    let k2 = k * k;
    let n = f64::from(order);
    (0..n_sections)
        .map(|s| {
            let theta = std::f64::consts::PI * (2.0 * s as f64 + 1.0) / (2.0 * n);
            let q = 1.0 / (2.0 * theta.cos());
            let norm = 1.0 / (1.0 + k / q + k2);
            BiquadCoefficients {
                b0: norm as f32,
                b1: (-2.0 * norm) as f32,
                b2: norm as f32,
                a1: (2.0 * (k2 - 1.0) * norm) as f32,
                a2: ((1.0 - k / q + k2) * norm) as f32,
            }
        })
        .collect()
}

fn effective_filter_order(order: u8) -> u8 {
    // Coerce to an even order ≥ 2; the cascade has no realizable odd-order form.
    let bounded = order.max(2);
    if bounded.is_multiple_of(2) {
        bounded
    } else {
        bounded + 1
    }
}

// ---------------------------------------------------------------------------
// Compiled stages — one per `PreprocessStage` variant. Each stage owns the
// per-channel filter memory it needs so frame-by-frame application preserves
// continuity (no edge transient at every frame boundary).
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum CompiledStage {
    Gain {
        linear: f32,
    },
    ChannelGain {
        linear_by_channel: Vec<f32>,
    },
    /// IIR cascade — coefficients are shared across channels; biquad memory is per-channel.
    Cascade {
        coeffs: Vec<BiquadCoefficients>,
        // state[channel_idx][section_idx]
        state: Vec<Vec<BiquadState>>,
    },
    DcBlock {
        alpha: f32,
        prev_x: Vec<f32>,
        prev_y: Vec<f32>,
    },
    Passthrough,
}

impl CompiledStage {
    fn from_stage(stage: &PreprocessStage, sample_rate_hz: u32, channel_count: usize) -> Self {
        match stage {
            PreprocessStage::Gain { db } => {
                // Defense-in-depth: API validation rejects NaN/non-finite,
                // but a directly-constructed PreprocessStage::Gain bypasses
                // that. Treat a non-finite db as unity (passthrough) so
                // corrupt audio cannot escape the preprocessing stage.
                let linear = if db.is_finite() {
                    10f64.powf(*db / 20.0) as f32
                } else {
                    1.0
                };
                CompiledStage::Gain { linear }
            }
            PreprocessStage::ChannelGain { db_by_channel } => CompiledStage::ChannelGain {
                linear_by_channel: db_by_channel
                    .iter()
                    .map(|db| {
                        if db.is_finite() {
                            10f64.powf(*db / 20.0) as f32
                        } else {
                            1.0
                        }
                    })
                    .collect(),
            },
            PreprocessStage::Highpass { cutoff_hz, order } => {
                let coeffs = butter_highpass_sos(*cutoff_hz, sample_rate_hz, *order);
                let state = vec![vec![BiquadState::default(); coeffs.len()]; channel_count];
                CompiledStage::Cascade { coeffs, state }
            }
            PreprocessStage::Lowpass { cutoff_hz, order } => {
                let coeffs = butter_lowpass_sos(*cutoff_hz, sample_rate_hz, *order);
                let state = vec![vec![BiquadState::default(); coeffs.len()]; channel_count];
                CompiledStage::Cascade { coeffs, state }
            }
            PreprocessStage::Bandpass {
                low_hz,
                high_hz,
                order,
            } => {
                // Bandpass = highpass at low_hz cascaded with lowpass at high_hz.
                let mut coeffs = butter_highpass_sos(*low_hz, sample_rate_hz, *order);
                coeffs.extend(butter_lowpass_sos(*high_hz, sample_rate_hz, *order));
                let state = vec![vec![BiquadState::default(); coeffs.len()]; channel_count];
                CompiledStage::Cascade { coeffs, state }
            }
            PreprocessStage::DcBlock => {
                // Standard one-pole DC blocker: y[n] = x[n] - x[n-1] + α·y[n-1]
                // with α chosen for ~5 Hz cutoff regardless of sample rate.
                let fs = f64::from(sample_rate_hz.max(1));
                let alpha = (-2.0 * std::f64::consts::PI * 5.0 / fs).exp() as f32;
                CompiledStage::DcBlock {
                    alpha,
                    prev_x: vec![0.0; channel_count],
                    prev_y: vec![0.0; channel_count],
                }
            }
            PreprocessStage::Passthrough => CompiledStage::Passthrough,
        }
    }

    fn apply(&mut self, channels: &mut [Vec<f32>]) {
        match self {
            CompiledStage::Gain { linear } => {
                if (*linear - 1.0).abs() < f32::EPSILON {
                    return;
                }
                for ch in channels.iter_mut() {
                    for sample in ch.iter_mut() {
                        *sample *= *linear;
                    }
                }
            }
            CompiledStage::ChannelGain { linear_by_channel } => {
                for (channel_index, channel) in channels.iter_mut().enumerate() {
                    let linear = linear_by_channel.get(channel_index).copied().unwrap_or(1.0);
                    for sample in channel.iter_mut() {
                        *sample *= linear;
                    }
                }
            }
            CompiledStage::Cascade { coeffs, state } => {
                for (ch_idx, ch) in channels.iter_mut().enumerate() {
                    let Some(ch_state) = state.get_mut(ch_idx) else {
                        continue;
                    };
                    for sample in ch.iter_mut() {
                        let mut x = *sample;
                        for (coef, sec_state) in coeffs.iter().zip(ch_state.iter_mut()) {
                            x = coef.process(x, sec_state);
                        }
                        *sample = x;
                    }
                }
            }
            CompiledStage::DcBlock {
                alpha,
                prev_x,
                prev_y,
            } => {
                for (ch_idx, ch) in channels.iter_mut().enumerate() {
                    let Some(px_slot) = prev_x.get_mut(ch_idx) else {
                        continue;
                    };
                    let Some(py_slot) = prev_y.get_mut(ch_idx) else {
                        continue;
                    };
                    for sample in ch.iter_mut() {
                        let x = *sample;
                        let y = x - *px_slot + *alpha * *py_slot;
                        *px_slot = x;
                        *py_slot = y;
                        *sample = y;
                    }
                }
            }
            CompiledStage::Passthrough => {}
        }
    }
}

// ---------------------------------------------------------------------------
// NodeAudioState — per-stream filter memory + recompile-on-config-change.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub(crate) struct NodeAudioState {
    /// The stage list this compiled state was built for. We re-derive
    /// coefficients (and reset memory) whenever this signature changes —
    /// preserving state across frames but cleanly resetting on reconfigure.
    config_signature: Vec<PreprocessStage>,
    sample_rate_hz: u32,
    channel_count: usize,
    stages: Vec<CompiledStage>,
}

impl NodeAudioState {
    pub(crate) fn apply(
        &mut self,
        channels: &mut [Vec<f32>],
        config: &NodeAudioConfig,
        sample_rate_hz: u32,
    ) {
        let stages = config.effective_stages();
        if stages.is_empty() {
            // Passthrough — reset signature so a later config change recompiles cleanly.
            if !self.config_signature.is_empty() {
                self.config_signature.clear();
                self.stages.clear();
            }
            return;
        }
        let channel_count = channels.len();
        if stages != self.config_signature
            || sample_rate_hz != self.sample_rate_hz
            || channel_count != self.channel_count
        {
            self.config_signature = stages;
            self.sample_rate_hz = sample_rate_hz;
            self.channel_count = channel_count;
            self.stages = self
                .config_signature
                .iter()
                .map(|stage| CompiledStage::from_stage(stage, sample_rate_hz, channel_count))
                .collect();
        }
        for stage in self.stages.iter_mut() {
            stage.apply(channels);
        }
    }
}

pub(crate) fn system_now_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .expect("system clock is before UNIX_EPOCH — RTC fault or misconfiguration")
}

#[cfg(test)]
#[path = "dsp_worker_tests.rs"]
mod dsp_worker_tests;
