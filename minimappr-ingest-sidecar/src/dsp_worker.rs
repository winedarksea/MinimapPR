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
    audio_payload::{decode_audio_payload, DecodedAudioPayload},
    classifier_helper::ManifestClassificationAnnotator,
    derived_cache::DerivedCache,
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

/// Centroid-relative Sirith tetrahedral mic positions [MK1, MK2, MK3, MK4].
pub(crate) const SIRITH_MIC_POSITIONS_M: [[f32; 3]; 4] = [
    [0.0, 0.050, 0.0],
    [0.0433, 0.025, 0.0],
    [0.0, 0.0, 0.0],
    [0.02165, 0.025, 0.04082],
];

#[derive(Clone, Debug)]
pub struct DspWorkerConfig {
    pub poll_interval_ms: u64,
    pub pending_manifest_batch_size: usize,
    pub window_seconds: f64,
    pub classification_window_seconds: f64,
    pub classifier_render_min_interval_seconds: f64,
    pub max_buffer_seconds: f64,
    pub min_coverage_ratio: f64,
    pub localization_band_hz: [f32; 2],
    pub localization_srp_grid_resolution_m: f32,
    pub localization_search_padding_m: f32,
    pub spatial_blend_band_hz: [f32; 2],
    pub pre_blend_highpass_hz: f32,
    pub min_localization_confidence: f32,
    pub skip_stale_manifests_for_live_buffer: bool,
    pub consumed_manifest_retention_max_files: usize,
    pub consumed_manifest_prune_interval: u64,
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
            localization_srp_grid_resolution_m: 0.5,
            localization_search_padding_m: 2.0,
            spatial_blend_band_hz: [1000.0, 3400.0],
            pre_blend_highpass_hz: 100.0,
            min_localization_confidence: 0.20,
            skip_stale_manifests_for_live_buffer: false,
            consumed_manifest_retention_max_files: 20_000,
            consumed_manifest_prune_interval: 256,
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

/// Per-node audio DSP override (gain and filters applied before buffer insertion).
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct NodeAudioConfig {
    /// Gain in dB to apply to all channels for this node (0.0 = no change).
    pub gain_db: Option<f64>,
    /// 1st-order IIR highpass cutoff in Hz (0 or None = disabled).
    pub hp_hz: Option<f64>,
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
}

struct OwnedManifestAudio {
    manifest: DspManifest,
    stream_key: String,
    decoded: DecodedAudioPayload,
    source_ids: Vec<String>,
    sr: u32,
    now_ns: u128,
    channel_count: usize,
    classification_window_sec: f64,
    mic_positions_m: Vec<[f32; 3]>,
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

struct RawAudioFramePublishRequest<'a> {
    publisher: &'a Option<DspEventPublisher>,
    source_manifest: &'a DspManifest,
    decoded: &'a DecodedAudioPayload,
    stream_key: &'a str,
    sample_rate_hz: u32,
    start_time_ns: i128,
    end_time_ns: i128,
    created_ns: u128,
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
    last_classifier_render_ns_by_stream: HashMap<String, u128>,
    last_localization_ns_by_stream: HashMap<String, u128>,
    last_trigger_ns_by_stream: HashMap<String, u128>,
    /// Time-only cadence gate used for node liveness heartbeats.
    /// Unlike `last_localization_ns_by_stream`, this is never blocked by the RMS
    /// energy gate or trigger cooldown — it fires unconditionally every
    /// `localization_cadence_ms` so nodes appear alive even during quiet periods.
    last_heartbeat_ns_by_stream: HashMap<String, u128>,
    consumed_manifests_since_prune: Arc<AtomicU64>,
    deferred_source_manifest_ids: Vec<String>,
    env_cache: Option<EnvironmentCache>,
    /// Shared publisher for DSP result manifests streamed to Python via SSE.
    dsp_event_publisher: Option<DspEventPublisher>,
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
            last_classifier_render_ns_by_stream: HashMap::new(),
            last_localization_ns_by_stream: HashMap::new(),
            last_trigger_ns_by_stream: HashMap::new(),
            last_heartbeat_ns_by_stream: HashMap::new(),
            consumed_manifests_since_prune: Arc::new(AtomicU64::new(0)),
            deferred_source_manifest_ids: Vec::new(),
            env_cache: None,
            dsp_event_publisher: None,
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

    /// Signals the worker to stop after the ingest channel has been fully drained.
    pub fn with_shutdown_signal(mut self, shutdown_requested: Arc<AtomicBool>) -> Self {
        self.shutdown_requested = Some(shutdown_requested);
        self
    }

    pub async fn run_loop(mut self) {
        info!("DSP worker started");
        let interval = time::Duration::from_millis(self.config.poll_interval_ms);
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
            .is_some_and(|flag| !flag.load(Ordering::Acquire));
        if !shutdown_requested || processed_manifest_count > 0 {
            return false;
        }
        self.raw_manifest_rx
            .as_ref()
            .is_none_or(|rx| rx.is_closed() && rx.is_empty())
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
        let owned = self.prepare_owned_manifest_audio(manifest).await?;

        self.publish_raw_audio_frame_for_owned_manifest(&owned).await;

        if self
            .should_skip_stale_manifest_for_live_buffer(&owned)
            .await
        {
            return None;
        }

        if owned.channel_count > 4 {
            return self.dispatch_large_array_manifest(owned, pending_backlog_depth);
        }

        let buffered = self.buffer_owned_manifest_audio(&owned).await?;
        self.dispatch_buffered_manifest(owned, buffered, pending_backlog_depth)
    }

    async fn prepare_owned_manifest_audio(
        &mut self,
        manifest: DspManifest,
    ) -> Option<OwnedManifestAudio> {
        let now_ns = system_now_ns();
        let Some(first_handle) = manifest.source_handles.first() else {
            self.mark_source_manifest_consumed_if_persisted(&manifest)
                .await;
            return None;
        };

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

        let mut decoded = match decode_audio_payload(&raw_payload) {
            Ok(decoded) => decoded,
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
        if decoded.channels.is_empty() {
            self.mark_source_manifest_consumed_if_persisted(&manifest)
                .await;
            return None;
        }

        {
            let node_id_key = stream_key.split("__").next().unwrap_or(&stream_key);
            let st = self.state.read().await;
            if let Some(cfg) = st.node_audio_overrides.get(node_id_key).cloned() {
                drop(st);
                apply_node_audio_to_channels(
                    &mut decoded.channels,
                    &cfg,
                    decoded.sample_rate_hz,
                );
            }
        }

        let source_ids = manifest
            .source_handles
            .iter()
            .map(|handle| handle.segment_id.clone())
            .collect::<Vec<_>>();
        let sr = decoded.sample_rate_hz.max(1);
        let (env_temp_c, env_humidity_fraction) = if decoded.temperature_c.is_none()
            && decoded.humidity_fraction.is_none()
        {
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
        let start_time_ns = resolve_buffer_start_time_ns(&decoded, first_handle, sr, now_ns);
        let node_timestamp_is_available = decoded.start_time_ns.is_some_and(|start_ns| start_ns > 0)
            || first_handle.toa_ns.is_some();
        let skew_ns = (start_time_ns - now_ns as i128).unsigned_abs();
        let max_skew_ns =
            (self.config.max_trusted_node_clock_skew_seconds * 1_000_000_000.0).round() as u128;
        let buffer_uses_receipt_time =
            should_use_receipt_time_alignment(node_timestamp_is_available, skew_ns, max_skew_ns);
        let (buffer_start_time_ns, buffer_end_time_ns) = if buffer_uses_receipt_time {
            let anchor_ns = first_handle.tor_ns.unwrap_or(now_ns as u64) as i128;
            let start_ns = anchor_ns.saturating_sub(render_duration_ns).max(1);
            (start_ns, anchor_ns)
        } else {
            (start_time_ns, start_time_ns + render_duration_ns)
        };

        let channel_count = decoded.channels.len();
        let classification_window_sec = self
            .config
            .classification_window_seconds
            .max(MIN_BIRDNET_CLASSIFICATION_WINDOW_SECONDS)
            .max(self.config.window_seconds);
        let mic_positions_m = mic_positions_from_manifest(&manifest);
        let start_sample_index = decoded.start_sample_index;
        let end_sample_index = decoded.end_sample_index;

        Some(OwnedManifestAudio {
            manifest,
            stream_key,
            decoded,
            source_ids,
            sr,
            now_ns,
            channel_count,
            classification_window_sec,
            mic_positions_m,
            effective_sound_speed_mps,
            buffer_start_time_ns,
            buffer_end_time_ns,
            start_sample_index,
            end_sample_index,
            node_timestamp_is_available,
            buffer_uses_receipt_time,
            arrived_via_channel,
        })
    }

    async fn publish_raw_audio_frame_for_owned_manifest(&self, owned: &OwnedManifestAudio) {
        publish_raw_audio_frame_event(RawAudioFramePublishRequest {
            publisher: &self.dsp_event_publisher,
            source_manifest: &owned.manifest,
            decoded: &owned.decoded,
            stream_key: &owned.stream_key,
            sample_rate_hz: owned.sr,
            start_time_ns: owned.buffer_start_time_ns,
            end_time_ns: owned.buffer_end_time_ns,
            created_ns: owned.now_ns,
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
        let buffers = self.buffers.entry(owned.stream_key.clone()).or_insert_with(|| {
            (0..owned.channel_count)
                .map(|_| SensorStreamBuffer::new(owned.sr, self.config.max_buffer_seconds))
                .collect()
        });

        let existing_sample_timeline_start_time_ns = if !owned.node_timestamp_is_available
            && !owned.buffer_uses_receipt_time
        {
            owned
                .start_sample_index
                .and_then(|sample_index| buffers[0].time_for_sample_index(sample_index))
        } else {
            None
        };
        let (buffer_start_time_ns, buffer_end_time_ns) =
            if let Some(existing_start_time_ns) = existing_sample_timeline_start_time_ns {
                (
                    existing_start_time_ns,
                    existing_start_time_ns
                        + (owned.buffer_end_time_ns - owned.buffer_start_time_ns),
                )
            } else {
                (owned.buffer_start_time_ns, owned.buffer_end_time_ns)
            };

        for (channel_index, buffer) in buffers.iter_mut().enumerate() {
            let Some(channel_samples) = owned.decoded.channels.get(channel_index) else {
                break;
            };
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
        }

        let end_ns = resolve_buffer_end_time_ns(
            buffers,
            owned.end_sample_index,
            buffer_end_time_ns,
            owned.sr,
        );
        let window_duration_ns = (self.config.window_seconds * 1_000_000_000.0).round() as i128;
        let center_time_ns = end_ns
            .saturating_sub(window_duration_ns / 2)
            .max(buffer_start_time_ns);
        let channel_states =
            localization_channel_states_centered(buffers, center_time_ns, self.config.window_seconds);
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
    ) -> Option<ComputePayload> {
        if !self.should_publish_classifier_render(
            &owned.stream_key,
            owned.now_ns,
            pending_backlog_depth,
        ) {
            self.defer_source_manifest_consumption(&owned.manifest);
            return None;
        }

        let omni_channels = owned.decoded.channels.clone();
        Some(ComputePayload {
            manifest: owned.manifest,
            stream_key: owned.stream_key,
            channel_states: (0..owned.channel_count)
                .map(|_| LocalizationChannelState {
                    coverage: None,
                    window: Vec::new(),
                })
                .collect(),
            active_channels: Vec::new(),
            classification_windows: (0..owned.channel_count).map(|_| Vec::new()).collect(),
            listenable_classification_windows: (0..owned.channel_count)
                .map(|_| Vec::new())
                .collect(),
            classification_coverage: (0..owned.channel_count).map(|_| None).collect(),
            classifier_render_start_ns: Some(owned.buffer_start_time_ns.max(0) as u128),
            classifier_render_end_ns: Some(owned.buffer_end_time_ns.max(0) as u128),
            mic_positions_m: owned.mic_positions_m,
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
            omni_fallback_reason: Some("single_sensor_or_non_tetrahedral_array".to_string()),
            omni_channels_override: Some(omni_channels.clone()),
            listenable_omni_channels_override: Some(omni_channels),
            manifest_store: self.manifest_store.clone(),
            derived_cache: self.derived_cache.clone(),
            state: self.state.clone(),
            classification_tx: self.classification_tx.clone(),
            dsp_event_publisher: self.dsp_event_publisher.clone(),
            config: self.config.clone(),
            consumed_since_prune: self.consumed_manifests_since_prune.clone(),
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
        let run_srp = channel_count == 4 && (on_localization_cadence || run_classifier_render);
        let on_heartbeat_cadence =
            on_localization_cadence || self.should_emit_heartbeat(stream_key, audio_end_ns);
        BufferedManifestTimingGates {
            on_heartbeat_cadence,
            run_classifier_render,
            run_srp,
        }
    }

    fn dispatch_buffered_manifest(
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
                &owned.stream_key,
                buffered.audio_end_ns,
                pending_backlog_depth,
            );
            if run_classifier_render {
                debug!(
                    manifest_id = %owned.manifest.manifest_id,
                    "DSP worker: no localization coverage window after buffering; publishing omni fallback render"
                );
                let Some(buffers) = self.buffers.get(&owned.stream_key) else {
                    error!(%owned.stream_key, "stream buffers missing after append in omni fallback path — skipping");
                    return None;
                };
                let fallback_render_windows =
                    channel_windows_ending_at(buffers, buffered.end_ns, owned.classification_window_sec);
                let listenable_fallback_render_windows =
                    channel_windows_ending_at_with_gap_concealment(
                        buffers,
                        buffered.end_ns,
                        owned.classification_window_sec,
                    );
                let fallback_render_coverage =
                    channel_coverage_ending_at(buffers, buffered.end_ns, owned.classification_window_sec);
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
                    stream_key: owned.stream_key,
                    channel_states: buffered.channel_states,
                    active_channels: eligible_coverage_channels(
                        &fallback_render_coverage,
                        self.config.min_coverage_ratio,
                    ),
                    classification_windows: (0..owned.channel_count).map(|_| Vec::new()).collect(),
                    listenable_classification_windows: (0..owned.channel_count)
                        .map(|_| Vec::new())
                        .collect(),
                    classification_coverage: fallback_render_coverage,
                    classifier_render_start_ns,
                    classifier_render_end_ns,
                    mic_positions_m: owned.mic_positions_m,
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
                });
            }
        }

        let active_channels =
            eligible_localization_channels(&buffered.channel_states, self.config.min_coverage_ratio);
        let windows: Vec<Vec<f32>> = buffered
            .channel_states
            .iter()
            .map(|state| state.window.clone())
            .collect();
        let timing_gates = self.resolve_buffered_manifest_timing_gates(
            &owned.stream_key,
            owned.channel_count,
            buffered.audio_end_ns,
            pending_backlog_depth,
            &windows,
        );

        if !timing_gates.on_heartbeat_cadence && !timing_gates.run_classifier_render {
            self.defer_source_manifest_consumption(&owned.manifest);
            return None;
        }

        let (classification_windows, listenable_classification_windows, classification_coverage) =
            if timing_gates.run_classifier_render {
                let Some(buffers) = self.buffers.get(&owned.stream_key) else {
                    error!(%owned.stream_key, "stream buffers missing after append in classifier render path — skipping");
                    return None;
                };
                (
                    channel_windows_ending_at(buffers, buffered.end_ns, owned.classification_window_sec),
                    channel_windows_ending_at_with_gap_concealment(
                        buffers,
                        buffered.end_ns,
                        owned.classification_window_sec,
                    ),
                    channel_coverage_ending_at(buffers, buffered.end_ns, owned.classification_window_sec),
                )
            } else {
                (
                    buffered.channel_states.iter().map(|_| Vec::new()).collect(),
                    buffered.channel_states.iter().map(|_| Vec::new()).collect(),
                    buffered.channel_states.iter().map(|_| None).collect(),
                )
            };
        let (classifier_render_start_ns, classifier_render_end_ns) = if timing_gates.run_classifier_render {
            classifier_render_bounds_from_windows(buffered.end_ns, &classification_windows, owned.sr)
        } else {
            (None, None)
        };

        Some(ComputePayload {
            manifest: owned.manifest,
            stream_key: owned.stream_key,
            channel_states: buffered.channel_states,
            active_channels,
            classification_windows,
            listenable_classification_windows,
            classification_coverage,
            classifier_render_start_ns,
            classifier_render_end_ns,
            mic_positions_m: owned.mic_positions_m,
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
            omni_fallback_reason: (owned.channel_count < 4).then(|| "single_point_node".to_string()),
            omni_channels_override: None,
            listenable_omni_channels_override: None,
            manifest_store: self.manifest_store.clone(),
            derived_cache: self.derived_cache.clone(),
            state: self.state.clone(),
            classification_tx: self.classification_tx.clone(),
            dsp_event_publisher: self.dsp_event_publisher.clone(),
            config: self.config.clone(),
            consumed_since_prune: self.consumed_manifests_since_prune.clone(),
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
        if previous / prune_interval < current / prune_interval {
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
    let coverage_stats_json = serde_json::to_value(coverage_stats(
        &frame_coverage,
        sample_rate_hz.max(1),
    ))
    .ok();
    let raw_manifest = DspManifest {
        manifest_id: format!("raw-audio-{}", source_manifest.manifest_id),
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
pub(crate) fn mic_positions_from_manifest(manifest: &crate::manifests::DspManifest) -> Vec<[f32; 3]> {
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
pub(crate) fn mic_positions_from_node_context(node_context: &Option<serde_json::Value>) -> Vec<[f32; 3]> {
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
                .first()
                .and_then(|buffer| buffer.time_for_sample_index(sample_index))
        })
        .unwrap_or(fallback_end_time_ns)
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
    let freshness_anchor_ns = newest_received_ns
        .unwrap_or_else(|| newest_source_ns.max(manifest.created_ns));
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

/// Apply per-node audio overrides (gain_db, hp_hz) to decoded channel buffers
/// in-place, before they are written into the SensorStreamBuffer.
///
/// Gain is applied as a linear multiplier derived from `gain_db`.
/// The highpass is a first-order IIR discrete approximation:
///   y[n] = α · (y[n-1] + x[n] − x[n-1])   where α = fs / (fs + 2π·fc)
pub(crate) fn apply_node_audio_to_channels(
    channels: &mut [Vec<f32>],
    config: &NodeAudioConfig,
    sample_rate_hz: u32,
) {
    let gain = config
        .gain_db
        .filter(|&db| db != 0.0)
        .map(|db| 10f64.powf(db / 20.0) as f32);

    let hp_alpha = config
        .hp_hz
        .filter(|&hz| hz > 0.0)
        .map(|hz| {
            let fs = sample_rate_hz as f64;
            (fs / (fs + 2.0 * std::f64::consts::PI * hz)) as f32
        });

    for ch in channels.iter_mut() {
        if let Some(g) = gain {
            for s in ch.iter_mut() {
                *s *= g;
            }
        }
        if let Some(alpha) = hp_alpha {
            if ch.len() >= 2 {
                let mut prev_x = ch[0];
                let mut prev_y = ch[0];
                for sample in ch.iter_mut().skip(1) {
                    let x = *sample;
                    let y = alpha * (prev_y + x - prev_x);
                    prev_x = x;
                    prev_y = y;
                    *sample = y;
                }
            }
        }
    }
}

pub(crate) fn system_now_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

#[cfg(test)]
#[path = "dsp_worker_tests.rs"]
mod dsp_worker_tests;
