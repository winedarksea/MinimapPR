use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use tokio::{sync::mpsc, sync::RwLock, time};
use tracing::{debug, info, warn};
use crate::{
    actors::{
        classification::ClassificationWorker,
        environment::EnvironmentCache,
    },
    audio_payload::{decode_audio_payload, DecodedAudioPayload},
    classifier_helper::ManifestClassificationAnnotator,
    derived_cache::DerivedCache,
    dsp::{AudioCoverageStats, SensorStreamBuffer},
    gcc_phat::TdoaResult,
    journal_reader::{read_payload_with_mmap, JournalPayloadHandle},
    manifests::{DspManifest, ManifestStore},
};

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
    pub birdnet_hybrid_render_enabled: bool,
    pub skip_stale_manifests_for_live_buffer: bool,
    pub consumed_manifest_retention_max_files: usize,
    pub consumed_manifest_prune_interval: u64,
    pub default_temperature_c: f32,
    pub default_humidity_fraction: f32,
    pub sound_speed_mps: f32,
    pub classifier_command_json: Option<String>,
    pub localization_cadence_ms: u64,
    pub localization_rms_gate: f32,
}

impl Default for DspWorkerConfig {
    fn default() -> Self {
        let default_temperature_c = 20.0;
        let default_humidity_fraction = 0.5;
        Self {
            poll_interval_ms: 20,
            pending_manifest_batch_size: 128,
            window_seconds: 512.0 / 16_000.0,
            classification_window_seconds: 512.0 / 16_000.0,
            classifier_render_min_interval_seconds: 0.0,
            max_buffer_seconds: 4.0 * 512.0 / 16_000.0,
            min_coverage_ratio: 0.85,
            localization_band_hz: [300.0, 3500.0],
            localization_srp_grid_resolution_m: 0.5,
            localization_search_padding_m: 2.0,
            spatial_blend_band_hz: [1000.0, 3400.0],
            pre_blend_highpass_hz: 100.0,
            min_localization_confidence: 0.20,
            birdnet_hybrid_render_enabled: false,
            skip_stale_manifests_for_live_buffer: true,
            consumed_manifest_retention_max_files: 20_000,
            consumed_manifest_prune_interval: 256,
            default_temperature_c,
            default_humidity_fraction,
            sound_speed_mps: 343.2,
            classifier_command_json: None,
            localization_cadence_ms: 250,
            localization_rms_gate: 0.0,
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

/// Shared state exposed via the /api/v1/dsp/* endpoints.
#[derive(Debug, Default)]
pub struct DspWorkerState {
    pub worker_running: bool,
    pub last_heartbeat_ns: Option<u128>,
    pub last_processed_ns: Option<u128>,
    pub total_tdoa_results: u64,
    pub total_localization_results: u64,
    pub total_classifier_renders: u64,
    pub total_failures: u64,
    pub total_stale_manifest_skips: u64,
    pub total_classification_drops: u64,
    pub pending_count: usize,
    pub recent_results: Vec<DspManifest>,
}

pub type SharedDspState = Arc<RwLock<DspWorkerState>>;

#[derive(Clone, Debug)]
pub(crate) struct LocalizationChannelState {
    pub(crate) coverage: Option<AudioCoverageStats>,
    pub(crate) window: Vec<f32>,
}

/// Carries a pre-built classifier_render manifest + the PCM path to the
/// ClassificationWorker, which calls BirdNET and publishes the annotated manifest.
pub struct ClassificationRequest {
    pub pcm_path: std::path::PathBuf,
    pub sample_rate_hz: u32,
    pub pending_manifest: DspManifest,
}

/// All owned data needed to run the compute phase (SRP-PHAT + render + publish)
/// for a single manifest after the ingest (buffer-append) phase completes.
pub(crate) struct ComputePayload {
    pub(crate) manifest: DspManifest,
    pub(crate) stream_key: String,
    pub(crate) channel_states: [LocalizationChannelState; 4],
    pub(crate) active_channels: Vec<usize>,
    pub(crate) classification_windows: [Vec<f32>; 4],
    pub(crate) classification_coverage: [Option<AudioCoverageStats>; 4],
    pub(crate) source_ids: Vec<String>,
    pub(crate) sr: u32,
    pub(crate) now_ns: u128,
    pub(crate) effective_sound_speed_mps: f32,
    pub(crate) run_srp: bool,
    pub(crate) run_classifier_render: bool,
    /// When set, overrides classification_windows for the omni render (single-sensor
    /// or no-coverage fallback paths that dispatch to Rayon instead of blocking inline).
    pub(crate) omni_channels_override: Option<Vec<Vec<f32>>>,
    /// When set, used as the render fallback_reason instead of deriving it from the
    /// localization algorithm string. Preserves semantic accuracy for fallback paths.
    pub(crate) omni_fallback_reason: Option<String>,
    /// When true, run_io skips publishing a localization_result manifest. Used for
    /// omni-only fallback paths that never had a real localization attempt.
    pub(crate) skip_localization_result: bool,
    pub(crate) manifest_store: ManifestStore,
    pub(crate) derived_cache: DerivedCache,
    pub(crate) state: SharedDspState,
    pub(crate) classification_tx: Option<flume::Sender<ClassificationRequest>>,
    pub(crate) config: DspWorkerConfig,
    pub(crate) consumed_since_prune: Arc<AtomicU64>,
}

pub struct DspWorker {
    manifest_store: ManifestStore,
    derived_cache: DerivedCache,
    config: DspWorkerConfig,
    state: SharedDspState,
    classifier_annotator: Option<ManifestClassificationAnnotator>,
    classification_tx: Option<flume::Sender<ClassificationRequest>>,
    raw_manifest_rx: Option<mpsc::Receiver<DspManifest>>,
    buffers: HashMap<String, [SensorStreamBuffer; 4]>,
    last_classifier_render_ns_by_stream: HashMap<String, u128>,
    last_localization_ns_by_stream: HashMap<String, u128>,
    consumed_manifests_since_prune: Arc<AtomicU64>,
    env_cache: Option<EnvironmentCache>,
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
            buffers: HashMap::new(),
            last_classifier_render_ns_by_stream: HashMap::new(),
            last_localization_ns_by_stream: HashMap::new(),
            consumed_manifests_since_prune: Arc::new(AtomicU64::new(0)),
            env_cache: None,
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
            manifest_store: self.manifest_store.clone(),
            rx,
        };
        (self, Some(worker))
    }

    /// Injects an in-process receiver for raw_journal_append manifests from the
    /// co-located ingest backend, bypassing the 20ms filesystem poll cycle.
    pub fn with_raw_manifest_receiver(mut self, rx: mpsc::Receiver<DspManifest>) -> Self {
        self.raw_manifest_rx = Some(rx);
        self
    }

    /// Injects a shared EnvironmentCache for sound-speed interpolation (Phase 6).
    pub fn with_env_cache(mut self, cache: EnvironmentCache) -> Self {
        self.env_cache = Some(cache);
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
            self.process_pending().await;
            time::sleep(interval).await;
        }
    }

    async fn process_pending(&mut self) {
        let batch_limit = self.config.pending_manifest_batch_size.max(1);

        // Fast path: drain the in-process channel before touching the filesystem.
        let channel_manifests: Vec<DspManifest> = if let Some(rx) = &mut self.raw_manifest_rx {
            let mut v = Vec::new();
            while let Ok(m) = rx.try_recv() {
                v.push(m);
                if v.len() >= batch_limit {
                    break;
                }
            }
            v
        } else {
            Vec::new()
        };

        let pending = if !channel_manifests.is_empty() {
            channel_manifests
        } else {
            match self
                .manifest_store
                .query_pending_limited("raw_journal_append", batch_limit)
                .await
            {
                Ok(m) => m,
                Err(err) => {
                    warn!(error = %err, "DSP worker failed to query pending manifests");
                    return;
                }
            }
        };

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
    }

    async fn ingest_one(
        &mut self,
        manifest: DspManifest,
        pending_backlog_depth: usize,
    ) -> Option<ComputePayload> {
        let now_ns = system_now_ns();

        let Some(first_handle) = manifest.source_handles.first() else {
            let _ = self
                .manifest_store
                .mark_consumed(&manifest.manifest_id)
                .await;
            return None;
        };

        let stream_key = first_handle.stream_key.clone();

        // Fast path: raw audio bytes were delivered through the in-process channel
        // alongside the manifest metadata — no disk read required.
        // Fallback: read from the journal segment via mmap. read_payload_with_mmap is
        // a blocking synchronous call, so run it in spawn_blocking to avoid stalling
        // the Tokio executor thread while the OS resolves the mmap page faults.
        let raw_payload: Vec<u8> = if let Some(bytes) = manifest.raw_payload.clone() {
            bytes
        } else {
            let handle = first_handle.clone();
            match tokio::task::spawn_blocking(move || read_payload_with_mmap(&handle)).await {
                Ok(Ok(bytes)) => bytes,
                Ok(Err(err)) => {
                    self.note_failure().await;
                    warn!(
                        manifest_id = %manifest.manifest_id,
                        error = %err,
                        "DSP worker failed to read journal payload; consuming unreadable source manifest"
                    );
                    self.consume_source_manifest(&manifest).await;
                    return None;
                }
                Err(join_err) => {
                    self.note_failure().await;
                    warn!(
                        manifest_id = %manifest.manifest_id,
                        error = %join_err,
                        "DSP worker spawn_blocking panicked reading journal payload"
                    );
                    self.consume_source_manifest(&manifest).await;
                    return None;
                }
            }
        };
        let decoded = match decode_audio_payload(&raw_payload) {
            Ok(decoded) => decoded,
            Err(err) => {
                self.note_failure().await;
                warn!(
                    manifest_id = %manifest.manifest_id,
                    error = %err,
                    "DSP worker failed to decode ingest audio; consuming manifest"
                );
                let _ = self
                    .manifest_store
                    .mark_consumed(&manifest.manifest_id)
                    .await;
                return None;
            }
        };
        if decoded.channels.is_empty() {
            let _ = self
                .manifest_store
                .mark_consumed(&manifest.manifest_id)
                .await;
            return None;
        }

        let source_ids = manifest
            .source_handles
            .iter()
            .map(|h| h.segment_id.clone())
            .collect::<Vec<_>>();
        let sr = decoded.sample_rate_hz.max(1);

        // Phase 6: sound-speed fallback chain:
        // (1) embedded MMB1 flag bytes > (2) EnvironmentCache interpolation > (3) config defaults.
        let (env_temp_c, env_humidity_fraction) = if decoded.temperature_c.is_none()
            && decoded.humidity_fraction.is_none()
        {
            if let Some(cache) = &self.env_cache {
                let query_ns = first_handle
                    .toa_ns
                    .unwrap_or_else(|| now_ns as u64);
                let node_id = stream_key.split("__").next().unwrap_or(&stream_key);
                cache
                    .interpolate(node_id, query_ns)
                    .await
                    .map(|(t, h)| (Some(t), Some(h)))
                    .unwrap_or((None, None))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        let effective_sound_speed_mps = resolve_effective_sound_speed_mps(
            &self.config,
            decoded.temperature_c.or(env_temp_c),
            decoded.humidity_fraction.or(env_humidity_fraction),
        );

        let frames_all = decoded.channels.iter().map(Vec::len).min().unwrap_or(0);
        let render_duration_ns =
            (frames_all as i128).saturating_mul(1_000_000_000) / i128::from(sr.max(1));
        let start_time_ns = resolve_buffer_start_time_ns(&decoded, first_handle, sr, now_ns);
        let audio_end_ns = start_time_ns.saturating_add(render_duration_ns).max(0) as u128;

        let source_manifest_is_stale = manifest_is_older_than_buffer_horizon(
            &manifest,
            now_ns,
            self.config.max_buffer_seconds,
        );
        if source_manifest_is_stale && self.config.skip_stale_manifests_for_live_buffer {
            debug!(
                manifest_id = %manifest.manifest_id,
                "DSP worker skipped stale source manifest to protect live buffer continuity"
            );
            self.consume_source_manifest(&manifest).await;
            let mut st = self.state.write().await;
            st.total_stale_manifest_skips += 1;
            st.last_processed_ns = Some(now_ns);
            return None;
        }

        let window_sec = self.config.window_seconds;
        let classification_window_sec = self.config.classification_window_seconds.max(window_sec);

        if decoded.channels.len() < 4 {
            if !self.config.birdnet_hybrid_render_enabled {
                self.consume_source_manifest(&manifest).await;
                return None;
            }
            if !self.should_publish_classifier_render(
                &stream_key,
                audio_end_ns,
                pending_backlog_depth,
            ) {
                self.consume_source_manifest(&manifest).await;
                return None;
            }
            // Dispatch to Rayon so disk/DB I/O doesn't block the ingest loop.
            let channel_states: [LocalizationChannelState; 4] =
                core::array::from_fn(|_| LocalizationChannelState { coverage: None, window: Vec::new() });
            let classification_windows: [Vec<f32>; 4] = core::array::from_fn(|_| Vec::new());
            return Some(ComputePayload {
                manifest,
                stream_key,
                channel_states,
                active_channels: Vec::new(),
                classification_windows,
                classification_coverage: [None, None, None, None],
                omni_channels_override: Some(decoded.channels.clone()),
                omni_fallback_reason: Some("single_sensor_or_non_array_node".to_string()),
                skip_localization_result: true,
                source_ids,
                sr,
                now_ns,
                effective_sound_speed_mps,
                run_srp: false,
                run_classifier_render: true,
                manifest_store: self.manifest_store.clone(),
                derived_cache: self.derived_cache.clone(),
                state: self.state.clone(),
                classification_tx: self.classification_tx.clone(),
                config: self.config.clone(),
                consumed_since_prune: self.consumed_manifests_since_prune.clone(),
            });
        }

        let start_sample_index = decoded.start_sample_index;
        let end_sample_index = decoded.end_sample_index;

        let buffers = self.buffers.entry(stream_key.clone()).or_insert_with(|| {
            core::array::from_fn(|_| SensorStreamBuffer::new(sr, self.config.max_buffer_seconds))
        });

        for (ch, buf) in buffers.iter_mut().enumerate() {
            if let Err(err) = buf.append(
                start_time_ns,
                &decoded.channels[ch],
                start_sample_index,
                end_sample_index,
            ) {
                self.note_failure().await;
                warn!(
                    manifest_id = %manifest.manifest_id,
                    channel = ch,
                    error = %err,
                    "DSP worker failed to append decoded audio; consuming malformed source manifest"
                );
                self.consume_source_manifest(&manifest).await;
                return None;
            }
        }

        let end_ns = start_time_ns + render_duration_ns;
        let channel_states = localization_channel_states(buffers, end_ns, window_sec);

        if channel_states.iter().all(|state| state.coverage.is_none()) {
            debug!(
                manifest_id = %manifest.manifest_id,
                "DSP worker: no localization coverage window after buffering; publishing omni fallback render"
            );
            if !self.config.birdnet_hybrid_render_enabled {
                self.consume_source_manifest(&manifest).await;
                return None;
            }
            let fallback_render_windows =
                latest_channel_windows(buffers, classification_window_sec);
            let fallback_render_coverage =
                latest_channel_coverage_stats(buffers, classification_window_sec);
            let fallback_render_channels = if fallback_render_windows
                .iter()
                .any(|window| !window.is_empty())
            {
                fallback_render_windows
                    .iter()
                    .cloned()
                    .collect::<Vec<Vec<f32>>>()
            } else {
                decoded.channels.clone()
            };
            if !self.should_publish_classifier_render(
                &stream_key,
                audio_end_ns,
                pending_backlog_depth,
            ) {
                self.consume_source_manifest(&manifest).await;
                return None;
            }
            // Dispatch to Rayon so disk/DB I/O doesn't block the ingest loop.
            // Pass fallback_render_coverage as classification_coverage so run_math
            // builds an accurate coverage_json for the classifier_render manifest.
            let active_channels = eligible_coverage_channels(&fallback_render_coverage);
            return Some(ComputePayload {
                manifest,
                stream_key,
                channel_states,
                active_channels,
                classification_windows: core::array::from_fn(|_| Vec::new()),
                classification_coverage: fallback_render_coverage,
                omni_channels_override: Some(fallback_render_channels),
                omni_fallback_reason: Some("localization_coverage_unavailable".to_string()),
                skip_localization_result: true,
                source_ids,
                sr,
                now_ns,
                effective_sound_speed_mps,
                run_srp: false,
                run_classifier_render: true,
                manifest_store: self.manifest_store.clone(),
                derived_cache: self.derived_cache.clone(),
                state: self.state.clone(),
                classification_tx: self.classification_tx.clone(),
                config: self.config.clone(),
                consumed_since_prune: self.consumed_manifests_since_prune.clone(),
            });
        }

        let active_channels =
            eligible_localization_channels(&channel_states, self.config.min_coverage_ratio);
        let classification_windows =
            channel_windows_ending_at(buffers, end_ns, classification_window_sec);
        let classification_coverage =
            channel_coverage_ending_at(buffers, end_ns, classification_window_sec);

        if self.config.birdnet_hybrid_render_enabled
            && !self.should_publish_classifier_render(
                &stream_key,
                audio_end_ns,
                pending_backlog_depth,
            )
        {
            self.consume_source_manifest(&manifest).await;
            return None;
        }

        let windows: [Vec<f32>; 4] = core::array::from_fn(|ch| channel_states[ch].window.clone());
        let run_srp = self.should_run_localization(&stream_key, audio_end_ns, &windows);
        let run_classifier_render = self.config.birdnet_hybrid_render_enabled;

        Some(ComputePayload {
            manifest,
            stream_key,
            channel_states,
            active_channels,
            classification_windows,
            classification_coverage,
            source_ids,
            sr,
            now_ns,
            effective_sound_speed_mps,
            run_srp,
            run_classifier_render,
            omni_channels_override: None,
            omni_fallback_reason: None,
            skip_localization_result: false,
            manifest_store: self.manifest_store.clone(),
            derived_cache: self.derived_cache.clone(),
            state: self.state.clone(),
            classification_tx: self.classification_tx.clone(),
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
            let result = crate::actors::dsp_compute::run_math(payload);
            crate::actors::dsp_compute::run_io(result).await;
        }
    }

    async fn consume_source_manifest(&mut self, manifest: &DspManifest) {
        consume_manifest_standalone(
            manifest,
            &self.manifest_store,
            &self.consumed_manifests_since_prune,
            self.config.consumed_manifest_prune_interval.max(1),
            self.config.consumed_manifest_retention_max_files,
        )
        .await;
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

    fn should_run_localization(
        &mut self,
        stream_key: &str,
        audio_ns: u128,
        windows: &[Vec<f32>; 4],
    ) -> bool {
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
    manifest_store: &ManifestStore,
    state: &SharedDspState,
) -> Option<DspManifest> {
    let Some(pending) = result.pending_manifest else {
        return None;
    };
    if let Some(tx) = classification_tx {
        let req = ClassificationRequest {
            pcm_path: result.pcm_path.unwrap_or_default(),
            sample_rate_hz: result.sample_rate_hz,
            pending_manifest: pending.clone(),
        };
        match tx.try_send(req) {
            Ok(()) => {}
            Err(flume::TrySendError::Full(_)) => {
                let mut st = state.write().await;
                st.total_classification_drops += 1;
                warn!(
                    drops = st.total_classification_drops,
                    "ClassificationWorker channel full; publishing manifest without BirdNET labels"
                );
                drop(st);
                let _ = manifest_store.publish(pending.clone()).await;
            }
            Err(flume::TrySendError::Disconnected(_)) => {
                warn!(
                    "ClassificationWorker channel closed; publishing manifest without BirdNET labels"
                );
                let _ = manifest_store.publish(pending.clone()).await;
            }
        }
    } else if let Err(err) = manifest_store.publish(pending.clone()).await {
        warn!(error = %err, "failed to publish classifier render manifest");
    }
    Some(pending)
}

pub(crate) async fn consume_manifest_standalone(
    manifest: &DspManifest,
    manifest_store: &ManifestStore,
    consumed_since_prune: &Arc<AtomicU64>,
    prune_interval: u64,
    retention_max_files: usize,
) {
    if let Err(err) = manifest_store.mark_consumed(&manifest.manifest_id).await {
        warn!(
            manifest_id = %manifest.manifest_id,
            error = %err,
            "DSP worker failed to mark manifest consumed"
        );
        return;
    }
    let count = consumed_since_prune.fetch_add(1, Ordering::Relaxed) + 1;
    if count % prune_interval == 0 {
        let store = manifest_store.clone();
        tokio::spawn(async move {
            if let Err(error) = store
                .prune_consumed_manifests(retention_max_files)
                .await
            {
                warn!(error = %error, "DSP worker failed to prune consumed manifests");
            }
        });
    }
}

pub(crate) fn render_coverage_json(
    channel_coverage: &[Option<AudioCoverageStats>; 4],
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

fn localization_channel_states(
    buffers: &[SensorStreamBuffer; 4],
    end_ns: i128,
    window_seconds: f64,
) -> [LocalizationChannelState; 4] {
    core::array::from_fn(|channel_index| LocalizationChannelState {
        coverage: buffers[channel_index].coverage_ending_at(end_ns, window_seconds),
        window: buffers[channel_index]
            .window_ending_at(end_ns, window_seconds)
            .unwrap_or_default(),
    })
}

fn channel_windows_ending_at(
    buffers: &[SensorStreamBuffer; 4],
    end_ns: i128,
    window_seconds: f64,
) -> [Vec<f32>; 4] {
    core::array::from_fn(|channel_index| {
        buffers[channel_index]
            .window_ending_at(end_ns, window_seconds)
            .unwrap_or_default()
    })
}

fn channel_coverage_ending_at(
    buffers: &[SensorStreamBuffer; 4],
    end_ns: i128,
    window_seconds: f64,
) -> [Option<AudioCoverageStats>; 4] {
    core::array::from_fn(|channel_index| {
        buffers[channel_index].coverage_ending_at(end_ns, window_seconds)
    })
}

fn latest_channel_windows(
    buffers: &[SensorStreamBuffer; 4],
    window_seconds: f64,
) -> [Vec<f32>; 4] {
    core::array::from_fn(|channel_index| buffers[channel_index].latest_window(window_seconds))
}

fn latest_channel_coverage_stats(
    buffers: &[SensorStreamBuffer; 4],
    window_seconds: f64,
) -> [Option<AudioCoverageStats>; 4] {
    core::array::from_fn(|channel_index| {
        buffers[channel_index].latest_coverage_stats(window_seconds)
    })
}

fn eligible_coverage_channels(channel_coverage: &[Option<AudioCoverageStats>; 4]) -> Vec<usize> {
    channel_coverage
        .iter()
        .enumerate()
        .filter_map(|(channel_index, coverage)| coverage.as_ref().map(|_| channel_index))
        .collect()
}

fn eligible_localization_channels(
    channel_states: &[LocalizationChannelState; 4],
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
            // Anchor sample-index math to Time of Receipt (tor_ns) rather than now_ns
            // so that consecutive packets align seamlessly regardless of queue jitter.
            let anchor_ns = first_handle.tor_ns.unwrap_or(now_ns as u64);
            decoded.start_sample_index.map(|sample_index| {
                sample_index_to_absolute_time_from_now_ns(sample_index, sample_rate_hz, anchor_ns.into())
            })
        })
        .or_else(|| {
            let anchor_ns = first_handle.tor_ns.unwrap_or(now_ns as u64);
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
    now_ns.saturating_sub(newest_source_ns) > horizon_ns
}

fn classifier_render_min_interval_ns(
    classifier_render_min_interval_seconds: f64,
    pending_backlog_depth: usize,
) -> u128 {
    if classifier_render_min_interval_seconds <= 0.0 {
        return 0;
    }
    let backlog_multiplier = match pending_backlog_depth {
        0..=128 => 1_u128,
        129..=256 => 2_u128,
        257..=512 => 3_u128,
        _ => 4_u128,
    };
    let base_interval_ns =
        (classifier_render_min_interval_seconds * 1_000_000_000.0).round() as u128;
    base_interval_ns.saturating_mul(backlog_multiplier)
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

pub(crate) fn system_now_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

#[cfg(test)]
#[path = "dsp_worker_tests.rs"]
mod dsp_worker_tests;
