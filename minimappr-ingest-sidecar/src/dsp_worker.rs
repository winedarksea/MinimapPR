use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use tokio::{sync::mpsc, sync::RwLock, task::JoinSet, time};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::{
    audio_payload::{decode_audio_payload, DecodedAudioPayload},
    classifier_helper::ManifestClassificationAnnotator,
    derived_cache::DerivedCache,
    dsp::{AudioCoverageStats, SensorStreamBuffer},
    dsp_render_output::{publish_classifier_render, publish_omni_render, RenderPublishContext},
    gcc_phat::TdoaResult,
    journal_reader::{read_payload_with_mmap, JournalPayloadHandle},
    manifests::{DspManifest, LocalizationManifestPayload, ManifestStore, PairTdoaDiagnostic},
    srp_phat::{
        estimate_tetrahedral_steering, SrpPhatConfig, SrpPhatEvaluation, SrpPhatLocalization,
    },
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
    /// How often to poll for pending manifests.
    pub poll_interval_ms: u64,
    /// Maximum number of pending manifests to process per poll iteration.
    pub pending_manifest_batch_size: usize,
    /// DSP window size in seconds (default: 512/16000 ≈ 32 ms).
    pub window_seconds: f64,
    /// Classification render window in seconds.
    pub classification_window_seconds: f64,
    /// Minimum interval between classifier renders per stream.
    pub classifier_render_min_interval_seconds: f64,
    /// Maximum retained history per stream buffer.
    pub max_buffer_seconds: f64,
    /// Skip windows with coverage below this ratio.
    pub min_coverage_ratio: f64,
    /// Runtime-profile localization band reported in localization provenance.
    pub localization_band_hz: [f32; 2],
    /// Search resolution used by the SRP-PHAT grid in meters.
    pub localization_srp_grid_resolution_m: f32,
    /// Padding around the active array bounds used for SRP search.
    pub localization_search_padding_m: f32,
    /// Frequency band where the BirdNET render uses spatial steering.
    pub spatial_blend_band_hz: [f32; 2],
    /// High-pass floor applied to the steered render before spatial blending.
    pub pre_blend_highpass_hz: f32,
    /// Minimum SRP confidence required before producing a hybrid render.
    pub min_localization_confidence: f32,
    /// Enables classifier-ready BirdNET hybrid render manifests.
    pub birdnet_hybrid_render_enabled: bool,
    /// Skip stale manifests instead of mutating live buffers with old audio.
    pub skip_stale_manifests_for_live_buffer: bool,
    /// Keep at most this many consumed manifest files on disk.
    pub consumed_manifest_retention_max_files: usize,
    /// Run consumed-manifest pruning after this many consumes.
    pub consumed_manifest_prune_interval: u64,
    pub default_temperature_c: f32,
    pub default_humidity_fraction: f32,
    pub sound_speed_mps: f32,
    pub classifier_command_json: Option<String>,
    /// Minimum interval between SRP-PHAT runs per stream (milliseconds).
    /// Reduces 30+ Hz grid searches to ~4 Hz. Set to 0 to run on every frame.
    pub localization_cadence_ms: u64,
    /// Skip SRP-PHAT when the window's peak-channel RMS is below this value.
    /// 0.0 disables the energy gate.
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
struct LocalizationChannelState {
    coverage: Option<AudioCoverageStats>,
    window: Vec<f32>,
}

/// Carries a pre-built classifier_render manifest + the PCM path to the
/// ClassificationWorker, which calls BirdNET and publishes the annotated manifest.
pub struct ClassificationRequest {
    pub pcm_path: std::path::PathBuf,
    pub sample_rate_hz: u32,
    /// Pre-built DspManifest with birdnet.label/scores = None.
    pub pending_manifest: DspManifest,
}

/// Dedicated task that owns ManifestClassificationAnnotator and processes
/// ClassificationRequests from a bounded channel, decoupling BirdNET latency
/// from the main DSP worker loop.
pub struct ClassificationWorker {
    annotator: ManifestClassificationAnnotator,
    manifest_store: ManifestStore,
    rx: mpsc::Receiver<ClassificationRequest>,
}

impl ClassificationWorker {
    pub async fn run_loop(mut self) {
        while let Some(req) = self.rx.recv().await {
            let mut manifest = req.pending_manifest;
            match self
                .annotator
                .classify_render(&req.pcm_path, req.sample_rate_hz)
                .await
            {
                Ok(Some(cls)) => {
                    if let Some(bn) = manifest.birdnet.as_mut() {
                        bn.label = Some(cls.label);
                        bn.label_confidence = Some(cls.label_confidence);
                        bn.scores = Some(cls.scores);
                    }
                }
                Ok(None) => {}
                Err(err) => {
                    warn!(error = %err, "ClassificationWorker: BirdNET annotation failed");
                }
            }
            if let Err(err) = self.manifest_store.publish(manifest).await {
                warn!(error = %err, "ClassificationWorker: failed to publish manifest");
            }
        }
    }
}

/// All owned data needed to run the compute phase (SRP-PHAT + render + publish)
/// for a single manifest after the ingest (buffer-append) phase completes.
/// Passed into `run_compute`, which is spawned as an independent tokio task so
/// the main ingest loop can immediately proceed to the next manifest.
struct ComputePayload {
    manifest: DspManifest,
    stream_key: String,
    channel_states: [LocalizationChannelState; 4],
    active_channels: Vec<usize>,
    classification_windows: [Vec<f32>; 4],
    classification_coverage: [Option<AudioCoverageStats>; 4],
    source_ids: Vec<String>,
    sr: u32,
    now_ns: u128,
    effective_sound_speed_mps: f32,
    run_srp: bool,
    run_classifier_render: bool,
    manifest_store: ManifestStore,
    derived_cache: DerivedCache,
    state: SharedDspState,
    classification_tx: Option<mpsc::Sender<ClassificationRequest>>,
    config: DspWorkerConfig,
    consumed_since_prune: Arc<AtomicU64>,
}

pub struct DspWorker {
    manifest_store: ManifestStore,
    derived_cache: DerivedCache,
    config: DspWorkerConfig,
    state: SharedDspState,
    classifier_annotator: Option<ManifestClassificationAnnotator>,
    classification_tx: Option<mpsc::Sender<ClassificationRequest>>,
    /// In-process channel receiver for raw_journal_append manifests, bypassing
    /// the 20ms filesystem polling cycle when the ingest backend is co-located.
    raw_manifest_rx: Option<mpsc::Receiver<DspManifest>>,
    /// Per-stream sample buffers (one channel-set per stream_key).
    buffers: HashMap<String, [SensorStreamBuffer; 4]>,
    last_classifier_render_ns_by_stream: HashMap<String, u128>,
    last_localization_ns_by_stream: HashMap<String, u128>,
    /// Shared with compute tasks so pruning is counted across spawned tasks.
    consumed_manifests_since_prune: Arc<AtomicU64>,
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
            Ok(classifier_annotator) => classifier_annotator,
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
        }
    }

    /// Extracts the `ManifestClassificationAnnotator` from the worker and
    /// wires it to a dedicated `ClassificationWorker` task via a bounded channel.
    /// Returns `(self, Some(worker))` when classification is configured,
    /// `(self, None)` when no classifier command is set.
    pub fn with_classification_worker(
        mut self,
        channel_capacity: usize,
    ) -> (Self, Option<ClassificationWorker>) {
        let Some(annotator) = self.classifier_annotator.take() else {
            return (self, None);
        };
        let (tx, rx) = mpsc::channel(channel_capacity);
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

    /// Main processing loop. Runs forever as a tokio task.
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

        // Prefer channel delivery; fall back to filesystem poll when channel is empty.
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

        let mut compute_tasks: JoinSet<()> = JoinSet::new();

        for manifest in pending {
            if let Some(payload) = self.ingest_one(manifest, pending_backlog_depth).await {
                compute_tasks.spawn(run_compute(payload));
            }
        }

        while let Some(result) = compute_tasks.join_next().await {
            if let Err(err) = result {
                warn!(error = ?err, "DSP compute task panicked");
            }
        }
    }

    /// Reads, decodes, and buffer-appends a single manifest.  Handles all
    /// early-exit paths (stale, decode error, insufficient channels, no coverage)
    /// inline.  Returns `Some(ComputePayload)` when the main localization +
    /// render path should run; the caller spawns `run_compute` on the payload
    /// so the next manifest's ingest can begin immediately.
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
        let raw_payload = match read_payload_with_mmap(first_handle) {
            Ok(payload) => payload,
            Err(err) => {
                self.note_failure().await;
                warn!(
                    manifest_id = %manifest.manifest_id,
                    error = %err,
                    "DSP worker failed to read journal payload; consuming unreadable source manifest"
                );
                self.consume_source_manifest(&manifest).await;
                return None;
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
        let effective_sound_speed_mps = resolve_effective_sound_speed_mps(
            &self.config,
            decoded.temperature_c,
            decoded.humidity_fraction,
        );

        // Compute the audio-timeline endpoint of this frame up front so that
        // cadence decisions below are based on audio time, not wall-clock time.
        // Wall-clock can be misleading when the worker processes a burst of
        // queued manifests faster than real-time (Bug 3 fix).
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
            let render_result = publish_omni_render(
                RenderPublishContext {
                    derived_cache: &self.derived_cache,
                    config: &self.config,
                    sound_speed_mps: effective_sound_speed_mps,
                },
                &manifest,
                &stream_key,
                &decoded.channels,
                None,
                sr,
                source_ids,
                now_ns,
                Some("single_sensor_or_non_array_node".to_string()),
            )
            .await;
            self.note_failures(render_result.failure_count).await;
            if let Some(pending) = self.dispatch_classification_result(render_result).await {
                let mut st = self.state.write().await;
                st.last_processed_ns = Some(now_ns);
                st.total_classifier_renders += 1;
                st.recent_results.push(pending);
                if st.recent_results.len() > 50 {
                    st.recent_results.remove(0);
                }
            }
            self.consume_source_manifest(&manifest).await;
            return None;
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
            let render_result = publish_omni_render(
                RenderPublishContext {
                    derived_cache: &self.derived_cache,
                    config: &self.config,
                    sound_speed_mps: effective_sound_speed_mps,
                },
                &manifest,
                &stream_key,
                &fallback_render_channels,
                render_coverage_json(
                    &fallback_render_coverage,
                    &eligible_coverage_channels(&fallback_render_coverage),
                    self.config.min_coverage_ratio,
                    "classification_latest",
                ),
                sr,
                source_ids,
                now_ns,
                Some("localization_coverage_unavailable".to_string()),
            )
            .await;
            self.note_failures(render_result.failure_count).await;
            if let Some(pending) = self.dispatch_classification_result(render_result).await {
                let mut st = self.state.write().await;
                st.last_processed_ns = Some(now_ns);
                st.total_classifier_renders += 1;
                st.recent_results.push(pending);
                if st.recent_results.len() > 50 {
                    st.recent_results.remove(0);
                }
            }
            self.consume_source_manifest(&manifest).await;
            return None;
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

        // Extract localization windows to pass to the compute task. The RMS check
        // inside should_run_localization is done here (ingest phase) because it
        // reads from the same window data that the compute task will use.
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
            manifest_store: self.manifest_store.clone(),
            derived_cache: self.derived_cache.clone(),
            state: self.state.clone(),
            classification_tx: self.classification_tx.clone(),
            config: self.config.clone(),
            consumed_since_prune: self.consumed_manifests_since_prune.clone(),
        })
    }

    #[cfg(test)]
    async fn process_one(&mut self, manifest: DspManifest, pending_backlog_depth: usize) {
        if let Some(payload) = self.ingest_one(manifest, pending_backlog_depth).await {
            run_compute(payload).await;
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

    /// Sends a `RenderPublishResult` to the `ClassificationWorker` channel.
    /// Falls back to direct `ManifestStore` publish when no worker is configured
    /// or the channel is full, so the manifest is never silently lost.
    async fn dispatch_classification_result(
        &mut self,
        result: crate::dsp_render_output::RenderPublishResult,
    ) -> Option<DspManifest> {
        dispatch_classification_result_standalone(
            result,
            &self.classification_tx,
            &self.manifest_store,
            &self.state,
        )
        .await
    }

    /// Cadence gate: returns true if SRP-PHAT should run for this frame.
    /// Uses the audio-timeline endpoint (`audio_ns`) instead of wall-clock so
    /// that burst processing of queued manifests doesn't skip frames based on
    /// CPU execution speed.
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

    /// Cadence gate: returns true if a classifier render should be produced.
    /// Uses audio-timeline endpoint (`audio_ns`) for the same reason as
    /// `should_run_localization`.
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
// Free functions shared by the DspWorker ingest path and the spawned compute tasks
// ---------------------------------------------------------------------------

/// SRP-PHAT + render + publish for one manifest.  Designed to run as an
/// independent `tokio::spawn` task so the ingest loop can process the next
/// manifest while this one's CPU/IO work is in flight.
async fn run_compute(payload: ComputePayload) {
    let windows: [Vec<f32>; 4] =
        core::array::from_fn(|ch| payload.channel_states[ch].window.clone());

    let localization_evaluation = if payload.run_srp {
        let srp_config = SrpPhatConfig {
            localization_band_hz: payload.config.localization_band_hz,
            grid_resolution_m: payload.config.localization_srp_grid_resolution_m,
            search_padding_m: payload.config.localization_search_padding_m,
            ..SrpPhatConfig::default()
        };
        let active_channels = payload.active_channels.clone();
        let mic_positions = SIRITH_MIC_POSITIONS_M;
        let sr = payload.sr;
        let sound_speed = payload.effective_sound_speed_mps;
        // CPU-bound FFT + grid search: run on the blocking thread pool so the
        // async executor remains free to accept new audio payloads.
        tokio::task::spawn_blocking(move || {
            estimate_tetrahedral_steering(
                &windows,
                &active_channels,
                &mic_positions,
                sr,
                sound_speed,
                srp_config,
            )
        })
        .await
        .expect("SRP-PHAT blocking task panicked")
    } else {
        SrpPhatEvaluation {
            localization: SrpPhatLocalization {
                attempted_algorithm: "localization_cadence_skipped".to_string(),
                resolved_algorithm: "localization_cadence_skipped".to_string(),
                steering_direction: [0.0, 0.0, 0.0],
                position_m: None,
                confidence: 0.0,
                residual_rms_seconds: 0.0,
                sound_speed_mps: payload.effective_sound_speed_mps,
            },
            pair_tdoas: vec![],
        }
    };

    let pair_diagnostics = localization_evaluation
        .pair_tdoas
        .iter()
        .map(pair_tdoa_diagnostic)
        .collect::<Vec<_>>();
    let localization = localization_evaluation.localization;
    let localization_payload = localization_manifest_payload(
        &localization,
        payload.config.localization_band_hz,
        pair_diagnostics.clone(),
    );
    let fallback_reason = if localization.resolved_algorithm != "srp_phat" {
        Some(localization.resolved_algorithm.clone())
    } else if localization.confidence < payload.config.min_localization_confidence {
        Some("low_localization_confidence".to_string())
    } else {
        None
    };

    let render_result = if payload.run_classifier_render {
        Some(
            publish_classifier_render(
                RenderPublishContext {
                    derived_cache: &payload.derived_cache,
                    config: &payload.config,
                    sound_speed_mps: payload.effective_sound_speed_mps,
                },
                &payload.manifest,
                &payload.stream_key,
                &payload.classification_windows,
                render_coverage_json(
                    &payload.classification_coverage,
                    &payload.active_channels,
                    payload.config.min_coverage_ratio,
                    "classification_trailing",
                ),
                payload.sr,
                payload.source_ids.clone(),
                payload.now_ns,
                Some(&localization),
                fallback_reason,
            )
            .await,
        )
    } else {
        None
    };

    if let Some(ref result) = render_result {
        if result.failure_count > 0 {
            let mut st = payload.state.write().await;
            st.total_failures += result.failure_count;
        }
    }

    let render_classifier_render = render_result
        .as_ref()
        .and_then(|r| r.pending_manifest.as_ref())
        .and_then(|m| m.classifier_render.clone());
    let render_pending = if let Some(result) = render_result {
        dispatch_classification_result_standalone(
            result,
            &payload.classification_tx,
            &payload.manifest_store,
            &payload.state,
        )
        .await
    } else {
        None
    };

    let coverage_json = serde_json::to_value(serde_json::json!({
        "per_channel": payload.channel_states
            .iter()
            .map(|state| state.coverage.clone())
            .collect::<Vec<_>>(),
        "active_channels": payload.active_channels,
        "threshold": payload.config.min_coverage_ratio,
    }))
    .ok();

    let published = DspManifest {
        manifest_id: format!("manifest-{}", Uuid::new_v4()),
        manifest_type: "localization_result".to_string(),
        created_ns: payload.now_ns,
        source_handles: payload.manifest.source_handles.clone(),
        derived_handle: None,
        localization: Some(localization_payload),
        classifier_render: render_classifier_render,
        birdnet: None,
        coverage_stats: coverage_json,
        promotion_ready: false,
    };

    if let Err(err) = payload.manifest_store.publish(published.clone()).await {
        warn!(
            manifest_id = %payload.manifest.manifest_id,
            error = %err,
            "DSP worker failed to publish localization manifest"
        );
    }

    consume_manifest_standalone(
        &payload.manifest,
        &payload.manifest_store,
        &payload.consumed_since_prune,
        payload.config.consumed_manifest_prune_interval.max(1),
        payload.config.consumed_manifest_retention_max_files,
    )
    .await;

    let mut st = payload.state.write().await;
    st.last_processed_ns = Some(payload.now_ns);
    let real_localization = localization.resolved_algorithm == "srp_phat"
        || localization
            .resolved_algorithm
            .starts_with("srp_phat_degraded");
    if real_localization {
        st.total_tdoa_results += 1;
    }
    if published.localization.is_some() && real_localization {
        st.total_localization_results += 1;
    }
    if published.classifier_render.is_some() {
        st.total_classifier_renders += 1;
    }
    if let Some(pending) = render_pending {
        st.recent_results.push(pending);
    }
    st.recent_results.push(published);
    if st.recent_results.len() > 50 {
        st.recent_results.remove(0);
    }
}

/// Routes a `RenderPublishResult` to the `ClassificationWorker` channel, or
/// falls back to a direct `ManifestStore` publish if no worker is wired up or
/// the channel is full.
async fn dispatch_classification_result_standalone(
    result: crate::dsp_render_output::RenderPublishResult,
    classification_tx: &Option<mpsc::Sender<ClassificationRequest>>,
    manifest_store: &ManifestStore,
    state: &SharedDspState,
) -> Option<DspManifest> {
    use tokio::sync::mpsc::error::TrySendError;
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
            Err(TrySendError::Full(_)) => {
                let mut st = state.write().await;
                st.total_classification_drops += 1;
                warn!(
                    drops = st.total_classification_drops,
                    "ClassificationWorker channel full; publishing manifest without BirdNET labels"
                );
                drop(st);
                let _ = manifest_store.publish(pending.clone()).await;
            }
            Err(TrySendError::Closed(_)) => {
                warn!("ClassificationWorker channel closed; publishing manifest without BirdNET labels");
                let _ = manifest_store.publish(pending.clone()).await;
            }
        }
    } else {
        if let Err(err) = manifest_store.publish(pending.clone()).await {
            warn!(error = %err, "failed to publish classifier render manifest");
        }
    }
    Some(pending)
}

/// Marks a manifest as consumed and periodically triggers a prune pass.
/// Called both from the ingest path (`&mut DspWorker`) and from spawned compute
/// tasks via a cloned `Arc<AtomicU64>` counter.
async fn consume_manifest_standalone(
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
        if let Err(error) = manifest_store
            .prune_consumed_manifests(retention_max_files)
            .await
        {
            warn!(error = %error, "DSP worker failed to prune consumed manifests");
        }
    }
}

// ---------------------------------------------------------------------------
// Pure helper functions (unchanged)
// ---------------------------------------------------------------------------

fn pair_tdoa_diagnostic(pair: &PairTdoa) -> PairTdoaDiagnostic {
    PairTdoaDiagnostic {
        ch_a: pair.ch_a,
        ch_b: pair.ch_b,
        delay_samples: pair.tdoa.delay_samples,
        lag_seconds: pair.tdoa.lag_seconds,
        confidence: pair.tdoa.confidence,
    }
}

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

fn latest_channel_windows(buffers: &[SensorStreamBuffer; 4], window_seconds: f64) -> [Vec<f32>; 4] {
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

fn render_coverage_json(
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

fn localization_manifest_payload(
    result: &SrpPhatLocalization,
    effective_band_hz: [f32; 2],
    pair_tdoas: Vec<PairTdoaDiagnostic>,
) -> LocalizationManifestPayload {
    LocalizationManifestPayload {
        attempted_algorithm: result.attempted_algorithm.clone(),
        resolved_algorithm: result.resolved_algorithm.clone(),
        steering_direction: (magnitude(result.steering_direction) > 1.0e-6)
            .then_some(result.steering_direction),
        position_m: result.position_m,
        confidence: result.confidence,
        residual_rms_seconds: result
            .residual_rms_seconds
            .is_finite()
            .then_some(result.residual_rms_seconds),
        sound_speed_mps: result.sound_speed_mps,
        effective_band_hz: Some(effective_band_hz),
        pair_tdoas,
    }
}

fn magnitude(vector: [f32; 3]) -> f32 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
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
            decoded.start_sample_index.map(|sample_index| {
                sample_index_to_absolute_time_from_now_ns(sample_index, sample_rate_hz, now_ns)
            })
        })
        .or_else(|| {
            first_handle.sample_index_start.map(|sample_index| {
                sample_index_to_absolute_time_from_now_ns(
                    sample_index as i64,
                    sample_rate_hz,
                    now_ns,
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

fn system_now_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

#[cfg(test)]
#[path = "dsp_worker_tests.rs"]
mod dsp_worker_tests;
