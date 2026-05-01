use std::{
    collections::HashMap,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use tokio::{sync::Mutex, time};
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
    srp_phat::{estimate_tetrahedral_steering, SrpPhatConfig, SrpPhatLocalization},
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
    pub pending_count: usize,
    pub recent_results: Vec<DspManifest>,
}

pub type SharedDspState = Arc<Mutex<DspWorkerState>>;

#[derive(Clone, Debug)]
struct LocalizationChannelState {
    coverage: Option<AudioCoverageStats>,
    window: Vec<f32>,
}

pub struct DspWorker {
    manifest_store: ManifestStore,
    derived_cache: DerivedCache,
    config: DspWorkerConfig,
    state: SharedDspState,
    classifier_annotator: Option<ManifestClassificationAnnotator>,
    /// Per-stream sample buffers (one channel-set per stream_key).
    buffers: HashMap<String, [SensorStreamBuffer; 4]>,
    last_classifier_render_ns_by_stream: HashMap<String, u128>,
    consumed_manifests_since_prune: u64,
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
            buffers: HashMap::new(),
            last_classifier_render_ns_by_stream: HashMap::new(),
            consumed_manifests_since_prune: 0,
        }
    }

    /// Main processing loop. Runs forever as a tokio task.
    pub async fn run_loop(mut self) {
        info!("DSP worker started");
        let interval = time::Duration::from_millis(self.config.poll_interval_ms);
        loop {
            {
                let mut st = self.state.lock().await;
                st.worker_running = true;
                st.last_heartbeat_ns = Some(system_now_ns());
            }
            self.process_pending().await;
            time::sleep(interval).await;
        }
    }

    async fn process_pending(&mut self) {
        let batch_limit = self.config.pending_manifest_batch_size.max(1);
        let pending = match self
            .manifest_store
            .query_pending_limited("raw_journal_append", batch_limit)
            .await
        {
            Ok(m) => m,
            Err(err) => {
                warn!(error = %err, "DSP worker failed to query pending manifests");
                return;
            }
        };

        {
            let mut st = self.state.lock().await;
            st.pending_count = pending.len();
        }

        let pending_backlog_depth = pending.len();
        for manifest in pending {
            self.process_one(manifest, pending_backlog_depth).await;
        }
    }

    async fn process_one(&mut self, manifest: DspManifest, pending_backlog_depth: usize) {
        let now_ns = system_now_ns();

        let Some(first_handle) = manifest.source_handles.first() else {
            let _ = self
                .manifest_store
                .mark_consumed(&manifest.manifest_id)
                .await;
            return;
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
                return;
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
                return;
            }
        };
        if decoded.channels.is_empty() {
            let _ = self
                .manifest_store
                .mark_consumed(&manifest.manifest_id)
                .await;
            return;
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
            let mut st = self.state.lock().await;
            st.total_stale_manifest_skips += 1;
            st.last_processed_ns = Some(now_ns);
            return;
        }
        let window_sec = self.config.window_seconds;
        let classification_window_sec = self.config.classification_window_seconds.max(window_sec);
        if decoded.channels.len() < 4 {
            if !self.config.birdnet_hybrid_render_enabled {
                self.consume_source_manifest(&manifest).await;
                return;
            }
            if !self.should_publish_classifier_render(&stream_key, now_ns, pending_backlog_depth) {
                self.consume_source_manifest(&manifest).await;
                return;
            }
            let render_result = publish_omni_render(
                RenderPublishContext {
                    manifest_store: &self.manifest_store,
                    derived_cache: &self.derived_cache,
                    config: &self.config,
                    sound_speed_mps: effective_sound_speed_mps,
                    classifier_annotator: self.classifier_annotator.as_mut(),
                },
                &manifest,
                &stream_key,
                &decoded.channels,
                sr,
                source_ids,
                now_ns,
                Some("single_sensor_or_non_array_node".to_string()),
            )
            .await;
            self.note_failures(render_result.failure_count).await;
            if let Some(render_manifest) = render_result.manifest {
                let mut st = self.state.lock().await;
                st.last_processed_ns = Some(now_ns);
                st.total_classifier_renders += 1;
                st.recent_results.push(render_manifest);
                if st.recent_results.len() > 50 {
                    st.recent_results.remove(0);
                }
            }
            self.consume_source_manifest(&manifest).await;
            return;
        }

        let frames = decoded
            .channels
            .iter()
            .take(4)
            .map(Vec::len)
            .min()
            .unwrap_or(0);
        let render_duration_ns =
            (frames as i128).saturating_mul(1_000_000_000) / i128::from(sr.max(1));
        let start_time_ns = resolve_buffer_start_time_ns(&decoded, first_handle, sr, now_ns);
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
                return;
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
                return;
            }
            let fallback_render_windows =
                latest_channel_windows(buffers, classification_window_sec);
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
            if !self.should_publish_classifier_render(&stream_key, now_ns, pending_backlog_depth) {
                self.consume_source_manifest(&manifest).await;
                return;
            }
            let render_result = publish_omni_render(
                RenderPublishContext {
                    manifest_store: &self.manifest_store,
                    derived_cache: &self.derived_cache,
                    config: &self.config,
                    sound_speed_mps: effective_sound_speed_mps,
                    classifier_annotator: self.classifier_annotator.as_mut(),
                },
                &manifest,
                &stream_key,
                &fallback_render_channels,
                sr,
                source_ids,
                now_ns,
                Some("localization_coverage_unavailable".to_string()),
            )
            .await;
            self.note_failures(render_result.failure_count).await;
            if let Some(render_manifest) = render_result.manifest {
                let mut st = self.state.lock().await;
                st.last_processed_ns = Some(now_ns);
                st.total_classifier_renders += 1;
                st.recent_results.push(render_manifest);
                if st.recent_results.len() > 50 {
                    st.recent_results.remove(0);
                }
            }
            self.consume_source_manifest(&manifest).await;
            return;
        }

        let active_channels =
            eligible_localization_channels(&channel_states, self.config.min_coverage_ratio);
        let windows: [Vec<f32>; 4] = core::array::from_fn(|ch| channel_states[ch].window.clone());
        let classification_windows =
            channel_windows_ending_at(buffers, end_ns, classification_window_sec);
        if self.config.birdnet_hybrid_render_enabled
            && !self.should_publish_classifier_render(&stream_key, now_ns, pending_backlog_depth)
        {
            self.consume_source_manifest(&manifest).await;
            return;
        }
        let localization_evaluation = estimate_tetrahedral_steering(
            &windows,
            &active_channels,
            &SIRITH_MIC_POSITIONS_M,
            sr,
            effective_sound_speed_mps,
            SrpPhatConfig {
                localization_band_hz: self.config.localization_band_hz,
                grid_resolution_m: self.config.localization_srp_grid_resolution_m,
                search_padding_m: self.config.localization_search_padding_m,
                ..SrpPhatConfig::default()
            },
        );
        let pair_diagnostics = localization_evaluation
            .pair_tdoas
            .iter()
            .map(pair_tdoa_diagnostic)
            .collect::<Vec<_>>();
        let localization = localization_evaluation.localization;
        let localization_payload = localization_manifest_payload(
            &localization,
            self.config.localization_band_hz,
            pair_diagnostics.clone(),
        );
        let fallback_reason = if localization.resolved_algorithm != "srp_phat" {
            Some(localization.resolved_algorithm.clone())
        } else if localization.confidence < self.config.min_localization_confidence {
            Some("low_localization_confidence".to_string())
        } else {
            None
        };
        let render_result = if self.config.birdnet_hybrid_render_enabled {
            Some(
                publish_classifier_render(
                    RenderPublishContext {
                        manifest_store: &self.manifest_store,
                        derived_cache: &self.derived_cache,
                        config: &self.config,
                        sound_speed_mps: effective_sound_speed_mps,
                        classifier_annotator: self.classifier_annotator.as_mut(),
                    },
                    &manifest,
                    &stream_key,
                    &classification_windows,
                    sr,
                    source_ids,
                    now_ns,
                    Some(&localization),
                    fallback_reason,
                )
                .await,
            )
        } else {
            None
        };
        if let Some(result) = render_result.as_ref() {
            self.note_failures(result.failure_count).await;
        }
        let render_manifest = render_result.and_then(|result| result.manifest);

        let coverage_json = serde_json::to_value(serde_json::json!({
            "per_channel": channel_states
                .iter()
                .map(|state| state.coverage.clone())
                .collect::<Vec<_>>(),
            "active_channels": active_channels,
            "threshold": self.config.min_coverage_ratio,
        }))
        .ok();
        let published = DspManifest {
            manifest_id: format!("manifest-{}", Uuid::new_v4()),
            manifest_type: "localization_result".to_string(),
            created_ns: now_ns,
            source_handles: manifest.source_handles.clone(),
            derived_handle: None,
            localization: Some(localization_payload),
            classifier_render: render_manifest
                .as_ref()
                .and_then(|manifest| manifest.classifier_render.clone()),
            birdnet: None,
            coverage_stats: coverage_json,
            promotion_ready: false,
        };

        if let Err(err) = self.manifest_store.publish(published.clone()).await {
            self.note_failure().await;
            warn!(
                manifest_id = %manifest.manifest_id,
                error = %err,
                "DSP worker failed to publish localization manifest"
            );
        }

        self.consume_source_manifest(&manifest).await;

        let mut st = self.state.lock().await;
        st.last_processed_ns = Some(now_ns);
        st.total_tdoa_results += 1;
        if published.localization.is_some() {
            st.total_localization_results += 1;
        }
        if published.classifier_render.is_some() {
            st.total_classifier_renders += 1;
        }
        if let Some(render_manifest) = render_manifest {
            st.recent_results.push(render_manifest);
        }
        st.recent_results.push(published);
        if st.recent_results.len() > 50 {
            st.recent_results.remove(0);
        }
    }

    async fn consume_source_manifest(&mut self, manifest: &DspManifest) {
        if let Err(err) = self
            .manifest_store
            .mark_consumed(&manifest.manifest_id)
            .await
        {
            warn!(
                manifest_id = %manifest.manifest_id,
                error = %err,
                "DSP worker failed to mark manifest consumed"
            );
            return;
        }

        self.consumed_manifests_since_prune = self.consumed_manifests_since_prune.saturating_add(1);
        let prune_interval = self.config.consumed_manifest_prune_interval.max(1);
        if self.consumed_manifests_since_prune % prune_interval == 0 {
            if let Err(error) = self
                .manifest_store
                .prune_consumed_manifests(self.config.consumed_manifest_retention_max_files)
                .await
            {
                warn!(
                    error = %error,
                    "DSP worker failed to prune consumed manifests"
                );
            }
        }
    }

    async fn note_failures(&self, count: u64) {
        if count == 0 {
            return;
        }
        let mut st = self.state.lock().await;
        st.total_failures += count;
    }

    async fn note_failure(&self) {
        self.note_failures(1).await;
    }

    fn should_publish_classifier_render(
        &mut self,
        stream_key: &str,
        now_ns: u128,
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
            Some(last_ns) if now_ns.saturating_sub(last_ns) < min_interval_ns => false,
            _ => {
                self.last_classifier_render_ns_by_stream
                    .insert(stream_key.to_string(), now_ns);
                true
            }
        }
    }
}

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

fn latest_channel_windows(buffers: &[SensorStreamBuffer; 4], window_seconds: f64) -> [Vec<f32>; 4] {
    core::array::from_fn(|channel_index| buffers[channel_index].latest_window(window_seconds))
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
        .or_else(|| first_handle.tor_ns.map(i128::from))
        .or_else(|| {
            decoded
                .start_sample_index
                .map(|sample_index| sample_index_to_relative_time_ns(sample_index, sample_rate_hz))
        })
        .or_else(|| {
            first_handle.sample_index_start.map(|sample_index| {
                sample_index_to_relative_time_ns(sample_index as i64, sample_rate_hz)
            })
        })
        .unwrap_or(now_ns as i128)
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
