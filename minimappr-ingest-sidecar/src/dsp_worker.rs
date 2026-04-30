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
    audio_payload::decode_audio_payload,
    derived_cache::DerivedCache,
    dsp::SensorStreamBuffer,
    dsp_render_output::{publish_classifier_render, publish_omni_render, RenderPublishContext},
    gcc_phat::{tetrahedral_gcc_phat, TdoaResult},
    journal_reader::read_payload_with_mmap,
    manifests::{DspManifest, LocalizationManifestPayload, ManifestStore, PairTdoaDiagnostic},
    srp_phat::{estimate_tetrahedral_steering, SrpPhatLocalization},
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
    /// DSP window size in seconds (default: 512/16000 ≈ 32 ms).
    pub window_seconds: f64,
    /// Skip windows with coverage below this ratio.
    pub min_coverage_ratio: f64,
    /// Runtime-profile localization band reported in localization provenance.
    pub localization_band_hz: [f32; 2],
    /// Frequency band where the BirdNET render uses spatial steering.
    pub spatial_blend_band_hz: [f32; 2],
    /// High-pass floor applied to the steered render before spatial blending.
    pub pre_blend_highpass_hz: f32,
    /// Minimum SRP confidence required before producing a hybrid render.
    pub min_localization_confidence: f32,
    /// Enables classifier-ready BirdNET hybrid render manifests.
    pub birdnet_hybrid_render_enabled: bool,
    pub sound_speed_mps: f32,
}

impl Default for DspWorkerConfig {
    fn default() -> Self {
        Self {
            poll_interval_ms: 20,
            window_seconds: 512.0 / 16_000.0,
            min_coverage_ratio: 0.85,
            localization_band_hz: [300.0, 3500.0],
            spatial_blend_band_hz: [1000.0, 3400.0],
            pre_blend_highpass_hz: 100.0,
            min_localization_confidence: 0.20,
            birdnet_hybrid_render_enabled: false,
            sound_speed_mps: 343.2,
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
    pub pending_count: usize,
    pub recent_results: Vec<DspManifest>,
}

pub type SharedDspState = Arc<Mutex<DspWorkerState>>;

pub struct DspWorker {
    manifest_store: ManifestStore,
    derived_cache: DerivedCache,
    config: DspWorkerConfig,
    state: SharedDspState,
    /// Per-stream sample buffers (one channel-set per stream_key).
    buffers: HashMap<String, [SensorStreamBuffer; 4]>,
}

impl DspWorker {
    pub fn new(
        manifest_store: ManifestStore,
        derived_cache: DerivedCache,
        config: DspWorkerConfig,
        state: SharedDspState,
    ) -> Self {
        Self {
            manifest_store,
            derived_cache,
            config,
            state,
            buffers: HashMap::new(),
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
        let pending = match self
            .manifest_store
            .query_pending("raw_journal_append")
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

        for manifest in pending {
            self.process_one(manifest).await;
        }
    }

    async fn process_one(&mut self, manifest: DspManifest) {
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
                    "DSP worker failed to read journal payload; skipping manifest"
                );
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
        let window_sec = self.config.window_seconds;
        if decoded.channels.len() < 4 {
            if !self.config.birdnet_hybrid_render_enabled {
                self.consume_source_manifest(&manifest).await;
                return;
            }
            let render_result = publish_omni_render(
                RenderPublishContext {
                    manifest_store: &self.manifest_store,
                    derived_cache: &self.derived_cache,
                    config: &self.config,
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
        let start_time_ns = decoded
            .start_time_ns
            .or_else(|| first_handle.sample_index_start.map(|s| s as i128))
            .unwrap_or(now_ns as i128);
        let start_sample_index = decoded.start_sample_index;
        let end_sample_index = decoded.end_sample_index;

        let buffers = self.buffers.entry(stream_key.clone()).or_insert_with(|| {
            core::array::from_fn(|_| SensorStreamBuffer::new(sr, window_sec * 4.0))
        });

        for (ch, buf) in buffers.iter_mut().enumerate() {
            let _ = buf.append(
                start_time_ns,
                &decoded.channels[ch],
                start_sample_index,
                end_sample_index,
            );
        }

        let end_ns = start_time_ns + (frames as i128 * 1_000_000_000 / i128::from(sr));
        let coverage = match buffers[0].coverage_ending_at(end_ns, window_sec) {
            Some(c) => c,
            None => {
                debug!(
                    manifest_id = %manifest.manifest_id,
                    "DSP worker: no coverage window yet; deferring"
                );
                return;
            }
        };

        if coverage.coverage_ratio < self.config.min_coverage_ratio {
            debug!(
                manifest_id = %manifest.manifest_id,
                coverage = coverage.coverage_ratio,
                threshold = self.config.min_coverage_ratio,
                "DSP worker: coverage below threshold; skipping TDOA"
            );
            let _ = self
                .manifest_store
                .mark_consumed(&manifest.manifest_id)
                .await;
            return;
        }

        let windows: [Vec<f32>; 4] = core::array::from_fn(|ch| {
            buffers[ch]
                .window_ending_at(end_ns, window_sec)
                .unwrap_or_default()
        });

        let tdoa_results = tetrahedral_gcc_phat(&windows, sr);
        const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let pair_tdoas: Vec<PairTdoa> = PAIRS
            .iter()
            .zip(tdoa_results.iter())
            .map(|(&(a, b), t)| PairTdoa {
                ch_a: a,
                ch_b: b,
                tdoa: t.clone(),
            })
            .collect();
        let pair_diagnostics = pair_tdoas
            .iter()
            .map(pair_tdoa_diagnostic)
            .collect::<Vec<_>>();
        let localization = estimate_tetrahedral_steering(
            &pair_tdoas,
            &SIRITH_MIC_POSITIONS_M,
            self.config.sound_speed_mps,
        );

        let localization_payload = localization.as_ref().map(|result| {
            localization_manifest_payload(
                result,
                self.config.localization_band_hz,
                pair_diagnostics.clone(),
            )
        });
        let fallback_reason = localization
            .as_ref()
            .filter(|result| result.confidence >= self.config.min_localization_confidence)
            .map(|_| None)
            .unwrap_or_else(|| Some("low_localization_confidence".to_string()));
        let render_result = if self.config.birdnet_hybrid_render_enabled {
            Some(
                publish_classifier_render(
                    RenderPublishContext {
                        manifest_store: &self.manifest_store,
                        derived_cache: &self.derived_cache,
                        config: &self.config,
                    },
                    &manifest,
                    &stream_key,
                    &windows,
                    sr,
                    source_ids,
                    now_ns,
                    localization.as_ref(),
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

        let coverage_json = serde_json::to_value(&coverage).ok();
        let published = DspManifest {
            manifest_id: format!("manifest-{}", Uuid::new_v4()),
            manifest_type: "localization_result".to_string(),
            created_ns: now_ns,
            source_handles: manifest.source_handles.clone(),
            derived_handle: None,
            localization: localization_payload,
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

    async fn consume_source_manifest(&self, manifest: &DspManifest) {
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

fn localization_manifest_payload(
    result: &SrpPhatLocalization,
    effective_band_hz: [f32; 2],
    pair_tdoas: Vec<PairTdoaDiagnostic>,
) -> LocalizationManifestPayload {
    LocalizationManifestPayload {
        attempted_algorithm: result.attempted_algorithm.clone(),
        resolved_algorithm: result.resolved_algorithm.clone(),
        steering_direction: Some(result.steering_direction),
        position_m: result.position_m,
        confidence: result.confidence,
        residual_rms_seconds: Some(result.residual_rms_seconds),
        sound_speed_mps: result.sound_speed_mps,
        effective_band_hz: Some(effective_band_hz),
        pair_tdoas,
    }
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
