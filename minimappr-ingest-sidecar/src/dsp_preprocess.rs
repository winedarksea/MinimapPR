//! DSP worker configuration and the per-node preprocessing chain
//! (`PreprocessStage` -> compiled filter stages -> `NodeAudioState`).

use super::*;

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
    /// Per-pair GCC-PHAT band and taper (`localization_band_policy.rs`).
    pub localization_band_policy: minimappr_core::localization_band_policy::LocalizationBandPolicy,
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
    /// Coverage race fix (`localization_coverage_wait.rs`): hold a cluster
    /// localization until every streaming member channel covers the window.
    /// Same keys/meaning as the Python `localization_coverage_*` settings.
    pub localization_coverage_wait_all_streaming_nodes_enabled: bool,
    pub localization_coverage_stall_budget_seconds: f64,
    pub localization_coverage_total_cap_seconds: f64,
    /// Localization-overhaul keys (`localization_solve_config.rs`).
    pub localization_solve: crate::localization_solve_config::LocalizationSolveConfig,
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
            localization_band_policy: Default::default(),
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
            localization_coverage_wait_all_streaming_nodes_enabled: true,
            localization_coverage_stall_budget_seconds: 0.5,
            localization_coverage_total_cap_seconds: 1.0,
            localization_solve: Default::default(),
        }
    }
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

// Biquad cascade design lives in `minimappr_core::butterworth_biquad`.
pub(crate) use minimappr_core::butterworth_biquad::{
    butter_highpass_sos, butter_lowpass_sos, BiquadCoefficients, BiquadState,
};

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
    pub(super) config_signature: Vec<PreprocessStage>,
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
