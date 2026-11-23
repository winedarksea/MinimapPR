//! The ingest -> compute hand-off: everything `actors/dsp_compute` needs to
//! solve, render and publish one manifest.

use super::*;

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
