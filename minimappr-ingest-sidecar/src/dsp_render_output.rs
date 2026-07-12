use tracing::warn;
use uuid::Uuid;

use crate::{
    birdnet_render::{render_band_split_pcm16le, render_omni_pcm16le, BandSplitRenderConfig},
    dsp_math::{norm3, scale3},
    dsp_worker::DspWorkerConfig,
    journal_reader::JournalPayloadHandle,
    manifests::{BirdnetHybridProvenance, ClassifierRenderManifestPayload, DspManifest},
    srp_phat::SrpPhatLocalization,
};

pub struct RenderPublishResult {
    /// Pre-built classifier_render manifest; not yet published to ManifestStore.
    /// None only on render failure.
    pub pending_manifest: Option<DspManifest>,
    /// Raw PCM16LE mono bytes produced by the render — carried in memory so
    /// ClassificationWorker can write to a temp file without a persistent cache.
    pub pcm_bytes: Option<Vec<u8>>,
    pub sample_rate_hz: u32,
    pub failure_count: u64,
}

/// Metadata produced by the sync CPU render phase, consumed by the async I/O phase.
pub struct RenderMeta {
    pub render_kind: String,
    pub render_start_ns: Option<u128>,
    pub render_end_ns: Option<u128>,
    pub spatial_band: Option<[f32; 2]>,
    pub steering_solution: Option<String>,
    pub confidence: Option<f32>,
    pub effective_fallback_reason: Option<String>,
    pub source_channel_count: usize,
    pub alias_cutoff_hz: Option<f32>,
    pub steering_model: Option<String>,
}

pub struct RenderComputeRequest<'a> {
    pub config: &'a DspWorkerConfig,
    pub sound_speed_mps: f32,
    pub channels: &'a [Vec<f32>],
    pub mic_positions_m: &'a [[f32; 3]],
    pub sample_rate_hz: u32,
    pub localization: Option<&'a SrpPhatLocalization>,
    pub fallback_reason: Option<String>,
    pub render_start_ns: Option<u128>,
    pub render_end_ns: Option<u128>,
}

/// Compute PCM render bytes and metadata synchronously.
/// Safe to call directly from a Rayon thread — no async I/O, no spawn_blocking.
pub fn compute_render_bytes(request: RenderComputeRequest<'_>) -> (Vec<u8>, RenderMeta) {
    let RenderComputeRequest {
        config,
        sound_speed_mps,
        channels,
        mic_positions_m,
        sample_rate_hz,
        localization,
        fallback_reason,
        render_start_ns,
        render_end_ns,
    } = request;
    let source_channel_count = channels.len();

    // Nearest-node omni source: always render the raw omni mix (no beamforming),
    // but preserve localization provenance so the Python side can pick the nearest
    // node / stamp confidence. The distinct render_kind lets the IAMF pipeline
    // treat it as a plain omni fallback (it only special-cases "birdnet_hybrid_*").
    if config.classification_audio_source == "nearest_node_omni" {
        let bytes = render_omni_pcm16le(channels);
        let meta = RenderMeta {
            render_kind: "nearest_node_omni".to_string(),
            render_start_ns,
            render_end_ns,
            spatial_band: None,
            steering_solution: None,
            confidence: localization.map(|sol| sol.confidence),
            effective_fallback_reason: fallback_reason,
            source_channel_count,
            alias_cutoff_hz: None,
            steering_model: None,
        };
        return (bytes, meta);
    }

    let use_hybrid = localization.filter(|sol| {
        sol.confidence >= config.min_localization_confidence
            && fallback_reason.is_none()
            && !mic_positions_m.is_empty()
    });

    if let Some(solution) = use_hybrid {
        let band_split_config = BandSplitRenderConfig {
            highpass_hz: config.band_split_highpass_hz,
            low_crossover_width_hz: config.band_split_low_crossover_width_hz,
            high_crossover_width_min_hz: config.band_split_high_crossover_width_min_hz,
            high_crossover_width_fraction: config.band_split_high_crossover_width_fraction,
            sound_speed_mps,
            band_max_clamp_hz: config.band_split_max_clamp_hz,
        };
        // Contract §1 steer-position rule: range_refined solves steer at the
        // solved position (near-field); range-unobservable modes project the
        // bearing at the solved/default range.
        let (steer_position_m, steering_model) = resolve_steer_position(solution, config);
        let output = render_band_split_pcm16le(
            channels,
            steer_position_m,
            mic_positions_m,
            sample_rate_hz,
            band_split_config,
        );
        let meta = RenderMeta {
            render_kind: "birdnet_band_split_das".to_string(),
            render_start_ns,
            render_end_ns,
            spatial_band: Some(output.effective_spatial_band),
            steering_solution: Some(format!(
                "srp_phat:{:.4},{:.4},{:.4}",
                solution.steering_direction[0],
                solution.steering_direction[1],
                solution.steering_direction[2],
            )),
            confidence: Some(solution.confidence),
            effective_fallback_reason: None,
            source_channel_count,
            alias_cutoff_hz: Some(output.alias_cutoff_hz),
            steering_model: Some(steering_model.to_string()),
        };
        (output.pcm_bytes, meta)
    } else {
        let bytes = render_omni_pcm16le(channels);
        let confidence = localization.map(|sol| sol.confidence);
        let effective_fallback_reason =
            fallback_reason.or_else(|| Some("localization_unavailable".to_string()));
        let meta = RenderMeta {
            render_kind: "birdnet_omni_fallback".to_string(),
            render_start_ns,
            render_end_ns,
            spatial_band: None,
            steering_solution: None,
            confidence,
            effective_fallback_reason,
            source_channel_count,
            alias_cutoff_hz: None,
            steering_model: None,
        };
        (bytes, meta)
    }
}

/// Resolve the steer position per BEAMFORMED_RENDER_CONTRACT.md §1.
fn resolve_steer_position(
    solution: &SrpPhatLocalization,
    config: &DspWorkerConfig,
) -> ([f32; 3], &'static str) {
    let mode = solution.range_projection_mode.as_deref();
    if mode == Some(crate::range_projection::RANGE_REFINED) {
        if let Some(position) = solution.position_m {
            return (position, "near_field");
        }
    }
    // Range-unobservable (or position missing): project the bearing at the
    // solved range when available, else the far-field default.
    let range_m = solution
        .position_m
        .map(norm3)
        .filter(|r| r.is_finite() && *r > 1e-3)
        .unwrap_or(config.localization_far_field_default_range_m);
    (
        scale3(solution.steering_direction, range_m),
        "bearing_projected",
    )
}

/// Build the classifier_render manifest and return raw PCM bytes in memory.
/// No disk writes — the caller is responsible for delivering PCM bytes to
/// ClassificationWorker via the in-process channel.
#[allow(clippy::too_many_arguments)]
pub fn build_render_result(
    manifest: &DspManifest,
    stream_key: &str,
    pcm_bytes: Vec<u8>,
    meta: RenderMeta,
    coverage_stats: Option<serde_json::Value>,
    _source_ids: Vec<String>,
    now_ns: u128,
    sample_rate_hz: u32,
) -> RenderPublishResult {
    let sample_count = pcm_bytes.len() / 2;
    // Sentinel derived_handle — segment_path is not a readable disk location in
    // the memory-only path. The inline raw_render_bytes field on the SSE manifest
    // carries the actual PCM; derived_handle conveys schema metadata only.
    let derived_handle = JournalPayloadHandle {
        journal_epoch: 0,
        segment_id: format!("derived-mem-{}", Uuid::new_v4()),
        stream_key: stream_key.to_string(),
        payload_offset_bytes: 0,
        payload_length_bytes: pcm_bytes.len() as u64,
        toa_ns: manifest.source_handles.first().and_then(|h| h.toa_ns),
        tor_ns: manifest.source_handles.first().and_then(|h| h.tor_ns),
        received_ns: None,
        sample_index_start: None,
        sample_count: Some(sample_count as u64),
        integrity_hash: String::new(),
        segment_path: std::path::PathBuf::new(), // not a readable path
    };
    let manifest_payload = ClassifierRenderManifestPayload {
        render_id: derived_handle.segment_id.clone(),
        render_kind: meta.render_kind.clone(),
        render_start_ns: meta.render_start_ns,
        render_end_ns: meta.render_end_ns,
        sample_rate_hz,
        channels: 1,
        sample_count,
        sample_format: "pcm16le".to_string(),
        effective_spatial_band: meta.spatial_band,
        source_channel_count: meta.source_channel_count,
        fallback_reason: meta.effective_fallback_reason.clone(),
        alias_cutoff_hz: meta.alias_cutoff_hz,
        steering_model: meta.steering_model.clone(),
    };
    let pending = DspManifest {
        manifest_id: format!("manifest-{}", Uuid::new_v4()),
        manifest_type: "classifier_render".to_string(),
        created_ns: now_ns,
        source_handles: manifest.source_handles.clone(),
        derived_handle: Some(derived_handle),
        localization: None,
        classifier_render: Some(manifest_payload),
        birdnet: Some(BirdnetHybridProvenance {
            steering_solution: meta.steering_solution,
            classifier_source_node: Some(stream_key.to_string()),
            spatial_blend_mode: meta.render_kind,
            effective_spatial_band: meta.spatial_band,
            confidence: meta.confidence,
            fallback_reason: meta.effective_fallback_reason,
            label: None,
            label_confidence: None,
            scores: None,
            features: None,
            alias_cutoff_hz: meta.alias_cutoff_hz,
            steering_model: meta.steering_model,
        }),
        coverage_stats,
        promotion_ready: true,
        env_samples: None,
        node_context: manifest.node_context.clone(),
        cluster_id: manifest.cluster_id.clone(),
        cluster_sensor_positions: manifest.cluster_sensor_positions.clone(),
        raw_payload: None,
        raw_render_bytes: None, // set by run_io before SSE broadcast
        raw_audio_frame: None,
        raw_audio_bytes: None,
    };
    RenderPublishResult {
        pending_manifest: Some(pending),
        pcm_bytes: Some(pcm_bytes),
        sample_rate_hz,
        failure_count: 0,
    }
}

/// Async wrapper kept for callers that previously used write_render_to_cache.
/// Now purely synchronous under the hood — no I/O, no DerivedCache.
#[allow(clippy::too_many_arguments)]
pub async fn write_render_to_cache(
    _derived_cache: &crate::derived_cache::DerivedCache,
    manifest: &DspManifest,
    stream_key: &str,
    pcm_bytes: Vec<u8>,
    meta: RenderMeta,
    coverage_stats: Option<serde_json::Value>,
    source_ids: Vec<String>,
    now_ns: u128,
    sample_rate_hz: u32,
) -> RenderPublishResult {
    if pcm_bytes.is_empty() {
        warn!(
            manifest_id = %manifest.manifest_id,
            "DSP worker: empty PCM render — skipping classifier render"
        );
        return RenderPublishResult {
            pending_manifest: None,
            pcm_bytes: None,
            sample_rate_hz,
            failure_count: 1,
        };
    }
    build_render_result(
        manifest,
        stream_key,
        pcm_bytes,
        meta,
        coverage_stats,
        source_ids,
        now_ns,
        sample_rate_hz,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with_source(source: &str) -> DspWorkerConfig {
        DspWorkerConfig {
            classification_audio_source: source.to_string(),
            ..DspWorkerConfig::default()
        }
    }

    #[test]
    fn nearest_node_omni_always_renders_omni_kind() {
        let config = config_with_source("nearest_node_omni");
        let channels = vec![
            vec![0.1_f32, -0.2, 0.3, -0.1],
            vec![0.2_f32, 0.1, -0.3, 0.0],
        ];
        let mic_positions_m = [[0.0_f32, 0.0, 0.0], [0.1, 0.0, 0.0]];
        let (bytes, meta) = compute_render_bytes(RenderComputeRequest {
            config: &config,
            sound_speed_mps: 343.0,
            channels: &channels,
            mic_positions_m: &mic_positions_m,
            sample_rate_hz: 16_000,
            localization: None,
            fallback_reason: None,
            render_start_ns: None,
            render_end_ns: None,
        });
        assert_eq!(meta.render_kind, "nearest_node_omni");
        assert!(meta.spatial_band.is_none());
        assert!(!bytes.is_empty());
    }

    #[test]
    fn beamformed_source_falls_back_to_omni_without_localization() {
        let config = config_with_source("beamformed");
        let channels = vec![vec![0.1_f32, -0.2, 0.3, -0.1]];
        let mic_positions_m = [[0.0_f32, 0.0, 0.0]];
        let (_bytes, meta) = compute_render_bytes(RenderComputeRequest {
            config: &config,
            sound_speed_mps: 343.0,
            channels: &channels,
            mic_positions_m: &mic_positions_m,
            sample_rate_hz: 16_000,
            localization: None,
            fallback_reason: None,
            render_start_ns: None,
            render_end_ns: None,
        });
        assert_eq!(meta.render_kind, "birdnet_omni_fallback");
    }
}
