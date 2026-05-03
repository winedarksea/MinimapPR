use std::path::PathBuf;

use tracing::warn;
use uuid::Uuid;

use crate::{
    birdnet_render::{render_hybrid_pcm16le, render_omni_pcm16le, HybridRenderConfig},
    derived_cache::DerivedCache,
    dsp_worker::{DspWorkerConfig, SIRITH_MIC_POSITIONS_M},
    journal_reader::JournalPayloadHandle,
    manifests::{BirdnetHybridProvenance, ClassifierRenderManifestPayload, DspManifest},
    srp_phat::SrpPhatLocalization,
};

pub struct RenderPublishResult {
    /// Pre-built classifier_render manifest; not yet published to ManifestStore.
    /// None only on DerivedCache write failure.
    pub pending_manifest: Option<DspManifest>,
    /// Path to the PCM file in DerivedCache, needed by ClassificationWorker.
    pub pcm_path: Option<PathBuf>,
    pub sample_rate_hz: u32,
    pub failure_count: u64,
}

/// Metadata produced by the sync CPU render phase, consumed by the async I/O phase.
pub struct RenderMeta {
    pub render_kind: String,
    pub spatial_band: Option<[f32; 2]>,
    pub steering_solution: Option<String>,
    pub confidence: Option<f32>,
    pub effective_fallback_reason: Option<String>,
    pub source_channel_count: usize,
}

/// Compute PCM render bytes and metadata synchronously.
/// Safe to call directly from a Rayon thread — no async I/O, no spawn_blocking.
pub fn compute_render_bytes(
    config: &DspWorkerConfig,
    sound_speed_mps: f32,
    channels: &[Vec<f32>],
    sample_rate_hz: u32,
    localization: Option<&SrpPhatLocalization>,
    fallback_reason: Option<String>,
) -> (Vec<u8>, RenderMeta) {
    let source_channel_count = channels.len();
    let use_hybrid = localization.filter(|sol| {
        sol.confidence >= config.min_localization_confidence && fallback_reason.is_none()
    });

    if let Some(solution) = use_hybrid {
        let hybrid_config = HybridRenderConfig {
            spatial_band_hz: config.spatial_blend_band_hz,
            pre_blend_highpass_hz: config.pre_blend_highpass_hz,
            sound_speed_mps,
        };
        let bytes = render_hybrid_pcm16le(
            channels,
            solution.steering_direction,
            &SIRITH_MIC_POSITIONS_M,
            sample_rate_hz,
            hybrid_config,
        );
        let meta = RenderMeta {
            render_kind: "birdnet_hybrid_spatial_blend".to_string(),
            spatial_band: Some(config.spatial_blend_band_hz),
            steering_solution: Some(format!(
                "srp_phat:{:.4},{:.4},{:.4}",
                solution.steering_direction[0],
                solution.steering_direction[1],
                solution.steering_direction[2],
            )),
            confidence: Some(solution.confidence),
            effective_fallback_reason: None,
            source_channel_count,
        };
        (bytes, meta)
    } else {
        let bytes = render_omni_pcm16le(channels);
        let confidence = localization.map(|sol| sol.confidence);
        let effective_fallback_reason =
            fallback_reason.or_else(|| Some("localization_unavailable".to_string()));
        let meta = RenderMeta {
            render_kind: "birdnet_omni_fallback".to_string(),
            spatial_band: None,
            steering_solution: None,
            confidence,
            effective_fallback_reason,
            source_channel_count,
        };
        (bytes, meta)
    }
}

/// Write PCM bytes to the DerivedCache and build a RenderPublishResult.
/// Async I/O only — no CPU render math.
#[allow(clippy::too_many_arguments)]
pub async fn write_render_to_cache(
    derived_cache: &DerivedCache,
    manifest: &DspManifest,
    stream_key: &str,
    pcm_bytes: Vec<u8>,
    meta: RenderMeta,
    coverage_stats: Option<serde_json::Value>,
    source_ids: Vec<String>,
    now_ns: u128,
    sample_rate_hz: u32,
) -> RenderPublishResult {
    let entry = match derived_cache
        .record_entry(
            "classifier_render".to_string(),
            &pcm_bytes,
            source_ids,
            now_ns,
        )
        .await
    {
        Ok(entry) => entry,
        Err(err) => {
            warn!(
                manifest_id = %manifest.manifest_id,
                error = %err,
                "DSP worker failed to record classifier render"
            );
            return RenderPublishResult {
                pending_manifest: None,
                pcm_path: None,
                sample_rate_hz,
                failure_count: 1,
            };
        }
    };
    let pcm_path = entry.path.clone();
    let sample_count = pcm_bytes.len() / 2;
    let manifest_payload = ClassifierRenderManifestPayload {
        render_id: entry.derived_id.clone(),
        render_kind: meta.render_kind.clone(),
        sample_rate_hz,
        channels: 1,
        sample_count,
        sample_format: "pcm16le".to_string(),
        effective_spatial_band: meta.spatial_band,
        source_channel_count: meta.source_channel_count,
        fallback_reason: meta.effective_fallback_reason.clone(),
    };
    let derived_handle = JournalPayloadHandle {
        journal_epoch: 0,
        segment_id: entry.derived_id.clone(),
        stream_key: stream_key.to_string(),
        payload_offset_bytes: 0,
        payload_length_bytes: entry.byte_length,
        toa_ns: manifest.source_handles.first().and_then(|h| h.toa_ns),
        tor_ns: manifest.source_handles.first().and_then(|h| h.tor_ns),
        sample_index_start: None,
        sample_count: Some(sample_count as u64),
        integrity_hash: String::new(),
        segment_path: entry.path.clone(),
    };
    // BirdNET labels are filled in by ClassificationWorker before ManifestStore.publish.
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
        }),
        coverage_stats,
        promotion_ready: true,
        env_samples: None,
        node_context: None,
            raw_payload: None,
    };
    RenderPublishResult {
        pending_manifest: Some(pending),
        pcm_path: Some(pcm_path),
        sample_rate_hz,
        failure_count: 0,
    }
}
