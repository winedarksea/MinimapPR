use std::path::PathBuf;

use tracing::warn;
use uuid::Uuid;

use crate::{
    birdnet_render::{
        render_hybrid_pcm16le, render_omni_pcm16le, ClassifierRenderPayload, HybridRenderConfig,
    },
    derived_cache::DerivedCache,
    dsp_worker::{DspWorkerConfig, SIRITH_MIC_POSITIONS_M},
    journal_reader::JournalPayloadHandle,
    manifests::{BirdnetHybridProvenance, ClassifierRenderManifestPayload, DspManifest},
    srp_phat::SrpPhatLocalization,
};

pub struct RenderPublishContext<'a> {
    pub derived_cache: &'a DerivedCache,
    pub config: &'a DspWorkerConfig,
    pub sound_speed_mps: f32,
}

pub struct RenderPublishResult {
    /// Pre-built classifier_render manifest; not yet published to ManifestStore.
    /// None only on DerivedCache write failure.
    pub pending_manifest: Option<DspManifest>,
    /// Path to the PCM file in DerivedCache, needed by ClassificationWorker.
    pub pcm_path: Option<PathBuf>,
    pub sample_rate_hz: u32,
    pub failure_count: u64,
}

pub async fn publish_classifier_render(
    context: RenderPublishContext<'_>,
    manifest: &DspManifest,
    stream_key: &str,
    windows: &[Vec<f32>; 4],
    coverage_stats: Option<serde_json::Value>,
    sample_rate_hz: u32,
    source_ids: Vec<String>,
    now_ns: u128,
    localization: Option<&SrpPhatLocalization>,
    fallback_reason: Option<String>,
) -> RenderPublishResult {
    let channels = windows.iter().cloned().collect::<Vec<_>>();

    let use_hybrid = localization.filter(|solution| {
        solution.confidence >= context.config.min_localization_confidence
            && fallback_reason.is_none()
    });

    // The render functions (rustfft beamforming, frequency blending) are CPU-bound.
    // Run them on the blocking thread pool so the async executor stays free.
    let render_kind;
    let spatial_band;
    let steering_solution;
    let confidence;
    let effective_fallback_reason;
    let payload: Vec<u8>;

    if let Some(solution) = use_hybrid {
        let steering_dir = solution.steering_direction;
        let mic_pos = SIRITH_MIC_POSITIONS_M;
        let hybrid_config = HybridRenderConfig {
            spatial_band_hz: context.config.spatial_blend_band_hz,
            pre_blend_highpass_hz: context.config.pre_blend_highpass_hz,
            sound_speed_mps: context.sound_speed_mps,
        };
        let channels_for_render = channels.clone();
        payload = tokio::task::spawn_blocking(move || {
            render_hybrid_pcm16le(
                &channels_for_render,
                steering_dir,
                &mic_pos,
                sample_rate_hz,
                hybrid_config,
            )
        })
        .await
        .expect("hybrid render task panicked");
        render_kind = "birdnet_hybrid_spatial_blend".to_string();
        spatial_band = Some(context.config.spatial_blend_band_hz);
        steering_solution = Some(format!(
            "srp_phat:{:.4},{:.4},{:.4}",
            solution.steering_direction[0],
            solution.steering_direction[1],
            solution.steering_direction[2],
        ));
        confidence = Some(solution.confidence);
        effective_fallback_reason = None;
    } else {
        let channels_for_render = channels.clone();
        payload = tokio::task::spawn_blocking(move || render_omni_pcm16le(&channels_for_render))
            .await
            .expect("omni render task panicked");
        render_kind = "birdnet_omni_fallback".to_string();
        spatial_band = None;
        steering_solution = None;
        confidence = localization.map(|solution| solution.confidence);
        effective_fallback_reason =
            fallback_reason.or_else(|| Some("localization_unavailable".to_string()));
    }

    build_render_manifest(
        context,
        manifest,
        stream_key,
        payload,
        coverage_stats,
        source_ids,
        now_ns,
        sample_rate_hz,
        channels.len(),
        render_kind,
        spatial_band,
        steering_solution,
        confidence,
        effective_fallback_reason,
    )
    .await
}

pub async fn publish_omni_render(
    context: RenderPublishContext<'_>,
    manifest: &DspManifest,
    stream_key: &str,
    channels: &[Vec<f32>],
    coverage_stats: Option<serde_json::Value>,
    sample_rate_hz: u32,
    source_ids: Vec<String>,
    now_ns: u128,
    fallback_reason: Option<String>,
) -> RenderPublishResult {
    // CPU-bound mix: run on blocking thread pool.
    let channels_owned = channels.to_vec();
    let payload = tokio::task::spawn_blocking(move || render_omni_pcm16le(&channels_owned))
        .await
        .expect("omni render task panicked");
    build_render_manifest(
        context,
        manifest,
        stream_key,
        payload,
        coverage_stats,
        source_ids,
        now_ns,
        sample_rate_hz,
        channels.len(),
        "birdnet_omni_fallback".to_string(),
        None,
        None,
        None,
        fallback_reason,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn build_render_manifest(
    context: RenderPublishContext<'_>,
    manifest: &DspManifest,
    stream_key: &str,
    payload: Vec<u8>,
    coverage_stats: Option<serde_json::Value>,
    source_ids: Vec<String>,
    now_ns: u128,
    sample_rate_hz: u32,
    source_channel_count: usize,
    render_kind: String,
    effective_spatial_band: Option<[f32; 2]>,
    steering_solution: Option<String>,
    confidence: Option<f32>,
    fallback_reason: Option<String>,
) -> RenderPublishResult {
    let entry = match context
        .derived_cache
        .record_entry(
            "classifier_render".to_string(),
            &payload,
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
    let sample_count = payload.len() / 2;
    let render_payload = ClassifierRenderPayload {
        render_id: entry.derived_id.clone(),
        render_kind: render_kind.clone(),
        sample_rate_hz,
        channels: 1,
        sample_count,
        sample_format: "pcm16le".to_string(),
        effective_spatial_band,
        source_channel_count,
        fallback_reason: fallback_reason.clone(),
    };
    let manifest_payload = ClassifierRenderManifestPayload {
        render_id: render_payload.render_id.clone(),
        render_kind: render_payload.render_kind.clone(),
        sample_rate_hz: render_payload.sample_rate_hz,
        channels: render_payload.channels,
        sample_count: render_payload.sample_count,
        sample_format: render_payload.sample_format.clone(),
        effective_spatial_band: render_payload.effective_spatial_band,
        source_channel_count: render_payload.source_channel_count,
        fallback_reason: render_payload.fallback_reason.clone(),
    };
    let derived_handle = JournalPayloadHandle {
        journal_epoch: 0,
        segment_id: entry.derived_id.clone(),
        stream_key: stream_key.to_string(),
        payload_offset_bytes: 0,
        payload_length_bytes: entry.byte_length,
        toa_ns: manifest
            .source_handles
            .first()
            .and_then(|handle| handle.toa_ns),
        tor_ns: manifest
            .source_handles
            .first()
            .and_then(|handle| handle.tor_ns),
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
            steering_solution,
            classifier_source_node: Some(stream_key.to_string()),
            spatial_blend_mode: render_kind,
            effective_spatial_band,
            confidence,
            fallback_reason,
            label: None,
            label_confidence: None,
            scores: None,
        }),
        coverage_stats,
        promotion_ready: true,
    };
    RenderPublishResult {
        pending_manifest: Some(pending),
        pcm_path: Some(pcm_path),
        sample_rate_hz,
        failure_count: 0,
    }
}
