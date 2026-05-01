use std::collections::BTreeMap;

use tracing::warn;
use uuid::Uuid;

use crate::{
    birdnet_render::{
        render_hybrid_pcm16le, render_omni_pcm16le, ClassifierRenderPayload, HybridRenderConfig,
    },
    classifier_helper::ManifestClassificationAnnotator,
    derived_cache::DerivedCache,
    dsp_worker::{DspWorkerConfig, SIRITH_MIC_POSITIONS_M},
    journal_reader::JournalPayloadHandle,
    manifests::{
        BirdnetHybridProvenance, ClassifierRenderManifestPayload, DspManifest, ManifestStore,
    },
    srp_phat::SrpPhatLocalization,
};

pub struct RenderPublishContext<'a> {
    pub manifest_store: &'a ManifestStore,
    pub derived_cache: &'a DerivedCache,
    pub config: &'a DspWorkerConfig,
    pub sound_speed_mps: f32,
    pub classifier_annotator: Option<&'a mut ManifestClassificationAnnotator>,
}

pub struct RenderPublishResult {
    pub manifest: Option<DspManifest>,
    pub failure_count: u64,
}

pub async fn publish_classifier_render(
    context: RenderPublishContext<'_>,
    manifest: &DspManifest,
    stream_key: &str,
    windows: &[Vec<f32>; 4],
    sample_rate_hz: u32,
    source_ids: Vec<String>,
    now_ns: u128,
    localization: Option<&SrpPhatLocalization>,
    fallback_reason: Option<String>,
) -> RenderPublishResult {
    let channels = windows.iter().cloned().collect::<Vec<_>>();
    let (payload, render_kind, spatial_band, steering_solution, confidence, fallback_reason) =
        if let Some(solution) = localization.filter(|solution| {
            solution.confidence >= context.config.min_localization_confidence
                && fallback_reason.is_none()
        }) {
            (
                render_hybrid_pcm16le(
                    &channels,
                    solution.steering_direction,
                    &SIRITH_MIC_POSITIONS_M,
                    sample_rate_hz,
                    HybridRenderConfig {
                        spatial_band_hz: context.config.spatial_blend_band_hz,
                        pre_blend_highpass_hz: context.config.pre_blend_highpass_hz,
                        sound_speed_mps: context.sound_speed_mps,
                    },
                ),
                "birdnet_hybrid_spatial_blend".to_string(),
                Some(context.config.spatial_blend_band_hz),
                Some(format!(
                    "srp_phat:{:.4},{:.4},{:.4}",
                    solution.steering_direction[0],
                    solution.steering_direction[1],
                    solution.steering_direction[2],
                )),
                Some(solution.confidence),
                None,
            )
        } else {
            (
                render_omni_pcm16le(&channels),
                "birdnet_omni_fallback".to_string(),
                None,
                None,
                localization.map(|solution| solution.confidence),
                fallback_reason.or_else(|| Some("localization_unavailable".to_string())),
            )
        };
    publish_render_manifest(
        context,
        manifest,
        stream_key,
        payload,
        source_ids,
        now_ns,
        sample_rate_hz,
        channels.len(),
        render_kind,
        spatial_band,
        steering_solution,
        confidence,
        fallback_reason,
    )
    .await
}

pub async fn publish_omni_render(
    context: RenderPublishContext<'_>,
    manifest: &DspManifest,
    stream_key: &str,
    channels: &[Vec<f32>],
    sample_rate_hz: u32,
    source_ids: Vec<String>,
    now_ns: u128,
    fallback_reason: Option<String>,
) -> RenderPublishResult {
    let payload = render_omni_pcm16le(channels);
    publish_render_manifest(
        context,
        manifest,
        stream_key,
        payload,
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
async fn publish_render_manifest(
    mut context: RenderPublishContext<'_>,
    manifest: &DspManifest,
    stream_key: &str,
    payload: Vec<u8>,
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
                manifest: None,
                failure_count: 1,
            };
        }
    };
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
    let mut authoritative_label: Option<String> = None;
    let mut authoritative_label_confidence: Option<f32> = None;
    let mut authoritative_scores: Option<BTreeMap<String, f32>> = None;
    if let Some(classifier_annotator) = context.classifier_annotator.as_mut() {
        match classifier_annotator
            .classify_render(&derived_handle.segment_path, sample_rate_hz)
            .await
        {
            Ok(Some(classification)) => {
                authoritative_label = Some(classification.label);
                authoritative_label_confidence = Some(classification.label_confidence);
                authoritative_scores = Some(classification.scores);
            }
            Ok(None) => {}
            Err(error) => {
                warn!(
                    manifest_id = %manifest.manifest_id,
                    error = %error,
                    "DSP worker classifier helper did not annotate render manifest"
                );
            }
        }
    }
    let published = DspManifest {
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
            label: authoritative_label,
            label_confidence: authoritative_label_confidence,
            scores: authoritative_scores,
        }),
        coverage_stats: None,
        promotion_ready: true,
    };
    if let Err(err) = context.manifest_store.publish(published.clone()).await {
        warn!(
            manifest_id = %manifest.manifest_id,
            error = %err,
            "DSP worker failed to publish classifier render manifest"
        );
        return RenderPublishResult {
            manifest: None,
            failure_count: 1,
        };
    }
    RenderPublishResult {
        manifest: Some(published),
        failure_count: 0,
    }
}
