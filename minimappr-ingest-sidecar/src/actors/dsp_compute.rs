use tracing::warn;
use uuid::Uuid;

use crate::{
    dsp_render_output::{compute_render_bytes, write_render_to_cache, RenderMeta},
    dsp_worker::{
        consume_manifest_standalone, dispatch_classification_result_standalone,
        render_coverage_json, ComputePayload, PairTdoa, SIRITH_MIC_POSITIONS_M,
    },
    manifests::{DspManifest, LocalizationManifestPayload, PairTdoaDiagnostic},
    srp_phat::{
        estimate_tetrahedral_steering, SrpPhatConfig, SrpPhatEvaluation, SrpPhatLocalization,
    },
};

/// Intermediate result passed from the sync CPU math phase to the async I/O phase.
pub(crate) struct ComputeMathResult {
    pub(crate) payload: ComputePayload,
    pub(crate) localization: SrpPhatLocalization,
    pub(crate) pair_diagnostics: Vec<PairTdoaDiagnostic>,
    /// Raw PCM bytes — Some when run_classifier_render is true and render succeeded.
    pub(crate) pcm_bytes: Option<Vec<u8>>,
    pub(crate) render_meta: Option<RenderMeta>,
    /// Coverage JSON for the classifier_render manifest.
    pub(crate) render_coverage_json: Option<serde_json::Value>,
    /// Coverage JSON for the localization_result manifest.
    pub(crate) localization_coverage_json: Option<serde_json::Value>,
}

/// CPU-only math phase. Runs on a Rayon thread — no I/O, no .await.
///
/// Performs SRP-PHAT and PCM rendering synchronously. All I/O is deferred to
/// `run_io`, which is spawned onto the Tokio runtime after this returns.
pub fn run_math(payload: ComputePayload) -> ComputeMathResult {
    let windows: [Vec<f32>; 4] =
        core::array::from_fn(|ch| payload.channel_states[ch].window.clone());

    let localization_evaluation = if payload.run_srp {
        let srp_config = SrpPhatConfig {
            localization_band_hz: payload.config.localization_band_hz,
            grid_resolution_m: payload.config.localization_srp_grid_resolution_m,
            search_padding_m: payload.config.localization_search_padding_m,
            ..SrpPhatConfig::default()
        };
        // Direct call: we are already on a dedicated Rayon thread.
        estimate_tetrahedral_steering(
            &windows,
            &payload.active_channels,
            &SIRITH_MIC_POSITIONS_M,
            payload.sr,
            payload.effective_sound_speed_mps,
            srp_config,
        )
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

    // For omni-only fallback paths, use the caller-supplied reason string so
    // classifier_render manifests carry accurate provenance ("single_sensor_or_non_array_node",
    // "localization_coverage_unavailable", etc.) instead of the generic skip label.
    let fallback_reason = if let Some(ref reason) = payload.omni_fallback_reason {
        Some(reason.clone())
    } else if localization.resolved_algorithm != "srp_phat" {
        Some(localization.resolved_algorithm.clone())
    } else if localization.confidence < payload.config.min_localization_confidence {
        Some("low_localization_confidence".to_string())
    } else {
        None
    };

    let (pcm_bytes, render_meta, render_cov) = if payload.run_classifier_render {
        // Use omni_channels_override when set (single-sensor or no-coverage fallback paths),
        // otherwise use the normal classification windows.
        let render_channels: &[Vec<f32>] = payload
            .omni_channels_override
            .as_deref()
            .unwrap_or(payload.classification_windows.as_slice());

        let cov = render_coverage_json(
            &payload.classification_coverage,
            &payload.active_channels,
            payload.config.min_coverage_ratio,
            "classification_trailing",
        );

        // compute_render_bytes is sync — safe to call directly on a Rayon thread.
        let (bytes, meta) = compute_render_bytes(
            &payload.config,
            payload.effective_sound_speed_mps,
            render_channels,
            payload.sr,
            Some(&localization),
            fallback_reason.clone(),
        );
        (Some(bytes), Some(meta), cov)
    } else {
        (None, None, None)
    };

    let localization_coverage_json = serde_json::to_value(serde_json::json!({
        "per_channel": payload.channel_states
            .iter()
            .map(|state| state.coverage.clone())
            .collect::<Vec<_>>(),
        "active_channels": payload.active_channels,
        "threshold": payload.config.min_coverage_ratio,
    }))
    .ok();

    ComputeMathResult {
        payload,
        localization,
        pair_diagnostics,
        pcm_bytes,
        render_meta,
        render_coverage_json: render_cov,
        localization_coverage_json,
    }
}

/// Async I/O phase: write render to cache, publish manifests, forward to ClassificationWorker.
///
/// Spawned onto the Tokio runtime by `run_compute` so the Rayon thread is freed
/// immediately after `run_math` returns.
pub async fn run_io(result: ComputeMathResult) {
    let ComputeMathResult {
        payload,
        localization,
        pair_diagnostics,
        pcm_bytes,
        render_meta,
        render_coverage_json,
        localization_coverage_json,
    } = result;

    let render_result = if let Some(bytes) = pcm_bytes {
        let meta = render_meta.expect("render_meta is set whenever pcm_bytes is set");
        Some(
            write_render_to_cache(
                &payload.derived_cache,
                &payload.manifest,
                &payload.stream_key,
                bytes,
                meta,
                render_coverage_json.clone(),
                payload.source_ids.clone(),
                payload.now_ns,
                payload.sr,
            )
            .await,
        )
    } else {
        None
    };

    if let Some(ref r) = render_result {
        if r.failure_count > 0 {
            let mut st = payload.state.write().await;
            st.total_failures += r.failure_count;
        }
    }

    let localization_payload = localization_manifest_payload(
        &localization,
        payload.config.localization_band_hz,
        pair_diagnostics,
    );

    let has_classifier_render = render_result
        .as_ref()
        .and_then(|r| r.pending_manifest.as_ref())
        .and_then(|m| m.classifier_render.as_ref())
        .is_some();

    let render_classifier_render = render_result
        .as_ref()
        .and_then(|r| r.pending_manifest.as_ref())
        .and_then(|m| m.classifier_render.clone());

    let render_pending = if let Some(r) = render_result {
        dispatch_classification_result_standalone(
            r,
            &payload.classification_tx,
            &payload.manifest_store,
            &payload.state,
        )
        .await
    } else {
        None
    };

    // Omni-only fallback paths skip publishing a localization_result — they never
    // had a real localization attempt, matching the behaviour of the old inline paths.
    let published = if !payload.skip_localization_result {
        let m = DspManifest {
            manifest_id: format!("manifest-{}", Uuid::new_v4()),
            manifest_type: "localization_result".to_string(),
            created_ns: payload.now_ns,
            source_handles: payload.manifest.source_handles.clone(),
            derived_handle: None,
            localization: Some(localization_payload),
            classifier_render: render_classifier_render,
            birdnet: None,
            // Use the classification/render window coverage (the full audio window) for
            // audio quality display. The localization window is only ~32ms (TDOA), which
            // produces misleading missing% on the rendered 30s audio.
            coverage_stats: render_coverage_json.or(localization_coverage_json),
            promotion_ready: false,
            env_samples: None,
            // Carry node context forward so Python can reconstruct the NodeSpec
            // without relying on any persisted raw audio payload.
            node_context: payload.manifest.node_context.clone(),
            raw_payload: None,
        };
        if let Err(err) = payload.manifest_store.publish(m.clone()).await {
            warn!(
                manifest_id = %payload.manifest.manifest_id,
                error = %err,
                "DSP worker failed to publish localization manifest"
            );
        }
        Some(m)
    } else {
        None
    };

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
    if let Some(ref m) = published {
        if m.localization.is_some() && real_localization {
            st.total_localization_results += 1;
        }
    }
    if has_classifier_render {
        st.total_classifier_renders += 1;
    }
    if let Some(ref pending) = render_pending {
        st.recent_results.push(pending.clone());
        if st.recent_results.len() > 50 {
            st.recent_results.remove(0);
        }
    }
    if let Some(ref m) = published {
        st.recent_results.push(m.clone());
        if st.recent_results.len() > 50 {
            st.recent_results.remove(0);
        }
    }

    // Broadcast result manifests to Python consumers via SSE — zero-disk path.
    if let Some(ref tx) = payload.dsp_result_tx {
        if let Some(ref m) = render_pending {
            let _ = tx.send(m.clone());
        }
        if let Some(ref m) = published {
            let _ = tx.send(m.clone());
        }
    }
}

/// Entry point dispatched onto the Rayon pool from `process_pending`.
///
/// Runs `run_math` synchronously on the Rayon thread (no I/O), then calls
/// `handle.spawn(run_io(...))` which queues the async I/O onto the Tokio runtime
/// and returns immediately — the Rayon thread is free for the next frame.
pub fn run_compute(payload: ComputePayload, handle: tokio::runtime::Handle) {
    let result = run_math(payload);
    handle.spawn(run_io(result));
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
