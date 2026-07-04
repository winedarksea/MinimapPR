use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde_json::{Map, Value};
use tracing::warn;
use uuid::Uuid;

use crate::{
    dsp_render_output::{
        compute_render_bytes, write_render_to_cache, RenderComputeRequest, RenderMeta,
    },
    dsp_worker::{
        consume_manifest_standalone, dispatch_classification_result_standalone,
        render_coverage_json, source_manifest_was_persisted, ComputePayload, PairTdoa,
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
    /// Canonical render PCM bytes used for BirdNET classification.
    pub(crate) pcm_bytes: Option<Vec<u8>>,
    /// Listener-facing render bytes that may conceal missing spans on copied windows.
    pub(crate) listenable_pcm_bytes: Option<Vec<u8>>,
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
    let localization_evaluation = if payload.run_srp && payload.channel_states.len() >= 4 {
        let windows = payload
            .channel_states
            .iter()
            .map(|state| state.window.clone())
            .collect::<Vec<_>>();
        // Amplitude/SNR range prior (Phase 1c): when enabled, derive the projection
        // distance for unobservable-range solves from the reference-channel received
        // level instead of the fixed default. Ships off; both `None` reproduces the
        // fixed-prior behaviour. The Python re-solve path applies its own prior from
        // the manifest `received_level_dbfs`; this governs the Rust-native SRP solve.
        let (range_prior_m, range_prior_std_m) =
            if payload.config.localization_amplitude_range_prior_enabled {
                match max_window_rms(&payload.channel_states) {
                    Some(rms) => {
                        let received_level_db = (20.0 * (rms.max(1.0e-9)).log10()) as f32;
                        let (prior_range_m, _clamped) =
                            crate::range_projection::amplitude_range_prior_m(
                                received_level_db,
                                payload.config.localization_amplitude_reference_level_db,
                                payload.config.localization_amplitude_prior_min_range_m,
                                payload.config.localization_amplitude_prior_max_range_m,
                            );
                        (
                            Some(prior_range_m),
                            Some(
                                payload.config.localization_amplitude_prior_std_factor
                                    * prior_range_m,
                            ),
                        )
                    }
                    None => (None, None),
                }
            } else {
                (None, None)
            };
        let srp_config = SrpPhatConfig {
            localization_band_hz: payload.config.localization_band_hz,
            grid_resolution_m: payload.config.localization_srp_grid_resolution_m,
            search_padding_m: payload.config.localization_search_padding_m,
            interp_factor: payload.config.gcc_phat_interp_factor,
            far_field_default_range_m: payload.config.localization_far_field_default_range_m,
            far_field_max_range_m: payload.config.localization_far_field_max_range_m,
            range_prior_m,
            range_prior_std_m,
            ..SrpPhatConfig::default()
        };
        estimate_tetrahedral_steering(
            windows.as_slice(),
            &payload.active_channels,
            payload.mic_positions_m.as_slice(),
            payload.sr,
            payload.effective_sound_speed_mps,
            srp_config,
        )
    } else {
        let skipped_reason = if payload.run_srp {
            // run_srp was true but channel count was insufficient — shouldn't happen
            // after the ingest_one guard, but kept as a safety net.
            "insufficient_channels"
        } else {
            "localization_cadence_skipped"
        };
        SrpPhatEvaluation {
            localization: SrpPhatLocalization {
                attempted_algorithm: skipped_reason.to_string(),
                resolved_algorithm: skipped_reason.to_string(),
                steering_direction: [0.0, 0.0, 0.0],
                position_m: None,
                position_covariance_m2: None,
                range_observability: None,
                range_projection_mode: None,
                confidence: 0.0,
                residual_rms_seconds: f32::INFINITY,
                sound_speed_mps: payload.effective_sound_speed_mps,
                dominant_frequency_hz: None,
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

    let fallback_reason = if let Some(ref reason) = payload.omni_fallback_reason {
        Some(reason.clone())
    } else if localization.resolved_algorithm != "srp_phat" {
        Some(localization.resolved_algorithm.clone())
    } else if localization.confidence < payload.config.min_localization_confidence {
        Some("low_localization_confidence".to_string())
    } else {
        None
    };

    let (pcm_bytes, listenable_pcm_bytes, render_meta, render_cov) =
        if payload.run_classifier_render {
            let render_channels: &[Vec<f32>] = payload
                .omni_channels_override
                .as_deref()
                .unwrap_or(payload.classification_windows.as_slice());
            let listenable_render_channels: &[Vec<f32>] = payload
                .listenable_omni_channels_override
                .as_deref()
                .unwrap_or(payload.listenable_classification_windows.as_slice());

            let cov = render_coverage_json(
                &payload.classification_coverage,
                &payload.active_channels,
                payload.config.min_coverage_ratio,
                "classification_trailing",
            );

            let (bytes, meta) = compute_render_bytes(RenderComputeRequest {
                config: &payload.config,
                sound_speed_mps: payload.effective_sound_speed_mps,
                channels: render_channels,
                mic_positions_m: payload.mic_positions_m.as_slice(),
                sample_rate_hz: payload.sr,
                localization: Some(&localization),
                fallback_reason: fallback_reason.clone(),
                render_start_ns: payload.classifier_render_start_ns,
                render_end_ns: payload.classifier_render_end_ns,
            });
            let (listenable_bytes, _) = compute_render_bytes(RenderComputeRequest {
                config: &payload.config,
                sound_speed_mps: payload.effective_sound_speed_mps,
                channels: listenable_render_channels,
                mic_positions_m: payload.mic_positions_m.as_slice(),
                sample_rate_hz: payload.sr,
                localization: Some(&localization),
                fallback_reason: fallback_reason.clone(),
                render_start_ns: payload.classifier_render_start_ns,
                render_end_ns: payload.classifier_render_end_ns,
            });
            (Some(bytes), Some(listenable_bytes), Some(meta), cov)
        } else {
            (None, None, None, None)
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
        listenable_pcm_bytes,
        render_meta,
        render_coverage_json: render_cov,
        localization_coverage_json,
    }
}

/// Async I/O phase: build manifests, broadcast via SSE, forward to ClassificationWorker.
/// No disk writes for memory-path (channel-delivered) manifests.
pub async fn run_io(result: ComputeMathResult) {
    let ComputeMathResult {
        mut payload,
        localization,
        pair_diagnostics,
        pcm_bytes,
        listenable_pcm_bytes,
        render_meta,
        render_coverage_json,
        localization_coverage_json,
    } = result;

    enrich_node_context_with_runtime_telemetry(&mut payload);

    // Build the classifier_render manifest and keep PCM bytes in memory.
    let mut render_result = if let (Some(bytes), Some(meta)) = (pcm_bytes, render_meta) {
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

    // Received level (dBFS) of the loudest (reference) channel, for the Python
    // amplitude/SNR range prior (Phase 1c). Per-node gain calibration is applied
    // downstream; here we report the raw full-scale level.
    let received_level_dbfs = max_window_rms(&payload.channel_states)
        .map(|rms| 20.0 * (rms.max(1.0e-9)).log10())
        .map(|db| db as f32);
    let localization_payload = localization_manifest_payload(
        &localization,
        payload.config.localization_band_hz,
        pair_diagnostics,
        received_level_dbfs,
    );

    let has_classifier_render = render_result
        .as_ref()
        .and_then(|r| r.pending_manifest.as_ref())
        .and_then(|m| m.classifier_render.as_ref())
        .is_some();

    // Base64-encode PCM bytes for inline SSE delivery (no disk file).
    let raw_render_b64: Option<String> = render_result
        .as_ref()
        .and_then(|r| listenable_pcm_bytes.as_ref().or(r.pcm_bytes.as_ref()))
        .map(|bytes| BASE64.encode(bytes));

    // derived_handle from the render result, propagated to localization_result so Python
    // can identify the render (Bug A fix — without this Python drops the event).
    let render_derived_handle = render_result
        .as_ref()
        .and_then(|r| r.pending_manifest.as_ref())
        .and_then(|m| m.derived_handle.clone());

    // Attach localization to classifier_render manifests so the standalone
    // classifier_render event remains self-contained when localization_result does
    // not embed classifier payloads.
    if let Some(ref mut r) = render_result {
        if let Some(ref mut pending) = r.pending_manifest {
            pending.localization = Some(localization_payload.clone());
        }
    }

    // Dispatch PCM bytes to ClassificationWorker (via flume channel, no disk).
    let render_pending = if let Some(r) = render_result {
        dispatch_classification_result_standalone(
            r,
            &payload.classification_tx,
            &payload.state,
            raw_render_b64.clone(),
        )
        .await
    } else {
        None
    };

    // Memory-path manifests (seg-mem-*) are delivered exclusively via SSE.
    // Only persist to disk for disk-backed ingest paths.
    let arrived_via_channel = !source_manifest_was_persisted(&payload.manifest);

    let published = if !payload.skip_localization_result {
        let m = DspManifest {
            manifest_id: format!("manifest-{}", Uuid::new_v4()),
            manifest_type: "localization_result".to_string(),
            created_ns: payload.now_ns,
            source_handles: payload.manifest.source_handles.clone(),
            // Carry derived_handle from the render so Python's load_localized_render_manifest_bundle
            // can find the render metadata (fixes the dropped-event bug).
            derived_handle: render_derived_handle,
            localization: Some(localization_payload),
            classifier_render: None,
            birdnet: None,
            coverage_stats: render_coverage_json.or(localization_coverage_json),
            promotion_ready: false,
            env_samples: None,
            node_context: payload.manifest.node_context.clone(),
            cluster_id: payload.manifest.cluster_id.clone(),
            cluster_sensor_positions: payload.manifest.cluster_sensor_positions.clone(),
            raw_payload: None,
            raw_render_bytes: None,
            raw_audio_frame: None,
            raw_audio_bytes: None,
        };
        if !arrived_via_channel {
            // Disk-backed ingest: persist manifest (without bulky base64 data).
            let mut disk_m = m.clone();
            disk_m.raw_render_bytes = None;
            if let Err(err) = payload.manifest_store.publish(disk_m).await {
                warn!(
                    manifest_id = %payload.manifest.manifest_id,
                    error = %err,
                    "DSP worker failed to publish localization manifest"
                );
            }
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

    {
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
    }

    if let Some(ref publisher) = payload.dsp_event_publisher {
        if let Some(ref pending) = render_pending {
            let _ = publisher.publish(pending.clone()).await;
            if let Some(ref m) = published {
                let _ = publisher.publish(m.clone()).await;
            }
        } else if let Some(ref m) = published {
            let _ = publisher.publish(m.clone()).await;
        }
    } else {
        let mut st = payload.state.write().await;
        if let Some(ref pending) = render_pending {
            st.recent_results.push(pending.clone());
            if st.recent_results.len() > 50 {
                st.recent_results.remove(0);
            }
            if let Some(ref m) = published {
                st.recent_results.push(m.clone());
                if st.recent_results.len() > 50 {
                    st.recent_results.remove(0);
                }
            }
        } else if let Some(ref m) = published {
            st.recent_results.push(m.clone());
            if st.recent_results.len() > 50 {
                st.recent_results.remove(0);
            }
        }
    }
}

fn enrich_node_context_with_runtime_telemetry(payload: &mut ComputePayload) {
    let Some(node_context) = payload.manifest.node_context.as_mut() else {
        return;
    };
    let Some(node_context_map) = node_context.as_object_mut() else {
        return;
    };

    let mut audio_debug = Map::new();
    audio_debug.insert(
        "sample_rate_hz".to_string(),
        Value::from(u64::from(payload.sr)),
    );
    audio_debug.insert(
        "active_sensor_count".to_string(),
        Value::from(payload.active_channels.len() as u64),
    );
    if let Some(rms) = max_window_rms(&payload.channel_states) {
        audio_debug.insert("rms".to_string(), Value::from(rms));
    }
    node_context_map.insert("audio_debug".to_string(), Value::Object(audio_debug));

    if payload.reported_temperature_c.is_none() && payload.reported_humidity_fraction.is_none() {
        return;
    }

    let mut environment = Map::new();
    if let Some(temperature_c) = payload.reported_temperature_c {
        environment.insert("temperature_c".to_string(), Value::from(temperature_c));
    }
    if let Some(humidity_fraction) = payload.reported_humidity_fraction {
        environment.insert(
            "humidity_fraction".to_string(),
            Value::from(humidity_fraction),
        );
    }
    if let Some(source) = payload.reported_environment_source.as_ref() {
        environment.insert("source".to_string(), Value::from(source.clone()));
    } else {
        environment.insert("source".to_string(), Value::from("embedded_audio_frame"));
    }
    node_context_map.insert("environment".to_string(), Value::Object(environment));
}

fn max_window_rms(channel_states: &[crate::dsp_worker::LocalizationChannelState]) -> Option<f64> {
    channel_states
        .iter()
        .filter_map(|state| rms_for_window(&state.window))
        .max_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
}

fn rms_for_window(window: &[f32]) -> Option<f64> {
    if window.is_empty() {
        return None;
    }
    let energy = window
        .iter()
        .map(|sample| {
            let sample64 = f64::from(*sample);
            sample64 * sample64
        })
        .sum::<f64>();
    Some((energy / window.len() as f64).sqrt())
}

/// Entry point dispatched onto the Rayon pool from `process_pending`.
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
    received_level_dbfs: Option<f32>,
) -> LocalizationManifestPayload {
    LocalizationManifestPayload {
        attempted_algorithm: result.attempted_algorithm.clone(),
        resolved_algorithm: result.resolved_algorithm.clone(),
        steering_direction: (magnitude(result.steering_direction) > 1.0e-6)
            .then_some(result.steering_direction),
        position_m: result.position_m,
        position_covariance_m2: result.position_covariance_m2,
        range_observability: result.range_observability,
        range_projection_mode: result.range_projection_mode.clone(),
        confidence: result.confidence,
        residual_rms_seconds: result
            .residual_rms_seconds
            .is_finite()
            .then_some(result.residual_rms_seconds),
        sound_speed_mps: result.sound_speed_mps,
        effective_band_hz: Some(effective_band_hz),
        dominant_frequency_hz: result.dominant_frequency_hz,
        received_level_dbfs,
        pair_tdoas,
    }
}

fn magnitude(vector: [f32; 3]) -> f32 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}
