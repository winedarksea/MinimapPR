use super::*;
use crate::{
    derived_cache::{DerivedCache, DerivedCacheConfig},
    dsp::SensorStreamBuffer,
    dsp_events::{DspEventPublisher, ReplayableDspEvent},
    journal_reader::JournalPayloadHandle,
    manifests::ManifestStore,
};
use base64::engine::general_purpose::STANDARD;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use tokio::sync::RwLock;

// ---------------------------------------------------------------------------
// Item 1: per-node preprocessing chain (PreprocessStage + biquad cascade).
// ---------------------------------------------------------------------------

/// Generate a single-frequency sine for filter response checks.
fn sine_wave(frequency_hz: f64, sample_rate_hz: u32, samples: usize) -> Vec<f32> {
    let fs = f64::from(sample_rate_hz);
    (0..samples)
        .map(|i| (2.0 * std::f64::consts::PI * frequency_hz * (i as f64) / fs).sin() as f32)
        .collect()
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|s| (*s as f64) * (*s as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt() as f32
}

#[test]
fn node_audio_config_effective_stages_synthesizes_legacy_fields_when_stages_empty() {
    let cfg = NodeAudioConfig {
        gain_db: Some(6.0),
        hp_hz: Some(100.0),
        stages: vec![],
    };
    let stages = cfg.effective_stages();
    assert_eq!(stages.len(), 2);
    assert!(matches!(stages[0], PreprocessStage::Gain { db } if (db - 6.0).abs() < 1e-9));
    assert!(matches!(
        stages[1],
        PreprocessStage::Highpass { cutoff_hz, order } if (cutoff_hz - 100.0).abs() < 1e-9 && order == 2
    ));
}

#[test]
fn node_audio_config_explicit_stages_take_priority_over_legacy_fields() {
    let cfg = NodeAudioConfig {
        gain_db: Some(6.0),
        hp_hz: Some(100.0),
        stages: vec![PreprocessStage::Passthrough],
    };
    let stages = cfg.effective_stages();
    assert_eq!(stages, vec![PreprocessStage::Passthrough]);
}

#[test]
fn node_audio_config_empty_means_passthrough() {
    let cfg = NodeAudioConfig::default();
    assert!(cfg.effective_stages().is_empty());

    // Apply must leave samples untouched when stages are empty.
    let mut state = NodeAudioState::default();
    let mut channels = vec![vec![0.1_f32, -0.2, 0.3, -0.4]];
    let original = channels[0].clone();
    state.apply(&mut channels, &cfg, 16_000);
    assert_eq!(channels[0], original);
}

#[test]
fn automatic_gps_geometry_uses_reported_positions_with_stable_sensor_indices() {
    let mut geometry = AutomaticGpsGeometry::default();
    let mut first = DspManifest {
        manifest_id: "first".to_string(),
        manifest_type: "raw_journal_append".to_string(),
        created_ns: 0,
        source_handles: Vec::new(),
        derived_handle: None,
        localization: None,
        classifier_render: None,
        birdnet: None,
        coverage_stats: None,
        promotion_ready: false,
        env_samples: None,
        node_context: Some(serde_json::json!({
            "node": {
                "position_geo": {"lat": 45.0, "lon": -93.0, "alt_m": 300.0},
                "sensor_offsets_m": [[0.0, 0.0, 0.0]]
            }
        })),
        cluster_id: None,
        cluster_sensor_positions: None,
        raw_payload: None,
        raw_render_bytes: None,
        raw_audio_frame: None,
        raw_audio_bytes: None,
    };
    let first_positions = geometry
        .update_from_manifest(&first, "node-a", 1)
        .expect("first GPS geometry");
    assert_eq!(
        first_positions,
        vec![("node-a:ch0".to_string(), [0.0, 0.0, 0.0])]
    );

    first.node_context = Some(serde_json::json!({
        "node": {
            "position_geo": {"lat": 45.0, "lon": -92.9999873, "alt_m": 301.5},
            "sensor_offsets_m": [[0.0, 0.0, 0.0]]
        }
    }));
    let second_positions = geometry
        .update_from_manifest(&first, "node-b", 1)
        .expect("second GPS geometry");

    assert_eq!(second_positions.len(), 2);
    assert_eq!(second_positions[0].0, "node-a:ch0");
    assert_eq!(second_positions[1].0, "node-b:ch0");
    assert!(second_positions[1].1[0] > 0.9 && second_positions[1].1[0] < 1.1);
    assert_eq!(second_positions[1].1[2], 1.5);
}

#[test]
fn gain_stage_applies_linear_multiplier_within_tolerance() {
    let mut state = NodeAudioState::default();
    let cfg = NodeAudioConfig {
        stages: vec![PreprocessStage::Gain { db: 6.0 }],
        ..NodeAudioConfig::default()
    };
    let mut channels = vec![vec![1.0_f32, -1.0, 0.5]];
    state.apply(&mut channels, &cfg, 16_000);
    // +6 dB ≈ ×1.9953 linear.
    let expected = 10f32.powf(6.0 / 20.0);
    for (got, original) in channels[0].iter().zip([1.0_f32, -1.0, 0.5].iter()) {
        assert!((got - original * expected).abs() < 1e-4);
    }
}

#[test]
fn channel_gain_stage_applies_independent_calibration_trim() {
    let config = NodeAudioConfig {
        stages: vec![PreprocessStage::ChannelGain {
            db_by_channel: vec![6.020_599_913, -6.020_599_913],
        }],
        ..NodeAudioConfig::default()
    };
    let mut channels = vec![vec![0.1_f32, -0.1], vec![0.1_f32, -0.1]];
    let mut state = NodeAudioState::default();

    state.apply(&mut channels, &config, 16_000);

    assert!((channels[0][0] - 0.2).abs() < 1.0e-5);
    assert!((channels[1][0] - 0.05).abs() < 1.0e-5);
}

#[test]
fn lowpass_attenuates_high_frequency_more_than_low_frequency() {
    let cfg = NodeAudioConfig {
        stages: vec![PreprocessStage::Lowpass {
            cutoff_hz: 500.0,
            order: 4,
        }],
        ..NodeAudioConfig::default()
    };
    let sr = 16_000;

    // 100 Hz tone — well below the 500 Hz cutoff, should pass nearly intact.
    let mut low_state = NodeAudioState::default();
    let mut low_channels = vec![sine_wave(100.0, sr, 4_096)];
    let low_in = low_channels[0].clone();
    low_state.apply(&mut low_channels, &cfg, sr);

    // 4 kHz tone — well above the cutoff, should be heavily attenuated.
    let mut hi_state = NodeAudioState::default();
    let mut hi_channels = vec![sine_wave(4_000.0, sr, 4_096)];
    let hi_in = hi_channels[0].clone();
    hi_state.apply(&mut hi_channels, &cfg, sr);

    // Skip the first N samples to let the filter settle.
    let settle = 1_024;
    let low_attenuation = rms(&low_channels[0][settle..]) / rms(&low_in[settle..]);
    let hi_attenuation = rms(&hi_channels[0][settle..]) / rms(&hi_in[settle..]);

    assert!(
        low_attenuation > 0.9,
        "100 Hz tone unexpectedly attenuated: {low_attenuation}"
    );
    assert!(
        hi_attenuation < 0.1,
        "4 kHz tone not attenuated enough: {hi_attenuation}"
    );
}

#[test]
fn highpass_attenuates_low_frequency_more_than_high_frequency() {
    let cfg = NodeAudioConfig {
        stages: vec![PreprocessStage::Highpass {
            cutoff_hz: 500.0,
            order: 4,
        }],
        ..NodeAudioConfig::default()
    };
    let sr = 16_000;

    let mut low_state = NodeAudioState::default();
    let mut low_channels = vec![sine_wave(100.0, sr, 4_096)];
    let low_in = low_channels[0].clone();
    low_state.apply(&mut low_channels, &cfg, sr);

    let mut hi_state = NodeAudioState::default();
    let mut hi_channels = vec![sine_wave(4_000.0, sr, 4_096)];
    let hi_in = hi_channels[0].clone();
    hi_state.apply(&mut hi_channels, &cfg, sr);

    let settle = 1_024;
    let low_attenuation = rms(&low_channels[0][settle..]) / rms(&low_in[settle..]);
    let hi_attenuation = rms(&hi_channels[0][settle..]) / rms(&hi_in[settle..]);

    assert!(
        low_attenuation < 0.1,
        "100 Hz tone not attenuated enough by HPF: {low_attenuation}"
    );
    assert!(
        hi_attenuation > 0.9,
        "4 kHz tone unexpectedly attenuated by HPF: {hi_attenuation}"
    );
}

#[test]
fn cascade_preserves_state_across_frame_boundaries() {
    // The whole point of the per-stream state is that splitting a buffer in
    // two and applying the filter to each half should produce the same output
    // as applying it to the whole buffer in one shot.
    let cfg = NodeAudioConfig {
        stages: vec![PreprocessStage::Highpass {
            cutoff_hz: 200.0,
            order: 4,
        }],
        ..NodeAudioConfig::default()
    };
    let sr = 16_000;

    let mut whole_state = NodeAudioState::default();
    let mut whole = vec![sine_wave(50.0, sr, 1_024)];
    whole_state.apply(&mut whole, &cfg, sr);

    let mut split_state = NodeAudioState::default();
    let full = sine_wave(50.0, sr, 1_024);
    let mut first_half = vec![full[..512].to_vec()];
    split_state.apply(&mut first_half, &cfg, sr);
    let mut second_half = vec![full[512..].to_vec()];
    split_state.apply(&mut second_half, &cfg, sr);

    let mut stitched = first_half[0].clone();
    stitched.extend_from_slice(&second_half[0]);

    assert_eq!(stitched.len(), whole[0].len());
    for (i, (a, b)) in stitched.iter().zip(whole[0].iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "sample {i} diverged: split={a} whole={b}"
        );
    }
}

#[test]
fn config_change_recompiles_and_resets_state() {
    let mut state = NodeAudioState::default();
    let sr = 16_000;

    // First config — lowpass at 1 kHz.
    let cfg1 = NodeAudioConfig {
        stages: vec![PreprocessStage::Lowpass {
            cutoff_hz: 1_000.0,
            order: 4,
        }],
        ..NodeAudioConfig::default()
    };
    let mut channels = vec![sine_wave(100.0, sr, 256)];
    state.apply(&mut channels, &cfg1, sr);
    let sig_after_first = state.config_signature.clone();

    // Switch to highpass at 2 kHz — coefficients and state must reset.
    let cfg2 = NodeAudioConfig {
        stages: vec![PreprocessStage::Highpass {
            cutoff_hz: 2_000.0,
            order: 4,
        }],
        ..NodeAudioConfig::default()
    };
    let mut channels = vec![sine_wave(100.0, sr, 256)];
    state.apply(&mut channels, &cfg2, sr);
    assert_ne!(state.config_signature, sig_after_first);
    assert!(matches!(
        state.config_signature[0],
        PreprocessStage::Highpass { .. }
    ));
}

#[test]
fn passthrough_stage_does_not_modify_samples() {
    let mut state = NodeAudioState::default();
    let cfg = NodeAudioConfig {
        stages: vec![PreprocessStage::Passthrough],
        ..NodeAudioConfig::default()
    };
    let original = sine_wave(440.0, 16_000, 512);
    let mut channels = vec![original.clone()];
    state.apply(&mut channels, &cfg, 16_000);
    assert_eq!(channels[0], original);
}

#[test]
fn multi_stage_chain_applies_in_order() {
    // gain → HP → LP: each stage transforms the output of the previous.
    let cfg = NodeAudioConfig {
        stages: vec![
            PreprocessStage::Gain { db: -6.0 },
            PreprocessStage::Highpass {
                cutoff_hz: 100.0,
                order: 2,
            },
            PreprocessStage::Lowpass {
                cutoff_hz: 4_000.0,
                order: 2,
            },
        ],
        ..NodeAudioConfig::default()
    };
    let mut state = NodeAudioState::default();
    let mut channels = vec![sine_wave(1_000.0, 16_000, 2_048)];
    let original_rms = rms(&channels[0]);
    state.apply(&mut channels, &cfg, 16_000);
    let processed_rms = rms(&channels[0][512..]);
    // 1 kHz tone is in the passband; after -6 dB gain it should be ~0.5×.
    let attenuation = processed_rms / original_rms;
    assert!(
        (attenuation - 0.5).abs() < 0.1,
        "expected ~0.5× attenuation through gain+HP+LP at 1 kHz, got {attenuation}"
    );
}

#[test]
fn mic_positions_from_manifest_prefers_cluster_sensor_positions() {
    let manifest = DspManifest {
        manifest_id: "manifest-cluster-positions".to_string(),
        manifest_type: "raw_journal_append".to_string(),
        created_ns: 1,
        source_handles: vec![],
        derived_handle: None,
        localization: None,
        classifier_render: None,
        birdnet: None,
        coverage_stats: None,
        promotion_ready: false,
        env_samples: None,
        node_context: Some(serde_json::json!({
            "node": {
                "sensor_offsets_m": [
                    [9.0, 9.0, 9.0],
                    [8.0, 8.0, 8.0],
                ]
            }
        })),
        cluster_id: Some("square".to_string()),
        cluster_sensor_positions: Some(vec![
            ("n0:ch0".to_string(), [0.0, 0.0, 2.0]),
            ("n1:ch0".to_string(), [2.0, 0.0, 2.0]),
            ("n2:ch0".to_string(), [2.0, 2.0, 2.0]),
            ("n3:ch0".to_string(), [0.0, 2.0, 2.0]),
        ]),
        raw_payload: None,
        raw_render_bytes: None,
        raw_audio_frame: None,
        raw_audio_bytes: None,
    };

    assert_eq!(
        mic_positions_from_manifest(&manifest),
        vec![
            [0.0, 0.0, 2.0],
            [2.0, 0.0, 2.0],
            [2.0, 2.0, 2.0],
            [0.0, 2.0, 2.0],
        ]
    );
}

#[test]
fn mic_positions_from_manifest_falls_back_to_node_context_positions() {
    let manifest = DspManifest {
        manifest_id: "manifest-node-context-positions".to_string(),
        manifest_type: "raw_journal_append".to_string(),
        created_ns: 1,
        source_handles: vec![],
        derived_handle: None,
        localization: None,
        classifier_render: None,
        birdnet: None,
        coverage_stats: None,
        promotion_ready: false,
        env_samples: None,
        node_context: Some(serde_json::json!({
            "node": {
                "sensor_offsets_m": [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ]
            }
        })),
        cluster_id: Some("square".to_string()),
        cluster_sensor_positions: None,
        raw_payload: None,
        raw_render_bytes: None,
        raw_audio_frame: None,
        raw_audio_bytes: None,
    };

    assert_eq!(
        mic_positions_from_manifest(&manifest),
        vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6],]
    );
}

#[tokio::test]
async fn worker_publishes_localization_and_classifier_render_contract() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest_store = ManifestStore::new(tmp.path());
    manifest_store.ensure_initialized().await.unwrap();
    let derived_cache = DerivedCache::new(
        tmp.path(),
        DerivedCacheConfig {
            budget_bytes: 16_777_216,
            admission_reserve_bytes: 0,
        },
    );
    derived_cache.ensure_initialized().await.unwrap();

    let payload = store_forward_payload();
    let now_ns = system_now_ns();
    let raw_manifest = DspManifest {
        manifest_id: "manifest-raw-test".to_string(),
        manifest_type: "raw_journal_append".to_string(),
        created_ns: now_ns,
        source_handles: vec![JournalPayloadHandle {
            journal_epoch: 1,
            segment_id: "seg-test".to_string(),
            stream_key: "sirith-test__audio_main__abcd".to_string(),
            payload_offset_bytes: 0,
            payload_length_bytes: payload.len() as u64,
            toa_ns: None,
            tor_ns: Some(now_ns as u64),
            received_ns: None,
            sample_index_start: Some(0),
            sample_count: Some(512),
            integrity_hash: String::new(),
            segment_path: tmp.path().join("in-memory-only.bin"),
        }],
        derived_handle: None,
        localization: None,
        classifier_render: None,
        birdnet: None,
        coverage_stats: None,
        promotion_ready: false,
        env_samples: None,
        node_context: None,
        cluster_id: None,
        cluster_sensor_positions: None,
        raw_payload: Some(payload.into_bytes()),
        raw_render_bytes: None,
        raw_audio_frame: None,
        raw_audio_bytes: None,
    };
    let state: SharedDspState = Arc::new(RwLock::new(Default::default()));
    let event_publisher = DspEventPublisher::new(state.clone(), 16, 16, 50);
    let mut dsp_result_rx = event_publisher.subscribe();
    let mut worker = DspWorker::new(
        manifest_store.clone(),
        derived_cache,
        DspWorkerConfig {
            ..DspWorkerConfig::default()
        },
        state.clone(),
    )
    .with_dsp_event_publisher(event_publisher);

    worker.process_one(raw_manifest, 1).await;

    // Collect SSE events broadcast by run_io (no disk writes for memory-path manifests).
    let events = drain_published_manifests(&mut dsp_result_rx);

    let localization_event = events
        .iter()
        .find(|m| m.manifest_type == "localization_result")
        .expect("localization_result SSE event");
    let raw_audio_event = events
        .iter()
        .find(|m| m.manifest_type == "raw_audio_frame")
        .expect("raw_audio_frame SSE event");
    assert_eq!(raw_audio_event.manifest_id, "raw-audio-manifest-raw-test");
    let render_event = events
        .iter()
        .find(|m| m.manifest_type == "classifier_render")
        .expect("classifier_render SSE event");

    let localization = localization_event
        .localization
        .as_ref()
        .expect("localization payload");
    assert_eq!(localization.attempted_algorithm, "srp_phat");
    assert_eq!(localization.resolved_algorithm, "srp_phat");
    assert_eq!(localization.pair_tdoas.len(), 6);
    // The reference-channel dominant frequency is computed and carried through the
    // serialized manifest so the Python fusion node can scale lateral covariance.
    let dominant_hz = localization
        .dominant_frequency_hz
        .expect("dominant_frequency_hz present in localization payload");
    assert!(
        dominant_hz > 0.0,
        "expected a positive dominant frequency, got {dominant_hz}"
    );

    // localization_result remains heartbeat/localization metadata only.
    assert!(localization_event.derived_handle.is_some());
    assert!(localization_event.raw_render_bytes.is_none());
    assert!(localization_event.classifier_render.is_none());

    let raw_audio_frame = raw_audio_event
        .raw_audio_frame
        .as_ref()
        .expect("raw audio frame payload");
    assert_eq!(raw_audio_frame["sample_format"], "pcm16le");
    assert_eq!(
        raw_audio_frame["stream_key"],
        "sirith-test__audio_main__abcd"
    );
    assert_eq!(raw_audio_frame["channel_count"], 4);
    assert!(raw_audio_frame["sample_count"].as_u64().unwrap() > 0);
    assert!(raw_audio_event.raw_audio_bytes.is_some());
    assert!(raw_audio_event.raw_render_bytes.is_none());

    let render_payload = render_event
        .classifier_render
        .as_ref()
        .expect("render payload");
    assert_eq!(render_payload.sample_format, "pcm16le");
    assert!(render_payload.render_start_ns.is_some());
    assert!(render_payload.render_end_ns.is_some());
    assert!(render_payload.render_start_ns < render_payload.render_end_ns);
    // Inline PCM delivered — segment_path is a sentinel (no disk file written).
    assert!(render_event.raw_render_bytes.is_some());
    assert!(!render_event
        .derived_handle
        .as_ref()
        .unwrap()
        .segment_path
        .exists());

    let state = state.read().await;
    assert_eq!(state.total_localization_attempts, 1);
    assert_eq!(state.total_classification_attempts, 1);
    assert_eq!(state.total_tdoa_results, 1);
    assert_eq!(state.total_localization_results, 1);
    assert_eq!(state.total_classifier_renders, 1);
}

#[tokio::test]
async fn worker_splits_gapped_batch_before_raw_publication_and_buffering() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest_store = ManifestStore::new(tmp.path());
    manifest_store.ensure_initialized().await.unwrap();
    let derived_cache = DerivedCache::new(
        tmp.path(),
        DerivedCacheConfig {
            budget_bytes: 16_777_216,
            admission_reserve_bytes: 0,
        },
    );
    derived_cache.ensure_initialized().await.unwrap();

    let state: SharedDspState = Arc::new(RwLock::new(Default::default()));
    let event_publisher = DspEventPublisher::new(state.clone(), 32, 32, 50);
    let mut dsp_result_rx = event_publisher.subscribe();
    let mut worker = DspWorker::new(
        manifest_store,
        derived_cache,
        DspWorkerConfig {
            max_buffer_seconds: 2.0,
            max_trusted_node_clock_skew_seconds: f64::MAX,
            ..DspWorkerConfig::default()
        },
        state,
    )
    .with_dsp_event_publisher(event_publisher);

    let manifest = raw_manifest_for_payload(
        tmp.path(),
        "manifest-gapped-batch",
        "segment-gapped-batch",
        store_forward_payload_with_internal_gap(),
    )
    .await;
    worker.process_one(manifest, 1).await;

    let events = drain_published_manifests(&mut dsp_result_rx);
    let raw_events = events
        .iter()
        .filter(|manifest| manifest.manifest_type == "raw_audio_frame")
        .collect::<Vec<_>>();
    assert_eq!(raw_events.len(), 2);
    assert_eq!(
        raw_events[0].manifest_id,
        "raw-audio-manifest-gapped-batch-segment-0"
    );
    assert_eq!(
        raw_events[1].manifest_id,
        "raw-audio-manifest-gapped-batch-segment-1"
    );
    assert_eq!(
        raw_events[0].raw_audio_frame.as_ref().unwrap()["start_sample_index"],
        0
    );
    assert_eq!(
        raw_events[0].raw_audio_frame.as_ref().unwrap()["end_sample_index"],
        512
    );
    assert_eq!(
        raw_events[1].raw_audio_frame.as_ref().unwrap()["start_sample_index"],
        1024
    );
    assert_eq!(
        raw_events[1].raw_audio_frame.as_ref().unwrap()["end_sample_index"],
        1536
    );
    assert!(raw_events.iter().all(|event| {
        event.raw_audio_frame.as_ref().unwrap()["sample_count"] == 512
            && event.raw_audio_frame.as_ref().unwrap()["source_manifest_id"]
                == "manifest-gapped-batch"
    }));

    let first_channel_buffer = &worker.buffers["sirith-test__audio_main__abcd"][0];
    let coverage = first_channel_buffer
        .coverage_ending_at(1_096_000_000, 0.096)
        .expect("coverage through both segments");
    assert_eq!(coverage.sample_count, 1536);
    assert_eq!(coverage.covered_samples, 1024);
    assert_eq!(coverage.missing_samples, 512);
    assert_eq!(coverage.max_gap_samples, 512);
}

#[tokio::test]
async fn worker_continues_localization_across_packet_gap_with_misaligned_toa() {
    // The previous shape of this test (worker_publishes_omni_render_when_localization_coverage_is_unavailable)
    // intentionally sent a backward-jumping TOA to force coverage_centered_at to
    // miss the buffer tail. That only "worked" because resolve_buffer_end_time_ns
    // was riding the publish-time TOA — the same TOA-trust that broke
    // GPS-locked tetrahedral arrays in production. Now that windowing follows
    // the sample-index timeline, a 512-sample gap is recovered and the second
    // frame's centered window lands on real samples, producing a real SRP
    // localization rather than the coverage_unavailable fallback.
    let tmp = tempfile::tempdir().unwrap();
    let manifest_store = ManifestStore::new(tmp.path());
    manifest_store.ensure_initialized().await.unwrap();
    let derived_cache = DerivedCache::new(
        tmp.path(),
        DerivedCacheConfig {
            budget_bytes: 16_777_216,
            admission_reserve_bytes: 0,
        },
    );
    derived_cache.ensure_initialized().await.unwrap();

    let state: SharedDspState = Arc::new(RwLock::new(Default::default()));
    let event_publisher = DspEventPublisher::new(state.clone(), 32, 32, 50);
    let mut dsp_result_rx = event_publisher.subscribe();
    let mut worker = DspWorker::new(
        manifest_store.clone(),
        derived_cache,
        DspWorkerConfig {
            classifier_render_min_interval_seconds: 0.0,
            max_buffer_seconds: 32.0,
            max_trusted_node_clock_skew_seconds: f64::MAX,
            ..DspWorkerConfig::default()
        },
        state.clone(),
    )
    .with_dsp_event_publisher(event_publisher);

    let first_payload = store_forward_payload_with_timing(1_000_000_000, 0, 1);
    let first_manifest =
        raw_manifest_for_payload(tmp.path(), "manifest-raw-first", "seg-first", first_payload)
            .await;
    worker.process_one(first_manifest, 2).await;

    let second_payload = store_forward_payload_with_timing(100_000_000, 1024, 2);
    let second_manifest = raw_manifest_for_payload(
        tmp.path(),
        "manifest-raw-second",
        "seg-second",
        second_payload,
    )
    .await;
    worker.process_one(second_manifest, 2).await;

    let events = drain_published_manifests(&mut dsp_result_rx);

    let localizations: Vec<_> = events
        .iter()
        .filter(|m| m.manifest_type == "localization_result")
        .collect();
    let renders: Vec<_> = events
        .iter()
        .filter(|m| m.manifest_type == "classifier_render")
        .collect();
    assert_eq!(localizations.len(), 2);
    assert_eq!(renders.len(), 2);

    // Neither render should land in the empty-PCM / coverage-unavailable branch.
    assert!(renders.iter().all(|render| {
        render
            .classifier_render
            .as_ref()
            .and_then(|payload| payload.fallback_reason.as_deref())
            != Some("localization_coverage_unavailable")
    }));
    assert!(renders.iter().all(|render| {
        render
            .classifier_render
            .as_ref()
            .map(|payload| payload.sample_count > 0)
            .unwrap_or(false)
    }));

    let state = state.read().await;
    assert_eq!(state.total_localization_results, 2);
    assert_eq!(state.total_classifier_renders, 2);
}

#[tokio::test]
async fn localization_continues_when_classifier_render_is_rate_limited() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest_store = ManifestStore::new(tmp.path());
    manifest_store.ensure_initialized().await.unwrap();
    let derived_cache = DerivedCache::new(
        tmp.path(),
        DerivedCacheConfig {
            budget_bytes: 16_777_216,
            admission_reserve_bytes: 0,
        },
    );
    derived_cache.ensure_initialized().await.unwrap();

    let state: SharedDspState = Arc::new(RwLock::new(Default::default()));
    let event_publisher = DspEventPublisher::new(state.clone(), 32, 32, 50);
    let mut dsp_result_rx = event_publisher.subscribe();
    let mut worker = DspWorker::new(
        manifest_store.clone(),
        derived_cache,
        DspWorkerConfig {
            classifier_render_min_interval_seconds: 999.0,
            localization_cadence_ms: 0,
            trigger_cooldown_seconds: 0.0,
            ..DspWorkerConfig::default()
        },
        state.clone(),
    )
    .with_dsp_event_publisher(event_publisher);

    let first_payload = store_forward_payload_with_timing(1_000_000_000, 0, 1);
    let first_manifest = raw_manifest_for_payload(
        tmp.path(),
        "manifest-raw-cadence-1",
        "seg-cadence-1",
        first_payload,
    )
    .await;
    worker.process_one(first_manifest, 1).await;

    let second_payload = store_forward_payload_with_timing(1_032_000_000, 512, 2);
    let second_manifest = raw_manifest_for_payload(
        tmp.path(),
        "manifest-raw-cadence-2",
        "seg-cadence-2",
        second_payload,
    )
    .await;
    worker.process_one(second_manifest, 1).await;

    let events = drain_published_manifests(&mut dsp_result_rx);

    let localization_count = events
        .iter()
        .filter(|m| m.manifest_type == "localization_result")
        .count();
    let render_count = events
        .iter()
        .filter(|m| m.manifest_type == "classifier_render")
        .count();

    // First frame publishes both; second still localizes even when render cadence suppresses.
    assert_eq!(localization_count, 2);
    assert_eq!(render_count, 1);

    let state = state.read().await;
    assert_eq!(state.total_localization_results, 2);
    assert_eq!(state.total_classifier_renders, 1);
}

#[tokio::test]
async fn tetra_classifier_render_forces_srp_between_localization_cadence_ticks() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest_store = ManifestStore::new(tmp.path());
    manifest_store.ensure_initialized().await.unwrap();
    let derived_cache = DerivedCache::new(
        tmp.path(),
        DerivedCacheConfig {
            budget_bytes: 16_777_216,
            admission_reserve_bytes: 0,
        },
    );
    derived_cache.ensure_initialized().await.unwrap();

    let state: SharedDspState = Arc::new(RwLock::new(Default::default()));
    let event_publisher = DspEventPublisher::new(state.clone(), 32, 32, 50);
    let mut dsp_result_rx = event_publisher.subscribe();
    let mut worker = DspWorker::new(
        manifest_store.clone(),
        derived_cache,
        DspWorkerConfig {
            classifier_render_min_interval_seconds: 0.0,
            localization_cadence_ms: 60_000,
            trigger_cooldown_seconds: 0.0,
            ..DspWorkerConfig::default()
        },
        state.clone(),
    )
    .with_dsp_event_publisher(event_publisher);

    let first_payload = store_forward_payload_with_timing(1_000_000_000, 0, 1);
    let first_manifest = raw_manifest_for_payload(
        tmp.path(),
        "manifest-raw-render-srp-1",
        "seg-render-srp-1",
        first_payload,
    )
    .await;
    worker.process_one(first_manifest, 1).await;

    let second_payload = store_forward_payload_with_timing(1_032_000_000, 512, 2);
    let second_manifest = raw_manifest_for_payload(
        tmp.path(),
        "manifest-raw-render-srp-2",
        "seg-render-srp-2",
        second_payload,
    )
    .await;
    worker.process_one(second_manifest, 1).await;

    let events = drain_published_manifests(&mut dsp_result_rx);
    let localization_events: Vec<_> = events
        .iter()
        .filter(|manifest| manifest.manifest_type == "localization_result")
        .collect();
    let render_events: Vec<_> = events
        .iter()
        .filter(|manifest| manifest.manifest_type == "classifier_render")
        .collect();

    assert_eq!(localization_events.len(), 2);
    assert_eq!(render_events.len(), 2);
    assert!(localization_events.iter().all(|manifest| {
        manifest
            .localization
            .as_ref()
            .is_some_and(|payload| payload.resolved_algorithm == "srp_phat")
    }));
    assert!(render_events.iter().all(|manifest| {
        manifest
            .classifier_render
            .as_ref()
            .is_some_and(|payload| payload.render_kind == "birdnet_band_split_das")
    }));
    assert!(render_events.iter().all(|manifest| {
        manifest.classifier_render.as_ref().is_some_and(|payload| {
            payload.render_start_ns.is_some()
                && payload.render_end_ns.is_some()
                && payload.render_start_ns < payload.render_end_ns
        })
    }));
}

#[tokio::test]
async fn small_packet_timestamp_jitter_does_not_drop_array_window_coverage() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest_store = ManifestStore::new(tmp.path());
    manifest_store.ensure_initialized().await.unwrap();
    let derived_cache = DerivedCache::new(
        tmp.path(),
        DerivedCacheConfig {
            budget_bytes: 16_777_216,
            admission_reserve_bytes: 0,
        },
    );
    derived_cache.ensure_initialized().await.unwrap();

    let state: SharedDspState = Arc::new(RwLock::new(Default::default()));
    let event_publisher = DspEventPublisher::new(state.clone(), 32, 32, 50);
    let mut dsp_result_rx = event_publisher.subscribe();
    let mut worker = DspWorker::new(
        manifest_store.clone(),
        derived_cache,
        DspWorkerConfig {
            classifier_render_min_interval_seconds: 0.0,
            localization_cadence_ms: 0,
            trigger_cooldown_seconds: 0.0,
            max_buffer_seconds: 32.0,
            max_trusted_node_clock_skew_seconds: f64::MAX,
            ..DspWorkerConfig::default()
        },
        state.clone(),
    )
    .with_dsp_event_publisher(event_publisher);

    let first_payload = store_forward_payload_with_timing(1_000_000_000, 0, 1);
    let first_manifest = raw_manifest_for_payload(
        tmp.path(),
        "manifest-raw-jitter-1",
        "seg-jitter-1",
        first_payload,
    )
    .await;
    worker.process_one(first_manifest, 1).await;

    let second_payload = store_forward_payload_with_timing_jitter(1_032_500_000, 512, 2);
    let second_manifest = raw_manifest_for_payload(
        tmp.path(),
        "manifest-raw-jitter-2",
        "seg-jitter-2",
        second_payload,
    )
    .await;
    worker.process_one(second_manifest, 1).await;

    let events = drain_published_manifests(&mut dsp_result_rx);
    let localization_events: Vec<_> = events
        .iter()
        .filter(|manifest| manifest.manifest_type == "localization_result")
        .collect();
    let render_events: Vec<_> = events
        .iter()
        .filter(|manifest| manifest.manifest_type == "classifier_render")
        .collect();

    assert_eq!(localization_events.len(), 2);
    assert_eq!(render_events.len(), 2);
    assert!(localization_events.iter().all(|manifest| {
        manifest
            .localization
            .as_ref()
            .is_some_and(|payload| payload.resolved_algorithm == "srp_phat")
    }));
    assert!(render_events.iter().all(|manifest| {
        manifest
            .classifier_render
            .as_ref()
            .and_then(|payload| payload.fallback_reason.as_deref())
            != Some("localization_coverage_unavailable")
    }));
}

#[tokio::test]
async fn gps_clock_correction_jitter_does_not_drop_classifier_render() {
    // Reproduces the live sirith-tetra-1a15 (4-mic, GPS-locked) failure mode:
    // NodeRunner re-samples the GPS-corrected UTC at every publish, so consecutive
    // packets carry start_time_ns deltas that don't match the sample-rate delta,
    // even though start_sample_index advances perfectly. Tens of milliseconds of
    // drift caused the previous 1 ms snap tolerance to fall back to TOA, which
    // pushed the windowing time past the buffer tail; coverage_centered_at and
    // window_ending_at returned None, the render produced empty PCM, and the
    // classifier_render manifest was silently dropped in write_render_to_cache.
    let tmp = tempfile::tempdir().unwrap();
    let manifest_store = ManifestStore::new(tmp.path());
    manifest_store.ensure_initialized().await.unwrap();
    let derived_cache = DerivedCache::new(
        tmp.path(),
        DerivedCacheConfig {
            budget_bytes: 16_777_216,
            admission_reserve_bytes: 0,
        },
    );
    derived_cache.ensure_initialized().await.unwrap();

    let state: SharedDspState = Arc::new(RwLock::new(Default::default()));
    let event_publisher = DspEventPublisher::new(state.clone(), 64, 64, 50);
    let mut dsp_result_rx = event_publisher.subscribe();
    let mut worker = DspWorker::new(
        manifest_store.clone(),
        derived_cache,
        DspWorkerConfig {
            classifier_render_min_interval_seconds: 0.0,
            localization_cadence_ms: 0,
            trigger_cooldown_seconds: 0.0,
            max_buffer_seconds: 32.0,
            max_trusted_node_clock_skew_seconds: f64::MAX,
            ..DspWorkerConfig::default()
        },
        state.clone(),
    )
    .with_dsp_event_publisher(event_publisher);

    // 16 packets of 512 samples each = 32 ms cadence on the sample timeline,
    // but each packet's TOA wanders by ±50 ms relative to the sample-rate delta
    // (representative of GPS PPS clock-correction shifts between publishes).
    let toa_jitter_pattern: [i64; 16] = [
        0,
        50_000_000,
        -30_000_000,
        40_000_000,
        -20_000_000,
        55_000_000,
        -45_000_000,
        35_000_000,
        25_000_000,
        -15_000_000,
        60_000_000,
        -25_000_000,
        45_000_000,
        -35_000_000,
        20_000_000,
        -10_000_000,
    ];
    let nominal_period_ns: u64 = 32_000_000;
    let mut start_time_ns: i64 = 1_000_000_000;
    let mut start_sample_index: u64 = 0;
    for (sequence, jitter_ns) in toa_jitter_pattern.iter().copied().enumerate() {
        let toa = (start_time_ns + jitter_ns).max(1) as u64;
        let payload =
            store_forward_payload_with_timing_jitter(toa, start_sample_index, sequence as u64 + 1);
        let manifest = raw_manifest_for_payload(
            tmp.path(),
            &format!("manifest-gps-jitter-{sequence}"),
            &format!("seg-gps-jitter-{sequence}"),
            payload,
        )
        .await;
        worker.process_one(manifest, 1).await;
        start_time_ns += nominal_period_ns as i64;
        start_sample_index += 512;
    }

    let events = drain_published_manifests(&mut dsp_result_rx);
    let render_events: Vec<_> = events
        .iter()
        .filter(|manifest| manifest.manifest_type == "classifier_render")
        .collect();
    assert!(
        !render_events.is_empty(),
        "GPS clock-correction jitter must not suppress classifier_render manifests"
    );
    let localization_events: Vec<_> = events
        .iter()
        .filter(|manifest| manifest.manifest_type == "localization_result")
        .collect();
    assert!(!localization_events.is_empty());

    // Once the buffer holds a localization-window's worth of samples, every
    // subsequent localization_result must resolve to srp_phat — a None coverage
    // would have downgraded resolved_algorithm to "localization_cadence_skipped".
    assert!(
        localization_events.iter().skip(2).all(|manifest| manifest
            .localization
            .as_ref()
            .is_some_and(|payload| payload.resolved_algorithm == "srp_phat")),
        "GPS-jittered packets should still produce srp_phat localizations"
    );

    // No render should land in the empty-PCM branch (which would have been
    // silently dropped by write_render_to_cache).
    assert!(
        render_events.iter().all(|manifest| {
            manifest
                .classifier_render
                .as_ref()
                .map(|payload| payload.sample_count > 0)
                .unwrap_or(false)
        }),
        "GPS-jittered renders must contain non-empty PCM"
    );
}

#[tokio::test]
async fn packet_timestamp_lead_does_not_force_localization_coverage_unavailable() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest_store = ManifestStore::new(tmp.path());
    manifest_store.ensure_initialized().await.unwrap();
    let derived_cache = DerivedCache::new(
        tmp.path(),
        DerivedCacheConfig {
            budget_bytes: 16_777_216,
            admission_reserve_bytes: 0,
        },
    );
    derived_cache.ensure_initialized().await.unwrap();

    let state: SharedDspState = Arc::new(RwLock::new(Default::default()));
    let event_publisher = DspEventPublisher::new(state.clone(), 64, 64, 50);
    let mut dsp_result_rx = event_publisher.subscribe();
    let mut worker = DspWorker::new(
        manifest_store.clone(),
        derived_cache,
        DspWorkerConfig {
            classifier_render_min_interval_seconds: 0.0,
            localization_cadence_ms: 0,
            trigger_cooldown_seconds: 0.0,
            max_buffer_seconds: 32.0,
            max_trusted_node_clock_skew_seconds: f64::MAX,
            ..DspWorkerConfig::default()
        },
        state.clone(),
    )
    .with_dsp_event_publisher(event_publisher);

    let first_payload = store_forward_payload_with_timing(1_000_000_000, 0, 1);
    let first_manifest = raw_manifest_for_payload(
        tmp.path(),
        "manifest-positive-lead-1",
        "seg-positive-lead-1",
        first_payload,
    )
    .await;
    worker.process_one(first_manifest, 1).await;

    let second_payload = store_forward_payload_with_timing_jitter(2_452_000_000, 512, 2);
    let second_manifest = raw_manifest_for_payload(
        tmp.path(),
        "manifest-positive-lead-2",
        "seg-positive-lead-2",
        second_payload,
    )
    .await;
    worker.process_one(second_manifest, 1).await;

    let events = drain_published_manifests(&mut dsp_result_rx);
    let localization_events: Vec<_> = events
        .iter()
        .filter(|manifest| manifest.manifest_type == "localization_result")
        .collect();
    let render_events: Vec<_> = events
        .iter()
        .filter(|manifest| manifest.manifest_type == "classifier_render")
        .collect();

    assert_eq!(localization_events.len(), 2);
    assert_eq!(render_events.len(), 2);
    assert!(render_events.iter().all(|manifest| {
        manifest
            .classifier_render
            .as_ref()
            .and_then(|payload| payload.fallback_reason.as_deref())
            != Some("localization_coverage_unavailable")
    }));
}

#[tokio::test]
async fn worker_publishes_omni_render_for_non_tetrahedral_channel_count() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest_store = ManifestStore::new(tmp.path());
    manifest_store.ensure_initialized().await.unwrap();
    let derived_cache = DerivedCache::new(
        tmp.path(),
        DerivedCacheConfig {
            budget_bytes: 16_777_216,
            admission_reserve_bytes: 0,
        },
    );
    derived_cache.ensure_initialized().await.unwrap();

    let state: SharedDspState = Arc::new(RwLock::new(Default::default()));
    let event_publisher = DspEventPublisher::new(state.clone(), 32, 32, 50);
    let mut dsp_result_rx = event_publisher.subscribe();
    let mut worker = DspWorker::new(
        manifest_store.clone(),
        derived_cache,
        DspWorkerConfig {
            ..DspWorkerConfig::default()
        },
        state.clone(),
    )
    .with_dsp_event_publisher(event_publisher);

    let payload = store_forward_payload_with_channel_count(1_000_000_000, 0, 1, 5);
    let manifest = raw_manifest_for_payload(
        tmp.path(),
        "manifest-raw-five-channel",
        "seg-five-channel",
        payload,
    )
    .await;

    worker.process_one(manifest, 1).await;

    let events = drain_published_manifests(&mut dsp_result_rx);
    let localization_count = events
        .iter()
        .filter(|m| m.manifest_type == "localization_result")
        .count();
    let render = events
        .iter()
        .find(|m| m.manifest_type == "classifier_render")
        .expect("classifier render event");

    assert_eq!(localization_count, 0);
    assert_eq!(
        render
            .classifier_render
            .as_ref()
            .and_then(|payload| payload.fallback_reason.as_deref()),
        Some("non_tetrahedral_array_geometry_unusable")
    );

    let state = state.read().await;
    assert_eq!(state.total_localization_attempts, 0);
    assert_eq!(state.total_classification_attempts, 1);
    assert_eq!(state.total_localization_results, 0);
    assert_eq!(state.total_classifier_renders, 1);
}

#[tokio::test]
async fn worker_dispatches_five_channel_planar_array_through_spatial_pipeline_not_omni() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest_store = ManifestStore::new(tmp.path());
    manifest_store.ensure_initialized().await.unwrap();
    let derived_cache = DerivedCache::new(
        tmp.path(),
        DerivedCacheConfig {
            budget_bytes: 16_777_216,
            admission_reserve_bytes: 0,
        },
    );
    derived_cache.ensure_initialized().await.unwrap();

    let state: SharedDspState = Arc::new(RwLock::new(Default::default()));
    let event_publisher = DspEventPublisher::new(state.clone(), 32, 32, 50);
    let mut dsp_result_rx = event_publisher.subscribe();
    let mut worker = DspWorker::new(
        manifest_store.clone(),
        derived_cache,
        DspWorkerConfig {
            ..DspWorkerConfig::default()
        },
        state.clone(),
    )
    .with_dsp_event_publisher(event_publisher);

    let r = 0.025 * std::f32::consts::FRAC_1_SQRT_2;
    let payload = store_forward_payload_with_planar_five_channel(Some("upper"));
    let mut manifest = raw_manifest_for_payload(
        tmp.path(),
        "manifest-raw-planar-five-channel",
        "seg-planar-five-channel",
        payload,
    )
    .await;
    manifest.node_context = Some(serde_json::json!({
        "node": {
            "id": "sirith-planar-test",
            "half_space": "upper",
            "sensor_offsets_m": [
                [r, r, 0.0],
                [-r, r, 0.0],
                [-r, -r, 0.0],
                [r, -r, 0.0],
                [0.0, 0.0, 0.0],
            ]
        }
    }));

    worker.process_one(manifest, 1).await;

    let events = drain_published_manifests(&mut dsp_result_rx);
    let localization_count = events
        .iter()
        .filter(|m| m.manifest_type == "localization_result")
        .count();
    let render = events
        .iter()
        .find(|m| m.manifest_type == "classifier_render")
        .expect("classifier render event");

    // Valid non-collinear 5-mic geometry must NOT hit the large-array omni
    // fallback gate (dsp_worker.rs's `array_spans_at_least_2d` check) even
    // though channel_count > 4. The synthetic pseudo-random test audio is not
    // a coherent point source, so a low-confidence fallback is expected —
    // what matters here is that it's *not* the geometry/channel-cap gate.
    let fallback_reason = render
        .classifier_render
        .as_ref()
        .and_then(|payload| payload.fallback_reason.as_deref());
    assert_ne!(
        fallback_reason,
        Some("non_tetrahedral_array_geometry_unusable"),
        "5-ch planar array with valid, non-collinear geometry incorrectly hit the geometry gate"
    );
    assert_ne!(
        fallback_reason,
        Some("array_exceeds_max_spatial_channels"),
        "5-ch planar array incorrectly hit the channel-cap gate"
    );
    assert_eq!(localization_count, 1);

    let state = state.read().await;
    assert_eq!(state.total_localization_attempts, 1);
}

#[tokio::test]
async fn clustered_single_channel_manifests_share_a_tetrahedral_localization_buffer() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest_store = ManifestStore::new(tmp.path());
    manifest_store.ensure_initialized().await.unwrap();
    let derived_cache = DerivedCache::new(
        tmp.path(),
        DerivedCacheConfig {
            budget_bytes: 16_777_216,
            admission_reserve_bytes: 0,
        },
    );
    derived_cache.ensure_initialized().await.unwrap();

    let state: SharedDspState = Arc::new(RwLock::new(Default::default()));
    let event_publisher = DspEventPublisher::new(state.clone(), 64, 64, 50);
    let mut dsp_result_rx = event_publisher.subscribe();
    let mut worker = DspWorker::new(
        manifest_store.clone(),
        derived_cache,
        DspWorkerConfig {
            classifier_render_min_interval_seconds: 0.0,
            localization_cadence_ms: 0,
            trigger_cooldown_seconds: 0.0,
            localization_rms_gate: 0.0,
            ..DspWorkerConfig::default()
        },
        state.clone(),
    )
    .with_dsp_event_publisher(event_publisher);

    let cluster_sensor_positions = vec![
        ("node-a:ch0".to_string(), [0.0, 0.0, 2.0]),
        ("node-b:ch0".to_string(), [2.0, 0.0, 2.0]),
        ("node-c:ch0".to_string(), [2.0, 2.0, 2.0]),
        ("node-d:ch0".to_string(), [0.0, 2.0, 2.0]),
    ];
    for (index, node_id) in ["node-a", "node-b", "node-c", "node-d"].iter().enumerate() {
        let payload =
            store_forward_payload_with_channel_count(1_000_000_000, 0, index as u64 + 1, 1);
        let manifest = raw_manifest_for_payload_with_metadata(
            tmp.path(),
            &format!("manifest-clustered-{index}"),
            &format!("seg-clustered-{index}"),
            &format!("{node_id}__audio_main__{index:04x}"),
            payload,
            Some("cluster-square".to_string()),
            Some(cluster_sensor_positions.clone()),
        )
        .await;
        worker.process_one(manifest, 1).await;
    }

    let events = drain_published_manifests(&mut dsp_result_rx);
    let clustered_localizations: Vec<_> = events
        .iter()
        .filter(|manifest| {
            manifest.manifest_type == "localization_result"
                && manifest.cluster_id.as_deref() == Some("cluster-square")
                && manifest
                    .localization
                    .as_ref()
                    .is_some_and(|payload| payload.resolved_algorithm == "srp_phat")
        })
        .collect();

    assert!(
        !clustered_localizations.is_empty(),
        "expected a real SRP localization from four clustered single-channel manifests"
    );
    let localization_manifest = clustered_localizations
        .last()
        .expect("clustered localization manifest");
    let localization = localization_manifest
        .localization
        .as_ref()
        .expect("clustered localization payload");
    assert_eq!(localization.pair_tdoas.len(), 6);
    assert_eq!(
        localization_manifest
            .cluster_sensor_positions
            .as_ref()
            .map(Vec::len),
        Some(4)
    );

    let state = state.read().await;
    assert!(state.total_localization_results >= 1);
}

#[tokio::test]
async fn automatic_gps_single_channel_manifests_share_a_localization_buffer() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest_store = ManifestStore::new(tmp.path());
    manifest_store.ensure_initialized().await.unwrap();
    let derived_cache = DerivedCache::new(
        tmp.path(),
        DerivedCacheConfig {
            budget_bytes: 16_777_216,
            admission_reserve_bytes: 0,
        },
    );
    derived_cache.ensure_initialized().await.unwrap();

    let state: SharedDspState = Arc::new(RwLock::new(Default::default()));
    let event_publisher = DspEventPublisher::new(state.clone(), 64, 64, 50);
    let mut dsp_result_rx = event_publisher.subscribe();
    let mut worker = DspWorker::new(
        manifest_store.clone(),
        derived_cache,
        DspWorkerConfig {
            classifier_render_min_interval_seconds: 0.0,
            localization_cadence_ms: 0,
            trigger_cooldown_seconds: 0.0,
            localization_rms_gate: 0.0,
            ..DspWorkerConfig::default()
        },
        state,
    )
    .with_dsp_event_publisher(event_publisher);

    // Approximately a two-metre square at 45°N. The manifests intentionally
    // omit every cluster field: GPS reports alone must create the shared route.
    for (index, (node_id, lat, lon)) in [
        ("gps-a", 45.0, -93.0),
        ("gps-b", 45.0, -92.9999746),
        ("gps-c", 45.0000180, -92.9999746),
        ("gps-d", 45.0000180, -93.0),
    ]
    .iter()
    .enumerate()
    {
        let payload =
            store_forward_payload_with_channel_count(1_000_000_000, 0, index as u64 + 1, 1);
        let mut manifest = raw_manifest_for_payload_with_metadata(
            tmp.path(),
            &format!("manifest-auto-gps-{index}"),
            &format!("seg-auto-gps-{index}"),
            &format!("{node_id}__audio_main__{index:04x}"),
            payload,
            None,
            None,
        )
        .await;
        manifest.node_context = Some(serde_json::json!({
            "node": {
                "id": node_id,
                "position_geo": {"lat": lat, "lon": lon, "alt_m": 2.0},
                "sensor_offsets_m": [[0.0, 0.0, 0.0]]
            }
        }));
        worker.process_one(manifest, 1).await;
    }

    let events = drain_published_manifests(&mut dsp_result_rx);
    let localization = events
        .iter()
        .rev()
        .find(|manifest| {
            manifest.manifest_type == "localization_result"
                && manifest.cluster_id.as_deref() == Some(AUTOMATIC_GPS_CLUSTER_ID)
                && manifest
                    .localization
                    .as_ref()
                    .is_some_and(|payload| payload.resolved_algorithm == "srp_phat")
        })
        .expect("automatic GPS geometry must produce a cross-node SRP result");
    assert_eq!(
        localization.cluster_sensor_positions.as_ref().map(Vec::len),
        Some(4)
    );
    assert_eq!(
        localization
            .localization
            .as_ref()
            .expect("localization payload")
            .pair_tdoas
            .len(),
        6
    );
}

#[tokio::test]
async fn mixed_clustered_tetrahedral_and_point_manifests_share_one_localization_buffer() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest_store = ManifestStore::new(tmp.path());
    manifest_store.ensure_initialized().await.unwrap();
    let derived_cache = DerivedCache::new(
        tmp.path(),
        DerivedCacheConfig {
            budget_bytes: 16_777_216,
            admission_reserve_bytes: 0,
        },
    );
    derived_cache.ensure_initialized().await.unwrap();

    let state: SharedDspState = Arc::new(RwLock::new(Default::default()));
    let event_publisher = DspEventPublisher::new(state.clone(), 64, 64, 50);
    let mut dsp_result_rx = event_publisher.subscribe();
    let mut worker = DspWorker::new(
        manifest_store.clone(),
        derived_cache,
        DspWorkerConfig {
            classifier_render_min_interval_seconds: 0.0,
            localization_cadence_ms: 0,
            trigger_cooldown_seconds: 0.0,
            localization_rms_gate: 0.0,
            ..DspWorkerConfig::default()
        },
        state.clone(),
    )
    .with_dsp_event_publisher(event_publisher);

    let cluster_sensor_positions = vec![
        ("tetra-node:ch0".to_string(), [0.0, 0.0, 2.0]),
        ("tetra-node:ch1".to_string(), [2.0, 0.0, 2.0]),
        ("tetra-node:ch2".to_string(), [2.0, 2.0, 2.0]),
        ("tetra-node:ch3".to_string(), [0.0, 2.0, 2.0]),
        ("point-node:ch0".to_string(), [1.0, 1.0, 3.5]),
    ];

    let tetra_payload = store_forward_payload_with_channel_count(1_000_000_000, 0, 1, 4);
    let tetra_manifest = raw_manifest_for_payload_with_metadata(
        tmp.path(),
        "manifest-mixed-tetra",
        "seg-mixed-tetra",
        "tetra-node__audio_main__0001",
        tetra_payload,
        Some("cluster-mixed".to_string()),
        Some(cluster_sensor_positions.clone()),
    )
    .await;
    worker.process_one(tetra_manifest, 1).await;

    let point_payload = store_forward_payload_with_channel_count(1_000_000_000, 0, 2, 1);
    let point_manifest = raw_manifest_for_payload_with_metadata(
        tmp.path(),
        "manifest-mixed-point",
        "seg-mixed-point",
        "point-node__audio_main__0002",
        point_payload,
        Some("cluster-mixed".to_string()),
        Some(cluster_sensor_positions.clone()),
    )
    .await;
    worker.process_one(point_manifest, 1).await;

    let events = drain_published_manifests(&mut dsp_result_rx);
    let mixed_cluster_localizations: Vec<_> = events
        .iter()
        .filter(|manifest| {
            manifest.manifest_type == "localization_result"
                && manifest.cluster_id.as_deref() == Some("cluster-mixed")
                && manifest
                    .localization
                    .as_ref()
                    .is_some_and(|payload| payload.resolved_algorithm == "srp_phat")
        })
        .collect();

    assert!(
        !mixed_cluster_localizations.is_empty(),
        "expected a real SRP localization from a mixed tetrahedral-plus-point cluster"
    );
    let localization_manifest = mixed_cluster_localizations
        .iter()
        .copied()
        .find(|manifest| {
            manifest
                .localization
                .as_ref()
                .is_some_and(|payload| payload.pair_tdoas.len() == 10)
        })
        .expect("mixed cluster localization manifest with five-sensor pair diagnostics");
    let localization = localization_manifest
        .localization
        .as_ref()
        .expect("mixed cluster localization payload");
    assert_eq!(localization.pair_tdoas.len(), 10);
    assert_eq!(
        localization_manifest
            .cluster_sensor_positions
            .as_ref()
            .map(Vec::len),
        Some(5)
    );

    let state = state.read().await;
    assert!(state.total_localization_results >= 1);
}

#[tokio::test]
async fn consume_manifest_standalone_skips_non_persisted_channel_manifests() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest_store = ManifestStore::new(tmp.path());
    manifest_store.ensure_initialized().await.unwrap();

    let consumed_since_prune = Arc::new(AtomicU64::new(0));
    let manifest = DspManifest {
        manifest_id: "manifest-in-memory-only".to_string(),
        manifest_type: "raw_journal_append".to_string(),
        created_ns: system_now_ns(),
        source_handles: vec![],
        derived_handle: None,
        localization: None,
        classifier_render: None,
        birdnet: None,
        coverage_stats: None,
        promotion_ready: false,
        env_samples: None,
        node_context: None,
        cluster_id: None,
        cluster_sensor_positions: None,
        raw_payload: Some(vec![1, 2, 3]),
        raw_render_bytes: None,
        raw_audio_frame: None,
        raw_audio_bytes: None,
    };

    consume_manifest_standalone(&manifest, &manifest_store, &consumed_since_prune, 1, 10).await;

    assert_eq!(consumed_since_prune.load(Ordering::Relaxed), 0);
}

async fn raw_manifest_for_payload(
    root: &std::path::Path,
    manifest_id: &str,
    segment_id: &str,
    payload: String,
) -> DspManifest {
    raw_manifest_for_payload_with_metadata(
        root,
        manifest_id,
        segment_id,
        "sirith-test__audio_main__abcd",
        payload,
        None,
        None,
    )
    .await
}

async fn raw_manifest_for_payload_with_metadata(
    root: &std::path::Path,
    manifest_id: &str,
    segment_id: &str,
    stream_key: &str,
    payload: String,
    cluster_id: Option<String>,
    cluster_sensor_positions: Option<Vec<(String, [f32; 3])>>,
) -> DspManifest {
    let now_ns = system_now_ns();
    let payload_bytes = payload.into_bytes();
    let payload_length_bytes = payload_bytes.len() as u64;
    DspManifest {
        manifest_id: manifest_id.to_string(),
        manifest_type: "raw_journal_append".to_string(),
        created_ns: now_ns,
        source_handles: vec![JournalPayloadHandle {
            journal_epoch: 1,
            segment_id: segment_id.to_string(),
            stream_key: stream_key.to_string(),
            payload_offset_bytes: 0,
            payload_length_bytes,
            toa_ns: None,
            tor_ns: Some(now_ns as u64),
            received_ns: None,
            sample_index_start: Some(0),
            sample_count: Some(512),
            integrity_hash: String::new(),
            segment_path: root.join(format!("{segment_id}.in-memory-only.bin")),
        }],
        derived_handle: None,
        localization: None,
        classifier_render: None,
        birdnet: None,
        coverage_stats: None,
        promotion_ready: false,
        env_samples: None,
        node_context: None,
        cluster_id,
        cluster_sensor_positions,
        raw_payload: Some(payload_bytes),
        raw_render_bytes: None,
        raw_audio_frame: None,
        raw_audio_bytes: None,
    }
}

fn store_forward_payload() -> String {
    store_forward_payload_with_timing(1_000_000_000, 0, 1)
}

fn store_forward_payload_with_timing(
    start_time_ns: u64,
    start_sample_index: u64,
    sequence: u64,
) -> String {
    store_forward_payload_with_channel_count(start_time_ns, start_sample_index, sequence, 4)
}

fn store_forward_payload_with_timing_jitter(
    start_time_ns: u64,
    start_sample_index: u64,
    sequence: u64,
) -> String {
    let sr = 16_000;
    let samples = pseudo_random(520);
    let channels = (0..4)
        .map(|channel_index| samples[channel_index..channel_index + 512].to_vec())
        .collect::<Vec<_>>();
    let sensor_offsets = SIRITH_MIC_POSITIONS_M
        .iter()
        .map(|position| vec![position[0], position[1], position[2]])
        .collect::<Vec<_>>();
    serde_json::json!({
        "node": {
            "id": "sirith-test",
            "sensor_offsets_m": sensor_offsets,
            "metadata": {}
        },
        "buffered_frames": [{
            "frame": {
                "start_time_ns": start_time_ns,
                "utc_start_ns": start_time_ns,
                "utc_end_ns": start_time_ns + 32_000_000,
                "start_sample_index": start_sample_index,
                "end_sample_index": start_sample_index + 512,
                "sample_rate_hz": sr,
                "channels": 4,
                "encoding": "pcm16le",
                "samples_per_channel": 512,
                "samples_b64": encode_pcm16le_b64(&channels),
                "sequence": sequence,
                "time_quality": "gps_locked"
            }
        }]
    })
    .to_string()
}

fn store_forward_payload_with_channel_count(
    start_time_ns: u64,
    start_sample_index: u64,
    sequence: u64,
    channel_count: usize,
) -> String {
    let sr = 16_000;
    let samples = pseudo_random(520);
    let channels = (0..channel_count)
        .map(|channel_index| samples[channel_index..channel_index + 512].to_vec())
        .collect::<Vec<_>>();
    let sensor_offsets = if channel_count == 4 {
        SIRITH_MIC_POSITIONS_M
            .iter()
            .map(|position| vec![position[0], position[1], position[2]])
            .collect::<Vec<_>>()
    } else {
        (0..channel_count)
            .map(|index| vec![index as f32 * 0.01, 0.0, 0.0])
            .collect::<Vec<_>>()
    };
    serde_json::json!({
        "node": {
            "id": "sirith-test",
            "sensor_offsets_m": sensor_offsets,
            "metadata": {}
        },
        "buffered_frames": [{
            "frame": {
                "start_time_ns": start_time_ns,
                "utc_start_ns": start_time_ns,
                "utc_end_ns": start_time_ns + 32_000_000,
                "start_sample_index": start_sample_index,
                "end_sample_index": start_sample_index + 512,
                "sample_rate_hz": sr,
                "channels": channel_count,
                "encoding": "pcm16le",
                "samples_per_channel": 512,
                "samples_b64": encode_pcm16le_b64(&channels),
                "sequence": sequence,
                "time_quality": "gps_locked"
            }
        }]
    })
    .to_string()
}

/// 5-channel planar array payload with real (non-collinear) 25mm-radius
/// corner+center geometry, so the dispatch gate's `array_spans_at_least_2d`
/// check admits it into the spatial pipeline instead of the omni fallback.
fn store_forward_payload_with_planar_five_channel(half_space: Option<&str>) -> String {
    let sr = 16_000;
    let samples = pseudo_random(520);
    let channels = (0..5)
        .map(|channel_index| samples[channel_index..channel_index + 512].to_vec())
        .collect::<Vec<_>>();
    let r = 0.025 * std::f32::consts::FRAC_1_SQRT_2;
    let sensor_offsets = vec![
        vec![r, r, 0.0],
        vec![-r, r, 0.0],
        vec![-r, -r, 0.0],
        vec![r, -r, 0.0],
        vec![0.0, 0.0, 0.0],
    ];
    let mut node = serde_json::json!({
        "id": "sirith-planar-test",
        "sensor_offsets_m": sensor_offsets,
        "metadata": {}
    });
    if let Some(half_space) = half_space {
        node["half_space"] = serde_json::Value::String(half_space.to_string());
    }
    serde_json::json!({
        "node": node,
        "buffered_frames": [{
            "frame": {
                "start_time_ns": 1_000_000_000u64,
                "utc_start_ns": 1_000_000_000u64,
                "utc_end_ns": 1_000_000_000u64 + 32_000_000,
                "start_sample_index": 0,
                "end_sample_index": 512,
                "sample_rate_hz": sr,
                "channels": 5,
                "encoding": "pcm16le",
                "samples_per_channel": 512,
                "samples_b64": encode_pcm16le_b64(&channels),
                "sequence": 1,
                "time_quality": "gps_locked"
            }
        }]
    })
    .to_string()
}

fn store_forward_payload_with_internal_gap() -> String {
    let first: serde_json::Value =
        serde_json::from_str(&store_forward_payload_with_timing(1_000_000_000, 0, 1)).unwrap();
    let second: serde_json::Value =
        serde_json::from_str(&store_forward_payload_with_timing(1_064_000_000, 1024, 2)).unwrap();
    serde_json::json!({
        "node": first["node"].clone(),
        "buffered_frames": [
            first["buffered_frames"][0].clone(),
            second["buffered_frames"][0].clone()
        ]
    })
    .to_string()
}

fn drain_published_manifests(
    rx: &mut tokio::sync::broadcast::Receiver<ReplayableDspEvent>,
) -> Vec<DspManifest> {
    let mut events = Vec::new();
    while let Ok(event) = rx.try_recv() {
        events.push(event.manifest);
    }
    events
}

fn pseudo_random(n: usize) -> Vec<f32> {
    let mut x = 0x12345678_u32;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (x as i32 as f32) / (i32::MAX as f32)
        })
        .collect()
}

fn encode_pcm16le_b64(channels: &[Vec<f32>]) -> String {
    let frame_count = channels.iter().map(Vec::len).min().unwrap_or(0);
    let mut bytes = Vec::with_capacity(frame_count * channels.len() * 2);
    for sample_index in 0..frame_count {
        for channel in channels {
            let pcm = (channel[sample_index].clamp(-1.0, 1.0) * 32767.0).round() as i16;
            bytes.extend_from_slice(&pcm.to_le_bytes());
        }
    }
    STANDARD.encode(bytes)
}

#[test]
fn low_coverage_channel_drops_out_of_localization_set() {
    let sample_rate_hz = 16_000;
    let window_seconds = 512.0 / sample_rate_hz as f64;
    let start_time_ns = 1_000_000_000_000_i128;
    let end_time_ns = start_time_ns + (512 * 1_000_000_000_i128 / i128::from(sample_rate_hz));

    let mut buffers: [SensorStreamBuffer; 4] =
        core::array::from_fn(|_| SensorStreamBuffer::new(sample_rate_hz, 1.0));
    for buffer in buffers.iter_mut().take(3) {
        buffer
            .append(start_time_ns, &vec![1.0; 512], Some(0), Some(512))
            .unwrap();
    }
    buffers[3]
        .append(start_time_ns, &vec![1.0; 256], Some(0), Some(256))
        .unwrap();
    buffers[3]
        .append(
            start_time_ns + (384 * 1_000_000_000_i128 / i128::from(sample_rate_hz)),
            &vec![1.0; 128],
            Some(384),
            Some(512),
        )
        .unwrap();

    let channel_states = localization_channel_states(&buffers, end_time_ns, window_seconds);
    let active_channels = eligible_localization_channels(&channel_states, 0.85);

    assert_eq!(active_channels, vec![0, 1, 2]);
    assert!(
        channel_states[3]
            .coverage
            .as_ref()
            .expect("coverage stats")
            .coverage_ratio
            < 0.85
    );
}

#[test]
fn resolve_localization_center_time_skips_invalid_clamp_when_buffer_is_shorter_than_window() {
    let sample_rate_hz = 16_000;
    let window_seconds = 512.0 / sample_rate_hz as f64;
    let start_time_ns = 1_000_000_000_000_i128;
    let end_time_ns = start_time_ns + (256 * 1_000_000_000_i128 / i128::from(sample_rate_hz));

    let mut buffers: [SensorStreamBuffer; 4] =
        core::array::from_fn(|_| SensorStreamBuffer::new(sample_rate_hz, 1.0));
    for buffer in &mut buffers {
        buffer
            .append(start_time_ns, &vec![1.0; 256], Some(0), Some(256))
            .unwrap();
    }

    let center_time_ns =
        resolve_localization_center_time_ns(&buffers, start_time_ns, end_time_ns, window_seconds);
    assert_eq!(center_time_ns, start_time_ns);

    let channel_states =
        localization_channel_states_centered(&buffers, center_time_ns, window_seconds);
    assert!(channel_states.iter().all(|state| state.coverage.is_none()));
    assert!(channel_states.iter().all(|state| state.window.is_empty()));
}

#[test]
fn resolve_buffer_start_time_prefers_packet_toa_over_other_fallbacks() {
    let decoded = DecodedAudioPayload {
        channels: vec![vec![0.0; 16]; 4],
        sample_rate_hz: 16_000,
        start_time_ns: None,
        start_sample_index: Some(16_000),
        end_sample_index: Some(16_016),
        temperature_c: None,
        humidity_fraction: None,
        environment_source: None,
    };
    let handle = JournalPayloadHandle {
        journal_epoch: 1,
        segment_id: "seg-test".to_string(),
        stream_key: "sirith-test__audio_main__abcd".to_string(),
        payload_offset_bytes: 0,
        payload_length_bytes: 0,
        toa_ns: Some(1_234_567_890),
        tor_ns: Some(9_876_543_210),
        received_ns: None,
        sample_index_start: Some(16_000),
        sample_count: Some(16),
        integrity_hash: String::new(),
        segment_path: std::path::PathBuf::new(),
    };

    assert_eq!(
        resolve_buffer_start_time_ns(&decoded, &handle, 16_000, 55_555),
        1_234_567_890
    );
}

#[test]
fn resolve_buffer_start_time_uses_tor_as_fallback() {
    let decoded = DecodedAudioPayload {
        channels: vec![vec![0.0; 16]; 4],
        sample_rate_hz: 16_000,
        start_time_ns: None,
        start_sample_index: Some(32_000),
        end_sample_index: Some(32_016),
        temperature_c: None,
        humidity_fraction: None,
        environment_source: None,
    };
    let handle = JournalPayloadHandle {
        journal_epoch: 1,
        segment_id: "seg-test".to_string(),
        stream_key: "sirith-test__audio_main__abcd".to_string(),
        payload_offset_bytes: 0,
        payload_length_bytes: 0,
        toa_ns: None,
        tor_ns: Some(4_900_000_000),
        received_ns: None,
        sample_index_start: Some(32_000),
        sample_count: Some(16),
        integrity_hash: String::new(),
        segment_path: std::path::PathBuf::new(),
    };

    assert_eq!(
        resolve_buffer_start_time_ns(&decoded, &handle, 16_000, 5_000_000_000),
        2_900_000_000
    );
}

#[test]
fn resolve_buffer_start_time_uses_now_as_fallback() {
    let decoded = DecodedAudioPayload {
        channels: vec![vec![0.0; 16]; 4],
        sample_rate_hz: 16_000,
        start_time_ns: None,
        start_sample_index: Some(32_000),
        end_sample_index: Some(32_016),
        temperature_c: None,
        humidity_fraction: None,
        environment_source: None,
    };
    let handle = JournalPayloadHandle {
        journal_epoch: 1,
        segment_id: "seg-test".to_string(),
        stream_key: "sirith-test__audio_main__abcd".to_string(),
        payload_offset_bytes: 0,
        payload_length_bytes: 0,
        toa_ns: None,
        tor_ns: None,
        received_ns: None,
        sample_index_start: Some(32_000),
        sample_count: Some(16),
        integrity_hash: String::new(),
        segment_path: std::path::PathBuf::new(),
    };

    assert_eq!(
        resolve_buffer_start_time_ns(&decoded, &handle, 16_000, 5_000_000_000),
        3_000_000_000
    );
}

#[test]
fn receipt_time_alignment_never_overrides_firmware_timestamps_for_tdoa() {
    // Even with extreme skew, firmware/node timing must remain authoritative
    // when available so TDOA continues to use packet timestamps.
    assert!(!should_use_receipt_time_alignment(
        true,
        9_000_000_000,
        100_000_000,
    ));
}

#[test]
fn receipt_time_alignment_is_only_fallback_when_node_timestamps_absent() {
    assert!(should_use_receipt_time_alignment(
        false,
        9_000_000_000,
        100_000_000,
    ));
    assert!(!should_use_receipt_time_alignment(
        false,
        50_000_000,
        100_000_000,
    ));
}

#[test]
fn existing_sample_timeline_is_stable_across_receipt_jitter() {
    let sample_rate_hz = 16_000;
    let first_packet_start_ns = 4_900_000_000_i128;
    let mut buffer = SensorStreamBuffer::new(sample_rate_hz, 32.0);
    buffer
        .append(first_packet_start_ns, &vec![0.0; 1280], Some(0), Some(1280))
        .unwrap();

    let expected_second_packet_start_ns =
        first_packet_start_ns + (1280 * 1_000_000_000_i128 / i128::from(sample_rate_hz));

    assert_eq!(
        buffer.time_for_sample_index(1280),
        Some(expected_second_packet_start_ns)
    );
}

#[test]
fn default_classifier_window_has_birdnet_context_and_sparse_stride() {
    let config = DspWorkerConfig::default();

    assert!(config.classification_window_seconds >= 15.0);
    assert_eq!(config.classification_window_seconds, 30.0);
    assert_eq!(config.classifier_render_min_interval_seconds, 28.0);
    assert!(config.max_buffer_seconds >= 32.0);
}

#[test]
fn stale_manifest_detection_uses_source_receipt_time() {
    let manifest = DspManifest {
        manifest_id: "manifest-stale".to_string(),
        manifest_type: "raw_journal_append".to_string(),
        created_ns: 1_000_000_000,
        source_handles: vec![JournalPayloadHandle {
            journal_epoch: 1,
            segment_id: "seg-stale".to_string(),
            stream_key: "sirith-test__audio_main__abcd".to_string(),
            payload_offset_bytes: 0,
            payload_length_bytes: 0,
            toa_ns: Some(1_000_000_000),
            tor_ns: Some(2_000_000_000),
            received_ns: None,
            sample_index_start: None,
            sample_count: None,
            integrity_hash: String::new(),
            segment_path: std::path::PathBuf::new(),
        }],
        derived_handle: None,
        localization: None,
        classifier_render: None,
        birdnet: None,
        coverage_stats: None,
        promotion_ready: false,
        env_samples: None,
        node_context: None,
        cluster_id: None,
        cluster_sensor_positions: None,
        raw_payload: None,
        raw_render_bytes: None,
        raw_audio_frame: None,
        raw_audio_bytes: None,
    };

    assert!(manifest_is_older_than_buffer_horizon(
        &manifest,
        40_000_000_001,
        32.0
    ));
    assert!(!manifest_is_older_than_buffer_horizon(
        &manifest,
        30_000_000_000,
        32.0
    ));
}

#[test]
fn stale_manifest_detection_keeps_fresh_manifest_with_old_packet_epoch() {
    let manifest = DspManifest {
        manifest_id: "manifest-old-epoch-fresh-arrival".to_string(),
        manifest_type: "raw_journal_append".to_string(),
        created_ns: 1_000_000_000,
        source_handles: vec![JournalPayloadHandle {
            journal_epoch: 1,
            segment_id: "seg-old-epoch".to_string(),
            stream_key: "sirith-test__audio_main__abcd".to_string(),
            payload_offset_bytes: 0,
            payload_length_bytes: 0,
            toa_ns: Some(1_000_000_000),
            tor_ns: Some(2_000_000_000),
            received_ns: Some(40_000_000_000),
            sample_index_start: None,
            sample_count: None,
            integrity_hash: String::new(),
            segment_path: std::path::PathBuf::new(),
        }],
        derived_handle: None,
        localization: None,
        classifier_render: None,
        birdnet: None,
        coverage_stats: None,
        promotion_ready: false,
        env_samples: None,
        node_context: None,
        cluster_id: None,
        cluster_sensor_positions: None,
        raw_payload: None,
        raw_render_bytes: None,
        raw_audio_frame: None,
        raw_audio_bytes: None,
    };

    // Even with old packet timestamps, the fresh ingest receipt time keeps the
    // manifest in the live path when using a 32 s stale horizon.
    assert!(!manifest_is_older_than_buffer_horizon(
        &manifest,
        40_000_000_001,
        32.0
    ));
}

#[test]
fn classifier_render_interval_is_zero_when_disabled() {
    assert_eq!(classifier_render_min_interval_ns(0.0, 999), 0);
    assert_eq!(classifier_render_min_interval_ns(-1.0, 999), 0);
}

#[test]
fn classifier_render_interval_does_not_scale_with_backlog() {
    let expected = 1_000_000_000_u128;
    assert_eq!(classifier_render_min_interval_ns(1.0, 0), expected);
    assert_eq!(classifier_render_min_interval_ns(1.0, 10_000), expected);
}

#[test]
fn merge_pending_manifests_prioritizes_fresh_disk_when_channel_is_busy() {
    let channel = vec![
        test_manifest_with_created_ns("channel-old-1", 1),
        test_manifest_with_created_ns("channel-old-2", 2),
        test_manifest_with_created_ns("channel-old-3", 3),
        test_manifest_with_created_ns("channel-old-4", 4),
    ];
    let disk = vec![test_manifest_with_created_ns("disk-fresh", 1_000)];

    let merged = merge_pending_manifests_for_batch(channel, disk, 4);
    let merged_ids: Vec<&str> = merged
        .iter()
        .map(|manifest| manifest.manifest_id.as_str())
        .collect();

    assert_eq!(merged.len(), 4);
    assert!(merged_ids.contains(&"disk-fresh"));
    assert_eq!(merged[0].manifest_id, "disk-fresh");
}

#[test]
fn merge_pending_manifests_deduplicates_by_manifest_id() {
    let channel = vec![test_manifest_with_created_ns("shared", 10)];
    let disk = vec![
        test_manifest_with_created_ns("shared", 20),
        test_manifest_with_created_ns("disk-other", 15),
    ];

    let merged = merge_pending_manifests_for_batch(channel, disk, 10);
    let shared_count = merged
        .iter()
        .filter(|manifest| manifest.manifest_id == "shared")
        .count();

    assert_eq!(shared_count, 1);
    assert_eq!(merged.len(), 2);
}

#[test]
fn poll_cycle_sleep_policy_only_sleeps_when_no_work_was_processed() {
    assert!(should_sleep_after_poll_cycle(0));
    assert!(!should_sleep_after_poll_cycle(1));
    assert!(!should_sleep_after_poll_cycle(128));
}

fn test_manifest_with_created_ns(manifest_id: &str, created_ns: u128) -> DspManifest {
    DspManifest {
        manifest_id: manifest_id.to_string(),
        manifest_type: "raw_journal_append".to_string(),
        created_ns,
        source_handles: vec![],
        derived_handle: None,
        localization: None,
        classifier_render: None,
        birdnet: None,
        coverage_stats: None,
        promotion_ready: false,
        env_samples: None,
        node_context: None,
        cluster_id: None,
        cluster_sensor_positions: None,
        raw_payload: None,
        raw_render_bytes: None,
        raw_audio_frame: None,
        raw_audio_bytes: None,
    }
}

/// Smoke test for the cross-language silent-drop counters surfaced via
/// `/api/v1/dsp/status`. The fields are the Rust counterpart to Python's
/// `localization_drops_by_reason` / `SensorStreamBuffer.reanchor_count` and
/// must start at zero, accept saturating increments, and survive serialization
/// — these are the load-bearing properties for the alert query that watches
/// both backends with one rule.
#[test]
fn dsp_worker_state_silent_drop_counters_default_zero_and_increment() {
    let mut state = crate::dsp_worker::DspWorkerState::default();
    assert_eq!(state.total_buffer_reanchors, 0);
    assert_eq!(state.total_window_underrun_drops, 0);

    state.total_buffer_reanchors = state.total_buffer_reanchors.saturating_add(3);
    state.total_window_underrun_drops = state.total_window_underrun_drops.saturating_add(1);

    assert_eq!(state.total_buffer_reanchors, 3);
    assert_eq!(state.total_window_underrun_drops, 1);
}

// ---------------------------------------------------------------------------
// Fix 1: purge_stale_streams — per-stream HashMap eviction
// ---------------------------------------------------------------------------

/// Helper that builds a minimal DspWorker without a real filesystem store.
async fn make_test_worker(tmp: &tempfile::TempDir) -> DspWorker {
    let manifest_store = ManifestStore::new(tmp.path());
    manifest_store.ensure_initialized().await.unwrap();
    let derived_cache = DerivedCache::new(
        tmp.path(),
        DerivedCacheConfig {
            budget_bytes: 16_777_216,
            admission_reserve_bytes: 0,
        },
    );
    derived_cache.ensure_initialized().await.unwrap();
    let state: SharedDspState = Arc::new(RwLock::new(Default::default()));
    DspWorker::new(
        manifest_store,
        derived_cache,
        DspWorkerConfig::default(),
        state,
    )
}

#[tokio::test]
async fn worker_shutdown_exits_after_drain_while_backend_sender_remains_open() {
    let tmp = tempfile::tempdir().unwrap();
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    let (_backend_sender, receiver) = tokio::sync::mpsc::channel(1);
    let worker = make_test_worker(&tmp)
        .await
        .with_raw_manifest_receiver(receiver)
        .with_shutdown_signal(shutdown_requested.clone());

    assert!(!worker.should_exit_after_drain(0));

    shutdown_requested.store(true, Ordering::Release);

    // The backend owns the sender until process teardown. Requiring the
    // receiver to close here would leave the sidecar waiting forever.
    assert!(worker.should_exit_after_drain(0));
    assert!(!worker.should_exit_after_drain(1));
}

#[tokio::test]
async fn purge_stale_streams_removes_idle_stream_entries() {
    let tmp = tempfile::tempdir().unwrap();
    let mut worker = make_test_worker(&tmp).await;

    // Insert two streams: one ancient, one recent.
    let old_key = "node-old__audio_main__0000".to_string();
    let live_key = "node-live__audio_main__0000".to_string();

    let now_ns = system_now_ns();
    let two_hours_ago = now_ns.saturating_sub(2 * 3600 * 1_000_000_000);

    worker
        .last_heartbeat_ns_by_stream
        .insert(old_key.clone(), two_hours_ago);
    worker
        .last_heartbeat_ns_by_stream
        .insert(live_key.clone(), now_ns);
    worker
        .last_localization_ns_by_stream
        .insert(old_key.clone(), two_hours_ago);
    worker
        .last_localization_ns_by_stream
        .insert(live_key.clone(), now_ns);
    worker
        .last_trigger_ns_by_stream
        .insert(old_key.clone(), two_hours_ago);
    worker
        .last_trigger_ns_by_stream
        .insert(live_key.clone(), now_ns);
    worker
        .last_classifier_render_ns_by_stream
        .insert(old_key.clone(), two_hours_ago);
    worker
        .last_classifier_render_ns_by_stream
        .insert(live_key.clone(), now_ns);

    // TTL is 1h (3600s default); old stream (2h) should be evicted, live stream kept.
    worker.purge_stale_streams().await;

    assert!(
        !worker.last_heartbeat_ns_by_stream.contains_key(&old_key),
        "stale stream should be evicted"
    );
    assert!(
        worker.last_heartbeat_ns_by_stream.contains_key(&live_key),
        "live stream should be retained"
    );
    assert!(!worker.last_localization_ns_by_stream.contains_key(&old_key));
    assert!(worker
        .last_localization_ns_by_stream
        .contains_key(&live_key));

    let st = worker.state.read().await;
    assert_eq!(
        st.total_stale_streams_evicted, 1,
        "eviction counter should be 1"
    );
}

#[tokio::test]
async fn purge_stale_streams_keeps_all_when_none_idle() {
    let tmp = tempfile::tempdir().unwrap();
    let mut worker = make_test_worker(&tmp).await;

    let now_ns = system_now_ns();
    for i in 0..3 {
        let key = format!("node-{i}__audio_main__0000");
        worker
            .last_heartbeat_ns_by_stream
            .insert(key.clone(), now_ns);
        worker.last_localization_ns_by_stream.insert(key, now_ns);
    }

    worker.purge_stale_streams().await;

    assert_eq!(
        worker.last_heartbeat_ns_by_stream.len(),
        3,
        "no streams should be evicted"
    );
    let st = worker.state.read().await;
    assert_eq!(st.total_stale_streams_evicted, 0);
}

#[tokio::test]
async fn purge_stale_streams_evicts_node_audio_state_when_all_channels_gone() {
    let tmp = tempfile::tempdir().unwrap();
    let mut worker = make_test_worker(&tmp).await;

    let now_ns = system_now_ns();
    let stale_ns = now_ns.saturating_sub(2 * 3600 * 1_000_000_000);

    // Insert all 4 channels for "stale-node" as stale.
    for ch in 0..4u32 {
        let key = format!("stale-node__audio_main__{ch:04x}");
        worker.last_heartbeat_ns_by_stream.insert(key, stale_ns);
    }
    worker
        .node_audio_state
        .insert("stale-node".to_string(), NodeAudioState::default());

    worker.purge_stale_streams().await;

    assert!(worker.last_heartbeat_ns_by_stream.is_empty());
    assert!(
        !worker.node_audio_state.contains_key("stale-node"),
        "node_audio_state should be removed when all its streams are evicted"
    );
}
