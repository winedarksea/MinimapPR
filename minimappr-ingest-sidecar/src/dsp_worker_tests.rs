use super::*;
use crate::{
    derived_cache::{DerivedCache, DerivedCacheConfig},
    dsp::SensorStreamBuffer,
    journal_reader::JournalPayloadHandle,
    manifests::ManifestStore,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use tokio::sync::RwLock;

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

    let (dsp_result_tx, mut dsp_result_rx) = tokio::sync::broadcast::channel(16);

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
        raw_payload: Some(payload.into_bytes()),
        raw_render_bytes: None,
    };
    let state: SharedDspState = Arc::new(RwLock::new(Default::default()));
    let mut worker = DspWorker::new(
        manifest_store.clone(),
        derived_cache,
        DspWorkerConfig {
            birdnet_hybrid_render_enabled: true,
            ..DspWorkerConfig::default()
        },
        state.clone(),
    )
    .with_dsp_result_sender(dsp_result_tx);

    worker.process_one(raw_manifest, 1).await;

    // Collect SSE events broadcast by run_io (no disk writes for memory-path manifests).
    let mut events: Vec<DspManifest> = Vec::new();
    while let Ok(m) = dsp_result_rx.try_recv() {
        events.push(m);
    }

    let localization_event = events
        .iter()
        .find(|m| m.manifest_type == "localization_result")
        .expect("localization_result SSE event");
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

    // localization_result carries derived_handle (Bug A fix) and inline PCM.
    assert!(localization_event.derived_handle.is_some());
    assert!(localization_event.raw_render_bytes.is_some());

    assert_eq!(
        render_event
            .classifier_render
            .as_ref()
            .expect("render payload")
            .sample_format,
        "pcm16le"
    );
    // Inline PCM delivered — segment_path is a sentinel (no disk file written).
    assert!(render_event.raw_render_bytes.is_some());
    assert!(!render_event.derived_handle.as_ref().unwrap().segment_path.exists());

    let state = state.read().await;
    assert_eq!(state.total_tdoa_results, 1);
    assert_eq!(state.total_localization_results, 1);
    assert_eq!(state.total_classifier_renders, 1);
}

#[tokio::test]
async fn worker_publishes_omni_render_when_localization_coverage_is_unavailable() {
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

    let (dsp_result_tx, mut dsp_result_rx) = tokio::sync::broadcast::channel(32);

    let state: SharedDspState = Arc::new(RwLock::new(Default::default()));
    let mut worker = DspWorker::new(
        manifest_store.clone(),
        derived_cache,
        DspWorkerConfig {
            birdnet_hybrid_render_enabled: true,
            classifier_render_min_interval_seconds: 0.0,
            max_buffer_seconds: 32.0,
            max_trusted_node_clock_skew_seconds: f64::MAX,
            ..DspWorkerConfig::default()
        },
        state.clone(),
    )
    .with_dsp_result_sender(dsp_result_tx);

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

    let mut events: Vec<DspManifest> = Vec::new();
    while let Ok(m) = dsp_result_rx.try_recv() {
        events.push(m);
    }

    let localizations: Vec<_> = events
        .iter()
        .filter(|m| m.manifest_type == "localization_result")
        .collect();
    assert_eq!(localizations.len(), 1);

    let renders: Vec<_> = events
        .iter()
        .filter(|m| m.manifest_type == "classifier_render")
        .collect();
    assert_eq!(renders.len(), 2);

    let fallback_render = renders
        .iter()
        .find(|render| {
            render
                .classifier_render
                .as_ref()
                .and_then(|payload| payload.fallback_reason.as_deref())
                == Some("localization_coverage_unavailable")
        })
        .expect("fallback render");
    assert_eq!(
        fallback_render
            .classifier_render
            .as_ref()
            .expect("render payload")
            .fallback_reason
            .as_deref(),
        Some("localization_coverage_unavailable")
    );

    let state = state.read().await;
    assert_eq!(state.total_localization_results, 1);
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

    let (dsp_result_tx, mut dsp_result_rx) = tokio::sync::broadcast::channel(32);

    let state: SharedDspState = Arc::new(RwLock::new(Default::default()));
    let mut worker = DspWorker::new(
        manifest_store.clone(),
        derived_cache,
        DspWorkerConfig {
            birdnet_hybrid_render_enabled: true,
            classifier_render_min_interval_seconds: 999.0,
            localization_cadence_ms: 0,
            trigger_cooldown_seconds: 0.0,
            ..DspWorkerConfig::default()
        },
        state.clone(),
    )
    .with_dsp_result_sender(dsp_result_tx);

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

    let mut events: Vec<DspManifest> = Vec::new();
    while let Ok(m) = dsp_result_rx.try_recv() {
        events.push(m);
    }

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
        raw_payload: Some(vec![1, 2, 3]),
        raw_render_bytes: None,
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
            stream_key: "sirith-test__audio_main__abcd".to_string(),
            payload_offset_bytes: 0,
            payload_length_bytes,
            toa_ns: None,
            tor_ns: Some(now_ns as u64),
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
        raw_payload: Some(payload_bytes),
        raw_render_bytes: None,
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
    let sr = 16_000;
    let samples = pseudo_random(520);
    let channels = vec![
        samples[0..512].to_vec(),
        samples[1..513].to_vec(),
        samples[2..514].to_vec(),
        samples[0..512].to_vec(),
    ];
    serde_json::json!({
        "node": {
            "id": "sirith-test",
            "sensor_offsets_m": SIRITH_MIC_POSITIONS_M,
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

    let mut buffers = core::array::from_fn(|_| SensorStreamBuffer::new(sample_rate_hz, 1.0));
    for channel_index in 0..3 {
        buffers[channel_index]
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
fn resolve_buffer_start_time_prefers_packet_toa_over_other_fallbacks() {
    let decoded = DecodedAudioPayload {
        channels: vec![vec![0.0; 16]; 4],
        sample_rate_hz: 16_000,
        start_time_ns: None,
        start_sample_index: Some(16_000),
        end_sample_index: Some(16_016),
        temperature_c: None,
        humidity_fraction: None,
    };
    let handle = JournalPayloadHandle {
        journal_epoch: 1,
        segment_id: "seg-test".to_string(),
        stream_key: "sirith-test__audio_main__abcd".to_string(),
        payload_offset_bytes: 0,
        payload_length_bytes: 0,
        toa_ns: Some(1_234_567_890),
        tor_ns: Some(9_876_543_210),
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
    };
    let handle = JournalPayloadHandle {
        journal_epoch: 1,
        segment_id: "seg-test".to_string(),
        stream_key: "sirith-test__audio_main__abcd".to_string(),
        payload_offset_bytes: 0,
        payload_length_bytes: 0,
        toa_ns: None,
        tor_ns: Some(4_900_000_000),
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
    };
    let handle = JournalPayloadHandle {
        journal_epoch: 1,
        segment_id: "seg-test".to_string(),
        stream_key: "sirith-test__audio_main__abcd".to_string(),
        payload_offset_bytes: 0,
        payload_length_bytes: 0,
        toa_ns: None,
        tor_ns: None,
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
        raw_payload: None,
        raw_render_bytes: None,
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
        created_ns: 40_000_000_000,
        source_handles: vec![JournalPayloadHandle {
            journal_epoch: 1,
            segment_id: "seg-old-epoch".to_string(),
            stream_key: "sirith-test__audio_main__abcd".to_string(),
            payload_offset_bytes: 0,
            payload_length_bytes: 0,
            toa_ns: Some(1_000_000_000),
            tor_ns: Some(2_000_000_000),
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
        raw_payload: None,
        raw_render_bytes: None,
    };

    // Even with old packet timestamps, the manifest just arrived and should
    // stay in the live path when using a 32s stale horizon.
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
        raw_payload: None,
        raw_render_bytes: None,
    }
}
