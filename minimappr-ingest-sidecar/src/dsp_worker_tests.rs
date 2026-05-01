use super::*;
use crate::{
    derived_cache::{DerivedCache, DerivedCacheConfig},
    dsp::SensorStreamBuffer,
    journal_reader::JournalPayloadHandle,
    manifests::ManifestStore,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use std::sync::Arc;
use tokio::{fs, sync::Mutex};

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
    let segment_path = tmp.path().join("segment.bin");
    fs::write(&segment_path, payload.as_bytes()).await.unwrap();
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
            segment_path,
        }],
        derived_handle: None,
        localization: None,
        classifier_render: None,
        birdnet: None,
        coverage_stats: None,
        promotion_ready: false,
    };
    let state: SharedDspState = Arc::new(Mutex::new(Default::default()));
    let mut worker = DspWorker::new(
        manifest_store.clone(),
        derived_cache,
        DspWorkerConfig {
            birdnet_hybrid_render_enabled: true,
            ..DspWorkerConfig::default()
        },
        state.clone(),
    );

    worker.process_one(raw_manifest, 1).await;

    let localizations = manifest_store
        .query_pending("localization_result")
        .await
        .unwrap();
    assert_eq!(localizations.len(), 1);
    let localization = localizations[0]
        .localization
        .as_ref()
        .expect("localization payload");
    assert_eq!(localization.attempted_algorithm, "srp_phat");
    assert_eq!(localization.resolved_algorithm, "srp_phat");
    assert_eq!(localization.pair_tdoas.len(), 6);

    let renders = manifest_store
        .query_pending("classifier_render")
        .await
        .unwrap();
    assert_eq!(renders.len(), 1);
    assert!(renders[0].promotion_ready);
    assert_eq!(
        renders[0]
            .classifier_render
            .as_ref()
            .expect("render payload")
            .sample_format,
        "pcm16le"
    );
    assert!(renders[0]
        .derived_handle
        .as_ref()
        .unwrap()
        .segment_path
        .exists());

    let state = state.lock().await;
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

    let state: SharedDspState = Arc::new(Mutex::new(Default::default()));
    let mut worker = DspWorker::new(
        manifest_store.clone(),
        derived_cache,
        DspWorkerConfig {
            birdnet_hybrid_render_enabled: true,
            max_buffer_seconds: 32.0,
            ..DspWorkerConfig::default()
        },
        state.clone(),
    );

    let first_payload = store_forward_payload_with_timing(1_000_000_000, 0, 1);
    let first_manifest =
        raw_manifest_for_payload(tmp.path(), "manifest-raw-first", "seg-first", first_payload)
            .await;
    worker.process_one(first_manifest, 2).await;

    // This mirrors the live Sirith failure mode: sample coverage appends, but
    // the frame timestamp maps the evaluation window outside the buffer.
    let second_payload = store_forward_payload_with_timing(100_000_000, 1024, 2);
    let second_manifest = raw_manifest_for_payload(
        tmp.path(),
        "manifest-raw-second",
        "seg-second",
        second_payload,
    )
    .await;
    worker.process_one(second_manifest, 2).await;

    let localizations = manifest_store
        .query_pending("localization_result")
        .await
        .unwrap();
    assert_eq!(localizations.len(), 1);

    let renders = manifest_store
        .query_pending("classifier_render")
        .await
        .unwrap();
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

    let state = state.lock().await;
    assert_eq!(state.total_localization_results, 1);
    assert_eq!(state.total_classifier_renders, 2);
}

#[tokio::test]
async fn worker_skips_stale_manifest_without_rewriting_timestamps() {
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

    let state: SharedDspState = Arc::new(Mutex::new(Default::default()));
    let mut worker = DspWorker::new(
        manifest_store.clone(),
        derived_cache,
        DspWorkerConfig {
            birdnet_hybrid_render_enabled: true,
            max_buffer_seconds: 0.5,
            skip_stale_manifests_for_live_buffer: true,
            ..DspWorkerConfig::default()
        },
        state.clone(),
    );

    let stale_payload = store_forward_payload_with_timing(1_000_000_000, 0, 1);
    let mut stale_manifest =
        raw_manifest_for_payload(tmp.path(), "manifest-raw-stale", "seg-stale", stale_payload)
            .await;
    stale_manifest.created_ns = 1;
    stale_manifest.source_handles[0].tor_ns = Some(1);
    stale_manifest.source_handles[0].toa_ns = Some(1);

    worker.process_one(stale_manifest, 1).await;

    let localizations = manifest_store
        .query_pending("localization_result")
        .await
        .unwrap();
    assert_eq!(localizations.len(), 0);
    let renders = manifest_store
        .query_pending("classifier_render")
        .await
        .unwrap();
    assert_eq!(renders.len(), 0);

    let state = state.lock().await;
    assert_eq!(state.total_stale_manifest_skips, 1);
    assert_eq!(state.total_localization_results, 0);
    assert_eq!(state.total_classifier_renders, 0);
}

async fn raw_manifest_for_payload(
    root: &std::path::Path,
    manifest_id: &str,
    segment_id: &str,
    payload: String,
) -> DspManifest {
    let segment_path = root.join(format!("{segment_id}.bin"));
    fs::write(&segment_path, payload.as_bytes()).await.unwrap();
    let now_ns = system_now_ns();
    DspManifest {
        manifest_id: manifest_id.to_string(),
        manifest_type: "raw_journal_append".to_string(),
        created_ns: now_ns,
        source_handles: vec![JournalPayloadHandle {
            journal_epoch: 1,
            segment_id: segment_id.to_string(),
            stream_key: "sirith-test__audio_main__abcd".to_string(),
            payload_offset_bytes: 0,
            payload_length_bytes: payload.len() as u64,
            toa_ns: None,
            tor_ns: Some(now_ns as u64),
            sample_index_start: Some(0),
            sample_count: Some(512),
            integrity_hash: String::new(),
            segment_path,
        }],
        derived_handle: None,
        localization: None,
        classifier_render: None,
        birdnet: None,
        coverage_stats: None,
        promotion_ready: false,
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
fn resolve_buffer_start_time_uses_relative_sample_time_before_now() {
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
        resolve_buffer_start_time_ns(&decoded, &handle, 16_000, 99_999),
        2_000_000_000
    );
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
