use super::*;
use crate::{
    derived_cache::{DerivedCache, DerivedCacheConfig},
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
    let raw_manifest = DspManifest {
        manifest_id: "manifest-raw-test".to_string(),
        manifest_type: "raw_journal_append".to_string(),
        created_ns: 1,
        source_handles: vec![JournalPayloadHandle {
            journal_epoch: 1,
            segment_id: "seg-test".to_string(),
            stream_key: "sirith-test__audio_main__abcd".to_string(),
            payload_offset_bytes: 0,
            payload_length_bytes: payload.len() as u64,
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

    worker.process_one(raw_manifest).await;

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

fn store_forward_payload() -> String {
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
                "start_time_ns": 1_000_000_000_u64,
                "utc_start_ns": 1_000_000_000_u64,
                "utc_end_ns": 1_032_000_000_u64,
                "start_sample_index": 0,
                "end_sample_index": 512,
                "sample_rate_hz": sr,
                "channels": 4,
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
