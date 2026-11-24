//! R3 sample-rate guard at worker level: an explicit cluster mixing a 16 kHz
//! tetra node with a 32 kHz point node. Each buffer channel runs at its own
//! sensor's rate, and the tetra node's solves exclude the point sensor
//! (no resampling this round).

use super::dsp_worker_coverage_race_tests::{race_config, RaceHarness};
use super::dsp_worker_solve_manifest_inspection::*;
use super::dsp_worker_tests::{
    drain_published_manifests, encode_pcm16le_b64, raw_manifest_for_payload_with_metadata,
};
use super::*;
use minimappr_core::localization_sensor_entry::SolveExclusionReason;

const CLUSTER_ID: &str = "rate-guard";
const CLUSTER_BUFFER_KEY: &str = "cluster::rate-guard::audio_main";
const TETRA_NODE: &str = "rg-tetra";
const POINT_NODE: &str = "rg-point";
const TETRA_RATE_HZ: u32 = 16_000;
const POINT_RATE_HZ: u32 = 32_000;
const FRAME_DURATION_NS: u64 = 32_000_000;
const STREAM_START_NS: u64 = 1_000_000_000;
const FRAME_COUNT: usize = 30;
const NOISE_PEAK: f32 = 0.3;

fn cluster_positions() -> Vec<(String, [f32; 3])> {
    let mut positions: Vec<(String, [f32; 3])> = SIRITH_MIC_POSITIONS_M
        .iter()
        .enumerate()
        .map(|(channel, offset)| (format!("{TETRA_NODE}:ch{channel}"), *offset))
        .collect();
    positions.push((format!("{POINT_NODE}:ch0"), [3.0, 1.0, 0.5]));
    positions
}

fn noise(sample_count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..sample_count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            NOISE_PEAK * (state as i32 as f32 / i32::MAX as f32)
        })
        .collect()
}

async fn frame_manifest(
    harness: &RaceHarness,
    node_id: &str,
    channel_count: usize,
    sample_rate_hz: u32,
    frame_index: usize,
) -> DspManifest {
    let samples_per_frame =
        (u64::from(sample_rate_hz) * FRAME_DURATION_NS / 1_000_000_000) as usize;
    let start_time_ns = STREAM_START_NS + frame_index as u64 * FRAME_DURATION_NS;
    let start_sample_index = (frame_index * samples_per_frame) as u64;
    let channels: Vec<Vec<f32>> = (0..channel_count)
        .map(|channel| noise(samples_per_frame, (frame_index * 8 + channel + 1) as u32))
        .collect();
    let payload = serde_json::json!({
        "node": {"id": node_id},
        "buffered_frames": [{
            "frame": {
                "start_time_ns": start_time_ns,
                "utc_start_ns": start_time_ns,
                "utc_end_ns": start_time_ns + FRAME_DURATION_NS,
                "start_sample_index": start_sample_index,
                "end_sample_index": start_sample_index + samples_per_frame as u64,
                "sample_rate_hz": sample_rate_hz,
                "channels": channel_count,
                "encoding": "pcm16le",
                "samples_per_channel": samples_per_frame,
                "samples_b64": encode_pcm16le_b64(&channels),
                "sequence": frame_index + 1,
                "time_quality": "gps_locked"
            }
        }]
    })
    .to_string();
    raw_manifest_for_payload_with_metadata(
        harness.storage_root(),
        &format!("manifest-{node_id}-{frame_index}"),
        &format!("seg-{node_id}-{frame_index}"),
        &format!("{node_id}__audio_main__0001"),
        payload,
        Some(CLUSTER_ID.to_string()),
        Some(cluster_positions()),
    )
    .await
}

#[tokio::test]
async fn mixed_rate_cluster_buffers_keep_each_rate_and_exclude_the_mismatch() {
    let mut harness = RaceHarness::new(race_config(true)).await;
    for frame_index in 0..FRAME_COUNT {
        harness.advance_clock_ms(FRAME_DURATION_NS / 1_000_000);
        for (node_id, channel_count, rate_hz) in [
            (TETRA_NODE, 4, TETRA_RATE_HZ),
            (POINT_NODE, 1, POINT_RATE_HZ),
        ] {
            let manifest =
                frame_manifest(&harness, node_id, channel_count, rate_hz, frame_index).await;
            harness.deliver_manifest(manifest).await;
        }
    }

    // Sensor order puts the point node (rg-point) ahead of the tetra.
    assert_eq!(
        harness.buffer_sample_rates_hz(CLUSTER_BUFFER_KEY),
        [
            POINT_RATE_HZ,
            TETRA_RATE_HZ,
            TETRA_RATE_HZ,
            TETRA_RATE_HZ,
            TETRA_RATE_HZ
        ]
    );

    let published = drain_published_manifests(&mut harness.events);
    let tetra_solve = attempted_localization_results(&published)
        .into_iter()
        .rev()
        .find(|manifest| solve_provenance(manifest).reporting_node_id == TETRA_NODE)
        .expect("a tetra-reported cluster solve");
    let provenance = solve_provenance(tetra_solve);
    assert_eq!(provenance.solve_sample_rate_hz, TETRA_RATE_HZ);
    assert_eq!(provenance.sensors[0].sensor_id, format!("{POINT_NODE}:ch0"));
    assert_eq!(provenance.sensors[0].sample_rate_hz, POINT_RATE_HZ);
    assert_eq!(provenance.excluded_sensors.len(), 1);
    assert_eq!(provenance.excluded_sensors[0].sensor_index, 0);
    assert_eq!(
        provenance.excluded_sensors[0].reason,
        SolveExclusionReason::SampleRateMismatch
    );
    assert!(!provenance.active_sensor_indices.contains(&0));
    let pairs = &tetra_solve.localization.as_ref().unwrap().pair_tdoas;
    assert!(!pairs.is_empty());
    assert!(pairs.iter().all(|pair| pair.ch_a != 0 && pair.ch_b != 0));
}
