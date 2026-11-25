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
    cluster_frame_manifest(
        harness,
        (CLUSTER_ID, cluster_positions()),
        node_id,
        channel_count,
        sample_rate_hz,
        frame_index,
    )
    .await
}

async fn cluster_frame_manifest(
    harness: &RaceHarness,
    (cluster_id, cluster_sensor_positions): (&str, Vec<(String, [f32; 3])>),
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
        Some(cluster_id.to_string()),
        Some(cluster_sensor_positions),
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

const TWO_TETRA_CLUSTER_ID: &str = "rate-guard-tetras";
const FIRST_TETRA_NODE: &str = "rg-tetra-a";
const SECOND_TETRA_NODE: &str = "rg-tetra-b";
const TWO_TETRA_SEPARATION_M: f32 = 3.0;

fn two_tetra_positions() -> Vec<(String, [f32; 3])> {
    [
        (FIRST_TETRA_NODE, 0.0),
        (SECOND_TETRA_NODE, TWO_TETRA_SEPARATION_M),
    ]
    .into_iter()
    .flat_map(|(node_id, east_m)| {
        SIRITH_MIC_POSITIONS_M
            .iter()
            .enumerate()
            .map(move |(channel, offset)| {
                (
                    format!("{node_id}:ch{channel}"),
                    [east_m + offset[0], offset[1], offset[2]],
                )
            })
    })
    .collect()
}

/// Event-triggered solves of a two-tetra cluster, as (reporter, solve rate,
/// rate-excluded sensor count), with `leader` posting first in every slot.
async fn two_tetra_event_solves(
    leader: (&str, u32),
    follower: (&str, u32),
) -> Vec<(String, u32, usize)> {
    let mut harness = RaceHarness::new(race_config(true)).await;
    for frame_index in 0..FRAME_COUNT {
        harness.advance_clock_ms(FRAME_DURATION_NS / 1_000_000);
        for (node_id, rate_hz) in [leader, follower] {
            let manifest = cluster_frame_manifest(
                &harness,
                (TWO_TETRA_CLUSTER_ID, two_tetra_positions()),
                node_id,
                4,
                rate_hz,
                frame_index,
            )
            .await;
            harness.deliver_manifest(manifest).await;
        }
    }
    let published = drain_published_manifests(&mut harness.events);
    attempted_localization_results(&published)
        .into_iter()
        .map(solve_provenance)
        .filter(|provenance| provenance.attempt_kind == SolveAttemptKind::EventTriggerAbsoluteRms)
        .map(|provenance| {
            let rate_excluded = provenance
                .excluded_sensors
                .iter()
                .filter(|entry| entry.reason == SolveExclusionReason::SampleRateMismatch)
                .count();
            (
                provenance.reporting_node_id.clone(),
                provenance.solve_sample_rate_hz,
                rate_excluded,
            )
        })
        .collect()
}

/// Regression (E4): every dispatch gate was keyed by the shared cluster
/// buffer, so the node posting first consumed the cluster's trigger, cadence
/// and render slots and a node at another rate (excluded from the leader's
/// solves) never solved at all.
#[tokio::test]
async fn each_rate_group_of_a_mixed_rate_cluster_gets_its_own_event_solves() {
    for (leader, follower) in [
        (
            (FIRST_TETRA_NODE, TETRA_RATE_HZ),
            (SECOND_TETRA_NODE, POINT_RATE_HZ),
        ),
        (
            (SECOND_TETRA_NODE, POINT_RATE_HZ),
            (FIRST_TETRA_NODE, TETRA_RATE_HZ),
        ),
    ] {
        let solves = two_tetra_event_solves(leader, follower).await;
        for (node_id, rate_hz) in [leader, follower] {
            let own: Vec<_> = solves
                .iter()
                .filter(|(reporter, _, _)| reporter == node_id)
                .collect();
            assert!(!own.is_empty(), "{node_id} never solved: {solves:?}");
            // The leader's frame-0 solve predates the follower's first packet,
            // so only "excludes the other node once it streams" is required.
            assert!(
                own.iter().all(|(_, solve_rate, _)| *solve_rate == rate_hz)
                    && own.iter().any(|(_, _, rate_excluded)| *rate_excluded == 4),
                "{node_id}: {solves:?}"
            );
        }
    }
}

#[tokio::test]
async fn a_same_rate_cluster_still_solves_each_event_once() {
    let solves = two_tetra_event_solves(
        (FIRST_TETRA_NODE, TETRA_RATE_HZ),
        (SECOND_TETRA_NODE, TETRA_RATE_HZ),
    )
    .await;
    // ~0.96 s of audio under a 0.8 s cooldown: two trigger windows, one
    // shared solve each, all led by the first poster with nothing excluded.
    assert_eq!(
        solves,
        vec![
            (FIRST_TETRA_NODE.to_string(), TETRA_RATE_HZ, 0),
            (FIRST_TETRA_NODE.to_string(), TETRA_RATE_HZ, 0),
        ]
    );
}
