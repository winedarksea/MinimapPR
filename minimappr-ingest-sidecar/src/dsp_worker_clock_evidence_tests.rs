//! Worker-level clock evidence (R5): per-frame timestamps of a binary upload
//! reach the node clock tracker and the solve provenance, and a parked solve
//! publishes the clock snapshot it triggered with.

use super::dsp_worker_coverage_race_tests::{
    race_config, RaceHarness, CLUSTER_BUFFER_KEY, NODE_A, NODE_B,
};
use super::dsp_worker_solve_manifest_inspection::{
    attempted_localization_results, cross_node_pair_count, solve_provenance,
};
use super::dsp_worker_test_node_fixtures::{
    fixture_frame_manifest, CoherentPointSource, FixtureClusterGeometry, FixtureNode,
    FIXTURE_FRAME_DURATION_NS, FIXTURE_FRAME_SAMPLES, FIXTURE_SAMPLE_RATE_HZ,
};
use super::dsp_worker_tests::drain_published_manifests;
use super::*;
use minimappr_core::localization_sensor_entry::{NodeTimeQuality, SolveExclusionReason};

const IN_UPLOAD_STEP_NS: u64 = 21_000_000;
const UPLOAD_FRAME_COUNT: usize = 30;
const STEP_FRAME_INDEX: usize = 20;
const UPLOAD_FIRST_FRAME_TIME_NS: u64 = 1_764_000_000_000_000_000;
const BOOT_SESSION: u32 = 4;
const TETRA_NODE_TYPE_CODE: u8 = 1;
const GPS_LOCKED_WIRE_CODE: u8 = 0;
const PCM16_FULL_SCALE: f32 = 32_767.0;
const NOISE_AMPLITUDE: f32 = 0.3;

fn push_string(payload: &mut Vec<u8>, value: &str) {
    payload.push(u8::try_from(value.len()).unwrap());
    payload.extend_from_slice(value.as_bytes());
}

/// MMB2 upload (no diagnostics sections) of sample-contiguous frames, each
/// with its own packet timestamp: the format `minimappr/sim/binary_ingest_encoder.py` writes.
fn mmb2_upload(node: &FixtureNode, frame_start_times_ns: &[u64]) -> Vec<u8> {
    let channel_count = node.channel_count();
    let mut payload = b"MMB2".to_vec();
    payload.extend_from_slice(&[2, 1]);
    payload.extend_from_slice(
        &u16::try_from(frame_start_times_ns.len())
            .unwrap()
            .to_le_bytes(),
    );
    push_string(&mut payload, &node.node_id);
    payload.extend_from_slice(&[TETRA_NODE_TYPE_CODE, 0]);
    payload.push(u8::try_from(channel_count).unwrap());
    for offset in node.body_offsets_m() {
        offset
            .iter()
            .for_each(|axis| payload.extend_from_slice(&axis.to_le_bytes()));
    }
    payload.push(1);
    for text in [
        "audio",
        "test-hardware",
        "test-firmware",
        "fix_3d",
        "gps_nmea_uart",
    ] {
        push_string(&mut payload, text);
    }
    payload.extend_from_slice(&BOOT_SESSION.to_le_bytes());
    let mut noise_state = 0x0bad_5eed_u32;
    for (frame_index, start_time_ns) in frame_start_times_ns.iter().copied().enumerate() {
        let start_sample_index = (frame_index * FIXTURE_FRAME_SAMPLES) as u64;
        let end_sample_index = start_sample_index + FIXTURE_FRAME_SAMPLES as u64;
        payload.extend_from_slice(&start_time_ns.to_le_bytes());
        payload.extend_from_slice(&(start_time_ns + FIXTURE_FRAME_DURATION_NS).to_le_bytes());
        payload.extend_from_slice(&start_sample_index.to_le_bytes());
        payload.extend_from_slice(&end_sample_index.to_le_bytes());
        payload.extend_from_slice(&FIXTURE_SAMPLE_RATE_HZ.to_le_bytes());
        payload.push(u8::try_from(channel_count).unwrap());
        payload.extend_from_slice(&(frame_index as u64 + 1).to_le_bytes());
        payload.extend_from_slice(&start_time_ns.to_le_bytes());
        payload.extend_from_slice(&(start_time_ns + FIXTURE_FRAME_DURATION_NS).to_le_bytes());
        payload.extend_from_slice(&[GPS_LOCKED_WIRE_CODE, 0, 0]);
        payload.extend_from_slice(&u32::try_from(FIXTURE_FRAME_SAMPLES).unwrap().to_le_bytes());
        for _ in 0..FIXTURE_FRAME_SAMPLES * channel_count {
            noise_state = noise_state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            let sample = NOISE_AMPLITUDE * (noise_state as i32 as f32 / i32::MAX as f32);
            payload.extend_from_slice(&((sample * PCM16_FULL_SCALE) as i16).to_le_bytes());
        }
    }
    payload
}

fn replace_raw_payload(manifest: &mut DspManifest, raw_payload: Vec<u8>) {
    manifest.source_handles[0].payload_length_bytes = raw_payload.len() as u64;
    manifest.raw_payload = Some(raw_payload);
}

#[tokio::test]
async fn a_21_ms_step_inside_one_binary_upload_reaches_node_timing() {
    let mut harness = RaceHarness::new(race_config(false)).await;
    let node = FixtureNode::tetra("tetra-step", [0.0, 0.0, 1.5]);
    let source = CoherentPointSource::new([5.0, 5.0, 1.5], 1);
    let mut manifest = fixture_frame_manifest(
        harness.storage_root(),
        &node,
        0,
        &source,
        &FixtureClusterGeometry::AutomaticGps,
    )
    .await;
    // No sequence gap and contiguous sample indices: only the timestamps step.
    let frame_start_times_ns: Vec<u64> = (0..UPLOAD_FRAME_COUNT)
        .map(|frame_index| {
            let nominal_ns =
                UPLOAD_FIRST_FRAME_TIME_NS + frame_index as u64 * FIXTURE_FRAME_DURATION_NS;
            nominal_ns
                + if frame_index >= STEP_FRAME_INDEX {
                    IN_UPLOAD_STEP_NS
                } else {
                    0
                }
        })
        .collect();
    replace_raw_payload(&mut manifest, mmb2_upload(&node, &frame_start_times_ns));
    harness.deliver_manifest(manifest).await;

    let published = drain_published_manifests(&mut harness.events);
    let solves = attempted_localization_results(&published);
    let solve = solves.first().expect("the upload triggers a solve");
    let provenance = solve_provenance(solve);
    let timing = provenance
        .node_timing
        .iter()
        .find(|row| row.node_id == node.node_id)
        .expect("node_timing row for the reporting node");
    assert_eq!(timing.time_quality, NodeTimeQuality::GpsLocked);
    assert_eq!(timing.timestamp_step_count, 1);
    assert_eq!(
        timing.last_timestamp_step_ns,
        Some(IN_UPLOAD_STEP_NS as i64)
    );
    // The buffer anchored at the upload's first packet, so the stepped frames
    // sit 21 ms off their packet times.
    assert_eq!(timing.timeline_residual_ns, Some(IN_UPLOAD_STEP_NS as i64));
    let expected_uncertainty_s = IN_UPLOAD_STEP_NS as f64 / 1e9;
    assert!((timing.clock_offset_std_s.unwrap() - expected_uncertainty_s).abs() < 1e-12);
    for sensor in provenance
        .sensors
        .iter()
        .filter(|sensor| sensor.node_id == node.node_id)
    {
        assert_eq!(sensor.clock_source, NodeTimeQuality::GpsLocked);
        assert_eq!(sensor.clock_offset_std_s, timing.clock_offset_std_s);
    }
}

/// Shifts every frame of a store-forward manifest's payload by `offset_ns`.
fn shift_packet_time(manifest: &mut DspManifest, offset_ns: u64) {
    let raw_payload = manifest.raw_payload.as_ref().expect("in-memory payload");
    let mut payload: serde_json::Value = serde_json::from_slice(raw_payload).unwrap();
    for buffered in payload["buffered_frames"].as_array_mut().unwrap() {
        for field in ["start_time_ns", "utc_start_ns", "utc_end_ns"] {
            let shifted = buffered["frame"][field].as_u64().unwrap() + offset_ns;
            buffered["frame"][field] = serde_json::Value::from(shifted);
        }
    }
    replace_raw_payload(manifest, serde_json::to_vec(&payload).unwrap());
}

#[tokio::test]
async fn a_parked_solve_publishes_the_clock_snapshot_it_triggered_with() {
    const TRIGGER_FRAME: usize = 25;
    let mut harness = RaceHarness::new(race_config(true)).await;
    // Node b runs two frames behind, so node a's trigger at frame 25 parks.
    for frame_index in 0..=TRIGGER_FRAME {
        harness.advance_clock_ms(FIXTURE_FRAME_DURATION_NS / 1_000_000);
        harness.deliver(NODE_A, frame_index).await;
        if frame_index >= 2 {
            harness.deliver(NODE_B, frame_index - 2).await;
        }
    }
    assert!(harness.has_pending_localization(CLUSTER_BUFFER_KEY));
    drain_published_manifests(&mut harness.events);

    // Node b's late frames carry a 21 ms clock step; they complete coverage.
    for frame_index in [TRIGGER_FRAME - 1, TRIGGER_FRAME] {
        let mut manifest = harness.cluster_frame_manifest(NODE_B, frame_index).await;
        shift_packet_time(&mut manifest, IN_UPLOAD_STEP_NS);
        harness.deliver_manifest(manifest).await;
    }
    assert!(!harness.has_pending_localization(CLUSTER_BUFFER_KEY));
    assert_eq!(
        harness.node_clock_assessment(NODE_B).timestamp_step_count,
        1
    );

    let published = drain_published_manifests(&mut harness.events);
    let trigger_segment_id = format!("seg-{NODE_A}-{TRIGGER_FRAME}");
    let released = attempted_localization_results(&published)
        .into_iter()
        .find(|manifest| {
            manifest
                .source_handles
                .iter()
                .any(|handle| handle.segment_id == trigger_segment_id)
        })
        .expect("the parked solve is released");
    let node_b_timing = solve_provenance(released)
        .node_timing
        .iter()
        .find(|row| row.node_id == NODE_B)
        .expect("node b contributes to the cluster solve");
    assert_eq!(
        node_b_timing.timestamp_step_count, 0,
        "trigger-time snapshot"
    );
    assert_eq!(node_b_timing.last_timestamp_step_ns, None);
}

#[tokio::test]
async fn a_residual_over_the_hard_limit_drops_only_that_nodes_cross_node_pairs() {
    const STEP_FRAME: usize = 10;
    const FRAME_COUNT: usize = 40;
    let mut harness = RaceHarness::new(race_config(true)).await;
    // Coverage wait on, so each solve waits for node b's same frame.
    // Node b's clock steps 21 ms at frame 10; its buffer stays anchored at its
    // first packet, so the residual (21 ms) stays over the 10 ms hard limit.
    for frame_index in 0..FRAME_COUNT {
        harness.advance_clock_ms(FIXTURE_FRAME_DURATION_NS / 1_000_000);
        harness.deliver(NODE_A, frame_index).await;
        let mut manifest = harness.cluster_frame_manifest(NODE_B, frame_index).await;
        if frame_index >= STEP_FRAME {
            shift_packet_time(&mut manifest, IN_UPLOAD_STEP_NS);
        }
        harness.deliver_manifest(manifest).await;
    }
    assert!(
        harness
            .node_clock_assessment(NODE_B)
            .residual_over_hard_limit
    );
    assert!(
        !harness
            .node_clock_assessment(NODE_A)
            .residual_over_hard_limit
    );

    let published = drain_published_manifests(&mut harness.events);
    let after_step: Vec<&DspManifest> = attempted_localization_results(&published)
        .into_iter()
        .filter(|manifest| {
            solve_provenance(manifest)
                .node_timing
                .iter()
                .any(|row| row.node_id == NODE_B && row.timestamp_step_count > 0)
        })
        .collect();
    assert!(!after_step.is_empty(), "a solve triggers after the step");
    let mut solves_with_both_nodes_active = 0;
    for manifest in after_step {
        let provenance = solve_provenance(manifest);
        let node_of = |sensor_index: usize| provenance.sensors[sensor_index].node_id.as_str();
        // Never a timeline exclusion inside one node; any other exclusion
        // here is a sensor whose audio had not arrived yet (coverage).
        for pair in &provenance.excluded_pairs {
            if pair.reason == SolveExclusionReason::TimelineResidualOverLimit {
                assert_ne!(node_of(pair.sensor_index_a), node_of(pair.sensor_index_b));
            }
        }
        assert_eq!(cross_node_pair_count(manifest), 0);
        let active = &provenance.active_sensor_indices;
        let active_on = |node_id: &str| active.iter().any(|index| node_of(*index) == node_id);
        if !(active_on(NODE_A) && active_on(NODE_B)) {
            continue;
        }
        solves_with_both_nodes_active += 1;
        for &a in active.iter().filter(|index| node_of(**index) == NODE_A) {
            for &b in active.iter().filter(|index| node_of(**index) == NODE_B) {
                assert!(
                    provenance.excluded_pairs.iter().any(|pair| {
                        pair.reason == SolveExclusionReason::TimelineResidualOverLimit
                            && [pair.sensor_index_a, pair.sensor_index_b].contains(&a)
                            && [pair.sensor_index_a, pair.sensor_index_b].contains(&b)
                    }),
                    "cross-node pair ({a}, {b}) must be excluded for the residual"
                );
            }
        }
        let pair_tdoas = &manifest.localization.as_ref().unwrap().pair_tdoas;
        assert!(
            pair_tdoas
                .iter()
                .any(|pair| node_of(pair.ch_a) == NODE_B && node_of(pair.ch_b) == NODE_B),
            "node b's own channels still solve"
        );
    }
    assert!(solves_with_both_nodes_active > 0);
}
