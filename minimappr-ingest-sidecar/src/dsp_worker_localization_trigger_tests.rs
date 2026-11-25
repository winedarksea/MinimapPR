//! Worker-level trigger and event-span tests (localization overhaul R4): the
//! noise-floor-relative trigger opens solves on a quiet site the absolute gate
//! misses, the cooldown starts only when a solve is dispatched, and a
//! long-baseline cluster's event solve reads the cross-node span.

use super::dsp_worker_coverage_race_tests::RaceHarness;
use super::dsp_worker_solve_manifest_inspection::*;
use super::dsp_worker_test_node_fixtures::*;
use super::dsp_worker_tests::drain_published_manifests;
use super::*;
use minimappr_core::localization_sensor_entry::SolveProvenance;

const QUIET_SITE_NOISE_RMS: f32 = 3.0e-4;
/// ~18 dB over the floor, still well under the 0.015 absolute gate.
const FAINT_CALL_RMS: f32 = 8.0 * QUIET_SITE_NOISE_RMS;
/// Uniform noise peaking at the fixture source amplitude has RMS peak/sqrt(3).
const FIXTURE_SOURCE_RMS: f32 = 0.3 / 1.732_050_8;
/// ~1.9 s of ambience: past the relative trigger's 1 s warmup.
const QUIET_FRAMES: usize = 60;
const CALL_FRAMES: usize = 8;
const NOISE_SEED: u32 = 0x0a11_ce55;

fn quiet_site_config(relative_trigger_enabled: bool) -> DspWorkerConfig {
    let mut config = DspWorkerConfig {
        classifier_render_min_interval_seconds: 1.0e6,
        ..DspWorkerConfig::default()
    };
    config
        .localization_solve
        .localization_trigger_noise_floor_relative_enabled = relative_trigger_enabled;
    config
}

/// Independent per-channel ambience, so only the call is coherent.
fn ambience(frame_index: usize, channel_count: usize) -> Vec<Vec<f32>> {
    let uniform_to_unit_rms = 3f32.sqrt();
    (0..channel_count)
        .map(|channel| {
            let mut state = NOISE_SEED
                .wrapping_add((frame_index * 131 + channel * 7919) as u32)
                .wrapping_mul(2_654_435_761);
            (0..FIXTURE_FRAME_SAMPLES)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (state as i32 as f32) / (i32::MAX as f32)
                        * uniform_to_unit_rms
                        * QUIET_SITE_NOISE_RMS
                })
                .collect()
        })
        .collect()
}

fn event_attempt_kinds(published: &[DspManifest]) -> Vec<(SolveAttemptKind, String)> {
    attempted_localization_results(published)
        .into_iter()
        .map(|manifest| solve_provenance(manifest))
        .filter(|provenance| {
            matches!(
                provenance.attempt_kind,
                SolveAttemptKind::EventTriggerAbsoluteRms
                    | SolveAttemptKind::EventTriggerNoiseFloorRelative
            )
        })
        .map(|provenance| (provenance.attempt_kind, provenance.solve_id.clone()))
        .collect()
}

/// Ambience, then a faint coherent call starting `call_onset_sample` into its
/// first frame; returns (published during the ambience, published during the call).
async fn quiet_site_session(
    config: DspWorkerConfig,
    call_onset_sample: usize,
) -> (Vec<DspManifest>, Vec<DspManifest>) {
    let node = FixtureNode::tetra("tetra-quiet", [0.0, 0.0, 1.5]);
    let call = CoherentPointSource::new([6.0, 3.0, 2.0], QUIET_FRAMES + CALL_FRAMES);
    let call_gain = FAINT_CALL_RMS / FIXTURE_SOURCE_RMS;
    let mut harness = RaceHarness::new(config).await;
    let mut during_ambience = Vec::new();
    for frame_index in 0..QUIET_FRAMES + CALL_FRAMES {
        if frame_index == QUIET_FRAMES {
            during_ambience = drain_published_manifests(&mut harness.events);
        }
        let mut channels = ambience(frame_index, node.channel_count());
        if frame_index >= QUIET_FRAMES {
            for (channel, call_channel) in channels
                .iter_mut()
                .zip(call.frame_channels(&node, frame_index))
            {
                let silent_prefix = if frame_index == QUIET_FRAMES {
                    call_onset_sample
                } else {
                    0
                };
                for (sample, call_sample) in
                    channel.iter_mut().zip(call_channel).skip(silent_prefix)
                {
                    *sample += call_gain * call_sample;
                }
            }
        }
        harness.advance_clock_ms(FIXTURE_FRAME_DURATION_NS / 1_000_000);
        let manifest = fixture_frame_manifest_from_channels(
            harness.storage_root(),
            &node,
            frame_index,
            &channels,
            &FixtureClusterGeometry::AutomaticGps,
        )
        .await;
        harness.deliver_manifest(manifest).await;
    }
    let during_call = drain_published_manifests(&mut harness.events);
    (during_ambience, during_call)
}

#[tokio::test]
async fn relative_trigger_opens_a_solve_on_a_faint_call_below_the_absolute_gate() {
    let (during_ambience, during_call) = quiet_site_session(quiet_site_config(true), 0).await;
    assert_eq!(event_attempt_kinds(&during_ambience), vec![]);

    let call_solves: Vec<&DspManifest> = attempted_localization_results(&during_call)
        .into_iter()
        .filter(|manifest| {
            solve_provenance(manifest).attempt_kind
                == SolveAttemptKind::EventTriggerNoiseFloorRelative
        })
        .collect();
    assert_eq!(call_solves.len(), 1, "one solve per cooldown");
    let localization = call_solves[0].localization.as_ref().unwrap();
    // The solver floor does not reject what the trigger admitted.
    assert_eq!(localization.resolved_algorithm, "srp_phat");
}

#[tokio::test]
async fn relative_trigger_solves_the_window_after_the_onset_not_the_noise_before_it() {
    // The call reaches the array 7 ms before its first frame ends. The window
    // ending at that frame would hold 7 ms of call in 32 ms; the event solve
    // must read the window that starts at the onset (E5 100 m single tetra).
    const ONSET_SAMPLE: usize = 400;
    let config = quiet_site_config(true);
    let window_ns = (config.window_seconds * 1.0e9).round() as i64;
    let (_, during_call) = quiet_site_session(config, ONSET_SAMPLE).await;
    let event_solves: Vec<&SolveProvenance> = attempted_localization_results(&during_call)
        .into_iter()
        .map(|manifest| solve_provenance(manifest))
        .filter(|provenance| {
            provenance.attempt_kind == SolveAttemptKind::EventTriggerNoiseFloorRelative
        })
        .collect();
    assert_eq!(event_solves.len(), 1, "one solve per cooldown");
    let onset_ns = (FIXTURE_STREAM_START_NS
        + QUIET_FRAMES as u64 * FIXTURE_FRAME_DURATION_NS
        + ONSET_SAMPLE as u64 * 1_000_000_000 / u64::from(FIXTURE_SAMPLE_RATE_HZ))
        as i64;
    let window_start_ns = event_solves[0].window_center_time_ns - window_ns / 2;
    let onset_block_ns = 5_000_000;
    assert!(
        (window_start_ns - onset_ns).abs() <= onset_block_ns,
        "window starts {} ns from the onset",
        window_start_ns - onset_ns
    );
}

#[tokio::test]
async fn absolute_only_trigger_misses_the_same_faint_call() {
    let (_, during_call) = quiet_site_session(quiet_site_config(false), 0).await;
    assert_eq!(event_attempt_kinds(&during_call), vec![]);
}

#[tokio::test]
async fn a_trigger_the_cadence_suppresses_does_not_start_the_cooldown() {
    // Cooldown 0.8 s under a 1 s cadence: the frame-25 trigger (0.8 s) is
    // suppressed by the cadence. Had it started the cooldown, the next solve
    // would wait to frame 50 (1.6 s); it fires at frame 32 (first past 1 s).
    const CADENCE_MS: u64 = 1_000;
    const FRAME_COUNT: usize = 40;
    let config = DspWorkerConfig {
        classifier_render_min_interval_seconds: 1.0e6,
        localization_cadence_ms: CADENCE_MS,
        ..DspWorkerConfig::default()
    };
    let node = FixtureNode::tetra("tetra-loud", [0.0, 0.0, 1.5]);
    let source = CoherentPointSource::new([6.0, 3.0, 2.0], FRAME_COUNT);
    let mut harness = RaceHarness::new(config).await;
    for frame_index in 0..FRAME_COUNT {
        harness.advance_clock_ms(FIXTURE_FRAME_DURATION_NS / 1_000_000);
        let manifest = fixture_frame_manifest(
            harness.storage_root(),
            &node,
            frame_index,
            &source,
            &FixtureClusterGeometry::AutomaticGps,
        )
        .await;
        harness.deliver_manifest(manifest).await;
    }
    let published = drain_published_manifests(&mut harness.events);
    let event_solve_frames: Vec<String> = attempted_localization_results(&published)
        .into_iter()
        .filter(|manifest| {
            solve_provenance(manifest).attempt_kind == SolveAttemptKind::EventTriggerAbsoluteRms
        })
        .map(|manifest| manifest.source_handles[0].segment_id.clone())
        .collect();
    assert_eq!(
        event_solve_frames,
        vec![
            "seg-tetra-loud-0".to_string(),
            "seg-tetra-loud-32".to_string()
        ]
    );
}

#[tokio::test]
async fn long_baseline_event_solve_reads_the_cross_node_span_and_recovers_its_lags() {
    const FRAME_COUNT: usize = 60;
    const BASELINE_M: f64 = 100.0;
    let config = DspWorkerConfig {
        localization_rms_gate: 0.0,
        ..quiet_site_config(true)
    };
    let node_local_window_seconds = config.window_seconds;
    let near = FixtureNode::tetra("tetra-near", [0.0, 0.0, 1.5]);
    let far = FixtureNode::tetra("tetra-far", [BASELINE_M, 0.0, 1.5]);
    let source = CoherentPointSource::new([3.0, 4.0, 2.0], FRAME_COUNT);
    let mut harness = RaceHarness::new(config).await;
    for frame_index in 0..FRAME_COUNT {
        harness.advance_clock_ms(FIXTURE_FRAME_DURATION_NS / 1_000_000);
        for node in [&near, &far] {
            let manifest = fixture_frame_manifest(
                harness.storage_root(),
                node,
                frame_index,
                &source,
                &FixtureClusterGeometry::AutomaticGps,
            )
            .await;
            harness.deliver_manifest(manifest).await;
        }
    }
    let published = drain_published_manifests(&mut harness.events);
    let cross_node_event_solve = attempted_localization_results(&published)
        .into_iter()
        .filter(|manifest| {
            let provenance = solve_provenance(manifest);
            provenance.attempt_kind == SolveAttemptKind::EventTriggerAbsoluteRms
                && provenance.active_sensor_indices.len() == 8
        })
        .last()
        .expect("an event solve over both nodes");

    let provenance = solve_provenance(cross_node_event_solve);
    let sound_speed_mps = f64::from(
        cross_node_event_solve
            .localization
            .as_ref()
            .unwrap()
            .sound_speed_mps,
    );
    let max_cross_node_distance_m = provenance
        .sensors
        .iter()
        .flat_map(|a| provenance.sensors.iter().map(move |b| (a, b)))
        .filter(|(a, b)| a.node_id != b.node_id)
        .map(|(a, b)| {
            (0..3)
                .map(|axis| (a.position_m[axis] - b.position_m[axis]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .fold(0.0, f64::max);
    let expected_span_seconds = 2.0 * max_cross_node_distance_m / sound_speed_mps;
    assert!(provenance.window_seconds > 10.0 * node_local_window_seconds);
    assert!(
        (provenance.window_seconds - expected_span_seconds).abs() < 1e-6,
        "span {} s, expected {expected_span_seconds} s",
        provenance.window_seconds
    );
    // The ~0.27 s cross-node lag is far outside a node-local window; the span
    // lets GCC-PHAT recover every pair at its geometric lag.
    assert_eq!(cross_node_pair_count(cross_node_event_solve), 16);
    assert_lags_match_geometry(
        &lags_by_sensor_pair(cross_node_event_solve),
        &source,
        &[&near, &far],
    );
}
