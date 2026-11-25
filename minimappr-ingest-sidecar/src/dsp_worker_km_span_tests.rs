//! Worker-level kilometre-baseline tests: the geometry-derived event span,
//! coverage cap and same-event window let a 1.5 km two-node cluster solve
//! with its cross-node pairs at their true (multi-second) lags, with no
//! setting changed; an explicit cap still restores the old bound.

use super::dsp_worker_coverage_race_tests::RaceHarness;
use super::dsp_worker_solve_manifest_inspection::*;
use super::dsp_worker_test_node_fixtures::*;
use super::dsp_worker_tests::drain_published_manifests;
use super::*;
use minimappr_core::localization_sensor_entry::SolveProvenance;
use crate::localization_coverage_wait::{
    STOP_REASON_ALL_COVERED, STOP_REASON_SUPERSEDED, STOP_REASON_TOTAL_CAP,
};

const BASELINE_M: f64 = 1_500.0;
/// Two 4-channel tetras.
const CLUSTER_CHANNEL_COUNT: usize = 8;
/// ~10.6 s of audio. A span reaches ~4.4 s either side of its trigger, so
/// only triggers past the first ~4.4 s (once the buffers hold that much
/// history) are covered; their far ends are delivered before the run ends.
const FRAME_COUNT: usize = 330;
/// The far node hears the source ~4.4 s after the near one.
const SOURCE_PRE_ROLL_SAMPLES: usize = 6 * FIXTURE_SAMPLE_RATE_HZ as usize;
const MILLISECONDS_PER_SECOND: f64 = 1_000.0;

fn km_config() -> DspWorkerConfig {
    DspWorkerConfig {
        classifier_render_min_interval_seconds: 1.0e6,
        // Every frame passes the gate: the source plays throughout.
        localization_rms_gate: 0.0,
        ..DspWorkerConfig::default()
    }
}

struct KmRun {
    harness: RaceHarness,
    published: Vec<DspManifest>,
    near: FixtureNode,
    far: FixtureNode,
    source: CoherentPointSource,
}

async fn run_km_cluster(config: DspWorkerConfig) -> KmRun {
    let near = FixtureNode::tetra("tetra-near", [0.0, 0.0, 1.5]);
    let far = FixtureNode::tetra("tetra-far", [BASELINE_M, 0.0, 1.5]);
    let source = CoherentPointSource::with_pre_roll_samples(
        [3.0, 4.0, 2.0],
        FRAME_COUNT,
        SOURCE_PRE_ROLL_SAMPLES,
    );
    let mut harness = RaceHarness::new(config).await;
    // Both nodes stream in real time on the manual clock.
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
    KmRun {
        harness,
        published,
        near,
        far,
        source,
    }
}

fn cross_node_event_solves(published: &[DspManifest]) -> Vec<&DspManifest> {
    attempted_localization_results(published)
        .into_iter()
        .filter(|manifest| {
            let provenance = solve_provenance(manifest);
            provenance.attempt_kind == SolveAttemptKind::EventTriggerAbsoluteRms
                && provenance.active_sensor_indices.len() == CLUSTER_CHANNEL_COUNT
        })
        .collect()
}

fn max_cross_node_lag_seconds(manifest: &DspManifest) -> f64 {
    let provenance = solve_provenance(manifest);
    let sound_speed_mps = f64::from(manifest.localization.as_ref().unwrap().sound_speed_mps);
    provenance
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
        .fold(0.0, f64::max)
        / sound_speed_mps
}

#[tokio::test]
async fn a_kilometre_cluster_solves_across_nodes_with_default_settings() {
    let run = run_km_cluster(km_config()).await;
    let solves = cross_node_event_solves(&run.published);
    let first = *solves.first().expect("a cross-node event solve");
    let lag_seconds = max_cross_node_lag_seconds(first);
    assert!(lag_seconds > 4.0, "lag {lag_seconds}");

    // The span reaches the far node's whole lag on either side of the trigger.
    let provenance = solve_provenance(first);
    assert!(
        (provenance.window_seconds - 2.0 * lag_seconds).abs() < 1e-6,
        "span {} s, lag {lag_seconds} s",
        provenance.window_seconds
    );
    // Every cross-node pair at its true multi-second lag.
    assert_eq!(cross_node_pair_count(first), 16);
    assert_lags_match_geometry(
        &lags_by_sensor_pair(first),
        &run.source,
        &[&run.near, &run.far],
    );

    // Each solve waited for the far node and was released fully covered:
    // never cut short by a 1 s cap or superseded by the next trigger.
    assert_eq!(
        run.harness.stop_reason_count(STOP_REASON_TOTAL_CAP).await,
        0
    );
    assert_eq!(
        run.harness.stop_reason_count(STOP_REASON_SUPERSEDED).await,
        0
    );
    assert!(
        run.harness.stop_reason_count(STOP_REASON_ALL_COVERED).await >= solves.len() as u64
    );
    assert!(run.harness.max_wait_ms().await > (lag_seconds - 0.5) * MILLISECONDS_PER_SECOND);

    // The far node's onsets of the same sound never opened their own solve:
    // every event solve was triggered on the near node, one per cooldown.
    for solve in &solves {
        assert!(
            solve.source_handles[0]
                .segment_id
                .starts_with("seg-tetra-near-"),
            "{}",
            solve.source_handles[0].segment_id
        );
    }
    assert!(solves.len() >= 2, "{} solves", solves.len());
}

/// The pre-derivation setting: the span is held to twice the cap, and the
/// wait (no longer than the far node's lag) gives up before the far node's
/// audio arrives, so the solve runs on the near node alone.
#[tokio::test]
async fn an_explicit_coverage_cap_restores_the_old_bound() {
    const EXPLICIT_CAP_SECONDS: f64 = 1.0;
    let run = run_km_cluster(DspWorkerConfig {
        localization_coverage_total_cap_seconds: EXPLICIT_CAP_SECONDS,
        ..km_config()
    })
    .await;
    let event_solves: Vec<&SolveProvenance> = attempted_localization_results(&run.published)
        .into_iter()
        .map(solve_provenance)
        .filter(|provenance| {
            provenance.attempt_kind == SolveAttemptKind::EventTriggerAbsoluteRms
                && provenance.window_seconds > km_config().window_seconds
        })
        .collect();
    assert!(!event_solves.is_empty());
    for provenance in &event_solves {
        assert!((provenance.window_seconds - 2.0 * EXPLICIT_CAP_SECONDS).abs() < 1e-6);
        assert!(provenance.active_sensor_indices.len() < CLUSTER_CHANNEL_COUNT);
    }
    assert!(cross_node_event_solves(&run.published).is_empty());
}

/// The parked-solve bound: with room for one, each new trigger (one per
/// cooldown) dispatches the solve still waiting seconds for the far node.
#[tokio::test]
async fn the_pending_bound_supersedes_the_oldest_parked_solve() {
    let mut config = km_config();
    config
        .localization_solve
        .localization_coverage_max_pending_per_buffer = 1;
    let run = run_km_cluster(config).await;
    assert!(run.harness.stop_reason_count(STOP_REASON_SUPERSEDED).await > 0);
    assert!(cross_node_event_solves(&run.published).is_empty());
}
