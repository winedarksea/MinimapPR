//! Worker-level timeline discipline: a stepped node is re-anchored and its
//! cross-node pairs come back with the step's widened uncertainty (then the
//! class default after the holdoff); a drifting node keeps its cross-node
//! pairs because its timeline is slewed before the residual reaches the limit.

use super::dsp_worker_clock_evidence_tests::shift_packet_time;
use super::dsp_worker_coverage_race_tests::{
    race_config, RaceHarness, FRAME_DURATION_NS, NODE_A, NODE_B,
};
use super::dsp_worker_solve_manifest_inspection::{
    attempted_localization_results, cross_node_pair_count, solve_provenance,
};
use super::dsp_worker_tests::drain_published_manifests;
use super::*;
use minimappr_core::localization_sensor_entry::{NodeTimingEvidence, SolveExclusionReason};

const STEP_NS: u64 = 21_000_000;
const STEP_FRAME: usize = 10;
const FRAME_COUNT: usize = 50;
/// Short enough to clear inside the run (about 7 frames of 32 ms).
const SHORT_HOLDOFF_SECONDS: f64 = 0.2;
/// 0.5 ms of packet-clock drift per 32 ms frame: far under the 2 ms step
/// threshold per frame, and past the 10 ms limit after 20 frames, the same
/// residual a 20 ppm crystal builds in about 8 minutes, time-compressed.
const DRIFT_NS_PER_FRAME: u64 = 500_000;

/// Delivers frames `0..FRAME_COUNT` of both nodes, node b's packet times
/// offset by `node_b_offset_ns(frame)`; returns the published manifests and
/// whether node b's hard-limit flag was ever set along the way.
async fn run_two_nodes(
    harness: &mut RaceHarness,
    node_b_offset_ns: impl Fn(usize) -> u64,
) -> (Vec<DspManifest>, bool) {
    let mut node_b_ever_over_hard_limit = false;
    let mut published = Vec::new();
    for frame_index in 0..FRAME_COUNT {
        harness.advance_clock_ms(FRAME_DURATION_NS / 1_000_000);
        harness.deliver(NODE_A, frame_index).await;
        let mut manifest = harness.cluster_frame_manifest(NODE_B, frame_index).await;
        let offset_ns = node_b_offset_ns(frame_index);
        if offset_ns > 0 {
            shift_packet_time(&mut manifest, offset_ns);
        }
        harness.deliver_manifest(manifest).await;
        node_b_ever_over_hard_limit |= harness
            .node_clock_assessment(NODE_B)
            .residual_over_hard_limit;
        published.extend(drain_published_manifests(&mut harness.events));
    }
    (published, node_b_ever_over_hard_limit)
}

/// Solves with both nodes active, triggered after `predicate` holds for node b's row.
fn two_node_solves<'a>(
    published: &'a [DspManifest],
    predicate: impl Fn(&NodeTimingEvidence) -> bool,
) -> Vec<&'a DspManifest> {
    attempted_localization_results(published)
        .into_iter()
        .filter(|manifest| {
            let provenance = solve_provenance(manifest);
            let node_b_row = provenance
                .node_timing
                .iter()
                .find(|row| row.node_id == NODE_B);
            let active_nodes: std::collections::BTreeSet<&str> = provenance
                .active_sensor_indices
                .iter()
                .map(|index| provenance.sensors[*index].node_id.as_str())
                .collect();
            active_nodes.len() == 2 && node_b_row.is_some_and(&predicate)
        })
        .collect()
}

fn assert_no_timeline_exclusions(manifest: &DspManifest) {
    let provenance = solve_provenance(manifest);
    assert!(
        provenance
            .excluded_pairs
            .iter()
            .all(|pair| pair.reason != SolveExclusionReason::TimelineResidualOverLimit),
        "{:?}",
        provenance.excluded_pairs
    );
    assert!(
        cross_node_pair_count(manifest) > 0,
        "cross-node pairs admitted"
    );
}

#[tokio::test]
async fn a_stepped_node_is_reanchored_and_its_cross_node_pairs_return_widened() {
    let mut harness = RaceHarness::new(race_config(true)).await;
    let (published, ever_over) =
        run_two_nodes(
            &mut harness,
            |frame| {
                if frame >= STEP_FRAME {
                    STEP_NS
                } else {
                    0
                }
            },
        )
        .await;
    assert!(!ever_over, "the hard-limit gate never saw the step");
    let node_b = harness.node_clock_assessment(NODE_B);
    assert_eq!(node_b.timestamp_step_count, 1);
    assert_eq!(node_b.timeline_step_reanchor_count, 1);
    assert_eq!(node_b.timeline_drift_slew_count, 0);
    assert!(node_b.timeline_residual_seconds.unwrap().abs() < 1e-6);
    // Default 30 s holdoff: still carrying the step's uncertainty.
    let step_seconds = STEP_NS as f64 / 1e9;
    assert!((node_b.timing_uncertainty_seconds - step_seconds).abs() < 1e-9);
    let state = harness.state.read().await;
    assert_eq!(state.total_timeline_step_reanchors, 1);
    assert_eq!(state.total_buffer_reanchors, 0, "no buffer was cleared");
    drop(state);

    let reanchored = two_node_solves(&published, |row| row.timeline_step_reanchor_count == 1);
    assert!(
        !reanchored.is_empty(),
        "a cluster solve runs after the re-anchor"
    );
    for manifest in reanchored {
        assert_no_timeline_exclusions(manifest);
        let provenance = solve_provenance(manifest);
        for pair in &manifest.localization.as_ref().unwrap().pair_tdoas {
            let crosses_node_b = pair.is_cross_node == Some(true)
                && [pair.ch_a, pair.ch_b]
                    .iter()
                    .any(|channel| provenance.sensors[*channel].node_id == NODE_B);
            if crosses_node_b {
                let uncertainty = f64::from(pair.timing_uncertainty_seconds.unwrap());
                assert!(
                    uncertainty >= step_seconds * 0.999,
                    "pair widened: {uncertainty}"
                );
            }
        }
    }
}

#[tokio::test]
async fn a_stepped_node_returns_to_its_class_uncertainty_after_the_holdoff() {
    let mut config = race_config(true);
    config
        .localization_solve
        .localization_timestamp_step_recovery_holdoff_seconds = SHORT_HOLDOFF_SECONDS;
    let mut harness = RaceHarness::new(config).await;
    let (published, ever_over) =
        run_two_nodes(
            &mut harness,
            |frame| {
                if frame >= STEP_FRAME {
                    STEP_NS
                } else {
                    0
                }
            },
        )
        .await;
    assert!(!ever_over);
    let node_a = harness.node_clock_assessment(NODE_A);
    let node_b = harness.node_clock_assessment(NODE_B);
    assert_eq!(node_b.timeline_step_reanchor_count, 1);
    assert_eq!(
        node_b.timing_uncertainty_seconds,
        node_a.timing_uncertainty_seconds
    );
    let recovered = two_node_solves(&published, |row| {
        row.timeline_step_reanchor_count == 1
            && row.clock_offset_std_s == Some(node_a.timing_uncertainty_seconds)
    });
    assert!(!recovered.is_empty(), "a cluster solve after the holdoff");
    recovered
        .into_iter()
        .for_each(assert_no_timeline_exclusions);
}

#[tokio::test]
async fn a_drifting_node_keeps_its_cross_node_pairs() {
    let drift = |frame: usize| frame as u64 * DRIFT_NS_PER_FRAME;
    // The same run with the discipline off loses node b to the gate.
    let mut undisciplined_config = race_config(true);
    undisciplined_config
        .localization_solve
        .localization_timeline_drift_slew_fraction = 0.0;
    let mut undisciplined = RaceHarness::new(undisciplined_config).await;
    let (_, undisciplined_ever_over) = run_two_nodes(&mut undisciplined, drift).await;
    assert!(
        undisciplined_ever_over,
        "without the slew the drift reaches the limit"
    );

    let mut harness = RaceHarness::new(race_config(true)).await;
    let (published, ever_over) = run_two_nodes(&mut harness, drift).await;
    assert!(!ever_over, "slewed before the residual reached the limit");
    let node_b = harness.node_clock_assessment(NODE_B);
    assert_eq!(node_b.timestamp_step_count, 0, "drift is never a step");
    assert_eq!(node_b.timeline_step_reanchor_count, 0);
    assert!(
        node_b.timeline_drift_slew_count >= 2,
        "{}",
        node_b.timeline_drift_slew_count
    );
    assert_eq!(
        u64::from(node_b.timeline_drift_slew_count),
        harness.state.read().await.total_timeline_drift_slews
    );
    let accumulated_ns = drift(FRAME_COUNT - 1);
    assert!(
        accumulated_ns > 2 * 10_000_000,
        "the run drifts past twice the limit"
    );

    let slewed_solves = two_node_solves(&published, |row| row.timeline_drift_slew_count >= 1);
    assert!(!slewed_solves.is_empty());
    slewed_solves
        .into_iter()
        .for_each(assert_no_timeline_exclusions);
}
