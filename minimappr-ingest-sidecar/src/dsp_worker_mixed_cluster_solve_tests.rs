//! Localization-overhaul E1 regressions (worker level, black box over the
//! published manifests). Written before the fix; the milestone named in each
//! `#[ignore]` (1: mixed-node routing + geometry push, 2: pair admission)
//! removes it.

use super::dsp_worker_coverage_race_tests::{race_config, RaceHarness};
use super::dsp_worker_solve_manifest_inspection::*;
use super::dsp_worker_test_node_fixtures::*;
use super::dsp_worker_tests::drain_published_manifests;
use super::*;
use minimappr_core::localization_sensor_entry::{CoordinateFrameKind, SolveExclusionReason};

/// Production cadence + cooldown fire at frames 0, 25 and 50 of a 60-frame run.
pub(super) const FRAMES_PER_RUN: usize = 60;
pub(super) const FIRST_SHARED_TRIGGER_FRAME: usize = 25;
pub(super) const SECOND_SHARED_TRIGGER_FRAME: usize = 50;
/// Identical audio in either arrival order must give the same lag to within
/// the correlator's interpolation step.
const ARRIVAL_ORDER_LAG_TOLERANCE_SECONDS: f64 = 0.25 / FIXTURE_SAMPLE_RATE_HZ as f64;
const AUTOMATIC_CLUSTER_BUFFER_KEY: &str = "cluster::auto-gps::audio_main";
const EXPLICIT_CLUSTER_ID: &str = "e1-three-node";
const EXPLICIT_CLUSTER_BUFFER_KEY: &str = "cluster::e1-three-node::audio_main";

/// Runs one planar + point session in the given arrival order and returns the
/// lags of the last six-sensor solve.
async fn planar_and_point_session(
    first: &FixtureNode,
    second: &FixtureNode,
    source: &CoherentPointSource,
) -> HashMap<SensorPairKey, f64> {
    let mut harness = RaceHarness::new(race_config(true)).await;
    for frame_index in 0..FRAMES_PER_RUN {
        harness.advance_clock_ms(FIXTURE_FRAME_DURATION_NS / 1_000_000);
        for node in [first, second] {
            let manifest = fixture_frame_manifest(
                harness.storage_root(),
                node,
                frame_index,
                source,
                &FixtureClusterGeometry::AutomaticGps,
            )
            .await;
            harness.deliver_manifest(manifest).await;
        }
    }
    let published = drain_published_manifests(&mut harness.events);
    let shared_solves: Vec<&DspManifest> = attempted_localization_results(&published)
        .into_iter()
        .filter(|manifest| {
            manifest
                .localization
                .as_ref()
                .is_some_and(|localization| localization.pair_tdoas.len() == 15)
        })
        .collect();
    let order = format!("{} then {}", first.node_id, second.node_id);
    let last_shared_solve = shared_solves.last().unwrap_or_else(|| {
        panic!("{order}: no localization_result solved all six planar + point sensors (15 pairs)")
    });

    let provenance = solve_provenance(last_shared_solve);
    let mut sensor_ids: Vec<&str> = provenance
        .sensors
        .iter()
        .map(|sensor| sensor.sensor_id.as_str())
        .collect();
    sensor_ids.sort_unstable();
    let mut expected_ids: Vec<String> = (0..5)
        .map(|channel| first_or_second_planar(first, second).sensor_id(channel))
        .chain([first_or_second_point(first, second).sensor_id(0)])
        .collect();
    expected_ids.sort_unstable();
    assert_eq!(sensor_ids, expected_ids, "{order}");
    assert_eq!(provenance.active_sensor_indices.len(), 6, "{order}");
    assert_eq!(cross_node_pair_count(last_shared_solve), 5, "{order}");
    let lags = lags_by_sensor_pair(last_shared_solve);
    assert_eq!(lags.len(), 15, "{order}");
    assert_lags_match_geometry(&lags, source, &[first, second]);
    lags
}

fn first_or_second_planar<'a>(first: &'a FixtureNode, second: &'a FixtureNode) -> &'a FixtureNode {
    if first.array_kind == FixtureArrayKind::SirithPlanar {
        first
    } else {
        second
    }
}

fn first_or_second_point<'a>(first: &'a FixtureNode, second: &'a FixtureNode) -> &'a FixtureNode {
    if first.array_kind == FixtureArrayKind::Point {
        first
    } else {
        second
    }
}

#[tokio::test]
async fn planar_and_point_nodes_share_one_fifteen_pair_solve_in_either_arrival_order() {
    let planar = FixtureNode::planar("planar-a", [0.0, 0.0, 1.5], 30.0);
    let point = FixtureNode::point("point-b", [3.0, 1.0, 1.8]);
    let source = CoherentPointSource::new([1.0, 6.0, 2.5], FRAMES_PER_RUN);

    let planar_first_lags = planar_and_point_session(&planar, &point, &source).await;
    let point_first_lags = planar_and_point_session(&point, &planar, &source).await;

    for (pair, planar_first_lag) in &planar_first_lags {
        let point_first_lag = point_first_lags[pair];
        assert!(
            (planar_first_lag - point_first_lag).abs() <= ARRIVAL_ORDER_LAG_TOLERANCE_SECONDS,
            "{pair:?}: {planar_first_lag} s planar-first vs {point_first_lag} s point-first"
        );
    }
}

/// Tetra node plus two point nodes 1 cm apart: the point-point pair is under
/// `localization_min_pair_baseline_m`, both tetra-point node pairs are usable.
struct ThreeNodeCluster {
    tetra: FixtureNode,
    near_point: FixtureNode,
    adjacent_point: FixtureNode,
    source: CoherentPointSource,
    geometry: FixtureClusterGeometry,
}

impl ThreeNodeCluster {
    fn new() -> Self {
        let tetra = FixtureNode::tetra("tetra-a", [0.0, 0.0, 1.5]);
        let near_point = FixtureNode::point("point-b", [4.0, 0.0, 1.5]);
        let adjacent_point = FixtureNode::point("point-c", [4.0, 0.01, 1.5]);
        let geometry = FixtureClusterGeometry::Explicit {
            cluster_id: EXPLICIT_CLUSTER_ID.to_string(),
            sensor_positions: explicit_cluster_sensor_positions(&[
                &tetra,
                &near_point,
                &adjacent_point,
            ]),
        };
        Self {
            tetra,
            near_point,
            adjacent_point,
            source: CoherentPointSource::new([-2.0, 5.0, 2.0], FRAMES_PER_RUN),
            geometry,
        }
    }

    async fn deliver(&self, harness: &mut RaceHarness, node: &FixtureNode, frame_index: usize) {
        let manifest = fixture_frame_manifest(
            harness.storage_root(),
            node,
            frame_index,
            &self.source,
            &self.geometry,
        )
        .await;
        harness.deliver_manifest(manifest).await;
    }

    fn assert_short_baseline_pair_excluded(&self, solve: &DspManifest) {
        let localization = solve.localization.as_ref().expect("localization payload");
        // Checked before provenance: this is the admission behaviour itself.
        assert_eq!(
            localization.pair_tdoas.len(),
            14,
            "15 pairs minus the short baseline"
        );
        let provenance = solve_provenance(solve);
        assert_eq!(provenance.active_sensor_indices.len(), 6);
        let short_pair = (
            self.near_point.sensor_id(0),
            self.adjacent_point.sensor_id(0),
        );
        let lags = lags_by_sensor_pair(solve);
        assert!(
            !lags.contains_key(&short_pair),
            "{short_pair:?} must not be correlated"
        );
        assert_eq!(
            cross_node_pair_count(solve),
            8,
            "two tetra-point node pairs x 4"
        );
        assert_eq!(provenance.excluded_pairs.len(), 1);
        let excluded = &provenance.excluded_pairs[0];
        let mut excluded_ids = [
            provenance.sensors[excluded.sensor_index_a]
                .sensor_id
                .clone(),
            provenance.sensors[excluded.sensor_index_b]
                .sensor_id
                .clone(),
        ];
        excluded_ids.sort_unstable();
        assert_eq!(excluded_ids, [short_pair.0, short_pair.1]);
        assert_eq!(
            excluded.reason,
            SolveExclusionReason::PairBaselineBelowMinimum
        );
        assert_lags_match_geometry(
            &lags,
            &self.source,
            &[&self.tetra, &self.near_point, &self.adjacent_point],
        );
    }
}

#[tokio::test]
#[ignore = "localization overhaul milestone 2: no pair admission yet; every pair is correlated (R7)"]
async fn short_baseline_cross_node_pair_is_excluded_while_both_other_node_pairs_solve() {
    let cluster = ThreeNodeCluster::new();
    let mut harness = RaceHarness::new(race_config(true)).await;
    for frame_index in 0..FRAMES_PER_RUN {
        harness.advance_clock_ms(FIXTURE_FRAME_DURATION_NS / 1_000_000);
        for node in [&cluster.tetra, &cluster.near_point, &cluster.adjacent_point] {
            cluster.deliver(&mut harness, node, frame_index).await;
        }
    }
    let published = drain_published_manifests(&mut harness.events);
    let six_sensor_solves: Vec<&DspManifest> = attempted_localization_results(&published)
        .into_iter()
        .filter(|manifest| {
            manifest.source_handles[0]
                .segment_id
                .ends_with(&format!("-{SECOND_SHARED_TRIGGER_FRAME}"))
        })
        .collect();
    let solve = six_sensor_solves
        .last()
        .expect("a cluster solve on the second trigger frame");
    cluster.assert_short_baseline_pair_excluded(solve);
}

#[tokio::test]
#[ignore = "localization overhaul milestone 2: a released parked solve rebuilds its pairs from coverage alone (R7)"]
async fn late_packet_releasing_a_parked_solve_keeps_the_short_baseline_exclusion() {
    let cluster = ThreeNodeCluster::new();
    let mut harness = RaceHarness::new(race_config(true)).await;
    for frame_index in 0..=FIRST_SHARED_TRIGGER_FRAME {
        harness.advance_clock_ms(FIXTURE_FRAME_DURATION_NS / 1_000_000);
        cluster
            .deliver(&mut harness, &cluster.tetra, frame_index)
            .await;
        cluster
            .deliver(&mut harness, &cluster.near_point, frame_index)
            .await;
        if frame_index > 0 {
            cluster
                .deliver(&mut harness, &cluster.adjacent_point, frame_index - 1)
                .await;
        }
    }
    assert!(
        harness.has_pending_localization(EXPLICIT_CLUSTER_BUFFER_KEY),
        "the trigger frame must park until the late point node covers it"
    );
    drain_published_manifests(&mut harness.events);

    cluster
        .deliver(
            &mut harness,
            &cluster.adjacent_point,
            FIRST_SHARED_TRIGGER_FRAME,
        )
        .await;
    assert!(!harness.has_pending_localization(EXPLICIT_CLUSTER_BUFFER_KEY));
    let published = drain_published_manifests(&mut harness.events);
    let released = attempted_localization_results(&published);
    let solve = released
        .last()
        .expect("the late packet releases the parked solve");
    cluster.assert_short_baseline_pair_excluded(solve);
}

/// Delivers node a's frames on time and node b one frame late, for frames
/// `first..=last`; returns with node b's `last` frame still undelivered.
pub(super) async fn deliver_with_late_node(
    harness: &mut RaceHarness,
    on_time: &FixtureNode,
    late: &FixtureNode,
    source: &CoherentPointSource,
    frames: std::ops::RangeInclusive<usize>,
) {
    for frame_index in frames {
        harness.advance_clock_ms(FIXTURE_FRAME_DURATION_NS / 1_000_000);
        let on_time_manifest = fixture_frame_manifest(
            harness.storage_root(),
            on_time,
            frame_index,
            source,
            &FixtureClusterGeometry::AutomaticGps,
        )
        .await;
        harness.deliver_manifest(on_time_manifest).await;
        if frame_index > 0 {
            deliver_late_frame(harness, late, source, frame_index - 1).await;
        }
    }
}

async fn deliver_late_frame(
    harness: &mut RaceHarness,
    late: &FixtureNode,
    source: &CoherentPointSource,
    frame_index: usize,
) {
    let manifest = fixture_frame_manifest(
        harness.storage_root(),
        late,
        frame_index,
        source,
        &FixtureClusterGeometry::AutomaticGps,
    )
    .await;
    harness.deliver_manifest(manifest).await;
}

#[tokio::test]
async fn geometry_push_while_a_solve_is_parked_applies_only_to_later_solves() {
    const SITE_FRAME_ID: &str = "e1-site-frame";
    const GEOMETRY_PUSH_REVISION: u64 = 7;
    const OVERRIDE_SHIFT_EAST_M: f64 = 5.0;
    let tetra_a = FixtureNode::tetra("tetra-a", [0.0, 0.0, 0.0]);
    let tetra_b = FixtureNode::tetra("tetra-b", [20.0, 0.0, 0.0]);
    let tetra_b_overridden =
        FixtureNode::tetra("tetra-b", [20.0 + OVERRIDE_SHIFT_EAST_M, 0.0, 0.0]);
    let source = CoherentPointSource::new([10.0, 8.0, 2.0], FRAMES_PER_RUN);
    let mut harness = RaceHarness::new(race_config(true)).await;

    deliver_with_late_node(
        &mut harness,
        &tetra_a,
        &tetra_b,
        &source,
        0..=FIRST_SHARED_TRIGGER_FRAME,
    )
    .await;
    assert!(harness.has_pending_localization(AUTOMATIC_CLUSTER_BUFFER_KEY));
    drain_published_manifests(&mut harness.events);

    // Operator moves node b while the trigger-frame solve is parked.
    harness
        .state
        .write()
        .await
        .site_geometry
        .apply_push(fixture_site_geometry_push(
            SITE_FRAME_ID,
            GEOMETRY_PUSH_REVISION,
            &[&tetra_a, &tetra_b_overridden],
        ))
        .expect("fixture push is valid");
    deliver_late_frame(&mut harness, &tetra_b, &source, FIRST_SHARED_TRIGGER_FRAME).await;
    assert!(!harness.has_pending_localization(AUTOMATIC_CLUSTER_BUFFER_KEY));
    let published = drain_published_manifests(&mut harness.events);
    let parked_solve = *attempted_localization_results(&published)
        .last()
        .expect("node b's late packet releases the parked solve");
    let parked = solve_provenance(parked_solve);
    assert_eq!(
        parked.geometry_push_revision, None,
        "snapshot taken at trigger"
    );
    let pre_push_b_ch0 = tetra_b.sensor_world_positions_m()[0].1;
    assert_position_near(
        sensor_position_m(parked, &tetra_b.sensor_id(0)),
        pre_push_b_ch0,
        "parked solve keeps the trigger-time geometry",
    );

    deliver_with_late_node(
        &mut harness,
        &tetra_a,
        &tetra_b,
        &source,
        FIRST_SHARED_TRIGGER_FRAME + 1..=SECOND_SHARED_TRIGGER_FRAME,
    )
    .await;
    deliver_late_frame(&mut harness, &tetra_b, &source, SECOND_SHARED_TRIGGER_FRAME).await;
    let published = drain_published_manifests(&mut harness.events);
    let later_solve = *attempted_localization_results(&published)
        .last()
        .expect("a solve after the push");
    let later = solve_provenance(later_solve);
    assert_eq!(later.geometry_push_revision, Some(GEOMETRY_PUSH_REVISION));
    assert_eq!(
        later.coordinate_frame.frame_id.as_deref(),
        Some(SITE_FRAME_ID)
    );
    assert_eq!(later.coordinate_frame.kind, CoordinateFrameKind::SiteEnu);
    assert!(later.layout_revision > parked.layout_revision);
    assert_position_near(
        sensor_position_m(later, &tetra_b.sensor_id(0)),
        tetra_b_overridden.sensor_world_positions_m()[0].1,
        "later solve uses the pushed placement",
    );
}

#[tokio::test]
async fn gps_move_while_a_solve_is_parked_applies_only_to_later_solves() {
    const MOVE_EAST_M: f64 = 5.0;
    let tetra_a = FixtureNode::tetra("tetra-a", [0.0, 0.0, 0.0]);
    let tetra_b = FixtureNode::tetra("tetra-b", [20.0, 0.0, 0.0]);
    let tetra_b_moved = FixtureNode::tetra("tetra-b", [20.0 + MOVE_EAST_M, 0.0, 0.0]);
    let source = CoherentPointSource::new([10.0, 8.0, 2.0], FRAMES_PER_RUN);
    let mut harness = RaceHarness::new(race_config(true)).await;

    deliver_with_late_node(
        &mut harness,
        &tetra_a,
        &tetra_b,
        &source,
        0..=FIRST_SHARED_TRIGGER_FRAME,
    )
    .await;
    assert!(harness.has_pending_localization(AUTOMATIC_CLUSTER_BUFFER_KEY));
    drain_published_manifests(&mut harness.events);

    // Node b's releasing packet already reports the new GPS fix.
    deliver_late_frame(
        &mut harness,
        &tetra_b_moved,
        &source,
        FIRST_SHARED_TRIGGER_FRAME,
    )
    .await;
    assert!(!harness.has_pending_localization(AUTOMATIC_CLUSTER_BUFFER_KEY));
    let published = drain_published_manifests(&mut harness.events);
    let parked_solve = *attempted_localization_results(&published)
        .last()
        .expect("node b's late packet releases the parked solve");
    let parked = solve_provenance(parked_solve);
    assert_eq!(parked.active_sensor_indices.len(), 8);
    assert_position_near(
        sensor_position_m(parked, &tetra_b.sensor_id(0)),
        tetra_b.sensor_world_positions_m()[0].1,
        "parked solve keeps the trigger-time geometry",
    );

    deliver_with_late_node(
        &mut harness,
        &tetra_a,
        &tetra_b_moved,
        &source,
        FIRST_SHARED_TRIGGER_FRAME + 1..=SECOND_SHARED_TRIGGER_FRAME,
    )
    .await;
    deliver_late_frame(
        &mut harness,
        &tetra_b_moved,
        &source,
        SECOND_SHARED_TRIGGER_FRAME,
    )
    .await;
    let published = drain_published_manifests(&mut harness.events);
    let later = solve_provenance(
        attempted_localization_results(&published)
            .last()
            .expect("a solve after the move"),
    );
    assert!(later.layout_revision > parked.layout_revision);
    assert_position_near(
        sensor_position_m(later, &tetra_b.sensor_id(0)),
        tetra_b_moved.sensor_world_positions_m()[0].1,
        "later solve uses the moved position",
    );
}
