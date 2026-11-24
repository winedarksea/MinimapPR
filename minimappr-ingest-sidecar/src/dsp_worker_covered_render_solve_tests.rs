//! A scheduled classifier render reports the event-triggered solve it
//! covers (`dsp_covered_event_solve`), worker level over published manifests.

use super::dsp_worker_coverage_race_tests::{race_config, RaceHarness};
use super::dsp_worker_mixed_cluster_solve_tests::{
    deliver_with_late_node, FIRST_SHARED_TRIGGER_FRAME, FRAMES_PER_RUN, SECOND_SHARED_TRIGGER_FRAME,
};
use super::dsp_worker_solve_manifest_inspection::*;
use super::dsp_worker_test_node_fixtures::*;
use super::dsp_worker_tests::drain_published_manifests;
use super::*;
use minimappr_core::localization_sensor_entry::SolveAttemptKind;

#[tokio::test]
async fn a_scheduled_render_repeats_the_event_triggered_solve_it_covers() {
    // Renders every ~16 frames; event triggers at frames 0, 25 and 50.
    const RENDER_INTERVAL_SECONDS: f64 = 0.5;
    let planar = FixtureNode::planar("planar-a", [0.0, 0.0, 1.5], 30.0);
    let point = FixtureNode::point("point-b", [3.0, 1.0, 1.8]);
    let source = CoherentPointSource::new([1.0, 6.0, 2.5], FRAMES_PER_RUN);
    let mut harness = RaceHarness::new(DspWorkerConfig {
        classifier_render_min_interval_seconds: RENDER_INTERVAL_SECONDS,
        ..race_config(true)
    })
    .await;
    for frame_index in 0..FRAMES_PER_RUN {
        harness.advance_clock_ms(FIXTURE_FRAME_DURATION_NS / 1_000_000);
        for node in [&planar, &point] {
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
    let event_solves: HashMap<&str, &DspManifest> = published
        .iter()
        .filter(|manifest| manifest.manifest_type == "localization_result")
        .filter_map(|manifest| {
            let provenance = manifest.localization.as_ref()?.solve_provenance.as_ref()?;
            (provenance.attempt_kind == SolveAttemptKind::EventTriggerAbsoluteRms)
                .then_some((provenance.solve_id.as_str(), manifest))
        })
        .collect();
    let covering_renders: Vec<&DspManifest> = published
        .iter()
        .filter(|manifest| manifest.manifest_type == "classifier_render")
        .filter(|manifest| {
            solve_provenance(manifest).attempt_kind == SolveAttemptKind::EventTriggerAbsoluteRms
        })
        .collect();
    assert!(
        covering_renders.len() >= 2,
        "renders after frames 0 and 25 cover an event solve"
    );
    for render in covering_renders {
        let render_provenance = solve_provenance(render);
        let event_solve = event_solves
            .get(render_provenance.solve_id.as_str())
            .unwrap_or_else(|| panic!("render repeats an unknown solve id"));
        let event_provenance = solve_provenance(event_solve);
        assert_eq!(
            render_provenance.window_center_time_ns,
            event_provenance.window_center_time_ns
        );
        assert_eq!(
            render.localization.as_ref().unwrap().position_m,
            event_solve.localization.as_ref().unwrap().position_m,
            "the render re-runs the same solve"
        );
    }
}

#[tokio::test]
async fn a_render_solves_the_fuller_event_window_once_a_late_member_covers_it() {
    // Coverage wait off: the frame-25 trigger solves tetra-a alone because
    // tetra-b runs one frame late. The render ~16 frames later finds b's audio
    // for that window and solves all eight sensors as a new solve.
    const RENDER_INTERVAL_SECONDS: f64 = 0.5;
    let tetra_a = FixtureNode::tetra("tetra-a", [0.0, 0.0, 0.0]);
    let tetra_b = FixtureNode::tetra("tetra-b", [20.0, 0.0, 0.0]);
    let source = CoherentPointSource::new([10.0, 8.0, 2.0], FRAMES_PER_RUN);
    let mut harness = RaceHarness::new(DspWorkerConfig {
        classifier_render_min_interval_seconds: RENDER_INTERVAL_SECONDS,
        ..race_config(false)
    })
    .await;
    deliver_with_late_node(
        &mut harness,
        &tetra_a,
        &tetra_b,
        &source,
        0..=SECOND_SHARED_TRIGGER_FRAME - 1,
    )
    .await;
    let published = drain_published_manifests(&mut harness.events);
    let trigger_solve = published
        .iter()
        .filter(|manifest| manifest.manifest_type == "localization_result")
        .find(|manifest| {
            manifest.source_handles[0]
                .segment_id
                .ends_with(&format!("tetra-a-{FIRST_SHARED_TRIGGER_FRAME}"))
                && manifest.localization.as_ref().is_some_and(|localization| {
                    localization
                        .solve_provenance
                        .as_ref()
                        .is_some_and(|provenance| {
                            provenance.attempt_kind == SolveAttemptKind::EventTriggerAbsoluteRms
                        })
                })
        })
        .expect("frame-25 trigger solve");
    let trigger = solve_provenance(trigger_solve);
    assert_eq!(trigger.active_sensor_indices.len(), 4, "tetra-b is late");
    let fuller_render = published
        .iter()
        .filter(|manifest| manifest.manifest_type == "classifier_render")
        .map(solve_provenance)
        .find(|provenance| provenance.window_center_time_ns == trigger.window_center_time_ns)
        .expect("a render covering the frame-25 trigger");
    assert_eq!(fuller_render.active_sensor_indices.len(), 8);
    assert_ne!(
        fuller_render.solve_id, trigger.solve_id,
        "a different, fuller solve"
    );
    assert_eq!(
        fuller_render.attempt_kind,
        SolveAttemptKind::EventTriggerAbsoluteRms
    );
}
