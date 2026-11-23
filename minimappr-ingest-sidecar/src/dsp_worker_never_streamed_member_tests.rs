//! A cluster member that never streams (live: one Bluefin node silent for
//! the whole process lifetime) must not empty the classifier render of the
//! members that do stream.

use super::dsp_worker_coverage_race_tests::{
    race_config, RaceHarness, FRAMES_PER_RUN, FRAME_DURATION_NS, NODE_A,
};
use super::dsp_worker_tests::drain_published_manifests;
use super::*;

const NANOSECONDS_PER_MILLISECOND: u64 = 1_000_000;

/// `beamformed` (the default) takes the band-split path once SRP passes the
/// confidence floor; `nearest_node_omni` always takes the omni mix.
const RENDER_SOURCES: [&str; 2] = ["beamformed", "nearest_node_omni"];

#[tokio::test]
async fn never_streamed_member_does_not_empty_the_beamformed_render() {
    assert_every_result_carries_a_render(RENDER_SOURCES[0]).await;
}

#[tokio::test]
async fn never_streamed_member_does_not_empty_the_omni_render() {
    assert_every_result_carries_a_render(RENDER_SOURCES[1]).await;
}

async fn assert_every_result_carries_a_render(classification_audio_source: &str) {
    // Renders share the trigger cooldown, so every real localization result
    // has a render due; node b is declared in the cluster but never delivers.
    let base_config = race_config(true);
    let config = DspWorkerConfig {
        classifier_render_min_interval_seconds: base_config.trigger_cooldown_seconds,
        classification_audio_source: classification_audio_source.to_string(),
        ..base_config
    };
    let mut harness = RaceHarness::new(config).await;
    for frame_index in 0..FRAMES_PER_RUN {
        harness.advance_clock_ms(FRAME_DURATION_NS / NANOSECONDS_PER_MILLISECOND);
        harness.deliver(NODE_A, frame_index).await;
    }
    let published = drain_published_manifests(&mut harness.events);
    let render_count = published
        .iter()
        .filter(|manifest| manifest.manifest_type == "classifier_render")
        .count();
    let real_results: Vec<&DspManifest> = published
        .iter()
        .filter(|manifest| {
            manifest.manifest_type == "localization_result"
                && manifest.localization.as_ref().is_some_and(|localization| {
                    localization.resolved_algorithm != "localization_cadence_skipped"
                })
        })
        .collect();
    assert!(
        !real_results.is_empty(),
        "no localization results published"
    );
    assert!(
        render_count >= real_results.len(),
        "{classification_audio_source} renders: {render_count}, results: {}",
        real_results.len()
    );
    for result in &real_results {
        assert!(
            result.derived_handle.is_some(),
            "{classification_audio_source} result for {} lost its render to the never-streamed member",
            result.source_handles[0].segment_id
        );
    }
}
