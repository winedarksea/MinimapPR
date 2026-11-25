use super::*;
use crate::dsp_worker::solve_layout_test_fixture::tetra_nodes_layout;

const SOUND_SPEED_MPS: f32 = 343.0;

fn two_nodes(baseline_m: f64) -> SolveSensorLayout {
    tetra_nodes_layout(&[("a", [0.0, 0.0, 0.0]), ("b", [baseline_m, 0.0, 0.0])])
}

#[test]
fn node_local_layout_keeps_todays_window() {
    let config = DspWorkerConfig::default();
    let solo = tetra_nodes_layout(&[("solo", [0.0, 0.0, 0.0])]);
    assert_eq!(
        event_solve_window_seconds(&solo, &config, SOUND_SPEED_MPS),
        config.window_seconds
    );
}

#[test]
fn cross_node_span_scales_with_the_baseline_and_covers_the_full_tdoa_range() {
    let config = DspWorkerConfig::default();
    let node_local = config.window_seconds;
    // Within half a node-local window of lag (~5 m) nothing changes.
    let short = event_solve_window_seconds(&two_nodes(2.0), &config, SOUND_SPEED_MPS);
    assert_eq!(short, node_local);
    let medium = event_solve_window_seconds(&two_nodes(20.0), &config, SOUND_SPEED_MPS);
    let long = event_solve_window_seconds(&two_nodes(200.0), &config, SOUND_SPEED_MPS);
    assert!(medium > node_local);
    assert!(long > medium);

    // Centred on the trigger, each half-span reaches the far node's whole lag
    // (the largest cross-node distance, `b`'s 4th mic 0.15 m further out,
    // over c): a trigger-length event centred there keeps half inside.
    let max_lag_seconds = (200.0 + 3.0 * 0.05) / f64::from(SOUND_SPEED_MPS);
    assert!((long / 2.0 - max_lag_seconds).abs() < 1e-9);
}

#[test]
fn lag_extent_is_capped_at_what_the_coverage_wait_can_collect() {
    let config = DspWorkerConfig::default();
    let span = event_solve_window_seconds(&two_nodes(5_000.0), &config, SOUND_SPEED_MPS);
    let capped_lag = config.localization_coverage_total_cap_seconds;
    assert!((span - (2.0 * capped_lag)).abs() < 1e-9, "{span}");
    let short_buffer = DspWorkerConfig {
        max_buffer_seconds: 0.5,
        ..DspWorkerConfig::default()
    };
    assert_eq!(
        event_solve_window_seconds(&two_nodes(5_000.0), &short_buffer, SOUND_SPEED_MPS),
        0.5
    );
}

#[test]
fn without_the_coverage_wait_the_event_keeps_todays_window() {
    let config = DspWorkerConfig {
        localization_coverage_wait_all_streaming_nodes_enabled: false,
        ..DspWorkerConfig::default()
    };
    assert_eq!(
        event_solve_window_seconds(&two_nodes(200.0), &config, SOUND_SPEED_MPS),
        config.window_seconds
    );
}
