use super::*;
use crate::dsp_worker::solve_layout_test_fixture::tetra_nodes_layout;

const SOUND_SPEED_MPS: f32 = 343.0;
/// `b`'s 4th fixture mic sits 3 x 5 cm further out than its node centre.
const FIXTURE_FAR_MIC_OFFSET_M: f64 = 0.15;

fn two_nodes(baseline_m: f64) -> SolveSensorLayout {
    tetra_nodes_layout(&[("a", [0.0, 0.0, 0.0]), ("b", [baseline_m, 0.0, 0.0])])
}

fn timing(layout: &SolveSensorLayout, config: &DspWorkerConfig) -> DerivedEventTiming {
    derived_event_timing(layout, config, SOUND_SPEED_MPS)
}

#[test]
fn node_local_layout_keeps_todays_window_and_wait() {
    let config = DspWorkerConfig::default();
    let solo = tetra_nodes_layout(&[("solo", [0.0, 0.0, 0.0])]);
    let solo_timing = timing(&solo, &config);
    assert_eq!(solo_timing.event_span_seconds, config.window_seconds);
    // The delivery margin alone: the old fixed 1 s wait.
    assert_eq!(
        solo_timing.coverage_total_cap_seconds,
        config.localization_solve.localization_coverage_delivery_margin_seconds
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

    // Centred on the trigger, each half-span reaches the far node's whole lag.
    let max_lag_seconds = (200.0 + FIXTURE_FAR_MIC_OFFSET_M) / f64::from(SOUND_SPEED_MPS);
    assert!((long / 2.0 - max_lag_seconds).abs() < 1e-9);
}

#[test]
fn compact_site_pays_no_extra_span_or_wait() {
    let config = DspWorkerConfig::default();
    let compact = timing(&two_nodes(20.0), &config);
    assert!(compact.event_span_seconds < 0.15, "{compact:?}");
    assert!(compact.coverage_total_cap_seconds < 1.1, "{compact:?}");
    assert!(!compact.lag_bounded);
}

/// The old fixed 1 s cap held a km network to ~340 m of span; the derived
/// span follows the baseline with no setting.
#[test]
fn a_kilometre_spread_gets_a_multi_second_span_and_wait() {
    let config = DspWorkerConfig::default();
    let spread = timing(&two_nodes(1_500.0), &config);
    let lag_seconds = (1_500.0 + FIXTURE_FAR_MIC_OFFSET_M) / f64::from(SOUND_SPEED_MPS);
    assert!((spread.event_span_seconds - 2.0 * lag_seconds).abs() < 1e-9, "{spread:?}");
    assert!(
        (spread.coverage_total_cap_seconds
            - (lag_seconds + config.localization_solve.localization_coverage_delivery_margin_seconds))
            .abs()
            < 1e-9
    );
    assert!(!spread.lag_bounded);
}

/// The span ignores clocks (GCC searches only to baseline / c); the
/// same-event window, judged on each node's own onset stamps, does not.
#[test]
fn a_weak_clock_widens_the_same_event_window_not_the_span() {
    let config = DspWorkerConfig::default();
    let layout = two_nodes(1_500.0);
    let window = |clock_uncertainty_seconds: f64| {
        same_event_window_seconds(&layout, &config, SOUND_SPEED_MPS, "a", "b", |_| {
            clock_uncertainty_seconds
        })
    };
    let margin_seconds = config.localization_solve.localization_event_span_clock_margin_sigma
        * 0.05_f64.hypot(0.05);
    assert!((window(0.05) - window(0.0) - margin_seconds).abs() < 1e-9);
}

#[test]
fn the_safety_bound_and_the_buffer_bound_the_lag() {
    let config = DspWorkerConfig::default();
    let bound_seconds = config
        .localization_solve
        .localization_event_span_max_cross_node_lag_seconds;
    let too_far = timing(&two_nodes(10_000.0), &config);
    assert!(too_far.lag_bounded);
    assert!((too_far.cross_node_lag_seconds - bound_seconds).abs() < 1e-9);
    assert!((too_far.event_span_seconds - 2.0 * bound_seconds).abs() < 1e-9);

    let short_buffer = DspWorkerConfig {
        max_buffer_seconds: 5.0,
        ..DspWorkerConfig::default()
    };
    let buffered = timing(&two_nodes(1_500.0), &short_buffer);
    assert!(buffered.lag_bounded);
    assert!(buffered.event_span_seconds < short_buffer.max_buffer_seconds);
}

#[test]
fn an_explicit_cap_overrides_the_wait_and_bounds_the_lag() {
    let config = DspWorkerConfig {
        localization_coverage_total_cap_seconds: 1.0,
        ..DspWorkerConfig::default()
    };
    let capped = timing(&two_nodes(5_000.0), &config);
    assert!((capped.event_span_seconds - 2.0).abs() < 1e-9, "{capped:?}");
    assert_eq!(capped.coverage_total_cap_seconds, 1.0);
    let solo = timing(&tetra_nodes_layout(&[("solo", [0.0, 0.0, 0.0])]), &config);
    assert_eq!(solo.coverage_total_cap_seconds, 1.0);
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

#[test]
fn the_same_event_window_is_the_cooldown_plus_the_pair_lag() {
    let config = DspWorkerConfig::default();
    let layout = tetra_nodes_layout(&[
        ("a", [0.0, 0.0, 0.0]),
        ("b", [20.0, 0.0, 0.0]),
        ("c", [1_500.0, 0.0, 0.0]),
    ]);
    let window = |last: &str, current: &str| {
        same_event_window_seconds(&layout, &config, SOUND_SPEED_MPS, last, current, |_| 0.0)
    };
    assert_eq!(window("a", "a"), config.trigger_cooldown_seconds);
    // The pair's own distance, not the site's largest baseline: a close pair
    // keeps a short window so distinct events stay distinct.
    let near_pair_lag = (20.0 + FIXTURE_FAR_MIC_OFFSET_M) / f64::from(SOUND_SPEED_MPS);
    assert!((window("a", "b") - config.trigger_cooldown_seconds - near_pair_lag).abs() < 1e-9);
    assert!(window("a", "c") > config.trigger_cooldown_seconds + 4.0);
    assert_eq!(window("a", "c"), window("c", "a"));
    // A node gone from the layout gets the largest baseline (never a new event early).
    assert!(window("gone", "a") >= window("a", "c"));
}
