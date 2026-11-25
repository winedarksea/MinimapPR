use super::*;

/// Shared with `tests/test_localization_event_timing.py`: both languages must
/// reproduce every case.
const VECTORS_JSON: &str = include_str!("../../tests/fixtures/localization_event_timing_vectors.json");
const TOLERANCE_SECONDS: f64 = 1.0e-9;

fn number(value: &serde_json::Value, key: &str) -> f64 {
    value[key].as_f64().unwrap_or_else(|| panic!("{key} is a number"))
}

fn config_from(value: &serde_json::Value) -> EventTimingConfig {
    EventTimingConfig {
        node_local_window_seconds: number(value, "node_local_window_seconds"),
        coverage_wait_enabled: value["coverage_wait_enabled"].as_bool().expect("bool"),
        explicit_coverage_total_cap_seconds: number(value, "explicit_coverage_total_cap_seconds"),
        coverage_delivery_margin_seconds: number(value, "coverage_delivery_margin_seconds"),
        clock_margin_sigma: number(value, "clock_margin_sigma"),
        max_cross_node_lag_seconds: number(value, "max_cross_node_lag_seconds"),
        max_buffer_seconds: number(value, "max_buffer_seconds"),
    }
}

fn vectors() -> serde_json::Value {
    serde_json::from_str(VECTORS_JSON).expect("vectors JSON")
}

#[test]
fn derived_event_timing_matches_the_shared_vectors() {
    let vectors = vectors();
    let cases = vectors["event_timing_cases"].as_array().expect("cases");
    assert!(cases.len() >= 8);
    for case in cases {
        let name = case["name"].as_str().expect("name");
        let geometry = (!case["geometry"].is_null()).then(|| CrossNodeGeometry {
            max_cross_node_distance_m: number(&case["geometry"], "max_cross_node_distance_m"),
        });
        let derived = derive_event_timing(
            geometry,
            number(case, "sound_speed_mps"),
            &config_from(&case["config"]),
        );
        let expected = &case["expected"];
        for (actual, key) in [
            (derived.cross_node_lag_seconds, "cross_node_lag_seconds"),
            (derived.event_span_seconds, "event_span_seconds"),
            (derived.coverage_total_cap_seconds, "coverage_total_cap_seconds"),
        ] {
            assert!(
                (actual - number(expected, key)).abs() < TOLERANCE_SECONDS,
                "{name}: {key} {actual} != {}",
                number(expected, key)
            );
        }
        assert_eq!(
            derived.lag_bounded,
            expected["lag_bounded"].as_bool().expect("bool"),
            "{name}: lag_bounded"
        );
    }
}

#[test]
fn same_event_window_matches_the_shared_vectors() {
    let vectors = vectors();
    for case in vectors["same_event_window_cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let window_seconds = same_event_trigger_window_seconds(
            number(case, "trigger_cooldown_seconds"),
            number(case, "node_distance_m"),
            number(case, "pair_clock_uncertainty_seconds"),
            number(case, "sound_speed_mps"),
            &config_from(&case["config"]),
        );
        assert!(
            (window_seconds - number(case, "expected_window_seconds")).abs() < TOLERANCE_SECONDS,
            "{name}: {window_seconds}"
        );
    }
}

#[test]
fn compact_site_span_and_wait_stay_short_and_km_spread_is_multi_second() {
    let vectors = vectors();
    let config = config_from(&vectors["event_timing_cases"][0]["config"]);
    let compact = derive_event_timing(
        Some(CrossNodeGeometry {
            max_cross_node_distance_m: 20.0,
        }),
        343.0,
        &config,
    );
    assert!(compact.event_span_seconds < 0.2, "{compact:?}");
    assert!(compact.coverage_total_cap_seconds < 1.1, "{compact:?}");
    let spread = derive_event_timing(
        Some(CrossNodeGeometry {
            max_cross_node_distance_m: 2_000.0,
        }),
        343.0,
        &config,
    );
    assert!(spread.event_span_seconds > 10.0, "{spread:?}");
    assert!(spread.coverage_total_cap_seconds > spread.cross_node_lag_seconds);
    assert!(!spread.lag_bounded);
}

#[test]
fn the_required_buffer_holds_the_largest_span_the_safety_bound_allows() {
    let vectors = vectors();
    let mut config = config_from(&vectors["event_timing_cases"][0]["config"]);
    config.max_buffer_seconds = required_sensor_buffer_seconds(&config);
    let far = derive_event_timing(
        Some(CrossNodeGeometry {
            max_cross_node_distance_m: 1.0e6,
        }),
        343.0,
        &config,
    );
    assert!((far.cross_node_lag_seconds - config.max_cross_node_lag_seconds).abs() < 1e-9);
    assert!(far.lag_bounded);
    assert!(far.event_span_seconds <= config.max_buffer_seconds);
}
