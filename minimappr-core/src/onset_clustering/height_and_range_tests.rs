//! Range-profile search-boundary tests (mirror `tests/test_onset_s_fixes_units.py`).

use super::*;
use crate::range_projection::RANGE_REFINED;

const ORIGIN: Point3 = [0.0, 0.0, 0.0];
const CLAIMED_POSITION: Point3 = [50.0, 0.0, 0.0];
const IDENTITY_COVARIANCE: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// The server defaults (`OnsetClusteringConfig::default`) the review probe used.
fn standard_settings(max_range_m: f64) -> RangeProfileSettings {
    RangeProfileSettings {
        min_range_m: 0.5,
        max_range_m,
        sample_count: 96,
        cone_half_angle_deg: 3.0,
        cone_step_deg: 1.0,
        support_ratio: 0.9,
        max_interval_ratio: 2.0,
        interval_floor_m: 10.0,
    }
}

fn range_from_origin(point: &Point3) -> f64 {
    distance(point, &ORIGIN)
}

fn rising_tail(points: &[Point3]) -> Vec<f64> {
    points.iter().map(range_from_origin).collect()
}

fn interior_peak(points: &[Point3]) -> Vec<f64> {
    points
        .iter()
        .map(|p| (-0.5 * ((range_from_origin(p) - 60.0) / 10.0).powi(2)).exp())
        .collect()
}

fn flat(points: &[Point3]) -> Vec<f64> {
    vec![1.0; points.len()]
}

fn support_for(score: &dyn Fn(&[Point3]) -> Vec<f64>, max_range_m: f64) -> RangeSupport {
    range_support_along_bearing(
        &CLAIMED_POSITION,
        &ORIGIN,
        score,
        &standard_settings(max_range_m),
        None,
    )
    .expect("positive profile")
}

fn judge(support: RangeSupport) -> RangeJudgedPosition {
    judge_range_from_profile(
        &CLAIMED_POSITION,
        Some(IDENTITY_COVARIANCE),
        Some(RANGE_REFINED),
        Some(0.9),
        &ORIGIN,
        Some(support),
    )
}

#[test]
fn interior_peak_is_closed_and_observable() {
    let support = support_for(&interior_peak, 1000.0);
    assert!(!support.open_near_side && !support.open_far_side);
    assert!(support.observable);
    assert!(50.0 <= support.interval_m.0 && support.interval_m.1 <= 70.0);
    assert_eq!(support.range_lower_bound_m(), None);
    let judged = judge(support);
    assert_eq!(judged.range_projection_mode.as_deref(), Some(RANGE_REFINED));
    assert_eq!(judged.range_lower_bound_m, None);
}

#[test]
fn review_probe_rising_tail_at_one_kilometre_is_not_observable() {
    // Review probe: score still rising at the 1000 m limit gave 923-1000 m, observable.
    let support = support_for(&rising_tail, 1000.0);
    assert!(support.open_far_side && !support.open_near_side);
    assert!(
        (support.interval_m.0 - 923.0).abs() < 1.0,
        "{:?}",
        support.interval_m
    );
    assert_eq!(support.interval_m.1, 1000.0);
    assert!(
        support.interval_m.1 <= 2.0 * support.interval_m.0,
        "the width test alone would pass"
    );
    assert!(!support.observable);
    assert_eq!(support.range_lower_bound_m(), Some(support.interval_m.0));

    let judged = judge(support);
    assert_eq!(
        judged.range_projection_mode.as_deref(),
        Some(RANGE_BEARING_PROJECTED)
    );
    assert_eq!(judged.range_lower_bound_m, Some(support.interval_m.0));
    assert_eq!(judged.range_interval_m, Some(support.interval_m));
}

#[test]
fn moving_the_search_limit_never_makes_a_rising_tail_observable() {
    for max_range_m in [100.0, 250.0, 923.0, 1000.0, 1500.0, 5000.0, 20_000.0] {
        let support = support_for(&rising_tail, max_range_m);
        assert!(support.open_far_side, "max {max_range_m}");
        assert!(
            !support.observable,
            "max {max_range_m}: {:?}",
            support.interval_m
        );
        assert!(support.range_lower_bound_m().is_some());
    }
}

#[test]
fn flat_profile_is_open_on_both_sides_with_no_lower_bound() {
    let support = support_for(&flat, 1000.0);
    assert!(support.open_near_side && support.open_far_side);
    assert!(!support.observable);
    assert_eq!(support.range_lower_bound_m(), None);
    let judged = judge(support);
    assert_eq!(
        judged.range_projection_mode.as_deref(),
        Some(RANGE_BEARING_PROJECTED)
    );
    assert_eq!(judged.range_lower_bound_m, None);
}

#[test]
fn admissibility_prior_closes_the_far_side() {
    // Samples beyond 400 m are inadmissible (e.g. above the canopy along this bearing):
    // the prior, not the search limit, ends the interval.
    let admissible = |point: &Point3| range_from_origin(point) <= 400.0;
    let support = range_support_along_bearing(
        &CLAIMED_POSITION,
        &ORIGIN,
        &rising_tail,
        &standard_settings(1000.0),
        Some(&admissible),
    )
    .expect("positive profile");
    assert!(!support.open_far_side);
    assert!(support.observable);
    assert!(support.interval_m.1 <= 400.0);
}
