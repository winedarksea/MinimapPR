//! Loop-consistency filter tests: clean sets pass, a reflection-locked pair is
//! rejected, non-finite delays are dropped.

use nalgebra::Vector3;

use super::loop_consistency::{filter_loop_consistent_pairs, LoopConsistencyConfig};
use super::test_support::*;

const ONE_SAMPLE_S: f64 = 1.0 / (SAMPLE_RATE_HZ as f64 * INTERPOLATION_FACTOR as f64);
/// A trunk reflection arriving ~2 ms late on one pair.
const REFLECTION_DELAY_S: f64 = 2.0e-3;

fn five_node_measurements(noise_std_s: f64) -> Vec<super::measurement_types::PairTdoaMeasurement> {
    let layout = positions(&[
        ("a", [0.0, 0.0, 1.0]),
        ("b", [12.0, 0.0, 1.0]),
        ("c", [12.0, 12.0, 1.0]),
        ("d", [0.0, 12.0, 1.0]),
        ("e", [6.0, -5.0, 1.0]),
    ]);
    synthetic_measurements(
        &layout,
        &Vector3::new(4.0, 7.0, 9.0),
        noise_std_s,
        &mut DeterministicGaussian::new(11),
    )
}

#[test]
fn consistent_set_keeps_every_pair() {
    let measurements = five_node_measurements(0.5 * ONE_SAMPLE_S);
    let report = filter_loop_consistent_pairs(
        &measurements,
        &LoopConsistencyConfig::for_sample_period(ONE_SAMPLE_S),
    );
    assert!(report.rejected.is_empty());
    assert_eq!(report.kept.len(), measurements.len());
    // 5 sensors with all pairs measured → C(5,3) loops.
    assert_eq!(report.initial_closures.len(), 10);
    assert_eq!(report.initial_inconsistent_loop_count, 0);
}

#[test]
fn reflection_locked_pair_is_rejected_and_only_that_pair() {
    let mut measurements = five_node_measurements(0.5 * ONE_SAMPLE_S);
    let corrupted_index = 4; // pair (b, c)
    measurements[corrupted_index].tdoa_seconds += REFLECTION_DELAY_S;
    let report = filter_loop_consistent_pairs(
        &measurements,
        &LoopConsistencyConfig::for_sample_period(ONE_SAMPLE_S),
    );
    assert_eq!(report.rejected.len(), 1);
    assert_eq!(report.rejected[0], measurements[corrupted_index]);
    // The bad pair sits in 3 of the 10 loops.
    assert_eq!(report.initial_inconsistent_loop_count, 3);
}

#[test]
fn non_finite_delays_are_rejected() {
    let mut measurements = five_node_measurements(0.0);
    measurements[0].tdoa_seconds = f64::NAN;
    let report = filter_loop_consistent_pairs(
        &measurements,
        &LoopConsistencyConfig::for_sample_period(ONE_SAMPLE_S),
    );
    assert_eq!(report.rejected.len(), 1);
    assert!(report.rejected[0].tdoa_seconds.is_nan());
}
