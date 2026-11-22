//! Loop-consistency filter for argmax pair TDOAs (CARDINAL-style).
//!
//! For any three sensors A, B, C with all three pairs measured, the direct-path
//! delays close: `τ_AB + τ_BC + τ_CA = 0`. A pair whose GCC peak locked onto a
//! reflection breaks every loop it is in. The filter greedily drops the pair
//! that breaks the most loops until all remaining loops close within
//! tolerance. Use it on the argmax-TDOA path only: before SRP-PHAT it would
//! discard pairs whose true direct path is the *second* peak (plan §Loop
//! consistency), where it is only a diagnostic.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::measurement_types::PairTdoaMeasurement;

/// Default timing std of one argmax pair TDOA, in interpolated sample periods.
pub const DEFAULT_PAIR_TIMING_STD_SAMPLE_PERIODS: f64 = 1.0;
/// Default closure tolerance in standard deviations of the loop sum.
pub const DEFAULT_CLOSURE_TOLERANCE_SIGMA: f64 = 3.0;
/// A loop sum adds three independent pair errors: σ_loop = √3 · σ_pair.
const PAIRS_PER_LOOP: f64 = 3.0;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LoopConsistencyConfig {
    /// One-sigma timing error of a single pair TDOA (seconds).
    pub pair_timing_std_s: f64,
    pub closure_tolerance_sigma: f64,
}

impl LoopConsistencyConfig {
    /// Defaults scaled to one interpolated sample period.
    pub fn for_sample_period(interpolated_sample_period_s: f64) -> Self {
        Self {
            pair_timing_std_s: interpolated_sample_period_s
                * DEFAULT_PAIR_TIMING_STD_SAMPLE_PERIODS,
            closure_tolerance_sigma: DEFAULT_CLOSURE_TOLERANCE_SIGMA,
        }
    }

    pub fn closure_tolerance_s(&self) -> f64 {
        self.closure_tolerance_sigma * self.pair_timing_std_s * PAIRS_PER_LOOP.sqrt()
    }
}

/// One closed triple and its loop sum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoopClosure {
    pub sensors: [String; 3],
    pub closure_error_s: f64,
    pub consistent: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoopConsistencyReport {
    pub kept: Vec<PairTdoaMeasurement>,
    pub rejected: Vec<PairTdoaMeasurement>,
    /// Loops evaluated on the *input* set (diagnostics for the SRP path).
    pub initial_closures: Vec<LoopClosure>,
    pub initial_inconsistent_loop_count: usize,
}

/// Directed lookup: `τ(a, b)` for both orientations of every measured pair.
fn directed_tdoa_table(
    measurements: &[PairTdoaMeasurement],
    active: &[bool],
) -> BTreeMap<(String, String), f64> {
    let mut table = BTreeMap::new();
    for (measurement, is_active) in measurements.iter().zip(active) {
        if !*is_active {
            continue;
        }
        let forward = (measurement.sensor_a.clone(), measurement.sensor_b.clone());
        let backward = (measurement.sensor_b.clone(), measurement.sensor_a.clone());
        table.insert(forward, measurement.tdoa_seconds);
        table.insert(backward, -measurement.tdoa_seconds);
    }
    table
}

fn sorted_sensor_ids(measurements: &[PairTdoaMeasurement]) -> Vec<String> {
    let mut sensors: Vec<String> = measurements
        .iter()
        .flat_map(|measurement| [measurement.sensor_a.clone(), measurement.sensor_b.clone()])
        .collect();
    sensors.sort();
    sensors.dedup();
    sensors
}

fn evaluate_loops(
    sensors: &[String],
    table: &BTreeMap<(String, String), f64>,
    tolerance_s: f64,
) -> Vec<LoopClosure> {
    let mut closures = Vec::new();
    for first in 0..sensors.len() {
        for second in first + 1..sensors.len() {
            for third in second + 1..sensors.len() {
                let (a, b, c) = (&sensors[first], &sensors[second], &sensors[third]);
                let legs = [
                    table.get(&(a.clone(), b.clone())),
                    table.get(&(b.clone(), c.clone())),
                    table.get(&(c.clone(), a.clone())),
                ];
                if let [Some(ab), Some(bc), Some(ca)] = legs {
                    let closure_error_s = ab + bc + ca;
                    closures.push(LoopClosure {
                        sensors: [a.clone(), b.clone(), c.clone()],
                        closure_error_s,
                        consistent: closure_error_s.abs() <= tolerance_s,
                    });
                }
            }
        }
    }
    closures
}

fn pair_in_loop(measurement: &PairTdoaMeasurement, closure: &LoopClosure) -> bool {
    closure.sensors.contains(&measurement.sensor_a)
        && closure.sensors.contains(&measurement.sensor_b)
}

/// Drop reflection-locked pairs. Pairs not in any closed loop are always kept
/// (nothing can contradict them). Tie-break when several pairs break equally
/// many loops: drop the one with the weaker correlation peak.
pub fn filter_loop_consistent_pairs(
    measurements: &[PairTdoaMeasurement],
    config: &LoopConsistencyConfig,
) -> LoopConsistencyReport {
    let tolerance_s = config.closure_tolerance_s();
    let sensors = sorted_sensor_ids(measurements);
    // Non-finite delays can never close a loop; reject them up front.
    let mut active: Vec<bool> = measurements
        .iter()
        .map(|measurement| measurement.tdoa_seconds.is_finite())
        .collect();
    let initial_closures = evaluate_loops(
        &sensors,
        &directed_tdoa_table(measurements, &active),
        tolerance_s,
    );
    let initial_inconsistent_loop_count = initial_closures
        .iter()
        .filter(|closure| !closure.consistent)
        .count();

    loop {
        let closures = evaluate_loops(
            &sensors,
            &directed_tdoa_table(measurements, &active),
            tolerance_s,
        );
        let broken: Vec<&LoopClosure> = closures
            .iter()
            .filter(|closure| !closure.consistent)
            .collect();
        if broken.is_empty() {
            break;
        }
        let worst = measurements
            .iter()
            .enumerate()
            .filter(|(index, _)| active[*index])
            .map(|(index, measurement)| {
                let broken_count = broken
                    .iter()
                    .filter(|closure| pair_in_loop(measurement, closure))
                    .count();
                let consistent_count = closures
                    .iter()
                    .filter(|closure| closure.consistent && pair_in_loop(measurement, closure))
                    .count();
                (
                    index,
                    broken_count,
                    consistent_count,
                    measurement.correlation_peak.abs(),
                )
            })
            .filter(|(_, broken_count, _, _)| *broken_count > 0)
            .max_by(|left, right| {
                left.1.cmp(&right.1).then(right.2.cmp(&left.2)).then(
                    right
                        .3
                        .partial_cmp(&left.3)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
            });
        match worst {
            Some((index, ..)) => active[index] = false,
            None => break,
        }
    }

    let mut kept = Vec::new();
    let mut rejected = Vec::new();
    for (measurement, is_active) in measurements.iter().zip(&active) {
        if *is_active {
            kept.push(measurement.clone());
        } else {
            rejected.push(measurement.clone());
        }
    }
    LoopConsistencyReport {
        kept,
        rejected,
        initial_closures,
        initial_inconsistent_loop_count,
    }
}
