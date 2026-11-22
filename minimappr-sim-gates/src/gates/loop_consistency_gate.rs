//! Loop consistency (`tdoa_solver::loop_consistency`): every rejected
//! triple must violate |τ_AB + τ_BC + τ_CA| > τ_thresh, every kept triple
//! must close, a decisive reflection lock (≥ 15σ detour) must be the pair
//! removed, and clean sets lose a pair only at the 3σ tail's rate.

use minimappr_core::tdoa_solver::loop_consistency::{filter_loop_consistent_pairs, LoopConsistencyConfig, DEFAULT_CLOSURE_TOLERANCE_SIGMA};
use minimappr_core::tdoa_solver::measurement_types::PairTdoaMeasurement;

use super::GateRun;
use crate::gate_seeds::{run_trials_in_parallel, GateOutcome};
use crate::statistics::{binomial_standard_error, two_sided_normal_tail};
use crate::trial_geometry::{count_in, distance, gaussian, inside_source, random_phone_layout, uniform_in};

const SOUND_SPEED_MPS: f64 = 343.0;
const STRONG_CORRELATION_PEAK: f64 = 0.8;
/// The reflection-locked pair's GCC peak is weaker than the direct paths.
const REFLECTION_CORRELATION_PEAK: f64 = 0.4;

struct LoopTrial {
    problems: Vec<String>,
    loop_count: usize,
    clean_false_rejections: usize,
    reflection_caught: bool,
}

fn sensor_name(index: usize) -> String {
    format!("s{index}")
}

fn measurements(seed: u64, purpose: &str, layout: &[[f64; 3]], source: [f64; 3], std_s: f64) -> Vec<PairTdoaMeasurement> {
    let mut pairs = Vec::new();
    for a in 0..layout.len() {
        for b in a + 1..layout.len() {
            let truth = (distance(source, layout[a]) - distance(source, layout[b])) / SOUND_SPEED_MPS;
            pairs.push(PairTdoaMeasurement {
                sensor_a: sensor_name(a),
                sensor_b: sensor_name(b),
                tdoa_seconds: truth + std_s * gaussian(seed, purpose, a as u64, b as u64),
                correlation_peak: STRONG_CORRELATION_PEAK,
                weight: 1.0,
            });
        }
    }
    pairs
}

fn check_report_consistency(report: &minimappr_core::tdoa_solver::loop_consistency::LoopConsistencyReport, tolerance_s: f64, label: &str, problems: &mut Vec<String>) {
    for closure in &report.initial_closures {
        if closure.consistent != (closure.closure_error_s.abs() <= tolerance_s) {
            problems.push(format!("{label}: loop {:?} flagged {} with closure {:.2e} s (tolerance {tolerance_s:.2e})", closure.sensors, closure.consistent, closure.closure_error_s));
        }
    }
    // Every loop among the kept pairs closes.
    let kept = filter_loop_consistent_pairs(&report.kept, &LoopConsistencyConfig { pair_timing_std_s: tolerance_s / (DEFAULT_CLOSURE_TOLERANCE_SIGMA * 3f64.sqrt()), closure_tolerance_sigma: DEFAULT_CLOSURE_TOLERANCE_SIGMA });
    if kept.initial_inconsistent_loop_count > 0 {
        problems.push(format!("{label}: {} loops among kept pairs still fail to close", kept.initial_inconsistent_loop_count));
    }
}

fn run_trial(seed: u64, run: &GateRun) -> LoopTrial {
    let config = &run.config.loop_consistency;
    let sensor_count = count_in(seed, "loop_sensor_count", config.sensor_count_range);
    let aperture = uniform_in(seed, "loop_aperture", 0, run.config.geometry.aperture_range_m);
    let layout = random_phone_layout(seed, sensor_count, aperture, &run.config.geometry);
    let source = inside_source(seed, &layout, aperture, &run.config.geometry);
    let loop_config = LoopConsistencyConfig { pair_timing_std_s: config.pair_timing_std_s, closure_tolerance_sigma: DEFAULT_CLOSURE_TOLERANCE_SIGMA };
    let tolerance = loop_config.closure_tolerance_s();
    let mut problems = Vec::new();

    let clean = measurements(seed, "loop_pair_noise", &layout, source, config.pair_timing_std_s);
    let clean_report = filter_loop_consistent_pairs(&clean, &loop_config);
    check_report_consistency(&clean_report, tolerance, "clean", &mut problems);

    // Same noise as the clean set: extra rejections are the reflection's doing.
    let mut locked = measurements(seed, "loop_pair_noise", &layout, source, config.pair_timing_std_s);
    let bad = (uniform_in(seed, "loop_bad_pair", 0, [0.0, 1.0]) * locked.len() as f64) as usize % locked.len();
    let sign = if uniform_in(seed, "loop_bad_sign", 0, [0.0, 1.0]) < 0.5 { -1.0 } else { 1.0 };
    locked[bad].tdoa_seconds += sign * uniform_in(seed, "loop_bad_delay", 0, config.reflection_delay_range_s);
    locked[bad].correlation_peak = REFLECTION_CORRELATION_PEAK;
    let locked_report = filter_loop_consistent_pairs(&locked, &loop_config);
    check_report_consistency(&locked_report, tolerance, "reflection", &mut problems);
    let bad_key = (locked[bad].sensor_a.clone(), locked[bad].sensor_b.clone());
    let reflection_caught = locked_report.rejected.iter().any(|r| (r.sensor_a.clone(), r.sensor_b.clone()) == bad_key);
    if !reflection_caught {
        problems.push(format!("{sensor_count} sensors: reflection-locked pair {bad_key:?} kept"));
    }
    // Only the 3-sensor case (one loop) may need to drop a good pair to
    // remove the bad one; with ≥ 4 sensors the bad pair alone breaks loops.
    if sensor_count > 3 && locked_report.rejected.len() > 1 + clean_report.rejected.len() {
        problems.push(format!("{sensor_count} sensors: {} pairs rejected for one reflection", locked_report.rejected.len()));
    }
    LoopTrial { problems, loop_count: clean_report.initial_closures.len(), clean_false_rejections: clean_report.rejected.len(), reflection_caught }
}

pub fn loop_consistency(run: &GateRun) -> GateOutcome {
    let results = run_trials_in_parallel(&run.seeds, run.threads, |seed| run_trial(seed, run));
    let mut outcome = GateOutcome::new("loop_consistency", results.len());
    for (seed, trial) in &results {
        for problem in &trial.problems {
            outcome.fail_seed(*seed, problem.clone());
        }
    }
    let with_false = results.iter().filter(|(_, t)| t.clean_false_rejections > 0).count();
    let fraction = with_false as f64 / results.len().max(1) as f64;
    let caught = results.iter().filter(|(_, t)| t.reflection_caught).count();
    let per_loop_false_alarm = two_sided_normal_tail(DEFAULT_CLOSURE_TOLERANCE_SIGMA);
    let union_bound = results.iter().map(|(_, t)| (t.loop_count as f64 * per_loop_false_alarm).min(1.0)).sum::<f64>() / results.len().max(1) as f64;
    let limit = union_bound
        + run.config.loop_consistency.clean_false_rejection_slack_standard_errors * binomial_standard_error(union_bound, results.len());
    outcome.summary(format!("clean sets with a false rejection: {with_false}/{} ({fraction:.3}, limit {limit:.3} = union bound {union_bound:.3} + slack); reflections caught {caught}/{}", results.len(), results.len()));
    let enough = results.len() >= run.config.coverage.minimum_trials_for_aggregate_gate;
    outcome.check_aggregate(fraction <= limit, enough, || format!("clean false-rejection fraction {fraction:.3} > {limit}"));
    outcome.contributing_seeds = results.iter().filter(|(_, t)| t.clean_false_rejections > 0).map(|(s, _)| *s).collect();
    outcome
}
