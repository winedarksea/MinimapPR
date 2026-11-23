//! Fit the two reported-region parameters on *training* seeds and report
//! coverage on training and held-out seeds.
//!
//! cargo run --release -p minimappr-sim-gates --example calibrate_region_parameters -- \
//!     [--mesh-sessions 150] [--srp-trials 120] [--held-out-mesh 40] [--held-out-srp 48] [--threads T] [--only mesh|srp]
//!
//! - Mesh contour σ multiplier (`EntityContourConfig::contour_sigma_multiplier`):
//!   each end-to-end Layer 2 fix's minimum covering multiplier is exact (the
//!   polygon gauge), so the fit is the target-quantile order statistic.
//! - SRP likelihood temperature (`RobustScoringConfig::likelihood_temperature`):
//!   each rendered trial's smallest covering temperature on a log ladder.
//!
//! Copy the printed fitted values into those core configs; the gate tests
//! then check held-out coverage against the shipped defaults.

use minimappr_core::mesh::config::{EntityContourConfig, FixCovarianceCalibrationConfig};
use minimappr_core::srp_layout::LayoutSrpConfig;
use minimappr_sim_gates::gate_config::SimGateConfig;
use minimappr_sim_gates::gate_seeds::{default_thread_count, pull_request_seeds, run_trials_in_parallel, training_seeds};
use minimappr_sim_gates::mesh_region_trials::{run_contour_session, MeshFixScore, MeshSessionConfig};
use minimappr_sim_gates::srp_trials::{render_srp_trial, temperature_ladder, SrpTrialConfig};
use minimappr_sim_gates::statistics::{coverage_at, fit_parameter_for_coverage, median, percentile};

const TEMPERATURE_LADDER_LOW: f64 = 0.001;
const TEMPERATURE_LADDER_HIGH: f64 = 1.0;
const TEMPERATURE_LADDER_STEPS: usize = 31;
const GAUSSIAN_95_TWELVE_GON_MULTIPLIER: f64 = 2.534;

struct Options {
    mesh_sessions: usize,
    srp_trials: usize,
    held_out_mesh: usize,
    held_out_srp: usize,
    threads: usize,
    only: Option<String>,
}

fn parse_options() -> Options {
    let mut options =
        Options { mesh_sessions: 150, srp_trials: 120, held_out_mesh: 40, held_out_srp: 48, threads: default_thread_count(), only: None };
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        let mut value = || arguments.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--mesh-sessions" => options.mesh_sessions = value().parse().expect("count"),
            "--srp-trials" => options.srp_trials = value().parse().expect("count"),
            "--held-out-mesh" => options.held_out_mesh = value().parse().expect("count"),
            "--held-out-srp" => options.held_out_srp = value().parse().expect("count"),
            "--threads" => options.threads = value().parse().expect("count"),
            "--only" => options.only = Some(value()),
            other => panic!("unknown flag {other}"),
        }
    }
    options
}

/// Non-exempt fixes scored with the shipped contour config.
fn mesh_scores(seeds: &[u64], threads: usize) -> Vec<MeshFixScore> {
    let contour = EntityContourConfig::default();
    let session = MeshSessionConfig::default();
    let scores: Vec<_> = run_trials_in_parallel(seeds, threads, |seed| run_contour_session(seed, &contour, &session))
        .into_iter()
        .flat_map(|(_, s)| s)
        .filter(|s| !s.is_flagged_mirror)
        .collect();
    if std::env::var("SIM_GATES_VERBOSE").is_ok() {
        for score in scores.iter().filter(|s| s.relative_frame_minimum_multiplier > 1.0) {
            println!("    miss@1: mirrored {} flagged {} seed {} call {} inside {} range {:.1} m err {:.2} m σmaj {:.2} m nodes {} mult {:.2}", score.layout_mirrored, score.is_flagged_mirror, score.seed, score.call_index, score.is_inside_call, score.true_range_from_centroid_m, score.relative_frame_error_m, score.reported_major_sigma_m, score.used_node_count, score.relative_frame_minimum_multiplier);
        }
    }
    scores
}

fn gauges(scores: &[MeshFixScore], inside: Option<bool>, world_frame: bool) -> Vec<f64> {
    scores
        .iter()
        .filter(|s| inside.is_none_or(|i| s.is_inside_call == i))
        .map(|s| if world_frame { s.world_frame_minimum_multiplier } else { s.relative_frame_minimum_multiplier })
        .collect()
}

fn calibrate_mesh(options: &Options, target: f64) {
    let shipped = EntityContourConfig::default().contour_sigma_multiplier;
    let training = training_seeds(options.mesh_sessions);
    let started = std::time::Instant::now();
    let scores = mesh_scores(&training, options.threads);
    let train = gauges(&scores, None, false);
    let train_errors: Vec<f64> = scores.iter().map(|s| s.relative_frame_error_m).collect();
    let fitted = fit_parameter_for_coverage(&train, target);
    println!("mesh contour: {} training fixes from {} sessions in {:.1} s", train.len(), training.len(), started.elapsed().as_secs_f64());
    println!("  relative-frame error median {:.2} m, P95 {:.2} m", median(&train_errors), percentile(&train_errors, 0.95));
    println!("  minimum covering multiplier: median {:.2}, P90 {:.2}, P95 {:.2}, P98 {:.2}", median(&train), percentile(&train, 0.9), percentile(&train, 0.95), percentile(&train, 0.98));
    println!("  training coverage: shipped {shipped:.3} -> {:.3}; Gaussian 95% 12-gon {GAUSSIAN_95_TWELVE_GON_MULTIPLIER} -> {:.3}", coverage_at(&train, shipped), coverage_at(&train, GAUSSIAN_95_TWELVE_GON_MULTIPLIER));
    println!("  (reference) one pooled multiplier for {target:.3} would be {fitted:.3}; the shipped multiplier is the Gaussian one and the per-class scales below carry the fit");
    let shipped_scales = FixCovarianceCalibrationConfig::default();
    for (label, inside, shipped_scale) in
        [("inside", true, shipped_scales.inside_layout_sigma_scale), ("outside", false, shipped_scales.outside_layout_sigma_scale)]
    {
        // Split by the host's own classification: that decides which scale applies.
        let class: Vec<f64> = scores.iter().filter(|s| s.host_classified_inside == inside).map(|s| s.relative_frame_minimum_multiplier).collect();
        let own_fit = fit_parameter_for_coverage(&class, target);
        println!(
            "    {label}: {} fixes, gauge median {:.2}, P95 {:.2}; coverage at shipped {shipped:.3}: {:.3}",
            class.len(),
            median(&class),
            percentile(&class, 0.95),
            coverage_at(&class, shipped),
        );
        // Gauges scale with 1/scale, so the scale that puts this class's
        // target quantile on the shipped Gaussian multiplier is:
        println!("    FITTED {label}_layout_sigma_scale: {:.3}", shipped_scale * own_fit / shipped);
    }
    let world = gauges(&scores, None, true);
    println!("  world frame (published polygon): gauge median {:.2}, P95 {:.2}; coverage at shipped {:.3}", median(&world), percentile(&world, 0.95), coverage_at(&world, shipped));
    let held_out = mesh_scores(&pull_request_seeds(options.held_out_mesh, &[]), options.threads);
    for (label, inside) in [("all", None), ("inside", Some(true)), ("outside", Some(false))] {
        let class = gauges(&held_out, inside, false);
        println!("  held-out {label} ({} fixes): coverage at shipped {:.3}", class.len(), coverage_at(&class, shipped));
    }
    let held_world = gauges(&held_out, None, true);
    println!("  held-out world frame: coverage at shipped {:.3}", coverage_at(&held_world, shipped));
}

fn calibrate_srp(options: &Options, target: f64) {
    let gate_config = SimGateConfig::default();
    let trial_config = SrpTrialConfig::default();
    let shipped = LayoutSrpConfig::default().scoring.likelihood_temperature;
    let ladder = temperature_ladder(TEMPERATURE_LADDER_LOW, TEMPERATURE_LADDER_HIGH, TEMPERATURE_LADDER_STEPS);
    let started = std::time::Instant::now();
    let results = run_trials_in_parallel(&training_seeds(options.srp_trials), options.threads, |seed| {
        let render_started = std::time::Instant::now();
        let trial = render_srp_trial(seed, &trial_config, &gate_config.geometry, &gate_config.perturbation)?;
        let render_s = render_started.elapsed().as_secs_f64();
        let minimum = trial.minimum_covering_temperature(&ladder)?;
        let at_shipped = trial.evaluate_at_temperature(shipped)?;
        Ok::<_, String>((trial.is_inside, minimum, render_s, at_shipped))
    });
    let mut minimums = Vec::new();
    for (seed, result) in &results {
        match result {
            Ok((inside, minimum, render_s, outcome)) => {
                minimums.push(*minimum);
                println!(
                    "  srp train seed {seed}: inside {inside} min T {minimum:.4} render {render_s:.2} s solve {:.3} s err {:.2} m bearing {:.2} deg fan {}",
                    outcome.solve_seconds, outcome.horizontal_error_m, outcome.bearing_error_deg, outcome.drew_bearing_fan
                );
            }
            Err(error) => println!("  srp train seed {seed}: ERROR {error}"),
        }
    }
    let fitted = fit_parameter_for_coverage(&minimums, target);
    println!("srp temperature: {} training trials in {:.1} s", minimums.len(), started.elapsed().as_secs_f64());
    println!("  training coverage at shipped T {shipped}: {:.3}", coverage_at(&minimums, shipped));
    println!("  FITTED likelihood_temperature for {target:.3}: {fitted:.4} (training coverage {:.3})", coverage_at(&minimums, fitted));
    let held_out = run_trials_in_parallel(&pull_request_seeds(options.held_out_srp, &[]), options.threads, |seed| {
        let trial = render_srp_trial(seed, &trial_config, &gate_config.geometry, &gate_config.perturbation)?;
        trial.evaluate_at_temperature(fitted)
    });
    let covered: Vec<bool> = held_out.iter().filter_map(|(_, r)| r.as_ref().ok().map(|o| o.drawn_region_covers_truth)).collect();
    println!("  held-out ({} trials): coverage at fitted {:.3}", covered.len(), covered.iter().filter(|c| **c).count() as f64 / covered.len().max(1) as f64);
    // Sharpness: a coverage reached by smearing is not a calibration.
    let inside: Vec<_> = held_out.iter().filter(|(seed, _)| seed % 2 == 0).filter_map(|(_, r)| r.as_ref().ok()).collect();
    let areas: Vec<f64> = inside.iter().map(|o| o.region_area_m2).collect();
    let fans = inside.iter().filter(|o| o.drew_bearing_fan).count();
    println!("  held-out inside sharpness at fitted: {fans}/{} bearing fans, median region area {:.1} m2", inside.len(), median(&areas));
}

fn main() {
    let options = parse_options();
    let target = SimGateConfig::default().coverage.calibration_target_coverage;
    if options.only.as_deref() != Some("srp") {
        calibrate_mesh(&options, target);
    }
    if options.only.as_deref() != Some("mesh") {
        calibrate_srp(&options, target);
    }
}
