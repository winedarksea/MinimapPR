//! Likelihood-contour coverage on held-out seeds, with the *shipped*
//! calibrated parameters (fitted on training seeds by
//! `examples/calibrate_region_parameters.rs`).
//!
//! Coverage rule: with ≥ `STRICT_TRIAL_COUNT` trials the point estimate must
//! sit in [90 %, 98 %]; with fewer, the Wilson interval (2σ) must overlap
//! the band (the strategy's note: 100 trials have a ±2.2-point binomial
//! spread, so smaller fixed sets can only be checked for consistency).

use minimappr_core::mesh::config::EntityContourConfig;
use minimappr_core::srp_layout::LayoutSrpConfig;

use super::GateRun;
use crate::gate_config::CoverageGateConfig;
use crate::gate_seeds::{run_trials_in_parallel, GateOutcome};
use crate::mesh_region_trials::{run_contour_session, MeshSessionConfig};
use crate::srp_trials::{render_srp_trial, SrpTrialConfig};
use crate::statistics::{median, percentile, wilson_interval};

const STRICT_TRIAL_COUNT: usize = 100;
const WILSON_STANDARD_ERRORS: f64 = 2.0;

/// Apply the coverage rule to `covered` of `trials`.
pub fn check_coverage(outcome: &mut GateOutcome, label: &str, covered: usize, trials: usize, config: &CoverageGateConfig) {
    let rate = covered as f64 / trials.max(1) as f64;
    let [low, high] = wilson_interval(covered, trials, WILSON_STANDARD_ERRORS);
    outcome.summary(format!("{label}: coverage {covered}/{trials} = {rate:.3} (Wilson 2σ [{low:.3}, {high:.3}])"));
    let enough = trials >= config.minimum_trials_for_aggregate_gate;
    if trials >= STRICT_TRIAL_COUNT {
        outcome.check_aggregate((config.minimum_coverage..=config.maximum_coverage).contains(&rate), enough, || {
            format!("{label} coverage {rate:.3} outside [{}, {}]", config.minimum_coverage, config.maximum_coverage)
        });
    } else {
        outcome.check_aggregate(high >= config.minimum_coverage && low <= config.maximum_coverage, enough, || {
            format!("{label} coverage {rate:.3} inconsistent with [{}, {}] (Wilson [{low:.3}, {high:.3}])", config.minimum_coverage, config.maximum_coverage)
        });
    }
}

/// End-to-end Layer 2 fixes against the published polygon.
pub fn mesh_contour_coverage(run: &GateRun) -> GateOutcome {
    let contour = EntityContourConfig::default();
    let session = MeshSessionConfig::default();
    let results = run_trials_in_parallel(&run.seeds, run.threads, |seed| run_contour_session(seed, &contour, &session));
    let all_scores: Vec<_> = results.iter().flat_map(|(_, s)| s.iter().cloned()).collect();
    // A flagged mirror (low reported chirality confidence) is exempt, as in
    // the mesh invariants; an unflagged mirror counts as a miss.
    let flagged = all_scores.iter().filter(|s| s.is_flagged_mirror).count();
    let unflagged_mirrors = all_scores.iter().filter(|s| s.layout_mirrored && !s.is_flagged_mirror).count();
    let scores: Vec<_> = all_scores.into_iter().filter(|s| !s.is_flagged_mirror).collect();
    let mut outcome = GateOutcome::new("mesh_contour_coverage", scores.len());
    outcome.summary(format!("{flagged} flagged-mirror fixes exempt, {unflagged_mirrors} unflagged mirrored fixes counted"));
    let multiplier = contour.contour_sigma_multiplier;
    let covered = |inside: Option<bool>| {
        let selected: Vec<_> = scores.iter().filter(|s| inside.is_none_or(|i| s.is_inside_call == i)).collect();
        (selected.iter().filter(|s| s.relative_frame_minimum_multiplier <= multiplier).count(), selected.len())
    };
    let (all_covered, all) = covered(None);
    check_coverage(&mut outcome, &format!("all fixes, σ multiplier {multiplier}"), all_covered, all, &run.config.coverage);
    for (label, inside) in [("inside calls", true), ("outside calls", false)] {
        let (c, n) = covered(Some(inside));
        outcome.summary(format!("{label}: {c}/{n} = {:.3} (reported, not gated separately)", c as f64 / n.max(1) as f64));
    }
    let world_covered = scores.iter().filter(|s| s.world_frame_minimum_multiplier <= multiplier).count();
    outcome.summary(format!("world frame (anchor error charged, diagnostic): {world_covered}/{all}"));
    let errors: Vec<f64> = scores.iter().map(|s| s.relative_frame_error_m).collect();
    outcome.summary(format!("relative-frame error median {:.2} m, P95 {:.2} m", median(&errors), percentile(&errors, 0.95)));
    let fixes_per_session = all as f64 / results.len().max(1) as f64;
    outcome.summary(format!("{} sessions, {fixes_per_session:.2} fixes per session", results.len()));
    outcome.contributing_seeds = results
        .iter()
        .filter(|(_, s)| s.iter().any(|f| f.relative_frame_minimum_multiplier > multiplier))
        .map(|(seed, _)| *seed)
        .collect();
    outcome
}

/// Audio-rendered SRP trials: drawn contour / fan coverage at the shipped
/// temperature, plus SRP accuracy against the strategy gates.
pub fn srp_contour_coverage(run: &GateRun) -> GateOutcome {
    let config = &run.config;
    let trial_config = SrpTrialConfig::default();
    let temperature = LayoutSrpConfig::default().scoring.likelihood_temperature;
    let results = run_trials_in_parallel(&run.seeds, run.threads, |seed| {
        let trial = render_srp_trial(seed, &trial_config, &config.geometry, &config.perturbation)?;
        let outcome = trial.evaluate_at_temperature(temperature)?;
        Ok::<_, String>((trial.is_inside, outcome))
    });
    let mut outcome = GateOutcome::new("srp_contour_coverage", results.len());
    let mut solved = Vec::new();
    for (seed, result) in &results {
        match result {
            Ok(value) => solved.push((*seed, value.0, value.1.clone())),
            Err(error) => outcome.fail_seed(*seed, format!("SRP trial failed: {error}")),
        }
    }
    let covered = solved.iter().filter(|(_, _, o)| o.drawn_region_covers_truth).count();
    check_coverage(&mut outcome, &format!("drawn region at T = {temperature}"), covered, solved.len(), &config.coverage);
    let enough = solved.len() >= config.coverage.minimum_trials_for_aggregate_gate;
    let inside_errors: Vec<f64> = solved.iter().filter(|(_, inside, _)| *inside).map(|(_, _, o)| o.horizontal_error_m).collect();
    let outside_bearings: Vec<f64> = solved.iter().filter(|(_, inside, _)| !*inside).map(|(_, _, o)| o.bearing_error_deg).collect();
    let (inside_median, inside_p95) = (median(&inside_errors), percentile(&inside_errors, 0.95));
    let outside_p95 = percentile(&outside_bearings, 0.95);
    let fans = solved.iter().filter(|(_, _, o)| o.drew_bearing_fan).count();
    let incomplete = solved.iter().filter(|(_, _, o)| !o.region_refinement_complete).count();
    let solve_times: Vec<f64> = solved.iter().map(|(_, _, o)| o.solve_seconds).collect();
    outcome.summary(format!("inside error median {inside_median:.3} m, P95 {inside_p95:.3} m ({} trials)", inside_errors.len()));
    outcome.summary(format!("outside (40–150 m) bearing median {:.2} deg, P95 {outside_p95:.2} deg; {fans} fans, {incomplete} region budgets exhausted", median(&outside_bearings)));
    outcome.summary(format!("solve time median {:.3} s, max {:.3} s", median(&solve_times), percentile(&solve_times, 1.0)));
    let inside_fans = solved.iter().filter(|(_, inside, o)| *inside && o.drew_bearing_fan).count();
    let inside_fan_fraction = inside_fans as f64 / inside_errors.len().max(1) as f64;
    let inside_areas: Vec<f64> = solved.iter().filter(|(_, inside, _)| *inside).map(|(_, _, o)| o.region_area_m2).collect();
    let inside_median_area = median(&inside_areas);
    outcome.summary(format!("inside sharpness: {inside_fans} fans ({inside_fan_fraction:.2}), median region area {inside_median_area:.2} m²"));
    outcome.check_aggregate(inside_fan_fraction <= config.coverage.maximum_inside_fan_fraction, enough, || {
        format!("inside layout drew a bearing fan in {inside_fan_fraction:.2} of trials (coverage by smearing)")
    });
    outcome.check_aggregate(inside_median_area <= config.coverage.maximum_inside_median_region_area_m2, enough, || {
        format!("inside median region area {inside_median_area:.1} m² (coverage by smearing)")
    });
    outcome.check_aggregate(inside_median < config.accuracy.inside_median_error_m, enough, || format!("SRP inside median {inside_median:.3} m"));
    outcome.check_aggregate(inside_p95 < config.accuracy.inside_p95_error_m, enough, || format!("SRP inside P95 {inside_p95:.3} m"));
    outcome.check_aggregate(outside_p95 < config.accuracy.outside_p95_bearing_error_deg, enough, || format!("SRP outside bearing P95 {outside_p95:.2} deg"));
    outcome.contributing_seeds = solved
        .iter()
        .filter(|(_, inside, o)| !o.drawn_region_covers_truth || (*inside && o.horizontal_error_m >= config.accuracy.inside_p95_error_m))
        .map(|(seed, _, _)| *seed)
        .collect();
    outcome
}
