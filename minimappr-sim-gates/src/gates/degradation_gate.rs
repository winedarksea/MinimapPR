//! Graceful degradation through the arrival localizer. Regions are the LM
//! covariance's Gaussian 95 % ellipse (declared σ = true σ here; NEES is
//! gated by `monte_carlo_accuracy`), distances are Mahalanobis.
//! - Muting one of 7 phones grows the region but leaves the estimate inside
//!   the old region (aggregate rate), and the new region always holds the
//!   old estimate (information only dropped, so it must agree).
//! - A pocketed phone (late arrival) leaves the estimate in the clean
//!   region (aggregate), and a *decisive* delay (≥ 8 pair σ) must never
//!   drag it catastrophically: the robust loss has to catch it.
//! - Dropping phones one by one (7 → 3): σ must not shrink (beyond
//!   linearization slack) and every returned fix must hold the truth
//!   within 4σ — few-node geometry is refused or honest, never confident.

use minimappr_core::mesh::arrival_time_localizer::ArrivalTimeLocalizer;

use super::GateRun;
use crate::arrival_trials::{solve_trial, ArrivalTrial, ScoredArrivalFix};
use crate::gate_seeds::{run_trials_in_parallel, GateOutcome};
use crate::statistics::{chi_square_two_dof_quantile, median};
use crate::trial_geometry::{inside_source, mahalanobis_squared_2d, random_phone_layout, uniform_in};

const MINIMUM_NODES: usize = 3;
const REGION_PROBABILITY: f64 = 0.95;

/// Mahalanobis distance (σ) of `point` from `fix`.
fn sigma_distance(point_xy_m: [f64; 2], fix: &ScoredArrivalFix) -> f64 {
    let position = fix.fix.fix.position_m;
    mahalanobis_squared_2d([point_xy_m[0] - position[0], point_xy_m[1] - position[1]], fix.horizontal_covariance_m2).sqrt()
}

fn horizontal(fix: &ScoredArrivalFix) -> [f64; 2] {
    [fix.fix.fix.position_m[0], fix.fix.fix.position_m[1]]
}

#[derive(Default)]
struct DegradationTrial {
    problems: Vec<String>,
    muted_outside_old_region: bool,
    pocketed_outside_clean_region: bool,
    pocketed_rejected: bool,
    pocket_was_decisive: bool,
    sigma_shrink_steps: usize,
    removal_steps: usize,
    errors_by_node_count: Vec<(usize, f64)>,
    refusals_by_node_count: Vec<usize>,
}

fn run_trial(seed: u64, run: &GateRun, localizer: &ArrivalTimeLocalizer) -> DegradationTrial {
    let config = &run.config;
    let degradation = &config.degradation;
    let region_sigma = chi_square_two_dof_quantile(REGION_PROBABILITY).sqrt();
    let catastrophic = degradation.catastrophic_mahalanobis_sigma;
    let aperture = uniform_in(seed, "degradation_aperture", 0, config.geometry.aperture_range_m);
    let layout = random_phone_layout(seed, degradation.full_node_count, aperture, &config.geometry);
    let trial = ArrivalTrial::new(seed, layout.clone(), inside_source(seed, &layout, aperture, &config.geometry));
    let truth = [trial.source_position_m[0], trial.source_position_m[1]];
    let mut result = DegradationTrial::default();
    let all = trial.all_node_indices();
    let Ok(full) = solve_trial(&trial, &all, localizer, &config.perturbation) else {
        result.problems.push("full 7-phone layout gave no fix".into());
        return result;
    };
    let victim = (uniform_in(seed, "degradation_victim", 0, [0.0, 1.0]) * all.len() as f64) as usize % all.len();
    let without_victim: Vec<usize> = all.iter().copied().filter(|n| *n != victim).collect();

    match solve_trial(&trial, &without_victim, localizer, &config.perturbation) {
        Ok(muted) => {
            result.muted_outside_old_region = sigma_distance(horizontal(&muted), &full) > region_sigma;
            let old_in_new = sigma_distance(horizontal(&full), &muted);
            if old_in_new > catastrophic {
                result.problems.push(format!("muting node {victim}: the 6-phone fix puts the 7-phone estimate {old_in_new:.1}σ away"));
            }
        }
        Err(error) => result.problems.push(format!("6 of 7 phones gave no fix: {error:?}")),
    }

    let perturbation = &config.perturbation;
    let pair_sigma_s = (2.0 * (perturbation.arrival_pick_std_s.hypot(perturbation.clock_residual_std_s)).powi(2)
        + 2.0 * (perturbation.layout_position_std_m.max(perturbation.phone_height_std_m) / localizer.sound_speed_mps).powi(2))
    .sqrt();
    let mut pocketed = trial.clone();
    let delay = uniform_in(seed, "degradation_pocket_delay", 0, degradation.pocketed_delay_range_s);
    pocketed.extra_arrival_delay_s[victim] = delay;
    result.pocket_was_decisive = delay >= degradation.decisive_outlier_pair_sigma * pair_sigma_s;
    match solve_trial(&pocketed, &all, localizer, perturbation) {
        Ok(fix) => {
            let distance = sigma_distance(horizontal(&fix), &full);
            result.pocketed_outside_clean_region = distance > region_sigma;
            result.pocketed_rejected = !fix.fix.used_arrival_indices.contains(&victim);
            if result.pocket_was_decisive && distance > catastrophic {
                result.problems.push(format!(
                    "pocketed node {victim} (+{:.2} ms = {:.1} pair σ) dragged the estimate {distance:.1}σ (rejected {})",
                    delay * 1e3,
                    delay / pair_sigma_s,
                    result.pocketed_rejected
                ));
            }
        }
        Err(error) => result.problems.push(format!("pocketed phone broke the solve: {error:?}")),
    }

    let mut order = all.clone();
    order.sort_by_key(|node| (uniform_in(seed, "degradation_order", *node as u64, [0.0, 1.0]) * 1e9) as u64);
    let mut kept = all.clone();
    let mut previous = full;
    result.errors_by_node_count.push((kept.len(), previous.horizontal_error_m));
    for node in order.into_iter().take(degradation.full_node_count - MINIMUM_NODES) {
        kept.retain(|k| *k != node);
        match solve_trial(&trial, &kept, localizer, perturbation) {
            Ok(fix) => {
                result.removal_steps += 1;
                let (old_sigma, new_sigma) = (previous.fix.fix.horizontal_major_sigma_m(), fix.fix.fix.horizontal_major_sigma_m());
                if new_sigma < (1.0 - degradation.maximum_sigma_shrink_fraction) * old_sigma {
                    result.sigma_shrink_steps += 1;
                }
                let truth_distance = sigma_distance(truth, &fix);
                if truth_distance > catastrophic {
                    result.problems.push(format!(
                        "{} phones: confident miss, truth {truth_distance:.1}σ out (error {:.2} m, σ {new_sigma:.2} m)",
                        kept.len(),
                        fix.horizontal_error_m
                    ));
                }
                result.errors_by_node_count.push((kept.len(), fix.horizontal_error_m));
                previous = fix;
            }
            Err(_) => result.refusals_by_node_count.push(kept.len()),
        }
    }
    result
}

pub fn graceful_degradation(run: &GateRun) -> GateOutcome {
    let localizer = ArrivalTimeLocalizer::default();
    let results = run_trials_in_parallel(&run.seeds, run.threads, |seed| run_trial(seed, run, &localizer));
    let mut outcome = GateOutcome::new("graceful_degradation", results.len());
    for (seed, trial) in &results {
        for problem in &trial.problems {
            outcome.fail_seed(*seed, problem.clone());
        }
    }
    let degradation = &run.config.degradation;
    let count = results.len().max(1) as f64;
    let muted_outside = results.iter().filter(|(_, t)| t.muted_outside_old_region).count();
    let pocketed_outside = results.iter().filter(|(_, t)| t.pocketed_outside_clean_region).count();
    let decisive = results.iter().filter(|(_, t)| t.pocket_was_decisive).count();
    let rejected_decisive = results.iter().filter(|(_, t)| t.pocket_was_decisive && t.pocketed_rejected).count();
    let shrink_steps: usize = results.iter().map(|(_, t)| t.sigma_shrink_steps).sum();
    let removal_steps: usize = results.iter().map(|(_, t)| t.removal_steps).sum();
    let shrink_fraction = shrink_steps as f64 / removal_steps.max(1) as f64;
    let limit = degradation.maximum_outside_previous_region_fraction;
    outcome.summary(format!("muted phone: estimate outside old 95 % region {muted_outside}/{} (limit {limit})", results.len()));
    outcome.summary(format!(
        "pocketed phone: outside clean 95 % region {pocketed_outside}/{}; decisive delays {decisive}, of which rejected {rejected_decisive}",
        results.len()
    ));
    outcome.summary(format!(
        "σ shrank > {:.0} % on {shrink_steps}/{removal_steps} node removals ({shrink_fraction:.3}, limit {})",
        degradation.maximum_sigma_shrink_fraction * 100.0,
        degradation.maximum_sigma_shrink_step_fraction
    ));
    for node_count in (MINIMUM_NODES..=degradation.full_node_count).rev() {
        let errors: Vec<f64> =
            results.iter().flat_map(|(_, t)| t.errors_by_node_count.iter().filter(|(n, _)| *n == node_count).map(|(_, e)| *e)).collect();
        let refusals = results.iter().filter(|(_, t)| t.refusals_by_node_count.contains(&node_count)).count();
        outcome.summary(format!("{node_count} phones: median error {:.2} m over {} fixes, {refusals} refused", median(&errors), errors.len()));
    }
    let enough = results.len() >= run.config.coverage.minimum_trials_for_aggregate_gate;
    outcome.check_aggregate(muted_outside as f64 / count <= limit, enough, || format!("muted phone moved the estimate out of its old region in {muted_outside}/{}", results.len()));
    outcome.check_aggregate(pocketed_outside as f64 / count <= limit, enough, || format!("pocketed phone moved the estimate out of the clean region in {pocketed_outside}/{}", results.len()));
    outcome.check_aggregate(shrink_fraction <= degradation.maximum_sigma_shrink_step_fraction, enough, || {
        format!("σ shrank on {shrink_fraction:.3} of node removals (information cannot grow)")
    });
    outcome.contributing_seeds = results
        .iter()
        .filter(|(_, t)| t.muted_outside_old_region || t.pocketed_outside_clean_region || t.sigma_shrink_steps > 0)
        .map(|(s, _)| *s)
        .collect();
    outcome
}
