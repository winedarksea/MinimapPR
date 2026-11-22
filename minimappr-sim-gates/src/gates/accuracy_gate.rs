//! Monte Carlo accuracy over layouts × ranges × noise through the arrival
//! localizer, plus the LM covariance's own calibration (NEES) under
//! correctly declared noise.
//!
//! Per seed: one random layout (4–8 phones, 15–40 m), an inside source and
//! sources at 50 / 100 / 200 m, all with the realistic residuals. Gated:
//! inside median and P95 horizontal error, P95 bearing error at 50 m, and a
//! fix for every well-posed inside trial. NEES: the mean of Δᵀ Σ⁻¹ Δ over
//! inside trials should be ≈ 2 (two horizontal dof); the band is wide
//! because the height-band marginalization deliberately inflates Σ.

use minimappr_core::mesh::arrival_time_localizer::ArrivalTimeLocalizer;

use super::GateRun;
use crate::arrival_trials::{solve_trial, ArrivalTrial};
use crate::gate_seeds::{run_trials_in_parallel, GateOutcome};
use crate::statistics::{median, percentile};
use crate::trial_geometry::{count_in, inside_source, random_phone_layout, source_at_range, uniform_in};

/// Distinct trial seeds per range share a layout but not noise.
const TRIALS_PER_SEED_STRIDE: u64 = 16;
/// Mean NEES band for 2 dof; above → overconfident Σ (a real failure).
const MAXIMUM_MEAN_NEES: f64 = 3.0;
const MINIMUM_MEAN_NEES: f64 = 0.2;

struct SeedResult {
    inside_error_m: Option<f64>,
    inside_nees: Option<f64>,
    bearing_errors_deg: Vec<(f64, Option<f64>)>,
    failure: Option<String>,
}

pub fn monte_carlo_accuracy(run: &GateRun) -> GateOutcome {
    let config = &run.config;
    let localizer = ArrivalTimeLocalizer::default();
    let ranges: Vec<f64> = std::iter::once(config.accuracy.outside_gate_range_m).chain(config.accuracy.reported_outside_ranges_m).collect();
    let results = run_trials_in_parallel(&run.seeds, run.threads, |seed| {
        let node_count = count_in(seed, "accuracy_node_count", config.geometry.node_count_range);
        let aperture = uniform_in(seed, "accuracy_aperture", 0, config.geometry.aperture_range_m);
        let layout = random_phone_layout(seed, node_count, aperture, &config.geometry);
        let inside = ArrivalTrial::new(seed * TRIALS_PER_SEED_STRIDE, layout.clone(), inside_source(seed, &layout, aperture, &config.geometry));
        let mut result = SeedResult { inside_error_m: None, inside_nees: None, bearing_errors_deg: Vec::new(), failure: None };
        match solve_trial(&inside, &inside.all_node_indices(), &localizer, &config.perturbation) {
            Ok(scored) => {
                result.inside_error_m = Some(scored.horizontal_error_m);
                result.inside_nees = Some(scored.mahalanobis_squared);
            }
            Err(error) => result.failure = Some(format!("{node_count} phones, inside source: no fix ({error:?})")),
        }
        for (offset, &range) in ranges.iter().enumerate() {
            let trial =
                ArrivalTrial::new(seed * TRIALS_PER_SEED_STRIDE + 1 + offset as u64, layout.clone(), source_at_range(seed, &layout, range, &config.geometry));
            let bearing = solve_trial(&trial, &trial.all_node_indices(), &localizer, &config.perturbation).ok().map(|s| s.bearing_error_deg);
            result.bearing_errors_deg.push((range, bearing));
        }
        result
    });

    let mut outcome = GateOutcome::new("monte_carlo_accuracy", results.len());
    let enough = results.len() >= config.coverage.minimum_trials_for_aggregate_gate;
    let mut inside_errors = Vec::new();
    let mut nees = Vec::new();
    for (seed, result) in &results {
        if let Some(message) = &result.failure {
            outcome.fail_seed(*seed, message.clone());
        }
        inside_errors.extend(result.inside_error_m);
        nees.extend(result.inside_nees.filter(|value| value.is_finite()));
    }
    let inside_median = median(&inside_errors);
    let inside_p95 = percentile(&inside_errors, 0.95);
    outcome.summary(format!("inside: median {inside_median:.3} m, P95 {inside_p95:.3} m over {} fixes", inside_errors.len()));
    outcome.check_aggregate(inside_median < config.accuracy.inside_median_error_m, enough, || {
        format!("inside median {inside_median:.3} m >= {}", config.accuracy.inside_median_error_m)
    });
    outcome.check_aggregate(inside_p95 < config.accuracy.inside_p95_error_m, enough, || {
        format!("inside P95 {inside_p95:.3} m >= {}", config.accuracy.inside_p95_error_m)
    });
    for &range in &ranges {
        let bearings: Vec<f64> = results.iter().filter_map(|(_, r)| r.bearing_errors_deg.iter().find(|(x, _)| *x == range)?.1).collect();
        let solved = bearings.len();
        let p95 = percentile(&bearings, 0.95);
        outcome.summary(format!("{range:.0} m: bearing median {:.2} deg, P95 {p95:.2} deg, {solved}/{} solved", median(&bearings), results.len()));
        if range == config.accuracy.outside_gate_range_m {
            outcome.check_aggregate(p95 < config.accuracy.outside_p95_bearing_error_deg, enough, || {
                format!("{range} m bearing P95 {p95:.2} deg >= {}", config.accuracy.outside_p95_bearing_error_deg)
            });
            outcome.check_aggregate(solved == results.len(), enough, || format!("{range} m: only {solved}/{} solved", results.len()));
            outcome.contributing_seeds.extend(
                results
                    .iter()
                    .filter(|(_, r)| r.bearing_errors_deg.iter().any(|(x, b)| *x == range && b.is_none_or(|b| b >= config.accuracy.outside_p95_bearing_error_deg)))
                    .map(|(seed, _)| *seed),
            );
        }
    }
    let mean_nees = nees.iter().sum::<f64>() / nees.len().max(1) as f64;
    outcome.summary(format!("LM NEES (2 dof) inside: mean {mean_nees:.2}, median {:.2}", median(&nees)));
    outcome.check_aggregate((MINIMUM_MEAN_NEES..=MAXIMUM_MEAN_NEES).contains(&mean_nees), enough, || {
        format!("mean NEES {mean_nees:.2} outside [{MINIMUM_MEAN_NEES}, {MAXIMUM_MEAN_NEES}]")
    });
    outcome.contributing_seeds.extend(
        results.iter().filter(|(_, r)| r.inside_error_m.is_some_and(|e| e >= config.accuracy.inside_p95_error_m)).map(|(seed, _)| *seed),
    );
    outcome
}
