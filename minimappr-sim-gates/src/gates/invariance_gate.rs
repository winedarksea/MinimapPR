//! Geometric invariance: rotating, translating or mirroring every node and
//! the bird must move the final estimate identically (within 1 mm) and the
//! covariance by the same linear map (eigenvalues within 1e-3 relative).
//!
//! Two chains: the arrival localizer (polar grid seed is *not* invariant;
//! the LM polish must make it so), and layout → GPS anchor (MDS frames are
//! arbitrary; the anchored positions must not be). The audio SRP + LM chain
//! has its own, more expensive gate in `srp_invariance_gate`.

use minimappr_core::layout::gps_anchor_fit::ChiralityCue;
use minimappr_core::mesh::arrival_time_localizer::ArrivalTimeLocalizer;

use super::GateRun;
use crate::arrival_trials::{score_fix, ArrivalTrial};
use crate::gate_seeds::{run_trials_in_parallel, GateOutcome};
use crate::layout_chain::{apply_all, fit_to_gps, gps_fixes, horizontal, solve_layout_from_truth};
use crate::trial_geometry::{
    count_in, inside_source, random_phone_layout, source_at_range, symmetric_eigenvalues_2x2, uniform_in, PlanarRigidMotion,
};

/// Range of the outside source in each invariance trial.
const OUTSIDE_SOURCE_RANGE_M: f64 = 50.0;

fn relative_difference(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.abs().max(b.abs()).max(f64::MIN_POSITIVE)
}

/// Compare one trial against its moved twin; `None` = equivalent.
fn arrival_discrepancy(trial: &ArrivalTrial, motion: &PlanarRigidMotion, run: &GateRun, localizer: &ArrivalTimeLocalizer) -> Option<String> {
    let nodes = trial.all_node_indices();
    let perturbation = &run.config.perturbation;
    let original = localizer.localize(&trial.arrivals(&nodes, None, perturbation, localizer.sound_speed_mps));
    let moved = localizer.localize(&trial.arrivals(&nodes, Some(motion), perturbation, localizer.sound_speed_mps));
    match (original, moved) {
        (Ok(original), Ok(moved)) => {
            let expected = motion.apply(original.fix.position_m);
            let discrepancy = (expected[0] - moved.fix.position_m[0])
                .hypot(expected[1] - moved.fix.position_m[1])
                .hypot(expected[2] - moved.fix.position_m[2]);
            let tolerance = &run.config.invariance;
            if discrepancy > tolerance.maximum_position_discrepancy_m {
                return Some(format!("{motion:?}: estimate moved {discrepancy:.4} m off its transformed twin"));
            }
            let original_sigma = symmetric_eigenvalues_2x2(motion.apply_to_covariance(score_fix(trial, original.clone()).horizontal_covariance_m2));
            let moved_scored = score_fix(trial, moved.clone());
            let moved_sigma = symmetric_eigenvalues_2x2(moved_scored.horizontal_covariance_m2);
            let worst = relative_difference(original_sigma[0], moved_sigma[0]).max(relative_difference(original_sigma[1], moved_sigma[1]));
            (worst > tolerance.maximum_relative_sigma_discrepancy)
                .then(|| format!("{motion:?}: covariance eigenvalues differ by {worst:.2e} relative"))
                .or_else(|| (original.used_arrival_indices != moved.used_arrival_indices).then(|| "different nodes rejected".to_string()))
        }
        (Err(_), Err(_)) => None,
        (original, moved) => Some(format!("{motion:?}: solvable only on one side ({:?} vs {:?})", original.is_ok(), moved.is_ok())),
    }
}

/// Anchored layout positions must move with the world (GPS moved too).
fn anchored_layout_discrepancy(seed: u64, truth: &[[f64; 3]], motion: &PlanarRigidMotion, run: &GateRun) -> Option<String> {
    let range_std = run.config.chirality.chirp_range_std_m;
    let moved_truth: Vec<[f64; 3]> = truth.iter().map(|p| motion.apply(*p)).collect();
    let anchored = |positions: &[[f64; 3]], gps: Vec<_>| -> Option<Vec<[f64; 2]>> {
        let layout = solve_layout_from_truth(seed, "invariance_ranges", positions, range_std).ok()?;
        let xy = horizontal(&layout.positions_m);
        let fit = fit_to_gps(&xy, &gps, ChiralityCue::Unknown).ok()?;
        Some(apply_all(&fit.transform, &xy))
    };
    let gps = gps_fixes(seed, "invariance_gps", 0, truth, run.config.perturbation.gps_error_std_m);
    // The same GPS errors, moved with the world (error vectors rotate too).
    let moved_gps = gps
        .iter()
        .zip(truth)
        .map(|(fix, p)| {
            fix.map(|mut fix| {
                let error = motion.rotate_vector([fix.east_m - p[0], fix.north_m - p[1]]);
                let moved = motion.apply(*p);
                fix.east_m = moved[0] + error[0];
                fix.north_m = moved[1] + error[1];
                fix
            })
        })
        .collect();
    let (Some(original), Some(moved)) = (anchored(truth, gps), anchored(&moved_truth, moved_gps)) else {
        return Some(format!("{motion:?}: layout chain failed on one side"));
    };
    let worst = original
        .iter()
        .zip(&moved)
        .map(|(o, m)| {
            let expected = motion.apply([o[0], o[1], 0.0]);
            (expected[0] - m[0]).hypot(expected[1] - m[1])
        })
        .fold(0.0, f64::max);
    (worst > run.config.invariance.maximum_position_discrepancy_m)
        .then(|| format!("{motion:?}: anchored layout off its transformed twin by {worst:.4} m"))
}

pub fn geometric_invariance(run: &GateRun) -> GateOutcome {
    let config = &run.config;
    let localizer = ArrivalTimeLocalizer::default();
    let results = run_trials_in_parallel(&run.seeds, run.threads, |seed| {
        let node_count = count_in(seed, "invariance_node_count", config.geometry.node_count_range);
        let aperture = uniform_in(seed, "invariance_aperture", 0, config.geometry.aperture_range_m);
        let layout = random_phone_layout(seed, node_count, aperture, &config.geometry);
        let sources = [inside_source(seed, &layout, aperture, &config.geometry), source_at_range(seed, &layout, OUTSIDE_SOURCE_RANGE_M, &config.geometry)];
        let mut problems = Vec::new();
        for mirrored in [false, true] {
            let motion = PlanarRigidMotion::random(seed ^ u64::from(mirrored), config.invariance.maximum_translation_m, mirrored);
            for (index, source) in sources.iter().enumerate() {
                let trial = ArrivalTrial::new(seed * 4 + index as u64, layout.clone(), *source);
                problems.extend(arrival_discrepancy(&trial, &motion, run, &localizer).map(|p| format!("arrival source {index}: {p}")));
            }
            problems.extend(anchored_layout_discrepancy(seed, &layout, &motion, run).map(|p| format!("layout anchor: {p}")));
        }
        problems
    });
    let mut outcome = GateOutcome::new("geometric_invariance", results.len());
    for (seed, problems) in &results {
        for problem in problems {
            outcome.fail_seed(*seed, problem.clone());
        }
    }
    outcome.summary(format!(
        "{} trials x (proper + mirrored motion) x (inside, {OUTSIDE_SOURCE_RANGE_M} m arrival fixes + anchored layout); {} discrepancies",
        results.len(),
        outcome.seed_failures.len()
    ));
    outcome
}
