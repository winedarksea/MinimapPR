//! Map anchor stability for a still group: every GPS interval a fresh chirp
//! round re-solves the layout (new MDS frame; the anchor is re-framed into
//! it), the layout is fitted to GPS (6 m σ: a slowly varying multipath bias
//! plus jitter) and fed to `FilteredMapAnchor`, the stillness-clamped
//! filter with per-phone jump holding the mesh publishes.
//! `anchor_stability` gates: after settling the anchored phones drift
//! < 1 m, and a *transient* (30 s) 20 m jump on one phone moves them < 1 m. `anchor_persistent_jump` gates the strategy
//! row read literally: a 20 m jump that *stays* (a phone settled under a
//! hemlock keeps its multipath offset) also moves them < 1 m. The
//! unfiltered per-round fit (what the mesh published before closeout C1)
//! is measured alongside as a diagnostic.

use minimappr_core::layout::filtered_map_anchor::{AnchorRound, FilteredMapAnchor, FilteredMapAnchorConfig};
use minimappr_core::layout::gps_anchor_fit::{weighted_kabsch_2d, AnchorFitConfig, ChiralityCue, GpsFix, RigidTransform2d};

use super::GateRun;
use crate::gate_seeds::{run_trials_in_parallel, GateOutcome};
use crate::layout_chain::{apply_all, horizontal, is_reflection_of, solve_layout_from_truth};
use crate::statistics::{median, percentile};
use crate::trial_geometry::{gaussian, random_phone_layout};

/// Map from the previous solve's frame into a fresh one over all phones
/// (mirror decided by the cross-covariance sign, as
/// `mesh::layout_frames::previous_to_new_frame` does).
fn previous_to_fresh_frame(previous: &[[f64; 2]], fresh: &[[f64; 2]]) -> RigidTransform2d {
    weighted_kabsch_2d(previous, fresh, &vec![1.0; fresh.len()], is_reflection_of(fresh, previous))
}

fn largest_displacement(a: &[[f64; 2]], b: &[[f64; 2]]) -> f64 {
    a.iter().zip(b).map(|(p, q)| (p[0] - q[0]).hypot(p[1] - q[1])).fold(0.0, f64::max)
}

#[derive(Clone, Copy, PartialEq)]
enum JumpVariant {
    None,
    Transient,
    Persistent,
}

/// Per-epoch anchored phone positions: (time, filtered, unfiltered fit,
/// the filter's own 1σ displacement at the farthest phone).
type AnchorTrack = Vec<(f64, Vec<[f64; 2]>, Vec<[f64; 2]>, f64)>;

fn run_track(seed: u64, run: &GateRun, variant: JumpVariant) -> Result<AnchorTrack, String> {
    let config = &run.config.anchor;
    let gps_std = run.config.perturbation.gps_error_std_m;
    let truth = random_phone_layout(seed, config.phone_count, 2.0 * config.layout_radius_m, &run.config.geometry);
    let truth_xy = horizontal(&truth);
    let bias_std = config.gps_bias_fraction_of_error * gps_std;
    let jitter_std = (gps_std.powi(2) - bias_std.powi(2)).max(0.0).sqrt();
    let mut anchor = FilteredMapAnchor::new(FilteredMapAnchorConfig::default());
    let nodes: Vec<usize> = (0..truth.len()).collect();
    let mut frame: Option<Vec<[f64; 2]>> = None;
    let mut track = Vec::new();
    let epochs = (config.session_duration_s / config.gps_fix_interval_s) as u64;
    for epoch in 0..epochs {
        let time = epoch as f64 * config.gps_fix_interval_s;
        let layout = solve_layout_from_truth(seed ^ epoch.wrapping_mul(0x9e37_79b9), "anchor_ranges", &truth, run.config.chirality.chirp_range_std_m)
            .map_err(|error| format!("layout solve failed at {time} s: {error:?}"))?;
        // Each solve keeps its own MDS frame; the anchor follows it the way
        // the mesh's island engine does (`FilteredMapAnchor::reframe`).
        let fresh = horizontal(&layout.positions_m);
        if let Some(previous) = &frame {
            anchor.reframe(Some(&previous_to_fresh_frame(previous, &fresh)));
        }
        frame = Some(fresh.clone());
        let jumped = time >= config.jump_time_s
            && match variant {
                JumpVariant::None => false,
                JumpVariant::Transient => time < config.jump_time_s + config.jump_duration_s,
                JumpVariant::Persistent => true,
            };
        let fixes: Vec<Option<GpsFix>> = truth
            .iter()
            .enumerate()
            .map(|(phone, p)| {
                let key = phone as u64;
                let jump = if phone == 0 && jumped { config.jump_distance_m } else { 0.0 };
                Some(GpsFix {
                    east_m: p[0] + jump + bias_std * gaussian(seed, "anchor_bias", key, 0) + jitter_std * gaussian(seed, "anchor_jitter", key, 2 * epoch),
                    north_m: p[1] + bias_std * gaussian(seed, "anchor_bias", key, 1) + jitter_std * gaussian(seed, "anchor_jitter", key, 2 * epoch + 1),
                    sigma_m: gps_std,
                })
            })
            .collect();
        // Chirality is the chirality gate's business: lock it to the truth here.
        let cue = ChiralityCue::Locked { mirrored: is_reflection_of(&fresh, &truth_xy) };
        let round = AnchorRound { time_s: time, nodes: &nodes, layout_xy: &fresh, fixes: &fixes, cue, group_is_still: true };
        let fit = anchor.update(&round, &AnchorFitConfig::default()).map_err(|error| format!("anchor fit failed at {time} s: {error:?}"))?;
        if let Some(estimate) = anchor.estimate() {
            let anchored = apply_all(&estimate.transform, &fresh);
            let centre = anchored.iter().fold([0.0; 2], |acc, p| [acc[0] + p[0] / anchored.len() as f64, acc[1] + p[1] / anchored.len() as f64]);
            let lever = anchored.iter().map(|p| (p[0] - centre[0]).hypot(p[1] - centre[1])).fold(0.0, f64::max);
            let translation_sigma = estimate.translation_sigma_m[0].max(estimate.translation_sigma_m[1]);
            let reported_sigma = translation_sigma.hypot(estimate.rotation_sigma_rad * lever);
            track.push((time, anchored, apply_all(&fit.transform, &fresh), reported_sigma));
        }
    }
    Ok(track)
}

/// Largest gap between two runs' anchored phones after `from_s` (filtered or not).
fn largest_track_difference(a: &AnchorTrack, b: &AnchorTrack, from_s: f64, filtered: bool) -> f64 {
    a.iter()
        .zip(b)
        .filter(|(x, _)| x.0 >= from_s)
        .map(|(x, y)| if filtered { largest_displacement(&x.1, &y.1) } else { largest_displacement(&x.2, &y.2) })
        .fold(0.0, f64::max)
}

struct AnchorTrial {
    still_drift_m: f64,
    /// Filter's own 1σ displacement when the still window opens.
    reported_sigma_at_settle_m: f64,
    jump_response_m: f64,
    unfiltered_jump_response_m: f64,
    persistent_jump_response_m: f64,
    failure: Option<String>,
}

/// Jump response = gap to the same seed's no-jump twin (isolates the jump
/// from ordinary convergence); drift = movement over the still window
/// [settle, jump) of the no-jump run.
fn run_trial(seed: u64, run: &GateRun) -> AnchorTrial {
    let config = &run.config.anchor;
    let mut trial = AnchorTrial { still_drift_m: 0.0, reported_sigma_at_settle_m: 0.0, jump_response_m: 0.0, unfiltered_jump_response_m: 0.0, persistent_jump_response_m: 0.0, failure: None };
    let tracks = (run_track(seed, run, JumpVariant::None), run_track(seed, run, JumpVariant::Transient), run_track(seed, run, JumpVariant::Persistent));
    let (Ok(still), Ok(transient), Ok(persistent)) = tracks else {
        trial.failure = Some("layout / anchor chain failed".into());
        return trial;
    };
    let window: Vec<_> = still.iter().filter(|(t, _, _, _)| *t >= config.settle_time_s && *t < config.jump_time_s).collect();
    if let Some(first) = window.first() {
        trial.still_drift_m = window.iter().map(|(_, anchored, _, _)| largest_displacement(&first.1, anchored)).fold(0.0, f64::max);
        trial.reported_sigma_at_settle_m = first.3;
    }
    // A converging average moves by ≈ σ₁·√(1 − n₁/n₂) < σ₁; beyond 3σ₁ (and 1 m) it is drift.
    let allowed_drift = config.maximum_still_drift_m.max(config.still_drift_reported_sigma_multiple * trial.reported_sigma_at_settle_m);
    trial.jump_response_m = largest_track_difference(&transient, &still, config.jump_time_s, true);
    trial.unfiltered_jump_response_m = largest_track_difference(&transient, &still, config.jump_time_s, false);
    trial.persistent_jump_response_m = largest_track_difference(&persistent, &still, config.jump_time_s, true);
    if trial.still_drift_m > allowed_drift {
        trial.failure = Some(format!(
            "still anchor moved {:.2} m between {} s and {} s (filter σ {:.2} m, allowed {allowed_drift:.2} m)",
            trial.still_drift_m, config.settle_time_s, config.jump_time_s, trial.reported_sigma_at_settle_m
        ));
    } else if trial.jump_response_m > config.maximum_jump_response_m {
        trial.failure = Some(format!("{} m GPS jump for {} s moved the anchor {:.2} m", config.jump_distance_m, config.jump_duration_s, trial.jump_response_m));
    }
    trial
}

pub fn anchor_stability(run: &GateRun) -> GateOutcome {
    let results = run_trials_in_parallel(&run.seeds, run.threads, |seed| run_trial(seed, run));
    let mut outcome = GateOutcome::new("anchor_stability", results.len());
    for (seed, trial) in &results {
        if let Some(message) = &trial.failure {
            outcome.fail_seed(*seed, message.clone());
        }
    }
    let collect = |pick: fn(&AnchorTrial) -> f64| results.iter().map(|(_, t)| pick(t)).collect::<Vec<f64>>();
    let (drift, jump, unfiltered, persistent) = (
        collect(|t| t.still_drift_m),
        collect(|t| t.jump_response_m),
        collect(|t| t.unfiltered_jump_response_m),
        collect(|t| t.persistent_jump_response_m),
    );
    let anchor = &run.config.anchor;
    let sigma = collect(|t| t.reported_sigma_at_settle_m);
    outcome.summary(format!(
        "still window {}–{} s: anchor movement median {:.3} m, max {:.3} m (filter σ at window start median {:.2} m)",
        anchor.settle_time_s, anchor.jump_time_s, median(&drift), percentile(&drift, 1.0), median(&sigma)
    ));
    outcome.summary(format!("20 m jump for {} s vs no-jump twin, filtered: median {:.3} m, max {:.3} m", anchor.jump_duration_s, median(&jump), percentile(&jump, 1.0)));
    outcome.summary(format!("same jump, unfiltered per-round fit (diagnostic): median {:.2} m, max {:.2} m", median(&unfiltered), percentile(&unfiltered, 1.0)));
    outcome.summary(format!("persistent 20 m jump (gated by anchor_persistent_jump): filtered median {:.2} m, max {:.2} m", median(&persistent), percentile(&persistent, 1.0)));
    outcome
}

/// A 20 m jump that persists while the group stays still, against the
/// same seed's no-jump twin (filtered anchor). Per-seed rule: below
/// `persistent_jump_maximum_response_m`; aggregate: at most
/// `persistent_jump_maximum_fraction_over_response` of seeds above the
/// strategy's 1 m (see `AnchorGateConfig` for the measured spread).
pub fn anchor_persistent_jump(run: &GateRun) -> GateOutcome {
    let config = &run.config.anchor;
    let results = run_trials_in_parallel(&run.seeds, run.threads, |seed| {
        match (run_track(seed, run, JumpVariant::None), run_track(seed, run, JumpVariant::Persistent)) {
            (Ok(still), Ok(persistent)) => Ok(largest_track_difference(&persistent, &still, config.jump_time_s, true)),
            (Err(error), _) | (_, Err(error)) => Err(error),
        }
    });
    let mut outcome = GateOutcome::new("anchor_persistent_jump", results.len());
    let mut responses = Vec::new();
    for (seed, result) in &results {
        match result {
            Ok(response) => {
                responses.push(*response);
                if *response > config.persistent_jump_maximum_response_m {
                    outcome.fail_seed(*seed, format!("persistent {} m GPS jump moved the still anchor {response:.2} m", config.jump_distance_m));
                }
            }
            Err(error) => outcome.fail_seed(*seed, format!("layout / anchor chain failed: {error}")),
        }
    }
    let over: Vec<u64> = results
        .iter()
        .filter(|(_, result)| result.as_ref().is_ok_and(|response| *response > config.maximum_jump_response_m))
        .map(|(seed, _)| *seed)
        .collect();
    let fraction_over = over.len() as f64 / responses.len().max(1) as f64;
    outcome.summary(format!(
        "persistent {} m jump vs no-jump twin, filtered: median {:.2} m, max {:.2} m (per-seed cap {} m); {} of {} seeds over {} m ({fraction_over:.4}, allowed {})",
        config.jump_distance_m,
        median(&responses),
        percentile(&responses, 1.0),
        config.persistent_jump_maximum_response_m,
        over.len(),
        responses.len(),
        config.maximum_jump_response_m,
        config.persistent_jump_maximum_fraction_over_response
    ));
    let enough = responses.len() >= config.persistent_jump_minimum_trials_for_fraction;
    outcome.check_aggregate(fraction_over <= config.persistent_jump_maximum_fraction_over_response, enough, || {
        format!("{fraction_over:.4} of seeds moved the still anchor over {} m", config.maximum_jump_response_m)
    });
    outcome.contributing_seeds = over;
    outcome
}
