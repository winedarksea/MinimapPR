//! Point-only localizer bake-off (E8): every candidate on the same seeded
//! point-network scenes, scored alike, decided by a rule fixed in advance.
//!
//! Candidates (`bakeoff_candidate_fix::BakeoffCandidate`):
//! - production: sidecar GCC-PHAT pairs + server Cartesian least squares,
//!   the path a point network runs today (`bakeoff_production_candidate`);
//! - the sidecar's own SRP grid estimate from the same call (diagnostic);
//! - production with the event-span lag cap lifted, and with a repaired
//!   front-end (cap lifted, window tapered, band reaching bird calls):
//!   diagnostics that separate production's front-end from its solver;
//! - shadow onset clustering with the R5 clock policy (`bakeoff_onset_candidate`).
//!
//! Protocol: tuning seeds pick the onset SRP grid reach (the only knob tuned;
//! production runs at its shipped defaults), then the held-out seeds are
//! scored once and `bakeoff_decision::decide` applies the rule. Entry points:
//! `run_point_bakeoff` (the full protocol), `run_bakeoff_scene` (one seed).
//! CLI: `examples/point_network_bakeoff.rs`; tests: `tests/point_network_bakeoff.rs`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crate::bakeoff_candidate_fix::{apply_reportability_gates, BakeoffCandidate, CandidateSolveOutcome};
use crate::bakeoff_config::{OnsetCandidateConfig, PointBakeoffConfig, SeedSplit, ONSET_GRID_RANGE_MULTIPLES_TRIED};
use crate::bakeoff_decision::{decide, BakeoffDecision};
use crate::bakeoff_onset_candidate::run_onset_candidate;
use crate::bakeoff_production_candidate::{run_production_candidate, ProductionFrontEndVariant};
use crate::bakeoff_scenes::{build_bakeoff_scene, BakeoffScene};
use crate::bakeoff_scoring::{scene_class, score_candidate, CandidateSceneScore, SceneScore};
use crate::bakeoff_summary::summarize;
use crate::bakeoff_windows::{detect_triggers, node_noise_floor_rms, BakeoffTrigger};

/// Layout lags an onset window spans after its trigger (see `bakeoff_onset_candidate`).
const ONSET_WINDOW_LAYOUT_LAGS_AFTER_TRIGGER: f64 = 2.0;

/// One seed's scores, plus the onset candidate under every grid reach tried.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneRun {
    pub score: SceneScore,
    pub onset_variant_scores: Vec<(f64, CandidateSceneScore)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OnsetTuningResult {
    /// (grid reach multiple, tuning-seed recall, tuning-seed p90 error).
    pub variants: Vec<(f64, f64, Option<f64>)>,
    pub chosen_grid_range_aperture_multiple: f64,
}

#[derive(Debug, Clone)]
pub struct PointBakeoffReport {
    pub config: PointBakeoffConfig,
    pub seeds: SeedSplit,
    pub tuning: OnsetTuningResult,
    pub tuning_scores: Vec<SceneScore>,
    pub held_out_scores: Vec<SceneScore>,
    pub failed_seeds: Vec<(u64, String)>,
    pub decision: BakeoffDecision,
    pub elapsed_seconds: f64,
}

/// Onset windows: a trigger inside an open window is already covered by it
/// (the onset localizer finds every source in its window).
fn onset_window_triggers(scene: &BakeoffScene, triggers: &[BakeoffTrigger], config: &OnsetCandidateConfig) -> Vec<BakeoffTrigger> {
    let span_after_trigger_s = ONSET_WINDOW_LAYOUT_LAGS_AFTER_TRIGGER * scene.largest_layout_lag_s() + config.window_post_roll_s;
    let mut opened: Vec<BakeoffTrigger> = Vec::new();
    for trigger in triggers {
        if opened.last().is_none_or(|last| trigger.stream_time_s >= last.stream_time_s + span_after_trigger_s) {
            opened.push(*trigger);
        }
    }
    opened
}

/// Every node contributes to every solve here, so the contributing centroid
/// is the layout's (measured positions, as the server knows them).
fn layout_centroid_m(scene: &BakeoffScene) -> [f64; 3] {
    let positions = scene.measured_positions_m();
    std::array::from_fn(|axis| positions.iter().map(|position| position[axis]).sum::<f64>() / positions.len() as f64)
}

fn quietest_node_floor_rms(scene: &BakeoffScene, frame_seconds: f64) -> f64 {
    (0..scene.nodes.len()).map(|node| node_noise_floor_rms(scene, node, frame_seconds)).fold(f64::INFINITY, f64::min)
}

/// Render seed `seed`, run every candidate on it and score them. The onset
/// candidate runs once per entry of `onset_grid_multiples`; the first entry
/// is the one recorded in `score`.
pub fn run_bakeoff_scene(seed: u64, config: &PointBakeoffConfig, onset_grid_multiples: &[f64]) -> Result<SceneRun, String> {
    let scene = build_bakeoff_scene(seed, &config.scene)?;
    let triggers = detect_triggers(&scene, &config.trigger);
    let floor_rms = quietest_node_floor_rms(&scene, config.trigger.frame_seconds);
    let capped_window_covers_layout = scene.largest_layout_lag_s() <= config.production.event_span_max_cross_node_lag_seconds;
    let production_window = ProductionFrontEndVariant { lag_cap_s: Some(config.production.event_span_max_cross_node_lag_seconds), taper_fraction: None, band_hz: None };
    let uncapped_window = ProductionFrontEndVariant { lag_cap_s: None, ..production_window };
    let repaired_front_end = ProductionFrontEndVariant {
        lag_cap_s: None,
        taper_fraction: Some(config.production.diagnostic_window_taper_fraction),
        band_hz: Some(config.production.diagnostic_band_hz),
    };
    let (mut deployed, mut sidecar, mut uncapped, mut repaired) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for trigger in &triggers {
        let run = |variant| run_production_candidate(&scene, trigger, &config.production, floor_rms, variant);
        let outcomes = run(production_window);
        // Same window when the cap does not bind: the uncapped run would repeat it exactly.
        uncapped.push(if capped_window_covers_layout { outcomes.deployed.clone() } else { run(uncapped_window).deployed });
        repaired.push(run(repaired_front_end).deployed);
        deployed.push(outcomes.deployed);
        sidecar.push(outcomes.sidecar_srp);
    }
    let onset_variant_outcomes: Vec<(f64, Vec<CandidateSolveOutcome>)> = onset_grid_multiples
        .iter()
        .map(|multiple| {
            let onset_config = OnsetCandidateConfig { grid_range_aperture_multiple: *multiple, ..config.onset.clone() };
            let outcomes =
                onset_window_triggers(&scene, &triggers, &onset_config).iter().map(|trigger| run_onset_candidate(&scene, trigger, &onset_config)).collect();
            (*multiple, outcomes)
        })
        .collect();
    let centroid = layout_centroid_m(&scene);
    for outcome in deployed.iter_mut().chain(sidecar.iter_mut()).chain(uncapped.iter_mut()).chain(repaired.iter_mut()) {
        apply_reportability_gates(outcome, &centroid, &config.reportability);
    }
    let mut onset_variant_outcomes = onset_variant_outcomes;
    for (_, outcomes) in onset_variant_outcomes.iter_mut() {
        for outcome in outcomes.iter_mut() {
            apply_reportability_gates(outcome, &centroid, &config.reportability);
        }
    }
    let scoring = &config.scoring;
    let onset_variant_scores: Vec<(f64, CandidateSceneScore)> =
        onset_variant_outcomes.iter().map(|(multiple, outcomes)| (*multiple, score_candidate(&scene, outcomes, scoring))).collect();
    let mut by_candidate = std::collections::BTreeMap::new();
    by_candidate.insert(BakeoffCandidate::ProductionGccPhatLsq, score_candidate(&scene, &deployed, scoring));
    by_candidate.insert(BakeoffCandidate::ProductionSidecarSrp, score_candidate(&scene, &sidecar, scoring));
    by_candidate.insert(BakeoffCandidate::ProductionUncappedWindow, score_candidate(&scene, &uncapped, scoring));
    by_candidate.insert(BakeoffCandidate::ProductionRepairedFrontEnd, score_candidate(&scene, &repaired, scoring));
    if let Some((_, onset_score)) = onset_variant_scores.first() {
        by_candidate.insert(BakeoffCandidate::OnsetClusteringShadow, onset_score.clone());
    }
    Ok(SceneRun { score: SceneScore { class: scene_class(&scene), call_count: scene.calls.len(), by_candidate }, onset_variant_scores })
}

/// Run `seeds` on `thread_count` threads; results come back in seed order.
pub fn run_bakeoff_seeds(
    seeds: &[u64],
    config: &PointBakeoffConfig,
    onset_grid_multiples: &[f64],
    thread_count: usize,
) -> Vec<(u64, Result<SceneRun, String>)> {
    let next = AtomicUsize::new(0);
    let results = Mutex::new(Vec::with_capacity(seeds.len()));
    std::thread::scope(|scope| {
        for _ in 0..thread_count.max(1).min(seeds.len().max(1)) {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(&seed) = seeds.get(index) else { break };
                let run = run_bakeoff_scene(seed, config, onset_grid_multiples);
                results.lock().expect("bake-off results lock").push((index, seed, run));
            });
        }
    });
    let mut results = results.into_inner().expect("bake-off results lock");
    results.sort_by_key(|(index, _, _)| *index);
    results.into_iter().map(|(_, seed, run)| (seed, run)).collect()
}

/// Highest tuning recall wins; a tie goes to the lower p90 error, then to the smaller grid.
fn choose_onset_grid_multiple(runs: &[SceneRun], multiples: &[f64]) -> OnsetTuningResult {
    let variants: Vec<(f64, f64, Option<f64>)> = multiples
        .iter()
        .enumerate()
        .map(|(variant, multiple)| {
            let scenes: Vec<SceneScore> = runs
                .iter()
                .map(|run| {
                    let mut score = run.score.clone();
                    score.by_candidate.insert(BakeoffCandidate::OnsetClusteringShadow, run.onset_variant_scores[variant].1.clone());
                    score
                })
                .collect();
            let references: Vec<&SceneScore> = scenes.iter().collect();
            let summary = summarize(&references, BakeoffCandidate::OnsetClusteringShadow);
            (*multiple, summary.recall, summary.p90_error_m)
        })
        .collect();
    let chosen = variants
        .iter()
        .min_by(|a, b| b.1.total_cmp(&a.1).then(a.2.unwrap_or(f64::INFINITY).total_cmp(&b.2.unwrap_or(f64::INFINITY))).then(a.0.total_cmp(&b.0)))
        .map_or(0.0, |variant| variant.0);
    OnsetTuningResult { variants, chosen_grid_range_aperture_multiple: chosen }
}

fn split_results(results: Vec<(u64, Result<SceneRun, String>)>, failed: &mut Vec<(u64, String)>) -> Vec<SceneRun> {
    results
        .into_iter()
        .filter_map(|(seed, run)| match run {
            Ok(run) => Some(run),
            Err(message) => {
                failed.push((seed, message));
                None
            }
        })
        .collect()
}

/// The whole protocol: tune on `seeds.tuning_seeds`, decide on `seeds.held_out_seeds`.
pub fn run_point_bakeoff(seeds: SeedSplit, mut config: PointBakeoffConfig, thread_count: usize) -> PointBakeoffReport {
    let started = Instant::now();
    let mut failed_seeds = Vec::new();
    let tuning_runs = split_results(run_bakeoff_seeds(&seeds.tuning_seeds, &config, &ONSET_GRID_RANGE_MULTIPLES_TRIED, thread_count), &mut failed_seeds);
    let tuning = choose_onset_grid_multiple(&tuning_runs, &ONSET_GRID_RANGE_MULTIPLES_TRIED);
    config.onset.grid_range_aperture_multiple = tuning.chosen_grid_range_aperture_multiple;
    let chosen = [tuning.chosen_grid_range_aperture_multiple];
    let held_out_runs = split_results(run_bakeoff_seeds(&seeds.held_out_seeds, &config, &chosen, thread_count), &mut failed_seeds);
    let held_out_scores: Vec<SceneScore> = held_out_runs.into_iter().map(|run| run.score).collect();
    let references: Vec<&SceneScore> = held_out_scores.iter().collect();
    let decision = decide(&references, &config.decision, &config.scoring);
    PointBakeoffReport {
        tuning_scores: tuning_runs.into_iter().map(|run| run.score).collect(),
        held_out_scores,
        failed_seeds,
        decision,
        tuning,
        seeds,
        config,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    }
}

