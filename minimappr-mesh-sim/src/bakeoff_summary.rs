//! Aggregate metrics for one candidate over a set of scored scenes.

use crate::bakeoff_candidate_fix::BakeoffCandidate;
use crate::bakeoff_scenes::BakeoffEventKind;
use crate::bakeoff_scoring::{PrimaryFixScore, SceneScore};

const MILLISECONDS_PER_SECOND: f64 = 1_000.0;
pub const MEDIAN_QUANTILE: f64 = 0.5;
pub const P90_QUANTILE: f64 = 0.9;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct CandidateSummary {
    pub call_count: usize,
    pub localized_count: usize,
    pub recall: f64,
    /// Calls with any attributed fix (error statistics are over these).
    pub fixed_call_count: usize,
    pub median_error_m: Option<f64>,
    pub p90_error_m: Option<f64>,
    pub coverage: Option<f64>,
    pub invalid_covariance_count: usize,
    pub two_caller_fix_count: usize,
    pub two_caller_misattributed_count: usize,
    pub attribution_error_rate: Option<f64>,
    pub all_fix_count: usize,
    pub all_misattributed_count: usize,
    pub solve_count: usize,
    pub median_solve_ms: Option<f64>,
    pub p90_solve_ms: Option<f64>,
}

/// Nearest-rank quantile of unsorted values.
pub fn quantile(values: &[f64], fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = ((fraction * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    Some(sorted[rank - 1])
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

pub fn primary_fixes<'a>(scenes: &'a [&SceneScore], candidate: BakeoffCandidate) -> impl Iterator<Item = &'a PrimaryFixScore> + 'a {
    scenes.iter().filter_map(move |scene| scene.by_candidate.get(&candidate)).flat_map(|score| score.primary_by_call.iter().flatten())
}

pub fn summarize(scenes: &[&SceneScore], candidate: BakeoffCandidate) -> CandidateSummary {
    let call_count = scenes.iter().map(|scene| scene.call_count).sum();
    let primaries: Vec<&PrimaryFixScore> = primary_fixes(scenes, candidate).collect();
    let errors: Vec<f64> = primaries.iter().map(|fix| fix.horizontal_error_m).collect();
    let localized_count = primaries.iter().filter(|fix| fix.localized).count();
    let covered_count = primaries.iter().filter(|fix| fix.covariance_covers_truth).count();
    let mut summary = CandidateSummary {
        call_count,
        localized_count,
        recall: ratio(localized_count, call_count).unwrap_or(0.0),
        fixed_call_count: primaries.len(),
        median_error_m: quantile(&errors, MEDIAN_QUANTILE),
        p90_error_m: quantile(&errors, P90_QUANTILE),
        coverage: ratio(covered_count, primaries.len()),
        invalid_covariance_count: primaries.iter().filter(|fix| !fix.covariance_valid).count(),
        ..CandidateSummary::default()
    };
    let mut solve_times_ms = Vec::new();
    for scene in scenes {
        let Some(score) = scene.by_candidate.get(&candidate) else { continue };
        summary.all_fix_count += score.fix_count;
        summary.all_misattributed_count += score.misattributed_fix_count;
        if scene.class.event_kind == BakeoffEventKind::TwoBirds {
            summary.two_caller_fix_count += score.fix_count;
            summary.two_caller_misattributed_count += score.misattributed_fix_count;
        }
        solve_times_ms.extend(score.solve_wall_times_s.iter().map(|seconds| seconds * MILLISECONDS_PER_SECOND));
    }
    summary.attribution_error_rate = ratio(summary.two_caller_misattributed_count, summary.two_caller_fix_count);
    summary.solve_count = solve_times_ms.len();
    summary.median_solve_ms = quantile(&solve_times_ms, MEDIAN_QUANTILE);
    summary.p90_solve_ms = quantile(&solve_times_ms, P90_QUANTILE);
    summary
}

/// Horizontal errors of calls both candidates fixed, as (left, right) pairs.
pub fn matched_errors(scenes: &[&SceneScore], left: BakeoffCandidate, right: BakeoffCandidate) -> Vec<(f64, f64)> {
    let mut pairs = Vec::new();
    for scene in scenes {
        let (Some(left_score), Some(right_score)) = (scene.by_candidate.get(&left), scene.by_candidate.get(&right)) else { continue };
        for (left_fix, right_fix) in left_score.primary_by_call.iter().zip(&right_score.primary_by_call) {
            if let (Some(left_fix), Some(right_fix)) = (left_fix, right_fix) {
                pairs.push((left_fix.horizontal_error_m, right_fix.horizontal_error_m));
            }
        }
    }
    pairs
}
