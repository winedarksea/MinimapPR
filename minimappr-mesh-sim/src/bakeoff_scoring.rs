//! Scoring one scene's candidate outputs against truth.
//!
//! - Attribution: each fix's cues (node, stamped time) pick the scripted call
//!   sounding nearest that moment at that node, which is the call a
//!   classifier would have labelled; no position is consulted, so a fix that
//!   lands on the wrong bird counts as an attribution error.
//! - Per call, the primary fix is the first fix attributed to it (the first
//!   trigger's solve; the strongest onset cluster). It is "localized" when
//!   its horizontal error is under max(floor, fraction × aperture).
//! - Coverage: the primary fix's stated horizontal covariance contains the
//!   truth when e^T S^-1 e is under the chi-square(2) quantile of the stated
//!   level (E3's `replay_uncertainty_coverage.py`, restricted to the
//!   horizontal plane because point layouts rarely observe height). A
//!   missing or singular covariance counts as not covering.

use std::collections::BTreeMap;

use crate::bakeoff_candidate_fix::{BakeoffCandidate, CandidateFix, CandidateSolveOutcome};
use crate::bakeoff_config::ScoringConfig;
use crate::bakeoff_scenes::{horizontal_distance_m, BakeoffEventKind, BakeoffScene, ClockRegime};

const SINGULAR_DETERMINANT_EPSILON: f64 = 1.0e-18;

/// What a scene was, for slicing results.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneClass {
    pub seed: u64,
    pub event_kind: BakeoffEventKind,
    pub clock_regime: ClockRegime,
    pub node_count: usize,
    pub aperture_m: f64,
    pub has_hand_measured_node: bool,
    /// Waveform of the scene's first call.
    pub waveform_label: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrimaryFixScore {
    pub horizontal_error_m: f64,
    pub localized: bool,
    pub covariance_covers_truth: bool,
    pub covariance_valid: bool,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct CandidateSceneScore {
    /// Indexed by call: the primary fix's score, or None when no fix was attributed.
    pub primary_by_call: Vec<Option<PrimaryFixScore>>,
    pub fix_count: usize,
    /// Fixes not within the gate of the call their cues attribute them to.
    pub misattributed_fix_count: usize,
    pub solve_wall_times_s: Vec<f64>,
    /// Why solves published nothing (degraded SRP reason, onset summary).
    pub failure_reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SceneScore {
    pub class: SceneClass,
    pub call_count: usize,
    pub by_candidate: BTreeMap<BakeoffCandidate, CandidateSceneScore>,
}

pub fn recall_gate_m(aperture_m: f64, config: &ScoringConfig) -> f64 {
    config.recall_gate_floor_m.max(config.recall_gate_aperture_fraction * aperture_m)
}

/// The call sounding nearest the fix's cues: smallest gap between a cue and
/// the call's direct-path sounding interval at the cue's node, ties broken
/// by distance from the call's arrival.
pub fn attributed_call(scene: &BakeoffScene, fix: &CandidateFix) -> Option<usize> {
    let mut best: Option<(usize, f64, f64)> = None;
    for &(node, cue_time_s) in &fix.attribution_cues {
        for call in 0..scene.calls.len() {
            let arrival_s = scene.arrival_stream_time_s(call, node);
            let gap_s = (arrival_s - cue_time_s).max(cue_time_s - arrival_s - scene.calls[call].duration_s).max(0.0);
            let onset_distance_s = (cue_time_s - arrival_s).abs();
            if best.is_none_or(|(_, best_gap, best_distance)| (gap_s, onset_distance_s) < (best_gap, best_distance)) {
                best = Some((call, gap_s, onset_distance_s));
            }
        }
    }
    best.map(|(call, _, _)| call)
}

fn horizontal_mahalanobis_squared(error: [f64; 2], covariance: &[[f64; 3]; 3]) -> Option<f64> {
    let [[a, b, _], [c, d, _], _] = *covariance;
    let symmetric_b = 0.5 * (b + c);
    let determinant = a * d - symmetric_b * symmetric_b;
    if !(determinant.is_finite() && determinant > SINGULAR_DETERMINANT_EPSILON && a > 0.0 && d > 0.0) {
        return None;
    }
    let value = (d * error[0] * error[0] - 2.0 * symmetric_b * error[0] * error[1] + a * error[1] * error[1]) / determinant;
    value.is_finite().then_some(value)
}

fn primary_score(scene: &BakeoffScene, call: usize, fix: &CandidateFix, config: &ScoringConfig) -> PrimaryFixScore {
    let truth = scene.calls[call].position_m;
    let horizontal_error_m = horizontal_distance_m(&fix.position_m, &truth);
    let error = [fix.position_m[0] - truth[0], fix.position_m[1] - truth[1]];
    let mahalanobis = fix.covariance_m2.as_ref().and_then(|covariance| horizontal_mahalanobis_squared(error, covariance));
    PrimaryFixScore {
        horizontal_error_m,
        localized: horizontal_error_m <= recall_gate_m(scene.aperture_m, config),
        covariance_covers_truth: mahalanobis.is_some_and(|value| value <= config.coverage_chi_square_threshold),
        covariance_valid: mahalanobis.is_some(),
    }
}

pub fn score_candidate(scene: &BakeoffScene, outcomes: &[CandidateSolveOutcome], config: &ScoringConfig) -> CandidateSceneScore {
    let gate_m = recall_gate_m(scene.aperture_m, config);
    let mut score = CandidateSceneScore { primary_by_call: vec![None; scene.calls.len()], ..CandidateSceneScore::default() };
    for outcome in outcomes {
        score.solve_wall_times_s.push(outcome.wall_time_s);
        score.failure_reasons.extend(outcome.failure_reason.clone());
        score.failure_reasons.extend(outcome.gate_rejections.iter().cloned());
        for fix in &outcome.fixes {
            score.fix_count += 1;
            let Some(call) = attributed_call(scene, fix) else {
                score.misattributed_fix_count += 1;
                continue;
            };
            if horizontal_distance_m(&fix.position_m, &scene.calls[call].position_m) > gate_m {
                score.misattributed_fix_count += 1;
            }
            if score.primary_by_call[call].is_none() {
                score.primary_by_call[call] = Some(primary_score(scene, call, fix, config));
            }
        }
    }
    score
}

pub fn scene_class(scene: &BakeoffScene) -> SceneClass {
    SceneClass {
        seed: scene.seed,
        event_kind: scene.event_kind,
        clock_regime: scene.clock_regime,
        node_count: scene.nodes.len(),
        aperture_m: scene.aperture_m,
        has_hand_measured_node: scene.has_hand_measured_node(),
        waveform_label: scene.calls.first().map_or("", |call| call.waveform_label),
    }
}
