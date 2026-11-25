//! The pre-registered adoption rule for live onset clustering (E8).
//!
//! Fixed before any held-out seed was scored; reads held-out scenes only.
//! Onset clustering (`OnsetClusteringShadow`) is recommended for live use in
//! the sidecar only if, against the deployed production path:
//!
//! 1. recall gain >= `minimum_recall_gain` and the one-sided paired-bootstrap
//!    lower bound (resampling seeds) at `bootstrap_confidence_level` is > 0;
//! 2. on calls both fixed, p90 horizontal error ratio onset / production is
//!    <= `maximum_matched_p90_error_ratio` (no accuracy traded for recall);
//! 3. horizontal 95 % coverage >= production's minus `maximum_coverage_shortfall`
//!    (the stated level when production states none);
//! 4. two-caller attribution error rate <= production's plus
//!    `maximum_attribution_error_increase`;
//! 5. guard: criterion 1 also holds against the stronger (higher held-out
//!    recall) of the two production front-end diagnostics (lag cap lifted;
//!    cap lifted + taper + bird-call band). If 1–4 pass and 5 fails, the gain
//!    is production's front-end, not its solver: keep production, fix that.

use rand::Rng;

use crate::bakeoff_candidate_fix::BakeoffCandidate;
use crate::bakeoff_config::{DecisionRuleConfig, ScoringConfig};
use crate::bakeoff_scoring::SceneScore;
use crate::bakeoff_summary::{matched_errors, quantile, summarize, P90_QUANTILE};
use crate::seeded_random::purpose_stream;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BakeoffVerdict {
    AdoptOnsetClustering,
    KeepProductionRepairFrontEnd,
    KeepProduction,
}

impl BakeoffVerdict {
    pub fn label(self) -> &'static str {
        match self {
            Self::AdoptOnsetClustering => "adopt live onset clustering",
            Self::KeepProductionRepairFrontEnd => "keep production; repair its front-end (lag cap, taper, band) instead",
            Self::KeepProduction => "keep production; onset clustering stays shadow-only",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionCriterion {
    pub name: &'static str,
    pub passed: bool,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairedRecallComparison {
    pub gain: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BakeoffDecision {
    pub criteria: Vec<DecisionCriterion>,
    pub verdict: BakeoffVerdict,
    pub onset_vs_production: PairedRecallComparison,
    pub strongest_production_window: BakeoffCandidate,
    pub onset_vs_strongest_production_window: PairedRecallComparison,
}

/// Per-seed (calls, localized by `left`, localized by `right`).
fn per_seed_localized(scenes: &[&SceneScore], left: BakeoffCandidate, right: BakeoffCandidate) -> Vec<(usize, usize, usize)> {
    let localized = |scene: &SceneScore, candidate| {
        scene.by_candidate.get(&candidate).map_or(0, |score| score.primary_by_call.iter().flatten().filter(|fix| fix.localized).count())
    };
    scenes.iter().map(|scene| (scene.call_count, localized(scene, left), localized(scene, right))).collect()
}

fn recall_gain(samples: &[&(usize, usize, usize)]) -> f64 {
    let calls: usize = samples.iter().map(|sample| sample.0).sum();
    if calls == 0 {
        return 0.0;
    }
    let left: usize = samples.iter().map(|sample| sample.1).sum();
    let right: usize = samples.iter().map(|sample| sample.2).sum();
    (left as f64 - right as f64) / calls as f64
}

/// Recall of `left` minus `right`, with a one-sided paired bootstrap over seeds.
pub fn paired_recall_comparison(
    scenes: &[&SceneScore],
    left: BakeoffCandidate,
    right: BakeoffCandidate,
    config: &DecisionRuleConfig,
) -> PairedRecallComparison {
    let per_seed = per_seed_localized(scenes, left, right);
    let gain = recall_gain(&per_seed.iter().collect::<Vec<_>>());
    if per_seed.is_empty() {
        return PairedRecallComparison { gain, lower_bound: f64::NEG_INFINITY, upper_bound: f64::INFINITY };
    }
    let mut rng = purpose_stream(config.bootstrap_seed, "point_bakeoff_bootstrap");
    let mut gains: Vec<f64> = (0..config.bootstrap_resample_count)
        .map(|_| {
            let resample: Vec<&(usize, usize, usize)> = (0..per_seed.len()).map(|_| &per_seed[rng.random_range(0..per_seed.len())]).collect();
            recall_gain(&resample)
        })
        .collect();
    gains.sort_by(f64::total_cmp);
    let tail = 1.0 - config.bootstrap_confidence_level;
    PairedRecallComparison {
        gain,
        lower_bound: quantile(&gains, tail).unwrap_or(f64::NEG_INFINITY),
        upper_bound: quantile(&gains, 1.0 - tail).unwrap_or(f64::INFINITY),
    }
}

fn recall_criterion(name: &'static str, comparison: &PairedRecallComparison, config: &DecisionRuleConfig) -> DecisionCriterion {
    DecisionCriterion {
        name,
        passed: comparison.gain >= config.minimum_recall_gain && comparison.lower_bound > 0.0,
        detail: format!(
            "gain {:+.3} (one-sided {:.0} % bootstrap lower bound {:+.3}); needs >= {:+.3} and lower bound > 0",
            comparison.gain,
            100.0 * config.bootstrap_confidence_level,
            comparison.lower_bound,
            config.minimum_recall_gain
        ),
    }
}

fn format_optional(value: Option<f64>) -> String {
    value.map_or_else(|| "n/a".to_string(), |value| format!("{value:.3}"))
}

/// Apply the rule to held-out scenes.
pub fn decide(held_out: &[&SceneScore], decision: &DecisionRuleConfig, scoring: &ScoringConfig) -> BakeoffDecision {
    use BakeoffCandidate::{OnsetClusteringShadow as Onset, ProductionGccPhatLsq as Production};
    let onset = summarize(held_out, Onset);
    let production = summarize(held_out, Production);
    let onset_vs_production = paired_recall_comparison(held_out, Onset, Production, decision);
    let strongest_production_window = [BakeoffCandidate::ProductionUncappedWindow, BakeoffCandidate::ProductionRepairedFrontEnd]
        .into_iter()
        .max_by(|a, b| summarize(held_out, *a).recall.total_cmp(&summarize(held_out, *b).recall))
        .unwrap_or(BakeoffCandidate::ProductionUncappedWindow);
    let onset_vs_strongest_production_window = paired_recall_comparison(held_out, Onset, strongest_production_window, decision);
    let matched = matched_errors(held_out, Onset, Production);
    let onset_matched: Vec<f64> = matched.iter().map(|pair| pair.0).collect();
    let production_matched: Vec<f64> = matched.iter().map(|pair| pair.1).collect();
    let p90_ratio = quantile(&onset_matched, P90_QUANTILE)
        .zip(quantile(&production_matched, P90_QUANTILE))
        .map(|(onset_p90, production_p90)| onset_p90 / production_p90.max(f64::MIN_POSITIVE));
    let coverage_reference = production.coverage.unwrap_or(scoring.coverage_confidence_level);
    let attribution_reference = production.attribution_error_rate.unwrap_or(0.0);
    let criteria = vec![
        recall_criterion("recall gain over production", &onset_vs_production, decision),
        DecisionCriterion {
            name: "matched-call p90 error ratio",
            passed: p90_ratio.is_some_and(|ratio| ratio <= decision.maximum_matched_p90_error_ratio),
            detail: format!("{} over {} calls both fixed; needs <= {:.2}", format_optional(p90_ratio), matched.len(), decision.maximum_matched_p90_error_ratio),
        },
        DecisionCriterion {
            name: "coverage not worse",
            passed: onset.coverage.is_some_and(|coverage| coverage >= coverage_reference - decision.maximum_coverage_shortfall),
            detail: format!(
                "onset {} vs reference {:.3}; shortfall allowed {:.2}",
                format_optional(onset.coverage),
                coverage_reference,
                decision.maximum_coverage_shortfall
            ),
        },
        DecisionCriterion {
            name: "two-caller attribution not worse",
            passed: onset.attribution_error_rate.unwrap_or(0.0) <= attribution_reference + decision.maximum_attribution_error_increase,
            detail: format!(
                "onset {} vs production {}; increase allowed {:.2}",
                format_optional(onset.attribution_error_rate),
                format_optional(production.attribution_error_rate),
                decision.maximum_attribution_error_increase
            ),
        },
        recall_criterion("guard: recall gain over the stronger production front-end diagnostic", &onset_vs_strongest_production_window, decision),
    ];
    let core_rule_passed = criteria[..4].iter().all(|criterion| criterion.passed);
    let verdict = match (core_rule_passed, criteria[4].passed) {
        (true, true) => BakeoffVerdict::AdoptOnsetClustering,
        (true, false) => BakeoffVerdict::KeepProductionRepairFrontEnd,
        (false, _) => BakeoffVerdict::KeepProduction,
    };
    BakeoffDecision { criteria, verdict, onset_vs_production, strongest_production_window, onset_vs_strongest_production_window }
}
