//! The attribution cascade for one classified cluster window (port of the
//! decision flow in `minimappr/core/onset_cluster_attribution_stage.py`):
//! no audible interferer → `SingleCluster`; SNR gradient settles →
//! `SnrGradient`; else (LOCO on, interferers sounding in the window, rerun
//! budget left) mask + rerun through the host's [`LocoRerunHost`] and let
//! [`loco_decision`] settle → `Loco`; otherwise `Unsettled` (unlabelled: the
//! position stands, no species is guessed). Masking and the classifier rerun
//! are signal work owned by the host (phones run their own classifier).

use std::collections::BTreeMap;

use super::attribution_decision::{
    interferers_in_window, loco_decision, snr_gradient_decision, AttributionThresholds, ClassificationWindow, ClassifierAnswer,
    GradientDecision, InterfererOnset, LocoDecision,
};
use super::attribution_evidence::ClusterAttributionEvidence;
use super::echo_checks::SAME_BAND_MIN_OVERLAP_FRACTION;
use super::numeric::python_round_to_i64;

/// Rerun budgets are remembered for this many recent analysis windows (memory bound only).
const MAX_TRACKED_RERUN_WINDOWS: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributionTier {
    SingleCluster,
    SnrGradient,
    Loco,
    Unsettled,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttributionCascadeConfig {
    pub thresholds: AttributionThresholds,
    pub loco_enabled: bool,
    pub loco_max_reruns_per_window: u32,
    /// Interferer onsets up to this long before the window still count (a note still sounding).
    pub loco_mask_max_seconds: f64,
}

impl Default for AttributionCascadeConfig {
    fn default() -> Self {
        Self { thresholds: AttributionThresholds::default(), loco_enabled: true, loco_max_reruns_per_window: 4, loco_mask_max_seconds: 1.0 }
    }
}

/// Everything attribution needs about one cluster's classification window.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterAttributionContext<'node> {
    /// Analysis window shared by sibling clusters (the rerun budget key).
    pub window_id: String,
    pub target_index: usize,
    /// Non-echo clusters of the window.
    pub cluster_evidence: BTreeMap<usize, ClusterAttributionEvidence>,
    pub window: ClassificationWindow<'node>,
}

/// Host side of LOCO: mask the interferer regions, then rerun the classifier.
pub trait LocoRerunHost {
    /// Mask these onset regions in the classification signal; returns the masked cell fraction.
    fn mask_interferers(&mut self, seeds: &[InterfererOnset]) -> f64;
    /// Classify the masked signal.
    fn rerun_masked(&mut self) -> ClassifierAnswer;
}

/// Per-analysis-window rerun counter shared by that window's sibling clusters.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LocoRerunBudget {
    used_by_window: BTreeMap<String, (u32, u64)>,
    use_sequence: u64,
}

impl LocoRerunBudget {
    pub fn try_consume(&mut self, window_id: &str, max_reruns: u32) -> bool {
        let used = self.used_by_window.get(window_id).map_or(0, |(used, _)| *used);
        if used >= max_reruns {
            return false;
        }
        self.use_sequence += 1;
        self.used_by_window.insert(window_id.to_string(), (used + 1, self.use_sequence));
        while self.used_by_window.len() > MAX_TRACKED_RERUN_WINDOWS {
            let oldest = self.used_by_window.iter().min_by_key(|(_, (_, sequence))| *sequence).map(|(window, _)| window.clone());
            if let Some(oldest) = oldest {
                self.used_by_window.remove(&oldest);
            }
        }
        true
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttributionOutcome {
    pub tier: AttributionTier,
    /// Why the cascade stopped where it did.
    pub reason: &'static str,
    /// The label the cluster carries: the original, the LOCO winner, or None (unlabelled).
    pub label: Option<String>,
    pub confidence: f64,
    pub interferer_count: usize,
    pub gradient: Option<GradientDecision>,
    pub loco: Option<LocoDecision>,
}

pub fn attribute_cluster(
    original: &ClassifierAnswer,
    context: &ClusterAttributionContext<'_>,
    config: &AttributionCascadeConfig,
    budget: &mut LocoRerunBudget,
    host: &mut dyn LocoRerunHost,
) -> AttributionOutcome {
    let keep = |tier, reason, interferer_count, gradient| AttributionOutcome {
        tier,
        reason,
        label: Some(original.label.clone()),
        confidence: original.confidence,
        interferer_count,
        gradient,
        loco: None,
    };
    let unlabelled = |reason, interferer_count, gradient, loco| AttributionOutcome {
        tier: AttributionTier::Unsettled,
        reason,
        label: None,
        confidence: 0.0,
        interferer_count,
        gradient,
        loco,
    };
    let Some(target) = context.cluster_evidence.get(&context.target_index) else {
        return keep(AttributionTier::SingleCluster, "target_evidence_missing", 0, None);
    };
    let lookback_samples = python_round_to_i64(config.loco_mask_max_seconds * f64::from(context.window.sample_rate_hz));
    let interferers = interferers_in_window(
        context.target_index,
        &context.cluster_evidence,
        &context.window,
        config.thresholds.audible_snr_db,
        lookback_samples,
        SAME_BAND_MIN_OVERLAP_FRACTION,
    );
    let count = interferers.len();
    if interferers.is_empty() {
        return keep(AttributionTier::SingleCluster, "no_interferers", 0, None);
    }
    let gradient = snr_gradient_decision(target, &interferers, context.window.classifying_node_id, &config.thresholds);
    if gradient.settled {
        return keep(AttributionTier::SnrGradient, gradient.reason, count, Some(gradient));
    }
    if !config.loco_enabled {
        return unlabelled("loco_disabled", count, Some(gradient), None);
    }
    let seeds: Vec<InterfererOnset> = interferers.iter().flat_map(|interferer| interferer.onsets_in_window.iter().copied()).collect();
    if host.mask_interferers(&seeds) <= 0.0 {
        // Interferer notes ended before this window: nothing of theirs was classified.
        return keep(AttributionTier::SingleCluster, "interferers_silent_in_window", count, Some(gradient));
    }
    if !budget.try_consume(&context.window_id, config.loco_max_reruns_per_window) {
        return unlabelled("loco_rerun_cap", count, Some(gradient), None);
    }
    let masked = host.rerun_masked();
    let decision = loco_decision(original, &masked, &config.thresholds);
    if !decision.settled {
        return unlabelled(decision.reason, count, Some(gradient), Some(decision));
    }
    AttributionOutcome {
        tier: AttributionTier::Loco,
        reason: decision.reason,
        label: decision.label.clone(),
        confidence: decision.confidence,
        interferer_count: count,
        gradient: Some(gradient),
        loco: Some(decision),
    }
}
