//! Per-track species belief as capped log-odds (port of
//! `minimappr/core/track_species_belief.py`, server track S3).
//!
//! Each label keeps an independent log-odds fused across detections:
//! `L += clip(logit(p) - logit(prior), ±per_update_cap)`, then
//! `L = clip(L, logit(prior) ± total_cap)`. One detection scoring `p` gives
//! belief `p` exactly, so three weak bouts from one perch add up to a
//! confident identification and one wrong bout cannot flip it. Labels a
//! detection does not score are left untouched (classifiers report top-k).
//!
//! Label maps are insertion-ordered `Vec`s so ties break exactly like the
//! reference's Python dicts (first maximum wins, stable sorts).

/// Clamp keeping logit finite for scores of exactly 0 or 1.
const PROBABILITY_EPSILON: f64 = 1e-6;
/// Exposed probabilities are rounded to this many decimals (API contract).
const EXPOSED_PROBABILITY_DECIMALS: usize = 4;
pub const UNKNOWN_LABEL: &str = "unknown";
/// Labels carrying no species information; they never enter a belief.
const NON_EVIDENCE_LABELS: [&str; 4] = ["", "unknown", "background", "silence"];

/// Insertion-ordered label → value map (the reference's dict semantics).
pub type LabelScores = Vec<(String, f64)>;

pub fn normalised_label(label: &str) -> String {
    label.trim().to_lowercase()
}

pub fn is_non_evidence_label(label: &str) -> bool {
    NON_EVIDENCE_LABELS.contains(&normalised_label(label).as_str())
}

pub fn probability_to_log_odds(probability: f64) -> f64 {
    let clamped = probability.clamp(PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON);
    (clamped / (1.0 - clamped)).ln()
}

pub fn log_odds_to_probability(log_odds: f64) -> f64 {
    1.0 / (1.0 + (-log_odds).exp())
}

/// `round(value, digits)` as Python does it (correctly rounded decimal).
pub fn round_decimal(value: f64, digits: usize) -> f64 {
    format!("{value:.digits$}").parse().unwrap_or(value)
}

pub fn score_for(scores: &[(String, f64)], label: &str) -> Option<f64> {
    scores.iter().find(|(key, _)| key == label).map(|(_, value)| *value)
}

pub fn set_score(scores: &mut LabelScores, label: &str, value: f64) {
    match scores.iter_mut().find(|(key, _)| key == label) {
        Some(entry) => entry.1 = value,
        None => scores.push((label.to_string(), value)),
    }
}

/// First entry with the largest value (Python `max(d.items(), key=...)`).
pub fn first_maximum(scores: &[(String, f64)]) -> Option<(&str, f64)> {
    let mut best: Option<(&str, f64)> = None;
    for (label, value) in scores {
        if best.is_none_or(|(_, best_value)| *value > best_value) {
            best = Some((label.as_str(), *value));
        }
    }
    best
}

/// Stable descending sort (Python `sorted(..., reverse=True)` keeps tie order).
pub fn ranked_descending(scores: &[(String, f64)]) -> LabelScores {
    let mut ranked = scores.to_vec();
    ranked.sort_by(|first, second| second.1.total_cmp(&first.1));
    ranked
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeciesBeliefConfig {
    pub prior_probability: f64,
    pub per_update_log_odds_cap: f64,
    pub total_log_odds_cap: f64,
    pub evidence_top_k: usize,
    pub display_min_belief: f64,
    pub exposed_top_n: usize,
}

impl Default for SpeciesBeliefConfig {
    fn default() -> Self {
        Self {
            prior_probability: 0.1,
            per_update_log_odds_cap: 3.0,
            total_log_odds_cap: 6.0,
            evidence_top_k: 3,
            display_min_belief: 0.5,
            exposed_top_n: 5,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeciesLogOddsBelief {
    config: SpeciesBeliefConfig,
    prior_log_odds: f64,
    log_odds_by_label: LabelScores,
}

impl SpeciesLogOddsBelief {
    pub fn new(config: SpeciesBeliefConfig) -> Self {
        Self { config, prior_log_odds: probability_to_log_odds(config.prior_probability), log_odds_by_label: Vec::new() }
    }

    pub fn update(&mut self, evidence_probability_by_label: &[(String, f64)]) {
        let cap = self.config.per_update_log_odds_cap;
        let floor = self.prior_log_odds - self.config.total_log_odds_cap;
        let ceiling = self.prior_log_odds + self.config.total_log_odds_cap;
        for (label, probability) in evidence_probability_by_label {
            // A NaN score would poison the label's log-odds for good
            // (the Python reference has this latent bug; see the review notes).
            if is_non_evidence_label(label) || probability.is_nan() {
                continue;
            }
            let delta = (probability_to_log_odds(*probability) - self.prior_log_odds).clamp(-cap, cap);
            let current = score_for(&self.log_odds_by_label, label).unwrap_or(self.prior_log_odds);
            set_score(&mut self.log_odds_by_label, label, (current + delta).clamp(floor, ceiling));
        }
    }

    pub fn probability_for(&self, label: &str) -> f64 {
        log_odds_to_probability(score_for(&self.log_odds_by_label, label).unwrap_or(self.prior_log_odds))
    }

    pub fn is_empty(&self) -> bool {
        self.log_odds_by_label.is_empty()
    }

    /// `(label, probability)` of the most-believed label; None when empty.
    pub fn top_label(&self) -> Option<(String, f64)> {
        first_maximum(&self.log_odds_by_label).map(|(label, log_odds)| (label.to_string(), log_odds_to_probability(log_odds)))
    }

    pub fn top_labels(&self, count: usize) -> Vec<String> {
        ranked_descending(&self.log_odds_by_label).into_iter().take(count).map(|(label, _)| label).collect()
    }

    /// Top-N label → probability rounded for exposure (what other layers read).
    pub fn exposed_probabilities(&self) -> LabelScores {
        ranked_descending(&self.log_odds_by_label)
            .into_iter()
            .take(self.config.exposed_top_n)
            .map(|(label, log_odds)| (label, round_decimal(log_odds_to_probability(log_odds), EXPOSED_PROBABILITY_DECIMALS)))
            .collect()
    }

    pub fn passes_display_threshold(&self) -> bool {
        self.top_label().is_some_and(|(_, probability)| probability >= self.config.display_min_belief)
    }
}

/// Label → probability evidence for one detection: the winning member's top-k
/// scores (composite keys carry a `member:` prefix) plus the detection's own
/// label at its confidence.
pub fn species_evidence_from_detection(
    label: &str,
    confidence: f64,
    classifier_scores: Option<&[(String, f64)]>,
    winner_member: Option<&str>,
    top_k: usize,
) -> LabelScores {
    let mut evidence: LabelScores = Vec::new();
    if let Some(scores) = classifier_scores.filter(|scores| !scores.is_empty()) {
        let prefix = winner_member.map(|member| format!("{member}:")).unwrap_or_default();
        let prefixed: LabelScores = if prefix.is_empty() {
            Vec::new()
        } else {
            scores
                .iter()
                .filter_map(|(key, value)| key.strip_prefix(prefix.as_str()).map(|rest| (rest.to_string(), *value)))
                .collect()
        };
        let member_scores = if prefixed.is_empty() {
            scores.iter().filter(|(key, _)| !key.contains(':')).cloned().collect()
        } else {
            prefixed
        };
        for (key, value) in ranked_descending(&member_scores).into_iter().take(top_k.max(1)) {
            set_score(&mut evidence, &key, value);
        }
    }
    let label = label.trim();
    if !label.is_empty() && !is_non_evidence_label(label) {
        let existing = score_for(&evidence, label).unwrap_or(0.0);
        set_score(&mut evidence, label, confidence.max(existing));
    }
    evidence
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(pairs: &[(&str, f64)]) -> LabelScores {
        pairs.iter().map(|(label, value)| (label.to_string(), *value)).collect()
    }

    #[test]
    fn one_detection_gives_its_own_probability_and_weak_bouts_accumulate() {
        let mut belief = SpeciesLogOddsBelief::new(SpeciesBeliefConfig::default());
        belief.update(&evidence(&[("robin", 0.3)]));
        assert!((belief.probability_for("robin") - 0.3).abs() < 1e-12);
        belief.update(&evidence(&[("robin", 0.45)]));
        belief.update(&evidence(&[("robin", 0.4)]));
        assert!(belief.passes_display_threshold());
        belief.update(&evidence(&[("tanager", 0.9), ("unknown", 0.99)]));
        assert_eq!(belief.top_label().unwrap().0, "robin");
        assert_eq!(belief.exposed_probabilities().len(), 2);
    }

    #[test]
    fn nan_scores_are_ignored_rather_than_poisoning_the_belief() {
        let mut belief = SpeciesLogOddsBelief::new(SpeciesBeliefConfig::default());
        belief.update(&evidence(&[("robin", 0.6)]));
        belief.update(&evidence(&[("robin", f64::NAN), ("wren", f64::NAN)]));
        assert!((belief.probability_for("robin") - 0.6).abs() < 1e-12);
        assert_eq!(belief.exposed_probabilities().len(), 1);
    }

    #[test]
    fn evidence_uses_the_winning_member_and_the_detection_label() {
        let scores = evidence(&[("Animal", 0.9), ("birdnet:robin", 0.4), ("birdnet:wren", 0.2)]);
        let built = species_evidence_from_detection("robin", 0.5, Some(&scores), Some("birdnet"), 3);
        assert_eq!(built, evidence(&[("robin", 0.5), ("wren", 0.2)]));
    }
}
