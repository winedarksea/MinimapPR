//! Which classifier labels belong to which onset cluster (port of the pure
//! decisions in `minimappr/core/onset_cluster_attribution.py`, server S5).
//!
//! 1. **SNR gradient (no inference).** The target out-reads every in-window
//!    interferer on the classifying node by `dominance_margin_db`, both
//!    clusters fit 1/r spreading within `spreading_tolerance_db`, and the
//!    measured dominance matches the positions' prediction: the classifier
//!    mostly heard the target. When the target's band covers an interferer's
//!    onsets only geometry can justify dominance (collinear-trail fix).
//! 2. **LOCO.** Otherwise the host masks the interferers and reruns its
//!    classifier; [`loco_decision`] settles the label or refuses.
//! 3. **Unsettled → unlabelled.** No guess: the position stands, no species.

use std::collections::BTreeMap;

use super::attribution_evidence::{ClusterAttributionEvidence, ClusterOnsetArrival, DECIBELS_PER_AMPLITUDE_DECADE, MINIMUM_SPREADING_DISTANCE_M};
use super::echo_checks::SAME_BAND_MIN_OVERLAP_FRACTION;
use crate::wildlife_tracking::species_log_odds::{ranked_descending, score_for};

/// Labels that say "no species here"; they can never settle a cluster's species.
const NON_SPECIES_LABELS: [&str; 5] = ["unknown", "background", "silence", "ambient", "no_drone"];
/// Echo-check status reason for a single-array cluster with too few independent onsets.
pub const PERSISTENCE_HOLD_REASON: &str = "not_persistent";

pub fn is_non_species_label(label: &str) -> bool {
    NON_SPECIES_LABELS.contains(&label.trim().to_lowercase().as_str())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttributionThresholds {
    pub audible_snr_db: f64,
    pub dominance_margin_db: f64,
    pub spreading_tolerance_db: f64,
    pub loco_min_confidence: f64,
    pub loco_label_margin: f64,
    pub loco_collapse_ratio: f64,
}

impl Default for AttributionThresholds {
    fn default() -> Self {
        Self {
            audible_snr_db: 3.0,
            dominance_margin_db: 10.0,
            spreading_tolerance_db: 6.0,
            loco_min_confidence: 0.3,
            loco_label_margin: 0.1,
            loco_collapse_ratio: 0.5,
        }
    }
}

/// One onset of an interferer inside the window: (seconds into the window, band).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterfererOnset {
    pub onset_seconds: f64,
    pub band_min_hz: f64,
    pub band_max_hz: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WindowInterferer<'evidence> {
    pub cluster_index: usize,
    pub evidence: &'evidence ClusterAttributionEvidence,
    pub onsets_in_window: Vec<InterfererOnset>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GradientDecision {
    pub settled: bool,
    pub reason: &'static str,
    pub dominance_db: BTreeMap<usize, f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocoDecision {
    pub settled: bool,
    pub reason: &'static str,
    pub label: Option<String>,
    pub confidence: f64,
    pub collapsed_labels: Vec<String>,
}

/// The classification window on the classifying node, in analysis-span samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassificationWindow<'node> {
    pub classifying_node_id: &'node str,
    pub start_sample: i64,
    pub end_sample: i64,
    pub sample_rate_hz: u32,
    /// Onsets this close (same band) to a target onset are the target's own note or echo.
    pub shared_onset_max_lag_samples: Option<i64>,
}

pub fn band_overlap_fraction(band_a: (f64, f64), band_b: (f64, f64)) -> f64 {
    let narrower = (band_a.1 - band_a.0).min(band_b.1 - band_b.0);
    if narrower <= 0.0 {
        return 0.0;
    }
    (band_a.1.min(band_b.1) - band_a.0.max(band_b.0)).max(0.0) / narrower
}

/// Other clusters audible on the classifying node with an onset arriving in
/// the window (or up to `lookback_samples` before it), minus onsets that are
/// the target's own note seen again (masking those would erase the target).
pub fn interferers_in_window<'evidence>(
    target_index: usize,
    cluster_evidence: &'evidence BTreeMap<usize, ClusterAttributionEvidence>,
    window: &ClassificationWindow<'_>,
    audible_snr_db: f64,
    lookback_samples: i64,
    same_band_min_overlap_fraction: f64,
) -> Vec<WindowInterferer<'evidence>> {
    let node = window.classifying_node_id;
    let no_arrivals: Vec<ClusterOnsetArrival> = Vec::new();
    let target_arrivals = cluster_evidence[&target_index].node_onset_arrivals.get(node).unwrap_or(&no_arrivals);
    let shared_with_target = |arrival: &ClusterOnsetArrival| {
        window.shared_onset_max_lag_samples.is_some_and(|max_lag| {
            target_arrivals.iter().any(|own| {
                (arrival.span_sample - own.span_sample).abs() <= max_lag
                    && band_overlap_fraction((arrival.band_min_hz, arrival.band_max_hz), (own.band_min_hz, own.band_max_hz))
                        >= same_band_min_overlap_fraction
            })
        })
    };
    let mut interferers = Vec::new();
    for (index, evidence) in cluster_evidence {
        if *index == target_index || evidence.node_band_snr_db.get(node).copied().unwrap_or(f64::NEG_INFINITY) < audible_snr_db {
            continue;
        }
        let onsets: Vec<InterfererOnset> = evidence
            .node_onset_arrivals
            .get(node)
            .unwrap_or(&no_arrivals)
            .iter()
            .filter(|arrival| {
                window.start_sample - lookback_samples <= arrival.span_sample && arrival.span_sample < window.end_sample && !shared_with_target(arrival)
            })
            .map(|arrival| InterfererOnset {
                onset_seconds: (arrival.span_sample - window.start_sample) as f64 / f64::from(window.sample_rate_hz),
                band_min_hz: arrival.band_min_hz,
                band_max_hz: arrival.band_max_hz,
            })
            .collect();
        if !onsets.is_empty() {
            interferers.push(WindowInterferer { cluster_index: *index, evidence, onsets_in_window: onsets });
        }
    }
    interferers
}

fn arrival_band_hz(arrivals: &[ClusterOnsetArrival]) -> Option<(f64, f64)> {
    if arrivals.is_empty() {
        return None;
    }
    let band_min = arrivals.iter().map(|arrival| arrival.band_min_hz).fold(f64::INFINITY, f64::min);
    let band_max = arrivals.iter().map(|arrival| arrival.band_max_hz).fold(f64::NEG_INFINITY, f64::max);
    Some((band_min, band_max))
}

/// Spreading-only level advantage of the target at a node (equal source levels).
fn geometric_dominance_db(target: &ClusterAttributionEvidence, interferer: &ClusterAttributionEvidence, node_id: &str) -> f64 {
    let target_distance = target.node_distances_m.get(node_id).copied().unwrap_or(0.0).max(MINIMUM_SPREADING_DISTANCE_M);
    let interferer_distance = interferer.node_distances_m.get(node_id).copied().unwrap_or(0.0).max(MINIMUM_SPREADING_DISTANCE_M);
    DECIBELS_PER_AMPLITUDE_DECADE * (interferer_distance / target_distance).log10()
}

pub fn snr_gradient_decision(
    target: &ClusterAttributionEvidence,
    interferers: &[WindowInterferer<'_>],
    classifying_node_id: &str,
    thresholds: &AttributionThresholds,
) -> GradientDecision {
    let mut dominance_db = BTreeMap::new();
    let refuse = |reason, dominance_db| GradientDecision { settled: false, reason, dominance_db };
    let Some(target_snr) = target.node_band_snr_db.get(classifying_node_id).copied() else {
        return refuse("target_level_missing", dominance_db);
    };
    let target_band = target.node_onset_arrivals.get(classifying_node_id).and_then(|arrivals| arrival_band_hz(arrivals));
    for interferer in interferers {
        let measured = target_snr - interferer.evidence.node_band_snr_db[classifying_node_id];
        dominance_db.insert(interferer.cluster_index, measured);
        if measured < thresholds.dominance_margin_db {
            return refuse("not_dominant", dominance_db);
        }
        // Single-array windows have no cross-node gradient to check against.
        let (Some(target_residual), Some(interferer_residual), Some(target_predicted), Some(interferer_predicted)) = (
            target.spreading_residual_db,
            interferer.evidence.spreading_residual_db,
            target.predicted_snr_db(classifying_node_id),
            interferer.evidence.predicted_snr_db(classifying_node_id),
        ) else {
            return refuse("no_cross_node_gradient", dominance_db);
        };
        if target_residual.max(interferer_residual) > thresholds.spreading_tolerance_db {
            return refuse("spreading_inconsistent", dominance_db);
        }
        if (measured - (target_predicted - interferer_predicted)).abs() > thresholds.spreading_tolerance_db {
            return refuse("dominance_not_predicted", dominance_db);
        }
        let bands_overlap = target_band.is_some_and(|band| {
            interferer.onsets_in_window.iter().any(|onset| {
                band_overlap_fraction(band, (onset.band_min_hz, onset.band_max_hz)) >= SAME_BAND_MIN_OVERLAP_FRACTION
            })
        });
        if bands_overlap
            && geometric_dominance_db(target, interferer.evidence, classifying_node_id)
                < thresholds.dominance_margin_db - thresholds.spreading_tolerance_db
        {
            return refuse("bands_overlap_not_geometric", dominance_db);
        }
    }
    GradientDecision { settled: true, reason: "dominant_and_consistent", dominance_db }
}

/// A single-array cluster held back *only* by persistence whose band overlaps
/// no accepted cluster's band is a second source (an echo or sidelobe would
/// repeat the source's spectrum), so it gets its own product.
pub fn is_band_distinct_persistence_hold(
    status_reasons: &[String],
    band_hz: (f64, f64),
    accepted_bands_hz: &[(f64, f64)],
    same_band_min_overlap_fraction: f64,
) -> bool {
    status_reasons.len() == 1
        && status_reasons[0] == PERSISTENCE_HOLD_REASON
        && accepted_bands_hz.iter().all(|accepted| band_overlap_fraction(band_hz, *accepted) < same_band_min_overlap_fraction)
}

/// The classifier's answer for one window: top label, its confidence, all scores.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassifierAnswer {
    pub label: String,
    pub confidence: f64,
    pub scores: Vec<(String, f64)>,
}

/// Settle the target's label from a leave-interferers-out rerun, or refuse.
pub fn loco_decision(original: &ClassifierAnswer, masked: &ClassifierAnswer, thresholds: &AttributionThresholds) -> LocoDecision {
    let masked_scores = if masked.scores.is_empty() { vec![(masked.label.clone(), masked.confidence)] } else { masked.scores.clone() };
    let collapsed_labels: Vec<String> = original
        .scores
        .iter()
        .filter(|(label, score)| *score > 0.0 && score_for(&masked_scores, label).unwrap_or(0.0) <= thresholds.loco_collapse_ratio * score)
        .map(|(label, _)| label.clone())
        .collect();
    let refuse = |reason, collapsed_labels| LocoDecision { settled: false, reason, label: None, confidence: 0.0, collapsed_labels };
    if is_non_species_label(&masked.label) {
        return refuse("masked_rerun_abstained", collapsed_labels);
    }
    if masked.confidence < thresholds.loco_min_confidence {
        return refuse("masked_rerun_not_confident", collapsed_labels);
    }
    let species_ranking: Vec<(String, f64)> =
        ranked_descending(&masked_scores).into_iter().filter(|(label, _)| !is_non_species_label(label) && *label != masked.label).collect();
    let runner_up = species_ranking.first().map_or(0.0, |(_, score)| *score);
    if masked.confidence - runner_up < thresholds.loco_label_margin {
        return refuse("masked_rerun_ambiguous", collapsed_labels);
    }
    // A changed label is trusted only if the old winner collapsed (it was the interferer's).
    if masked.label != original.label && !is_non_species_label(&original.label) && !collapsed_labels.contains(&original.label) {
        return refuse("original_label_survived", collapsed_labels);
    }
    LocoDecision { settled: true, reason: "target_label_retained", label: Some(masked.label.clone()), confidence: masked.confidence, collapsed_labels }
}
