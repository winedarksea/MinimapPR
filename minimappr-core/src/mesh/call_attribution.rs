//! S5 species attribution for a call window holding several source clusters
//! (the ported `onset_clustering::attribution_cascade` wired into the host).
//! Each cluster's classification window is classified on its loudest node;
//! the SNR-gradient tier or LOCO (mask the interferers, rerun) decides whether
//! that label belongs to the cluster; otherwise the cluster is unlabelled and
//! only its position stands. A single-cluster window keeps the call's belief.
//!
//! The window context is built as the server's onset stage builds it
//! (`_attach_attribution_context`): clusters colocated with the target (one
//! source split in two) are not interferers, and interferer onsets within the
//! echo lag of a target onset in the same band are the target's own note.

use std::collections::BTreeMap;

use crate::onset_clustering::attribution_cascade::{
    attribute_cluster, AttributionTier, ClusterAttributionContext, LocoRerunBudget, LocoRerunHost,
};
use crate::onset_clustering::attribution_decision::{ClassificationWindow, ClassifierAnswer, InterfererOnset};
use crate::onset_clustering::numeric::python_round_to_i64;
use crate::onset_clustering::srp_grid::spatial_cluster_radius_m;
use crate::onset_clustering::window_label_reconciliation::{labels_to_drop_by_cluster, top_scoring_label};
use crate::protocol::wire_codes::SpeciesScore;

use super::call_pipeline_types::{CallLocalizationRequest, ClipSetLocalizer, LocalizedCluster};
use super::config::{MeshAttributionConfig, DEVICE_SAMPLE_RATE_HZ};

/// Masking and rerun through the localizer's classifier seam. The rerun runs
/// only when the cascade asks for it, i.e. after the rerun budget admitted it,
/// so the per-window cap bounds classifier CPU as on the server.
struct LocalizerRerunHost<'call> {
    localizer: &'call dyn ClipSetLocalizer,
    request: &'call CallLocalizationRequest,
    cluster: &'call LocalizedCluster,
    masked_seeds: Vec<InterfererOnset>,
    masked_answer: Option<ClassifierAnswer>,
}

impl LocoRerunHost for LocalizerRerunHost<'_> {
    fn mask_interferers(&mut self, seeds: &[InterfererOnset]) -> f64 {
        self.masked_seeds = seeds.to_vec();
        self.localizer.masked_cell_fraction(self.request, self.cluster, seeds)
    }

    fn rerun_masked(&mut self) -> ClassifierAnswer {
        let answer = self
            .localizer
            .classify_cluster_window(self.request, self.cluster, &self.masked_seeds)
            .map(|(answer, _)| answer)
            // No answer from the classifier: an abstention, which LOCO never settles on.
            .unwrap_or(ClassifierAnswer { label: String::new(), confidence: 0.0, scores: Vec::new() });
        self.masked_answer = Some(answer.clone());
        answer
    }
}

fn species_scores(answer: &ClassifierAnswer) -> Vec<SpeciesScore> {
    answer
        .scores
        .iter()
        .filter_map(|(label, score)| Some(SpeciesScore { label_id: label.parse().ok()?, score: *score as f32 }))
        .collect()
}

fn distance_m(first: &[f64; 3], second: &[f64; 3]) -> f64 {
    (0..3).map(|axis| (first[axis] - second[axis]).powi(2)).sum::<f64>().sqrt()
}

/// The request's node centroid (ranges for the colocation radius are measured from it).
fn node_centroid_m(request: &CallLocalizationRequest) -> Option<[f64; 3]> {
    if request.clips.is_empty() {
        return None;
    }
    let count = request.clips.len() as f64;
    Some([0, 1, 2].map(|axis| request.clips.iter().map(|clip| clip.position.position_m[axis]).sum::<f64>() / count))
}

/// Within one clustering radius: spatially the same source, never an interferer.
fn colocated(first: &[f64; 3], second: &[f64; 3], centroid_m: &[f64; 3], config: &MeshAttributionConfig) -> bool {
    let range_m = distance_m(first, centroid_m).min(distance_m(second, centroid_m));
    let radius_m = spatial_cluster_radius_m(range_m, config.colocated_cluster_radius_m, config.colocated_cluster_range_fraction);
    distance_m(first, second) <= radius_m
}

/// Gradient-settled clusters drop labels another settled sibling claims as its
/// top label (closeout B2; server: `onset_window_label_reconciliation.py`).
/// Each entry is `(cluster index, tier, attributed species)`.
pub fn reconcile_window_labels(decisions: &mut [(usize, AttributionTier, Vec<SpeciesScore>)]) {
    let tiers_and_tops: Vec<(AttributionTier, Option<u32>)> = decisions
        .iter()
        .map(|(_, tier, species)| {
            let scores: Vec<(u32, f64)> = species.iter().map(|entry| (entry.label_id, f64::from(entry.score))).collect();
            (*tier, top_scoring_label(&scores))
        })
        .collect();
    for ((_, _, species), dropped) in decisions.iter_mut().zip(labels_to_drop_by_cluster(&tiers_and_tops)) {
        species.retain(|entry| !dropped.contains(&entry.label_id));
    }
}

pub fn attribute_window_clusters(
    request: &CallLocalizationRequest,
    clusters: &mut [LocalizedCluster],
    localizer: &dyn ClipSetLocalizer,
    config: &MeshAttributionConfig,
    budget: &mut LocoRerunBudget,
) {
    if clusters.len() < 2 {
        return;
    }
    let Some(centroid_m) = node_centroid_m(request) else { return };
    let start_sample = python_round_to_i64((request.centre_reference_s - request.half_window_s) * DEVICE_SAMPLE_RATE_HZ);
    let end_sample = python_round_to_i64((request.centre_reference_s + request.half_window_s) * DEVICE_SAMPLE_RATE_HZ);
    let shared_onset_max_lag_samples = python_round_to_i64(config.shared_onset_max_lag_seconds * DEVICE_SAMPLE_RATE_HZ);
    let mut decisions: Vec<(usize, AttributionTier, Vec<SpeciesScore>)> = Vec::new();
    for (index, cluster) in clusters.iter().enumerate() {
        let Some(target_evidence) = cluster.attribution_evidence.clone() else { continue };
        let mut evidence: BTreeMap<usize, _> = clusters
            .iter()
            .enumerate()
            .filter(|(other_index, other)| {
                *other_index != index && !colocated(&other.fix.position_m, &cluster.fix.position_m, &centroid_m, config)
            })
            .filter_map(|(other_index, other)| Some((other_index, other.attribution_evidence.clone()?)))
            .collect();
        evidence.insert(index, target_evidence);
        let Some((original, _)) = localizer.classify_cluster_window(request, cluster, &[]) else { continue };
        let node_id = cluster.classifying_node.0.to_string();
        let context = ClusterAttributionContext {
            window_id: request.call_id.to_string(),
            target_index: index,
            cluster_evidence: evidence,
            window: ClassificationWindow {
                classifying_node_id: &node_id,
                start_sample,
                end_sample,
                sample_rate_hz: DEVICE_SAMPLE_RATE_HZ as u32,
                shared_onset_max_lag_samples: Some(shared_onset_max_lag_samples),
            },
        };
        let mut host = LocalizerRerunHost { localizer, request, cluster, masked_seeds: Vec::new(), masked_answer: None };
        let outcome = attribute_cluster(&original, &context, &config.cascade, budget, &mut host);
        let species = match outcome.tier {
            AttributionTier::SingleCluster | AttributionTier::SnrGradient => species_scores(&original),
            AttributionTier::Loco => host.masked_answer.as_ref().map(species_scores).unwrap_or_default(),
            AttributionTier::Unsettled => Vec::new(),
        };
        decisions.push((index, outcome.tier, species));
    }
    if config.window_label_reconciliation_enabled {
        reconcile_window_labels(&mut decisions);
    }
    for (index, tier, species) in decisions {
        clusters[index].attribution_tier = Some(tier);
        clusters[index].attributed_species = Some(species);
    }
}
