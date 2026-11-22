//! S5 species attribution for a call window holding several source clusters
//! (the ported `onset_clustering::attribution_cascade` wired into the host).
//! Each cluster's classification window is classified on its loudest node;
//! the SNR-gradient tier or LOCO (mask the interferers, rerun) decides whether
//! that label belongs to the cluster; otherwise the cluster is unlabelled and
//! only its position stands. A single-cluster window keeps the call's belief.

use std::collections::BTreeMap;

use crate::onset_clustering::attribution_cascade::{
    attribute_cluster, AttributionCascadeConfig, AttributionTier, ClusterAttributionContext, LocoRerunBudget, LocoRerunHost,
};
use crate::onset_clustering::attribution_decision::{ClassificationWindow, ClassifierAnswer, InterfererOnset};
use crate::onset_clustering::numeric::python_round_to_i64;
use crate::protocol::wire_codes::SpeciesScore;

use super::call_pipeline_types::{CallLocalizationRequest, ClipSetLocalizer, LocalizedCluster};
use super::config::DEVICE_SAMPLE_RATE_HZ;

/// Masking and rerun through the localizer's classifier seam.
struct LocalizerRerunHost<'call> {
    localizer: &'call dyn ClipSetLocalizer,
    request: &'call CallLocalizationRequest,
    cluster: &'call LocalizedCluster,
    masked_answer: Option<ClassifierAnswer>,
}

impl LocoRerunHost for LocalizerRerunHost<'_> {
    fn mask_interferers(&mut self, seeds: &[InterfererOnset]) -> f64 {
        match self.localizer.classify_cluster_window(self.request, self.cluster, seeds) {
            Some((answer, masked_fraction)) => {
                self.masked_answer = Some(answer);
                masked_fraction
            }
            None => 0.0,
        }
    }

    fn rerun_masked(&mut self) -> ClassifierAnswer {
        self.masked_answer.clone().unwrap_or(ClassifierAnswer { label: String::new(), confidence: 0.0, scores: Vec::new() })
    }
}

fn species_scores(answer: &ClassifierAnswer) -> Vec<SpeciesScore> {
    answer
        .scores
        .iter()
        .filter_map(|(label, score)| Some(SpeciesScore { label_id: label.parse().ok()?, score: *score as f32 }))
        .collect()
}

pub fn attribute_window_clusters(
    request: &CallLocalizationRequest,
    clusters: &mut [LocalizedCluster],
    localizer: &dyn ClipSetLocalizer,
    config: &AttributionCascadeConfig,
    budget: &mut LocoRerunBudget,
) {
    if clusters.len() < 2 {
        return;
    }
    let evidence: BTreeMap<usize, _> = clusters
        .iter()
        .enumerate()
        .filter_map(|(index, cluster)| Some((index, cluster.attribution_evidence.clone()?)))
        .collect();
    let start_sample = python_round_to_i64((request.centre_reference_s - request.half_window_s) * DEVICE_SAMPLE_RATE_HZ);
    let end_sample = python_round_to_i64((request.centre_reference_s + request.half_window_s) * DEVICE_SAMPLE_RATE_HZ);
    let mut decisions: Vec<(usize, AttributionTier, Vec<SpeciesScore>)> = Vec::new();
    for (index, cluster) in clusters.iter().enumerate() {
        if !evidence.contains_key(&index) {
            continue;
        }
        let Some((original, _)) = localizer.classify_cluster_window(request, cluster, &[]) else { continue };
        let node_id = cluster.classifying_node.0.to_string();
        let context = ClusterAttributionContext {
            window_id: request.call_id.to_string(),
            target_index: index,
            cluster_evidence: evidence.clone(),
            window: ClassificationWindow {
                classifying_node_id: &node_id,
                start_sample,
                end_sample,
                sample_rate_hz: DEVICE_SAMPLE_RATE_HZ as u32,
                shared_onset_max_lag_samples: None,
            },
        };
        let mut host = LocalizerRerunHost { localizer, request, cluster, masked_answer: None };
        let outcome = attribute_cluster(&original, &context, config, budget, &mut host);
        let species = match outcome.tier {
            AttributionTier::SingleCluster | AttributionTier::SnrGradient => species_scores(&original),
            AttributionTier::Loco => host.masked_answer.as_ref().map(species_scores).unwrap_or_default(),
            AttributionTier::Unsettled => Vec::new(),
        };
        decisions.push((index, outcome.tier, species));
    }
    for (index, tier, species) in decisions {
        clusters[index].attribution_tier = Some(tier);
        clusters[index].attributed_species = Some(species);
    }
}
