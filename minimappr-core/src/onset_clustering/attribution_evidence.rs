//! Per-cluster level evidence for species attribution (port of
//! `minimappr/core/onset_cluster_attribution_evidence.py`, S5).
//!
//! Per node: the noise-floor-relative band SNR at the cluster's led onsets
//! (read at each onset's predicted arrival, in that onset's band), those
//! predicted arrivals (so a masking step knows the cluster's time-frequency
//! regions in any node's window), and how well the SNR profile fits 1/r
//! spreading from the cluster's position. Nodes under `audible_snr_db` are
//! censored: excluded from the source-level fit, penalised only when the fit
//! says they should clearly have heard it. The attribution *decision* (SNR
//! gradient tier, LOCO) consumes this; it is not made here.

use std::collections::BTreeMap;

use super::echo_checks::node_band_snr_db;
use super::numeric::{median, python_round_to_i64};
use super::onset_detection::SubbandHistory;
use super::srp_grid::{distance, Point3};
use super::types::OnsetFrame;

pub const DECIBELS_PER_AMPLITUDE_DECADE: f64 = 20.0;
/// Distances below this are clamped before taking log10 (a node at the source).
pub const MINIMUM_SPREADING_DISTANCE_M: f64 = 0.1;

/// One led onset of a cluster as it reaches one node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusterOnsetArrival {
    pub span_sample: i64,
    pub band_min_hz: f64,
    pub band_max_hz: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClusterAttributionEvidence {
    pub node_band_snr_db: BTreeMap<String, f64>,
    pub node_distances_m: BTreeMap<String, f64>,
    pub node_onset_arrivals: BTreeMap<String, Vec<ClusterOnsetArrival>>,
    /// Source level (dB above floor at 1 m) from audible nodes; None with none audible.
    pub fitted_source_level_db: Option<f64>,
    /// RMS of measured-minus-predicted (audible + censored violations); None with < 2 nodes.
    pub spreading_residual_db: Option<f64>,
}

fn spreading_db(distance_m: f64) -> f64 {
    DECIBELS_PER_AMPLITUDE_DECADE * distance_m.max(MINIMUM_SPREADING_DISTANCE_M).log10()
}

impl ClusterAttributionEvidence {
    /// Spreading-model SNR of this cluster at a node (None without a fit).
    pub fn predicted_snr_db(&self, node_id: &str) -> Option<f64> {
        Some(self.fitted_source_level_db? - spreading_db(*self.node_distances_m.get(node_id)?))
    }
}

fn spreading_fit(
    snr_by_node: &BTreeMap<String, f64>,
    distances_m: &BTreeMap<String, f64>,
    audible_snr_db: f64,
) -> (Option<f64>, Option<f64>) {
    let audible: Vec<&String> = snr_by_node
        .iter()
        .filter(|(_, snr)| **snr >= audible_snr_db)
        .map(|(node, _)| node)
        .collect();
    if audible.is_empty() {
        return (None, None);
    }
    let source_level = audible
        .iter()
        .map(|node| snr_by_node[*node] + spreading_db(distances_m[*node]))
        .sum::<f64>()
        / audible.len() as f64;
    if snr_by_node.len() < 2 {
        return (Some(source_level), None);
    }
    let mut residuals = Vec::new();
    for (node, measured) in snr_by_node {
        let predicted = source_level - spreading_db(distances_m[node]);
        if audible.contains(&node) {
            residuals.push(measured - predicted);
        } else if predicted > audible_snr_db {
            // Censored: only the part of the prediction above audibility is a violation.
            residuals.push(predicted - measured.max(audible_snr_db));
        }
    }
    if residuals.is_empty() {
        return (Some(source_level), None);
    }
    let mean_square =
        residuals.iter().map(|value| value * value).sum::<f64>() / residuals.len() as f64;
    (Some(source_level), Some(mean_square.sqrt()))
}

/// Inputs shared by every cluster of a window.
pub struct AttributionEvidenceContext<'window> {
    pub node_histories: &'window BTreeMap<String, SubbandHistory>,
    pub node_positions_m: &'window BTreeMap<String, Point3>,
    pub sound_speed_mps: f64,
    pub sample_rate_hz: u32,
    pub frame_samples: usize,
    pub audible_snr_db: f64,
}

pub fn build_cluster_attribution_evidence(
    position_m: &Point3,
    led_frames: &[&OnsetFrame],
    context: &AttributionEvidenceContext<'_>,
) -> ClusterAttributionEvidence {
    let distances: BTreeMap<String, f64> = context
        .node_positions_m
        .iter()
        .map(|(node, centre)| (node.clone(), distance(position_m, centre)))
        .collect();
    let closest = distances.values().copied().fold(f64::INFINITY, f64::min);
    let mut arrivals: BTreeMap<String, Vec<ClusterOnsetArrival>> = distances
        .keys()
        .map(|node| (node.clone(), Vec::new()))
        .collect();
    let mut readings: BTreeMap<String, Vec<f64>> = distances
        .keys()
        .map(|node| (node.clone(), Vec::new()))
        .collect();
    for frame in led_frames {
        let template_distance = distances
            .get(&frame.template_node_id)
            .copied()
            .unwrap_or(closest);
        for (node, node_distance) in &distances {
            let delay = (node_distance - template_distance) / context.sound_speed_mps
                * context.sample_rate_hz as f64;
            let arrival = frame.onset_sample as i64 + python_round_to_i64(delay);
            arrivals
                .get_mut(node)
                .expect("node")
                .push(ClusterOnsetArrival {
                    span_sample: arrival,
                    band_min_hz: frame.band_min_hz,
                    band_max_hz: frame.band_max_hz,
                });
            let Some(history) = context.node_histories.get(node) else {
                continue;
            };
            if let Some(snr) = node_band_snr_db(
                history,
                arrival,
                context.frame_samples,
                frame.band_min_hz,
                frame.band_max_hz,
            ) {
                readings.get_mut(node).expect("node").push(snr);
            }
        }
    }
    let snr_by_node: BTreeMap<String, f64> = readings
        .iter()
        .filter(|(_, values)| !values.is_empty())
        .map(|(node, values)| (node.clone(), median(values)))
        .collect();
    let (fitted_source_level_db, spreading_residual_db) =
        spreading_fit(&snr_by_node, &distances, context.audible_snr_db);
    ClusterAttributionEvidence {
        node_band_snr_db: snr_by_node,
        node_distances_m: distances,
        node_onset_arrivals: arrivals,
        fitted_source_level_db,
        spreading_residual_db,
    }
}
