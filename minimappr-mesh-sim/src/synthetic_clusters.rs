//! Layer 2 stand-in for onset clustering and the classifier on arrival-time
//! clips. Every scripted call that several clips share is a source cluster;
//! a secondary call whose arrivals are time-locked 3–30 ms behind an
//! accepted cluster's in the same band at most nodes is an echo (the S2
//! independent-onset rule, thresholds from `OnsetClusteringConfig`). Each
//! cluster carries S5 level evidence (arrival SNR per node, a 1/r fit via
//! `spreading_fit`) so the host's ported attribution cascade can run. The
//! classifier stand-in scores the calls audible on a node, graded by SNR
//! like the server's `SnrGradedSpeciesOracle`, with masked regions removed.

use std::collections::BTreeMap;

use minimappr_core::mesh::arrival_time_localizer::{ArrivalTimeLocalizer, NodeArrival};
use minimappr_core::mesh::call_pipeline::{CallFix, CallLocalizationRequest, LocalizedCluster};
use minimappr_core::mesh::clock_mapping::NodeClockMapping;
use minimappr_core::mesh::config::DEVICE_SAMPLE_RATE_HZ;
use minimappr_core::onset_clustering::attribution_decision::{band_overlap_fraction, AttributionThresholds, ClassifierAnswer, InterfererOnset};
use minimappr_core::onset_clustering::attribution_evidence::{spreading_fit, ClusterAttributionEvidence, ClusterOnsetArrival};
use minimappr_core::onset_clustering::echo_checks::SAME_BAND_MIN_OVERLAP_FRACTION;
use minimappr_core::onset_clustering::OnsetClusteringConfig;
use minimappr_core::protocol::NodeId;

use crate::runner_actions::{CALL_BAND_HIGH_FRACTION, CALL_BAND_LOW_FRACTION};
use crate::synthetic_clip::{decode_synthetic_clip, SolveTrace, SyntheticArrival};

/// Level recorded for a node whose clip holds no arrival of a cluster: the
/// clip-correlation floor (0 dB), i.e. censored below audibility.
const CENSORED_NODE_LEVEL_DB: f64 = 0.0;
/// The server oracle's SNR grading: 50 % of the ceiling at 12 dB, 3 dB slope.
const CLASSIFIER_MIDPOINT_SNR_DB: f64 = 12.0;
const CLASSIFIER_SLOPE_DB: f64 = 3.0;
/// A masked onset matches an arrival within this (sample rounding of both sides).
const MASK_MATCH_TOLERANCE_S: f64 = 1e-3;

pub type DecodedClips = Vec<(usize, Vec<SyntheticArrival>)>;

pub fn decode_request_clips(request: &CallLocalizationRequest) -> DecodedClips {
    request
        .clips
        .iter()
        .enumerate()
        .filter_map(|(index, clip)| Some((index, decode_synthetic_clip(&clip.payload.clip_container)?)))
        .collect()
}

fn arrival_band(arrival: &SyntheticArrival) -> (f64, f64) {
    let centre = f64::from(arrival.centre_frequency_hz);
    (centre * CALL_BAND_LOW_FRACTION, centre * CALL_BAND_HIGH_FRACTION)
}

fn arrival_of(arrivals: &[SyntheticArrival], call_index: u32) -> Option<&SyntheticArrival> {
    arrivals.iter().find(|arrival| arrival.call_index == call_index)
}

/// Calls by (clips holding them, summed SNR), strongest first; the first is
/// the call the single-source localizer always picked.
fn ranked_calls(decoded: &DecodedClips) -> Vec<u32> {
    let mut votes: BTreeMap<u32, (usize, f32)> = BTreeMap::new();
    for (_, arrivals) in decoded {
        for arrival in arrivals {
            let vote = votes.entry(arrival.call_index).or_insert((0, 0.0));
            vote.0 += 1;
            vote.1 += arrival.snr_db;
        }
    }
    let Some(primary) = votes.iter().max_by(|a, b| a.1 .0.cmp(&b.1 .0).then(a.1 .1.total_cmp(&b.1 .1))).map(|(call, _)| *call) else {
        return Vec::new();
    };
    let mut others: Vec<(u32, (usize, f32))> = votes.into_iter().filter(|(call, _)| *call != primary).collect();
    others.sort_by(|a, b| b.1 .0.cmp(&a.1 .0).then(b.1 .1.total_cmp(&a.1 .1)));
    std::iter::once(primary).chain(others.into_iter().map(|(call, _)| call)).collect()
}

/// Is `candidate` an echo of `leader`: time-locked behind it, same band, at most common nodes?
fn is_echo_of(decoded: &DecodedClips, leader: u32, candidate: u32, config: &OnsetClusteringConfig) -> bool {
    let (mut common, mut locked) = (0usize, 0usize);
    for (_, arrivals) in decoded {
        let (Some(lead), Some(follow)) = (arrival_of(arrivals, leader), arrival_of(arrivals, candidate)) else { continue };
        common += 1;
        let lag_s = (follow.arrival_device_sample - lead.arrival_device_sample) / DEVICE_SAMPLE_RATE_HZ;
        let same_band = band_overlap_fraction(arrival_band(lead), arrival_band(follow)) >= SAME_BAND_MIN_OVERLAP_FRACTION;
        if same_band && (config.echo_min_lag_seconds..=config.echo_max_lag_seconds).contains(&lag_s) {
            locked += 1;
        }
    }
    common > 0 && ((common - locked) as f64 / common as f64) < config.echo_min_independent_fraction
}

fn solve_call(
    localizer: &ArrivalTimeLocalizer,
    request: &CallLocalizationRequest,
    decoded: &DecodedClips,
    call_index: u32,
) -> (SolveTrace, Result<LocalizedCluster, String>) {
    let mut node_ids = Vec::new();
    let mut arrivals = Vec::new();
    let mut levels: Vec<(NodeId, f64, [f64; 3], f64)> = Vec::new();
    for (index, clip_arrivals) in decoded {
        let clip = &request.clips[*index];
        match arrival_of(clip_arrivals, call_index) {
            Some(arrival) => {
                let reference_s = clip.clock.reference_seconds_from_node_sample(arrival.arrival_device_sample);
                node_ids.push(clip.node_id);
                arrivals.push(NodeArrival {
                    position_m: clip.position.position_m,
                    position_std_m: clip.position.std_m,
                    reference_arrival_s: reference_s,
                    timing_std_s: (arrival.timing_std_s as f64).hypot(clip.clock.offset_sigma_s),
                });
                levels.push((clip.node_id, f64::from(arrival.snr_db), clip.position.position_m, reference_s));
            }
            None => levels.push((clip.node_id, CENSORED_NODE_LEVEL_DB, clip.position.position_m, f64::NAN)),
        }
    }
    let mut trace = SolveTrace {
        host_call_id: request.call_id,
        true_call_index: call_index,
        node_ids: node_ids.clone(),
        estimated_node_positions_m: arrivals.iter().map(|a| a.position_m).collect(),
        estimated_node_std_m: arrivals.iter().map(|a| a.position_std_m).collect(),
        localizer_inputs: arrivals.clone(),
        fix: None,
        chirality_confidence: request.chirality_confidence,
        chirality_locked: request.chirality_locked,
    };
    let band = decoded.iter().find_map(|(_, clip)| arrival_of(clip, call_index)).map(arrival_band).unwrap_or_default();
    let result = localizer.localize(&arrivals).map_err(|error| format!("{error:?}")).map(|solved| {
        let used_nodes: Vec<NodeId> = solved.used_arrival_indices.iter().map(|&index| node_ids[index]).collect();
        let fix = CallFix {
            horizontal_major_sigma_m: solved.fix.horizontal_major_sigma_m(),
            position_m: solved.fix.position_m,
            covariance_m2: solved.fix.covariance_m2,
            rejected_nodes: node_ids.iter().copied().filter(|n| !used_nodes.contains(n)).collect(),
            used_nodes,
        };
        let loudest = levels.iter().filter(|level| level.3.is_finite()).max_by(|a, b| a.1.total_cmp(&b.1)).map_or(request.classifying_node, |level| level.0);
        LocalizedCluster {
            attribution_evidence: Some(level_evidence(&levels, &fix.position_m, band)),
            fix,
            classifying_node: loudest,
            attributed_species: None,
            attribution_tier: None,
        }
    });
    trace.fix = result.as_ref().ok().map(|cluster| cluster.fix.clone());
    (trace, result)
}

/// S5 evidence from arrival levels: per-node SNR, distance from the fix, the arrival as an onset.
fn level_evidence(levels: &[(NodeId, f64, [f64; 3], f64)], fix_m: &[f64; 3], band: (f64, f64)) -> ClusterAttributionEvidence {
    let mut evidence = ClusterAttributionEvidence {
        node_band_snr_db: BTreeMap::new(),
        node_distances_m: BTreeMap::new(),
        node_onset_arrivals: BTreeMap::new(),
        fitted_source_level_db: None,
        spreading_residual_db: None,
    };
    for (node, snr_db, position_m, reference_s) in levels {
        let key = node.0.to_string();
        let distance_m = (0..3).map(|axis| (position_m[axis] - fix_m[axis]).powi(2)).sum::<f64>().sqrt();
        evidence.node_band_snr_db.insert(key.clone(), *snr_db);
        evidence.node_distances_m.insert(key.clone(), distance_m);
        if reference_s.is_finite() {
            let span_sample = (reference_s * DEVICE_SAMPLE_RATE_HZ).round() as i64;
            evidence.node_onset_arrivals.insert(key, vec![ClusterOnsetArrival { span_sample, band_min_hz: band.0, band_max_hz: band.1 }]);
        }
    }
    let audible_snr_db = AttributionThresholds::default().audible_snr_db;
    let (level, residual) = spreading_fit(&evidence.node_band_snr_db, &evidence.node_distances_m, audible_snr_db);
    evidence.fitted_source_level_db = level;
    evidence.spreading_residual_db = residual;
    evidence
}

/// Every accepted source in the window plus a solve trace per attempt. The
/// strongest call failing fails the window, as the single-source localizer did.
pub fn localize_window_clusters(
    localizer: &ArrivalTimeLocalizer,
    request: &CallLocalizationRequest,
    decoded: &DecodedClips,
) -> (Result<Vec<LocalizedCluster>, String>, Vec<SolveTrace>) {
    let ranked = ranked_calls(decoded);
    let Some(&primary) = ranked.first() else {
        return (Err("no correlatable arrival in any clip".to_string()), Vec::new());
    };
    let echo_config = OnsetClusteringConfig::default();
    let mut traces = Vec::new();
    let (trace, primary_result) = solve_call(localizer, request, decoded, primary);
    traces.push(trace);
    let mut clusters = match primary_result {
        Ok(cluster) => vec![cluster],
        Err(reason) => return (Err(reason), traces),
    };
    let mut accepted = vec![primary];
    for candidate in ranked.into_iter().skip(1) {
        if accepted.iter().any(|leader| is_echo_of(decoded, *leader, candidate, &echo_config)) {
            continue;
        }
        let (trace, result) = solve_call(localizer, request, decoded, candidate);
        if let Ok(cluster) = result {
            traces.push(trace);
            clusters.push(cluster);
            accepted.push(candidate);
        }
    }
    (Ok(clusters), traces)
}

/// The classifier stand-in on `node`'s clip with `masked` onset regions removed:
/// (answer, masked share of the audible calls). None when the node has no clip
/// or hears nothing above the detection threshold.
pub fn classify_window(
    request: &CallLocalizationRequest,
    decoded: &DecodedClips,
    node: NodeId,
    masked: &[InterfererOnset],
    detection_snr_db: f64,
) -> Option<(ClassifierAnswer, f64)> {
    let (_, arrivals) = decoded.iter().find(|(index, _)| request.clips[*index].node_id == node)?;
    let clip = request.clips.iter().find(|clip| clip.node_id == node)?;
    let window_start_s = request.centre_reference_s - request.half_window_s;
    classify_arrivals(arrivals, &clip.clock, window_start_s, masked, detection_snr_db, MASK_MATCH_TOLERANCE_S)
}

/// The SNR-graded classifier stand-in over the calls a node's window holds;
/// a call is masked when a masked onset (seconds after `window_start_s` on
/// the reference clock) lies within `mask_tolerance_s` of it in a shared band.
pub fn classify_arrivals(
    arrivals: &[SyntheticArrival],
    clock: &NodeClockMapping,
    window_start_s: f64,
    masked: &[InterfererOnset],
    detection_snr_db: f64,
    mask_tolerance_s: f64,
) -> Option<(ClassifierAnswer, f64)> {
    let audible: Vec<&SyntheticArrival> = arrivals.iter().filter(|arrival| f64::from(arrival.snr_db) >= detection_snr_db).collect();
    if audible.is_empty() {
        return None;
    }
    let is_masked = |arrival: &SyntheticArrival| {
        let onset_s = clock.reference_seconds_from_node_sample(arrival.arrival_device_sample) - window_start_s;
        masked.iter().any(|seed| {
            (seed.onset_seconds - onset_s).abs() <= mask_tolerance_s
                && band_overlap_fraction(arrival_band(arrival), (seed.band_min_hz, seed.band_max_hz)) >= SAME_BAND_MIN_OVERLAP_FRACTION
        })
    };
    let mut scores: Vec<(String, f64)> = Vec::new();
    let mut masked_count = 0usize;
    for arrival in &audible {
        if is_masked(arrival) {
            masked_count += 1;
            continue;
        }
        let graded = f64::from(arrival.classifier_score) / (1.0 + (-(f64::from(arrival.snr_db) - CLASSIFIER_MIDPOINT_SNR_DB) / CLASSIFIER_SLOPE_DB).exp());
        let label = arrival.species_label.to_string();
        match scores.iter_mut().find(|(existing, _)| *existing == label) {
            Some(entry) => entry.1 = entry.1.max(graded),
            None => scores.push((label, graded)),
        }
    }
    scores.sort_by(|a, b| b.1.total_cmp(&a.1));
    let (label, confidence) = scores.first().cloned().unwrap_or_else(|| ("unknown".to_string(), 0.0));
    Some((ClassifierAnswer { label, confidence, scores }, masked_count as f64 / audible.len() as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arrival(call_index: u32, sample: f64) -> SyntheticArrival {
        SyntheticArrival {
            call_index,
            arrival_device_sample: sample,
            snr_db: 20.0,
            timing_std_s: 1e-5,
            species_label: 1,
            classifier_score: 0.8,
            centre_frequency_hz: 4_000.0,
        }
    }

    fn clips_with_lag(lags_s: &[f64]) -> DecodedClips {
        lags_s
            .iter()
            .enumerate()
            .map(|(node, lag_s)| (node, vec![arrival(0, 1_000.0), arrival(1, 1_000.0 + lag_s * DEVICE_SAMPLE_RATE_HZ)]))
            .collect()
    }

    #[test]
    fn time_locked_same_band_follower_is_an_echo_and_an_independent_one_is_not() {
        let config = OnsetClusteringConfig::default();
        assert!(is_echo_of(&clips_with_lag(&[0.004, 0.008, 0.012, 0.02, -0.002]), 0, 1, &config));
        assert!(!is_echo_of(&clips_with_lag(&[0.6, 0.55, 0.62, 0.004, 0.58]), 0, 1, &config));
    }
}
