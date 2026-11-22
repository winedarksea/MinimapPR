//! Species-compatibility term for association (port of
//! `minimappr/core/track_species_association_gate.py`, S-fixes tracking).
//!
//! A detection whose top species claims at least `detection_claim_probability`
//! never joins a track whose belief is confident (≥ `confident_probability`)
//! in another species, unless either side supports the other's species. A
//! *weak* contradicting claim may still join when it sits on the track's own
//! perch, so one misclassified bout feeds its bird's belief instead of
//! spawning a track; off the perch it is refused (sub-display tanager bouts
//! must not build a tanager belief inside the grosbeak's track). Unknown
//! labels and scores under the claim floor stay wildcards.

use super::species_log_odds::{first_maximum, score_for};
use super::TrackId;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeciesAssociationGateConfig {
    pub enabled: bool,
    /// A track belief at or above this is a confident species (the display floor).
    pub confident_probability: f64,
    /// A detection score at or above this is a claim that can contradict a confident track.
    pub detection_claim_probability: f64,
}

impl Default for SpeciesAssociationGateConfig {
    fn default() -> Self {
        Self { enabled: true, confident_probability: 0.5, detection_claim_probability: 0.15 }
    }
}

fn confident_top(probability_by_label: &[(String, f64)], floor: f64) -> Option<&str> {
    first_maximum(probability_by_label).and_then(|(label, probability)| (probability >= floor).then_some(label))
}

/// True when a detection's species claim contradicts a confident track belief.
pub fn species_claims_contradict(
    detection_evidence: &[(String, f64)],
    track_belief: &[(String, f64)],
    confident_probability: f64,
    detection_claim_probability: f64,
) -> bool {
    let (Some(detection_top), Some(track_top)) = (
        confident_top(detection_evidence, detection_claim_probability),
        confident_top(track_belief, confident_probability),
    ) else {
        return false;
    };
    if detection_top == track_top {
        return false;
    }
    let track_supports_detection = score_for(track_belief, detection_top).unwrap_or(0.0) >= confident_probability;
    let detection_supports_track = score_for(detection_evidence, track_top).unwrap_or(0.0) >= detection_claim_probability;
    !(track_supports_detection || detection_supports_track)
}

/// The candidate track ids minus those whose confident belief the detection contradicts.
pub fn species_compatible_track_ids(
    track_ids: &[TrackId],
    detection_evidence: &[(String, f64)],
    belief_for_track: impl Fn(TrackId) -> Vec<(String, f64)>,
    weak_claim_fits_track: impl Fn(TrackId) -> bool,
    config: &SpeciesAssociationGateConfig,
) -> Vec<TrackId> {
    if !config.enabled || detection_evidence.is_empty() {
        return track_ids.to_vec();
    }
    let strongest_claim = detection_evidence.iter().map(|(_, value)| *value).fold(f64::NEG_INFINITY, f64::max);
    let claim_is_confident = strongest_claim >= config.confident_probability;
    track_ids
        .iter()
        .copied()
        .filter(|track_id| {
            let contradicts = species_claims_contradict(
                detection_evidence,
                &belief_for_track(*track_id),
                config.confident_probability,
                config.detection_claim_probability,
            );
            !contradicts || (!claim_is_confident && weak_claim_fits_track(*track_id))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scores(pairs: &[(&str, f64)]) -> Vec<(String, f64)> {
        pairs.iter().map(|(label, value)| (label.to_string(), *value)).collect()
    }

    #[test]
    fn contradiction_needs_two_confident_disagreeing_claims() {
        let track = scores(&[("tanager", 0.8)]);
        assert!(species_claims_contradict(&scores(&[("grosbeak", 0.9)]), &track, 0.5, 0.15));
        assert!(species_claims_contradict(&scores(&[("grosbeak", 0.3)]), &track, 0.5, 0.15));
        assert!(!species_claims_contradict(&scores(&[("grosbeak", 0.1)]), &track, 0.5, 0.15));
        assert!(!species_claims_contradict(&scores(&[("grosbeak", 0.9)]), &scores(&[("tanager", 0.4)]), 0.5, 0.15));
        assert!(!species_claims_contradict(&scores(&[("grosbeak", 0.9), ("tanager", 0.6)]), &track, 0.5, 0.15));
    }
}
