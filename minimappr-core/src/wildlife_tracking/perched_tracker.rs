//! The perched wildlife tracker: the server's `TrackManager.update` for a
//! category in the `stationary_hold` tracking mode (wildlife by default, see
//! [`super::tracking_mode`]) with species log-odds on (perched filter,
//! relocation gate + ghost revert, concurrent-perch guard, capped species
//! log-odds and the species association gate).
//!
//! Per fix: age tracks → undo a hop if the fix lands back on a ghost perch →
//! predict → drop species-incompatible and concurrently-sung perches → nearest
//! neighbour → relocation gate → new track. Not ported: dormant reacquire and
//! classifier fingerprints (the perched + entity layers keep identity here),
//! partial-constraint binding and two-threshold hidden tracks (S4, not
//! passed). Sans-IO: time is the caller's nanoseconds.

use std::collections::BTreeMap;

use super::nearest_neighbor_association::{associate_nearest_neighbor, AssociationCandidate};
use super::perched_concurrency::{concurrently_active_perch_conflicts, fix_lands_on_another_perch, PerchGuardTrack};
use super::perched_filter::PerchedRandomWalkFilter;
use super::perched_relocation::{PerchedRelocationGate, RelocationCandidate, RelocationRevert};
use super::species_association_gate::species_compatible_track_ids;
use super::species_log_odds::{first_maximum, normalised_label, LabelScores, SpeciesLogOddsBelief, UNKNOWN_LABEL};
use super::tracker_config::PerchedTrackerConfig;
use super::wildlife_track::{WildlifeTrack, WildlifeTrackStatus};
use super::{seconds_to_nanoseconds, Covariance3, Position3, TrackId, NANOSECONDS_PER_SECOND};

/// Smallest dt (s) between updates of one track (the reference's guard).
const MINIMUM_UPDATE_DT_SECONDS: f64 = 1e-3;

/// One localized, classified detection.
#[derive(Debug, Clone, PartialEq)]
pub struct WildlifeFix {
    pub timestamp_ns: i64,
    pub position_m: Position3,
    pub covariance_m2: Option<Covariance3>,
    pub label: String,
    pub confidence: f64,
    /// Label → probability (see `species_evidence_from_detection`).
    pub species_evidence: LabelScores,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrackUpdateOutcome {
    pub track_id: TrackId,
    pub created: bool,
    pub relocated: bool,
    /// A hop undone by this fix (the entity layer must forget post-hop fixes).
    pub relocation_revert: Option<RelocationRevert>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PerchedWildlifeTracker {
    config: PerchedTrackerConfig,
    tracks: BTreeMap<TrackId, WildlifeTrack>,
    beliefs: BTreeMap<TrackId, SpeciesLogOddsBelief>,
    filter: PerchedRandomWalkFilter,
    pub relocation_gate: PerchedRelocationGate,
    last_track_id: TrackId,
}

impl PerchedWildlifeTracker {
    pub fn new(config: PerchedTrackerConfig) -> Self {
        Self {
            config,
            tracks: BTreeMap::new(),
            beliefs: BTreeMap::new(),
            filter: PerchedRandomWalkFilter::new(config.perched_filter),
            relocation_gate: PerchedRelocationGate::new(config.relocation),
            last_track_id: 0,
        }
    }

    pub fn config(&self) -> &PerchedTrackerConfig {
        &self.config
    }

    /// Every track not yet reaped (dropped ones included), in creation order.
    pub fn tracks(&self) -> impl Iterator<Item = &WildlifeTrack> {
        self.tracks.values()
    }

    pub fn track(&self, track_id: TrackId) -> Option<&WildlifeTrack> {
        self.tracks.get(&track_id)
    }

    pub fn update(&mut self, fix: &WildlifeFix) -> TrackUpdateOutcome {
        self.age_tracks(fix.timestamp_ns);
        let detection_species = first_maximum(&fix.species_evidence).map(|(label, _)| label.to_string());
        let relocation_revert = detection_species.as_deref().and_then(|species| {
            let ghost = self.relocation_gate.find_ghost_hit(fix.timestamp_ns, &fix.position_m, species)?;
            self.revert_relocation(&ghost, fix.timestamp_ns)
        });
        let predicted = self.predicted_tracks(fix.timestamp_ns);
        let candidates = self.association_candidates(fix, detection_species.as_deref(), &predicted);
        let association_candidates: Vec<AssociationCandidate> = candidates
            .iter()
            .map(|track| AssociationCandidate {
                track_id: track.id,
                position_m: track.position_m,
                covariance_m2: Some(track.position_covariance_m2),
            })
            .collect();
        let mut matched =
            associate_nearest_neighbor(&fix.position_m, fix.covariance_m2.as_ref(), &association_candidates, &self.config.association);
        let mut relocated = false;
        if matched.is_none() {
            if let Some(species) = detection_species.as_deref() {
                let relocation_candidates: Vec<RelocationCandidate> = candidates
                    .iter()
                    .map(|track| RelocationCandidate {
                        track_id: track.id,
                        position_m: track.position_m,
                        last_seen_ns: track.last_seen_ns,
                        is_perched: self.filter.has_track(track.id),
                        top_species: self.normalised_top_species(track),
                    })
                    .collect();
                matched = self.relocation_gate.find_relocation_match(fix.timestamp_ns, &fix.position_m, species, &relocation_candidates);
                if let Some(track_id) = matched {
                    self.filter.initialize_track(track_id, &fix.position_m);
                    relocated = true;
                }
            }
        }
        let (track_id, created) = match matched {
            Some(track_id) => {
                self.update_matched(track_id, fix);
                (track_id, false)
            }
            None => (self.create_track(fix), true),
        };
        TrackUpdateOutcome { track_id, created, relocated, relocation_revert }
    }

    /// Coast, drop and reap tracks by silence (the reference's `_age_tracks`).
    pub fn age_tracks(&mut self, now_ns: i64) {
        let stale_ns = seconds_to_nanoseconds(self.config.track_stale_seconds);
        let drop_ns = seconds_to_nanoseconds(self.config.track_stale_seconds * self.config.track_drop_multiplier);
        let reap_ns = seconds_to_nanoseconds(self.config.track_stale_seconds * self.config.track_reap_multiplier);
        let mut reaped = Vec::new();
        for track in self.tracks.values_mut() {
            let gap_ns = now_ns - track.last_seen_ns;
            if track.is_dropped() {
                if gap_ns > reap_ns {
                    reaped.push(track.id);
                }
                continue;
            }
            if gap_ns > drop_ns {
                track.status = WildlifeTrackStatus::Dropped;
                self.filter.remove_track(track.id);
            } else if gap_ns > stale_ns
                && matches!(track.status, WildlifeTrackStatus::Tentative | WildlifeTrackStatus::Confirmed)
            {
                track.status = WildlifeTrackStatus::Coasting;
            }
        }
        for track_id in reaped {
            self.tracks.remove(&track_id);
            self.filter.remove_track(track_id);
            self.beliefs.remove(&track_id);
            self.relocation_gate.forget(track_id);
        }
    }

    fn next_track_id(&mut self) -> TrackId {
        self.last_track_id += 1;
        self.last_track_id
    }

    fn predicted_tracks(&self, timestamp_ns: i64) -> Vec<WildlifeTrack> {
        self.tracks
            .values()
            .filter(|track| !track.is_dropped())
            .map(|track| {
                let dt_s = ((timestamp_ns - track.last_seen_ns) as f64 / NANOSECONDS_PER_SECOND as f64).max(0.0);
                let mut predicted = track.clone();
                if let Some(estimate) = self.filter.predict(track.id, dt_s) {
                    predicted.position_m = estimate.position_m;
                    predicted.position_covariance_m2 = estimate.covariance_m2;
                }
                predicted
            })
            .collect()
    }

    /// Tracks the associator may consider: species-compatible, and not a perch
    /// another concurrently singing bird of the species would be merged into.
    fn association_candidates(&self, fix: &WildlifeFix, detection_species: Option<&str>, predicted: &[WildlifeTrack]) -> Vec<WildlifeTrack> {
        let by_id: BTreeMap<TrackId, &WildlifeTrack> = predicted.iter().map(|track| (track.id, track)).collect();
        let snap_distance_m = self.config.concurrent_perch_guard.snap_distance_m;
        let ids: Vec<TrackId> = predicted.iter().map(|track| track.id).collect();
        let compatible = species_compatible_track_ids(
            &ids,
            &fix.species_evidence,
            |track_id| self.belief_probabilities(track_id),
            |track_id| {
                let track = by_id[&track_id];
                !fix_lands_on_another_perch(
                    &fix.position_m,
                    fix.covariance_m2.as_ref(),
                    &track.position_m,
                    Some(&track.position_covariance_m2),
                    snap_distance_m,
                )
            },
            &self.config.species_association_gate,
        );
        let guard_tracks: Vec<PerchGuardTrack> = compatible
            .iter()
            .map(|track_id| {
                let track = by_id[track_id];
                PerchGuardTrack {
                    track_id: track.id,
                    position_m: track.position_m,
                    covariance_m2: Some(track.position_covariance_m2),
                    last_seen_ns: track.last_seen_ns,
                    is_perched: self.filter.has_track(track.id),
                    top_species: self.normalised_top_species(track),
                }
            })
            .collect();
        let conflicts = concurrently_active_perch_conflicts(
            fix.timestamp_ns,
            &fix.position_m,
            fix.covariance_m2.as_ref(),
            detection_species,
            &guard_tracks,
            &self.config.concurrent_perch_guard,
        );
        compatible
            .into_iter()
            .filter(|track_id| !conflicts.contains(track_id))
            .map(|track_id| by_id[&track_id].clone())
            .collect()
    }

    /// The track's top species (its belief's top label, else its own label).
    fn normalised_top_species(&self, track: &WildlifeTrack) -> Vec<String> {
        match self.beliefs.get(&track.id) {
            Some(belief) => belief.top_labels(1).iter().map(|label| normalised_label(label)).collect(),
            None => {
                let label = normalised_label(&track.label);
                if label.is_empty() || label == UNKNOWN_LABEL { Vec::new() } else { vec![label] }
            }
        }
    }

    pub fn belief_probabilities(&self, track_id: TrackId) -> LabelScores {
        self.beliefs.get(&track_id).map(SpeciesLogOddsBelief::exposed_probabilities).unwrap_or_default()
    }

    fn apply_belief(&mut self, track_id: TrackId, evidence: &[(String, f64)]) {
        let belief_config = self.config.species_belief;
        let belief = self.beliefs.entry(track_id).or_insert_with(|| SpeciesLogOddsBelief::new(belief_config));
        belief.update(evidence);
        let top = belief.top_label();
        let exposed = belief.exposed_probabilities();
        let track = self.tracks.get_mut(&track_id).expect("belief applied to a live track");
        if let Some((label, probability)) = &top {
            track.label = label.clone();
            track.confidence = *probability;
        }
        track.species_belief = exposed;
        track.species_top_label = top.as_ref().map(|(label, _)| label.clone());
        track.species_top_probability = top.map(|(_, probability)| probability);
    }

    fn create_track(&mut self, fix: &WildlifeFix) -> TrackId {
        let track_id = self.next_track_id();
        let initial_variance = self.config.perched_filter.initial_position_variance_m2;
        let mut isotropic = [[0.0; 3]; 3];
        (0..3).for_each(|axis| isotropic[axis][axis] = initial_variance);
        self.tracks.insert(
            track_id,
            WildlifeTrack {
                id: track_id,
                first_seen_ns: fix.timestamp_ns,
                last_seen_ns: fix.timestamp_ns,
                position_m: fix.position_m,
                position_covariance_m2: fix.covariance_m2.unwrap_or(isotropic),
                label: fix.label.clone(),
                confidence: fix.confidence,
                update_count: 1,
                status: WildlifeTrackStatus::Tentative,
                species_belief: Vec::new(),
                species_top_label: None,
                species_top_probability: None,
            },
        );
        self.filter.initialize_track(track_id, &fix.position_m);
        self.apply_belief(track_id, &fix.species_evidence);
        track_id
    }

    fn update_matched(&mut self, track_id: TrackId, fix: &WildlifeFix) {
        let track = self.tracks.get(&track_id).expect("matched a live track");
        let dt_s = ((fix.timestamp_ns - track.last_seen_ns) as f64 / NANOSECONDS_PER_SECOND as f64).max(MINIMUM_UPDATE_DT_SECONDS);
        let fallback_position_m = track.position_m;
        let estimate = self.filter.update(track_id, &fallback_position_m, &fix.position_m, dt_s, fix.covariance_m2.as_ref());
        let confirm_threshold = self.config.confirm_threshold;
        let track = self.tracks.get_mut(&track_id).expect("matched a live track");
        track.last_seen_ns = fix.timestamp_ns;
        track.position_m = estimate.position_m;
        track.position_covariance_m2 = estimate.covariance_m2;
        track.label = fix.label.clone();
        track.update_count += 1;
        track.status = match track.status {
            WildlifeTrackStatus::Tentative if track.update_count < confirm_threshold => WildlifeTrackStatus::Tentative,
            _ => WildlifeTrackStatus::Confirmed,
        };
        self.apply_belief(track_id, &fix.species_evidence);
    }

    /// A fix landed back on the perch a track just left: it was two birds. The
    /// track returns to its old perch; what it picked up continues as a new track.
    fn revert_relocation(&mut self, ghost: &super::perched_relocation::RelocationGhost, timestamp_ns: i64) -> Option<RelocationRevert> {
        let relocated = match self.tracks.get(&ghost.track_id) {
            Some(track) if !track.is_dropped() => track.clone(),
            _ => {
                // Its filter state is gone; reviving it would resurrect a dead track.
                self.relocation_gate.forget(ghost.track_id);
                return None;
            }
        };
        let split_id = self.next_track_id();
        let split = WildlifeTrack {
            id: split_id,
            first_seen_ns: ghost.left_at_ns,
            update_count: 1,
            status: WildlifeTrackStatus::Tentative,
            ..relocated.clone()
        };
        self.filter.initialize_track(split_id, &split.position_m);
        self.tracks.insert(split_id, split);
        if let Some(belief) = self.beliefs.get(&relocated.id).cloned() {
            self.beliefs.insert(split_id, belief);
        }
        if let Some(track) = self.tracks.get_mut(&relocated.id) {
            track.position_m = ghost.perch_position_m;
        }
        self.filter.initialize_track(relocated.id, &ghost.perch_position_m);
        Some(self.relocation_gate.mark_reverted(ghost, split_id, timestamp_ns))
    }
}
