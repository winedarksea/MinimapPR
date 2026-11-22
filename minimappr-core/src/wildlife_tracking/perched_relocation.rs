//! Perched relocation gate with ghost-perch revert (port of
//! `minimappr/core/track_perched_relocation.py`).
//!
//! A fix 10–30 m from a perched track is the same bird on a new perch when its
//! species matches the track's top species, no other track of that species is
//! nearer, and the track was silent long enough that one bird could not have
//! sung at both perches. The old perch is kept as a *ghost*: a same-species
//! fix back on it before it expires means the hop was a second bird, so the
//! hop is reverted and both tracks are barred from relocating for the ghost
//! window (one mistake per pair, no identity ping-pong across a flock). A
//! track hops at most once per ghost window. Reverts are reported so layers
//! that followed the hop (the entity registry) can re-read its fixes.

use std::collections::BTreeMap;

use super::covariance_math::distance_m;
use super::species_log_odds::normalised_label;
use super::{seconds_to_nanoseconds, Position3, TrackId};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelocationGateConfig {
    pub association_distance_m: f64,
    pub relocation_gate_max_m: f64,
    pub relocation_min_silence_seconds: f64,
    pub relocation_ghost_seconds: f64,
}

impl Default for RelocationGateConfig {
    fn default() -> Self {
        Self {
            association_distance_m: 8.0,
            relocation_gate_max_m: 30.0,
            relocation_min_silence_seconds: 2.0,
            relocation_ghost_seconds: 30.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RelocationGhost {
    pub track_id: TrackId,
    pub perch_position_m: Position3,
    pub left_at_ns: i64,
    pub species: String,
}

/// A relocation undone: `track_id` went back to its old perch; whatever it
/// picked up since `hop_time_ns` belongs to `split_track_id` (another bird).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelocationRevert {
    pub track_id: TrackId,
    pub split_track_id: TrackId,
    pub hop_time_ns: i64,
    pub reverted_at_ns: i64,
}

/// What the gate needs about one candidate track.
#[derive(Debug, Clone, PartialEq)]
pub struct RelocationCandidate {
    pub track_id: TrackId,
    pub position_m: Position3,
    pub last_seen_ns: i64,
    pub is_perched: bool,
    /// Normalised top species.
    pub top_species: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PerchedRelocationGate {
    config: RelocationGateConfig,
    min_silence_ns: i64,
    ghost_ns: i64,
    /// Insertion-ordered (the reference's dict order decides ghost-hit ties).
    ghosts: Vec<RelocationGhost>,
    relocation_blocked_until_ns: BTreeMap<TrackId, i64>,
    pub relocations_accepted: u64,
    pub relocations_reverted: u64,
}

impl PerchedRelocationGate {
    pub fn new(config: RelocationGateConfig) -> Self {
        Self {
            config,
            min_silence_ns: seconds_to_nanoseconds(config.relocation_min_silence_seconds),
            ghost_ns: seconds_to_nanoseconds(config.relocation_ghost_seconds),
            ghosts: Vec::new(),
            relocation_blocked_until_ns: BTreeMap::new(),
            relocations_accepted: 0,
            relocations_reverted: 0,
        }
    }

    /// The track this fix relocates (keeping its identity), or None.
    pub fn find_relocation_match(
        &mut self,
        timestamp_ns: i64,
        position_m: &Position3,
        detection_species: &str,
        candidates: &[RelocationCandidate],
    ) -> Option<TrackId> {
        self.expire(timestamp_ns);
        let species = normalised_label(detection_species);
        let mut nearest: Option<(f64, &RelocationCandidate)> = None;
        for candidate in candidates.iter().filter(|candidate| candidate.top_species.contains(&species)) {
            let distance = distance_m(position_m, &candidate.position_m);
            if nearest.is_none_or(|(best, _)| distance < best) {
                nearest = Some((distance, candidate));
            }
        }
        let (distance, nearest) = nearest?;
        // Only the nearest same-species track may relocate.
        if !nearest.is_perched {
            return None;
        }
        if !(self.config.association_distance_m <= distance && distance <= self.config.relocation_gate_max_m) {
            return None;
        }
        // One bird cannot sing at both perches within min_silence.
        if timestamp_ns - nearest.last_seen_ns < self.min_silence_ns {
            return None;
        }
        if self.relocation_blocked_until_ns.get(&nearest.track_id).copied().unwrap_or(0) > timestamp_ns {
            return None;
        }
        // One hop per ghost window: chaining would erase the ghost that tells a flock apart.
        if self.ghosts.iter().any(|ghost| ghost.track_id == nearest.track_id) {
            return None;
        }
        self.relocations_accepted += 1;
        if self.ghost_ns > 0 {
            self.ghosts.push(RelocationGhost {
                track_id: nearest.track_id,
                perch_position_m: nearest.position_m,
                left_at_ns: timestamp_ns,
                species,
            });
        }
        Some(nearest.track_id)
    }

    /// A same-species fix back on a fresh ghost perch, or None.
    pub fn find_ghost_hit(&mut self, timestamp_ns: i64, position_m: &Position3, detection_species: &str) -> Option<RelocationGhost> {
        self.expire(timestamp_ns);
        let species = normalised_label(detection_species);
        self.ghosts
            .iter()
            .find(|ghost| {
                ghost.species == species && distance_m(position_m, &ghost.perch_position_m) <= self.config.association_distance_m
            })
            .cloned()
    }

    pub fn mark_reverted(&mut self, ghost: &RelocationGhost, split_track_id: TrackId, timestamp_ns: i64) -> RelocationRevert {
        self.ghosts.retain(|existing| existing.track_id != ghost.track_id);
        let blocked_until = timestamp_ns + self.ghost_ns;
        self.relocation_blocked_until_ns.insert(ghost.track_id, blocked_until);
        self.relocation_blocked_until_ns.insert(split_track_id, blocked_until);
        self.relocations_reverted += 1;
        RelocationRevert { track_id: ghost.track_id, split_track_id, hop_time_ns: ghost.left_at_ns, reverted_at_ns: timestamp_ns }
    }

    pub fn forget(&mut self, track_id: TrackId) {
        self.ghosts.retain(|ghost| ghost.track_id != track_id);
        self.relocation_blocked_until_ns.remove(&track_id);
    }

    fn expire(&mut self, timestamp_ns: i64) {
        let ghost_ns = self.ghost_ns;
        self.ghosts.retain(|ghost| timestamp_ns - ghost.left_at_ns <= ghost_ns);
        self.relocation_blocked_until_ns.retain(|_, until| *until > timestamp_ns);
    }
}
