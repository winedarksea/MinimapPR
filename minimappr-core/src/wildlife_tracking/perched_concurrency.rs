//! Concurrent-perch guard (port of `minimappr/core/track_perched_concurrency.py`).
//!
//! A snap to a far perch is only "one bird hopping" when no other bird of the
//! species could be the singer. A perched same-species track may not take a
//! fix that lands on another perch when it sang less than `min_silence` ago,
//! or when another same-species track sang after its last bout (within the
//! concurrent-activity window): the silence was a neighbour's turn. The most
//! recent singer may still hop. "Another perch" is what the perched filter
//! would snap to: beyond the snap distance and χ²-inconsistent with the
//! predicted perch, so a far bird with tens of metres of fix error still
//! averages into its one track.

use std::collections::BTreeSet;

use nalgebra::Matrix3;

use super::covariance_math::{finite_symmetrized, mahalanobis_squared, vector_from_position};
use super::perched_filter::DEFAULT_RELOCATION_SNAP_MIN_MAHALANOBIS_SQUARED;
use super::species_log_odds::normalised_label;
use super::{seconds_to_nanoseconds, Covariance3, Position3, TrackId};

/// Variance floor (m²) so a missing or degenerate covariance cannot make the
/// innovation covariance singular.
const MINIMUM_VARIANCE_M2: f64 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConcurrentPerchGuardConfig {
    pub enabled: bool,
    pub snap_distance_m: f64,
    pub min_silence_seconds: f64,
    pub concurrent_activity_seconds: f64,
}

impl Default for ConcurrentPerchGuardConfig {
    fn default() -> Self {
        Self { enabled: true, snap_distance_m: 6.0, min_silence_seconds: 2.0, concurrent_activity_seconds: 30.0 }
    }
}

/// What the guard needs about one predicted, non-dropped track.
#[derive(Debug, Clone, PartialEq)]
pub struct PerchGuardTrack {
    pub track_id: TrackId,
    pub position_m: Position3,
    pub covariance_m2: Option<Covariance3>,
    pub last_seen_ns: i64,
    pub is_perched: bool,
    /// Normalised top species (empty when the track has none).
    pub top_species: Vec<String>,
}

/// Would the perched filter read this fix as a new perch for the track?
pub fn fix_lands_on_another_perch(
    position_m: &Position3,
    measurement_covariance_m2: Option<&Covariance3>,
    track_position_m: &Position3,
    track_covariance_m2: Option<&Covariance3>,
    snap_distance_m: f64,
) -> bool {
    let innovation = vector_from_position(position_m) - vector_from_position(track_position_m);
    if !(0.0 < snap_distance_m && snap_distance_m < innovation.norm()) {
        return false;
    }
    let mut combined = Matrix3::identity() * MINIMUM_VARIANCE_M2;
    for covariance in [finite_symmetrized(measurement_covariance_m2), finite_symmetrized(track_covariance_m2)]
        .into_iter()
        .flatten()
    {
        combined += covariance;
    }
    match mahalanobis_squared(&innovation, &combined) {
        Some(value) => value > DEFAULT_RELOCATION_SNAP_MIN_MAHALANOBIS_SQUARED,
        // Singular: the reference treats an unsolvable innovation as a new perch;
        // a non-finite score compares false there.
        None => combined.lu().solve(&innovation).is_none(),
    }
}

/// Ids of perched same-species tracks this fix must not snap onto.
pub fn concurrently_active_perch_conflicts(
    timestamp_ns: i64,
    position_m: &Position3,
    measurement_covariance_m2: Option<&Covariance3>,
    detection_species: Option<&str>,
    tracks: &[PerchGuardTrack],
    config: &ConcurrentPerchGuardConfig,
) -> BTreeSet<TrackId> {
    let mut conflicts = BTreeSet::new();
    let species = detection_species.map(normalised_label).unwrap_or_default();
    if !config.enabled || species.is_empty() {
        return conflicts;
    }
    let same_species: Vec<&PerchGuardTrack> =
        tracks.iter().filter(|track| track.is_perched && track.top_species.contains(&species)).collect();
    let min_silence_ns = seconds_to_nanoseconds(config.min_silence_seconds);
    let concurrent_ns = seconds_to_nanoseconds(config.concurrent_activity_seconds);
    for track in &same_species {
        if !fix_lands_on_another_perch(
            position_m,
            measurement_covariance_m2,
            &track.position_m,
            track.covariance_m2.as_ref(),
            config.snap_distance_m,
        ) {
            continue;
        }
        let sang_too_recently = timestamp_ns - track.last_seen_ns < min_silence_ns;
        let neighbour_sang_since = same_species.iter().any(|other| {
            other.track_id != track.track_id
                && other.last_seen_ns > track.last_seen_ns
                && timestamp_ns - other.last_seen_ns <= concurrent_ns
        });
        if sang_too_recently || neighbour_sang_since {
            conflicts.insert(track.track_id);
        }
    }
    conflicts
}
