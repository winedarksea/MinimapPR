//! Configuration of the perched wildlife tracker. Defaults are the server's
//! (`TrackingConfig` with wildlife in `stationary_hold`, species log-odds on,
//! and the wildlife stale time).

use super::nearest_neighbor_association::NearestNeighborAssociationConfig;
use super::perched_concurrency::ConcurrentPerchGuardConfig;
use super::perched_filter::{PerchedFilterConfig, DEFAULT_RELOCATION_SNAP_MIN_MAHALANOBIS_SQUARED};
use super::perched_relocation::RelocationGateConfig;
use super::species_association_gate::SpeciesAssociationGateConfig;
use super::species_log_odds::SpeciesBeliefConfig;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerchedTrackerConfig {
    pub association: NearestNeighborAssociationConfig,
    pub perched_filter: PerchedFilterConfig,
    pub relocation: RelocationGateConfig,
    pub concurrent_perch_guard: ConcurrentPerchGuardConfig,
    pub species_association_gate: SpeciesAssociationGateConfig,
    pub species_belief: SpeciesBeliefConfig,
    /// Seconds without a detection before a track coasts (server
    /// `track_stale_seconds_wildlife`).
    pub track_stale_seconds: f64,
    /// Dropped after `stale × drop_multiplier`, forgotten after `stale × reap_multiplier`.
    pub track_drop_multiplier: f64,
    pub track_reap_multiplier: f64,
    /// Detections needed to confirm a track.
    pub confirm_threshold: u32,
}

/// Server defaults for the knobs several sub-configs share.
pub const DEFAULT_ASSOCIATION_DISTANCE_M: f64 = 8.0;
pub const DEFAULT_RELOCATION_SNAP_DISTANCE_M: f64 = 6.0;
pub const DEFAULT_RELOCATION_MIN_SILENCE_SECONDS: f64 = 2.0;

impl Default for PerchedTrackerConfig {
    fn default() -> Self {
        Self {
            association: NearestNeighborAssociationConfig {
                association_distance_m: DEFAULT_ASSOCIATION_DISTANCE_M,
                ..NearestNeighborAssociationConfig::default()
            },
            perched_filter: PerchedFilterConfig {
                relocation_snap_distance_m: DEFAULT_RELOCATION_SNAP_DISTANCE_M,
                relocation_snap_min_mahalanobis_squared: DEFAULT_RELOCATION_SNAP_MIN_MAHALANOBIS_SQUARED,
                ..PerchedFilterConfig::default()
            },
            relocation: RelocationGateConfig {
                association_distance_m: DEFAULT_ASSOCIATION_DISTANCE_M,
                relocation_min_silence_seconds: DEFAULT_RELOCATION_MIN_SILENCE_SECONDS,
                ..RelocationGateConfig::default()
            },
            concurrent_perch_guard: ConcurrentPerchGuardConfig {
                snap_distance_m: DEFAULT_RELOCATION_SNAP_DISTANCE_M,
                min_silence_seconds: DEFAULT_RELOCATION_MIN_SILENCE_SECONDS,
                ..ConcurrentPerchGuardConfig::default()
            },
            species_association_gate: SpeciesAssociationGateConfig::default(),
            species_belief: SpeciesBeliefConfig::default(),
            track_stale_seconds: 120.0,
            track_drop_multiplier: 3.0,
            track_reap_multiplier: 5.0,
            confirm_threshold: 2,
        }
    }
}
