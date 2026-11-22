//! The track record the perched tracker keeps for one bird, and its lifecycle
//! status (the subset of the server's `TrackState` the wildlife path reads).

use super::species_log_odds::LabelScores;
use super::{Covariance3, Position3, TrackId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WildlifeTrackStatus {
    Tentative,
    Confirmed,
    Coasting,
    Dropped,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WildlifeTrack {
    pub id: TrackId,
    pub first_seen_ns: i64,
    pub last_seen_ns: i64,
    pub position_m: Position3,
    /// Filtered posterior after the last update (the fix covariance at creation).
    pub position_covariance_m2: Covariance3,
    pub label: String,
    pub confidence: f64,
    pub update_count: u32,
    pub status: WildlifeTrackStatus,
    /// Exposed top-N belief (rounded), as other layers read it.
    pub species_belief: LabelScores,
    pub species_top_label: Option<String>,
    pub species_top_probability: Option<f64>,
}

impl WildlifeTrack {
    pub fn is_dropped(&self) -> bool {
        self.status == WildlifeTrackStatus::Dropped
    }
}
