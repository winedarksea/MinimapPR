//! Tracker + entity registry in the order the server's FusionNode runs them
//! for an entity-routed (wildlife) detection:
//!
//! 1. reporting decision (pre-assembly): absorb unpositioned bouts and
//!    bearing evidence into a matching entity and stop there;
//! 2. assembly: a localized fix updates the perched tracker (a reverted hop
//!    releases the post-hop fixes from the hopper's entity);
//! 3. surfacing: unknown zero-confidence detections are dropped, the rest
//!    are observed into an entity, carrying the bound track's fused position
//!    when the canonical row persists it.
//!
//! Sans-IO: the mesh host (and tests) feed detections and maintenance ticks.

use std::collections::BTreeMap;

use super::absorption_geometry::forget_fixes_since;
use super::config::BirdEntityConfig;
use super::model::{EntityId, SpatialDisplayMode};
use super::registry::{BirdEntityChange, BirdEntityObservation, BirdEntityRegistry};
use super::reporting_decision::{decide_entity_reporting, EntityDecisionInput, EntityReportingAction, EntityReportingDecision};
use crate::wildlife_tracking::perched_tracker::{PerchedWildlifeTracker, TrackUpdateOutcome, WildlifeFix};
use crate::wildlife_tracking::species_log_odds::{LabelScores, UNKNOWN_LABEL};
use crate::wildlife_tracking::tracker_config::PerchedTrackerConfig;
use crate::wildlife_tracking::wildlife_track::WildlifeTrack;
use crate::wildlife_tracking::{Covariance3, Position3, TrackId};

/// One classified wildlife detection.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityPipelineDetection {
    pub detection_id: String,
    pub event_time_ns: i64,
    pub label: String,
    pub confidence: f64,
    pub species_evidence: LabelScores,
    pub spatial_display_mode: SpatialDisplayMode,
    pub reporting_node_id: Option<String>,
    pub reporting_node_position_m: Option<Position3>,
    pub position_m: Option<Position3>,
    pub covariance_m2: Option<Covariance3>,
    pub range_projection_mode: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntityPipelineOutcome {
    pub decision: EntityReportingDecision,
    pub track_update: Option<TrackUpdateOutcome>,
    /// The entity change from absorption or observation; None when the
    /// detection never reached an entity (dropped at surfacing).
    pub change: Option<BirdEntityChange>,
    /// Fixes released from an entity because the hop it followed was undone.
    pub revert_fixes_released: usize,
}

impl EntityPipelineOutcome {
    pub fn entity_id(&self) -> Option<EntityId> {
        self.change.as_ref().map(|change| change.entity_id)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WildlifeEntityPipeline {
    pub tracker: PerchedWildlifeTracker,
    pub registry: BirdEntityRegistry,
}

impl WildlifeEntityPipeline {
    pub fn new(tracker_config: PerchedTrackerConfig, entity_config: BirdEntityConfig) -> Self {
        Self { tracker: PerchedWildlifeTracker::new(tracker_config), registry: BirdEntityRegistry::new(entity_config) }
    }

    pub fn process(&mut self, detection: &EntityPipelineDetection) -> EntityPipelineOutcome {
        let decision = decide_entity_reporting(
            &mut self.registry,
            &EntityDecisionInput {
                event_time_ns: detection.event_time_ns,
                label: &detection.label,
                confidence: detection.confidence,
                spatial_display_mode: detection.spatial_display_mode,
                reporting_node_id: detection.reporting_node_id.as_deref(),
                reporting_node_position_m: detection.reporting_node_position_m.as_ref(),
                position_m: detection.position_m.as_ref(),
                covariance_m2: detection.covariance_m2.as_ref(),
                range_projection_mode: detection.range_projection_mode.as_deref(),
                below_display_threshold: false,
            },
        );
        if decision.action == EntityReportingAction::EnrichExisting {
            let change = decision.absorbed_change.clone();
            return EntityPipelineOutcome { decision, track_update: None, change, revert_fixes_released: 0 };
        }
        let mut revert_fixes_released = 0;
        let track_update = match (detection.spatial_display_mode, detection.position_m) {
            (SpatialDisplayMode::Localized, Some(position_m)) => {
                let outcome = self.tracker.update(&WildlifeFix {
                    timestamp_ns: detection.event_time_ns,
                    position_m,
                    covariance_m2: detection.covariance_m2,
                    label: detection.label.clone(),
                    confidence: detection.confidence,
                    species_evidence: detection.species_evidence.clone(),
                });
                if let Some(revert) = outcome.relocation_revert {
                    revert_fixes_released = self.release_post_hop_fixes(revert.track_id, revert.hop_time_ns);
                }
                Some(outcome)
            }
            _ => None,
        };
        // FusionNode never broadcasts an unknown, zero-confidence detection.
        if detection.label == UNKNOWN_LABEL && detection.confidence == 0.0 {
            return EntityPipelineOutcome { decision, track_update, change: None, revert_fixes_released };
        }
        let track = track_update.and_then(|outcome| self.tracker.track(outcome.track_id));
        let fused = self.registry.config().canonical_row_fused_position_enabled;
        let (position_m, covariance_m2) = match track {
            Some(track) if fused => (Some(track.position_m), Some(track.position_covariance_m2)),
            _ => (detection.position_m, detection.covariance_m2),
        };
        let observation = BirdEntityObservation {
            event_time_ns: detection.event_time_ns,
            label: detection.label.clone(),
            confidence: detection.confidence,
            spatial_display_mode: detection.spatial_display_mode,
            reporting_node_id: detection.reporting_node_id.clone(),
            reporting_node_position_m: detection.reporting_node_position_m,
            detection_position_m: position_m,
            detection_covariance_m2: covariance_m2,
            detection_id: Some(detection.detection_id.clone()),
            track,
        };
        let change = self.registry.observe(&observation, decision.entity_id);
        EntityPipelineOutcome { decision, track_update, change: Some(change), revert_fixes_released }
    }

    /// Maintenance tick: entities follow their tracks, step back, expire.
    pub fn maintain(&mut self, now_ns: i64) -> Vec<BirdEntityChange> {
        let tracks: BTreeMap<TrackId, WildlifeTrack> = self.tracker.tracks().map(|track| (track.id, track.clone())).collect();
        self.registry.maintain(now_ns, Some(&tracks))
    }

    fn release_post_hop_fixes(&mut self, track_id: TrackId, hop_time_ns: i64) -> usize {
        let Some(entity_id) = self.registry.entity_for_track(Some(track_id)) else { return 0 };
        self.registry.get_mut(entity_id).map_or(0, |entity| forget_fixes_since(entity, hop_time_ns))
    }
}
