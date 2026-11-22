//! Entity-keyed reporting decision (port of the decision half of
//! `minimappr/core/entity_reporting_policy.py`; the storage half stays in
//! the server). Keyed on the bird entity and its sliding absorption window:
//!
//! - no matching entity → `Insert` (the post-assembly observe creates it);
//! - an unpositioned bout, or bearing evidence (a fix too uncertain to
//!   pinpoint, e.g. a far bird's unobservable range) matching an entity →
//!   `EnrichExisting`: absorbed, no new row, no track, the entity pulses;
//! - a positioned bout matching an entity → `UpgradeExisting`: the tracker
//!   gets the fix and the canonical row is updated in place.

use super::absorption_geometry::should_absorb_as_bearing_evidence;
use super::matching::DetectionGeometry;
use super::model::{EntityId, SpatialDisplayMode};
use super::registry::{BirdEntityChange, BirdEntityRegistry};
use crate::wildlife_tracking::{Covariance3, Position3};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityReportingAction {
    Insert,
    EnrichExisting,
    UpgradeExisting,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntityReportingDecision {
    pub action: EntityReportingAction,
    /// The matched entity (also set on an insert whose entity has no canonical row yet).
    pub entity_id: Option<EntityId>,
    pub absorbed_as_bearing_evidence: bool,
    pub absorbed_change: Option<BirdEntityChange>,
}

/// One classified detection before assembly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntityDecisionInput<'detection> {
    pub event_time_ns: i64,
    pub label: &'detection str,
    pub confidence: f64,
    pub spatial_display_mode: SpatialDisplayMode,
    pub reporting_node_id: Option<&'detection str>,
    pub reporting_node_position_m: Option<&'detection Position3>,
    pub position_m: Option<&'detection Position3>,
    pub covariance_m2: Option<&'detection Covariance3>,
    pub range_projection_mode: Option<&'detection str>,
    /// Two-threshold sub-display detections are never absorbed.
    pub below_display_threshold: bool,
}

pub fn decide_entity_reporting(registry: &mut BirdEntityRegistry, input: &EntityDecisionInput<'_>) -> EntityReportingDecision {
    let insert = EntityReportingDecision { action: EntityReportingAction::Insert, entity_id: None, absorbed_as_bearing_evidence: false, absorbed_change: None };
    if input.below_display_threshold {
        return insert;
    }
    let positioned = input.spatial_display_mode != SpatialDisplayMode::NodeOnly;
    let geometry = DetectionGeometry {
        spatial_display_mode: if positioned { input.spatial_display_mode } else { SpatialDisplayMode::NodeOnly },
        reporting_node_id: input.reporting_node_id,
        reporting_node_position_m: input.reporting_node_position_m,
        detection_position_m: if positioned { input.position_m } else { None },
        detection_covariance_m2: if positioned { input.covariance_m2 } else { None },
    };
    let Some(entity_id) = registry.find_absorbing_entity(input.label, input.event_time_ns, &geometry, None) else {
        return insert;
    };
    let entity = registry.get(entity_id).expect("found above");
    let bearing_evidence =
        positioned && should_absorb_as_bearing_evidence(entity, input.covariance_m2, input.range_projection_mode, registry.config());
    if entity.canonical_detection_id.is_none() {
        // Canonical row not written yet: a new row, same entity.
        return EntityReportingDecision { entity_id: Some(entity_id), ..insert };
    }
    if !positioned || bearing_evidence {
        let change = registry.absorb_unlocalized(entity_id, input.event_time_ns, input.label, input.confidence, input.reporting_node_id);
        return EntityReportingDecision {
            action: EntityReportingAction::EnrichExisting,
            entity_id: Some(entity_id),
            absorbed_as_bearing_evidence: bearing_evidence,
            absorbed_change: change,
        };
    }
    EntityReportingDecision { action: EntityReportingAction::UpgradeExisting, entity_id: Some(entity_id), absorbed_as_bearing_evidence: false, absorbed_change: None }
}
