//! Is a detection spatially consistent with an entity? (port of
//! `minimappr/core/bird_entity_matching.py`). Scores are normalised
//! mismatches in [0, 1]; heard-nearby entities add a penalty so unpositioned
//! bouts prefer an entity already on the map. None = inconsistent.

use super::absorption_geometry::covariance_gated_fix_score;
use super::config::BirdEntityConfig;
use super::model::{angular_difference_deg, compass_bearing_deg, BirdEntityRecord, BirdEntitySpatialState, SpatialDisplayMode};
use crate::wildlife_tracking::covariance_math::horizontal_distance_m;
use crate::wildlife_tracking::{Covariance3, Position3};

/// Added to a heard-nearby candidate's score: one bird, one entity on the map.
pub const HEARD_NEARBY_MATCH_SCORE_PENALTY: f64 = 1.0;

/// What a detection offers for spatial matching.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectionGeometry<'detection> {
    pub spatial_display_mode: SpatialDisplayMode,
    pub reporting_node_id: Option<&'detection str>,
    pub reporting_node_position_m: Option<&'detection Position3>,
    pub detection_position_m: Option<&'detection Position3>,
    pub detection_covariance_m2: Option<&'detection Covariance3>,
}

pub fn spatial_mismatch_score(entity: &BirdEntityRecord, config: &BirdEntityConfig, detection: &DetectionGeometry<'_>) -> Option<f64> {
    if !entity.is_spatial() {
        // Heard-nearby entities have only a reporting node to compare against.
        let anchor = if detection.spatial_display_mode == SpatialDisplayMode::Localized {
            detection.detection_position_m
        } else {
            detection.reporting_node_position_m
        };
        let (Some(anchor), Some(entity_node_position)) = (anchor, entity.reporting_node_position_m.as_ref()) else {
            let same_node = detection.reporting_node_id.is_some() && detection.reporting_node_id == entity.reporting_node_id.as_deref();
            return same_node.then_some(HEARD_NEARBY_MATCH_SCORE_PENALTY);
        };
        let distance_m = horizontal_distance_m(anchor, entity_node_position);
        if distance_m > config.absorption_max_node_distance_m {
            return None;
        }
        return Some(HEARD_NEARBY_MATCH_SCORE_PENALTY + distance_m / config.absorption_max_node_distance_m);
    }
    let entity_position_m = entity.position_m.as_ref()?;
    match (detection.spatial_display_mode, detection.detection_position_m) {
        (SpatialDisplayMode::Localized, Some(detection_position_m)) => {
            if entity.state == BirdEntitySpatialState::Pinpointed && config.absorption_covariance_gate_enabled {
                return covariance_gated_fix_score(
                    entity_position_m,
                    entity.position_covariance_m2.as_ref(),
                    detection_position_m,
                    detection.detection_covariance_m2,
                    config,
                );
            }
            if entity.state == BirdEntitySpatialState::Pinpointed {
                let distance_m = horizontal_distance_m(detection_position_m, entity_position_m);
                return (distance_m <= config.absorption_max_fix_distance_m)
                    .then(|| distance_m / config.absorption_max_fix_distance_m);
            }
            bearing_mismatch(entity, detection_position_m, config)
        }
        (SpatialDisplayMode::BearingOnly, Some(detection_position_m)) => {
            let node_position_m = detection.reporting_node_position_m?;
            let difference_deg = angular_difference_deg(
                compass_bearing_deg(node_position_m, detection_position_m),
                compass_bearing_deg(node_position_m, entity_position_m),
            );
            (difference_deg <= config.absorption_max_bearing_difference_deg)
                .then(|| difference_deg / config.absorption_max_bearing_difference_deg)
        }
        _ => {
            // Unpositioned (node only): the reporting node must be near the bird.
            let node_position_m = detection.reporting_node_position_m?;
            let distance_m = horizontal_distance_m(node_position_m, entity_position_m);
            (distance_m <= config.absorption_max_node_distance_m).then(|| distance_m / config.absorption_max_node_distance_m)
        }
    }
}

/// A fix against a direction-known entity: inside its fan (with slack)?
fn bearing_mismatch(entity: &BirdEntityRecord, detection_position_m: &Position3, config: &BirdEntityConfig) -> Option<f64> {
    let bearing = entity.bearing.as_ref()?;
    let difference_deg =
        angular_difference_deg(compass_bearing_deg(&bearing.origin_position_m, detection_position_m), bearing.bearing_deg);
    let allowed_deg = bearing.half_width_deg + config.absorption_max_bearing_difference_deg;
    (difference_deg <= allowed_deg).then(|| difference_deg / allowed_deg)
}
