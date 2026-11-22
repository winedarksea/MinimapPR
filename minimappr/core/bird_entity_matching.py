"""Field Mode W3-entity: is a detection spatially consistent with an entity?

The spatial half of the sliding absorption window (species + time live in
``BirdEntityRegistry.find_absorbing_entity``). Scores are normalised
mismatches in [0, 1]; heard-nearby entities add a penalty so unpositioned
bouts prefer an entity already on the map. None = inconsistent.
"""

from __future__ import annotations

from minimappr.core.bird_entity_absorption_geometry import covariance_gated_fix_score
from minimappr.core.bird_entity_model import (
    ENTITY_STATE_PINPOINTED,
    BirdEntityRecord,
    Vec3,
    angular_difference_deg,
    compass_bearing_deg,
    horizontal_distance_m,
)
from minimappr.core.entity_reporting_config import EntityReportingConfig

SPATIAL_DISPLAY_LOCALIZED = "localized"
SPATIAL_DISPLAY_BEARING_ONLY = "bearing_only"
# Added to a heard-nearby candidate's match score so an unpositioned detection
# prefers absorbing into an entity already on the map (one bird, one entity).
HEARD_NEARBY_MATCH_SCORE_PENALTY = 1.0


def spatial_mismatch_score(
    entity: BirdEntityRecord,
    config: EntityReportingConfig,
    *,
    spatial_display_mode: str,
    reporting_node_id: str | None,
    reporting_node_position_m: Vec3 | None,
    detection_position_m: Vec3 | None,
    detection_covariance_m2: list[list[float]] | None = None,
) -> float | None:
    """Normalised mismatch in [0, 1] (+ penalty for heard-nearby), None = inconsistent."""
    if not entity.is_spatial:
        # Heard-nearby entities have only a reporting node to compare against.
        anchor = (
            detection_position_m
            if spatial_display_mode == SPATIAL_DISPLAY_LOCALIZED
            else reporting_node_position_m
        )
        if anchor is None or entity.reporting_node_position_m is None:
            same_node = reporting_node_id is not None and reporting_node_id == entity.reporting_node_id
            return HEARD_NEARBY_MATCH_SCORE_PENALTY if same_node else None
        distance_m = horizontal_distance_m(anchor, entity.reporting_node_position_m)
        if distance_m > config.absorption_max_node_distance_m:
            return None
        return HEARD_NEARBY_MATCH_SCORE_PENALTY + distance_m / config.absorption_max_node_distance_m
    entity_position_m = entity.position_m
    if entity_position_m is None:
        return None
    if spatial_display_mode == SPATIAL_DISPLAY_LOCALIZED and detection_position_m is not None:
        if entity.state == ENTITY_STATE_PINPOINTED and config.absorption_covariance_gate_enabled:
            # Field Mode S-fixes tracking: the fix's own covariance, not a fixed radius.
            return covariance_gated_fix_score(
                entity_position_m,
                entity.position_covariance_m2,
                detection_position_m,
                detection_covariance_m2,
                config,
            )
        if entity.state == ENTITY_STATE_PINPOINTED:
            distance_m = horizontal_distance_m(detection_position_m, entity_position_m)
            if distance_m > config.absorption_max_fix_distance_m:
                return None
            return distance_m / config.absorption_max_fix_distance_m
        return _bearing_mismatch(entity, detection_position_m, config)
    if spatial_display_mode == SPATIAL_DISPLAY_BEARING_ONLY and detection_position_m is not None:
        if reporting_node_position_m is None:
            return None
        difference_deg = angular_difference_deg(
            compass_bearing_deg(reporting_node_position_m, detection_position_m),
            compass_bearing_deg(reporting_node_position_m, entity_position_m),
        )
        if difference_deg > config.absorption_max_bearing_difference_deg:
            return None
        return difference_deg / config.absorption_max_bearing_difference_deg
    # Unpositioned (node_only): the reporting node must be near the bird.
    if reporting_node_position_m is None:
        return None
    distance_m = horizontal_distance_m(reporting_node_position_m, entity_position_m)
    if distance_m > config.absorption_max_node_distance_m:
        return None
    return distance_m / config.absorption_max_node_distance_m

def _bearing_mismatch(
    entity: BirdEntityRecord, detection_position_m: Vec3, config: EntityReportingConfig
) -> float | None:
    """A fix against a direction-known entity: inside its fan (with slack)?"""
    bearing = entity.bearing
    if bearing is None:
        return None
    difference_deg = angular_difference_deg(
        compass_bearing_deg(bearing.origin_position_m, detection_position_m), bearing.bearing_deg
    )
    allowed_deg = bearing.half_width_deg + config.absorption_max_bearing_difference_deg
    if difference_deg > allowed_deg:
        return None
    return difference_deg / allowed_deg
