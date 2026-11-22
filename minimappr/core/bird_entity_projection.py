"""Field Mode W3-entity: registry records -> API/WS models (adds geo + ellipse)."""

from __future__ import annotations

import logging
from typing import Callable

from minimappr.core.bird_entity_model import BirdEntityRecord, Vec3, uncertainty_ellipse
from minimappr.core.bird_entity_registry import BirdEntityChange
from minimappr.models import (
    BirdEntity,
    BirdEntityBearing,
    BirdEntityEvent,
    BirdEntityUncertaintyEllipse,
    GeoPoint,
)

_logger = logging.getLogger(__name__)

LocalToGeo = Callable[[Vec3], GeoPoint]


def _geo_or_none(local_to_geo: LocalToGeo | None, position_m: Vec3 | None) -> GeoPoint | None:
    if local_to_geo is None or position_m is None:
        return None
    try:
        return local_to_geo(position_m)
    except (ValueError, TypeError, ArithmeticError) as exc:
        # A bad origin/frame must not take down the entity feed; the local
        # position still ships.
        _logger.debug("entity geo projection failed: %s", exc)
        return None


def bird_entity_to_model(
    entity: BirdEntityRecord,
    *,
    local_to_geo: LocalToGeo | None,
    sigma_multiplier: float,
) -> BirdEntity:
    bearing = None
    if entity.bearing is not None:
        bearing = BirdEntityBearing(
            origin_node_id=entity.bearing.origin_node_id,
            origin_position_m=entity.bearing.origin_position_m,
            origin_geo=_geo_or_none(local_to_geo, entity.bearing.origin_position_m),
            bearing_deg=entity.bearing.bearing_deg,
            half_width_deg=entity.bearing.half_width_deg,
            range_min_m=entity.bearing.range_min_m,
            range_max_m=entity.bearing.range_max_m,
        )
    ellipse = None
    ellipse_axes = uncertainty_ellipse(entity.position_covariance_m2, sigma_multiplier)
    if ellipse_axes is not None and entity.state == "pinpointed":
        semi_major_m, semi_minor_m, orientation_deg = ellipse_axes
        ellipse = BirdEntityUncertaintyEllipse(
            semi_major_m=semi_major_m,
            semi_minor_m=semi_minor_m,
            orientation_deg=orientation_deg,
            sigma_multiplier=sigma_multiplier,
        )
    return BirdEntity(
        id=entity.entity_id,
        state=entity.state,
        label_category=entity.label_category,
        species_top_label=entity.species_top_label,
        species_top_probability=entity.species_top_probability,
        species_belief=dict(entity.species_belief),
        reporting_node_id=entity.reporting_node_id,
        reporting_node_ids=list(entity.reporting_node_ids),
        # The map is spatial-only: a heard-nearby entity carries no position.
        position_m=entity.position_m if entity.is_spatial else None,
        position_geo=_geo_or_none(local_to_geo, entity.position_m) if entity.is_spatial else None,
        horizontal_uncertainty_m=entity.horizontal_uncertainty_m,
        uncertainty_ellipse=ellipse,
        bearing=bearing,
        track_id=entity.track_id,
        canonical_detection_id=entity.canonical_detection_id,
        first_heard_ns=entity.first_heard_ns,
        last_heard_ns=entity.last_heard_ns,
        last_spatial_update_ns=entity.last_spatial_update_ns,
        last_pulse_ns=entity.last_pulse_ns,
        detection_count=entity.detection_count,
        absorbed_detection_count=entity.absorbed_detection_count,
    )


def bird_entity_event_payload(
    change: BirdEntityChange,
    *,
    local_to_geo: LocalToGeo | None,
    sigma_multiplier: float,
    server_time_ns: int,
) -> dict:
    event = BirdEntityEvent(
        event=change.kind,
        previous_state=change.previous_state,
        entity=bird_entity_to_model(change.entity, local_to_geo=local_to_geo, sigma_multiplier=sigma_multiplier),
        server_time_ns=server_time_ns,
    )
    return event.model_dump(mode="json")
