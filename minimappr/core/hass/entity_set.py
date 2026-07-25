"""The desired Home Assistant entity set for the current zones and nodes.

Split out of ``state_mapper`` because it answers a different question: the mapper
says *what state to publish*, this says *which entities should exist at all*. The
reconciler diffs the result against the ledger to decide what to create, update,
and delete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from minimappr.core.hass import discovery as disc
from minimappr.core.hass.models import HassBridgeConfig
from minimappr.core.hass.topics import HassTopics

if TYPE_CHECKING:
    # Type-only: state_mapper imports this module, so the reverse must stay lazy.
    from minimappr.core.hass.state_mapper import NodeStateInput, ZoneStateInput


@dataclass(frozen=True, slots=True)
class DiscoveryEntity:
    """One desired HA entity: its config topic, payload, and owned state topics.

    ``state_topics`` exists so removal can also blank the entity's state and
    attribute topics — otherwise the broker keeps serving a zombie retained value
    for an entity HA no longer knows about.
    """
    config_topic: str
    payload: dict[str, Any]
    state_topics: tuple[str, ...]


def desired_discovery(
    *,
    config: HassBridgeConfig,
    topics: HassTopics,
    device: dict[str, Any],
    detection_class_names: tuple[str, ...],
    zones: tuple[ZoneStateInput, ...],
    nodes: tuple[NodeStateInput, ...],
) -> dict[str, DiscoveryEntity]:
    """The full set of entities that should exist right now, by config topic.

    Every ``publish_*`` toggle gates exactly its own entities, so turning one
    off makes the corresponding entities *absent from the desired set* —
    which the reconciler then removes from HA rather than leaving stale.
    """
    entities: dict[str, DiscoveryEntity] = {}

    def add(component: str, object_id: str, payload: dict[str, Any], *state_topics: str) -> None:
        topic = topics.discovery(component, object_id)
        entities[topic] = DiscoveryEntity(
            config_topic=topic, payload=payload, state_topics=tuple(state_topics)
        )

    for zone in zones:
        if config.publish_zone_occupancy:
            add(
                disc.COMPONENT_BINARY_SENSOR,
                disc.zone_occupancy_object_id(zone.zone_id),
                disc.zone_occupancy_config(
                    topics,
                    zone_id=zone.zone_id,
                    zone_name=zone.zone_name,
                    zone_type=zone.zone_type,
                    device=device,
                ),
                topics.zone_occupancy(zone.zone_id),
                topics.zone_occupancy_attributes(zone.zone_id),
            )
        if config.publish_zone_spl:
            add(
                disc.COMPONENT_SENSOR,
                disc.zone_spl_object_id(zone.zone_id),
                disc.zone_spl_config(
                    topics,
                    zone_id=zone.zone_id,
                    zone_name=zone.zone_name,
                    device=device,
                ),
                topics.zone_spl(zone.zone_id),
            )

    if config.publish_detection_classes:
        for name in detection_class_names:
            add(
                disc.COMPONENT_BINARY_SENSOR,
                disc.detection_class_object_id(name),
                disc.detection_class_config(
                    topics,
                    label=name,
                    device=device,
                    off_delay_seconds=config.detection_off_delay_seconds,
                ),
                topics.detection_class(name),
            )

    if config.publish_node_status:
        for node in nodes:
            add(
                disc.COMPONENT_BINARY_SENSOR,
                disc.node_connectivity_object_id(node.node_id),
                disc.node_connectivity_config(
                    topics,
                    node_id=node.node_id,
                    node_name=node.node_name,
                    device=device,
                ),
                topics.node_connectivity(node.node_id),
                topics.node_attributes(node.node_id),
            )

    if config.publish_system_health:
        add(
            disc.COMPONENT_SENSOR,
            "system_health",
            disc.system_health_config(topics, device=device),
            topics.system_health,
            topics.system_health_attributes,
        )
        add(
            disc.COMPONENT_SENSOR,
            "active_track_count",
            disc.active_track_count_config(topics, device=device),
            topics.active_track_count,
        )

    if config.publish_events:
        add(
            disc.COMPONENT_EVENT,
            "detection",
            disc.detection_event_config(
                topics, device=device, event_types=disc.detection_event_types()
            ),
            topics.detection_event,
        )
        add(
            disc.COMPONENT_EVENT,
            "alert",
            disc.alert_event_config(
                topics, device=device, event_types=list(disc.ALERT_EVENT_TYPES)
            ),
            topics.alert_event,
        )

    if config.publish_track_slots:
        for slot_index in range(config.track_slot_count):
            add(
                disc.COMPONENT_DEVICE_TRACKER,
                disc.track_slot_object_id(slot_index),
                disc.track_slot_config(
                    topics, slot_index=slot_index, device=device
                ),
                topics.track_slot(slot_index),
                topics.track_slot_attributes(slot_index),
            )

    return entities
