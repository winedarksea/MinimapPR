"""MinimapPR state -> ``list[MqttPublish]``.

This module owns the **retain decision** for every topic. The bridge never sets
``retain``; it only ships what it is handed. That keeps retain policy in one
golden-testable place, which matters because getting it wrong is not a cosmetic
bug: a retained ``event`` re-fires every automation on HA restart, and a retained
ON for an ``off_delay`` sensor restores-then-auto-offs into a phantom trigger.

Retained: availability, discovery configs, and all *stateful* entities (zone
occupancy/SPL, node connectivity, system health, track slots) so an HA restart
restores last-known state without waiting for our next poll.

Not retained: ``event`` entities and the impulse detection binary_sensors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from minimappr.core.hass import discovery as disc
from minimappr.core.hass.entity_set import DiscoveryEntity, desired_discovery
from minimappr.core.hass.models import HassBridgeConfig
from minimappr.core.hass.spl_aggregator import ZoneSplAggregator
from minimappr.core.hass.topics import HassTopics, slugify
from minimappr.core.hass.track_slots import TrackSlotAllocator, TrackSlotCandidate
from minimappr.core.hass.transport import MqttPublish

# Statuses that count as "connected" for the connectivity binary_sensor.
# Connectivity is binary; a degraded or BIT-failing node is still *reachable*,
# so it reports ON with the detail in attributes rather than vanishing from HA
# as if it were unplugged.
_OFFLINE_HEALTH_STATUSES = frozenset({"offline", "unknown", ""})


@dataclass(frozen=True, slots=True)
class ZoneStateInput:
    zone_id: str
    zone_name: str
    zone_type: str
    occupied: bool = False
    occupying_track_ids: tuple[str, ...] = ()
    occupying_labels: tuple[str, ...] = ()
    updated_ns: int = 0


@dataclass(frozen=True, slots=True)
class NodeStateInput:
    node_id: str
    node_name: str
    health_status: str
    last_seen_ns: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SystemStateInput:
    system_health: str = "unknown"
    active_track_count: int = 0
    online_nodes: int = 0
    degraded_nodes: int = 0
    offline_nodes: int = 0
    generated_ns: int = 0


@dataclass(frozen=True, slots=True)
class HassStateSnapshot:
    """Everything one publish cycle needs, gathered by the bridge's poll."""
    zones: tuple[ZoneStateInput, ...] = ()
    nodes: tuple[NodeStateInput, ...] = ()
    system: SystemStateInput = SystemStateInput()
    tracks: tuple[TrackSlotCandidate, ...] = ()


def _json(payload: dict[str, Any]) -> str:
    # Sorted keys so an unchanged payload serializes byte-identically and the
    # bridge's dedupe cache can suppress it.
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class HassStateMapper:
    def __init__(self, config: HassBridgeConfig) -> None:
        self._config = config
        self._topics = HassTopics(
            discovery_prefix=config.discovery_prefix,
            base_topic=config.base_topic,
            device_id=config.device_id,
        )
        self._device = disc.device_block(
            device_id=self._topics.device_id,
            device_name=config.device_name,
            version=config.version,
        )
        self._slots = TrackSlotAllocator(config.track_slot_count)
        self._spl = ZoneSplAggregator(config.zone_spl_window_seconds)
        # Slug -> configured name, so a detection's label or category can be
        # matched against the configured set without re-slugifying per event.
        self._detection_class_slugs = {
            slugify(name): name for name in config.detection_classes if str(name).strip()
        }

    @property
    def topics(self) -> HassTopics:
        return self._topics

    @property
    def spl(self) -> ZoneSplAggregator:
        return self._spl

    @property
    def slots(self) -> TrackSlotAllocator:
        return self._slots

    # -- availability -------------------------------------------------------

    def availability(self, *, online: bool) -> MqttPublish:
        return MqttPublish(
            topic=self._topics.availability,
            payload=disc.PAYLOAD_AVAILABLE if online else disc.PAYLOAD_NOT_AVAILABLE,
            retain=True,
        )

    # -- discovery ----------------------------------------------------------

    def desired_discovery(
        self,
        *,
        zones: tuple[ZoneStateInput, ...],
        nodes: tuple[NodeStateInput, ...],
    ) -> dict[str, DiscoveryEntity]:
        """The entity set that should exist right now — see ``entity_set.py``."""
        return desired_discovery(
            config=self._config,
            topics=self._topics,
            device=self._device,
            detection_class_names=tuple(self._detection_class_slugs.values()),
            zones=zones,
            nodes=nodes,
        )

    def discovery_publish(self, entity: DiscoveryEntity) -> MqttPublish:
        return MqttPublish(topic=entity.config_topic, payload=_json(entity.payload), retain=True)

    def removal_publishes(self, *, config_topic: str, state_topics: tuple[str, ...]) -> list[MqttPublish]:
        """Delete an entity from HA and clear what the broker retained for it.

        An empty retained payload to the config topic is HA's documented delete.
        The state/attribute topics need the same treatment or the broker keeps
        replaying a value for an entity that no longer exists.
        """
        messages = [MqttPublish(topic=config_topic, payload="", retain=True)]
        messages.extend(MqttPublish(topic=topic, payload="", retain=True) for topic in state_topics)
        return messages

    # -- stateful snapshot --------------------------------------------------

    def snapshot_publishes(self, snapshot: HassStateSnapshot, *, now_ns: int) -> list[MqttPublish]:
        messages: list[MqttPublish] = []
        self._spl.prune(now_ns=now_ns)

        for zone in snapshot.zones:
            if self._config.publish_zone_occupancy:
                messages.append(
                    MqttPublish(
                        topic=self._topics.zone_occupancy(zone.zone_id),
                        payload=disc.PAYLOAD_ON if zone.occupied else disc.PAYLOAD_OFF,
                        retain=True,
                    )
                )
                messages.append(
                    MqttPublish(
                        topic=self._topics.zone_occupancy_attributes(zone.zone_id),
                        payload=_json(
                            {
                                "zone_id": zone.zone_id,
                                "zone_name": zone.zone_name,
                                "zone_type": zone.zone_type,
                                "track_count": len(zone.occupying_track_ids),
                                "track_ids": list(zone.occupying_track_ids),
                                "labels": list(zone.occupying_labels),
                                "updated_ns": zone.updated_ns,
                            }
                        ),
                        retain=True,
                    )
                )
            if self._config.publish_zone_spl:
                spl = self._spl.max_for_zone(zone.zone_id, now_ns=now_ns)
                messages.append(
                    MqttPublish(
                        topic=self._topics.zone_spl(zone.zone_id),
                        payload=disc.PAYLOAD_UNKNOWN if spl is None else f"{spl:.1f}",
                        retain=True,
                    )
                )

        if self._config.publish_node_status:
            for node in snapshot.nodes:
                health = str(node.health_status or "").strip().lower()
                online = health not in _OFFLINE_HEALTH_STATUSES
                messages.append(
                    MqttPublish(
                        topic=self._topics.node_connectivity(node.node_id),
                        payload=disc.PAYLOAD_ON if online else disc.PAYLOAD_OFF,
                        retain=True,
                    )
                )
                messages.append(
                    MqttPublish(
                        topic=self._topics.node_attributes(node.node_id),
                        payload=_json(
                            {
                                "node_id": node.node_id,
                                "node_name": node.node_name,
                                "health_status": health,
                                "degraded": online and health != "online",
                                "last_seen_ns": node.last_seen_ns,
                                **node.detail,
                            }
                        ),
                        retain=True,
                    )
                )

        if self._config.publish_system_health:
            system = snapshot.system
            messages.append(
                MqttPublish(
                    topic=self._topics.system_health,
                    payload=_system_health_state(system.system_health),
                    retain=True,
                )
            )
            messages.append(
                MqttPublish(
                    topic=self._topics.system_health_attributes,
                    payload=_json(
                        {
                            "raw_health": system.system_health,
                            "online_nodes": system.online_nodes,
                            "degraded_nodes": system.degraded_nodes,
                            "offline_nodes": system.offline_nodes,
                            "active_tracks": system.active_track_count,
                            "generated_ns": system.generated_ns,
                        }
                    ),
                    retain=True,
                )
            )
            messages.append(
                MqttPublish(
                    topic=self._topics.active_track_count,
                    payload=str(int(system.active_track_count)),
                    retain=True,
                )
            )

        if self._config.publish_track_slots:
            messages.extend(self._track_slot_publishes(snapshot.tracks))

        return messages

    def _track_slot_publishes(self, tracks: tuple[TrackSlotCandidate, ...]) -> list[MqttPublish]:
        messages: list[MqttPublish] = []
        for assignment in self._slots.assign(list(tracks)):
            candidate = assignment.candidate
            if candidate is None:
                messages.append(
                    MqttPublish(
                        topic=self._topics.track_slot(assignment.slot_index),
                        payload="not_home",
                        retain=True,
                    )
                )
                messages.append(
                    MqttPublish(
                        topic=self._topics.track_slot_attributes(assignment.slot_index),
                        payload=_json({"track_id": None, "source_type": "gps"}),
                        retain=True,
                    )
                )
                continue

            attributes: dict[str, Any] = {
                # HA ignores latitude/longitude unless source_type says gps.
                "source_type": "gps",
                "track_id": candidate.track_id,
                "label": candidate.label,
                "status": candidate.status,
                "tqi": round(candidate.tqi, 4),
            }
            if candidate.confidence is not None:
                attributes["confidence"] = round(candidate.confidence, 4)
            if candidate.lat is not None and candidate.lon is not None:
                attributes["latitude"] = candidate.lat
                attributes["longitude"] = candidate.lon
            if candidate.altitude_m is not None:
                attributes["altitude"] = candidate.altitude_m
            # "home" is meaningless for an acoustic track; the useful signal is
            # lat/lon in the attributes, so an occupied slot always reads home.
            messages.append(
                MqttPublish(
                    topic=self._topics.track_slot(assignment.slot_index),
                    payload="home",
                    retain=True,
                )
            )
            messages.append(
                MqttPublish(
                    topic=self._topics.track_slot_attributes(assignment.slot_index),
                    payload=_json(attributes),
                    retain=True,
                )
            )
        return messages

    # -- impulses -----------------------------------------------------------

    def observe_detection(self, detection: dict[str, Any], *, now_ns: int) -> list[MqttPublish]:
        """Feed one detection event into the SPL window and emit its impulses.

        ``coalescable=False`` throughout: two identical gunshots one second apart
        are two events, and collapsing them would lose the second one.
        """
        zone_ids = [str(item) for item in detection.get("zone_ids") or []]
        timestamp_ns = _as_int(detection.get("timestamp_ns"), now_ns)
        self._spl.observe(
            zone_ids=zone_ids, spl_db=detection.get("spl_db"), timestamp_ns=timestamp_ns
        )

        messages: list[MqttPublish] = []
        label = str(detection.get("label") or "")
        category = str(detection.get("label_category") or "unknown")

        if self._config.publish_detection_classes:
            for matched in self._matched_detection_classes(label, category):
                messages.append(
                    MqttPublish(
                        topic=self._topics.detection_class(matched),
                        payload=disc.PAYLOAD_ON,
                        retain=False,
                        coalescable=False,
                    )
                )

        if self._config.publish_events:
            payload = {
                # HA discards an event_type absent from the declared event_types,
                # so this is clamped to the closed taxonomy category set.
                "event_type": category if category in disc.detection_event_types() else "unknown",
                "detection_id": detection.get("id"),
                "label": label,
                "label_category": category,
                "confidence": detection.get("label_confidence"),
                "zone_ids": zone_ids,
                "timestamp_ns": timestamp_ns,
            }
            spl_db = detection.get("spl_db")
            if spl_db is not None:
                payload["spl_db"] = spl_db
            position = detection.get("position_geo")
            if isinstance(position, dict):
                payload["latitude"] = position.get("lat")
                payload["longitude"] = position.get("lon")
            messages.append(
                MqttPublish(
                    topic=self._topics.detection_event,
                    payload=_json(payload),
                    retain=False,
                    coalescable=False,
                )
            )
        return messages

    def _matched_detection_classes(self, label: str, category: str) -> list[str]:
        """Configured names matching this detection, by label first then category.

        Matching both means the shipped default (taxonomy categories) yields
        working sensors out of the box while an operator can add a specific
        label like ``gunshot`` for a finer-grained sensor.
        """
        matched: list[str] = []
        for candidate in (label, category):
            name = self._detection_class_slugs.get(slugify(candidate)) if candidate else None
            if name is not None and name not in matched:
                matched.append(name)
        return matched

    def alert_publish(self, alert: dict[str, Any]) -> list[MqttPublish]:
        if not self._config.publish_events:
            return []
        priority = str(alert.get("priority") or "normal").strip().lower()
        payload = {
            "event_type": priority if priority in disc.ALERT_EVENT_TYPES else "normal",
            "alert_id": alert.get("alert_id"),
            "rule_id": alert.get("rule_id"),
            "priority": priority,
            "destination": alert.get("destination"),
            "detection_id": alert.get("detection_id"),
            "track_id": alert.get("track_id"),
            "timestamp_ns": alert.get("timestamp_ns"),
        }
        return [
            MqttPublish(
                topic=self._topics.alert_event,
                payload=_json({key: value for key, value in payload.items() if value is not None}),
                retain=False,
                coalescable=False,
            )
        ]


SYSTEM_HEALTH_OPTIONS: frozenset[str] = frozenset({"ok", "degraded", "error"})


def _system_health_state(raw: str) -> str:
    """Clamp to the enum sensor's declared ``options``; unknown stays unknown.

    HA rejects an enum state outside the declared options, so an unrecognized
    value must be normalized — but normalizing it to ``degraded`` would invent a
    diagnosis, so it publishes HA's unknown sentinel instead.
    """
    value = str(raw or "").strip().lower()
    return value if value in SYSTEM_HEALTH_OPTIONS else disc.PAYLOAD_UNKNOWN


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
