"""Pure Home Assistant MQTT Discovery payload builders.

Every function here is a pure ``(topics, config, entity identity) -> dict``. No
I/O, no clock, no randomness — so ``tests/test_hass_discovery_payloads.py`` can
golden-compare the full payload set and the spec lint can walk it.

Discovery style is the classic per-component form,
``<discovery_prefix>/<component>/<device_id>/<object_id>/config``, with full
(non-abbreviated) key names. Abbreviations (``t``, ``stat_t``, ``dev_cla``) are
valid but make the golden fixtures unreadable, and the wire saving is
irrelevant for a payload published once per entity per reconcile.

Retain policy for the *state* of these entities lives in ``state_mapper``; the
config payloads themselves are always published retained (that is what makes HA
rediscover the device after a restart without us republishing).
"""

from __future__ import annotations

from typing import Any

from minimappr.core.hass.topics import HassTopics, slugify

# HA component names we publish. Kept explicit so the spec lint can assert that
# every payload's device_class is legal for its component.
COMPONENT_BINARY_SENSOR = "binary_sensor"
COMPONENT_SENSOR = "sensor"
COMPONENT_EVENT = "event"
COMPONENT_DEVICE_TRACKER = "device_tracker"

PAYLOAD_ON = "ON"
PAYLOAD_OFF = "OFF"
PAYLOAD_AVAILABLE = "online"
PAYLOAD_NOT_AVAILABLE = "offline"

# HA's MQTT platforms treat the literal string "None" as "no value" and set the
# entity state to unknown (``homeassistant.const.PAYLOAD_NONE``). Used for a
# zone with no SPL sample in its window: 0.0 dB would be a measurement we never
# took. Publishing the word "unknown" instead would fail numeric parsing on a
# sensor carrying ``state_class: measurement``.
PAYLOAD_UNKNOWN = "None"

# Zone kind -> HA binary_sensor device_class. An alert zone is a safety concern
# and an exclusion zone is a rule violation ("problem"); coverage/interest zones
# are plain occupancy.
_ZONE_TYPE_DEVICE_CLASS: dict[str, str] = {
    "alert_zone": "safety",
    "exclusion_zone": "problem",
}
_ZONE_DEVICE_CLASS_DEFAULT = "occupancy"

# Detection label -> HA binary_sensor device_class. Only labels with a genuinely
# better-fitting class are listed; everything else falls through to "sound",
# which is what a microphone-derived detection actually is.
_DETECTION_DEVICE_CLASS: dict[str, str] = {
    "gunshot": "safety",
    "glass_break": "safety",
    "scream": "safety",
    "explosion": "safety",
    "smoke_alarm": "problem",
    "fire_alarm": "problem",
    "speech": "sound",
    "vehicle": "motion",
    "drone": "motion",
}
_DETECTION_DEVICE_CLASS_DEFAULT = "sound"

# device_class values HA accepts per component, for the spec lint. Not
# exhaustive for HA — exhaustive for what we emit.
ALLOWED_DEVICE_CLASSES: dict[str, frozenset[str]] = {
    COMPONENT_BINARY_SENSOR: frozenset(
        {"safety", "problem", "occupancy", "sound", "motion", "connectivity"}
    ),
    COMPONENT_SENSOR: frozenset({"sound_pressure", "enum", "signal_strength"}),
    COMPONENT_EVENT: frozenset({"doorbell", "button", "motion"}),
    COMPONENT_DEVICE_TRACKER: frozenset(),
}


def zone_device_class(zone_type: str) -> str:
    return _ZONE_TYPE_DEVICE_CLASS.get(str(zone_type).strip().lower(), _ZONE_DEVICE_CLASS_DEFAULT)


def detection_device_class(label: str) -> str:
    return _DETECTION_DEVICE_CLASS.get(
        str(label).strip().lower(), _DETECTION_DEVICE_CLASS_DEFAULT
    )


def device_block(*, device_id: str, device_name: str, version: str) -> dict[str, Any]:
    """The shared HA device all our entities attach to.

    One device for the whole site keeps the HA UI navigable: zones, nodes,
    system diagnostics and events group under a single card instead of
    scattering dozens of orphan entities.
    """
    return {
        "identifiers": [f"minimappr_{device_id}"],
        "name": device_name,
        "manufacturer": "MinimapPR",
        "model": "Acoustic Localization COP",
        "sw_version": version,
    }


def _availability(topics: HassTopics) -> dict[str, Any]:
    return {
        "availability_topic": topics.availability,
        "payload_available": PAYLOAD_AVAILABLE,
        "payload_not_available": PAYLOAD_NOT_AVAILABLE,
    }


def _base(
    topics: HassTopics,
    *,
    object_id: str,
    name: str,
    device: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "object_id": f"{topics.device_id}_{object_id}",
        "unique_id": topics.unique_id(object_id),
        "device": device,
        **_availability(topics),
    }


# -- zones -----------------------------------------------------------------


def zone_occupancy_object_id(zone_id: str) -> str:
    return f"zone_occupancy_{slugify(zone_id)}"


def zone_spl_object_id(zone_id: str) -> str:
    return f"zone_spl_{slugify(zone_id)}"


def zone_occupancy_config(
    topics: HassTopics,
    *,
    zone_id: str,
    zone_name: str,
    zone_type: str,
    device: dict[str, Any],
) -> dict[str, Any]:
    object_id = zone_occupancy_object_id(zone_id)
    payload = _base(topics, object_id=object_id, name=f"{zone_name} occupancy", device=device)
    payload.update(
        {
            "state_topic": topics.zone_occupancy(zone_id),
            "json_attributes_topic": topics.zone_occupancy_attributes(zone_id),
            "payload_on": PAYLOAD_ON,
            "payload_off": PAYLOAD_OFF,
            "device_class": zone_device_class(zone_type),
        }
    )
    return payload


def zone_spl_config(
    topics: HassTopics,
    *,
    zone_id: str,
    zone_name: str,
    device: dict[str, Any],
) -> dict[str, Any]:
    object_id = zone_spl_object_id(zone_id)
    payload = _base(topics, object_id=object_id, name=f"{zone_name} sound level", device=device)
    payload.update(
        {
            "state_topic": topics.zone_spl(zone_id),
            "device_class": "sound_pressure",
            "unit_of_measurement": "dB",
            "state_class": "measurement",
            "suggested_display_precision": 1,
        }
    )
    return payload


# -- detection classes ------------------------------------------------------


def detection_class_object_id(label: str) -> str:
    return f"detection_{slugify(label)}"


def detection_class_config(
    topics: HassTopics,
    *,
    label: str,
    device: dict[str, Any],
    off_delay_seconds: int,
) -> dict[str, Any]:
    """An impulse binary_sensor: goes ON when heard, auto-OFF after ``off_delay``.

    ``off_delay`` is why the *state* of this entity is never retained — a
    retained ON would restore on HA restart and then auto-off, firing every
    automation attached to it for a sound that happened hours ago.
    """
    object_id = detection_class_object_id(label)
    payload = _base(
        topics, object_id=object_id, name=f"{label.replace('_', ' ').title()} detected", device=device
    )
    payload.update(
        {
            "state_topic": topics.detection_class(label),
            "payload_on": PAYLOAD_ON,
            "payload_off": PAYLOAD_OFF,
            "device_class": detection_device_class(label),
            "off_delay": int(off_delay_seconds),
        }
    )
    return payload


# -- nodes -------------------------------------------------------------------


def node_connectivity_object_id(node_id: str) -> str:
    return f"node_online_{slugify(node_id)}"


def node_connectivity_config(
    topics: HassTopics,
    *,
    node_id: str,
    node_name: str,
    device: dict[str, Any],
) -> dict[str, Any]:
    object_id = node_connectivity_object_id(node_id)
    payload = _base(topics, object_id=object_id, name=f"Node {node_name}", device=device)
    payload.update(
        {
            "state_topic": topics.node_connectivity(node_id),
            "json_attributes_topic": topics.node_attributes(node_id),
            "payload_on": PAYLOAD_ON,
            "payload_off": PAYLOAD_OFF,
            "device_class": "connectivity",
            "entity_category": "diagnostic",
        }
    )
    return payload


# -- system ------------------------------------------------------------------


def system_health_config(topics: HassTopics, *, device: dict[str, Any]) -> dict[str, Any]:
    payload = _base(topics, object_id="system_health", name="System health", device=device)
    payload.update(
        {
            "state_topic": topics.system_health,
            "json_attributes_topic": topics.system_health_attributes,
            "device_class": "enum",
            # Mirrors the vocabulary GET /api/v1/context/current already returns,
            # so the HA state and the COP agree word-for-word.
            "options": ["ok", "degraded", "error"],
            "entity_category": "diagnostic",
        }
    )
    return payload


def active_track_count_config(topics: HassTopics, *, device: dict[str, Any]) -> dict[str, Any]:
    payload = _base(topics, object_id="active_track_count", name="Active tracks", device=device)
    payload.update(
        {
            "state_topic": topics.active_track_count,
            "state_class": "measurement",
            "unit_of_measurement": "tracks",
            "entity_category": "diagnostic",
        }
    )
    return payload


# -- track slots -------------------------------------------------------------


def track_slot_object_id(slot_index: int) -> str:
    return f"track_slot_{slot_index:02d}"


def track_slot_config(
    topics: HassTopics,
    *,
    slot_index: int,
    device: dict[str, Any],
) -> dict[str, Any]:
    """A fixed device_tracker slot.

    HA's entity registry persists every ``unique_id`` forever, so mapping
    ephemeral track ids to entities would grow the registry without bound and
    leave thousands of orphans. A fixed pool of slots is bounded and
    deterministic; the allocator decides which track occupies which slot.
    """
    object_id = track_slot_object_id(slot_index)
    payload = _base(topics, object_id=object_id, name=f"Track slot {slot_index:02d}", device=device)
    payload.update(
        {
            "state_topic": topics.track_slot(slot_index),
            "json_attributes_topic": topics.track_slot_attributes(slot_index),
            "payload_home": "home",
            "payload_not_home": "not_home",
            "source_type": "gps",
        }
    )
    return payload


# -- impulse events ----------------------------------------------------------


def detection_event_config(
    topics: HassTopics,
    *,
    device: dict[str, Any],
    event_types: list[str],
) -> dict[str, Any]:
    """HA drops an ``event_type`` that was not declared in ``event_types``,
    so the caller must pass the closed taxonomy category set."""
    payload = _base(topics, object_id="detection", name="Detection", device=device)
    payload.update(
        {
            "state_topic": topics.detection_event,
            "event_types": list(event_types),
        }
    )
    return payload


def alert_event_config(
    topics: HassTopics,
    *,
    device: dict[str, Any],
    event_types: list[str],
) -> dict[str, Any]:
    payload = _base(topics, object_id="alert", name="Alert", device=device)
    payload.update(
        {
            "state_topic": topics.alert_event,
            "event_types": list(event_types),
        }
    )
    return payload

ALERT_EVENT_TYPES: tuple[str, ...] = ("low", "normal", "high", "critical")


def detection_event_types() -> list[str]:
    """The closed set of taxonomy categories, sorted for a stable payload.

    HA discards an incoming ``event_type`` that was not declared in the
    entity's ``event_types``, so the declaration and the clamp must draw on the
    same source.
    """
    from minimappr.core.taxonomy import DEFAULT_CATEGORY_TO_IFF

    return sorted(DEFAULT_CATEGORY_TO_IFF)
