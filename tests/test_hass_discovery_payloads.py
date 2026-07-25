"""Golden + spec-lint coverage for Home Assistant MQTT Discovery payloads.

**No live Home Assistant instance was available when this integration was
built.** These fixtures and the ``_assert_valid_ha_discovery`` lint below *are*
the spec-validation artifact in its place. They were hand-checked on 2026-07-25
against the Home Assistant MQTT Discovery documentation, specifically:

* "MQTT Discovery" — discovery topic form
  ``<discovery_prefix>/<component>/[<node_id>/]<object_id>/config``, retained
  config payloads, and deletion by publishing an empty retained payload.
* "MQTT Binary sensor" — ``payload_on``/``payload_off``, ``off_delay``,
  ``device_class`` value list, ``entity_category``.
* "MQTT Sensor" — ``device_class: sound_pressure`` with ``unit_of_measurement``,
  ``state_class: measurement``, ``suggested_display_precision``; and
  ``device_class: enum`` requiring an ``options`` list and forbidding
  ``state_class``/``unit_of_measurement``.
* "MQTT Event" — ``event_types`` must be non-empty and an incoming
  ``event_type`` outside that list is discarded by HA.
* "MQTT Device tracker" — ``source_type: gps`` is required for HA to accept
  ``latitude``/``longitude`` from the attributes topic.
* "MQTT Availability" — ``availability_topic`` with
  ``payload_available``/``payload_not_available``, paired with a broker LWT.

A manual smoke test against a real HA instance is the first follow-up for this
subsystem; until it happens, do not describe these payloads as spec-verified.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minimappr.core.hass import discovery as disc
from minimappr.core.hass.topics import HassTopics, is_valid_topic_level, slugify

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "hass_discovery"

DEVICE = disc.device_block(device_id="minimappr", device_name="MinimapPR", version="test")
EVENT_TYPES = ["human", "security", "unknown", "vehicle", "wildlife"]


@pytest.fixture()
def topics() -> HassTopics:
    return HassTopics(discovery_prefix="homeassistant", base_topic="minimappr", device_id="minimappr")


def _assert_golden(name: str, payload: dict) -> None:
    path = FIXTURE_DIR / f"{name}.json"
    assert path.exists(), f"missing golden fixture {path}"
    expected = json.loads(path.read_text())
    assert payload == expected, f"{name} drifted from its golden fixture"


# -- golden comparisons -----------------------------------------------------


def test_device_block_golden() -> None:
    _assert_golden("device_block", DEVICE)


def test_zone_occupancy_golden(topics: HassTopics) -> None:
    payload = disc.zone_occupancy_config(
        topics, zone_id="front_lawn", zone_name="Front Lawn", zone_type="alert_zone", device=DEVICE
    )
    _assert_golden("zone_occupancy", payload)


def test_zone_spl_golden(topics: HassTopics) -> None:
    payload = disc.zone_spl_config(
        topics, zone_id="front_lawn", zone_name="Front Lawn", device=DEVICE
    )
    _assert_golden("zone_spl", payload)


def test_detection_class_golden(topics: HassTopics) -> None:
    payload = disc.detection_class_config(
        topics, label="gunshot", device=DEVICE, off_delay_seconds=30
    )
    _assert_golden("detection_class", payload)


def test_node_connectivity_golden(topics: HassTopics) -> None:
    payload = disc.node_connectivity_config(
        topics, node_id="node-a", node_name="node-a", device=DEVICE
    )
    _assert_golden("node_connectivity", payload)


def test_system_health_golden(topics: HassTopics) -> None:
    _assert_golden("system_health", disc.system_health_config(topics, device=DEVICE))


def test_active_track_count_golden(topics: HassTopics) -> None:
    _assert_golden("active_track_count", disc.active_track_count_config(topics, device=DEVICE))


def test_track_slot_golden(topics: HassTopics) -> None:
    _assert_golden("track_slot", disc.track_slot_config(topics, slot_index=0, device=DEVICE))


def test_detection_event_golden(topics: HassTopics) -> None:
    payload = disc.detection_event_config(topics, device=DEVICE, event_types=EVENT_TYPES)
    _assert_golden("detection_event", payload)


def test_alert_event_golden(topics: HassTopics) -> None:
    payload = disc.alert_event_config(
        topics, device=DEVICE, event_types=["high", "normal", "low"]
    )
    _assert_golden("alert_event", payload)


# -- spec lint --------------------------------------------------------------


def _desired_entity_set(topics: HassTopics) -> list[tuple[str, dict]]:
    """One of every entity kind we can publish, as (component, payload)."""
    entities: list[tuple[str, dict]] = [
        (disc.COMPONENT_BINARY_SENSOR, disc.zone_occupancy_config(
            topics, zone_id="z1", zone_name="Z1", zone_type="alert_zone", device=DEVICE)),
        (disc.COMPONENT_BINARY_SENSOR, disc.zone_occupancy_config(
            topics, zone_id="z2", zone_name="Z2", zone_type="coverage_zone", device=DEVICE)),
        (disc.COMPONENT_SENSOR, disc.zone_spl_config(
            topics, zone_id="z1", zone_name="Z1", device=DEVICE)),
        (disc.COMPONENT_BINARY_SENSOR, disc.detection_class_config(
            topics, label="gunshot", device=DEVICE, off_delay_seconds=30)),
        (disc.COMPONENT_BINARY_SENSOR, disc.detection_class_config(
            topics, label="speech", device=DEVICE, off_delay_seconds=30)),
        (disc.COMPONENT_BINARY_SENSOR, disc.node_connectivity_config(
            topics, node_id="n1", node_name="n1", device=DEVICE)),
        (disc.COMPONENT_SENSOR, disc.system_health_config(topics, device=DEVICE)),
        (disc.COMPONENT_SENSOR, disc.active_track_count_config(topics, device=DEVICE)),
        (disc.COMPONENT_EVENT, disc.detection_event_config(
            topics, device=DEVICE, event_types=EVENT_TYPES)),
        (disc.COMPONENT_EVENT, disc.alert_event_config(
            topics, device=DEVICE, event_types=["high", "normal"])),
    ]
    entities.extend(
        (disc.COMPONENT_DEVICE_TRACKER, disc.track_slot_config(topics, slot_index=i, device=DEVICE))
        for i in range(4)
    )
    return entities


def _assert_valid_ha_discovery(component: str, payload: dict) -> None:
    for required in ("name", "unique_id", "state_topic", "device", "availability_topic"):
        assert required in payload, f"{component} payload missing required key {required}"
    assert payload["payload_available"] == "online"
    assert payload["payload_not_available"] == "offline"
    assert payload["device"]["identifiers"], "device block needs identifiers for HA grouping"

    device_class = payload.get("device_class")
    if device_class is not None:
        allowed = disc.ALLOWED_DEVICE_CLASSES[component]
        assert device_class in allowed, f"{device_class} is not a valid {component} device_class"

    if component == disc.COMPONENT_EVENT:
        assert payload.get("event_types"), "event entities require a non-empty event_types list"

    if component == disc.COMPONENT_DEVICE_TRACKER:
        # Without source_type=gps HA ignores latitude/longitude attributes.
        assert payload.get("source_type") == "gps"

    if device_class == "enum":
        assert payload.get("options"), "enum sensors require an options list"
        assert "state_class" not in payload, "HA rejects state_class on an enum sensor"
        assert "unit_of_measurement" not in payload

    if component == disc.COMPONENT_SENSOR and payload.get("unit_of_measurement"):
        assert payload.get("device_class") != "enum"


def test_every_entity_kind_passes_the_spec_lint(topics: HassTopics) -> None:
    for component, payload in _desired_entity_set(topics):
        _assert_valid_ha_discovery(component, payload)


def test_unique_ids_are_unique_across_the_desired_set(topics: HassTopics) -> None:
    unique_ids = [payload["unique_id"] for _, payload in _desired_entity_set(topics)]
    assert len(unique_ids) == len(set(unique_ids))


def test_object_ids_are_unique_across_the_desired_set(topics: HassTopics) -> None:
    object_ids = [payload["object_id"] for _, payload in _desired_entity_set(topics)]
    assert len(object_ids) == len(set(object_ids))


def test_discovery_topics_are_unique_and_well_formed(topics: HassTopics) -> None:
    seen: set[str] = set()
    for component, payload in _desired_entity_set(topics):
        # object_id in the payload is namespaced with the device id; the topic
        # segment is the bare object id, recovered by stripping that prefix.
        bare = payload["object_id"].removeprefix(f"{topics.device_id}_")
        topic = topics.discovery(component, bare)
        assert topic.startswith("homeassistant/")
        assert topic.endswith("/config")
        assert "+" not in topic and "#" not in topic
        assert topic not in seen
        seen.add(topic)


def test_off_delay_entities_carry_no_retain_hint(topics: HassTopics) -> None:
    """An ``off_delay`` binary_sensor must not have retained state.

    A retained ON restores on HA restart and then auto-offs, which fires every
    attached automation for a sound that happened hours ago. Retain is decided
    in ``state_mapper``; this asserts the discovery payload does not ask HA to
    treat the topic as retained.
    """
    payload = disc.detection_class_config(topics, label="gunshot", device=DEVICE, off_delay_seconds=30)
    assert payload["off_delay"] == 30
    assert "retain" not in payload


def test_zone_device_class_map() -> None:
    assert disc.zone_device_class("alert_zone") == "safety"
    assert disc.zone_device_class("exclusion_zone") == "problem"
    assert disc.zone_device_class("coverage_zone") == "occupancy"
    assert disc.zone_device_class("interest_zone") == "occupancy"
    assert disc.zone_device_class("ALERT_ZONE") == "safety"


def test_detection_device_class_map() -> None:
    assert disc.detection_device_class("gunshot") == "safety"
    assert disc.detection_device_class("smoke_alarm") == "problem"
    assert disc.detection_device_class("speech") == "sound"
    assert disc.detection_device_class("vehicle") == "motion"
    assert disc.detection_device_class("wren_song") == "sound"


# -- slugify / topics -------------------------------------------------------


def test_slugify_is_deterministic_and_safe() -> None:
    assert slugify("Front Lawn") == "front_lawn"
    assert slugify("zone/with/slashes") == "zone_with_slashes"
    assert slugify("  Trailing--Dashes  ") == "trailing_dashes"
    assert slugify("Ünïcodé") == "n_cod"
    assert slugify("Front Lawn") == slugify("Front Lawn")


def test_slugify_punctuation_only_ids_get_stable_distinct_slugs() -> None:
    a = slugify("///")
    b = slugify("+++")
    assert a != b
    assert a == slugify("///")
    assert is_valid_topic_level(a)


def test_slugify_truncates_with_a_collision_suffix() -> None:
    long_a = "a" * 60 + "_first"
    long_b = "a" * 60 + "_second"
    slug_a, slug_b = slugify(long_a), slugify(long_b)
    assert len(slug_a) <= 48 and len(slug_b) <= 48
    assert slug_a != slug_b, "shared-prefix long ids must not collapse to one entity"
    assert slug_a == slugify(long_a)


def test_is_valid_topic_level() -> None:
    assert is_valid_topic_level("minimappr")
    assert not is_valid_topic_level("")
    assert not is_valid_topic_level("  ")
    assert not is_valid_topic_level("a/b")
    assert not is_valid_topic_level("a+b")
    assert not is_valid_topic_level("a#")


def test_rule_topic_normalizes_into_the_base_namespace(topics: HassTopics) -> None:
    assert topics.rule_topic("rooms/kitchen/occupancy") == "minimappr/rooms/kitchen/occupancy"
    assert topics.rule_topic("/rooms/kitchen") is None
    assert topics.rule_topic("minimappr/alerts/high") == "minimappr/alerts/high"
    assert topics.rule_topic("homeassistant/binary_sensor/x/config") == (
        "minimappr/homeassistant/binary_sensor/x/config"
    )
    assert topics.rule_topic("a/#") is None
    assert topics.rule_topic("a/+/b") is None
    assert topics.rule_topic("") is None
    assert topics.rule_topic("   ") is None
    assert topics.rule_topic("a//b") is None


def test_topic_builders_use_slugs(topics: HassTopics) -> None:
    assert topics.zone_occupancy("Front Lawn") == "minimappr/zone/front_lawn/occupancy"
    assert topics.node_connectivity("node-A") == "minimappr/node/node_a/connectivity"
    assert topics.track_slot(3) == "minimappr/track/03/state"
    assert topics.availability == "minimappr/status"
    assert topics.unique_id("system_health") == "minimappr_minimappr_system_health"
