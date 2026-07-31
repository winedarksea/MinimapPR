"""MinimapPR state -> MQTT message mapping, slot allocation, and SPL windowing.

Pure-function coverage: no bridge, no transport, no event loop.
"""

from __future__ import annotations

import json

import pytest

from minimappr.core.hass.bridge import HassBridgeConfig
from minimappr.core.hass.discovery import detection_event_types
from minimappr.core.hass.spl_aggregator import ZoneSplAggregator
from minimappr.core.hass.state_mapper import (
    HassStateMapper,
    HassStateSnapshot,
    NodeStateInput,
    SystemStateInput,
    ZoneStateInput,
)
from minimappr.core.hass.track_slots import TrackSlotAllocator, TrackSlotCandidate

SECOND_NS = 1_000_000_000
NOW_NS = 1_000 * SECOND_NS


def _config(**overrides) -> HassBridgeConfig:
    base = {
        "enabled": True,
        "mqtt_host": "broker.local",
        "detection_classes": ("security", "gunshot"),
        "track_slot_count": 3,
        "publish_track_slots": True,
        "version": "test",
    }
    base.update(overrides)
    return HassBridgeConfig(**base)


def _mapper(**overrides) -> HassStateMapper:
    return HassStateMapper(_config(**overrides))


def _by_topic(messages) -> dict[str, str]:
    return {message.topic: message.payload for message in messages}


def _snapshot(**overrides) -> HassStateSnapshot:
    defaults = {
        "zones": (
            ZoneStateInput(
                zone_id="front_lawn",
                zone_name="Front Lawn",
                zone_type="alert_zone",
                occupied=True,
                occupying_track_ids=("t1", "t2"),
                occupying_labels=("speech",),
                updated_ns=NOW_NS,
            ),
        ),
        "nodes": (
            NodeStateInput(node_id="node-a", node_name="node-a", health_status="online", last_seen_ns=NOW_NS),
        ),
        "system": SystemStateInput(
            system_health="ok",
            active_track_count=2,
            online_nodes=1,
            generated_ns=NOW_NS,
        ),
        "tracks": (),
    }
    defaults.update(overrides)
    return HassStateSnapshot(**defaults)


# -- availability ------------------------------------------------------------


def test_availability_is_retained_online_and_offline() -> None:
    mapper = _mapper()
    online = mapper.availability(online=True)
    offline = mapper.availability(online=False)
    assert (online.topic, online.payload, online.retain) == ("minimappr/status", "online", True)
    assert (offline.topic, offline.payload, offline.retain) == ("minimappr/status", "offline", True)


# -- zone occupancy ----------------------------------------------------------


def test_zone_occupancy_maps_to_on_off_with_attributes() -> None:
    mapper = _mapper()
    occupied = _by_topic(mapper.snapshot_publishes(_snapshot(), now_ns=NOW_NS))
    assert occupied["minimappr/zone/front_lawn/occupancy"] == "ON"
    attributes = json.loads(occupied["minimappr/zone/front_lawn/attributes"])
    assert attributes["track_ids"] == ["t1", "t2"]
    assert attributes["track_count"] == 2
    assert attributes["labels"] == ["speech"]
    assert attributes["zone_type"] == "alert_zone"

    vacant = _snapshot(
        zones=(ZoneStateInput(zone_id="front_lawn", zone_name="Front Lawn", zone_type="alert_zone"),)
    )
    assert _by_topic(mapper.snapshot_publishes(vacant, now_ns=NOW_NS))[
        "minimappr/zone/front_lawn/occupancy"
    ] == "OFF"


def test_all_stateful_topics_are_retained() -> None:
    """HA restarting must restore last-known state without waiting for our poll."""
    mapper = _mapper()
    messages = mapper.snapshot_publishes(_snapshot(tracks=()), now_ns=NOW_NS)
    assert messages and all(message.retain for message in messages)
    assert all(message.coalescable for message in messages)


# -- zone SPL ----------------------------------------------------------------


def test_zone_with_no_spl_sample_publishes_unknown_not_zero() -> None:
    """0.0 dB would be a reading we never took."""
    mapper = _mapper()
    published = _by_topic(mapper.snapshot_publishes(_snapshot(), now_ns=NOW_NS))
    assert published["minimappr/zone/front_lawn/spl_db"] == "None"


def test_zone_spl_publishes_the_window_max() -> None:
    mapper = _mapper()
    for value in (61.5, 88.25, 70.0):
        mapper.observe_detection(
            {"id": "d", "label": "speech", "zone_ids": ["front_lawn"], "spl_db": value, "timestamp_ns": NOW_NS},
            now_ns=NOW_NS,
        )
    published = _by_topic(mapper.snapshot_publishes(_snapshot(), now_ns=NOW_NS))
    assert published["minimappr/zone/front_lawn/spl_db"] == "88.2"


def test_zone_spl_expires_back_to_unknown() -> None:
    mapper = _mapper(zone_spl_window_seconds=10.0)
    mapper.observe_detection(
        {"id": "d", "label": "speech", "zone_ids": ["front_lawn"], "spl_db": 90.0, "timestamp_ns": NOW_NS},
        now_ns=NOW_NS,
    )
    inside = _by_topic(mapper.snapshot_publishes(_snapshot(), now_ns=NOW_NS + 5 * SECOND_NS))
    assert inside["minimappr/zone/front_lawn/spl_db"] == "90.0"
    outside = _by_topic(mapper.snapshot_publishes(_snapshot(), now_ns=NOW_NS + 20 * SECOND_NS))
    assert outside["minimappr/zone/front_lawn/spl_db"] == "None"


def test_spl_is_attributed_only_to_the_detections_zones() -> None:
    mapper = _mapper()
    mapper.observe_detection(
        {"id": "d", "label": "speech", "zone_ids": ["back_yard"], "spl_db": 95.0, "timestamp_ns": NOW_NS},
        now_ns=NOW_NS,
    )
    published = _by_topic(mapper.snapshot_publishes(_snapshot(), now_ns=NOW_NS))
    assert published["minimappr/zone/front_lawn/spl_db"] == "None"


def test_aggregator_rejects_a_nonpositive_window() -> None:
    with pytest.raises(ValueError):
        ZoneSplAggregator(0.0)


def test_aggregator_ignores_missing_and_unparseable_spl() -> None:
    aggregator = ZoneSplAggregator(60.0)
    aggregator.observe(zone_ids=["z"], received_level_db=None, timestamp_ns=NOW_NS)
    aggregator.observe(zone_ids=["z"], received_level_db="loud", timestamp_ns=NOW_NS)  # type: ignore[arg-type]
    assert aggregator.max_for_zone("z", now_ns=NOW_NS) is None


def test_aggregator_drops_fully_expired_zones() -> None:
    aggregator = ZoneSplAggregator(1.0)
    aggregator.observe(zone_ids=["z"], received_level_db=50.0, timestamp_ns=NOW_NS)
    assert aggregator.tracked_zone_count() == 1
    aggregator.prune(now_ns=NOW_NS + 10 * SECOND_NS)
    assert aggregator.tracked_zone_count() == 0


# -- nodes -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("health", "expected_state", "expected_degraded"),
    [
        ("online", "ON", False),
        ("degraded", "ON", True),
        ("bit_fail", "ON", True),
        ("offline", "OFF", False),
        ("", "OFF", False),
    ],
)
def test_node_connectivity_is_binary_with_degradation_in_attributes(
    health: str, expected_state: str, expected_degraded: bool
) -> None:
    """A degraded node is still reachable — it must not read as unplugged."""
    mapper = _mapper()
    snapshot = _snapshot(
        nodes=(NodeStateInput(node_id="node-a", node_name="node-a", health_status=health),)
    )
    published = _by_topic(mapper.snapshot_publishes(snapshot, now_ns=NOW_NS))
    assert published["minimappr/node/node_a/connectivity"] == expected_state
    attributes = json.loads(published["minimappr/node/node_a/attributes"])
    assert attributes["health_status"] == health.lower()
    assert attributes["degraded"] is expected_degraded


def test_node_detail_is_merged_into_attributes() -> None:
    mapper = _mapper()
    snapshot = _snapshot(
        nodes=(
            NodeStateInput(
                node_id="node-a",
                node_name="Node A",
                health_status="degraded",
                detail={"time_quality": "free_running"},
            ),
        )
    )
    published = _by_topic(mapper.snapshot_publishes(snapshot, now_ns=NOW_NS))
    attributes = json.loads(published["minimappr/node/node_a/attributes"])
    assert attributes["time_quality"] == "free_running"
    assert attributes["node_name"] == "Node A"


# -- system ------------------------------------------------------------------


def test_system_health_and_track_count() -> None:
    mapper = _mapper()
    snapshot = _snapshot(
        system=SystemStateInput(
            system_health="degraded",
            active_track_count=7,
            online_nodes=2,
            degraded_nodes=1,
            offline_nodes=3,
            generated_ns=NOW_NS,
        )
    )
    published = _by_topic(mapper.snapshot_publishes(snapshot, now_ns=NOW_NS))
    assert published["minimappr/system/health"] == "degraded"
    assert published["minimappr/system/active_track_count"] == "7"
    attributes = json.loads(published["minimappr/system/attributes"])
    assert attributes["online_nodes"] == 2
    assert attributes["degraded_nodes"] == 1
    assert attributes["offline_nodes"] == 3
    assert attributes["active_tracks"] == 7


def test_unrecognized_system_health_publishes_unknown_not_a_guess() -> None:
    mapper = _mapper()
    snapshot = _snapshot(system=SystemStateInput(system_health="wedged"))
    published = _by_topic(mapper.snapshot_publishes(snapshot, now_ns=NOW_NS))
    assert published["minimappr/system/health"] == "None"


# -- detections --------------------------------------------------------------


def test_detection_matches_both_its_label_and_its_category() -> None:
    mapper = _mapper()
    messages = mapper.observe_detection(
        {
            "id": "d1",
            "label": "gunshot",
            "label_category": "security",
            "label_confidence": 0.9,
            "zone_ids": ["front_lawn"],
            "timestamp_ns": NOW_NS,
        },
        now_ns=NOW_NS,
    )
    topics = {message.topic for message in messages}
    assert "minimappr/detection_class/gunshot" in topics
    assert "minimappr/detection_class/security" in topics


def test_unconfigured_detection_class_publishes_no_binary_sensor() -> None:
    mapper = _mapper(detection_classes=("gunshot",))
    messages = mapper.observe_detection(
        {"id": "d", "label": "birdsong", "label_category": "wildlife", "timestamp_ns": NOW_NS},
        now_ns=NOW_NS,
    )
    assert all("detection_class" not in message.topic for message in messages)


def test_detection_impulses_are_never_retained_or_coalescable() -> None:
    """A retained ON restores on HA restart and auto-offs into a phantom trigger."""
    mapper = _mapper()
    messages = mapper.observe_detection(
        {"id": "d", "label": "gunshot", "label_category": "security", "timestamp_ns": NOW_NS},
        now_ns=NOW_NS,
    )
    assert messages
    assert not any(message.retain for message in messages)
    assert not any(message.coalescable for message in messages)


def test_detection_event_type_is_clamped_to_the_taxonomy() -> None:
    mapper = _mapper()
    messages = mapper.observe_detection(
        {"id": "d", "label": "gunshot", "label_category": "not_a_category", "timestamp_ns": NOW_NS},
        now_ns=NOW_NS,
    )
    event = next(m for m in messages if m.topic == "minimappr/event/detection")
    assert json.loads(event.payload)["event_type"] == "unknown"


def test_detection_event_carries_position_when_present() -> None:
    mapper = _mapper()
    messages = mapper.observe_detection(
        {
            "id": "d",
            "label": "gunshot",
            "label_category": "security",
            "position_geo": {"lat": 37.5, "lon": -122.1},
            "spl_db": 101.2,
            "timestamp_ns": NOW_NS,
        },
        now_ns=NOW_NS,
    )
    event = json.loads(next(m for m in messages if m.topic == "minimappr/event/detection").payload)
    assert event["latitude"] == 37.5
    assert event["longitude"] == -122.1
    assert event["spl_db"] == 101.2


def test_declared_event_types_cover_every_taxonomy_category() -> None:
    mapper = _mapper()
    entities = mapper.desired_discovery(zones=(), nodes=())
    config = entities["homeassistant/event/minimappr/detection/config"]
    assert config.payload["event_types"] == detection_event_types()
    assert "unknown" in config.payload["event_types"], "the clamp fallback must be declared"


def test_alert_event_priority_is_clamped_and_impulsive() -> None:
    mapper = _mapper()
    messages = mapper.alert_publish(
        {"alert_id": "a1", "rule_id": "r1", "priority": "WEIRD", "timestamp_ns": NOW_NS}
    )
    assert len(messages) == 1
    assert not messages[0].retain and not messages[0].coalescable
    assert json.loads(messages[0].payload)["event_type"] == "normal"


def test_alert_event_omits_absent_fields() -> None:
    mapper = _mapper()
    payload = json.loads(mapper.alert_publish({"alert_id": "a1", "priority": "high"})[0].payload)
    assert payload["event_type"] == "high"
    assert "detection_id" not in payload, "absent fields are omitted, not null-filled"


# -- entity toggles ----------------------------------------------------------


@pytest.mark.parametrize(
    ("toggle", "suppressed_fragment"),
    [
        ("publish_zone_occupancy", "/zone/front_lawn/occupancy"),
        ("publish_zone_spl", "/zone/front_lawn/spl_db"),
        ("publish_node_status", "/node/node_a/connectivity"),
        ("publish_system_health", "/system/health"),
        ("publish_track_slots", "/track/00/state"),
    ],
)
def test_each_toggle_suppresses_exactly_its_own_topics(toggle: str, suppressed_fragment: str) -> None:
    on_snapshot = _snapshot(tracks=(TrackSlotCandidate(track_id="t1", tqi=0.9),))
    assert any(
        suppressed_fragment in message.topic
        for message in _mapper().snapshot_publishes(on_snapshot, now_ns=NOW_NS)
    ), "sanity: the topic exists when the toggle is on"

    off_mapper = _mapper(**{toggle: False})
    assert not any(
        suppressed_fragment in message.topic
        for message in off_mapper.snapshot_publishes(on_snapshot, now_ns=NOW_NS)
    )


def test_toggles_remove_entities_from_the_desired_discovery_set() -> None:
    zones = (ZoneStateInput(zone_id="z1", zone_name="Z1", zone_type="alert_zone"),)
    nodes = (NodeStateInput(node_id="n1", node_name="n1", health_status="online"),)
    full = _mapper().desired_discovery(zones=zones, nodes=nodes)
    assert any("zone_occupancy_z1" in topic for topic in full)

    trimmed = _mapper(
        publish_zone_occupancy=False,
        publish_zone_spl=False,
        publish_node_status=False,
        publish_events=False,
        publish_detection_classes=False,
        publish_track_slots=False,
    ).desired_discovery(zones=zones, nodes=nodes)
    # Only the system diagnostics pair survives.
    assert set(trimmed) == {
        "homeassistant/sensor/minimappr/system_health/config",
        "homeassistant/sensor/minimappr/active_track_count/config",
    }


def test_discovery_entities_declare_the_state_topics_they_own() -> None:
    """Removal must be able to blank the state topics, not just the config."""
    zones = (ZoneStateInput(zone_id="z1", zone_name="Z1", zone_type="alert_zone"),)
    entities = _mapper().desired_discovery(zones=zones, nodes=())
    occupancy = entities["homeassistant/binary_sensor/minimappr/zone_occupancy_z1/config"]
    assert occupancy.state_topics == (
        "minimappr/zone/z1/occupancy",
        "minimappr/zone/z1/attributes",
    )


def test_removal_blanks_the_config_and_every_state_topic() -> None:
    mapper = _mapper()
    messages = mapper.removal_publishes(
        config_topic="homeassistant/binary_sensor/minimappr/zone_occupancy_z1/config",
        state_topics=("minimappr/zone/z1/occupancy", "minimappr/zone/z1/attributes"),
    )
    assert len(messages) == 3
    assert all(message.payload == "" and message.retain for message in messages)


def test_discovery_config_is_published_retained() -> None:
    mapper = _mapper()
    entity = next(iter(mapper.desired_discovery(zones=(), nodes=()).values()))
    assert mapper.discovery_publish(entity).retain is True


# -- track slots -------------------------------------------------------------


def test_slots_fill_by_descending_tqi() -> None:
    allocator = TrackSlotAllocator(2)
    assignments = allocator.assign(
        [
            TrackSlotCandidate(track_id="low", tqi=0.1),
            TrackSlotCandidate(track_id="high", tqi=0.9),
            TrackSlotCandidate(track_id="mid", tqi=0.5),
        ]
    )
    assert [a.candidate.track_id for a in assignments] == ["high", "mid"]


def test_slot_assignment_is_sticky_across_cycles() -> None:
    """An automation watching one slot must keep following one target."""
    allocator = TrackSlotAllocator(2)
    allocator.assign([TrackSlotCandidate(track_id="t1", tqi=0.2)])
    # A far better track arrives; the incumbent keeps its slot.
    assignments = allocator.assign(
        [TrackSlotCandidate(track_id="t1", tqi=0.2), TrackSlotCandidate(track_id="t2", tqi=0.99)]
    )
    assert [a.candidate.track_id for a in assignments] == ["t1", "t2"]


def test_ties_break_on_lowest_track_id_for_determinism() -> None:
    first = TrackSlotAllocator(1).assign(
        [TrackSlotCandidate(track_id="bbb", tqi=0.5), TrackSlotCandidate(track_id="aaa", tqi=0.5)]
    )
    second = TrackSlotAllocator(1).assign(
        [TrackSlotCandidate(track_id="aaa", tqi=0.5), TrackSlotCandidate(track_id="bbb", tqi=0.5)]
    )
    assert first[0].candidate.track_id == second[0].candidate.track_id == "aaa"


def test_departed_track_frees_its_slot_for_a_newcomer_same_cycle() -> None:
    allocator = TrackSlotAllocator(1)
    allocator.assign([TrackSlotCandidate(track_id="t1", tqi=0.9)])
    assignments = allocator.assign([TrackSlotCandidate(track_id="t2", tqi=0.1)])
    assert assignments[0].candidate.track_id == "t2"


def test_vacant_slot_publishes_not_home_with_a_null_track_id() -> None:
    mapper = _mapper(track_slot_count=2)
    published = _by_topic(
        mapper.snapshot_publishes(
            _snapshot(tracks=(TrackSlotCandidate(track_id="t1", tqi=0.9),)), now_ns=NOW_NS
        )
    )
    assert published["minimappr/track/00/state"] == "home"
    assert published["minimappr/track/01/state"] == "not_home"
    assert json.loads(published["minimappr/track/01/attributes"])["track_id"] is None


def test_evicted_track_slot_reverts_to_not_home() -> None:
    mapper = _mapper(track_slot_count=1)
    mapper.snapshot_publishes(
        _snapshot(tracks=(TrackSlotCandidate(track_id="t1", tqi=0.9),)), now_ns=NOW_NS
    )
    published = _by_topic(mapper.snapshot_publishes(_snapshot(tracks=()), now_ns=NOW_NS))
    assert published["minimappr/track/00/state"] == "not_home"


def test_occupied_slot_attributes_carry_gps_source_and_position() -> None:
    mapper = _mapper(track_slot_count=1)
    published = _by_topic(
        mapper.snapshot_publishes(
            _snapshot(
                tracks=(
                    TrackSlotCandidate(
                        track_id="t1",
                        tqi=0.8,
                        label="speech",
                        lat=37.5,
                        lon=-122.1,
                        altitude_m=12.0,
                        status="confirmed",
                        confidence=0.75,
                    ),
                )
            ),
            now_ns=NOW_NS,
        )
    )
    attributes = json.loads(published["minimappr/track/00/attributes"])
    assert attributes["source_type"] == "gps", "HA ignores lat/lon without this"
    assert (attributes["latitude"], attributes["longitude"]) == (37.5, -122.1)
    assert attributes["altitude"] == 12.0
    assert attributes["label"] == "speech"
    assert attributes["status"] == "confirmed"


def test_track_without_a_position_omits_lat_lon() -> None:
    mapper = _mapper(track_slot_count=1)
    published = _by_topic(
        mapper.snapshot_publishes(
            _snapshot(tracks=(TrackSlotCandidate(track_id="t1", tqi=0.8),)), now_ns=NOW_NS
        )
    )
    attributes = json.loads(published["minimappr/track/00/attributes"])
    assert "latitude" not in attributes and "longitude" not in attributes


def test_zero_slot_count_publishes_nothing_and_allocates_nothing() -> None:
    allocator = TrackSlotAllocator(0)
    assert allocator.assign([TrackSlotCandidate(track_id="t1", tqi=0.9)]) == []
    mapper = _mapper(track_slot_count=0)
    published = _by_topic(
        mapper.snapshot_publishes(
            _snapshot(tracks=(TrackSlotCandidate(track_id="t1", tqi=0.9),)), now_ns=NOW_NS
        )
    )
    assert not any("/track/" in topic for topic in published)


def test_negative_slot_count_is_rejected() -> None:
    with pytest.raises(ValueError):
        TrackSlotAllocator(-1)
