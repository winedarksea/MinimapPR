"""HassRuleActionHandler — fast return, honest status, and the topic guard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minimappr.core.hass.rules_handler import HassRuleActionHandler
from minimappr.interfaces import ActionDescriptor
from minimappr.models import DetectionEvent, TrackState
from tests.hass_helpers import build_test_bridge


def _descriptor(**payload) -> ActionDescriptor:
    return ActionDescriptor(
        action_type="alert", destination="hass", priority="high", payload=payload
    )


def _detection() -> DetectionEvent:
    return DetectionEvent(
        id="d1",
        timestamp_ns=1,
        position_m=(1.0, 2.0, 3.0),
        confidence=0.8,
        gdop=1.2,
        label="gunshot",
        label_category="security",
        label_confidence=0.9,
        reference_sensor="node-a",
    )


def _track() -> TrackState:
    return TrackState(
        id="t1",
        label="gunshot",
        position_m=(1.0, 2.0, 3.0),
        velocity_mps=(0.0, 0.0, 0.0),
        confidence=0.7,
        updated_ns=1,
        first_seen_ns=1,
        last_seen_ns=1,
    )


@pytest.mark.asyncio
async def test_happy_path_queues_one_noncoalescable_message(tmp_path: Path, monkeypatch) -> None:
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    handler = HassRuleActionHandler(bridge)

    result = await handler.handle(_descriptor(topic="rooms/kitchen/activity"))

    assert result["delivered"] is True
    assert result["status"] == "QUEUED"
    assert result["topic"] == "minimappr/rooms/kitchen/activity"
    assert bridge.metrics.rule_actions_queued == 1

    await bridge._connect_once()
    assert transport.count_for("minimappr/rooms/kitchen/activity") == 1
    await bridge.stop()


@pytest.mark.asyncio
async def test_handle_never_awaits_the_transport(tmp_path: Path, monkeypatch) -> None:
    """A broker round-trip here would stall the fusion pipeline's detection emit."""
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    await bridge._connect_once()
    transport.clear()
    handler = HassRuleActionHandler(bridge)

    await handler.handle(_descriptor(topic="alerts/high"))

    assert transport.published == [], "handle() must only enqueue, never publish"
    await bridge.stop()


@pytest.mark.asyncio
async def test_missing_topic_is_rejected(tmp_path: Path, monkeypatch) -> None:
    bridge, _ = build_test_bridge(tmp_path, monkeypatch)
    handler = HassRuleActionHandler(bridge)

    for payload in ({}, {"topic": ""}, {"topic": "   "}):
        result = await handler.handle(_descriptor(**payload))
        assert result["delivered"] is False
        assert result["failure_class"] == "missing_topic"
    assert bridge.metrics.rule_actions_queued == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "topic",
    [
        "a/#",
        "a/+/b",
        "/rooms/kitchen",
        "a//b",
    ],
)
async def test_unsafe_topics_are_rejected_and_enqueue_nothing(
    tmp_path: Path, monkeypatch, topic: str
) -> None:
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    await bridge._connect_once()
    transport.clear()
    handler = HassRuleActionHandler(bridge)

    result = await handler.handle(_descriptor(topic=topic))

    assert result["failure_class"] == "invalid_topic"
    await bridge._flush_outbound()
    assert transport.published == []
    await bridge.stop()


@pytest.mark.asyncio
async def test_a_discovery_topic_cannot_escape_the_base_namespace(tmp_path: Path, monkeypatch) -> None:
    """Without the guard a stored rule could overwrite every discovery config."""
    bridge, _ = build_test_bridge(tmp_path, monkeypatch)
    handler = HassRuleActionHandler(bridge)

    result = await handler.handle(
        _descriptor(topic="homeassistant/binary_sensor/minimappr/zone_occupancy_z1/config")
    )

    assert result["delivered"] is True, "prefixed, not refused — it lands in our namespace"
    assert result["topic"].startswith("minimappr/")
    assert not result["topic"].startswith("homeassistant/")
    await bridge.stop()


@pytest.mark.asyncio
async def test_queue_full_is_rejected(tmp_path: Path, monkeypatch) -> None:
    bridge, _ = build_test_bridge(tmp_path, monkeypatch, queue_size=1)
    handler = HassRuleActionHandler(bridge)

    first = await handler.handle(_descriptor(topic="a"))
    second = await handler.handle(_descriptor(topic="b"))

    assert first["delivered"] is True
    assert second["delivered"] is False
    assert second["failure_class"] == "queue_full"
    assert bridge.metrics.rule_actions_rejected == 1


@pytest.mark.asyncio
async def test_disabled_bridge_is_rejected_not_silently_dropped(tmp_path: Path, monkeypatch) -> None:
    bridge, _ = build_test_bridge(tmp_path, monkeypatch, enabled=False)
    handler = HassRuleActionHandler(bridge)

    result = await handler.handle(_descriptor(topic="rooms/kitchen"))

    assert result["delivered"] is False
    assert result["failure_class"] == "bridge_disabled"


# -- message body ------------------------------------------------------------


@pytest.mark.asyncio
async def test_transcript_path_omits_rather_than_null_fills(tmp_path: Path, monkeypatch) -> None:
    """FusionNode's transcript dispatch passes detection=None and track=None."""
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    handler = HassRuleActionHandler(bridge)
    await handler.handle(_descriptor(topic="alerts/high"))
    await bridge._connect_once()

    body = json.loads(transport.payload_for("minimappr/alerts/high"))
    assert "detection_id" not in body
    assert "track_id" not in body
    assert body["action_type"] == "alert"
    assert body["priority"] == "high"
    await bridge.stop()


@pytest.mark.asyncio
async def test_detection_and_track_context_is_included_when_present(
    tmp_path: Path, monkeypatch
) -> None:
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    handler = HassRuleActionHandler(bridge)
    await handler.handle(
        _descriptor(topic="alerts/high"), detection=_detection(), track=_track()
    )
    await bridge._connect_once()

    body = json.loads(transport.payload_for("minimappr/alerts/high"))
    assert body["detection_id"] == "d1"
    assert body["track_id"] == "t1"
    assert body["label_category"] == "security"
    await bridge.stop()


@pytest.mark.asyncio
async def test_a_message_payload_publishes_a_bare_scalar(tmp_path: Path, monkeypatch) -> None:
    """Some HA entities expect "ON", not a JSON object."""
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    handler = HassRuleActionHandler(bridge)
    await handler.handle(_descriptor(topic="rooms/kitchen/occupancy", message="ON"))
    await bridge._connect_once()

    assert transport.payload_for("minimappr/rooms/kitchen/occupancy") == "ON"
    await bridge.stop()


@pytest.mark.asyncio
async def test_extra_payload_keys_are_carried_through(tmp_path: Path, monkeypatch) -> None:
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    handler = HassRuleActionHandler(bridge)
    await handler.handle(_descriptor(topic="equipment/pump1/anomaly", severity="high"))
    await bridge._connect_once()

    body = json.loads(transport.payload_for("minimappr/equipment/pump1/anomaly"))
    assert body["payload"] == {"severity": "high"}
    await bridge.stop()


@pytest.mark.asyncio
async def test_rule_can_opt_into_retain(tmp_path: Path, monkeypatch) -> None:
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    handler = HassRuleActionHandler(bridge)
    await handler.handle(_descriptor(topic="rooms/kitchen/occupancy", message="ON", retain=True))
    await bridge._connect_once()

    assert transport.retained()["minimappr/rooms/kitchen/occupancy"] == "ON"
    await bridge.stop()


@pytest.mark.asyncio
async def test_rule_actions_are_never_coalesced(tmp_path: Path, monkeypatch) -> None:
    """Two rule matches producing the same body are two events."""
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    await bridge._connect_once()
    transport.clear()
    handler = HassRuleActionHandler(bridge)

    for _ in range(3):
        await handler.handle(_descriptor(topic="alerts/high"))
    await bridge._flush_outbound()

    assert transport.count_for("minimappr/alerts/high") == 3
    assert bridge.metrics.messages_coalesced == 0
    await bridge.stop()
