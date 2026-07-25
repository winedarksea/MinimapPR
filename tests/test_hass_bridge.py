"""HassBridge lifecycle, availability, coalescing/dedupe, and failure handling."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from minimappr.core.hass import bridge as bridge_module
from minimappr.core.hass.bridge import HassBridge, HassBridgeConfig
from minimappr.core.hass.outbound import OutboundQueue
from minimappr.core.hass.transport import MqttPublish
from tests.hass_helpers import (
    RecordingMqttTransport,
    build_test_bridge,
    node,
    snapshot,
    static_snapshot_provider,
    zone,
)

SECOND_NS = 1_000_000_000


# -- dormancy ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_bridge_creates_no_task_and_publishes_nothing(tmp_path: Path, monkeypatch) -> None:
    bridge, transport = build_test_bridge(tmp_path, monkeypatch, enabled=False)
    await bridge.start()
    assert bridge.connection_state == "disabled"
    assert transport.connect_calls == 0
    assert not any(task.get_name() == "hass-publisher" for task in asyncio.all_tasks())
    await bridge.stop()


@pytest.mark.asyncio
async def test_disabled_bridge_reports_disabled_status(tmp_path: Path, monkeypatch) -> None:
    bridge, _ = build_test_bridge(tmp_path, monkeypatch, enabled=False)
    status = bridge.status()
    assert status["enabled"] is False
    assert status["connection_state"] == "disabled"


# -- availability -----------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_publishes_retained_birth_message(tmp_path: Path, monkeypatch) -> None:
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    assert await bridge._connect_once() is True
    assert transport.retained()["minimappr/status"] == "online"
    birth = next(m for m in transport.published if m.topic == "minimappr/status")
    assert birth.retain is True
    await bridge.stop()


@pytest.mark.asyncio
async def test_transport_config_carries_the_offline_will(tmp_path: Path, monkeypatch) -> None:
    """The LWT is what makes HA mark us unavailable on an ungraceful death."""
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    await bridge._connect_once()
    will = transport.will
    assert will is not None
    assert (will.topic, will.payload, will.retain) == ("minimappr/status", "offline", True)
    await bridge.stop()


@pytest.mark.asyncio
async def test_graceful_stop_publishes_offline_then_disconnects(tmp_path: Path, monkeypatch) -> None:
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    await bridge._connect_once()
    transport.clear()

    await bridge.stop()

    assert transport.published[-1].topic == "minimappr/status"
    assert transport.published[-1].payload == "offline"
    assert transport.published[-1].retain is True
    assert transport.disconnect_calls == 1


# -- connect failure / backoff ----------------------------------------------


@pytest.mark.asyncio
async def test_connect_failure_records_error_state_and_counts_the_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    transport = RecordingMqttTransport(fail_connect_times=1)
    bridge, _ = build_test_bridge(tmp_path, monkeypatch, transport=transport)

    assert await bridge._connect_once() is False
    assert bridge.connection_state == "error"
    assert "simulated connect failure" in str(bridge.status()["last_connect_error"])
    assert bridge.metrics.reconnect_count == 1

    assert await bridge._connect_once() is True
    assert bridge.connection_state == "connected"
    assert bridge.status()["last_connect_error"] is None
    await bridge.stop()


@pytest.mark.asyncio
async def test_reconnect_clears_the_unchanged_dedupe_cache(tmp_path: Path, monkeypatch) -> None:
    """A broker restart may have dropped retained messages; suppressing the
    republish would leave HA permanently stale."""
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    bridge.set_state_snapshot_provider(static_snapshot_provider(snapshot(zones=(zone("z1"),))))
    await bridge._connect_once()
    assert transport.count_for("minimappr/zone/z1/occupancy") == 1

    await bridge._connect_once()

    assert transport.count_for("minimappr/zone/z1/occupancy") == 2
    await bridge.stop()


def test_backoff_jitter_stays_within_twenty_percent() -> None:
    for _ in range(200):
        value = bridge_module._with_jitter(10.0)
        assert 8.0 <= value <= 12.0


def test_backoff_jitter_never_goes_negative() -> None:
    assert bridge_module._with_jitter(0.0) == 0.0


# -- offline queueing -------------------------------------------------------


@pytest.mark.asyncio
async def test_messages_queued_while_disconnected_flush_on_connect(tmp_path: Path, monkeypatch) -> None:
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    bridge.enqueue(MqttPublish(topic="minimappr/rooms/kitchen", payload="ON", retain=True))

    await bridge._connect_once()

    assert transport.payload_for("minimappr/rooms/kitchen") == "ON"
    await bridge.stop()


@pytest.mark.asyncio
async def test_publishing_while_disconnected_requeues_rather_than_dropping(
    tmp_path: Path, monkeypatch
) -> None:
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    bridge.enqueue(MqttPublish(topic="minimappr/x", payload="1", retain=True))

    await bridge._flush_outbound()  # never connected: no transport
    assert transport.published == []

    await bridge._connect_once()
    assert transport.payload_for("minimappr/x") == "1"
    await bridge.stop()


# -- queue full -------------------------------------------------------------


@pytest.mark.asyncio
async def test_queue_full_increments_the_drop_counter_and_returns_false(
    tmp_path: Path, monkeypatch
) -> None:
    bridge, _ = build_test_bridge(tmp_path, monkeypatch, queue_size=2)
    assert bridge.enqueue(MqttPublish(topic="a", payload="1")) is True
    assert bridge.enqueue(MqttPublish(topic="b", payload="2")) is True

    assert bridge.enqueue(MqttPublish(topic="c", payload="3")) is False
    assert bridge.metrics.messages_dropped_queue_full == 1


@pytest.mark.asyncio
async def test_live_event_queue_full_counts_separately(tmp_path: Path, monkeypatch) -> None:
    bridge, _ = build_test_bridge(tmp_path, monkeypatch, queue_size=1)
    bridge.handle_live_event({"type": "detection"})
    bridge.handle_live_event({"type": "detection"})

    assert bridge.metrics.live_events_dropped == 1
    assert bridge.metrics.messages_dropped_queue_full == 0


@pytest.mark.asyncio
async def test_handle_live_event_never_raises(tmp_path: Path, monkeypatch) -> None:
    """It runs on the fusion hot path via LiveEventHub; raising would propagate."""
    bridge, _ = build_test_bridge(tmp_path, monkeypatch, queue_size=1)
    for _ in range(5):
        bridge.handle_live_event({"type": "detection"})


# -- coalescing / dedupe ----------------------------------------------------


@pytest.mark.asyncio
async def test_three_updates_to_one_topic_collapse_to_one_publish(tmp_path: Path, monkeypatch) -> None:
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    await bridge._connect_once()
    transport.clear()

    for value in ("OFF", "ON", "OFF"):
        bridge.enqueue(MqttPublish(topic="minimappr/zone/z1/occupancy", payload=value, retain=True))
    await bridge._flush_outbound()

    assert transport.count_for("minimappr/zone/z1/occupancy") == 1
    assert transport.payload_for("minimappr/zone/z1/occupancy") == "OFF", "last write wins"
    assert bridge.metrics.messages_coalesced == 2
    await bridge.stop()


@pytest.mark.asyncio
async def test_unchanged_retained_payload_is_suppressed_next_cycle(tmp_path: Path, monkeypatch) -> None:
    """This is what stops republishing identical zone state every interval."""
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    bridge.set_state_snapshot_provider(static_snapshot_provider(snapshot(zones=(zone("z1"),))))
    await bridge._connect_once()
    published_first = len(transport.published)
    assert published_first > 0

    await bridge._publish_cycle_once()

    assert len(transport.published) == published_first, "nothing changed, nothing republished"
    assert bridge.metrics.messages_suppressed_unchanged > 0
    await bridge.stop()


@pytest.mark.asyncio
async def test_changed_state_is_republished(tmp_path: Path, monkeypatch) -> None:
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    bridge.set_state_snapshot_provider(static_snapshot_provider(snapshot(zones=(zone("z1"),))))
    await bridge._connect_once()
    transport.clear()

    bridge.set_state_snapshot_provider(
        static_snapshot_provider(snapshot(zones=(zone("z1", occupied=True),)))
    )
    await bridge._publish_cycle_once()

    assert transport.payload_for("minimappr/zone/z1/occupancy") == "ON"
    await bridge.stop()


@pytest.mark.asyncio
async def test_impulses_are_never_coalesced_or_deduped(tmp_path: Path, monkeypatch) -> None:
    """Two identical gunshots one second apart are two events."""
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    await bridge._connect_once()
    transport.clear()

    for _ in range(3):
        bridge.enqueue(
            MqttPublish(
                topic="minimappr/event/detection", payload='{"event_type":"security"}', coalescable=False
            )
        )
    await bridge._flush_outbound()

    assert transport.count_for("minimappr/event/detection") == 3
    assert bridge.metrics.messages_coalesced == 0
    assert bridge.metrics.messages_suppressed_unchanged == 0
    await bridge.stop()


@pytest.mark.asyncio
async def test_rate_floor_holds_a_topic_without_losing_it(tmp_path: Path, monkeypatch) -> None:
    bridge, transport = build_test_bridge(
        tmp_path, monkeypatch, publish_min_interval_seconds=3600.0
    )
    await bridge._connect_once()
    transport.clear()

    bridge.enqueue(MqttPublish(topic="minimappr/z", payload="1", retain=True))
    await bridge._flush_outbound()
    assert transport.count_for("minimappr/z") == 1

    # Second value inside the floor: held back, not published, not dropped.
    bridge.enqueue(MqttPublish(topic="minimappr/z", payload="2", retain=True))
    await bridge._flush_outbound()
    assert transport.count_for("minimappr/z") == 1
    assert bridge.metrics.messages_dropped_queue_full == 0
    await bridge.stop()


def test_deferred_message_is_superseded_by_a_newer_value() -> None:
    """The rate floor is lossless precisely because deferral is coalescable:
    once the floor lapses the *latest* value goes out, not a stale backlog."""
    queue = OutboundQueue(maxsize=10, min_interval_seconds=10.0)
    now_ns = 1_000_000_000_000

    queue.enqueue(MqttPublish(topic="t", payload="1", retain=True))
    first = queue.collect_ready(now_ns=now_ns)
    assert [message.payload for message in first] == ["1"]
    queue.record_published(first[0], now_ns=now_ns)

    # Inside the floor: deferred rather than sent.
    queue.enqueue(MqttPublish(topic="t", payload="2", retain=True))
    assert queue.collect_ready(now_ns=now_ns + SECOND_NS) == []

    # A newer value arrives, then the floor lapses: only the newest goes out.
    queue.enqueue(MqttPublish(topic="t", payload="3", retain=True))
    ready = queue.collect_ready(now_ns=now_ns + 20 * SECOND_NS)
    assert [message.payload for message in ready] == ["3"]


def test_rate_floor_of_zero_disables_deferral() -> None:
    queue = OutboundQueue(maxsize=10, min_interval_seconds=0.0)
    queue.enqueue(MqttPublish(topic="t", payload="1", retain=True))
    published = queue.collect_ready(now_ns=1_000)[0]
    queue.record_published(published, now_ns=1_000)

    queue.enqueue(MqttPublish(topic="t", payload="2", retain=True))
    assert [message.payload for message in queue.collect_ready(now_ns=1_000)] == ["2"]


def test_deleting_a_topic_clears_it_from_the_unchanged_cache() -> None:
    """An empty retained payload is a delete, so a later re-create of the same
    payload must not be suppressed as 'unchanged'."""
    queue = OutboundQueue(maxsize=10, min_interval_seconds=0.0)
    created = MqttPublish(topic="t", payload="ON", retain=True)

    queue.record_published(created, now_ns=1)
    assert queue.published_state_topic_count == 1

    queue.record_published(MqttPublish(topic="t", payload="", retain=True), now_ns=2)
    assert queue.published_state_topic_count == 0

    queue.enqueue(created)
    assert [message.payload for message in queue.collect_ready(now_ns=3)] == ["ON"]


@pytest.mark.asyncio
async def test_rate_floor_does_not_apply_to_impulses(tmp_path: Path, monkeypatch) -> None:
    bridge, transport = build_test_bridge(
        tmp_path, monkeypatch, publish_min_interval_seconds=3600.0
    )
    await bridge._connect_once()
    transport.clear()

    for _ in range(2):
        bridge.enqueue(MqttPublish(topic="minimappr/event/alert", payload="{}", coalescable=False))
        await bridge._flush_outbound()

    assert transport.count_for("minimappr/event/alert") == 2
    await bridge.stop()


# -- publish failure --------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_failure_marks_disconnected_without_killing_the_loop(
    tmp_path: Path, monkeypatch
) -> None:
    transport = RecordingMqttTransport(fail_publish_on={"minimappr/bad"})
    bridge, _ = build_test_bridge(tmp_path, monkeypatch, transport=transport)
    await bridge._connect_once()

    bridge.enqueue(MqttPublish(topic="minimappr/bad", payload="x", retain=True))
    await bridge._flush_outbound()

    assert bridge.connection_state == "error"
    assert bridge.metrics.messages_failed == 1
    assert transport.disconnect_calls >= 1

    # The next cycle must still work rather than the loop having died.
    await bridge._publish_cycle_once()
    await bridge.stop()


@pytest.mark.asyncio
async def test_a_failing_snapshot_provider_does_not_break_the_cycle(
    tmp_path: Path, monkeypatch
) -> None:
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)

    async def broken_provider():
        raise RuntimeError("storage is down")

    bridge.set_state_snapshot_provider(broken_provider)
    await bridge._connect_once()
    transport.clear()

    await bridge._publish_cycle_once()  # must not raise

    await bridge.stop()


# -- inbound mapping --------------------------------------------------------


@pytest.mark.asyncio
async def test_tee_detections_become_impulse_publishes(tmp_path: Path, monkeypatch) -> None:
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    await bridge._connect_once()
    transport.clear()

    bridge.handle_live_event(
        {
            "type": "detection",
            "detection": {
                "id": "d1",
                "label": "gunshot",
                "label_category": "security",
                "label_confidence": 0.9,
                "zone_ids": ["z1"],
                "spl_db": 104.0,
                "timestamp_ns": 1,
            },
        }
    )
    await bridge._publish_cycle_once()

    assert "minimappr/detection_class/gunshot" in transport.topics()
    assert "minimappr/event/detection" in transport.topics()
    await bridge.stop()


@pytest.mark.asyncio
async def test_tee_alerts_become_alert_events(tmp_path: Path, monkeypatch) -> None:
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    await bridge._connect_once()
    transport.clear()

    bridge.handle_live_event({"type": "alert", "alert_id": "a1", "priority": "high"})
    await bridge._publish_cycle_once()

    assert "minimappr/event/alert" in transport.topics()
    await bridge.stop()


@pytest.mark.asyncio
async def test_zone_change_events_request_a_reconcile(tmp_path: Path, monkeypatch) -> None:
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    bridge.set_state_snapshot_provider(static_snapshot_provider(snapshot(zones=(zone("z1"),))))
    await bridge._connect_once()
    transport.clear()

    bridge.handle_live_event({"type": "zone_updated", "zone_id": "z2"})
    bridge.set_state_snapshot_provider(
        static_snapshot_provider(snapshot(zones=(zone("z1"), zone("z2"))))
    )
    await bridge._publish_cycle_once()

    assert "homeassistant/binary_sensor/minimappr/zone_occupancy_z2/config" in transport.topics()
    await bridge.stop()


@pytest.mark.asyncio
async def test_unparseable_live_event_is_swallowed(tmp_path: Path, monkeypatch) -> None:
    bridge, _ = build_test_bridge(tmp_path, monkeypatch)
    await bridge._connect_once()
    bridge.handle_live_event({"type": "detection", "detection": "not a dict"})
    await bridge._publish_cycle_once()  # must not raise
    await bridge.stop()


# -- status -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_reports_queue_depth_and_entity_counts(tmp_path: Path, monkeypatch) -> None:
    bridge, _ = build_test_bridge(tmp_path, monkeypatch)
    bridge.set_state_snapshot_provider(
        static_snapshot_provider(snapshot(zones=(zone("z1"),), nodes=(node("n1"),)))
    )
    await bridge._connect_once()

    status = bridge.status()
    assert status["connection_state"] == "connected"
    assert status["transport"] == "recording"
    assert status["discovery_entity_count"] > 0
    assert status["published_state_topic_count"] > 0
    assert status["queue_capacity"] == bridge.config.queue_size
    assert status["connected_since_ns"] is not None
    assert set(status["metrics"]) >= {
        "messages_published",
        "messages_failed",
        "messages_dropped_queue_full",
        "messages_suppressed_unchanged",
        "messages_coalesced",
        "live_events_dropped",
        "reconnect_count",
    }
    await bridge.stop()


@pytest.mark.asyncio
async def test_status_transitions_broadcast_once_each(tmp_path: Path, monkeypatch) -> None:
    events: list[dict] = []

    async def live_callback(payload: dict) -> None:
        events.append(payload)

    bridge, _ = build_test_bridge(tmp_path, monkeypatch, live_callback=live_callback)
    await bridge._connect_once()
    await asyncio.sleep(0)  # let the transition broadcast tasks run

    kinds = [event["type"] for event in events]
    assert kinds and set(kinds) == {"hass_status"}
    assert all(event["event_type"] == "hass_status" for event in events), (
        "the Leptos LiveEvent enum is tagged on 'type'; older readers use 'event_type'"
    )
    await bridge.stop()


@pytest.mark.asyncio
async def test_start_is_idempotent_and_stop_cancels_the_task(tmp_path: Path, monkeypatch) -> None:
    bridge, _ = build_test_bridge(tmp_path, monkeypatch)
    await bridge.start()
    await bridge.start()
    running = [task for task in asyncio.all_tasks() if task.get_name() == "hass-publisher"]
    assert len(running) == 1

    await bridge.stop()
    assert not [task for task in asyncio.all_tasks() if task.get_name() == "hass-publisher"]


@pytest.mark.asyncio
async def test_stop_without_start_is_safe(tmp_path: Path, monkeypatch) -> None:
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    await bridge.stop()
    assert transport.disconnect_calls == 0


def test_bridge_can_be_constructed_without_an_mqtt_client_installed(tmp_path: Path) -> None:
    """Phases 1-7 must be importable and testable with aiomqtt absent."""
    bridge = HassBridge(
        config=HassBridgeConfig(
            enabled=True,
            mqtt_host="broker.test",
            discovery_ledger_path=tmp_path / "ledger.json",
        )
    )
    assert bridge.enabled is True
    assert bridge.transport_name is None
