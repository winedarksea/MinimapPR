"""HTTP surface for the Home Assistant bridge: status, republish, purge."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from minimappr.main import _build_hass_state_snapshot_provider, app
from tests.hass_helpers import build_test_bridge, node, snapshot, static_snapshot_provider, zone

STATUS_URL = "/api/v1/integrations/hass/status"
REPUBLISH_URL = "/api/v1/integrations/hass/republish-discovery"
PURGE_URL = "/api/v1/integrations/hass/purge-discovery"


def _configure_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")


# -- status ------------------------------------------------------------------


def test_status_is_200_disabled_when_hass_is_unconfigured(monkeypatch, tmp_path: Path) -> None:
    """A 404 would leave the Settings page with nothing to render."""
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.get(STATUS_URL)
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is False
        assert body["connection_state"] == "disabled"
        assert body["transport"] is None
        assert body["base_topic"] == "minimappr"
        assert isinstance(body["transport_available"], bool)


def test_status_reflects_the_probe_for_the_optional_mqtt_client(monkeypatch, tmp_path: Path) -> None:
    from minimappr.core.hass.aiomqtt_transport import aiomqtt_available

    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        assert client.get(STATUS_URL).json()["transport_available"] == aiomqtt_available()


@pytest.mark.asyncio
async def test_status_returns_the_full_field_set_with_a_bound_bridge(
    monkeypatch, tmp_path: Path
) -> None:
    _configure_env(monkeypatch, tmp_path)
    bridge, _ = build_test_bridge(tmp_path, monkeypatch, mqtt_host="broker.test")
    bridge.set_state_snapshot_provider(
        static_snapshot_provider(snapshot(zones=(zone("z1"),), nodes=(node("n1"),)))
    )
    await bridge._connect_once()

    with TestClient(app) as client:
        app.state.hass_bridge = bridge
        try:
            body = client.get(STATUS_URL).json()
        finally:
            del app.state.hass_bridge

    assert body["enabled"] is True
    assert body["connection_state"] == "connected"
    assert body["transport"] == "recording"
    assert body["mqtt_host"] == "broker.test"
    assert body["discovery_entity_count"] > 0
    assert body["published_state_topic_count"] > 0
    assert body["connected_since_ns"] is not None
    assert body["last_publish_ns"] is not None
    assert body["last_reconcile_ns"] is not None
    assert body["queue_capacity"] == bridge.config.queue_size
    assert set(body["metrics"]) >= {
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
async def test_status_surfaces_the_last_connect_error(monkeypatch, tmp_path: Path) -> None:
    from tests.hass_helpers import RecordingMqttTransport

    _configure_env(monkeypatch, tmp_path)
    bridge, _ = build_test_bridge(
        tmp_path, monkeypatch, transport=RecordingMqttTransport(fail_connect_times=1)
    )
    await bridge._connect_once()

    with TestClient(app) as client:
        app.state.hass_bridge = bridge
        try:
            body = client.get(STATUS_URL).json()
        finally:
            del app.state.hass_bridge

    assert body["connection_state"] == "error"
    assert "simulated connect failure" in body["last_connect_error"]
    await bridge.stop()


# -- republish / purge -------------------------------------------------------


def test_republish_and_purge_are_503_when_the_bridge_is_absent(monkeypatch, tmp_path: Path) -> None:
    """Silently succeeding would tell an operator the purge ran when it did not."""
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        for url in (REPUBLISH_URL, PURGE_URL):
            resp = client.post(url)
            assert resp.status_code == 503
            assert "not enabled" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_republish_and_purge_are_503_when_the_bridge_is_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    _configure_env(monkeypatch, tmp_path)
    bridge, _ = build_test_bridge(tmp_path, monkeypatch, enabled=False)
    with TestClient(app) as client:
        app.state.hass_bridge = bridge
        try:
            assert client.post(REPUBLISH_URL).status_code == 503
            assert client.post(PURGE_URL).status_code == 503
        finally:
            del app.state.hass_bridge


@pytest.mark.asyncio
async def test_republish_forces_a_full_reconcile_next_cycle(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    bridge.set_state_snapshot_provider(static_snapshot_provider(snapshot(zones=(zone("z1"),))))
    await bridge._connect_once()
    transport.clear()

    with TestClient(app) as client:
        app.state.hass_bridge = bridge
        try:
            resp = client.post(REPUBLISH_URL)
        finally:
            del app.state.hass_bridge
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    await bridge._publish_cycle_once()

    config_topic = "homeassistant/binary_sensor/minimappr/zone_occupancy_z1/config"
    assert config_topic in transport.topics(), "the dedupe cache must not suppress this"
    assert "minimappr/zone/z1/occupancy" in transport.topics()
    await bridge.stop()


@pytest.mark.asyncio
async def test_republish_keeps_orphan_removal_working(monkeypatch, tmp_path: Path) -> None:
    """Invalidating digests must not forget which entities we can still delete."""
    _configure_env(monkeypatch, tmp_path)
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    bridge.set_state_snapshot_provider(
        static_snapshot_provider(snapshot(zones=(zone("z1"), zone("gone"))))
    )
    await bridge._connect_once()
    transport.clear()

    bridge.forget_published_state()
    bridge.request_reconcile()
    bridge.set_state_snapshot_provider(static_snapshot_provider(snapshot(zones=(zone("z1"),))))
    await bridge._publish_cycle_once()

    blanked = {m.topic for m in transport.published if m.payload == "" and m.retain}
    assert "homeassistant/binary_sensor/minimappr/zone_occupancy_gone/config" in blanked
    await bridge.stop()


@pytest.mark.asyncio
async def test_purge_queues_removals_and_reports_the_count(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    bridge, transport = build_test_bridge(tmp_path, monkeypatch)
    bridge.set_state_snapshot_provider(
        static_snapshot_provider(snapshot(zones=(zone("z1"),), nodes=(node("n1"),)))
    )
    await bridge._connect_once()
    transport.clear()

    with TestClient(app) as client:
        app.state.hass_bridge = bridge
        try:
            body = client.post(PURGE_URL).json()
        finally:
            del app.state.hass_bridge

    assert body["ok"] is True
    assert body["queued_removals"] > 0
    await bridge._flush_outbound()
    assert transport.published, "removals must actually reach the transport"
    assert all(message.payload == "" for message in transport.published)
    assert transport.retained() == {}, "the broker must hold nothing after a purge"
    await bridge.stop()


# -- runtime wiring ----------------------------------------------------------


def test_snapshot_provider_runs_against_real_runtime_state(monkeypatch, tmp_path: Path) -> None:
    """The poll side of the bridge, driven against a genuinely bound app state.

    Everything else substitutes a static provider, so this is the only coverage
    that `_build_hass_state_snapshot_provider` actually composes with the real
    tracker, zone matcher, storage, and BIT evaluator. It runs on the
    TestClient's portal because the provider must execute on the same event loop
    as the app's storage.
    """
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        client.put(
            "/api/v1/zones/z1",
            json={
                "id": "z1",
                "name": "Zone One",
                "zone_type": "alert_zone",
                "polygon_geo": [[37.0, -122.0], [37.001, -122.0], [37.001, -121.999]],
            },
        ).raise_for_status()

        provider = _build_hass_state_snapshot_provider(app.state)
        snapshot = client.portal.call(provider)

    assert [zone.zone_id for zone in snapshot.zones] == ["z1"]
    assert snapshot.zones[0].zone_type == "alert_zone"
    assert snapshot.zones[0].occupied is False
    assert snapshot.system.system_health in {"ok", "degraded", "error"}
    assert snapshot.system.active_track_count == 0
    assert snapshot.nodes == ()


def test_zone_crud_requests_a_reconcile_on_the_bound_bridge(monkeypatch, tmp_path: Path) -> None:
    """The CRUD hook must be reachable through `app.state`, not just in theory."""
    _configure_env(monkeypatch, tmp_path)

    class _RecordingBridge:
        def __init__(self) -> None:
            self.reconcile_calls = 0

        def request_reconcile(self) -> None:
            self.reconcile_calls += 1

    bridge = _RecordingBridge()
    with TestClient(app) as client:
        app.state.hass_bridge = bridge
        try:
            client.put(
                "/api/v1/zones/z1",
                json={
                    "id": "z1",
                    "name": "Zone One",
                    "zone_type": "alert_zone",
                    "polygon_geo": [[37.0, -122.0], [37.001, -122.0], [37.001, -121.999]],
                },
            ).raise_for_status()
            assert bridge.reconcile_calls == 1

            client.delete("/api/v1/zones/z1").raise_for_status()
            assert bridge.reconcile_calls == 2
        finally:
            del app.state.hass_bridge


def test_zone_crud_works_with_no_bridge_bound(monkeypatch, tmp_path: Path) -> None:
    """The api-only role never builds a bridge, so `app.state` has no attribute.

    Simulated by removing it: the CRUD hook is a `getattr(..., None)` guard, and
    a missing bridge must not turn a zone write into a 500.
    """
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        del app.state.hass_bridge
        resp = client.put(
            "/api/v1/zones/z1",
            json={
                "id": "z1",
                "name": "Zone One",
                "zone_type": "alert_zone",
                "polygon_geo": [[37.0, -122.0], [37.001, -122.0], [37.001, -121.999]],
            },
        )
        assert resp.status_code == 200
        # And the status route falls back to the settings-only view.
        body = client.get(STATUS_URL).json()
        assert body["connection_state"] == "disabled"
        assert body["transport"] is None


def test_an_unconfigured_bridge_is_still_bound_and_reports_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    """The combined lifespan always builds the bridge so the status route and the
    CRUD reconcile hooks need no special-casing; `start()` is the no-op."""
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        bridge = app.state.hass_bridge
        assert bridge.enabled is False
        assert bridge.connection_state == "disabled"
        assert client.get(STATUS_URL).json()["connection_state"] == "disabled"


def test_the_bridge_is_teed_into_the_live_event_hub(monkeypatch, tmp_path: Path) -> None:
    """Without this subscription the bridge never sees a detection at all, and
    nothing else in the system would report the omission."""
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        assert "hass_bridge" in app.state.live_hub.subscriber_names()

        bridge = app.state.hass_bridge
        client.portal.call(
            app.state.live_hub.broadcast,
            {"type": "detection", "detection": {"id": "d1", "label": "gunshot"}},
        )
        # Delivered synchronously into the bridge's inbound queue, and the tee
        # must never have raised into broadcast()'s caller.
        assert app.state.live_hub.subscriber_error_count("hass_bridge") == 0
        assert bridge.metrics.live_events_dropped == 0


# -- live event --------------------------------------------------------------


@pytest.mark.asyncio
async def test_hass_status_live_event_carries_both_type_and_event_type(
    tmp_path: Path, monkeypatch
) -> None:
    """The Leptos LiveEvent enum is `#[serde(tag = "type")]`; an event without
    "type" fails to deserialize and is dropped entirely."""
    events: list[dict] = []

    async def live_callback(payload: dict) -> None:
        events.append(payload)

    bridge, _ = build_test_bridge(tmp_path, monkeypatch, live_callback=live_callback)
    bridge.set_state_snapshot_provider(static_snapshot_provider(snapshot(zones=(zone("z1"),))))
    await bridge._connect_once()

    assert events, "connecting must broadcast at least one status event"
    for event in events:
        assert event["type"] == "hass_status"
        assert event["event_type"] == "hass_status"
        assert event["status"]["connection_state"] in {"connecting", "connected"}
    await bridge.stop()
