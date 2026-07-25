"""Shared fixtures for the Home Assistant bridge tests (AGENTS §3).

Used by test_hass_bridge, test_hass_bridge_reconcile, test_hass_rules_handler,
and test_hass_api — the recorder and the builder were identical in all four.

``build_test_bridge`` neuters the timing knobs the same way the effector tests
neuter ``status_poll_interval_seconds``: the interval is pushed far out of reach
and cycles are driven explicitly via ``await bridge._publish_cycle_once()``, so
no test ever sleeps or races the loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from minimappr.core.hass import bridge as bridge_module
from minimappr.core.hass.bridge import HassBridge, HassBridgeConfig
from minimappr.core.hass.state_mapper import (
    HassStateSnapshot,
    NodeStateInput,
    SystemStateInput,
    ZoneStateInput,
)
from minimappr.core.hass.transport import MqttPublish, MqttTransportConfig, MqttTransportError


class RecordingMqttTransport:
    """In-memory transport that records publishes and simulates broker retain.

    Duck-typed against ``MqttTransport`` rather than subclassing the Protocol, so
    a drift in the Protocol surfaces as a real test failure instead of being
    papered over by inheritance.
    """

    def __init__(
        self,
        config: MqttTransportConfig | None = None,
        *,
        fail_connect_times: int = 0,
        fail_publish_on: set[str] | None = None,
    ) -> None:
        self.config = config
        self.published: list[MqttPublish] = []
        self.connect_calls = 0
        self.disconnect_calls = 0
        self._fail_connect_times = fail_connect_times
        self._fail_publish_on = fail_publish_on or set()
        self._retained: dict[str, str] = {}

    @property
    def name(self) -> str:
        return "recording"

    @property
    def will(self):
        return self.config.will if self.config is not None else None

    async def connect(self) -> None:
        self.connect_calls += 1
        if self._fail_connect_times > 0:
            self._fail_connect_times -= 1
            raise MqttTransportError("simulated connect failure")

    async def publish(self, message: MqttPublish) -> None:
        if message.topic in self._fail_publish_on:
            raise MqttTransportError(f"simulated publish failure for {message.topic}")
        self.published.append(message)
        if message.retain:
            # An empty retained payload is a broker-side delete.
            if message.payload:
                self._retained[message.topic] = message.payload
            else:
                self._retained.pop(message.topic, None)

    async def disconnect(self) -> None:
        self.disconnect_calls += 1

    # -- assertions helpers -------------------------------------------------

    def retained(self) -> dict[str, str]:
        """What a fresh subscriber would receive on connect."""
        return dict(self._retained)

    def topics(self) -> list[str]:
        return [message.topic for message in self.published]

    def payload_for(self, topic: str) -> str | None:
        for message in reversed(self.published):
            if message.topic == topic:
                return message.payload
        return None

    def count_for(self, topic: str) -> int:
        return sum(1 for message in self.published if message.topic == topic)

    def clear(self) -> None:
        self.published.clear()


def hass_test_config(tmp_path: Path, **overrides: Any) -> HassBridgeConfig:
    base: dict[str, Any] = {
        "enabled": True,
        "mqtt_host": "broker.test",
        # Far out of reach: tests drive cycles explicitly.
        "publish_interval_seconds": 3600.0,
        "publish_min_interval_seconds": 0.0,
        "reconcile_interval_seconds": 3600.0,
        "detection_classes": ("security", "gunshot"),
        "discovery_ledger_path": tmp_path / "hass_discovery_ledger.json",
        "version": "test",
    }
    base.update(overrides)
    return HassBridgeConfig(**base)


def build_test_bridge(
    tmp_path: Path,
    monkeypatch: Any,
    *,
    transport: RecordingMqttTransport | None = None,
    live_callback: Any = None,
    **overrides: Any,
) -> tuple[HassBridge, RecordingMqttTransport]:
    """Build a bridge whose transport is a recorder, via the _build_transport seam."""
    recorder = transport if transport is not None else RecordingMqttTransport()

    def _fake_build_transport(config: HassBridgeConfig):
        recorder.config = MqttTransportConfig(
            host=config.mqtt_host,
            port=config.mqtt_port,
            username=config.mqtt_username,
            password=config.mqtt_password,
            client_id=config.mqtt_client_id,
            keepalive_seconds=config.mqtt_keepalive_seconds,
            tls_enabled=config.mqtt_tls_enabled,
            tls_insecure=config.mqtt_tls_insecure,
            will=bridge_module.MqttWill(
                topic=f"{config.base_topic.strip('/')}/status",
                payload="offline",
                retain=True,
            ),
        )
        return recorder

    monkeypatch.setattr(bridge_module, "_build_transport", _fake_build_transport)
    bridge = HassBridge(config=hass_test_config(tmp_path, **overrides), live_callback=live_callback)
    return bridge, recorder


def snapshot(
    *,
    zones: tuple[ZoneStateInput, ...] = (),
    nodes: tuple[NodeStateInput, ...] = (),
    system: SystemStateInput | None = None,
    tracks: tuple = (),
) -> HassStateSnapshot:
    return HassStateSnapshot(
        zones=zones,
        nodes=nodes,
        system=system if system is not None else SystemStateInput(system_health="ok"),
        tracks=tracks,
    )


def static_snapshot_provider(value: HassStateSnapshot):
    async def provider() -> HassStateSnapshot:
        return value

    return provider


def zone(zone_id: str, *, occupied: bool = False, zone_type: str = "alert_zone") -> ZoneStateInput:
    return ZoneStateInput(
        zone_id=zone_id,
        zone_name=zone_id.replace("_", " ").title(),
        zone_type=zone_type,
        occupied=occupied,
    )


def node(node_id: str, *, health_status: str = "online") -> NodeStateInput:
    return NodeStateInput(node_id=node_id, node_name=node_id, health_status=health_status)
