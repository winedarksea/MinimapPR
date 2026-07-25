"""End-to-end MQTT publishing against a real broker. **Skipped by default.**

Everything else in the HA test suite substitutes a fake at the
``bridge._build_transport`` seam, which proves our logic but never exercises
``aiomqtt`` or a broker's actual retain/LWT semantics. This file closes that gap
when a broker is available:

    brew install mosquitto && mosquitto -p 1883 &
    pip install -e '.[hass]'
    MINIMAPPR_HASS_LIVE_BROKER_TEST=1 .venv/bin/python -m pytest \\
        tests/test_hass_broker_integration.py -q

Host/port come from ``MINIMAPPR_HASS_LIVE_BROKER_HOST`` / ``_PORT``
(default 127.0.0.1:1883).

This still does **not** validate Home Assistant compatibility — only that our
messages reach a broker with the retain and LWT flags we intend. HA-side
verification requires a manual smoke test; see
``tests/test_hass_discovery_payloads.py`` and
``docs/home_assistant_integration.md``.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MINIMAPPR_HASS_LIVE_BROKER_TEST") != "1",
    reason="set MINIMAPPR_HASS_LIVE_BROKER_TEST=1 with a reachable MQTT broker to run",
)

aiomqtt = pytest.importorskip("aiomqtt")

from minimappr.core.hass.aiomqtt_transport import AiomqttTransport  # noqa: E402
from minimappr.core.hass.bridge import HassBridge, HassBridgeConfig  # noqa: E402
from minimappr.core.hass.transport import MqttPublish, MqttTransportConfig, MqttWill  # noqa: E402
from tests.hass_helpers import node, snapshot, static_snapshot_provider, zone  # noqa: E402

BROKER_HOST = os.environ.get("MINIMAPPR_HASS_LIVE_BROKER_HOST", "127.0.0.1")
BROKER_PORT = int(os.environ.get("MINIMAPPR_HASS_LIVE_BROKER_PORT", "1883"))
_COLLECT_TIMEOUT_S = 5.0


def _unique_topics() -> tuple[str, str]:
    """Namespace each run so a shared broker's retained messages cannot leak
    between test runs and produce false passes."""
    suffix = uuid.uuid4().hex[:8]
    return f"ha_test_{suffix}", f"mm_test_{suffix}"


async def _collect_retained(pattern: str, *, expected: int, timeout_s: float = _COLLECT_TIMEOUT_S) -> dict[str, str]:
    """Subscribe as a fresh client and gather what the broker replays."""
    received: dict[str, str] = {}

    async def _reader() -> None:
        async with aiomqtt.Client(hostname=BROKER_HOST, port=BROKER_PORT) as client:
            await client.subscribe(pattern)
            async for message in client.messages:
                received[str(message.topic)] = message.payload.decode("utf-8")
                if len(received) >= expected:
                    return

    # The timeout is the normal exit for "fewer than `expected` retained
    # messages" — callers assert on what actually arrived.
    try:
        await asyncio.wait_for(_reader(), timeout=timeout_s)
    except (asyncio.TimeoutError, TimeoutError):
        pass
    return received


async def _purge(pattern_root: str) -> None:
    """Leave the shared broker clean regardless of assertion outcome."""
    retained = await _collect_retained(f"{pattern_root}/#", expected=10_000, timeout_s=1.5)
    if not retained:
        return
    async with aiomqtt.Client(hostname=BROKER_HOST, port=BROKER_PORT) as client:
        for topic in retained:
            await client.publish(topic, payload=b"", retain=True)


@pytest.mark.asyncio
async def test_real_transport_connects_and_publishes_retained() -> None:
    _, base_topic = _unique_topics()
    transport = AiomqttTransport(
        MqttTransportConfig(host=BROKER_HOST, port=BROKER_PORT, client_id=f"mm-{uuid.uuid4().hex[:6]}")
    )
    await transport.connect()
    try:
        await transport.publish(
            MqttPublish(topic=f"{base_topic}/system/health", payload="ok", retain=True)
        )
    finally:
        await transport.disconnect()

    retained = await _collect_retained(f"{base_topic}/#", expected=1)
    assert retained.get(f"{base_topic}/system/health") == "ok"
    await _purge(base_topic)


@pytest.mark.asyncio
async def test_discovery_configs_land_retained_on_the_broker(tmp_path: Path) -> None:
    discovery_prefix, base_topic = _unique_topics()
    bridge = HassBridge(
        config=HassBridgeConfig(
            enabled=True,
            mqtt_host=BROKER_HOST,
            mqtt_port=BROKER_PORT,
            mqtt_client_id=f"mm-{uuid.uuid4().hex[:6]}",
            discovery_prefix=discovery_prefix,
            base_topic=base_topic,
            publish_interval_seconds=3600.0,
            publish_min_interval_seconds=0.0,
            reconcile_interval_seconds=3600.0,
            discovery_ledger_path=tmp_path / "ledger.json",
            version="live-test",
        )
    )
    bridge.set_state_snapshot_provider(
        static_snapshot_provider(snapshot(zones=(zone("z1", occupied=True),), nodes=(node("n1"),)))
    )
    try:
        assert await bridge._connect_once() is True
        assert bridge.connection_state == "connected"

        configs = await _collect_retained(f"{discovery_prefix}/#", expected=2)
        assert any("zone_occupancy_z1/config" in topic for topic in configs)
        assert any("node_online_n1/config" in topic for topic in configs)

        state = await _collect_retained(f"{base_topic}/#", expected=2)
        assert state.get(f"{base_topic}/status") == "online"
        assert state.get(f"{base_topic}/zone/z1/occupancy") == "ON"
    finally:
        await bridge.stop()
        await _purge(discovery_prefix)
        await _purge(base_topic)


@pytest.mark.asyncio
async def test_graceful_stop_publishes_offline(tmp_path: Path) -> None:
    _, base_topic = _unique_topics()
    bridge = HassBridge(
        config=HassBridgeConfig(
            enabled=True,
            mqtt_host=BROKER_HOST,
            mqtt_port=BROKER_PORT,
            base_topic=base_topic,
            publish_interval_seconds=3600.0,
            reconcile_interval_seconds=3600.0,
            discovery_ledger_path=tmp_path / "ledger.json",
        )
    )
    await bridge._connect_once()
    await bridge.stop()

    retained = await _collect_retained(f"{base_topic}/status", expected=1)
    assert retained.get(f"{base_topic}/status") == "offline"
    await _purge(base_topic)


@pytest.mark.asyncio
async def test_last_will_fires_when_the_client_vanishes(tmp_path: Path) -> None:
    """Drop the socket without a graceful disconnect: the broker must publish
    the LWT, which is what makes HA mark the device unavailable on a crash."""
    _, base_topic = _unique_topics()
    will_topic = f"{base_topic}/status"
    transport = AiomqttTransport(
        MqttTransportConfig(
            host=BROKER_HOST,
            port=BROKER_PORT,
            keepalive_seconds=1,
            will=MqttWill(topic=will_topic, payload="offline", retain=True),
        )
    )
    await transport.connect()
    await transport.publish(MqttPublish(topic=will_topic, payload="online", retain=True))

    # Kill the underlying socket so the broker sees an unclean disconnect.
    client = transport._client  # noqa: SLF001 - deliberately simulating a crash
    client._client.socket().close()  # type: ignore[union-attr]

    retained: dict[str, str] = {}
    for _ in range(20):
        await asyncio.sleep(0.5)
        retained = await _collect_retained(will_topic, expected=1, timeout_s=1.0)
        if retained.get(will_topic) == "offline":
            break
    assert retained.get(will_topic) == "offline", "broker did not deliver the LWT"
    await _purge(base_topic)


@pytest.mark.asyncio
async def test_purge_removes_retained_configs_from_the_broker(tmp_path: Path) -> None:
    discovery_prefix, base_topic = _unique_topics()
    bridge = HassBridge(
        config=HassBridgeConfig(
            enabled=True,
            mqtt_host=BROKER_HOST,
            mqtt_port=BROKER_PORT,
            discovery_prefix=discovery_prefix,
            base_topic=base_topic,
            publish_interval_seconds=3600.0,
            publish_min_interval_seconds=0.0,
            reconcile_interval_seconds=3600.0,
            discovery_ledger_path=tmp_path / "ledger.json",
        )
    )
    bridge.set_state_snapshot_provider(
        static_snapshot_provider(snapshot(zones=(zone("z1"),)))
    )
    try:
        await bridge._connect_once()
        assert await _collect_retained(f"{discovery_prefix}/#", expected=1)

        await bridge.purge_discovery()
        await bridge._flush_outbound()

        assert await _collect_retained(f"{discovery_prefix}/#", expected=1, timeout_s=2.0) == {}
    finally:
        await bridge.stop()
        await _purge(discovery_prefix)
        await _purge(base_topic)
