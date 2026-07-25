"""HassBridge: the outbound Home Assistant MQTT publisher.

Structural template is ``EffectorManager`` (core/effectors/registry.py): an
optional subsystem that is fully dormant when unconfigured, a single named
background task whose per-iteration exceptions never kill the loop, and a
module-level ``_build_transport`` factory as the monkeypatch seam for tests.

The one deliberate divergence from ``EffectorManager._poll_loop``: this loop
publishes a full snapshot immediately after connecting rather than sleeping
first, because a freshly-connected HA needs state now, not one interval from now.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from minimappr.core.hass.discovery import PAYLOAD_NOT_AVAILABLE
from minimappr.core.hass.ledger import HassDiscoveryLedger
from minimappr.core.hass.models import HassBridgeConfig, HassBridgeMetrics
from minimappr.core.hass.outbound import OutboundQueue
from minimappr.core.hass.reconcile import reconcile_discovery
from minimappr.core.hass.state_mapper import HassStateMapper, HassStateSnapshot
from minimappr.core.hass.transport import (
    MqttPublish,
    MqttTransport,
    MqttTransportConfig,
    MqttWill,
)

logger = logging.getLogger(__name__)

LiveCallback = Callable[[dict], Awaitable[None]]
StateSnapshotProvider = Callable[[], Awaitable[HassStateSnapshot]]

ConnectionState = str  # disabled | disconnected | connecting | connected | error

_RECONNECT_JITTER_FRACTION = 0.2
_DROP_LOG_EVERY = 100


def _with_jitter(delay_seconds: float) -> float:
    """+/-20% jitter so a fleet of nodes does not stampede one broker in lockstep."""
    spread = delay_seconds * _RECONNECT_JITTER_FRACTION
    return max(0.0, delay_seconds + random.uniform(-spread, spread))


def _build_transport(config: HassBridgeConfig) -> MqttTransport | None:
    """Construct the MQTT client for this config, or None if unavailable.

    Module-level factory, mirroring ``effectors/registry.py:_build_driver`` — it
    is the monkeypatch seam every bridge test substitutes, which is what lets
    phases 1-7 be fully testable with no MQTT client installed at all.
    """
    from minimappr.core.hass.aiomqtt_transport import AiomqttTransport, aiomqtt_available

    if not aiomqtt_available():
        logger.warning(
            "hass bridge enabled but the aiomqtt package is not installed; "
            "install the 'hass' extra to enable MQTT publishing"
        )
        return None
    return AiomqttTransport(
        MqttTransportConfig(
            host=config.mqtt_host,
            port=config.mqtt_port,
            username=config.mqtt_username,
            password=config.mqtt_password,
            client_id=config.mqtt_client_id,
            keepalive_seconds=config.mqtt_keepalive_seconds,
            tls_enabled=config.mqtt_tls_enabled,
            tls_insecure=config.mqtt_tls_insecure,
            will=MqttWill(
                topic=f"{config.base_topic.strip('/')}/status",
                payload=PAYLOAD_NOT_AVAILABLE,
                retain=True,
            ),
        )
    )


class HassBridge:
    def __init__(
        self,
        *,
        config: HassBridgeConfig,
        live_callback: LiveCallback | None = None,
    ) -> None:
        self._config = config
        self._live_callback = live_callback
        self._mapper = HassStateMapper(config)
        self._ledger = HassDiscoveryLedger(config.discovery_ledger_path)
        self._inbound: asyncio.Queue[dict] = asyncio.Queue(maxsize=config.queue_size)
        self._outbound = OutboundQueue(
            maxsize=config.queue_size,
            min_interval_seconds=config.publish_min_interval_seconds,
        )
        self._metrics = HassBridgeMetrics()
        self._transport: MqttTransport | None = None
        self._task: asyncio.Task[None] | None = None
        self._snapshot_provider: StateSnapshotProvider | None = None
        self._connection_state: ConnectionState = "disabled" if not config.enabled else "disconnected"
        self._connected_since_ns: int | None = None
        self._last_connect_error: str | None = None
        self._last_publish_ns: int | None = None
        self._last_reconcile_ns: int | None = None
        self._reconcile_requested = True

    # -- accessors ----------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def config(self) -> HassBridgeConfig:
        return self._config

    @property
    def metrics(self) -> HassBridgeMetrics:
        return self._sync_metrics()

    @property
    def connection_state(self) -> ConnectionState:
        return self._connection_state

    @property
    def transport_name(self) -> str | None:
        return self._transport.name if self._transport is not None else None

    @property
    def mapper(self) -> HassStateMapper:
        """Exposed for the rules handler's topic guard, which needs the same
        ``base_topic`` normalization the bridge publishes under."""
        return self._mapper

    def status(self) -> dict[str, object]:
        from minimappr.core.hass.aiomqtt_transport import aiomqtt_available

        return {
            "enabled": self._config.enabled,
            "connection_state": self._connection_state,
            "transport": self.transport_name,
            "transport_available": aiomqtt_available(),
            "mqtt_host": self._config.mqtt_host,
            "mqtt_port": self._config.mqtt_port,
            "mqtt_tls_enabled": self._config.mqtt_tls_enabled,
            "discovery_prefix": self._config.discovery_prefix,
            "base_topic": self._config.base_topic,
            "device_id": self._config.device_id,
            "connected_since_ns": self._connected_since_ns,
            "last_connect_error": self._last_connect_error,
            "last_publish_ns": self._last_publish_ns,
            "last_reconcile_ns": self._last_reconcile_ns,
            "queue_depth": self._outbound.depth,
            "queue_capacity": self._outbound.capacity,
            "discovery_entity_count": len(self._ledger.entries()),
            "published_state_topic_count": self._outbound.published_state_topic_count,
            "metrics": self._sync_metrics().as_dict(),
        }

    def set_state_snapshot_provider(self, provider: StateSnapshotProvider | None) -> None:
        """Late-bound, mirroring ``EffectorManager.set_target_zone_resolver``:
        the provider needs runtime state that does not exist at build time."""
        self._snapshot_provider = provider

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        if not self._config.enabled:
            # Fully dormant: no task, no queues drained, no ledger touched.
            self._set_connection_state("disabled")
            return
        self._ledger.load()
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._publish_loop(), name="hass-publisher")

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        transport = self._transport
        if transport is not None:
            # Graceful shutdown: say offline explicitly rather than relying on the
            # LWT, which only fires on an *ungraceful* disconnect.
            with contextlib.suppress(Exception):
                await transport.publish(self._mapper.availability(online=False))
            with contextlib.suppress(Exception):
                await transport.disconnect()
        self._transport = None
        self._set_connection_state("disabled" if not self._config.enabled else "disconnected")

    # -- ingest -------------------------------------------------------------

    def handle_live_event(self, payload: dict) -> None:
        """The ``LiveEventHub`` tee sink. Synchronous, non-blocking, non-raising."""
        try:
            self._inbound.put_nowait(payload)
        except asyncio.QueueFull:
            self._count_live_event_drop()

    def enqueue(self, message: MqttPublish) -> bool:
        """Queue one message. False means it was dropped (queue full)."""
        return self._outbound.enqueue(message)

    def enqueue_rule_action(self, message: MqttPublish) -> bool:
        """Queue a rule-authored publish. Synchronous: the caller is on the
        fusion hot path (see ``core/hass/rules_handler.py``)."""
        accepted = self.enqueue(message)
        if accepted:
            self._metrics.rule_actions_queued += 1
        else:
            self._metrics.rule_actions_rejected += 1
        return accepted

    def request_reconcile(self) -> None:
        """Synchronous so zone/node CRUD routes can call it without awaiting."""
        self._reconcile_requested = True

    def forget_published_state(self) -> None:
        """Drop the dedupe cache and the ledger's digests so the next reconcile
        republishes everything. Needed after retained messages are cleared on the
        broker out-of-band, which we have no way to observe."""
        self._outbound.forget_published()
        self._ledger.invalidate_digests()

    async def purge_discovery(self) -> int:
        """Blank every retained topic we have ever published, and clear the ledger.

        Run before uninstalling, otherwise HA keeps the entities forever as
        permanently-unavailable rows in its registry.

        The availability topic is blanked too, so the broker is left holding
        nothing of ours. That means a still-running bridge shows as unavailable in
        HA until its next reconnect — acceptable, because purge is explicitly a
        pre-uninstall action, not routine maintenance.
        """
        removed = 0
        for entry in self._ledger.entries().values():
            for message in self._mapper.removal_publishes(
                config_topic=entry.config_topic, state_topics=entry.state_topics
            ):
                if self.enqueue(message):
                    removed += 1
        if self.enqueue(
            MqttPublish(topic=self._mapper.topics.availability, payload="", retain=True)
        ):
            removed += 1
        self._ledger.clear()
        self._ledger.save()
        self._outbound.clear_published_state_topics()
        self._outbound.forget_published()
        return removed

    # -- publish loop -------------------------------------------------------

    async def _publish_loop(self) -> None:
        backoff = self._config.reconnect_backoff_initial_seconds
        while True:
            if self._transport is None:
                connected = await self._connect_once()
                if not connected:
                    # Keep draining and polling into _outbound while offline;
                    # coalescing means an hour down costs one message per topic.
                    with contextlib.suppress(Exception):
                        await self._collect_pending()
                    await asyncio.sleep(_with_jitter(backoff))
                    backoff = min(backoff * 2.0, self._config.reconnect_backoff_max_seconds)
                    continue
                # _connect_once() already published a full snapshot, so fall
                # through to the sleep rather than running a second cycle now.
                backoff = self._config.reconnect_backoff_initial_seconds
                await asyncio.sleep(self._config.publish_interval_seconds)
                continue

            try:
                await self._publish_cycle_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                # One bad cycle must never kill the publisher.
                logger.warning("hass publish cycle failed", exc_info=True)
            await asyncio.sleep(self._config.publish_interval_seconds)

    async def _connect_once(self) -> bool:
        self._set_connection_state("connecting")
        transport = _build_transport(self._config)
        if transport is None:
            self._last_connect_error = "no MQTT transport available (install the 'hass' extra)"
            self._set_connection_state("error")
            return False
        try:
            await transport.connect()
        except Exception as exc:
            self._last_connect_error = f"{type(exc).__name__}: {exc}"
            self._metrics.reconnect_count += 1
            self._set_connection_state("error")
            return False

        self._transport = transport
        self._last_connect_error = None
        self._connected_since_ns = time.time_ns()
        self._metrics.reconnect_count += 1
        # A broker restart may have lost every retained message, so nothing about
        # what we sent before this connection can be assumed to still be there.
        self._outbound.forget_published()
        self._set_connection_state("connected")

        self.enqueue(self._mapper.availability(online=True))
        self._reconcile_requested = True
        await self._publish_cycle_once()
        return True

    async def _publish_cycle_once(self) -> None:
        """One full cycle: drain the tee, poll state, reconcile, flush.

        Exposed (underscore-private but stable) so tests can drive cycles
        deterministically instead of waiting on ``publish_interval_seconds``.
        """
        await self._collect_pending()
        await self._flush_outbound()

    async def _collect_pending(self) -> None:
        self._drain_inbound()
        snapshot = await self._poll_snapshot()
        if snapshot is not None:
            if self._reconcile_requested or self._reconcile_is_due():
                await self._reconcile(snapshot)
            for message in self._mapper.snapshot_publishes(snapshot, now_ns=time.time_ns()):
                self.enqueue(message)

    def _drain_inbound(self) -> None:
        """Interpret tee'd live events off the hot path."""
        while True:
            try:
                payload = self._inbound.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                self._handle_inbound_payload(payload)
            except Exception:
                logger.debug("hass bridge could not map live event", exc_info=True)

    def _handle_inbound_payload(self, payload: dict) -> None:
        event_type = str(payload.get("type") or payload.get("event_type") or "")
        if event_type == "detection":
            detection = payload.get("detection")
            if isinstance(detection, dict):
                for message in self._mapper.observe_detection(detection, now_ns=time.time_ns()):
                    self.enqueue(message)
        elif event_type == "alert":
            # Alerts arrive either flattened or nested depending on the emitter.
            alert = payload.get("alert") if isinstance(payload.get("alert"), dict) else payload
            for message in self._mapper.alert_publish(alert):
                self.enqueue(message)
        elif event_type in ("zone_updated", "zone_deleted", "node_updated", "node_deleted"):
            self._reconcile_requested = True

    async def _poll_snapshot(self) -> "HassStateSnapshot | None":
        """Zone occupancy, node health and metrics are never broadcast, so they
        are polled rather than tee'd."""
        provider = self._snapshot_provider
        if provider is None:
            return None
        try:
            return await provider()
        except Exception:
            logger.warning("hass state snapshot provider failed", exc_info=True)
            return None

    def _reconcile_is_due(self) -> bool:
        if self._last_reconcile_ns is None:
            return True
        elapsed_s = (time.time_ns() - self._last_reconcile_ns) / 1e9
        return elapsed_s >= self._config.reconcile_interval_seconds

    async def _reconcile(self, snapshot: HassStateSnapshot) -> None:
        self._reconcile_requested = False
        reconcile_discovery(
            mapper=self._mapper,
            ledger=self._ledger,
            enqueue=self.enqueue,
            zones=snapshot.zones,
            nodes=snapshot.nodes,
        )
        self._last_reconcile_ns = time.time_ns()
        self._metrics.reconcile_count += 1
        await self._broadcast_status()

    async def _flush_outbound(self) -> None:
        """Publish everything ``OutboundQueue`` says is ready this cycle.

        Stops at the first failure: ``_publish_one`` has already torn the session
        down by then, so the remaining messages belong to the next connection.
        """
        for message in self._outbound.collect_ready(now_ns=time.time_ns()):
            if not await self._publish_one(message):
                return

    async def _publish_one(self, message: MqttPublish) -> bool:
        transport = self._transport
        if transport is None:
            self.enqueue(message)
            return False
        try:
            await transport.publish(message)
        except Exception as exc:
            self._metrics.messages_failed += 1
            self._last_connect_error = f"publish failed: {type(exc).__name__}: {exc}"
            # Treat a publish failure as a lost session: the loop reconnects and
            # republishes from a cleared dedupe cache rather than assuming the
            # broker still holds what we thought we sent.
            with contextlib.suppress(Exception):
                await transport.disconnect()
            self._transport = None
            self._connected_since_ns = None
            self._set_connection_state("error")
            return False

        self._metrics.messages_published += 1
        self._last_publish_ns = time.time_ns()
        self._outbound.record_published(message, now_ns=self._last_publish_ns)
        return True

    # -- status -------------------------------------------------------------

    def _set_connection_state(self, state: ConnectionState) -> None:
        if state == self._connection_state:
            return
        self._connection_state = state
        if state != "connected":
            self._connected_since_ns = None
        # Broadcast on transition only; a per-cycle status event would be noise.
        if self._live_callback is not None:
            with contextlib.suppress(RuntimeError):
                asyncio.get_running_loop().create_task(
                    self._broadcast_status(), name="hass-status-broadcast"
                )

    async def _broadcast_status(self) -> None:
        if self._live_callback is None:
            return
        payload = {
            # Both keys: the frontend LiveEvent enum is tagged on "type", while
            # older consumers and the REST-shaped paths read "event_type".
            "type": "hass_status",
            "event_type": "hass_status",
            "status": self.status(),
        }
        try:
            await self._live_callback(payload)
        except Exception:
            logger.debug("hass status broadcast failed", exc_info=True)

    def _count_live_event_drop(self) -> None:
        """Drop-newest on a full inbound queue.

        A dropped detection is a lost impulse, not self-healing state, so the
        counter is surfaced in the status endpoint rather than only logged.
        """
        self._metrics.live_events_dropped += 1
        total = self._metrics.live_events_dropped
        if total == 1 or total % _DROP_LOG_EVERY == 0:
            logger.warning("hass bridge dropped a live event (queue full, %d total)", total)

    def _sync_metrics(self) -> HassBridgeMetrics:
        """Fold the queue's own counters into the reported metrics.

        ``OutboundQueue`` owns coalescing/dedupe/drop accounting because it is
        where those decisions are made; the bridge is just the reporting surface.
        """
        self._metrics.messages_dropped_queue_full = self._outbound.dropped_count
        self._metrics.messages_coalesced = self._outbound.coalesced_count
        self._metrics.messages_suppressed_unchanged = self._outbound.suppressed_unchanged_count
        return self._metrics
