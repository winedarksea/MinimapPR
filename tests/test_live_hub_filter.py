from __future__ import annotations

import asyncio
import json

import pytest

from minimappr.api.live import LiveEventHub, LiveSubscriptionFilter


def test_node_state_events_bypass_content_filters() -> None:
    """A subscriber with content filters must still receive node-state events.

    node_updated / node_capability_status payloads carry no detection or track, so
    the content filters (categories/labels/zones/statuses) do not apply to them.
    """
    flt = LiveSubscriptionFilter(label_categories={"vehicle"}, labels={"truck"})

    node_updated = {"type": "node_updated", "node": {"id": "node-1"}}
    node_capability = {
        "type": "node_capability_status",
        "node_id": "node-1",
        "capability": "ptz_camera",
        "status": {"state": "idle"},
    }

    assert LiveEventHub._matches_filter(node_updated, flt) is True
    assert LiveEventHub._matches_filter(node_capability, flt) is True


def test_detection_events_still_filtered() -> None:
    """Content filters still gate detection/track payloads."""
    flt = LiveSubscriptionFilter(label_categories={"vehicle"})

    matching = {"detection": {"label_category": "vehicle"}}
    non_matching = {"detection": {"label_category": "animal"}}

    assert LiveEventHub._matches_filter(matching, flt) is True
    assert LiveEventHub._matches_filter(non_matching, flt) is False


def test_no_filter_delivers_everything() -> None:
    flt = LiveSubscriptionFilter()
    assert LiveEventHub._matches_filter({"detection": {"label_category": "x"}}, flt) is True
    assert LiveEventHub._matches_filter({"type": "node_updated"}, flt) is True


class _FakeWebSocket:
    """Minimal stand-in for the send side of a starlette WebSocket."""

    def __init__(self, *, fail: bool = False, block: asyncio.Event | None = None) -> None:
        self.sent: list[str] = []
        self._fail = fail
        self._block = block

    async def send_text(self, text: str) -> None:
        if self._block is not None:
            await self._block.wait()
        if self._fail:
            raise RuntimeError("client gone")
        self.sent.append(text)


async def _register(hub: LiveEventHub, client, flt: LiveSubscriptionFilter | None = None) -> None:
    async with hub._lock:
        hub._clients.add(client)
        hub._filters[client] = flt or LiveSubscriptionFilter()


@pytest.mark.asyncio
async def test_broadcast_sends_starlette_compatible_json_to_matching_clients() -> None:
    hub = LiveEventHub()
    client = _FakeWebSocket()
    await _register(hub, client)

    payload = {"type": "detection", "detection": {"label_category": "vehicle", "note": "café"}}
    await hub.broadcast(payload)

    # Must match starlette's WebSocket.send_json wire format exactly.
    assert client.sent == [json.dumps(payload, separators=(",", ":"), ensure_ascii=False)]
    assert json.loads(client.sent[0]) == payload


@pytest.mark.asyncio
async def test_broadcast_skips_filtered_out_clients() -> None:
    hub = LiveEventHub()
    matching = _FakeWebSocket()
    filtered = _FakeWebSocket()
    await _register(hub, matching, LiveSubscriptionFilter(label_categories={"vehicle"}))
    await _register(hub, filtered, LiveSubscriptionFilter(label_categories={"animal"}))

    await hub.broadcast({"detection": {"label_category": "vehicle"}})

    assert len(matching.sent) == 1
    assert filtered.sent == []


@pytest.mark.asyncio
async def test_broadcast_evicts_failing_client_without_dropping_others() -> None:
    hub = LiveEventHub()
    healthy = _FakeWebSocket()
    broken = _FakeWebSocket(fail=True)
    await _register(hub, healthy)
    await _register(hub, broken)

    await hub.broadcast({"type": "node_updated"})

    assert len(healthy.sent) == 1
    async with hub._lock:
        assert broken not in hub._clients
        assert healthy in hub._clients


@pytest.mark.asyncio
async def test_broadcast_dispatches_subscribers_even_with_no_clients() -> None:
    hub = LiveEventHub()
    seen: list[dict] = []
    hub.subscribe("sink", seen.append)

    await hub.broadcast({"type": "node_updated"})

    assert seen == [{"type": "node_updated"}]


@pytest.mark.asyncio
async def test_broadcast_does_not_serialize_a_stalled_client_ahead_of_others() -> None:
    """A blocked client must not delay delivery to healthy peers."""
    hub = LiveEventHub()
    gate = asyncio.Event()
    stalled = _FakeWebSocket(block=gate)
    healthy = _FakeWebSocket()
    await _register(hub, stalled)
    await _register(hub, healthy)

    task = asyncio.create_task(hub.broadcast({"type": "node_updated"}))
    # The healthy client is served while the stalled one is still blocked.
    for _ in range(10):
        await asyncio.sleep(0)
        if healthy.sent:
            break
    assert len(healthy.sent) == 1

    gate.set()
    await task
