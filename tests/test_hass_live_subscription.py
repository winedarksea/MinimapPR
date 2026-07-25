"""LiveEventHub.subscribe() — the synchronous fan-out tee the HA bridge feeds on.

These tests guard the hot-path contract: the tee must not await, must not be
blocked by websocket clients, and must not let a broken subscriber escape into
``broadcast()``'s callers (which are on the fusion pipeline).
"""

from __future__ import annotations

import asyncio

import pytest

from minimappr.api.live import LiveEventHub


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def accept(self) -> None:
        return None

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


@pytest.mark.asyncio
async def test_subscriber_sees_every_payload() -> None:
    hub = LiveEventHub()
    seen: list[dict] = []
    hub.subscribe("sink", seen.append)

    await hub.broadcast({"type": "detection", "n": 1})
    await hub.broadcast({"type": "track", "n": 2})

    assert [item["n"] for item in seen] == [1, 2]


@pytest.mark.asyncio
async def test_subscriber_receives_events_with_no_websocket_clients() -> None:
    """broadcast() returns early when there are no clients; the tee runs first."""
    hub = LiveEventHub()
    seen: list[dict] = []
    hub.subscribe("sink", seen.append)

    await hub.broadcast({"type": "node_updated"})

    assert len(seen) == 1


@pytest.mark.asyncio
async def test_subscribers_bypass_per_client_filters() -> None:
    """Filters are a per-websocket concern; a tee gets the raw firehose."""
    hub = LiveEventHub()
    socket = _FakeWebSocket()
    await hub.connect(socket)  # type: ignore[arg-type]
    await hub.update_filter(socket, {"filter": {"labels": ["gunshot"]}})  # type: ignore[arg-type]
    seen: list[dict] = []
    hub.subscribe("sink", seen.append)

    await hub.broadcast({"type": "detection", "detection": {"label": "birdsong"}})

    assert socket.sent == [], "filter should have excluded the websocket client"
    assert len(seen) == 1, "the tee must still see the filtered-out event"


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery_and_is_idempotent() -> None:
    hub = LiveEventHub()
    seen: list[dict] = []
    unsubscribe = hub.subscribe("sink", seen.append)

    await hub.broadcast({"n": 1})
    unsubscribe()
    unsubscribe()
    await hub.broadcast({"n": 2})

    assert [item["n"] for item in seen] == [1]
    assert hub.subscriber_names() == []


@pytest.mark.asyncio
async def test_resubscribing_the_same_name_replaces_the_sink() -> None:
    hub = LiveEventHub()
    first: list[dict] = []
    second: list[dict] = []
    stale_unsubscribe = hub.subscribe("sink", first.append)
    hub.subscribe("sink", second.append)

    # The superseded handle must not remove the live subscriber.
    stale_unsubscribe()
    await hub.broadcast({"n": 1})

    assert first == []
    assert len(second) == 1
    assert hub.subscriber_names() == ["sink"]


@pytest.mark.asyncio
async def test_raising_subscriber_neither_escapes_nor_blocks_websocket_send() -> None:
    hub = LiveEventHub()
    socket = _FakeWebSocket()
    await hub.connect(socket)  # type: ignore[arg-type]

    def explode(_payload: dict) -> None:
        raise RuntimeError("subscriber is broken")

    hub.subscribe("bad", explode)
    good: list[dict] = []
    hub.subscribe("good", good.append)

    await hub.broadcast({"type": "detection"})

    assert socket.sent == [{"type": "detection"}], "WS delivery must be unaffected"
    assert len(good) == 1, "one bad subscriber must not starve the others"
    assert hub.subscriber_error_count("bad") == 1


@pytest.mark.asyncio
async def test_broken_subscriber_is_never_auto_removed() -> None:
    """Dropping the sink would silently kill a downstream feed — count instead."""
    hub = LiveEventHub()

    def explode(_payload: dict) -> None:
        raise RuntimeError("still broken")

    hub.subscribe("bad", explode)
    for _ in range(5):
        await hub.broadcast({"n": 1})

    assert hub.subscriber_names() == ["bad"]
    assert hub.subscriber_error_count("bad") == 5


@pytest.mark.asyncio
async def test_subscriber_can_unsubscribe_itself_mid_dispatch() -> None:
    """Dispatch iterates a snapshot, so self-removal cannot mutate-during-iterate."""
    hub = LiveEventHub()
    seen: list[dict] = []
    unsubscribe: list = []

    def once(payload: dict) -> None:
        seen.append(payload)
        unsubscribe[0]()

    unsubscribe.append(hub.subscribe("once", once))
    other: list[dict] = []
    hub.subscribe("other", other.append)

    await hub.broadcast({"n": 1})
    await hub.broadcast({"n": 2})

    assert [item["n"] for item in seen] == [1]
    assert [item["n"] for item in other] == [1, 2]


@pytest.mark.asyncio
async def test_dispatch_adds_no_await_points() -> None:
    """The tee must complete before broadcast() yields to the event loop.

    Asserted by racing a task started immediately after broadcast(): if
    dispatch awaited anything, the competing task would interleave.
    """
    hub = LiveEventHub()
    order: list[str] = []
    hub.subscribe("sink", lambda _payload: order.append("subscriber"))

    async def competitor() -> None:
        order.append("competitor")

    task = asyncio.create_task(competitor())
    await hub.broadcast({"n": 1})
    await task

    assert order[0] == "subscriber"
