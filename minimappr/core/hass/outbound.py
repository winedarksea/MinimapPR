"""The outbound publish queue: coalescing, suppress-unchanged, and the rate floor.

Separated from ``HassBridge`` because it is the one part of the bridge with no
I/O at all — it owns the queue and the "what did we already send" caches, and
decides *which* messages are ready. The bridge does the transport work. That
split keeps the delivery policy testable without a transport and keeps both
files inside the size guideline (AGENTS §1.1).
"""

from __future__ import annotations

import asyncio
import logging

from minimappr.core.hass.transport import MqttPublish

logger = logging.getLogger(__name__)

_DROP_LOG_EVERY = 100


class OutboundQueue:
    def __init__(self, *, maxsize: int, min_interval_seconds: float) -> None:
        self._queue: asyncio.Queue[MqttPublish] = asyncio.Queue(maxsize=maxsize)
        self._min_interval_ns = int(min_interval_seconds * 1_000_000_000)
        self._maxsize = maxsize
        # topic -> last payload actually sent, for suppress-unchanged.
        self._last_published: dict[str, str] = {}
        self._last_publish_ns_by_topic: dict[str, int] = {}
        # Rate-limited messages held for the next cycle.
        self._deferred: list[MqttPublish] = []
        self._published_state_topics: set[str] = set()
        self.dropped_count = 0
        self.coalesced_count = 0
        self.suppressed_unchanged_count = 0

    # -- accessors ----------------------------------------------------------

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    @property
    def capacity(self) -> int:
        return self._maxsize

    @property
    def published_state_topic_count(self) -> int:
        return len(self._published_state_topics)

    # -- ingest -------------------------------------------------------------

    def enqueue(self, message: MqttPublish) -> bool:
        """Queue one message. False means it was dropped because the queue is full.

        **Drop-newest**, deliberately: retained state is fully re-derived by the
        next poll, so a dropped state message self-heals within one interval, and
        dropping the oldest instead would cost a queue pass for no benefit.
        Impulses can genuinely be lost here, which is why ``dropped_count`` is
        surfaced in the status endpoint rather than only logged.
        """
        try:
            self._queue.put_nowait(message)
        except asyncio.QueueFull:
            self.dropped_count += 1
            if self.dropped_count == 1 or self.dropped_count % _DROP_LOG_EVERY == 0:
                logger.warning(
                    "hass bridge dropped a publish (queue full, %d total)", self.dropped_count
                )
            return False
        return True

    # -- flush planning -----------------------------------------------------

    def collect_ready(self, *, now_ns: int) -> list[MqttPublish]:
        """Drain the queue and return the messages that should go out this cycle.

        Ordering matters: coalescing first means the per-topic rate floor applies
        to one message per topic rather than to a backlog, and dedupe before
        rate-limiting means an unchanged topic costs nothing at all.

        Impulses (``coalescable=False``) bypass both dedupe and the rate floor —
        two identical gunshots are two events, and rate-limiting an alert would
        delay the thing an operator most needs.
        """
        coalesced: dict[str, MqttPublish] = {message.topic: message for message in self._deferred}
        self._deferred = []
        impulses: list[MqttPublish] = []

        while True:
            try:
                message = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not message.coalescable:
                impulses.append(message)
                continue
            if message.topic in coalesced:
                self.coalesced_count += 1
            coalesced[message.topic] = message

        ready: list[MqttPublish] = list(impulses)
        for topic, message in coalesced.items():
            if self._last_published.get(topic) == message.payload:
                self.suppressed_unchanged_count += 1
                continue
            last_ns = self._last_publish_ns_by_topic.get(topic)
            if (
                self._min_interval_ns > 0
                and last_ns is not None
                and now_ns - last_ns < self._min_interval_ns
            ):
                # Lossless: it is coalescable, so re-injecting next cycle sends
                # either this payload or a newer one for the same topic.
                self._deferred.append(message)
                continue
            ready.append(message)
        return ready

    # -- post-publish bookkeeping ------------------------------------------

    def record_published(self, message: MqttPublish, *, now_ns: int) -> None:
        if message.retain:
            if message.payload:
                self._last_published[message.topic] = message.payload
                self._published_state_topics.add(message.topic)
            else:
                # An empty retained payload is a delete; forget it so a later
                # re-create is not suppressed as "unchanged".
                self._last_published.pop(message.topic, None)
                self._published_state_topics.discard(message.topic)
        self._last_publish_ns_by_topic[message.topic] = now_ns

    def forget_published(self) -> None:
        """Drop the dedupe caches so everything republishes.

        Called on every (re)connect: a broker restart may have lost its retained
        messages, and a stale cache would suppress exactly the republish that
        heals it.
        """
        self._last_published.clear()
        self._last_publish_ns_by_topic.clear()

    def clear_published_state_topics(self) -> None:
        self._published_state_topics.clear()
