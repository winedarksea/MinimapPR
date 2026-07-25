"""Live event websocket hub."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import WebSocket

logger = logging.getLogger(__name__)

LiveEventSubscriber = Callable[[dict], None]
"""A fan-out sink. MUST be synchronous, non-blocking, and non-raising.

``broadcast()`` runs on the fusion hot path, so a subscriber that awaits, blocks,
or raises directly degrades detection emission. The intended body is a single
``queue.put_nowait(payload)``; all interpretation belongs in the consumer's own
task.
"""

_SUBSCRIBER_ERROR_LOG_EVERY = 100


@dataclass(slots=True)
class LiveSubscriptionFilter:
    zone_ids: set[str] = field(default_factory=set)
    label_categories: set[str] = field(default_factory=set)
    labels: set[str] = field(default_factory=set)
    track_statuses: set[str] = field(default_factory=set)
    min_confidence: float | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "LiveSubscriptionFilter":
        raw_filter = payload.get("filter") if isinstance(payload.get("filter"), dict) else payload
        zone_ids = _normalize_set(raw_filter.get("zone_ids"))
        label_categories = _normalize_set(raw_filter.get("label_categories"))
        labels = _normalize_set(raw_filter.get("labels"))
        track_statuses = _normalize_set(raw_filter.get("track_status"))
        min_confidence = None
        raw_conf = raw_filter.get("min_confidence")
        if raw_conf is not None:
            try:
                min_confidence = float(raw_conf)
            except (TypeError, ValueError):
                min_confidence = None
        if min_confidence is not None:
            min_confidence = max(0.0, min(1.0, min_confidence))
        return cls(
            zone_ids=zone_ids,
            label_categories=label_categories,
            labels=labels,
            track_statuses=track_statuses,
            min_confidence=min_confidence,
        )


def _normalize_set(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value.strip().lower()} if value.strip() else set()
    if isinstance(value, (list, tuple, set)):
        result = set()
        for item in value:
            if not isinstance(item, str):
                continue
            text = item.strip().lower()
            if text:
                result.add(text)
        return result
    return set()


class LiveEventHub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._filters: dict[WebSocket, LiveSubscriptionFilter] = {}
        self._lock = asyncio.Lock()
        self._subscribers: dict[str, LiveEventSubscriber] = {}
        self._subscriber_error_counts: dict[str, int] = {}

    def subscribe(self, name: str, subscriber: LiveEventSubscriber) -> Callable[[], None]:
        """Register a synchronous fan-out sink; returns an idempotent unsubscribe.

        Subscribers receive every broadcast payload *unfiltered* — the
        websocket subscription filters are per-client and do not apply. A second
        ``subscribe`` under the same name replaces the first.
        """
        self._subscribers[name] = subscriber
        self._subscriber_error_counts.pop(name, None)

        def unsubscribe() -> None:
            if self._subscribers.get(name) is subscriber:
                del self._subscribers[name]
                self._subscriber_error_counts.pop(name, None)

        return unsubscribe

    def subscriber_names(self) -> list[str]:
        return sorted(self._subscribers)

    def subscriber_error_count(self, name: str) -> int:
        return self._subscriber_error_counts.get(name, 0)

    def _dispatch_subscribers(self, payload: dict) -> None:
        """Fan out to sinks with zero await points and no lock.

        A failing subscriber is counted and throttle-logged but never removed:
        silently dropping a downstream integration's whole feed is a worse
        failure mode than a noisy log, and the operator cannot tell the
        difference between "unsubscribed itself" and "never wired up".
        """
        if not self._subscribers:
            return
        for name, subscriber in list(self._subscribers.items()):
            try:
                subscriber(payload)
            except Exception:
                count = self._subscriber_error_counts.get(name, 0) + 1
                self._subscriber_error_counts[name] = count
                if count == 1 or count % _SUBSCRIBER_ERROR_LOG_EVERY == 0:
                    logger.warning(
                        "live event subscriber %s failed (%d total)", name, count, exc_info=True
                    )

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients.add(websocket)
            self._filters[websocket] = LiveSubscriptionFilter()

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)
            self._filters.pop(websocket, None)

    async def update_filter(self, websocket: WebSocket, payload: dict[str, Any]) -> LiveSubscriptionFilter:
        parsed = LiveSubscriptionFilter.from_payload(payload)
        async with self._lock:
            if websocket in self._clients:
                self._filters[websocket] = parsed
        return parsed

    async def broadcast(self, payload: dict) -> None:
        # Fan out to sinks before touching the lock or any await, so a slow/absent
        # websocket client cannot delay or drop a downstream integration's copy.
        self._dispatch_subscribers(payload)

        async with self._lock:
            clients = list(self._clients)
            filters = {client: self._filters.get(client, LiveSubscriptionFilter()) for client in clients}

        if not clients:
            return

        _SEND_TIMEOUT_S = 2.0
        stale: list[WebSocket] = []
        for client in clients:
            if not self._matches_filter(payload, filters.get(client, LiveSubscriptionFilter())):
                continue
            try:
                await asyncio.wait_for(client.send_json(payload), timeout=_SEND_TIMEOUT_S)
            except (Exception, asyncio.TimeoutError):
                stale.append(client)

        if stale:
            async with self._lock:
                for client in stale:
                    self._clients.discard(client)
                    self._filters.pop(client, None)

    @staticmethod
    def _matches_filter(payload: dict[str, Any], flt: LiveSubscriptionFilter) -> bool:
        detection = payload.get("detection") if isinstance(payload.get("detection"), dict) else {}
        track = payload.get("track") if isinstance(payload.get("track"), dict) else {}

        # Content filters (zones/labels/categories/confidence/statuses) only apply to
        # detection/track payloads. System/state events (e.g. node_updated,
        # node_capability_status) carry neither, so always deliver them — otherwise a
        # subscriber with any content filter set would silently miss node-state events.
        if not detection and not track:
            return True

        if flt.zone_ids:
            zone_ids = {
                str(item).strip().lower()
                for item in detection.get("zone_ids", [])
                if str(item).strip()
            }
            if zone_ids.isdisjoint(flt.zone_ids):
                return False

        if flt.label_categories:
            category = str(
                detection.get("label_category")
                or track.get("label_category")
                or "unknown"
            ).strip().lower()
            if category not in flt.label_categories:
                return False

        if flt.labels:
            label = str(detection.get("label") or track.get("label") or "").strip().lower()
            if label not in flt.labels:
                return False

        if flt.track_statuses:
            status = str(track.get("status") or "").strip().lower()
            if status not in flt.track_statuses:
                return False

        if flt.min_confidence is not None:
            conf_raw = detection.get("label_confidence")
            if conf_raw is None:
                conf_raw = detection.get("confidence")
            if conf_raw is None:
                conf_raw = track.get("confidence")
            try:
                conf = float(conf_raw)
            except (TypeError, ValueError):
                conf = 0.0
            if conf < flt.min_confidence:
                return False

        return True
