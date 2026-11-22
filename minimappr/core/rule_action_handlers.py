"""Built-in rule action handlers (log, websocket), split out of ``core/rules.py``."""

from __future__ import annotations

import logging
import time
from typing import Any

from minimappr.interfaces import ActionDescriptor, RuleActionHandler
from minimappr.models import DetectionEvent, TrackState

logger = logging.getLogger("minimappr.core.rules")


class LoggingRuleActionHandler(RuleActionHandler):
    async def handle(
        self,
        descriptor: ActionDescriptor,
        *,
        detection: DetectionEvent | None = None,
        track: TrackState | None = None,
        alert_id: str | None = None,
        rule_id: str | None = None,
    ) -> dict[str, Any]:
        logger.info(
            "rule_action type=%s destination=%s priority=%s detection=%s track=%s",
            descriptor.action_type,
            descriptor.destination,
            descriptor.priority,
            detection.id if detection else None,
            track.id if track else None,
        )
        return {"delivered": True, "handler": "log"}


class WebsocketRuleActionHandler(RuleActionHandler):
    """Push a fired alert to live websocket clients.

    The payload is **flat**, not nested under an ``"alert"`` key, and carries both
    ``type`` and ``event_type``. The frontend's ``LiveEvent`` enum is
    ``#[serde(tag = "type")]`` over an internally-tagged ``Alert`` struct, so the
    previous nested/untagged shape failed to deserialize and every live alert was
    silently dropped at the websocket boundary. Flat also means the websocket and
    the ``GET /api/v1/alerts`` poll deliver the same shape, so the two paths
    cannot drift again.
    """

    def __init__(self, send_callback) -> None:
        self._send_callback = send_callback

    async def handle(
        self,
        descriptor: ActionDescriptor,
        *,
        detection: DetectionEvent | None = None,
        track: TrackState | None = None,
        alert_id: str | None = None,
        rule_id: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "type": "alert",
            "event_type": "alert",
            "action_type": descriptor.action_type,
            "destination": descriptor.destination,
            "priority": descriptor.priority,
            "status": "sent",
            "timestamp_ns": time.time_ns(),
            "payload": descriptor.payload,
            "detection_id": detection.id if detection else None,
            "track_id": track.id if track else None,
        }
        # Omit rather than null-fill: the consumer types these as required
        # strings, and an explicit null fails deserialization where absence
        # falls back to a default.
        if alert_id is not None:
            payload["alert_id"] = alert_id
        if rule_id is not None:
            payload["rule_id"] = rule_id
        await self._send_callback(payload)
        return {"delivered": True, "handler": "websocket"}


__all__ = ["LoggingRuleActionHandler", "WebsocketRuleActionHandler"]
