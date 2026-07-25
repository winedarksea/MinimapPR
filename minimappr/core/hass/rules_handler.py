"""Rules-engine action handler for Home Assistant delivery (destination="hass").

Registered in ``FusionNode._action_handlers`` keyed by ``"hass"`` when the bridge
is enabled; dispatch already routes by ``descriptor.destination`` (see
``FusionNode._dispatch_rule_action`` in ``core/fusion_node.py``), so no
rule-engine change is required.

``_dispatch_rule_action`` **awaits handlers inline on the fusion pipeline**, so a
broker round-trip here would stall detection emission for the duration of the
publish. ``handle()` therefore only calls the bridge's synchronous
``enqueue_rule_action`` and returns.

**Documented trade-off:** ``delivered=True`` means *accepted for delivery*, not
*broker-acked*, so the alert row says "sent" for a message that may never reach
the broker (queue drop, broker down past the queue's capacity). ``alert_id`` is
now passed in, so the remaining work to close the loop is for the bridge to patch
that row once the publish actually lands — out of scope for v1 and recorded in
TODO.md.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from minimappr.core.hass.bridge import HassBridge
from minimappr.core.hass.transport import MqttPublish
from minimappr.interfaces import ActionDescriptor, RuleActionHandler
from minimappr.models import DetectionEvent, TrackState

logger = logging.getLogger(__name__)


class HassRuleActionHandler(RuleActionHandler):
    def __init__(self, bridge: HassBridge) -> None:
        self._bridge = bridge

    async def handle(
        self,
        descriptor: ActionDescriptor,
        *,
        detection: DetectionEvent | None = None,
        track: TrackState | None = None,
        alert_id: str | None = None,
        rule_id: str | None = None,
    ) -> dict[str, Any]:
        payload = descriptor.payload if isinstance(descriptor.payload, dict) else {}
        requested_topic = str(payload.get("topic") or "").strip()
        if not requested_topic:
            return _rejected("missing_topic", "rule payload has no 'topic'")

        if not self._bridge.enabled:
            return _rejected("bridge_disabled", "the Home Assistant bridge is not enabled")

        topic = self._bridge.mapper.topics.rule_topic(requested_topic)
        if topic is None:
            # Real safety control, not cosmetics: without it a stored rule could
            # publish to homeassistant/#/config and corrupt every discovery
            # payload on the broker.
            logger.warning("hass rule action refused an unsafe topic: %r", requested_topic)
            return _rejected("invalid_topic", f"topic {requested_topic!r} is not publishable")

        message = MqttPublish(
            topic=topic,
            payload=_message_body(descriptor, payload, detection=detection, track=track),
            retain=bool(payload.get("retain", False)),
            # Always False: a rule action is an impulse. Two identical
            # "occupancy ON" publishes from two matches are two events.
            coalescable=False,
        )
        if not self._bridge.enqueue_rule_action(message):
            return _rejected("queue_full", "the outbound publish queue is full")
        return {
            "delivered": True,
            "handler": "hass",
            "status": "QUEUED",
            "topic": topic,
        }


def _message_body(
    descriptor: ActionDescriptor,
    payload: dict[str, Any],
    *,
    detection: DetectionEvent | None,
    track: TrackState | None,
) -> str:
    """Build the published JSON body.

    A rule may supply ``"message"`` to publish a bare scalar (``"ON"``, a number)
    for an HA entity that expects one; otherwise the body is a JSON object.
    Detection/track ids are **omitted rather than null-filled** — the transcript
    dispatch path passes both as None, and a body full of nulls reads as "we
    looked and found nothing" instead of "not applicable here".
    """
    if "message" in payload:
        message = payload["message"]
        return message if isinstance(message, str) else json.dumps(message, sort_keys=True)

    body: dict[str, Any] = {
        "action_type": descriptor.action_type,
        "priority": descriptor.priority,
    }
    extra = {key: value for key, value in payload.items() if key not in ("topic", "retain")}
    if extra:
        body["payload"] = extra
    if detection is not None:
        body["detection_id"] = detection.id
        body["label"] = detection.label
        body["label_category"] = detection.label_category
    if track is not None:
        body["track_id"] = track.id
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def _rejected(failure_class: str, detail: str) -> dict[str, Any]:
    return {
        "delivered": False,
        "handler": "hass",
        "status": "REJECTED",
        "failure_class": failure_class,
        "detail": detail,
    }
