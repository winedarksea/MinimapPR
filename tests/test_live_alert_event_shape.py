"""The live `alert` websocket payload must match the frontend's `Alert` struct.

The frontend's `LiveEvent` enum is `#[serde(tag = "type")]` wrapping an
internally-tagged `Alert`, so a nested `{"event_type": "alert", "alert": {...}}`
payload failed to deserialize and every live alert was dropped at the websocket
boundary without any server-side symptom. These assertions pin the flat shape so
the websocket and REST-poll paths cannot drift apart again.

Mirrors `minimappr-frontend/src/state/models.rs::Alert` — update both together.
"""

from __future__ import annotations

import pytest

from minimappr.core.rules import WebsocketRuleActionHandler
from minimappr.interfaces import ActionDescriptor
from minimappr.models import DetectionEvent

# Field -> whether the Rust struct tolerates the key being absent. Keys typed as
# a bare String there fail on an explicit null, which is why the handler omits
# rather than null-fills them.
_REQUIRED_KEYS = {"type", "event_type", "priority", "destination", "timestamp_ns"}
_NON_NULLABLE_KEYS = {"type", "event_type", "alert_id"}


def _detection() -> DetectionEvent:
    return DetectionEvent(
        id="det-1",
        timestamp_ns=1,
        position_m=(1.0, 2.0, 3.0),
        confidence=0.8,
        gdop=1.1,
        label="gunshot",
        label_category="security",
        label_confidence=0.9,
        reference_sensor="node-a",
    )


async def _capture(**handle_kwargs) -> dict:
    sent: list[dict] = []

    async def send(payload: dict) -> None:
        sent.append(payload)

    handler = WebsocketRuleActionHandler(send)
    descriptor = ActionDescriptor(
        action_type="alert",
        destination="cop",
        priority="high",
        payload={"message": "Coyote detected"},
    )
    result = await handler.handle(descriptor, **handle_kwargs)
    assert result["delivered"] is True
    assert len(sent) == 1
    return sent[0]


@pytest.mark.asyncio
async def test_payload_is_flat_and_carries_a_type_tag() -> None:
    payload = await _capture(alert_id="a-1", rule_id="coyote_alert")

    assert payload["type"] == "alert", "the frontend enum is tagged on 'type'"
    assert payload["event_type"] == "alert", "older readers key off 'event_type'"
    assert "alert" not in payload, "the nested shape is what broke deserialization"
    assert payload["alert_id"] == "a-1"
    assert payload["rule_id"] == "coyote_alert"


@pytest.mark.asyncio
async def test_payload_has_every_field_the_frontend_requires() -> None:
    payload = await _capture(alert_id="a-1", rule_id="r-1", detection=_detection())

    for key in _REQUIRED_KEYS:
        assert key in payload, f"frontend Alert requires {key}"
    for key in _NON_NULLABLE_KEYS:
        assert payload.get(key) is not None, f"{key} is a bare String in Rust; null fails"
    assert payload["detection_id"] == "det-1"
    assert payload["payload"] == {"message": "Coyote detected"}
    assert payload["status"] == "sent"


@pytest.mark.asyncio
async def test_missing_ids_are_omitted_not_null_filled() -> None:
    """A caller that has no alert row must not emit `"alert_id": null`."""
    payload = await _capture()

    assert "alert_id" not in payload
    assert "rule_id" not in payload
    assert payload["type"] == "alert"


@pytest.mark.asyncio
async def test_timestamp_is_populated() -> None:
    payload = await _capture(alert_id="a-1")
    assert isinstance(payload["timestamp_ns"], int)
    assert payload["timestamp_ns"] > 0
