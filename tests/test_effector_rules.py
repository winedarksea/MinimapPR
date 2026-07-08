"""EffectorRuleActionHandler dispatch — the destination="effector" rule seam."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from minimappr.core.effector_rules import EffectorRuleActionHandler
from minimappr.core.effectors.base import ExecutionResult
from minimappr.interfaces import ActionDescriptor
from minimappr.models import DetectionEvent, TimeQuality, TrackState


def _track(track_id: str = "trk-1") -> TrackState:
    return TrackState(
        id=track_id,
        first_seen_ns=1,
        last_seen_ns=1,
        position_m=(3.0, 4.0, 0.0),
    )


def _detection() -> DetectionEvent:
    return DetectionEvent(
        id="det-1",
        event_id="det-1",
        source_type="raw_sensor",
        timestamp_ns=1,
        toa_ns=1,
        tor_ns=1,
        time_quality=TimeQuality.FREE_RUNNING,
        position_m=(1.0, 2.0, 0.0),
        confidence=0.9,
        gdop=1.0,
        label="human",
        label_confidence=0.9,
        reference_sensor="node-1:ch0",
        source_sensors=["node-1:ch0"],
    )


@pytest.mark.asyncio
async def test_handle_slews_to_track_position() -> None:
    manager = AsyncMock()
    manager.slew_to_target.return_value = ExecutionResult(status="COMPLETED", execution_id="ex-1")
    handler = EffectorRuleActionHandler(manager)
    descriptor = ActionDescriptor(action_type="cue", destination="effector", payload={"effector_id": "cam-1"})

    result = await handler.handle(descriptor, track=_track())

    manager.slew_to_target.assert_awaited_once_with(
        "cam-1", (3.0, 4.0, 0.0), track_id="trk-1", detection_id=None
    )
    assert result == {
        "delivered": True,
        "handler": "effector",
        "status": "COMPLETED",
        "execution_id": "ex-1",
        "failure_class": None,
    }


@pytest.mark.asyncio
async def test_handle_uses_detection_position_when_no_track() -> None:
    manager = AsyncMock()
    manager.slew_to_target.return_value = ExecutionResult(status="COMPLETED")
    handler = EffectorRuleActionHandler(manager)
    descriptor = ActionDescriptor(action_type="cue", destination="effector", payload={"effector_id": "cam-1"})

    await handler.handle(descriptor, detection=_detection())

    manager.slew_to_target.assert_awaited_once_with(
        "cam-1", (1.0, 2.0, 0.0), track_id=None, detection_id="det-1"
    )


@pytest.mark.asyncio
async def test_handle_accepts_node_id_payload() -> None:
    manager = AsyncMock()
    manager.slew_to_target.return_value = ExecutionResult(status="COMPLETED")
    handler = EffectorRuleActionHandler(manager)
    descriptor = ActionDescriptor(action_type="cue", destination="effector", payload={"node_id": "cam-node"})

    await handler.handle(descriptor, track=_track())

    manager.slew_to_target.assert_awaited_once_with(
        "cam-node", (3.0, 4.0, 0.0), track_id="trk-1", detection_id=None
    )


@pytest.mark.asyncio
async def test_handle_capture_action_type_calls_capture() -> None:
    manager = AsyncMock()
    manager.capture.return_value = ExecutionResult(status="COMPLETED", result_refs=["art-1"])
    handler = EffectorRuleActionHandler(manager)
    descriptor = ActionDescriptor(action_type="capture", destination="effector", payload={"effector_id": "cam-1"})

    result = await handler.handle(descriptor, track=_track())

    manager.capture.assert_awaited_once_with("cam-1", track_id="trk-1", detection_id=None)
    manager.slew_to_target.assert_not_called()
    assert result["status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_handle_missing_effector_id_short_circuits() -> None:
    manager = AsyncMock()
    handler = EffectorRuleActionHandler(manager)
    descriptor = ActionDescriptor(action_type="cue", destination="effector", payload={})

    result = await handler.handle(descriptor, track=_track())

    assert result == {"delivered": False, "handler": "effector", "status": "missing_effector_id"}
    manager.slew_to_target.assert_not_called()


@pytest.mark.asyncio
async def test_handle_no_target_position_short_circuits() -> None:
    manager = AsyncMock()
    handler = EffectorRuleActionHandler(manager)
    descriptor = ActionDescriptor(action_type="cue", destination="effector", payload={"effector_id": "cam-1"})

    result = await handler.handle(descriptor)

    assert result == {"delivered": False, "handler": "effector", "status": "no_target_position"}


@pytest.mark.asyncio
async def test_handle_reports_rejected_as_not_delivered() -> None:
    manager = AsyncMock()
    manager.slew_to_target.return_value = ExecutionResult(status="REJECTED", failure_class="rate_limited")
    handler = EffectorRuleActionHandler(manager)
    descriptor = ActionDescriptor(action_type="cue", destination="effector", payload={"effector_id": "cam-1"})

    result = await handler.handle(descriptor, track=_track())

    assert result["delivered"] is False
    assert result["status"] == "REJECTED"
    assert result["failure_class"] == "rate_limited"
