"""Rules-engine action handler for effector cueing (destination="effector").

Registered in ``FusionNode._action_handlers`` keyed by ``"effector"`` when the
effector subsystem is enabled; dispatch already routes by ``descriptor.destination``
(see ``FusionNode._dispatch_rule_action`` in ``core/fusion_node.py``), so no rule-engine
change is required. Not exercised by the default rules in v1 (no auto-cue rules
ship yet), but present so a user can add a cue rule immediately.
"""

from __future__ import annotations

import logging
from typing import Any

from minimappr.core.effectors.registry import EffectorManager
from minimappr.interfaces import ActionDescriptor, RuleActionHandler
from minimappr.models import DetectionEvent, TrackState

logger = logging.getLogger(__name__)


class EffectorRuleActionHandler(RuleActionHandler):
    """Handles action_type="cue"/"capture" rules with destination="effector".

    Expects ``descriptor.payload["node_id"]`` to name the target PTZ-capable node.
    Legacy stored rules with ``effector_id`` are still accepted.
    """

    def __init__(self, effector_manager: EffectorManager) -> None:
        self._effector_manager = effector_manager

    async def handle(
        self,
        descriptor: ActionDescriptor,
        *,
        detection: DetectionEvent | None = None,
        track: TrackState | None = None,
        alert_id: str | None = None,
        rule_id: str | None = None,
    ) -> dict[str, Any]:
        node_id = descriptor.payload.get("node_id") or descriptor.payload.get("effector_id")
        if not node_id:
            return {"delivered": False, "handler": "effector", "status": "missing_effector_id"}

        target_pos = track.position_m if track is not None else (
            detection.position_m if detection is not None else None
        )
        if target_pos is None:
            return {"delivered": False, "handler": "effector", "status": "no_target_position"}

        track_id = track.id if track is not None else None
        detection_id = detection.id if detection is not None else None

        if descriptor.action_type == "capture":
            result = await self._effector_manager.capture(
                node_id, track_id=track_id, detection_id=detection_id
            )
        else:
            result = await self._effector_manager.slew_to_target(
                node_id, target_pos, track_id=track_id, detection_id=detection_id
            )

        return {
            "delivered": result.status == "COMPLETED",
            "handler": "effector",
            "status": result.status,
            "execution_id": result.execution_id,
            "failure_class": result.failure_class,
        }
