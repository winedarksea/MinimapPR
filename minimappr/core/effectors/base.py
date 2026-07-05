"""Effector abstraction: the Protocol every driver implements.

``ExecutionResult`` deliberately mirrors speed-core's ``CommandOutcome`` shape
(speed_core/kernel/event.py) — same ``status`` vocabulary (``COMPLETED`` /
``FAILED`` / ``REJECTED``), ``execution_id``, ``failure_class`` (not
``error``), and ``result_refs`` for artifact ids. ``EffectorManager`` entry
points accept an optional caller-supplied ``execution_id``/``idempotency_key``
mirroring ``DomainPort.execute(command, *, execution_id, idempotency_key)``
(speed_core/domain/port.py) so a future thin adapter can delegate straight
through with no change to effector internals.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from minimappr.models import Vec3


ExecutionStatus = Literal["COMPLETED", "FAILED", "REJECTED"]


@dataclass(slots=True)
class EffectorCapabilities:
    """What an effector was probed to support, plus the strategies selected."""
    movement_strategies: list[str] = field(default_factory=list)
    """Supported move operations in probe order, e.g. ['AbsoluteMove', 'RelativeMove', 'ContinuousMove']."""
    selected_movement_strategy: str | None = None
    snapshot_strategies: list[str] = field(default_factory=list)
    """e.g. ['snapshot_uri', 'rtsp_frame_grab']."""
    selected_snapshot_strategy: str | None = None
    has_zoom: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EffectorCommand:
    target_position_m: Vec3
    track_id: str | None = None
    detection_id: str | None = None
    priority: str = "normal"


@dataclass(slots=True)
class ExecutionResult:
    status: ExecutionStatus
    execution_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    failure_class: str | None = None
    result_refs: list[str] = field(default_factory=list)
    detail: str | None = None


@runtime_checkable
class Effector(Protocol):
    async def get_capabilities(self) -> EffectorCapabilities:
        ...

    async def arm(self, *, zone_id: str | None = None) -> bool:
        ...

    async def execute(self, command: EffectorCommand) -> ExecutionResult:
        ...

    async def disarm(self) -> bool:
        ...

    async def get_status(self) -> dict[str, Any]:
        ...
