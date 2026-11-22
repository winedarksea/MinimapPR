"""Field Mode W3-entity: hold rules on a new unlocalized entity for a grace window.

A species heard without a fix usually gets localized (or absorbed into a
mapped entity) a moment later. Firing rules at once would alert twice: once as
"heard nearby", once as the localized bird. Instead the first unlocalized
detection of a new entity is held for ``grace_seconds``:

- the entity gains a bearing/position inside the window -> rules fire once, on
  the localizing detection (``release_for_spatial``);
- the window ends with no fix -> rules fire on the held detection;
- the entity expires first -> nothing fires.

Timers are asyncio tasks off the ingest hot path; ``stop`` cancels them.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

_logger = logging.getLogger(__name__)

FireRulesCallback = Callable[[Any, Any], Awaitable[None]]


@dataclass(slots=True)
class _HeldNotification:
    detection: Any
    track: Any
    timer_task: asyncio.Task | None = None


class NotificationGraceWindow:
    def __init__(self, *, grace_seconds: float, fire_rules: FireRulesCallback) -> None:
        self._grace_seconds = max(0.0, float(grace_seconds))
        self._fire_rules = fire_rules
        self._held_by_entity_id: dict[str, _HeldNotification] = {}
        # Entities whose rules already fired: later bouts never re-notify.
        self._notified_entity_ids: set[str] = set()
        self.fired_after_grace_count = 0
        self.fired_on_localization_count = 0

    @property
    def grace_seconds(self) -> float:
        return self._grace_seconds

    def is_pending(self, entity_id: str) -> bool:
        return entity_id in self._held_by_entity_id

    def was_notified(self, entity_id: str) -> bool:
        return entity_id in self._notified_entity_ids

    async def notify_now(self, entity_id: str, detection: Any, track: Any) -> None:
        if entity_id in self._notified_entity_ids:
            return
        self._notified_entity_ids.add(entity_id)
        await self._fire_rules(detection, track)

    async def hold_or_fire(self, entity_id: str, detection: Any, track: Any) -> None:
        """Start the grace window for a new unlocalized entity (0 s = fire now)."""
        if entity_id in self._notified_entity_ids or entity_id in self._held_by_entity_id:
            return
        if self._grace_seconds <= 0.0:
            await self.notify_now(entity_id, detection, track)
            return
        held = _HeldNotification(detection=detection, track=track)
        self._held_by_entity_id[entity_id] = held
        held.timer_task = asyncio.create_task(
            self._fire_after_grace(entity_id), name=f"entity-grace-{entity_id}"
        )

    async def release_for_spatial(self, entity_id: str, detection: Any, track: Any) -> bool:
        """The entity got a fix inside its window: fire once, on this detection."""
        held = self._held_by_entity_id.pop(entity_id, None)
        if held is None:
            return False
        if held.timer_task is not None:
            held.timer_task.cancel()
        self.fired_on_localization_count += 1
        await self.notify_now(entity_id, detection, track)
        return True

    def forget(self, entity_id: str) -> None:
        """Entity expired: drop its hold (nothing fires) and its notified mark."""
        held = self._held_by_entity_id.pop(entity_id, None)
        if held is not None and held.timer_task is not None:
            held.timer_task.cancel()
        self._notified_entity_ids.discard(entity_id)

    def retain_only(self, live_entity_ids: set[str]) -> None:
        """Bound memory: forget entities the registry no longer holds (e.g. evicted)."""
        for entity_id in [eid for eid in self._held_by_entity_id if eid not in live_entity_ids]:
            self.forget(entity_id)
        self._notified_entity_ids &= live_entity_ids

    async def stop(self) -> None:
        tasks = [held.timer_task for held in self._held_by_entity_id.values() if held.timer_task is not None]
        self._held_by_entity_id.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _fire_after_grace(self, entity_id: str) -> None:
        try:
            await asyncio.sleep(self._grace_seconds)
        except asyncio.CancelledError:
            return
        held = self._held_by_entity_id.pop(entity_id, None)
        if held is None:
            return
        self.fired_after_grace_count += 1
        try:
            await self.notify_now(entity_id, held.detection, held.track)
        except Exception:  # noqa: BLE001 - a failing rule action must not kill the timer loop silently
            _logger.exception("grace-window rule evaluation failed for entity %s", entity_id)
