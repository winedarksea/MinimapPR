"""Field Mode W3-entity: ``GET /api/v1/entities`` + PATCH helpers for entity keys.

Entities are memory-only and live in the process that runs ``FusionNode``.
With entity reporting off — or in a split ``process_role=api`` process that has
no fusion node — the endpoint answers ``enabled: false`` with no entities
rather than an error, so the frontend can hide the tray. Live updates ride
``/ws/live`` as ``{"type": "entity", ...}`` events.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request

from minimappr.core.entity_reporting_config import (
    ENTITY_REPORTING_SETTING_KEY_PREFIX,
    entity_reporting_setting_errors,
)
from minimappr.models import BirdEntityListResponse

entity_router = APIRouter()


def is_entity_reporting_setting_key(key: str) -> bool:
    return key.startswith(ENTITY_REPORTING_SETTING_KEY_PREFIX)


def entity_reporting_patch_errors(coerced: dict[str, object], settings: Any) -> list[str]:
    """Validate W3-entity keys on the merged (patched over current) view."""
    if not any(is_entity_reporting_setting_key(key) for key in coerced):
        return []

    class _MergedView:
        def __getattr__(self, key: str) -> Any:
            return coerced.get(key, getattr(settings, key))

    try:
        return entity_reporting_setting_errors(_MergedView())
    except (TypeError, ValueError) as exc:
        return [f"entity reporting settings: {exc}"]


@entity_router.get("/api/v1/entities", response_model=BirdEntityListResponse)
async def list_bird_entities(request: Request) -> BirdEntityListResponse:
    fusion_node = getattr(request.app.state, "fusion_node", None)
    coordinator = getattr(fusion_node, "entity_reporting", None)
    if coordinator is None:
        return BirdEntityListResponse(enabled=False, generated_ns=time.time_ns(), entities=[])
    return await coordinator.list_entities()
