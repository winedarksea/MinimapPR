"""Ingest transport implementations."""

from __future__ import annotations

from minimappr.core.fusion_node import FusionNode
from minimappr.interfaces import IngestTransport
from minimappr.models import IngestFrameRequest


class HttpIngestTransport(IngestTransport):
    def __init__(self, fusion_node: FusionNode) -> None:
        self._fusion_node = fusion_node

    async def deliver_frame(self, payload: IngestFrameRequest):
        return await self._fusion_node.ingest(payload)
