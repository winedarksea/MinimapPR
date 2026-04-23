"""Ingest transport implementations."""

from __future__ import annotations

import asyncio

from minimappr.core.fusion_node import FusionNode
from minimappr.interfaces import IngestTransport
from minimappr.models import (
    IngestFrameRequest,
    IngestFrameResponse,
    StoreForwardBufferedFrameResponse,
    StoreForwardIngestRequest,
    StoreForwardIngestResponse,
)


class HttpIngestTransport(IngestTransport):
    def __init__(self, fusion_node: FusionNode) -> None:
        self._fusion_node = fusion_node

    async def deliver_frame(self, payload: IngestFrameRequest) -> IngestFrameResponse:
        return await self._fusion_node.ingest(payload)

    async def deliver_store_forward(self, payload: StoreForwardIngestRequest) -> StoreForwardIngestResponse:
        ordered_frames = payload.buffered_frames
        if payload.sort_by_toa:
            ordered_frames = sorted(
                payload.buffered_frames,
                key=lambda item: (
                    item.frame.toa_ns if item.frame.toa_ns is not None else item.frame.start_time_ns,
                    item.frame.sequence if item.frame.sequence is not None else -1,
                    item.frame.start_time_ns,
                ),
            )

        accepted_frames = 0
        duplicate_frames = 0
        rejected_frames = 0
        queued_events = 0
        results: list[StoreForwardBufferedFrameResponse] = []

        for buffered in ordered_frames:
            try:
                frame_response = await self.deliver_frame(
                    IngestFrameRequest(
                        node=payload.node,
                        frame=buffered.frame,
                        environment=buffered.environment,
                    )
                )
            except ValueError as exc:
                rejected_frames += 1
                results.append(
                    StoreForwardBufferedFrameResponse(
                        sequence=buffered.frame.sequence,
                        start_time_ns=buffered.frame.start_time_ns,
                        accepted=False,
                        duplicate=False,
                        triggered=False,
                        frame_energy=0.0,
                        detail=str(exc),
                    )
                )
                await asyncio.sleep(0)
                continue

            accepted_frames += 1
            duplicate_frames += 1 if frame_response.duplicate else 0
            queued_events += 1 if frame_response.queued_event_id is not None else 0
            # Yield after each frame so the pipeline event loop can run between deliveries.
            await asyncio.sleep(0)
            results.append(
                StoreForwardBufferedFrameResponse(
                    sequence=buffered.frame.sequence,
                    start_time_ns=buffered.frame.start_time_ns,
                    accepted=frame_response.accepted,
                    duplicate=frame_response.duplicate,
                    triggered=frame_response.triggered,
                    frame_energy=frame_response.frame_energy,
                    queued_event_id=frame_response.queued_event_id,
                    detail=None,
                )
            )

        return StoreForwardIngestResponse(
            accepted=rejected_frames == 0,
            total_frames=len(ordered_frames),
            accepted_frames=accepted_frames,
            duplicate_frames=duplicate_frames,
            rejected_frames=rejected_frames,
            queued_events=queued_events,
            results=results,
        )
