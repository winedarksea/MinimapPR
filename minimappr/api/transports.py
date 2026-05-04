"""Ingest transport implementations."""

from __future__ import annotations

import asyncio
import time

from minimappr.api.binary_ingest import BinaryIngestPayload
from minimappr.api.rust_dsp_manifests import LocalizedClassifierRenderRequest
from minimappr.core.fusion_node import FusionNode
from minimappr.interfaces import IngestTransport
from minimappr.models import (
    EnvironmentSampleIn,
    IngestFrameRequest,
    IngestFrameResponse,
    StoreForwardBufferedFrameResponse,
    StoreForwardIngestRequest,
    StoreForwardIngestResponse,
    NodeSpec,
)


class HttpIngestTransport(IngestTransport):
    def __init__(self, fusion_node: FusionNode) -> None:
        self._fusion_node = fusion_node

    async def deliver_frame(self, payload: IngestFrameRequest) -> IngestFrameResponse:
        return await self._fusion_node.ingest(payload)

    async def deliver_store_forward(self, payload: StoreForwardIngestRequest) -> StoreForwardIngestResponse:
        return await self._deliver_buffered_frames(
            node=payload.node,
            buffered_frames=payload.buffered_frames,
            sort_by_toa=payload.sort_by_toa,
        )

    async def deliver_binary(self, payload: BinaryIngestPayload) -> StoreForwardIngestResponse:
        return await self._deliver_buffered_frames(
            node=payload.node,
            buffered_frames=payload.buffered_frames,
            sort_by_toa=payload.sort_by_toa,
        )

    async def deliver_localized_render(self, payload: LocalizedClassifierRenderRequest) -> None:
        await self._fusion_node.ingest_localized_render(payload)

    async def deliver_node_heartbeat(
        self,
        node: NodeSpec,
        *,
        last_sample_time_ns: int | None = None,
        sample_rate_hz: int | None = None,
        active_sensor_count: int | None = None,
        rms: float | None = None,
    ) -> None:
        normalized_node, geo_position = self._fusion_node._ingest_processor._normalize_node_spec(node)
        sensor_count = len(normalized_node.sensor_offsets_m or [])
        audio_summary = {
            "sensor_count": sensor_count,
            "active_sensor_count": active_sensor_count if active_sensor_count is not None else sensor_count,
            "sample_rate_hz": sample_rate_hz,
            "last_sample_time_ns": last_sample_time_ns,
            "age_seconds": None,
            "rms": rms,
            "recent_coverage_ratio": None,
            "recent_missing_ratio": None,
            "recent_max_gap_seconds": None,
            "max_buffer_samples": None,
            "max_buffer_seconds": None,
            "status": "sidecar_stream_consumer",
        }
        async with self._fusion_node._storage_batch():
            await self._fusion_node.storage.upsert_node(
                spec=normalized_node,
                last_seen_ns=time.time_ns(),
                position_geo=geo_position,
            )
            await self._fusion_node.storage.upsert_node_audio_summary(
                node_id=normalized_node.id,
                summary=audio_summary,
                updated_ns=time.time_ns(),
            )

    async def deliver_environment_sample(self, *, node_id: str, sample: EnvironmentSampleIn) -> None:
        if not sample.has_any_measurement():
            return

        timestamp_ns = sample.timestamp_ns or time.time_ns()
        metadata = dict(sample.metadata)
        if sample.source:
            metadata = {"source": sample.source, **metadata}

        async with self._fusion_node._storage_batch():
            await self._fusion_node.storage.insert_environment(
                node_id=node_id,
                timestamp_ns=timestamp_ns,
                temperature_c=sample.temperature_c,
                pressure_pa=sample.pressure_pa,
                humidity_fraction=sample.humidity_fraction,
                wind_speed_mps=sample.wind_speed_mps,
                wind_dir_deg=sample.wind_dir_deg,
                solar_lux=sample.solar_lux,
                metadata=metadata,
            )

            existing_node = await self._fusion_node.storage.get_node_by_id(node_id)
            if isinstance(existing_node, dict):
                try:
                    node_spec = NodeSpec.model_validate(existing_node)
                except Exception:  # noqa: BLE001
                    node_spec = None
                if node_spec is not None:
                    normalized_node, geo_position = self._fusion_node._ingest_processor._normalize_node_spec(node_spec)
                    await self._fusion_node.storage.upsert_node(
                        spec=normalized_node,
                        last_seen_ns=time.time_ns(),
                        position_geo=geo_position,
                    )

        environment_provider = getattr(self._fusion_node, "environment_provider", None)
        if environment_provider is not None and hasattr(environment_provider, "ingest_sample"):
            environment_provider.ingest_sample(
                node_id=node_id,
                timestamp_ns=timestamp_ns,
                temperature_c=sample.temperature_c,
                humidity_fraction=sample.humidity_fraction,
                pressure_pa=sample.pressure_pa,
                wind_speed_mps=sample.wind_speed_mps,
                wind_dir_deg=sample.wind_dir_deg,
                solar_lux=sample.solar_lux,
                location_m=None,
                metadata=metadata,
            )

    async def _deliver_buffered_frames(self, *, node, buffered_frames, sort_by_toa: bool) -> StoreForwardIngestResponse:
        ordered_frames = buffered_frames
        if sort_by_toa:
            ordered_frames = sorted(
                buffered_frames,
                key=lambda item: (
                    item.frame.utc_start_ns if item.frame.utc_start_ns is not None else item.frame.start_time_ns,
                    item.frame.start_sample_index if item.frame.start_sample_index is not None else -1,
                    item.frame.toa_ns if item.frame.toa_ns is not None else item.frame.start_time_ns,
                    item.frame.start_time_ns,
                    item.frame.sequence if item.frame.sequence is not None else -1,
                ),
            )

        accepted_frames = 0
        duplicate_frames = 0
        rejected_frames = 0
        queued_events = 0
        results: list[StoreForwardBufferedFrameResponse] = []

        for buffered in ordered_frames:
            try:
                request = IngestFrameRequest(
                    node=node,
                    frame=buffered.frame,
                    environment=buffered.environment,
                )
                decoded_audio = getattr(buffered, "decoded_audio", None)
                if decoded_audio is None:
                    frame_response = await self.deliver_frame(request)
                else:
                    frame_response = await self._fusion_node.ingest_decoded(request, decoded_audio)
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
            total_frames=len(buffered_frames),
            accepted_frames=accepted_frames,
            duplicate_frames=duplicate_frames,
            rejected_frames=rejected_frames,
            queued_events=queued_events,
            results=results,
        )
