from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from minimappr.classifiers.heuristic import HeuristicClassifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.fusion_node import FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization import LocalizationEngine
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import GeoPoint, IngestFrameRequest, NodeSpec, NodeType
from minimappr.storage.db import Storage
from minimappr.utils.audio import encode_pcm16le_b64


@pytest.mark.asyncio
async def test_ingest_tracks_sequence_gaps_and_resets_on_boot_rollover(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "fusion_sequence_gap.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        trigger_rms=1.0,
        trigger_cooldown_seconds=0.0,
        localization_window_seconds=0.04,
        classification_window_seconds=0.0,
        max_sensor_buffer_seconds=2.0,
        fusion_worker_count=1,
    )
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)

    storage = Storage(settings.db_path)
    await storage.initialize()

    fusion = FusionNode(
        settings=settings,
        registry=NodeRegistry(),
        buffer=MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds),
        localizer=LocalizationEngine(max_tau_s=0.03),
        classifier=HeuristicClassifier(),
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=lambda payload: asyncio.sleep(0, result=None),
        coordinate_frame=LocalCoordinateFrame(origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"),
        zone_matcher=ZoneMatcher(storage=storage),
    )
    await fusion.start()

    base_node = NodeSpec(
        id="point-sequence-gap",
        node_type=NodeType.POINT,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
        metadata={"boot_count": 1},
    )
    samples = np.zeros((1, 1024), dtype=np.float32)
    frame_duration_ns = int(round((1024 / 16_000) * 1_000_000_000))
    start_time_ns = 1_739_810_500_000_000_000

    for offset, sequence in enumerate((3, 4, 7, 8)):
        response = await fusion.ingest(
            IngestFrameRequest(
                node=base_node,
                frame={
                    "start_time_ns": start_time_ns + (offset * frame_duration_ns),
                    "sample_rate_hz": 16_000,
                    "channels": 1,
                    "encoding": "pcm16le",
                    "samples_b64": encode_pcm16le_b64(samples),
                    "sequence": sequence,
                },
            )
        )
        assert response.accepted is True

    rollover_node = base_node.model_copy(update={"metadata": {"boot_count": 2}})
    for offset, sequence in enumerate((1, 2), start=4):
        response = await fusion.ingest(
            IngestFrameRequest(
                node=rollover_node,
                frame={
                    "start_time_ns": start_time_ns + (offset * frame_duration_ns),
                    "sample_rate_hz": 16_000,
                    "channels": 1,
                    "encoding": "pcm16le",
                    "samples_b64": encode_pcm16le_b64(samples),
                    "sequence": sequence,
                },
            )
        )
        assert response.accepted is True

    status = await fusion.status()
    assert status["metrics"]["frame_sequence_gaps"] == 2

    await fusion.stop()
    await storage.close()