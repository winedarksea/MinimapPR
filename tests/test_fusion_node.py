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
async def test_fusion_node_ingest_and_status(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "fusion.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        trigger_rms=0.001,
        trigger_cooldown_seconds=0.0,
        localization_window_seconds=0.04,
        max_sensor_buffer_seconds=2.0,
        fusion_worker_count=1,
        fusion_event_queue_size=8,
    )
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)

    storage = Storage(settings.db_path)
    await storage.initialize()

    registry = NodeRegistry()
    buffer = MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds)
    localizer = LocalizationEngine(max_tau_s=0.03)
    classifier = HeuristicClassifier()
    tracker = TrackManager(settings)
    coordinate_frame = LocalCoordinateFrame(origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat")
    zone_matcher = ZoneMatcher(storage=storage)

    async def _live(payload: dict) -> None:
        _ = payload

    fusion = FusionNode(
        settings=settings,
        registry=registry,
        buffer=buffer,
        localizer=localizer,
        classifier=classifier,
        tracker=tracker,
        storage=storage,
        live_callback=_live,
        coordinate_frame=coordinate_frame,
        zone_matcher=zone_matcher,
    )

    await fusion.start()

    node = NodeSpec(
        id="sirith-tetra-test",
        node_type=NodeType.SIRITH_TETRA,
        position_m=(1.0, 2.0, 3.0),
        sensor_offsets_m=[
            (-0.02, -0.01, 0.0),
            (0.02, -0.01, 0.0),
            (0.0, 0.02, 0.0),
            (0.0, 0.0, 0.03),
        ],
        capabilities=["audio"],
        metadata={},
    )

    rng = np.random.default_rng(11)
    channels_first = rng.normal(0.0, 0.2, size=(4, 1024)).astype(np.float32)

    request = IngestFrameRequest(
        node=node,
        frame={
            "start_time_ns": 1_739_810_000_000_000_000,
            "sample_rate_hz": 16000,
            "channels": 4,
            "encoding": "pcm16le",
            "samples_b64": encode_pcm16le_b64(channels_first),
            "sequence": 1,
        },
    )

    response = await fusion.ingest(request)

    assert response.accepted is True
    assert response.frame_energy > 0.0
    assert response.triggered is True
    assert response.queued_event_id is not None

    await asyncio.sleep(0.1)

    status = await fusion.status()
    assert status["started"] is True
    assert status["workers"]["localization_running"] == 1
    assert status["workers"]["classification_running"] == 1
    assert status["workers"]["rules_running"] == 1
    assert status["metrics"]["ingest_requests"] == 1
    assert status["metrics"]["frames_accepted"] == 1
    assert status["metrics"]["triggers_enqueued"] == 1

    await fusion.stop()
    await storage.close()
