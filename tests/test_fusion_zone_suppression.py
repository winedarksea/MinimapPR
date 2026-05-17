from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
import pytest

from minimappr.classifiers.base import AudioClassifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.fusion_node import FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization import LocalizationEngine
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import ClassificationResult, GeoPoint, IngestFrameRequest, NodeSpec, NodeType
from minimappr.storage.db import Storage
from minimappr.utils.audio import encode_pcm16le_b64


class ConstantBirdClassifier(AudioClassifier):
    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        del samples, sample_rate_hz
        return ClassificationResult(
            label="bird",
            confidence=0.95,
            scores={"bird": 0.95},
            features={"mock": 1.0},
        )


@pytest.mark.asyncio
async def test_zone_suppressed_detection_does_not_emit_tracks_or_alerts(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "suppression.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        trigger_rms=0.01,
        trigger_cooldown_seconds=0.0,
        localization_window_seconds=0.04,
        classification_window_seconds=0.0,
        max_sensor_buffer_seconds=2.0,
        fusion_worker_count=1,
        site_origin_lat=37.0,
        site_origin_lon=-122.0,
        site_origin_alt_m=0.0,
    )
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)

    storage = Storage(settings.db_path)
    await storage.initialize()
    await storage.upsert_zone(
        zone_id="wildlife-suppress",
        name="Wildlife Suppress",
        zone_type="exclusion_zone",
        polygon_geo=[
            [36.999, -122.001],
            [36.999, -121.999],
            [37.001, -121.999],
            [37.001, -122.001],
        ],
        properties={"suppress_label_categories": ["wildlife"]},
        created_ns=time.time_ns(),
    )

    fusion = FusionNode(
        settings=settings,
        registry=NodeRegistry(),
        buffer=MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds),
        localizer=LocalizationEngine(max_tau_s=0.02),
        classifier=ConstantBirdClassifier(),
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=lambda payload: asyncio.sleep(0, result=None),
        coordinate_frame=LocalCoordinateFrame(origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"),
        zone_matcher=ZoneMatcher(storage=storage),
    )
    await fusion.start()

    request = IngestFrameRequest(
        node=NodeSpec(
            id="point-zone-test",
            node_type=NodeType.POINT,
            position_m=(0.0, 0.0, 0.0),
            sensor_offsets_m=[(0.0, 0.0, 0.0)],
            capabilities=["audio"],
        ),
        frame={
            "start_time_ns": 1_739_810_000_000_000_000,
            "sample_rate_hz": 16_000,
            "channels": 1,
            "encoding": "pcm16le",
            "samples_b64": encode_pcm16le_b64(
                np.random.default_rng(444).normal(0.0, 0.2, size=(1, 1024)).astype(np.float32)
            ),
            "sequence": 1,
        },
    )
    response = await fusion.ingest(request)
    assert response.accepted is True
    assert response.triggered is True

    await asyncio.sleep(0.2)
    await fusion.housekeeping_tick(now_ns=time.time_ns())

    detections = await storage.list_detections(limit=10)
    assert detections
    assert detections[0]["track_id"] is None

    tracks = await storage.list_tracks(limit=10)
    assert tracks == []

    alerts = await storage.list_alerts(limit=10)
    assert alerts == []

    status = await fusion.status()
    assert status["metrics"]["detections_suppressed_by_zone"] >= 1

    await fusion.stop()
    await storage.close()
