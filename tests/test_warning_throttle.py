"""Regression tests for log-throttling of high-frequency ingest warnings.

A persistently lossy/degraded single node was observed emitting ~7-11 warnings
per second ("Zero-padded degraded audio coverage detected" and "Detected ingest
frame sequence gap"), saturating the in-memory log ring buffer. These warnings
are now rate-limited per node while the cumulative metrics stay exact.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import numpy as np
import pytest

from minimappr.classifiers.heuristic import HeuristicClassifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import AudioCoverageStats, MultiSensorBuffer
from minimappr.core.fusion_node import EventCandidate, FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization import LocalizationEngine
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import GeoPoint, IngestFrameRequest, NodeSpec, NodeType, TimeQuality
from minimappr.storage.db import Storage
from minimappr.utils.audio import encode_pcm16le_b64


async def _build_fusion(tmp_path: Path) -> tuple[FusionNode, Storage]:
    settings = Settings(
        db_path=tmp_path / "throttle.db",
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
        coordinate_frame=LocalCoordinateFrame(
            origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"
        ),
        zone_matcher=ZoneMatcher(storage=storage),
    )
    return fusion, storage


def _degraded_stats() -> AudioCoverageStats:
    return AudioCoverageStats(
        sample_count=1000,
        covered_samples=800,
        missing_samples=200,
        coverage_ratio=0.8,
        missing_ratio=0.2,
        max_gap_samples=8000,
        max_gap_seconds=0.5,
        warning=True,
        degraded=True,
    )


@pytest.mark.asyncio
async def test_degraded_audio_warning_is_throttled_but_metric_is_exact(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fusion, storage = await _build_fusion(tmp_path)
    try:
        candidate = EventCandidate(
            id="cand-1",
            source_node_id="node-A",
            event_time_ns=1_000,
            sample_rate_hz=16_000,
            source_type="raw_sensor",
            time_quality=TimeQuality.GPS_LOCKED,
            source_observation_ids=[],
        )
        audio_quality = {"s0": _degraded_stats(), "s1": _degraded_stats()}

        with caplog.at_level(logging.WARNING, logger="minimappr.core.fusion_node"):
            for _ in range(5):
                fusion._record_degraded_audio_quality_metrics(
                    candidate=candidate,
                    source_window_type="localization_centered",
                    audio_quality=audio_quality,
                )

        # Every degraded sensor on every call is counted (5 calls * 2 sensors).
        assert fusion._metrics.frames_zero_padded_degraded == 10
        # But the warning is logged at most once within the throttle window.
        degraded_logs = [
            record
            for record in caplog.records
            if record.message == "Zero-padded degraded audio coverage detected"
        ]
        assert len(degraded_logs) == 1

        # Simulate the throttle window elapsing -> one more warning is allowed.
        fusion._degraded_audio_warning_last_logged_s["node-A"] = (
            fusion._degraded_audio_warning_last_logged_s["node-A"]
            - fusion._degraded_audio_warning_interval_seconds
            - 1.0
        )
        with caplog.at_level(logging.WARNING, logger="minimappr.core.fusion_node"):
            fusion._record_degraded_audio_quality_metrics(
                candidate=candidate,
                source_window_type="localization_centered",
                audio_quality=audio_quality,
            )
        degraded_logs = [
            record
            for record in caplog.records
            if record.message == "Zero-padded degraded audio coverage detected"
        ]
        assert len(degraded_logs) == 2
        assert fusion._metrics.frames_zero_padded_degraded == 12
    finally:
        await fusion.stop()
        await storage.close()


@pytest.mark.asyncio
async def test_sequence_gap_warning_is_throttled_but_metric_is_exact(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fusion, storage = await _build_fusion(tmp_path)
    try:
        node = NodeSpec(
            id="point-gap-throttle",
            node_type=NodeType.POINT,
            position_m=(0.0, 0.0, 0.0),
            sensor_offsets_m=[(0.0, 0.0, 0.0)],
            capabilities=["audio"],
            metadata={"boot_count": 1},
        )
        samples = np.zeros((1, 1024), dtype=np.float32)
        frame_duration_ns = int(round((1024 / 16_000) * 1_000_000_000))
        start_time_ns = 1_739_810_500_000_000_000

        # Sequences 1,3,5,7 -> a one-frame gap before every frame after the first.
        sequences = (1, 3, 5, 7)
        with caplog.at_level(logging.WARNING, logger="minimappr.core.ingest"):
            for offset, sequence in enumerate(sequences):
                response = await fusion.ingest(
                    IngestFrameRequest(
                        node=node,
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

        gap_logs = [
            record
            for record in caplog.records
            if record.message == "Detected ingest frame sequence gap"
        ]
        # Three gaps occurred but they are throttled to a single log per node.
        assert len(gap_logs) == 1

        status = await fusion.status()
        # The cumulative metric still reflects every missed frame (3 gaps).
        assert status["metrics"]["frame_sequence_gaps"] == 3
    finally:
        await fusion.stop()
        await storage.close()
