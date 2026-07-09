"""Tests that firmware NodeRunner publish/queue counters are exposed via the API.

The node ships counters such as runner_queue_overflows (which drive server-side
frame sequence gaps) and a publish-failure breakdown inside each frame's
timing_diagnostics. These are persisted with the node audio summary so they
surface under /api/v1/nodes -> audio_debug.runner_stats for live monitoring.
"""

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
from minimappr.core.ingest_health import IngestHealthClassifier, runner_stats_from_timing_diagnostics
from minimappr.core.localization import LocalizationEngine
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import GeoPoint, IngestFrameRequest, NodeSpec, NodeType
from minimappr.storage.db import Storage
from minimappr.utils.audio import encode_pcm16le_b64


def test_extract_runner_stats_selects_known_counters() -> None:
    timing = {
        "runner_queue_overflows": 12,
        "runner_frames_dropped": 5,
        "runner_publish_timeout_failures": 3,
        "runner_last_publish_status": -5,
        "unrelated_field": 99,
        "runner_publish_dns_failures": None,  # absent -> skipped
    }
    stats = runner_stats_from_timing_diagnostics(timing)
    assert stats == {
        "runner_queue_overflows": 12,
        "runner_frames_dropped": 5,
        "runner_publish_timeout_failures": 3,
        "runner_last_publish_status": -5,
    }
    # Nodes without runner telemetry leave the summary untouched.
    assert runner_stats_from_timing_diagnostics({}) is None
    assert runner_stats_from_timing_diagnostics(None) is None


def test_extract_runner_stats_includes_mmb3_transport_health() -> None:
    stats = runner_stats_from_timing_diagnostics({
        "runner_queue_overflows": 2,
        "transport_health": {
            "queue_slots_high_water": 39,
            "queue_slots_capacity": 40,
            "boot_id": 123,
        },
    })

    assert stats == {
        "runner_queue_overflows": 2,
        "transport_health": {
            "queue_slots_high_water": 39,
            "queue_slots_capacity": 40,
            "boot_id": 123,
        },
    }


def test_ingest_health_classifies_gap_causes() -> None:
    classifier = IngestHealthClassifier()
    base = {
        "runner_queue_overflows": 0,
        "runner_frames_dropped": 0,
        "runner_publish_errors": 0,
        "transport_health": {
            "ring_frames_high_water": 1,
            "ring_frames_capacity": 16,
            "queue_slots_high_water": 1,
            "queue_slots_capacity": 40,
            "wifi_rssi_dbm": -55,
            "boot_id": 10,
        },
    }
    assert classifier.observe(node_id="n1", sequence_gap_count=0, timing_diagnostics=base)["verdict"] == "HEALTHY"

    rate_limited = {
        **base,
        "runner_queue_overflows": 1,
    }
    assert classifier.observe(node_id="n1", sequence_gap_count=3, timing_diagnostics=rate_limited)["verdict"] == "LOSSY_RATE_LIMITED"

    wifi = {
        **rate_limited,
        "runner_publish_errors": 2,
        "transport_health": {**base["transport_health"], "queue_slots_high_water": 1},
    }
    assert classifier.observe(node_id="n1", sequence_gap_count=2, timing_diagnostics=wifi)["verdict"] == "LOSSY_WIFI"

    restarted = {
        **wifi,
        "transport_health": {**base["transport_health"], "boot_id": 11},
    }
    assert classifier.observe(node_id="n1", sequence_gap_count=1, timing_diagnostics=restarted)["verdict"] == "NODE_RESTARTED"

    server_gap = {
        **restarted,
        "transport_health": {**base["transport_health"], "boot_id": 11},
    }
    assert classifier.observe(node_id="n1", sequence_gap_count=1, timing_diagnostics=server_gap)["verdict"] == "SERVER_GAP_ONLY"


@pytest.mark.asyncio
async def test_runner_stats_persisted_with_audio_summary(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "runner_stats.db",
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
    await fusion.start()
    try:
        node = NodeSpec(
            id="point-runner-stats",
            node_type=NodeType.POINT,
            position_m=(0.0, 0.0, 0.0),
            sensor_offsets_m=[(0.0, 0.0, 0.0)],
            capabilities=["audio"],
            metadata={"boot_count": 1},
        )
        samples = np.zeros((1, 1024), dtype=np.float32)
        response = await fusion.ingest(
            IngestFrameRequest(
                node=node,
                frame={
                    "start_time_ns": 1_739_810_500_000_000_000,
                    "sample_rate_hz": 16_000,
                    "channels": 1,
                    "encoding": "pcm16le",
                    "samples_b64": encode_pcm16le_b64(samples),
                    "sequence": 1,
                    "timing_diagnostics": {
                        "runner_queue_overflows": 7,
                        "runner_frames_dropped": 2,
                        "runner_publish_wifi_down_failures": 4,
                        "transport_health": {
                            "ring_frames_high_water": 1,
                            "ring_frames_capacity": 16,
                            "queue_slots_high_water": 2,
                            "queue_slots_capacity": 40,
                            "wifi_rssi_dbm": -53,
                            "boot_id": 999,
                        },
                    },
                },
            )
        )
        assert response.accepted is True

        summaries = await storage.list_node_audio_summaries()
        by_node = {item["node_id"]: item for item in summaries}
        runner_stats = by_node["point-runner-stats"].get("runner_stats")
        assert runner_stats is not None
        assert runner_stats["runner_queue_overflows"] == 7
        assert runner_stats["runner_frames_dropped"] == 2
        assert runner_stats["runner_publish_wifi_down_failures"] == 4
        assert runner_stats["transport_health"]["boot_id"] == 999
        ingest_health = by_node["point-runner-stats"].get("ingest_health")
        assert ingest_health is not None
        assert ingest_health["verdict"] == "LOSSY_RATE_LIMITED"
    finally:
        await fusion.stop()
        await storage.close()
