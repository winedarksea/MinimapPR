"""Python ingest path: a node's packet-clock step re-anchors only that node's
buffers (every channel), through ``FusionNode.ingest`` and ``status()``."""

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
from minimappr.core.timeline_discipline import TimelineDisciplinePolicy
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import GeoPoint, IngestFrameRequest, NodeSpec, NodeType
from minimappr.storage.db import Storage
from minimappr.utils.audio import encode_pcm16le_b64

SAMPLE_RATE_HZ = 16_000
FRAME_SAMPLES = 512
FRAME_NS = 32_000_000
FRAME_COUNT = 30
STEP_FRAME = 12
STEP_NS = 21_000_000
START_NS = 1_764_000_000_000_000_000
TETRA_OFFSETS_M = [(0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.0, 0.05, 0.0), (0.0, 0.0, 0.05)]
# Quiet frames: no trigger, so the test isolates buffering.
QUIET_AMPLITUDE = 1e-4


def _tetra(node_id: str, x_m: float) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        node_type=NodeType.SIRITH_TETRA,
        position_m=(x_m, 0.0, 1.5),
        sensor_offsets_m=TETRA_OFFSETS_M,
        capabilities=["audio"],
        metadata={},
    )


def _frame(frame_index: int, offset_ns: int) -> dict:
    return {
        "start_time_ns": START_NS + frame_index * FRAME_NS + offset_ns,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "channels": len(TETRA_OFFSETS_M),
        "encoding": "pcm16le",
        "samples_b64": encode_pcm16le_b64(np.full((len(TETRA_OFFSETS_M), FRAME_SAMPLES), QUIET_AMPLITUDE, dtype=np.float32)),
        "sequence": frame_index + 1,
        "start_sample_index": frame_index * FRAME_SAMPLES,
        "end_sample_index": (frame_index + 1) * FRAME_SAMPLES,
    }


@pytest.mark.asyncio
async def test_a_stepped_node_is_reanchored_on_the_python_ingest_path(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "timeline.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        trigger_rms=10.0,
        localization_window_seconds=0.04,
        classification_window_seconds=0.5,
        max_sensor_buffer_seconds=4.0,
        fusion_worker_count=1,
    )
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)
    storage = Storage(settings.db_path)
    await storage.initialize()
    fusion = FusionNode(
        settings=settings,
        registry=NodeRegistry(),
        buffer=MultiSensorBuffer(
            max_duration_seconds=settings.max_sensor_buffer_seconds,
            timeline_discipline_policy=TimelineDisciplinePolicy.from_settings(settings),
        ),
        localizer=LocalizationEngine(max_tau_s=0.03),
        classifier=HeuristicClassifier(),
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=lambda payload: asyncio.sleep(0, result=None),
        coordinate_frame=LocalCoordinateFrame(origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"),
        zone_matcher=ZoneMatcher(storage=storage),
    )
    await fusion.start()
    try:
        steady, stepped = _tetra("steady-node", 0.0), _tetra("stepped-node", 5.0)
        for frame_index in range(FRAME_COUNT):
            await fusion.ingest(IngestFrameRequest(node=steady, frame=_frame(frame_index, 0)))
            step_offset_ns = STEP_NS if frame_index >= STEP_FRAME else 0
            await fusion.ingest(IngestFrameRequest(node=stepped, frame=_frame(frame_index, step_offset_ns)))
        status = await fusion.status()
    finally:
        await fusion.stop()
        await storage.close()

    rows = [row for row in status["buffer_state"] if row.get("present")]
    stepped_rows = [row for row in rows if row["sensor_id"].startswith("stepped-node")]
    steady_rows = [row for row in rows if row["sensor_id"].startswith("steady-node")]
    assert len(stepped_rows) == len(steady_rows) == len(TETRA_OFFSETS_M)
    for row in stepped_rows:
        assert row["timeline_step_reanchor_count"] == 1, row
        assert row["reanchor_count"] == 0, "no audio cleared"
        assert row["sample_count"] == FRAME_COUNT * FRAME_SAMPLES
        # The whole buffer moved onto the corrected clock.
        assert row["start_time_ns"] == START_NS + STEP_NS
    for row in steady_rows:
        assert row["timeline_step_reanchor_count"] == 0
        assert row["timeline_drift_slew_count"] == 0
        assert row["start_time_ns"] == START_NS
