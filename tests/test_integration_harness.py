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
from tests.helpers import shift_signal


@pytest.mark.asyncio
async def test_integration_harness_multi_node_tdoa_pipeline(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "integration.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        trigger_rms=0.001,
        trigger_cooldown_seconds=0.0,
        localization_window_seconds=0.02,
        max_sensor_buffer_seconds=2.0,
        fusion_worker_count=1,
        fusion_localization_queue_size=16,
        fusion_classification_queue_size=16,
        fusion_rules_queue_size=16,
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

    sample_rate_hz = 16_000
    n = 1024
    rng = np.random.default_rng(99)
    excitation = (rng.standard_normal(n) * np.hanning(n)).astype(np.float32)
    sound_speed = 343.2

    node_positions = [
        (0.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (0.0, 4.0, 0.0),
        (4.0, 4.0, 0.0),
    ]
    source = np.asarray([2.0, 1.6, 0.7], dtype=np.float64)
    start_time_ns = 1_739_810_000_000_000_000
    frame_ns = int((n / sample_rate_hz) * 1_000_000_000)

    for sequence in (1, 2):
        for idx, position in enumerate(node_positions):
            pos_arr = np.asarray(position, dtype=np.float64)
            distance = float(np.linalg.norm(source - pos_arr))
            signal = shift_signal(excitation, sample_rate_hz, distance / sound_speed)
            request = IngestFrameRequest(
                node=NodeSpec(
                    id=f"point-node-{idx}",
                    node_type=NodeType.POINT,
                    position_m=position,
                    sensor_offsets_m=[(0.0, 0.0, 0.0)],
                    capabilities=["audio"],
                ),
                frame={
                    "start_time_ns": start_time_ns + (sequence - 1) * frame_ns,
                    "sample_rate_hz": sample_rate_hz,
                    "channels": 1,
                    "encoding": "pcm16le",
                    "samples_b64": encode_pcm16le_b64(signal[None, :]),
                    "sequence": sequence,
                },
            )
            response = await fusion.ingest(request)
            assert response.accepted is True

    await asyncio.sleep(0.4)
    detections = await storage.list_detections(limit=20)
    assert detections, "Expected at least one detection from multi-node ingest harness"

    best = min(
        detections,
        key=lambda item: abs(float(item["position_m"][0]) - source[0]) + abs(float(item["position_m"][1]) - source[1]),
    )
    estimate = np.asarray(best["position_m"], dtype=np.float64)
    error_m = float(np.linalg.norm(estimate - source))
    assert error_m < 2.2

    await fusion.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_integration_harness_three_node_degrades_to_2d(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "integration_2d.db",
        snippet_dir=tmp_path / "snippets_2d",
        snippet_retention_seconds=0,
        trigger_rms=0.001,
        trigger_cooldown_seconds=0.0,
        localization_window_seconds=0.02,
        max_sensor_buffer_seconds=2.0,
        fusion_worker_count=1,
        min_sensors_for_3d=4,
        min_sensors_for_2d=3,
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

    sample_rate_hz = 16_000
    n = 1024
    rng = np.random.default_rng(123)
    excitation = (rng.standard_normal(n) * np.hanning(n)).astype(np.float32)
    sound_speed = 343.2
    positions = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0)]
    source = np.asarray([1.8, 1.2, 0.0], dtype=np.float64)
    start_time_ns = 1_739_810_200_000_000_000
    frame_ns = int((n / sample_rate_hz) * 1_000_000_000)

    for sequence in (1, 2):
        for idx, position in enumerate(positions):
            distance = float(np.linalg.norm(source - np.asarray(position, dtype=np.float64)))
            signal = shift_signal(excitation, sample_rate_hz, distance / sound_speed)
            response = await fusion.ingest(
                IngestFrameRequest(
                    node=NodeSpec(
                        id=f"point2d-{idx}",
                        node_type=NodeType.POINT,
                        position_m=position,
                        sensor_offsets_m=[(0.0, 0.0, 0.0)],
                        capabilities=["audio"],
                    ),
                    frame={
                        "start_time_ns": start_time_ns + (sequence - 1) * frame_ns,
                        "sample_rate_hz": sample_rate_hz,
                        "channels": 1,
                        "encoding": "pcm16le",
                        "samples_b64": encode_pcm16le_b64(signal[None, :]),
                        "sequence": sequence,
                    },
                )
            )
            assert response.accepted is True

    await asyncio.sleep(0.35)
    tracks = await storage.list_tracks(limit=20)
    assert tracks
    assert any(track.get("capability_tier") == "2d" for track in tracks)

    await fusion.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_integration_harness_multi_source_simultaneous(tmp_path: Path) -> None:
    """Two simultaneous acoustic sources superimposed at each node produce at least one detection."""
    settings = Settings(
        db_path=tmp_path / "multi_src.db",
        snippet_dir=tmp_path / "snippets_ms",
        snippet_retention_seconds=0,
        trigger_rms=0.001,
        trigger_cooldown_seconds=0.0,
        localization_window_seconds=0.02,
        max_sensor_buffer_seconds=2.0,
        fusion_worker_count=1,
        fusion_localization_queue_size=16,
        fusion_classification_queue_size=16,
        fusion_rules_queue_size=16,
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

    sample_rate_hz = 16_000
    n = 1024
    rng = np.random.default_rng(42)
    sound_speed = 343.2

    # Two distinct sources at spatially separate positions
    source_a = np.asarray([1.0, 0.5, 0.0], dtype=np.float64)
    source_b = np.asarray([3.5, 3.5, 0.0], dtype=np.float64)
    signal_a = (rng.standard_normal(n) * np.hanning(n)).astype(np.float32)
    signal_b = (rng.standard_normal(n) * np.hanning(n)).astype(np.float32)

    node_positions = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (4.0, 4.0, 0.0)]
    start_time_ns = 1_739_820_000_000_000_000

    for idx, position in enumerate(node_positions):
        pos_arr = np.asarray(position, dtype=np.float64)
        dist_a = float(np.linalg.norm(source_a - pos_arr))
        dist_b = float(np.linalg.norm(source_b - pos_arr))
        combined = (
            shift_signal(signal_a, sample_rate_hz, dist_a / sound_speed)
            + shift_signal(signal_b, sample_rate_hz, dist_b / sound_speed)
        ).astype(np.float32)
        response = await fusion.ingest(
            IngestFrameRequest(
                node=NodeSpec(
                    id=f"node-ms-{idx}",
                    node_type=NodeType.POINT,
                    position_m=position,
                    sensor_offsets_m=[(0.0, 0.0, 0.0)],
                    capabilities=["audio"],
                ),
                frame={
                    "start_time_ns": start_time_ns,
                    "sample_rate_hz": sample_rate_hz,
                    "channels": 1,
                    "encoding": "pcm16le",
                    "samples_b64": encode_pcm16le_b64(combined[None, :]),
                    "sequence": 1,
                },
            )
        )
        assert response.accepted is True

    await asyncio.sleep(0.4)
    detections = await storage.list_detections(limit=20)
    assert detections, "Expected at least one detection from simultaneous multi-source ingest"

    await fusion.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_integration_harness_mixed_node_types(tmp_path: Path) -> None:
    """A Sirith tetra array and POINT nodes can coexist and both contribute frames to the fusion pipeline."""
    settings = Settings(
        db_path=tmp_path / "mixed_nodes.db",
        snippet_dir=tmp_path / "snippets_mix",
        snippet_retention_seconds=0,
        trigger_rms=0.001,
        trigger_cooldown_seconds=0.0,
        localization_window_seconds=0.04,
        max_sensor_buffer_seconds=2.0,
        fusion_worker_count=1,
        fusion_localization_queue_size=16,
        fusion_classification_queue_size=16,
        fusion_rules_queue_size=16,
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

    sample_rate_hz = 16_000
    n = 1024
    rng = np.random.default_rng(77)
    excitation = (rng.standard_normal(n) * np.hanning(n)).astype(np.float32)
    sound_speed = 343.2
    source = np.asarray([2.0, 2.0, 0.5], dtype=np.float64)
    start_time_ns = 1_739_830_000_000_000_000

    # Sirith tetra array node at origin with 4 microphones
    tetra_offsets = [(-0.02, -0.01, 0.0), (0.02, -0.01, 0.0), (0.0, 0.02, 0.0), (0.0, 0.0, 0.03)]
    tetra_origin = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    channels = [
        shift_signal(
            excitation,
            sample_rate_hz,
            float(np.linalg.norm(source - (tetra_origin + np.asarray(off, dtype=np.float64)))) / sound_speed,
        )
        for off in tetra_offsets
    ]
    tetra_samples = np.stack(channels, axis=0).astype(np.float32)

    tetra_resp = await fusion.ingest(
        IngestFrameRequest(
            node=NodeSpec(
                id="sirith-tetra-mix",
                node_type=NodeType.SIRITH_TETRA,
                position_m=(0.0, 0.0, 0.0),
                sensor_offsets_m=tetra_offsets,
                capabilities=["audio"],
            ),
            frame={
                "start_time_ns": start_time_ns,
                "sample_rate_hz": sample_rate_hz,
                "channels": 4,
                "encoding": "pcm16le",
                "samples_b64": encode_pcm16le_b64(tetra_samples),
                "sequence": 1,
            },
        )
    )
    assert tetra_resp.accepted is True

    # Two POINT nodes at separate positions
    for idx, position in enumerate([(4.0, 0.0, 0.0), (4.0, 4.0, 0.0)]):
        pos_arr = np.asarray(position, dtype=np.float64)
        dist = float(np.linalg.norm(source - pos_arr))
        signal = shift_signal(excitation, sample_rate_hz, dist / sound_speed)
        resp = await fusion.ingest(
            IngestFrameRequest(
                node=NodeSpec(
                    id=f"point-mix-{idx}",
                    node_type=NodeType.POINT,
                    position_m=position,
                    sensor_offsets_m=[(0.0, 0.0, 0.0)],
                    capabilities=["audio"],
                ),
                frame={
                    "start_time_ns": start_time_ns,
                    "sample_rate_hz": sample_rate_hz,
                    "channels": 1,
                    "encoding": "pcm16le",
                    "samples_b64": encode_pcm16le_b64(signal[None, :]),
                    "sequence": 1,
                },
            )
        )
        assert resp.accepted is True

    await asyncio.sleep(0.4)
    status = await fusion.status()
    assert status["metrics"]["ingest_requests"] == 3
    assert status["metrics"]["frames_accepted"] == 3
    assert status["metrics"]["triggers_enqueued"] >= 1

    await fusion.stop()
    await storage.close()
