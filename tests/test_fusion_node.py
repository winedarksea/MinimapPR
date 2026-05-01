from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from minimappr.api.rust_dsp_manifests import LocalizedClassifierRenderRequest
from minimappr.classifiers.base import AudioClassifier, ClassificationResult
from minimappr.classifiers.heuristic import HeuristicClassifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.fusion_node import FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization import LocalizationEngine, LocalizationResult
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import GeoPoint, IngestFrameRequest, NodeSpec, NodeType
from minimappr.storage.db import Storage
from minimappr.utils.audio import encode_pcm16le_b64, mono_mix, rms


class _FixedReferenceLocalizer:
    def __init__(self, reference_sensor: str) -> None:
        self.reference_sensor = reference_sensor

    def localize(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
    ) -> LocalizationResult:
        del sensor_positions, sensor_windows, sample_rate_hz, temperature_c, humidity_fraction
        return LocalizationResult(
            position_m=(0.0, 0.0, 0.0),
            confidence=0.9,
            gdop=1.0,
            reference_sensor=self.reference_sensor,
            tdoa_s={},
        )


class _CountingClassifier(AudioClassifier):
    def __init__(self) -> None:
        self.calls = 0

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        del samples, sample_rate_hz
        self.calls += 1
        return ClassificationResult(
            label="sparrow",
            confidence=0.91,
            scores={"sparrow": 0.91},
            features={},
        )


class _UnknownThenSparrowClassifier(AudioClassifier):
    def __init__(self) -> None:
        self.calls = 0

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        del samples, sample_rate_hz
        self.calls += 1
        if self.calls == 1:
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                scores={"unknown": 0.0},
                features={},
            )
        return ClassificationResult(
            label="sparrow",
            confidence=0.91,
            scores={"sparrow": 0.91},
            features={},
        )


class _AlwaysUnknownClassifier(AudioClassifier):
    def __init__(self) -> None:
        self.calls = 0

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        del samples, sample_rate_hz
        self.calls += 1
        return ClassificationResult(
            label="unknown",
            confidence=0.0,
            scores={"unknown": 0.0},
            features={},
        )


class _AlwaysRaisingClassifier(AudioClassifier):
    def __init__(self) -> None:
        self.calls = 0

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        del samples, sample_rate_hz
        self.calls += 1
        raise RuntimeError("classifier unavailable")


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
    assert "realtime" in status
    assert "pipeline_seconds_behind_realtime" in status["realtime"]
    assert status["drop_on_backpressure"] is True

    await fusion.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_fusion_ingest_accepts_explicit_packet_coverage_metadata(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "fusion_explicit_coverage.db",
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

    node = NodeSpec(
        id="sirith-explicit-coverage",
        node_type=NodeType.SIRITH_TETRA,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=[
            (-0.02, -0.01, 0.0),
            (0.02, -0.01, 0.0),
            (0.0, 0.02, 0.0),
            (0.0, 0.0, 0.03),
        ],
        capabilities=["audio", "array_localization"],
        metadata={},
    )

    channels_first = np.ones((4, 1024), dtype=np.float32) * 0.1
    response = await fusion.ingest(
        IngestFrameRequest(
            node=node,
            frame={
                "start_time_ns": 1_739_810_000_000_000_000,
                "utc_end_ns": 1_739_810_000_064_000_000,
                "start_sample_index": 32_000,
                "end_sample_index": 33_024,
                "sample_rate_hz": 16_000,
                "channels": 4,
                "encoding": "pcm16le",
                "samples_per_channel": 1024,
                "samples_b64": encode_pcm16le_b64(channels_first),
                "sequence": 1,
                "time_quality": "gps_locked",
            },
        )
    )

    assert response.accepted is True
    assert response.frame_energy > 0.0

    await fusion.stop()
    await storage.close()


def test_ingest_model_rejects_negative_utc_coverage() -> None:
    with pytest.raises(ValueError, match="utc_end_ns must be >= start_time_ns"):
        IngestFrameRequest(
            node=NodeSpec(
                id="bad-coverage",
                node_type=NodeType.SIRITH_TETRA,
                position_m=(0.0, 0.0, 0.0),
                sensor_offsets_m=[
                    (-0.02, -0.01, 0.0),
                    (0.02, -0.01, 0.0),
                    (0.0, 0.02, 0.0),
                    (0.0, 0.0, 0.03),
                ],
                capabilities=["audio"],
                metadata={},
            ),
            frame={
                "start_time_ns": 100,
                "utc_end_ns": 99,
                "sample_rate_hz": 16_000,
                "channels": 4,
                "encoding": "pcm16le",
                "samples_per_channel": 1,
                "samples_b64": encode_pcm16le_b64(np.zeros((4, 1), dtype=np.float32)),
            },
        )


@pytest.mark.asyncio
async def test_multichannel_trigger_avoids_phase_cancellation(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "fusion_phase_cancel.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        trigger_rms=0.05,
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

    node = NodeSpec(
        id="sirith-phase-cancel",
        node_type=NodeType.SIRITH_TETRA,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=[
            (-0.02, -0.01, 0.0),
            (0.02, -0.01, 0.0),
            (0.0, 0.02, 0.0),
            (0.0, 0.0, 0.03),
        ],
        capabilities=["audio", "array_localization"],
        metadata={},
    )

    sample_rate_hz = 16000
    sample_count = 1024
    t = np.arange(sample_count, dtype=np.float32) / float(sample_rate_hz)
    tone = 0.08 * np.sin(2.0 * np.pi * 3200.0 * t)
    channels_first = np.stack([tone, -tone, tone, -tone]).astype(np.float32)

    assert rms(channels_first[0]) > settings.trigger_rms
    assert rms(mono_mix(channels_first)) < (settings.trigger_rms * 0.1)

    response = await fusion.ingest(
        IngestFrameRequest(
            node=node,
            frame={
                "start_time_ns": 1_739_810_050_000_000_000,
                "sample_rate_hz": sample_rate_hz,
                "channels": 4,
                "encoding": "pcm16le",
                "samples_b64": encode_pcm16le_b64(channels_first),
                "sequence": 1,
            },
        )
    )

    assert response.accepted is True
    assert response.triggered is True
    assert response.frame_energy > settings.trigger_rms

    await fusion.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_fusion_backpressure_drops_when_queue_full(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "fusion_backpressure.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        trigger_rms=0.001,
        trigger_cooldown_seconds=0.0,
        localization_window_seconds=0.04,
        max_sensor_buffer_seconds=2.0,
        fusion_worker_count=1,
        fusion_localization_queue_size=1,
        fusion_classification_queue_size=1,
        fusion_rules_queue_size=1,
        fusion_drop_on_backpressure=True,
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

    node = NodeSpec(
        id="point-backpressure",
        node_type=NodeType.POINT,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
        metadata={},
    )
    samples = np.random.default_rng(2025).normal(0.0, 0.2, size=(1, 1024)).astype(np.float32)
    payload = {
        "sample_rate_hz": 16000,
        "channels": 1,
        "encoding": "pcm16le",
        "samples_b64": encode_pcm16le_b64(samples),
        "sequence": 1,
    }

    first = await fusion.ingest(
        IngestFrameRequest(
            node=node,
            frame={"start_time_ns": 1_739_810_100_000_000_000, **payload},
        )
    )
    second = await fusion.ingest(
        IngestFrameRequest(
            node=node,
            frame={"start_time_ns": 1_739_810_100_100_000_000, **payload},
        )
    )
    third = await fusion.ingest(
        IngestFrameRequest(
            node=node,
            frame={"start_time_ns": 1_739_810_100_200_000_000, **payload},
        )
    )

    assert first.triggered is True
    assert second.triggered is False
    assert third.triggered is False

    status = await fusion.status()
    assert status["metrics"]["triggers_dropped_queue_full"] >= 1
    assert status["metrics"]["stage_drops_backpressure"] >= 1
    assert status["realtime"]["stages"]["localization"]["queued_items"] >= 0

    await storage.close()


@pytest.mark.asyncio
async def test_fusion_reuses_localized_classification_for_matching_omni_reference(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "fusion_reuse.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        trigger_rms=0.001,
        trigger_cooldown_seconds=0.0,
        localization_window_seconds=0.04,
        classification_window_seconds=0.08,
        max_sensor_buffer_seconds=2.0,
        fusion_worker_count=1,
        beamformed_classification_enabled=False,
        preprocess_enabled=False,
    )
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)

    storage = Storage(settings.db_path)
    await storage.initialize()

    classifier = _CountingClassifier()
    fusion = FusionNode(
        settings=settings,
        registry=NodeRegistry(),
        buffer=MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds),
        localizer=_FixedReferenceLocalizer(reference_sensor="reuse-node:ch0"),
        classifier=classifier,
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=lambda payload: asyncio.sleep(0, result=None),
        coordinate_frame=LocalCoordinateFrame(origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"),
        zone_matcher=ZoneMatcher(storage=storage),
    )
    await fusion.start()

    node = NodeSpec(
        id="reuse-node",
        node_type=NodeType.SIRITH_TETRA,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=[
            (-0.02, -0.01, 0.0),
            (0.02, -0.01, 0.0),
            (0.0, 0.02, 0.0),
            (0.0, 0.0, 0.03),
        ],
        capabilities=["audio", "array_localization"],
        metadata={},
    )
    t = np.arange(1024, dtype=np.float32) / 16000.0
    channels = np.stack(
        [
            0.4 * np.sin(2.0 * np.pi * 1200.0 * t),
            0.2 * np.sin(2.0 * np.pi * 1200.0 * t),
            0.1 * np.sin(2.0 * np.pi * 1200.0 * t),
            0.05 * np.sin(2.0 * np.pi * 1200.0 * t),
        ]
    ).astype(np.float32)

    response = await fusion.ingest(
        IngestFrameRequest(
            node=node,
            frame={
                "start_time_ns": 1_739_810_300_000_000_000,
                "sample_rate_hz": 16000,
                "channels": 4,
                "encoding": "pcm16le",
                "samples_b64": encode_pcm16le_b64(channels),
                "sequence": 1,
            },
        )
    )
    assert response.triggered is True

    await asyncio.sleep(0.2)

    status = await fusion.status()
    assert classifier.calls == 1
    assert status["metrics"]["classification_reuse_hits"] == 1

    await fusion.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_birdnet_chunked_dispatch_suppresses_overlapping_candidates(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "fusion_chunking.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        trigger_rms=0.001,
        trigger_cooldown_seconds=0.0,
        localization_window_seconds=0.04,
        classification_window_seconds=30.0,
        max_sensor_buffer_seconds=32.0,
        classifier_backend="birdnet",
        birdnet_chunked_dispatch_enabled=True,
        birdnet_chunk_overlap_seconds=2.0,
        preprocess_enabled=False,
        fusion_worker_count=1,
    )
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)

    storage = Storage(settings.db_path)
    await storage.initialize()

    classifier = _CountingClassifier()
    fusion = FusionNode(
        settings=settings,
        registry=NodeRegistry(),
        buffer=MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds),
        localizer=LocalizationEngine(max_tau_s=0.03),
        classifier=classifier,
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=lambda payload: asyncio.sleep(0, result=None),
        coordinate_frame=LocalCoordinateFrame(origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"),
        zone_matcher=ZoneMatcher(storage=storage),
    )
    await fusion.start()

    node = NodeSpec(
        id="chunk-node",
        node_type=NodeType.POINT,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
        metadata={},
    )
    t = np.arange(1024, dtype=np.float32) / 16000.0
    samples = (0.4 * np.sin(2.0 * np.pi * 1200.0 * t)).reshape(1, -1).astype(np.float32)
    stride_ns = int((settings.classification_window_seconds - settings.birdnet_chunk_overlap_seconds) * 1_000_000_000)
    aligned_start_ns = 1_739_810_400_000_000_000
    aligned_start_ns -= aligned_start_ns % stride_ns

    first = await fusion.ingest(
        IngestFrameRequest(
            node=node,
            frame={
                "start_time_ns": aligned_start_ns,
                "sample_rate_hz": 16000,
                "channels": 1,
                "encoding": "pcm16le",
                "samples_b64": encode_pcm16le_b64(samples),
                "sequence": 1,
            },
        )
    )
    second = await fusion.ingest(
        IngestFrameRequest(
            node=node,
            frame={
                "start_time_ns": aligned_start_ns + 5_000_000_000,
                "sample_rate_hz": 16000,
                "channels": 1,
                "encoding": "pcm16le",
                "samples_b64": encode_pcm16le_b64(samples),
                "sequence": 2,
            },
        )
    )

    assert first.triggered is True
    assert second.triggered is True

    await asyncio.sleep(0.25)

    status = await fusion.status()
    assert classifier.calls == 1
    assert status["metrics"]["birdnet_chunk_dispatches_suppressed"] == 1

    await fusion.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_birdnet_chunked_dispatch_retries_after_non_actionable_result(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "fusion_chunk_retry.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        trigger_rms=0.001,
        trigger_cooldown_seconds=0.0,
        localization_window_seconds=0.04,
        classification_window_seconds=30.0,
        max_sensor_buffer_seconds=32.0,
        classifier_backend="birdnet",
        birdnet_chunked_dispatch_enabled=True,
        birdnet_chunk_overlap_seconds=2.0,
        birdnet_chunk_min_retry_progress_seconds=0.0,
        preprocess_enabled=False,
        fusion_worker_count=1,
    )
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)

    storage = Storage(settings.db_path)
    await storage.initialize()

    classifier = _UnknownThenSparrowClassifier()
    fusion = FusionNode(
        settings=settings,
        registry=NodeRegistry(),
        buffer=MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds),
        localizer=LocalizationEngine(max_tau_s=0.03),
        classifier=classifier,
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=lambda payload: asyncio.sleep(0, result=None),
        coordinate_frame=LocalCoordinateFrame(origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"),
        zone_matcher=ZoneMatcher(storage=storage),
    )
    await fusion.start()

    node = NodeSpec(
        id="chunk-retry-node",
        node_type=NodeType.POINT,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
        metadata={},
    )
    t = np.arange(1024, dtype=np.float32) / 16000.0
    samples = (0.4 * np.sin(2.0 * np.pi * 900.0 * t)).reshape(1, -1).astype(np.float32)
    stride_ns = int((settings.classification_window_seconds - settings.birdnet_chunk_overlap_seconds) * 1_000_000_000)
    aligned_start_ns = 1_739_810_500_000_000_000
    aligned_start_ns -= aligned_start_ns % stride_ns

    first = await fusion.ingest(
        IngestFrameRequest(
            node=node,
            frame={
                "start_time_ns": aligned_start_ns,
                "sample_rate_hz": 16000,
                "channels": 1,
                "encoding": "pcm16le",
                "samples_b64": encode_pcm16le_b64(samples),
                "sequence": 1,
            },
        )
    )
    second = await fusion.ingest(
        IngestFrameRequest(
            node=node,
            frame={
                "start_time_ns": aligned_start_ns + 5_000_000_000,
                "sample_rate_hz": 16000,
                "channels": 1,
                "encoding": "pcm16le",
                "samples_b64": encode_pcm16le_b64(samples),
                "sequence": 2,
            },
        )
    )

    assert first.triggered is True
    assert second.triggered is True

    await asyncio.sleep(0.25)

    status = await fusion.status()
    assert classifier.calls == 2
    assert status["metrics"]["birdnet_chunk_dispatches_suppressed"] == 0

    await fusion.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_birdnet_chunked_dispatch_limits_same_chunk_retries(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "fusion_chunk_retry_limited.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        trigger_rms=0.001,
        trigger_cooldown_seconds=0.0,
        localization_window_seconds=0.04,
        classification_window_seconds=30.0,
        max_sensor_buffer_seconds=32.0,
        classifier_backend="birdnet",
        birdnet_chunked_dispatch_enabled=True,
        birdnet_chunk_overlap_seconds=2.0,
        birdnet_chunk_max_retries_per_chunk=1,
        birdnet_chunk_min_retry_progress_seconds=0.0,
        preprocess_enabled=False,
        fusion_worker_count=1,
    )
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)

    storage = Storage(settings.db_path)
    await storage.initialize()

    classifier = _AlwaysUnknownClassifier()
    fusion = FusionNode(
        settings=settings,
        registry=NodeRegistry(),
        buffer=MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds),
        localizer=LocalizationEngine(max_tau_s=0.03),
        classifier=classifier,
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=lambda payload: asyncio.sleep(0, result=None),
        coordinate_frame=LocalCoordinateFrame(origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"),
        zone_matcher=ZoneMatcher(storage=storage),
    )
    await fusion.start()

    node = NodeSpec(
        id="chunk-retry-limit-node",
        node_type=NodeType.POINT,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
        metadata={},
    )
    t = np.arange(1024, dtype=np.float32) / 16000.0
    samples = (0.4 * np.sin(2.0 * np.pi * 800.0 * t)).reshape(1, -1).astype(np.float32)
    stride_ns = int((settings.classification_window_seconds - settings.birdnet_chunk_overlap_seconds) * 1_000_000_000)
    aligned_start_ns = 1_739_810_700_000_000_000
    aligned_start_ns -= aligned_start_ns % stride_ns

    for sequence, start_time_ns in enumerate(
        (
            aligned_start_ns,
            aligned_start_ns + 10_000_000_000,
            aligned_start_ns + 20_000_000_000,
        ),
        start=1,
    ):
        response = await fusion.ingest(
            IngestFrameRequest(
                node=node,
                frame={
                    "start_time_ns": start_time_ns,
                    "sample_rate_hz": 16000,
                    "channels": 1,
                    "encoding": "pcm16le",
                    "samples_b64": encode_pcm16le_b64(samples),
                    "sequence": sequence,
                },
            )
        )
        assert response.triggered is True

    await asyncio.sleep(0.25)

    status = await fusion.status()
    assert classifier.calls == 2
    assert status["metrics"]["birdnet_chunk_dispatches_suppressed"] == 1

    await fusion.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_birdnet_chunked_dispatch_does_not_retry_after_classifier_error(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "fusion_chunk_error_no_retry.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        trigger_rms=0.001,
        trigger_cooldown_seconds=0.0,
        localization_window_seconds=0.04,
        classification_window_seconds=30.0,
        max_sensor_buffer_seconds=32.0,
        classifier_backend="birdnet",
        birdnet_chunked_dispatch_enabled=True,
        birdnet_chunk_overlap_seconds=2.0,
        birdnet_chunk_retry_on_classifier_error=False,
        preprocess_enabled=False,
        fusion_worker_count=1,
    )
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)

    storage = Storage(settings.db_path)
    await storage.initialize()

    classifier = _AlwaysRaisingClassifier()
    fusion = FusionNode(
        settings=settings,
        registry=NodeRegistry(),
        buffer=MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds),
        localizer=LocalizationEngine(max_tau_s=0.03),
        classifier=classifier,
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=lambda payload: asyncio.sleep(0, result=None),
        coordinate_frame=LocalCoordinateFrame(origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"),
        zone_matcher=ZoneMatcher(storage=storage),
    )
    await fusion.start()

    node = NodeSpec(
        id="chunk-error-no-retry-node",
        node_type=NodeType.POINT,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
        metadata={},
    )
    t = np.arange(1024, dtype=np.float32) / 16000.0
    samples = (0.4 * np.sin(2.0 * np.pi * 1000.0 * t)).reshape(1, -1).astype(np.float32)
    stride_ns = int((settings.classification_window_seconds - settings.birdnet_chunk_overlap_seconds) * 1_000_000_000)
    aligned_start_ns = 1_739_810_800_000_000_000
    aligned_start_ns -= aligned_start_ns % stride_ns

    for sequence, start_time_ns in enumerate(
        (
            aligned_start_ns,
            aligned_start_ns + 5_000_000_000,
        ),
        start=1,
    ):
        response = await fusion.ingest(
            IngestFrameRequest(
                node=node,
                frame={
                    "start_time_ns": start_time_ns,
                    "sample_rate_hz": 16000,
                    "channels": 1,
                    "encoding": "pcm16le",
                    "samples_b64": encode_pcm16le_b64(samples),
                    "sequence": sequence,
                },
            )
        )
        assert response.triggered is True

    await asyncio.sleep(0.25)

    status = await fusion.status()
    assert classifier.calls == 1
    assert status["metrics"]["classification_failures"] == 1
    assert status["metrics"]["birdnet_chunk_dispatches_suppressed"] == 1

    await fusion.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_single_sensor_classification_only_detection_does_not_create_track(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "fusion_single_sensor.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        trigger_rms=0.001,
        trigger_cooldown_seconds=0.0,
        localization_window_seconds=0.04,
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

    node = NodeSpec(
        id="point-single-sensor",
        node_type=NodeType.POINT,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
        metadata={},
    )
    samples = np.random.default_rng(31415).normal(0.0, 0.2, size=(1, 1024)).astype(np.float32)

    response = await fusion.ingest(
        IngestFrameRequest(
            node=node,
            frame={
                "start_time_ns": 1_739_810_250_000_000_000,
                "sample_rate_hz": 16000,
                "channels": 1,
                "encoding": "pcm16le",
                "samples_b64": encode_pcm16le_b64(samples),
                "sequence": 1,
            },
        )
    )

    assert response.accepted is True
    assert response.triggered is True

    await asyncio.sleep(0.15)

    detections = await storage.list_detections(limit=10)
    assert len(detections) == 1
    assert detections[0]["track_id"] is None
    assert detections[0]["feature_summary"]["capability_tier"] == "classification_only"

    tracks = await storage.list_tracks(limit=10)
    assert tracks == []

    await fusion.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_detection_feature_summary_flags_reconstructed_audio_gap(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "fusion_audio_quality.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        trigger_rms=0.001,
        trigger_cooldown_seconds=0.0,
        localization_window_seconds=0.04,
        classification_window_seconds=0.70,
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

    node = NodeSpec(
        id="point-audio-gap",
        node_type=NodeType.POINT,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
        metadata={},
    )
    sample_rate_hz = 16_000
    frame_samples = 1024
    gap_start_index = 12000
    start_time_ns = 1_739_810_300_000_000_000
    second_start_ns = start_time_ns + int(round((gap_start_index / sample_rate_hz) * 1_000_000_000))

    first = await fusion.ingest(
        IngestFrameRequest(
            node=node,
            frame={
                "start_time_ns": start_time_ns,
                "sample_rate_hz": sample_rate_hz,
                "channels": 1,
                "encoding": "pcm16le",
                "samples_b64": encode_pcm16le_b64(np.zeros((1, frame_samples), dtype=np.float32)),
                "sequence": 1,
                "start_sample_index": 0,
                "end_sample_index": frame_samples,
            },
        )
    )
    assert first.triggered is False

    second = await fusion.ingest(
        IngestFrameRequest(
            node=node,
            frame={
                "start_time_ns": second_start_ns,
                "sample_rate_hz": sample_rate_hz,
                "channels": 1,
                "encoding": "pcm16le",
                "samples_b64": encode_pcm16le_b64(
                    np.random.default_rng(991).normal(0.0, 0.2, size=(1, frame_samples)).astype(np.float32)
                ),
                "sequence": 2,
                "start_sample_index": gap_start_index,
                "end_sample_index": gap_start_index + frame_samples,
            },
        )
    )
    assert second.triggered is True

    await asyncio.sleep(0.15)

    detections = await storage.list_detections(limit=10)
    assert len(detections) == 1
    audio_quality = detections[0]["feature_summary"]["audio_quality"]
    assert audio_quality["degraded"] is True
    assert audio_quality["missing_ratio"] > 0.05
    assert audio_quality["max_gap_seconds"] > 0.25

    await fusion.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_fusion_ingest_deduplicates_replayed_frame(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "fusion_dedupe.db",
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

    node = NodeSpec(
        id="point-dedupe",
        node_type=NodeType.POINT,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
        metadata={},
    )
    samples = np.random.default_rng(22).normal(0.0, 0.25, size=(1, 1024)).astype(np.float32)

    request = IngestFrameRequest(
        node=node,
        frame={
            "start_time_ns": 1_739_810_200_000_000_000,
            "sample_rate_hz": 16000,
            "channels": 1,
            "encoding": "pcm16le",
            "samples_b64": encode_pcm16le_b64(samples),
            "sequence": 44,
        },
    )
    first = await fusion.ingest(request)
    second = await fusion.ingest(request)

    assert first.accepted is True
    assert first.duplicate is False
    assert second.accepted is True
    assert second.duplicate is True

    db = storage._require_db()
    observations_row = await (
        await db.execute(
            """
            SELECT COUNT(1) AS c
            FROM observations
            WHERE node_id = ?
            """,
            (node.id,),
        )
    ).fetchone()
    receipts_row = await (
        await db.execute(
            """
            SELECT COUNT(1) AS c
            FROM ingested_frames
            WHERE node_id = ?
            """,
            (node.id,),
        )
    ).fetchone()
    assert int(observations_row["c"]) == 1
    assert int(receipts_row["c"]) == 1

    await storage.close()


@pytest.mark.asyncio
async def test_fusion_ingest_skips_observation_rows_for_staged_journal_profile(tmp_path: Path) -> None:
    settings = Settings(
        runtime_profile="birdnet_hybrid_production",
        db_path=tmp_path / "fusion_journal_stage.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        trigger_rms=0.001,
        trigger_cooldown_seconds=0.0,
        localization_window_seconds=0.04,
        max_sensor_buffer_seconds=2.0,
        fusion_worker_count=1,
        fusion_event_queue_size=8,
        ingest_storage_mode="journal",
        direct_ingest_enabled=False,
    )
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)

    assert settings.persist_observations_on_ingest is False

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

    node = NodeSpec(
        id="point-journal-stage",
        node_type=NodeType.POINT,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
        metadata={},
    )
    samples = np.random.default_rng(71).normal(0.0, 0.25, size=(1, 1024)).astype(np.float32)

    response = await fusion.ingest(
        IngestFrameRequest(
            node=node,
            frame={
                "start_time_ns": 1_739_810_400_000_000_000,
                "sample_rate_hz": 16000,
                "channels": 1,
                "encoding": "pcm16le",
                "samples_b64": encode_pcm16le_b64(samples),
                "sequence": 77,
            },
        )
    )

    assert response.accepted is True
    assert response.duplicate is False

    db = storage._require_db()
    observations_row = await (
        await db.execute(
            """
            SELECT COUNT(1) AS c
            FROM observations
            WHERE node_id = ?
            """,
            (node.id,),
        )
    ).fetchone()
    receipts_row = await (
        await db.execute(
            """
            SELECT COUNT(1) AS c
            FROM ingested_frames
            WHERE node_id = ?
            """,
            (node.id,),
        )
    ).fetchone()
    assert int(observations_row["c"]) == 0
    assert int(receipts_row["c"]) == 1

    await storage.close()


@pytest.mark.asyncio
async def test_fusion_ingests_rust_localized_render_directly(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "fusion_rust_render.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        fusion_worker_count=1,
        fusion_event_queue_size=8,
    )
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)

    storage = Storage(settings.db_path)
    await storage.initialize()

    buffer = MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds)
    fusion = FusionNode(
        settings=settings,
        registry=NodeRegistry(),
        buffer=buffer,
        localizer=LocalizationEngine(max_tau_s=0.03),
        classifier=HeuristicClassifier(),
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=lambda payload: asyncio.sleep(0, result=None),
        coordinate_frame=LocalCoordinateFrame(origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"),
        zone_matcher=ZoneMatcher(storage=storage),
    )
    await fusion.start()

    node = NodeSpec(
        id="sirith-rust-render",
        node_type=NodeType.SIRITH_TETRA,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=[
            (0.0, 0.05, 0.0),
            (0.0433, 0.025, 0.0),
            (0.0, 0.0, 0.0),
            (0.02165, 0.025, 0.04082),
        ],
        capabilities=["audio"],
        metadata={},
    )
    audio = np.random.default_rng(901).normal(0.0, 0.3, size=48_000).astype(np.float32)

    await fusion.ingest_localized_render(
        LocalizedClassifierRenderRequest(
            manifest_id="manifest-rust-render-1",
            node=node,
            event_time_ns=1_739_950_000_000_000_000,
            sample_rate_hz=48_000,
            decoded_audio=audio,
            localization_position_m=(1.5, -0.5, 0.0),
            localization_confidence=0.87,
            localization_gdop=1.2,
            localization_method="rust_srp_phat",
            render_kind="birdnet_hybrid_spatial_blend",
            environment={"temperature_c": 18.0, "humidity_fraction": 0.4},
        )
    )

    detections = await storage.list_detections(limit=10)
    assert len(detections) == 1
    detection = detections[0]
    assert detection["source_node_id"] == node.id
    assert detection["feature_summary"]["localization_method"] == "rust_srp_phat"
    assert detection["feature_summary"]["rust_render_kind"] == "birdnet_hybrid_spatial_blend"
    assert tuple(detection["position_m"]) == pytest.approx((1.5, -0.5, 0.0))

    sensor_descriptors = await fusion.registry.sensors_for_node(node.id)
    summary = await buffer.summarize_sensors(
        sensor_ids=[descriptor.sensor_id for descriptor in sensor_descriptors],
        now_ns=1_739_950_000_000_000_000,
    )
    assert summary["active_sensor_count"] == 1
    assert summary["sample_rate_hz"] == 48_000
    assert summary["rms"] is not None

    await fusion.stop()
    await storage.close()
