from __future__ import annotations

import asyncio
import time
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
    def __init__(
        self,
        reference_sensor: str,
        *,
        confidence: float = 0.9,
        wavelength_factor: float | None = None,
        dominant_frequency_hz: float | None = None,
        alias_cutoff_hz: float | None = None,
    ) -> None:
        self.reference_sensor = reference_sensor
        self.confidence = confidence
        self.wavelength_factor = wavelength_factor
        self.dominant_frequency_hz = dominant_frequency_hz
        self.alias_cutoff_hz = alias_cutoff_hz

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
            confidence=self.confidence,
            gdop=1.0,
            reference_sensor=self.reference_sensor,
            tdoa_s={},
            wavelength_factor=self.wavelength_factor,
            dominant_frequency_hz=self.dominant_frequency_hz,
            alias_cutoff_hz=self.alias_cutoff_hz,
        )


class _LatencyInjectedFixedReferenceLocalizer(_FixedReferenceLocalizer):
    def __init__(self, reference_sensor: str, *, latency_seconds: float, **kwargs: object) -> None:
        super().__init__(reference_sensor, **kwargs)
        self.latency_seconds = latency_seconds

    def localize(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
    ) -> LocalizationResult:
        time.sleep(self.latency_seconds)
        return super().localize(
            sensor_positions,
            sensor_windows,
            sample_rate_hz,
            temperature_c,
            humidity_fraction,
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
        classification_window_seconds=0.0,
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
            "time_quality": "gps_locked",
        },
    )

    response = await fusion.ingest(request)

    assert response.accepted is True
    assert response.frame_energy > 0.0
    assert response.triggered is True
    assert response.queued_event_id is not None
    assert all(
        grade.value == "gps_pps"
        for grade in (await registry.sensor_sync_grades()).values()
    )

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
        classification_window_seconds=0.0,
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
        classification_window_seconds=0.0,
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
        classification_window_seconds=0.0,
        max_sensor_buffer_seconds=2.0,
        fusion_worker_count=1,
        fusion_localization_queue_size=1,
        fusion_classification_queue_size=1,
        fusion_rules_queue_size=1,
        drop_on_backpressure=True,
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
        localizer=_LatencyInjectedFixedReferenceLocalizer(
            reference_sensor="reuse-node:ch0",
            latency_seconds=0.005,
        ),
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
    assert status["metrics"]["localization_stage_in"] >= 1
    assert status["metrics"]["localization_stage_out"] >= 1
    assert status["metrics"]["localization_stage_total_time_ms"] >= 4.0
    assert status["metrics"]["localization_stage_max_time_ms"] >= 4.0
    assert status["metrics"]["stage_timeout_count"] == 0

    await fusion.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_fusion_records_wavelength_alias_metrics_and_features(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "fusion_wavelength.db",
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

    fusion = FusionNode(
        settings=settings,
        registry=NodeRegistry(),
        buffer=MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds),
        localizer=_FixedReferenceLocalizer(
            reference_sensor="alias-node:ch0",
            confidence=0.36,
            wavelength_factor=0.4,
            dominant_frequency_hz=12_000.0,
            alias_cutoff_hz=3_200.0,
        ),
        classifier=_CountingClassifier(),
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=lambda payload: asyncio.sleep(0, result=None),
        coordinate_frame=LocalCoordinateFrame(origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"),
        zone_matcher=ZoneMatcher(storage=storage),
    )
    await fusion.start()

    node = NodeSpec(
        id="alias-node",
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
    time_axis = np.arange(1024, dtype=np.float32) / 16_000.0
    channels = np.stack(
        [
            0.4 * np.sin(2.0 * np.pi * 1200.0 * time_axis),
            0.2 * np.sin(2.0 * np.pi * 1200.0 * time_axis),
            0.1 * np.sin(2.0 * np.pi * 1200.0 * time_axis),
            0.05 * np.sin(2.0 * np.pi * 1200.0 * time_axis),
        ]
    ).astype(np.float32)

    response = await fusion.ingest(
        IngestFrameRequest(
            node=node,
            frame={
                "start_time_ns": 1_739_810_320_000_000_000,
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
    assert status["metrics"]["localization_band_aliased_count"] == 1

    detections = await storage.list_detections(limit=10)
    assert len(detections) == 1
    feature_summary = detections[0]["feature_summary"]
    assert feature_summary["wavelength_factor"] == pytest.approx(0.4)
    assert feature_summary["dominant_frequency_hz"] == pytest.approx(12_000.0)
    assert feature_summary["alias_cutoff_hz"] == pytest.approx(3_200.0)

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

    status = await fusion.status()
    assert status["metrics"]["frames_zero_padded_degraded"] >= 1

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
        classification_window_seconds=0.0,
        max_sensor_buffer_seconds=2.0,
        fusion_worker_count=1,
        fusion_event_queue_size=8,
        persist_observations_on_ingest=True,
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


def test_settings_default_skip_observation_rows_for_birdnet_hybrid_direct_ingest(tmp_path: Path) -> None:
    settings = Settings(
        runtime_profile="birdnet_hybrid_production",
        db_path=tmp_path / "fusion_birdnet_direct.db",
        snippet_dir=tmp_path / "snippets",
        direct_ingest_enabled=True,
        ingest_storage_mode="spool",
    )

    assert settings.persist_observations_on_ingest is False


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

    ingest_before_ns = time.time_ns()
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
            localization_position_covariance_m2=[
                [2.0, 0.1, 0.0],
                [0.1, 1.5, 0.0],
                [0.0, 0.0, 3.0],
            ],
            localization_range_observability=0.42,
            localization_residual_rms_seconds=2.5e-4,
            localization_range_projection_mode="prior_projected",
            localization_method="rust_srp_phat",
            render_kind="birdnet_hybrid_spatial_blend",
            environment={"temperature_c": 18.0, "humidity_fraction": 0.4},
        )
    )
    ingest_after_ns = time.time_ns()

    detections = await storage.list_detections(limit=10)
    assert len(detections) == 1
    detection = detections[0]
    assert detection["source_node_id"] == node.id
    assert detection["feature_summary"]["localization_method"] == "rust_srp_phat"
    # Legacy Rust "prior_projected" is canonicalized to "range_asymptotic" and, because
    # range is unobservable, the path-agnostic haircut caps confidence (0.87 -> <=0.20)
    # and range_observability (0.42 -> <=0.05) — the same treatment the Python solver applies.
    assert detection["confidence"] <= 0.20
    assert detection["feature_summary"]["localization_range_observability"] == pytest.approx(0.05)
    assert detection["feature_summary"]["localization_residual_rms_seconds"] == pytest.approx(2.5e-4)
    assert detection["feature_summary"]["localization_range_projection_mode"] == "range_asymptotic"
    assert detection["spatial_display_mode"] == "bearing_only"
    assert detection["track_id"] is None
    assert detection["feature_summary"]["position_geo_uncertainty"]["horizontal_major_std_m"] > 0.0
    assert detection["feature_summary"]["rust_render_kind"] == "birdnet_hybrid_spatial_blend"
    assert tuple(detection["position_m"]) == pytest.approx((1.5, -0.5, 0.0))
    assert np.allclose(
        np.asarray(detection["position_covariance_m2"], dtype=np.float64),
        np.asarray([[2.0, 0.1, 0.0], [0.1, 1.5, 0.0], [0.0, 0.0, 3.0]], dtype=np.float64),
    )
    assert await storage.list_tracks(limit=10) == []

    nodes = await storage.list_nodes(limit=10)
    assert len(nodes) == 1
    assert nodes[0]["id"] == node.id
    last_seen_ns = int(nodes[0]["last_seen_ns"])
    assert ingest_before_ns <= last_seen_ns <= ingest_after_ns

    sensor_descriptors = await fusion.registry.sensors_for_node(node.id)
    summary = await buffer.summarize_sensors(
        sensor_ids=[descriptor.sensor_id for descriptor in sensor_descriptors],
        now_ns=ingest_after_ns,
    )
    assert summary["active_sensor_count"] == len(sensor_descriptors)
    assert summary["sample_rate_hz"] == 48_000
    assert summary["last_sample_time_ns"] is not None
    assert summary["age_seconds"] is not None
    assert summary["age_seconds"] <= 1.0
    assert summary["rms"] is not None

    await fusion.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_fusion_rust_render_python_cartesian_solver_rehomes_solve(tmp_path: Path) -> None:
    import itertools

    settings = Settings(
        db_path=tmp_path / "fusion_rust_render_python_cartesian.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        fusion_worker_count=1,
        fusion_event_queue_size=8,
        localization_single_node_solver="python_cartesian",
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

    mic_offsets = [
        (0.0, 0.050, 0.0),
        (0.0433, 0.025, 0.0),
        (0.0, 0.0, 0.0),
        (0.02165, 0.025, 0.04082),
    ]
    node = NodeSpec(
        id="sirith-python-cartesian",
        node_type=NodeType.SIRITH_TETRA,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=mic_offsets,
        capabilities=["audio"],
        metadata={},
    )

    sound_speed_mps = 343.2
    mic_positions = [np.asarray(offset, dtype=np.float64) for offset in mic_offsets]
    centroid = np.mean(mic_positions, axis=0)
    source_m = np.array([0.20, 0.10, 0.15])
    pair_tdoas = []
    for channel_a, channel_b in itertools.combinations(range(4), 2):
        lag = (
            float(np.linalg.norm(source_m - mic_positions[channel_a]))
            - float(np.linalg.norm(source_m - mic_positions[channel_b]))
        ) / sound_speed_mps
        pair_tdoas.append(
            {"ch_a": channel_a, "ch_b": channel_b, "lag_seconds": lag, "confidence": 0.8}
        )
    steering = source_m - centroid
    steering = steering / float(np.linalg.norm(steering))

    audio = np.random.default_rng(902).normal(0.0, 0.3, size=48_000).astype(np.float32)
    await fusion.ingest_localized_render(
        LocalizedClassifierRenderRequest(
            manifest_id="manifest-python-cartesian-1",
            node=node,
            event_time_ns=1_739_950_000_000_000_000,
            sample_rate_hz=48_000,
            decoded_audio=audio,
            # The sidecar's own position is deliberately wrong; the python_cartesian
            # solver must override it from the pairwise TDOAs + bearing.
            localization_position_m=(99.0, 99.0, 99.0),
            localization_confidence=0.9,
            localization_range_projection_mode="range_refined",
            localization_pair_tdoas=pair_tdoas,
            localization_steering_direction=tuple(steering),
            localization_sound_speed_mps=sound_speed_mps,
            localization_method="rust_srp_phat",
            environment={"temperature_c": 18.0, "humidity_fraction": 0.4},
        )
    )

    assert fusion._metrics.localization_single_node_python_solved_count == 1
    assert fusion._metrics.localization_single_node_python_fallback_count == 0

    detections = await storage.list_detections(limit=10)
    assert len(detections) == 1
    detection = detections[0]
    assert detection["feature_summary"]["localization_method"] == "python_cartesian_rust_tdoa"
    # The Rust-supplied (99,99,99) must have been discarded for the Python solve.
    position = np.asarray(detection["position_m"], dtype=np.float64)
    assert np.linalg.norm(position - np.array([99.0, 99.0, 99.0])) > 1.0
    recovered_bearing = (position - centroid) / float(np.linalg.norm(position - centroid))
    true_bearing = (source_m - centroid) / float(np.linalg.norm(source_m - centroid))
    assert float(np.dot(recovered_bearing, true_bearing)) > 0.9

    await fusion.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_rust_render_production_classification_is_coalesced(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "fusion_rust_render_coalesced.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        runtime_profile="birdnet_hybrid_production",
        fusion_worker_count=1,
        fusion_classification_queue_size=8,
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
        id="sirith-rust-render-coalesce",
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
    audio = np.random.default_rng(902).normal(0.0, 0.3, size=16_000).astype(np.float32)

    for index in range(3):
        await fusion.ingest_localized_render(
            LocalizedClassifierRenderRequest(
                manifest_id=f"manifest-rust-render-coalesce-{index}",
                node=node,
                event_time_ns=1_739_950_000_000_000_000 + index,
                sample_rate_hz=16_000,
                decoded_audio=audio,
                localization_position_m=(1.5, -0.5, 0.0),
                localization_confidence=0.87,
                localization_gdop=1.2,
                localization_method="rust_srp_phat",
                render_kind="birdnet_hybrid_spatial_blend",
                environment={"temperature_c": 18.0, "humidity_fraction": 0.4},
            )
        )

    status = await fusion.status()
    # ingest_localized_render classifies synchronously, so the queue stays empty
    assert status["queue"]["classification_depth"] == 0
    assert status["metrics"]["birdnet_chunk_dispatches_suppressed"] == 2

    nodes = await storage.list_nodes(limit=10)
    assert len(nodes) == 1
    assert nodes[0]["id"] == node.id

    await storage.close()


@pytest.mark.asyncio
async def test_fusion_ingests_rust_classifier_render_fallback_as_omni(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "fusion_rust_render_fallback.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
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
        id="sirith-rust-point-fallback",
        node_type=NodeType.POINT,
        position_m=(3.0, 1.0, 2.0),
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
        metadata={},
    )
    audio = np.random.default_rng(903).normal(0.0, 0.3, size=16_000).astype(np.float32)

    await fusion.ingest_localized_render(
        LocalizedClassifierRenderRequest(
            manifest_id="manifest-rust-point-fallback-1",
            node=node,
            event_time_ns=1_739_950_000_000_000_100,
            sample_rate_hz=16_000,
            decoded_audio=audio,
            localization_position_m=node.position_m,
            localization_confidence=0.0,
            localization_gdop=float("inf"),
            localization_method="rust_classifier_render_fallback",
            source_type="raw_sensor",
            reporting_modality="omni",
            fallback_reason="single_point_node",
            render_kind="birdnet_omni_fallback",
            environment={"temperature_c": 18.0, "humidity_fraction": 0.4},
        )
    )

    detections = await storage.list_detections(limit=10)
    assert len(detections) == 1
    detection = detections[0]
    assert detection["source_node_id"] == node.id
    assert detection["reporting_modality"] == "omni"
    assert detection["feature_summary"]["localization_method"] == "rust_classifier_render_fallback"
    assert detection["feature_summary"]["rust_fallback_reason"] == "single_point_node"
    assert tuple(detection["position_m"]) == pytest.approx(node.position_m)

    await fusion.stop()
    await storage.close()


def _make_minimal_fusion_node(tmp_path: Path):
    """Build a FusionNode wired with real components but no nodes registered.

    The silent-drop instrumentation can be exercised by calling its helpers
    directly; we don't need a full ingest round-trip for these unit tests.
    """
    settings = Settings(
        db_path=tmp_path / "fusion_drops.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        trigger_rms=0.001,
        trigger_cooldown_seconds=0.0,
        localization_window_seconds=0.04,
        classification_window_seconds=0.0,
        max_sensor_buffer_seconds=2.0,
        fusion_worker_count=1,
    )
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)
    return settings


@pytest.mark.asyncio
async def test_record_silent_drop_increments_counters_and_rate_limits_log(
    tmp_path: Path, caplog
) -> None:
    """`_record_silent_drop` is the single source of truth for silent-drop
    visibility — every silent return in the pipeline routes through it. It must
    bump the right counter and emit a rate-limited WARNING."""
    import logging as _logging

    settings = _make_minimal_fusion_node(tmp_path)
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

    # Tighten the rate-limit so the test can observe a single warning per reason
    # without sleeping.
    fusion._drop_warning_interval_seconds = 0.0

    with caplog.at_level(_logging.WARNING, logger="minimappr.core.fusion_node"):
        fusion._record_silent_drop(
            stage="localization", reason="no_sensors", candidate_id="evt-1"
        )
        fusion._record_silent_drop(
            stage="localization", reason="no_window", candidate_id="evt-2"
        )
        fusion._record_silent_drop(
            stage="classification", reason="chunk_suppressed", candidate_id="evt-3"
        )
        fusion._record_silent_drop(
            stage="rules", reason="suppressed_by_zone", candidate_id="evt-4"
        )

    metrics = fusion._metrics
    assert metrics.localization_drops_by_reason == {"no_sensors": 1, "no_window": 1}
    assert metrics.classification_drops_by_reason == {"chunk_suppressed": 1}
    assert metrics.rules_drops_by_reason == {"suppressed_by_zone": 1}

    warning_messages = [r.message for r in caplog.records if r.levelno == _logging.WARNING]
    assert sum(1 for m in warning_messages if m == "Silent pipeline drop") == 4

    await storage.close()


@pytest.mark.asyncio
async def test_localize_candidate_no_sensors_drop_visible_in_metrics(
    tmp_path: Path,
) -> None:
    """End-to-end through the helper: when no sensors are registered, the
    silent-None return path increments the `no_sensors` counter."""
    from minimappr.core.fusion_node import EventCandidate
    from minimappr.models import TimeQuality

    settings = _make_minimal_fusion_node(tmp_path)
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

    candidate = EventCandidate(
        id="evt-empty-registry",
        source_node_id="ghost-node",
        event_time_ns=1_739_900_000_000_000_000,
        sample_rate_hz=16_000,
        source_type="raw_sensor",
        time_quality=TimeQuality.GPS_LOCKED,
        source_observation_ids=[],
    )

    result = await fusion._localize_candidate(candidate)

    assert result is None
    assert fusion._metrics.localization_drops_by_reason.get("no_sensors") == 1

    await storage.close()


@pytest.mark.asyncio
async def test_fusion_status_exposes_health_and_buffer_state(
    tmp_path: Path,
) -> None:
    """The new `health` and `buffer_state` blocks must surface through
    FusionNode.status() so `/api/v1/fusion/status` carries the silent-stall
    watchdog signals."""
    settings = _make_minimal_fusion_node(tmp_path)
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

    status = await fusion.status()

    assert "health" in status
    health = status["health"]
    # No traffic yet — both timestamps are 0 → seconds_since_* is None.
    assert health["seconds_since_last_emission"] is None
    assert health["seconds_since_last_trigger"] is None
    assert health["active_drought"] is False

    assert "buffer_state" in status
    assert isinstance(status["buffer_state"], list)

    # Simulate trigger-without-emission to flip active_drought via timestamps.
    now_ns = time.time_ns()
    fusion._metrics.last_trigger_enqueue_ns = now_ns
    fusion._metrics.last_detection_emission_ns = now_ns - 120 * 1_000_000_000

    status2 = await fusion.status()
    assert status2["health"]["active_drought"] is True
    assert status2["health"]["seconds_since_last_emission"] >= 120.0

    await storage.close()


@pytest.mark.asyncio
async def test_await_window_coverage_returns_true_when_all_sensors_covered(
    tmp_path: Path,
) -> None:
    """Helper returns immediately when every sensor already has coverage
    straddling the requested centered window — no sleeping, no polling."""
    settings = _make_minimal_fusion_node(tmp_path)
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
        coordinate_frame=LocalCoordinateFrame(
            origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"
        ),
        zone_matcher=ZoneMatcher(storage=storage),
    )

    sample_rate = 16_000
    samples_per_chunk = 1024
    base_ns = 1_739_900_000_000_000_000
    chunk_duration_ns = int(samples_per_chunk / sample_rate * 1_000_000_000)
    # Append two consecutive chunks per sensor so the buffer covers a 100+ ms
    # span centered around event_time below.
    for sensor_id in ("a", "b", "c"):
        for k in range(3):
            await buffer.append(
                sensor_id=sensor_id,
                sample_rate_hz=sample_rate,
                start_time_ns=base_ns + k * chunk_duration_ns,
                samples=np.ones(samples_per_chunk, dtype=np.float32),
            )

    center_time_ns = base_ns + chunk_duration_ns  # second chunk center
    started = time.monotonic()
    ready, snapshot = await fusion._await_window_coverage(
        sensor_ids=["a", "b", "c"],
        center_time_ns=center_time_ns,
        window_seconds=0.04,
        timeout_s=0.5,
    )
    elapsed = time.monotonic() - started

    assert ready is True
    assert len(snapshot) == 3
    assert all(s["present"] for s in snapshot)
    # Helper must short-circuit on the first poll when coverage is already
    # there — no sleep, no second snapshot.
    assert elapsed < 0.05

    await storage.close()


@pytest.mark.asyncio
async def test_await_window_coverage_waits_until_late_sensor_catches_up(
    tmp_path: Path,
) -> None:
    """A trailing sensor arriving mid-wait should unblock the helper before
    the deadline — this is the canonical race the fix targets."""
    settings = _make_minimal_fusion_node(tmp_path)
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
        coordinate_frame=LocalCoordinateFrame(
            origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"
        ),
        zone_matcher=ZoneMatcher(storage=storage),
    )

    sample_rate = 16_000
    samples_per_chunk = 1024
    base_ns = 1_739_900_000_000_000_000
    chunk_duration_ns = int(samples_per_chunk / sample_rate * 1_000_000_000)

    # Sensors a, b cover [base, base + 3*chunk]; c will arrive late.
    for sensor_id in ("a", "b"):
        for k in range(3):
            await buffer.append(
                sensor_id=sensor_id,
                sample_rate_hz=sample_rate,
                start_time_ns=base_ns + k * chunk_duration_ns,
                samples=np.ones(samples_per_chunk, dtype=np.float32),
            )

    center_time_ns = base_ns + chunk_duration_ns

    async def _late_append() -> None:
        await asyncio.sleep(0.04)
        for k in range(3):
            await buffer.append(
                sensor_id="c",
                sample_rate_hz=sample_rate,
                start_time_ns=base_ns + k * chunk_duration_ns,
                samples=np.ones(samples_per_chunk, dtype=np.float32),
            )

    started = time.monotonic()
    late_task = asyncio.create_task(_late_append())
    ready, snapshot = await fusion._await_window_coverage(
        sensor_ids=["a", "b", "c"],
        center_time_ns=center_time_ns,
        window_seconds=0.04,
        timeout_s=0.5,
        poll_interval_s=0.005,
    )
    elapsed = time.monotonic() - started
    await late_task

    assert ready is True
    # Helper should not have returned before the late append fired (~40 ms),
    # and should not have spent more than the budget.
    assert 0.03 < elapsed < 0.3
    by_id = {s["sensor_id"]: s for s in snapshot}
    assert all(by_id[sid].get("present") for sid in ("a", "b", "c"))

    await storage.close()


@pytest.mark.asyncio
async def test_await_window_coverage_times_out_when_buffer_never_advances(
    tmp_path: Path,
) -> None:
    """If the buffer never catches up to target_end, the helper returns False
    after the timeout — driving the `buffer_lag_timeout` drop in the caller."""
    settings = _make_minimal_fusion_node(tmp_path)
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
        coordinate_frame=LocalCoordinateFrame(
            origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"
        ),
        zone_matcher=ZoneMatcher(storage=storage),
    )

    sample_rate = 16_000
    samples_per_chunk = 1024
    base_ns = 1_739_900_000_000_000_000
    # Only sensor a has data; b and c never arrive.
    await buffer.append(
        sensor_id="a",
        sample_rate_hz=sample_rate,
        start_time_ns=base_ns,
        samples=np.ones(samples_per_chunk, dtype=np.float32),
    )

    started = time.monotonic()
    ready, snapshot = await fusion._await_window_coverage(
        sensor_ids=["a", "b", "c"],
        center_time_ns=base_ns,
        window_seconds=0.04,
        timeout_s=0.08,
        poll_interval_s=0.005,
    )
    elapsed = time.monotonic() - started

    assert ready is False
    # Should not return *before* the timeout, and not much after.
    assert 0.06 < elapsed < 0.25
    by_id = {s["sensor_id"]: s for s in snapshot}
    assert by_id["a"]["present"] is True
    assert by_id["b"]["present"] is False
    assert by_id["c"]["present"] is False

    await storage.close()


@pytest.mark.asyncio
async def test_localize_candidate_drops_with_buffer_lag_timeout_reason(
    tmp_path: Path,
) -> None:
    """End-to-end: a candidate whose required window will never be covered
    drops with `buffer_lag_timeout` and the warning carries `buffer_snapshot`
    so operators can see which sensor was lagging."""
    from minimappr.core.fusion_node import EventCandidate
    from minimappr.models import TimeQuality

    settings = _make_minimal_fusion_node(tmp_path)
    storage = Storage(settings.db_path)
    await storage.initialize()
    registry = NodeRegistry()
    buffer = MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds)

    node = NodeSpec(
        id="lag-node",
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
    )
    await registry.upsert(node, last_seen_ns=1_739_900_000_000_000_000)

    fusion = FusionNode(
        settings=settings,
        registry=registry,
        buffer=buffer,
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
    # Cap the wait so the test runs fast.
    fusion.localization_config.localization_buffer_wait_max_seconds = 0.08
    fusion._drop_warning_interval_seconds = 0.0

    # Append only to one sensor — the other 3 stay absent → coverage never
    # reaches min_sensors_for_2d (default 3).
    sample_rate = 16_000
    samples_per_chunk = 1024
    base_ns = 1_739_900_000_000_000_000
    await buffer.append(
        sensor_id="lag-node:ch0",
        sample_rate_hz=sample_rate,
        start_time_ns=base_ns,
        samples=np.ones(samples_per_chunk, dtype=np.float32),
    )

    candidate = EventCandidate(
        id="evt-lag",
        source_node_id="lag-node",
        event_time_ns=base_ns + int(0.02 * 1_000_000_000),
        sample_rate_hz=sample_rate,
        source_type="raw_sensor",
        time_quality=TimeQuality.GPS_LOCKED,
        source_observation_ids=[],
    )

    result = await fusion._localize_candidate(candidate)

    assert result is None
    assert fusion._metrics.localization_drops_by_reason.get("buffer_lag_timeout") == 1
    assert "no_window" not in fusion._metrics.localization_drops_by_reason
    assert "event_too_old" not in fusion._metrics.localization_drops_by_reason

    await storage.close()


@pytest.mark.asyncio
async def test_localize_candidate_drops_with_event_too_old_reason(
    tmp_path: Path,
) -> None:
    """When the candidate's window trailing edge sits before every buffered
    sample, waiting can't help — the helper bails immediately and the caller
    drops with `event_too_old` rather than burning the full timeout."""
    from minimappr.core.fusion_node import EventCandidate
    from minimappr.models import TimeQuality

    settings = _make_minimal_fusion_node(tmp_path)
    storage = Storage(settings.db_path)
    await storage.initialize()
    registry = NodeRegistry()
    buffer = MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds)

    node = NodeSpec(
        id="old-node",
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
    )
    await registry.upsert(node, last_seen_ns=1_739_900_000_000_000_000)

    fusion = FusionNode(
        settings=settings,
        registry=registry,
        buffer=buffer,
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
    fusion.localization_config.localization_buffer_wait_max_seconds = 2.0
    fusion._drop_warning_interval_seconds = 0.0

    sample_rate = 16_000
    samples_per_chunk = 1024
    buffer_anchor_ns = 1_739_900_000_000_000_000
    # Populate sensors with audio starting far in the "future" relative to
    # the candidate's event_time_ns.
    for k in range(4):
        await buffer.append(
            sensor_id=f"old-node:ch{k}",
            sample_rate_hz=sample_rate,
            start_time_ns=buffer_anchor_ns,
            samples=np.ones(samples_per_chunk, dtype=np.float32),
        )

    # event_time_ns is one full second before the buffer's start.
    candidate = EventCandidate(
        id="evt-old",
        source_node_id="old-node",
        event_time_ns=buffer_anchor_ns - 1_000_000_000,
        sample_rate_hz=sample_rate,
        source_type="raw_sensor",
        time_quality=TimeQuality.GPS_LOCKED,
        source_observation_ids=[],
    )

    started = time.monotonic()
    result = await fusion._localize_candidate(candidate)
    elapsed = time.monotonic() - started

    assert result is None
    assert fusion._metrics.localization_drops_by_reason.get("event_too_old") == 1
    # Should NOT have waited the full 2 s timeout — bails on first poll.
    assert elapsed < 0.2

    await storage.close()


# ---------------------------------------------------------------------------
# Fix 4: throttled per-stage exception counter
# ---------------------------------------------------------------------------


def _make_minimal_fusion(tmp_path: Path) -> tuple[FusionNode, Storage]:
    """Return a FusionNode wired with a minimal in-memory config for unit tests."""
    settings = Settings(
        db_path=tmp_path / "fusion.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        trigger_rms=0.001,
        trigger_cooldown_seconds=0.0,
        localization_window_seconds=0.04,
        classification_window_seconds=0.0,
        max_sensor_buffer_seconds=2.0,
        fusion_worker_count=1,
        fusion_event_queue_size=8,
    )
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)

    storage = Storage(settings.db_path)
    coordinate_frame = LocalCoordinateFrame(
        origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"
    )

    fusion = FusionNode(
        settings=settings,
        registry=NodeRegistry(),
        buffer=MultiSensorBuffer(max_duration_seconds=2.0),
        localizer=LocalizationEngine(max_tau_s=0.03),
        classifier=HeuristicClassifier(),
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=None,
        coordinate_frame=coordinate_frame,
        zone_matcher=ZoneMatcher(storage=storage),
    )
    return fusion, storage


def test_record_worker_exception_increments_counter(tmp_path: Path) -> None:
    """_record_worker_exception increments the per-stage exceptions_by_type counter."""
    fusion, _ = _make_minimal_fusion(tmp_path)

    exc = ValueError("boom")
    fusion._record_worker_exception(stage="localization", exc=exc)

    assert fusion._metrics.localization_exceptions_by_type == {"ValueError": 1}
    assert fusion._metrics.classification_exceptions_by_type == {}
    assert fusion._metrics.rules_exceptions_by_type == {}

    # Second call on same type increments, not re-keyed.
    fusion._record_worker_exception(stage="localization", exc=ValueError("again"))
    assert fusion._metrics.localization_exceptions_by_type == {"ValueError": 2}

    # Different exception type gets its own key.
    fusion._record_worker_exception(stage="localization", exc=RuntimeError("other"))
    assert fusion._metrics.localization_exceptions_by_type == {"ValueError": 2, "RuntimeError": 1}


def test_record_worker_exception_throttles_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """_record_worker_exception logs at most once per (stage, exc_type) per interval."""
    import logging

    fusion, _ = _make_minimal_fusion(tmp_path)
    # Force the throttle interval to be very long so repeat calls are suppressed.
    fusion._drop_warning_interval_seconds = 9999.0

    exc = ValueError("boom")
    with caplog.at_level(logging.WARNING, logger="minimappr.core.fusion_node"):
        fusion._record_worker_exception(stage="localization", exc=exc)
        fusion._record_worker_exception(stage="localization", exc=exc)
        fusion._record_worker_exception(stage="localization", exc=exc)

    worker_exc_logs = [r for r in caplog.records if r.message == "Fusion worker exception"]
    assert len(worker_exc_logs) == 1, "Expected exactly one log entry per throttle window"
    assert worker_exc_logs[0].__dict__["exception_type"] == "ValueError"
    assert worker_exc_logs[0].__dict__["stage_name"] == "localization"


def test_record_worker_exception_different_stages_independent(tmp_path: Path) -> None:
    """Exceptions on different stages are tracked independently and don't interfere."""
    fusion, _ = _make_minimal_fusion(tmp_path)

    fusion._record_worker_exception(stage="localization", exc=ValueError("a"))
    fusion._record_worker_exception(stage="classification", exc=ValueError("b"))
    fusion._record_worker_exception(stage="rules", exc=ValueError("c"))

    assert fusion._metrics.localization_exceptions_by_type == {"ValueError": 1}
    assert fusion._metrics.classification_exceptions_by_type == {"ValueError": 1}
    assert fusion._metrics.rules_exceptions_by_type == {"ValueError": 1}


def test_exception_counters_appear_in_status(tmp_path: Path) -> None:
    """FusionMetrics exception dicts are serialized into the status() metrics dict."""
    import dataclasses

    fusion, _ = _make_minimal_fusion(tmp_path)
    fusion._record_worker_exception(stage="classification", exc=RuntimeError("test"))

    metrics = dataclasses.asdict(fusion._metrics)
    assert "classification_exceptions_by_type" in metrics
    assert metrics["classification_exceptions_by_type"] == {"RuntimeError": 1}
    assert "localization_exceptions_by_type" in metrics
    assert "rules_exceptions_by_type" in metrics


def test_fusion_node_module_never_imports_iamf_pipeline_or_writes_wav() -> None:
    """The live FusionNode pipeline (ingest/localize/classify/track/rules)
    must never produce raw recording audio. IamfPipeline.run() is only
    reachable from the capture-session start/stop flow
    (core/capture_session.py, main.py /api/v1/capture/*); pin that
    core/fusion_node.py neither imports it nor writes .wav files directly,
    so this boundary can't silently regress.
    """
    import ast
    import minimappr.core.fusion_node as fusion_node_module

    source_path = Path(fusion_node_module.__file__)
    source_text = source_path.read_text()
    tree = ast.parse(source_text)

    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert not any("iamf_pipeline" in mod for mod in imported_modules), (
        f"core/fusion_node.py must not import iamf_pipeline: {imported_modules}"
    )

    wav_write_markers = ("import wave", "write_wav", '.wav"', ".wav'")
    found = [m for m in wav_write_markers if m in source_text]
    assert not found, f"core/fusion_node.py must not write .wav files directly: {found}"


def _lateral_variance_perp_to_bearing(covariance, bearing_vec) -> float:
    cov = np.asarray(covariance, dtype=np.float64)
    b = np.asarray(bearing_vec, dtype=np.float64)
    b = b / (float(np.linalg.norm(b)) + 1e-12)
    radial_var = float(b @ cov @ b)
    return (float(np.trace(cov)) - radial_var) / 2.0


async def _ingest_single_node_render_covariance(
    tmp_path: Path, *, dominant_frequency_hz: float, db_name: str
):
    import itertools

    settings = Settings(
        db_path=tmp_path / db_name,
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        fusion_worker_count=1,
        fusion_event_queue_size=8,
        localization_single_node_solver="python_cartesian",
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

    mic_offsets = [
        (0.0, 0.050, 0.0),
        (0.0433, 0.025, 0.0),
        (0.0, 0.0, 0.0),
        (0.02165, 0.025, 0.04082),
    ]
    node = NodeSpec(
        id="sirith-freq-scaling",
        node_type=NodeType.SIRITH_TETRA,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=mic_offsets,
        capabilities=["audio"],
        metadata={},
    )
    sound_speed_mps = 343.2
    mic_positions = [np.asarray(off, dtype=np.float64) for off in mic_offsets]
    centroid = np.mean(mic_positions, axis=0)
    source_m = np.array([0.20, 0.10, 0.15])
    pair_tdoas = []
    for a, b in itertools.combinations(range(4), 2):
        lag = (
            float(np.linalg.norm(source_m - mic_positions[a]))
            - float(np.linalg.norm(source_m - mic_positions[b]))
        ) / sound_speed_mps
        pair_tdoas.append({"ch_a": a, "ch_b": b, "lag_seconds": lag, "confidence": 0.8})
    steering = source_m - centroid
    steering = steering / float(np.linalg.norm(steering))
    audio = np.random.default_rng(7).normal(0.0, 0.3, size=48_000).astype(np.float32)

    await fusion.ingest_localized_render(
        LocalizedClassifierRenderRequest(
            manifest_id=f"manifest-freq-{int(dominant_frequency_hz)}",
            node=node,
            event_time_ns=1_739_950_000_000_000_000,
            sample_rate_hz=48_000,
            decoded_audio=audio,
            localization_position_m=(0.20, 0.10, 0.15),
            localization_confidence=0.9,
            localization_range_projection_mode="range_refined",
            localization_pair_tdoas=pair_tdoas,
            localization_steering_direction=tuple(steering),
            localization_sound_speed_mps=sound_speed_mps,
            localization_dominant_frequency_hz=dominant_frequency_hz,
            localization_method="rust_srp_phat",
            environment={"temperature_c": 18.0, "humidity_fraction": 0.4},
        )
    )
    detections = await storage.list_detections(limit=10)
    assert len(detections) == 1
    cov = detections[0]["position_covariance_m2"]
    position = np.asarray(detections[0]["position_m"], dtype=np.float64)
    bearing = position - centroid
    await fusion.stop()
    await storage.close()
    assert cov is not None
    return _lateral_variance_perp_to_bearing(cov, bearing)


@pytest.mark.asyncio
async def test_frequency_covariance_scaling_inflates_lateral_single_node_path(tmp_path: Path) -> None:
    """Phase 1d: on the single-node Rust render seam (fusion_node), a low-frequency
    source (well below the array alias cutoff) yields a laterally inflated covariance
    versus a high-frequency source with the identical TDOA geometry."""
    low = await _ingest_single_node_render_covariance(
        tmp_path, dominant_frequency_hz=250.0, db_name="freq_low.db"
    )
    high = await _ingest_single_node_render_covariance(
        tmp_path, dominant_frequency_hz=12_000.0, db_name="freq_high.db"
    )
    # Below the ~2 kHz alias cutoff the lateral covariance must inflate materially.
    assert low > high * 1.5, f"expected lateral inflation, low={low} high={high}"


def test_frequency_covariance_scaling_inflates_lateral_dispatch_path() -> None:
    """Phase 1d: the multi-node dispatch seam (localization_dispatch) inflates lateral
    covariance for a low-frequency tone relative to a high-frequency tone."""
    from minimappr.core.localization_dispatch import build_localizer_from_settings

    settings = Settings(localization_algorithm="gcc_phat", localization_strategy="fixed")
    dispatcher = build_localizer_from_settings(settings)

    # 5 cm tetra; alias cutoff ~2 kHz. Source in the near field so a covariance exists.
    sensor_positions = {
        "n:ch0": np.array([0.0, 0.050, 0.0]),
        "n:ch1": np.array([0.0433, 0.025, 0.0]),
        "n:ch2": np.array([0.0, 0.0, 0.0]),
        "n:ch3": np.array([0.02165, 0.025, 0.04082]),
    }
    sample_rate_hz = 48_000
    sound_speed_mps = 343.2
    source_m = np.array([1.2, 0.6, 0.9])
    n = 4096
    t = np.arange(n, dtype=np.float64) / sample_rate_hz

    def _windows(freq_hz: float) -> dict[str, np.ndarray]:
        windows = {}
        for sid, pos in sensor_positions.items():
            delay_s = float(np.linalg.norm(source_m - pos)) / sound_speed_mps
            sig = np.sin(2.0 * np.pi * freq_hz * (t - delay_s)).astype(np.float32)
            windows[sid] = sig
        return windows

    def _lateral(freq_hz: float) -> float:
        result = dispatcher.localize(
            sensor_positions, _windows(freq_hz), sample_rate_hz, 18.0, 0.4
        )
        assert result.position_covariance_m2 is not None
        centroid = np.mean(np.vstack(list(sensor_positions.values())), axis=0)
        bearing = np.asarray(result.position_m) - centroid
        return _lateral_variance_perp_to_bearing(result.position_covariance_m2, bearing)

    low = _lateral(400.0)
    high = _lateral(6_000.0)
    assert low > high * 1.2, f"expected lateral inflation on dispatch path, low={low} high={high}"


def _make_fusion_node(settings: Settings, storage: Storage) -> FusionNode:
    return FusionNode(
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


@pytest.mark.asyncio
async def test_phase2_sanity_gate_measures_range_from_centroid_not_origin(tmp_path: Path) -> None:
    """Phase 2: a node surveyed far from the site origin must still get its full 1 km
    envelope. The sanity gate measures range from the contributing-sensor centroid,
    not the origin, with a 5 km absolute-origin backstop for runaway coordinates."""
    settings = Settings(
        db_path=tmp_path / "phase2_gate.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
    )
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)
    storage = Storage(settings.db_path)
    await storage.initialize()
    fusion = _make_fusion_node(settings, storage)

    # Node array centroid surveyed 800 m from origin along +x.
    centroid = np.array([800.0, 0.0, 0.0])
    ref = "far-node:ch0"
    windows = {ref: np.zeros(256, dtype=np.float32)}

    def _loc(position):
        return LocalizationResult(
            position_m=(float(position[0]), float(position[1]), float(position[2])),
            confidence=0.7,
            gdop=2.0,
            reference_sensor=ref,
            tdoa_s={ref: 0.0},
            position_covariance_m2=[[100.0, 0, 0], [0, 100.0, 0], [0, 0, 100.0]],
            range_observability=0.5,
            residual_rms_seconds=1e-4,
            range_projection_mode="range_refined",
            attempted_algorithm="gcc_phat",
            resolved_algorithm="gcc_phat",
        )

    # Source 900 m from the array (1700 m from origin): allowed now, dropped under the
    # legacy origin-relative 500 m gate.
    branch = fusion._build_localization_branch(
        localization=_loc(centroid + np.array([900.0, 0.0, 0.0])),
        selected_windows=windows,
        classification_windows=windows,
        capability_tier="full_3d",
        contributing_centroid_m=centroid,
    )
    assert branch is not None
    assert fusion._metrics.localization_rejected_out_of_range_count == 0

    # A runaway solve 5200 m from the array is still rejected (> max_range_m).
    branch = fusion._build_localization_branch(
        localization=_loc(centroid + np.array([5200.0, 0.0, 0.0])),
        selected_windows=windows,
        classification_windows=windows,
        capability_tier="full_3d",
        contributing_centroid_m=centroid,
    )
    assert branch is None
    assert fusion._metrics.localization_rejected_out_of_range_count == 1

    await storage.close()


def _cone_branch(position, centroid, *, lateral_std_m=4.0, radial_std_m=500.0, confidence=0.8):
    from minimappr.core.fusion_node import LocalizationBranch

    pos = np.asarray(position, dtype=np.float64)
    cen = np.asarray(centroid, dtype=np.float64)
    r = (pos - cen) / np.linalg.norm(pos - cen)
    cov = (lateral_std_m**2 * (np.eye(3) - np.outer(r, r)) + radial_std_m**2 * np.outer(r, r)).tolist()
    return LocalizationBranch(
        localization_position_m=(float(pos[0]), float(pos[1]), float(pos[2])),
        localization_confidence=confidence,
        localization_gdop=2.0,
        localization_position_covariance_m2=cov,
        localization_range_observability=0.05,
        localization_residual_rms_seconds=1e-4,
        localization_range_projection_mode="range_bearing_projected",
        reference_sensor="n:ch0",
        reference_signal=np.zeros(256, dtype=np.float32),
        classification_reference_signal=np.zeros(256, dtype=np.float32),
        tdoa_s={},
        localization_method="python_cartesian_rust_tdoa",
        capability_tier="full_3d",
    )


@pytest.mark.asyncio
async def test_phase4_bearing_fusion_hook_upgrades_second_node_cone(tmp_path: Path) -> None:
    """Phase 4: with the flag on, a second node's corroborating cone upgrades the
    branch in place to a range-refined multi-node triangulation."""
    settings = Settings(
        db_path=tmp_path / "phase4.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        multi_node_bearing_fusion_enabled=True,
    )
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)
    storage = Storage(settings.db_path)
    await storage.initialize()
    fusion = _make_fusion_node(settings, storage)

    source = np.array([15.0, 200.0, 0.0])
    ca = np.array([0.0, 0.0, 0.0])
    cb = np.array([30.0, 0.0, 0.0])
    event_ns = 1_739_950_000_000_000_000

    # Node A cone arrives first: no corroborator yet, branch is unchanged.
    branch_a = await fusion._maybe_fuse_multi_node_bearing(
        _cone_branch(source, ca),
        node_id="node-a",
        contributing_centroid_m=ca,
        event_time_ns=event_ns,
        sound_speed_mps=343.0,
    )
    assert branch_a.localization_range_projection_mode == "range_bearing_projected"
    assert fusion._metrics.localization_bearing_fusion_fused_count == 0

    # Node B cone arrives within the window: triangulation upgrades it.
    branch_b = await fusion._maybe_fuse_multi_node_bearing(
        _cone_branch(source, cb),
        node_id="node-b",
        contributing_centroid_m=cb,
        event_time_ns=event_ns + 100_000_000,
        sound_speed_mps=343.0,
    )
    assert branch_b.localization_range_projection_mode == "range_refined"
    assert branch_b.localization_method == "multi_node_bearing_triangulation"
    assert fusion._metrics.localization_bearing_fusion_fused_count == 1
    assert fusion._metrics.last_bearing_fusion_contributor_count == 2
    fused = np.asarray(branch_b.localization_position_m)
    assert float(np.linalg.norm(fused - source)) / 200.0 < 0.15

    await storage.close()


@pytest.mark.asyncio
async def test_phase4_bearing_fusion_disabled_by_default(tmp_path: Path) -> None:
    """Regression: with the flag off (default), the hook is a no-op."""
    settings = Settings(
        db_path=tmp_path / "phase4_off.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
    )
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)
    storage = Storage(settings.db_path)
    await storage.initialize()
    fusion = _make_fusion_node(settings, storage)
    source = np.array([15.0, 200.0, 0.0])
    branch = _cone_branch(source, np.array([0.0, 0.0, 0.0]))
    out = await fusion._maybe_fuse_multi_node_bearing(
        branch,
        node_id="node-a",
        contributing_centroid_m=np.array([0.0, 0.0, 0.0]),
        event_time_ns=1_000,
        sound_speed_mps=343.0,
    )
    assert out is branch
    assert fusion._metrics.localization_bearing_fusion_attempt_count == 0
    await storage.close()
