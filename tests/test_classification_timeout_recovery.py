from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
import pytest

from minimappr.classifiers.base import AudioClassifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.classification import ClassificationOrchestrator
from minimappr.core.fusion_node import FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization import LocalizationEngine
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import ClassificationResult, GeoPoint, IngestFrameRequest, NodeSpec, NodeType
from minimappr.storage.db import Storage
from minimappr.utils.audio import encode_pcm16le_b64


class _HangingClassifier(AudioClassifier):
    def __init__(self) -> None:
        self.cancel_pending_calls = 0

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        del samples, sample_rate_hz
        time.sleep(0.2)
        return ClassificationResult(label="bird", confidence=1.0, scores={"bird": 1.0}, features={})

    def cancel_pending(self) -> None:
        self.cancel_pending_calls += 1


class _CancellingThenRecoveringClassifier(AudioClassifier):
    def __init__(self) -> None:
        self.calls = 0

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        del samples, sample_rate_hz
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("Analysis was cancelled")
        return ClassificationResult(label="warbler", confidence=0.72, scores={"warbler": 0.72}, features={})


class _AlwaysFailingClassifier(AudioClassifier):
    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        del samples, sample_rate_hz
        raise RuntimeError("birdnet worker crashed")


class _FirstCallHangsThenRecoversClassifier(AudioClassifier):
    def __init__(self) -> None:
        self.calls = 0

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        del samples, sample_rate_hz
        self.calls += 1
        if self.calls == 1:
            time.sleep(0.2)
        return ClassificationResult(label="warbler", confidence=0.74, scores={"warbler": 0.74}, features={})


class _StorageStub:
    async def upsert_label(self, *, name: str, category: str, source: str, created_ns: int) -> str:
        del name, category, source, created_ns
        return "label-timeout"


class _TaxonomyStub:
    def category_for_label(self, label: str) -> str:
        del label
        return "unknown"

    def iff_for_category(self, category: str) -> str:
        del category
        return "unknown"


class _EnvironmentStub:
    def get_speed_of_sound(self, position_m: tuple[float, float, float]) -> float:
        del position_m
        return 343.0


@pytest.mark.asyncio
async def test_timeout_invokes_classifier_cancellation() -> None:
    classifier = _HangingClassifier()
    orchestrator = ClassificationOrchestrator(
        classifier=classifier,
        storage=_StorageStub(),
        taxonomy_provider=_TaxonomyStub(),
        environment_provider=_EnvironmentStub(),
        stage_timeout_seconds=0.01,
    )

    result = await orchestrator.classify_omni_only(
        reference_signal=np.zeros(1024, dtype=np.float32),
        sample_rate_hz=16_000,
        event_time_ns=123,
    )

    assert result.classification.label == "timeout"
    assert result.classification.features.get("reason") == "classification_timeout"
    assert classifier.cancel_pending_calls == 1


@pytest.mark.asyncio
async def test_cancelled_analysis_retries_once_and_recovers() -> None:
    classifier = _CancellingThenRecoveringClassifier()
    orchestrator = ClassificationOrchestrator(
        classifier=classifier,
        storage=_StorageStub(),
        taxonomy_provider=_TaxonomyStub(),
        environment_provider=_EnvironmentStub(),
    )

    result = await orchestrator.classify_omni_only(
        reference_signal=np.zeros(1024, dtype=np.float32),
        sample_rate_hz=16_000,
        event_time_ns=123,
    )

    assert result.classification.label == "warbler"
    assert classifier.calls == 2


@pytest.mark.asyncio
async def test_classifier_exception_degrades_to_unknown() -> None:
    orchestrator = ClassificationOrchestrator(
        classifier=_AlwaysFailingClassifier(),
        storage=_StorageStub(),
        taxonomy_provider=_TaxonomyStub(),
        environment_provider=_EnvironmentStub(),
    )

    result = await orchestrator.classify_omni_only(
        reference_signal=np.zeros(1024, dtype=np.float32),
        sample_rate_hz=16_000,
        event_time_ns=123,
    )

    assert result.classification.label == "unknown"
    assert result.classification.features.get("reason") == "classification_error"


@pytest.mark.asyncio
async def test_fusion_stage_timeout_drops_stalled_candidate_and_recovers(tmp_path: Path) -> None:
    settings = Settings(
        db_path=tmp_path / "fusion_timeout_recovery.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        trigger_rms=0.001,
        trigger_cooldown_seconds=0.0,
        localization_window_seconds=0.04,
        classification_window_seconds=0.0,
        max_sensor_buffer_seconds=2.0,
        fusion_worker_count=1,
        classifier_stage_timeout_seconds=0.01,
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
        classifier=_FirstCallHangsThenRecoversClassifier(),
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=lambda payload: asyncio.sleep(0, result=None),
        coordinate_frame=LocalCoordinateFrame(origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"),
        zone_matcher=ZoneMatcher(storage=storage),
    )
    await fusion.start()

    node = NodeSpec(
        id="timeout-recovery-node",
        node_type=NodeType.POINT,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
        metadata={},
    )
    samples = np.random.default_rng(55).normal(0.0, 0.2, size=(1, 1024)).astype(np.float32)
    start_time_ns = 1_739_810_600_000_000_000
    frame_duration_ns = int(round((1024 / 16_000) * 1_000_000_000))

    for offset in range(2):
        response = await fusion.ingest(
            IngestFrameRequest(
                node=node,
                frame={
                    "start_time_ns": start_time_ns + (offset * frame_duration_ns),
                    "sample_rate_hz": 16_000,
                    "channels": 1,
                    "encoding": "pcm16le",
                    "samples_b64": encode_pcm16le_b64(samples),
                    "sequence": offset + 1,
                },
            )
        )
        assert response.accepted is True
        assert response.triggered is True

    await asyncio.sleep(0.35)

    status = await fusion.status()
    assert status["metrics"]["stage_timeout_count"] >= 1

    detections = await storage.list_detections(limit=10)
    assert len(detections) >= 1
    assert any(detection["label"] == "warbler" for detection in detections)

    await fusion.stop()
    await storage.close()
