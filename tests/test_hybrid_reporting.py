from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from minimappr.classifiers.base import AudioClassifier
from minimappr.config import Settings
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.core.fusion_node import FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization import LocalizationError
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import ClassificationResult, GeoPoint, IngestFrameRequest, LocalizationResult, NodeSpec, NodeType
from minimappr.storage.db import Storage
from minimappr.utils.audio import encode_pcm16le_b64


def _dominant_frequency_hz(samples: np.ndarray, sample_rate_hz: int) -> float:
    spectrum = np.abs(np.fft.rfft(samples.astype(np.float64)))
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate_hz)
    return float(freqs[int(np.argmax(spectrum))])


class RecordingLocalizer:
    def __init__(self, *, reference_sensor: str | None = None, fail: bool = False) -> None:
        self.reference_sensor = reference_sensor
        self.fail = fail
        self.recorded_frequency_hz: float | None = None

    def localize(
        self,
        sensor_positions: dict[str, np.ndarray],
        sensor_windows: dict[str, np.ndarray],
        sample_rate_hz: int,
        temperature_c: float,
        humidity_fraction: float,
    ) -> LocalizationResult:
        del temperature_c, humidity_fraction
        if self.fail:
            raise LocalizationError("planned failure")
        first_window = next(iter(sensor_windows.values()))
        self.recorded_frequency_hz = _dominant_frequency_hz(first_window, sample_rate_hz)
        reference_sensor = self.reference_sensor or sorted(sensor_positions.keys())[0]
        return LocalizationResult(
            position_m=(1.0, 2.0, 1.5),
            confidence=0.9,
            gdop=1.0,
            reference_sensor=reference_sensor,
            tdoa_s={},
        )


class RecordingClassifier(AudioClassifier):
    def __init__(self, *, label_for_positive: str = "robin", label_for_negative: str = "robin") -> None:
        self.label_for_positive = label_for_positive
        self.label_for_negative = label_for_negative
        self.recorded_frequencies_hz: list[float] = []

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        self.recorded_frequencies_hz.append(_dominant_frequency_hz(samples, sample_rate_hz))
        label = self.label_for_positive if float(np.mean(samples)) >= 0.0 else self.label_for_negative
        return ClassificationResult(
            label=label,
            confidence=0.92,
            scores={label: 0.92},
            features={"mean": float(np.mean(samples))},
        )


async def _start_fusion(
    tmp_path: Path,
    *,
    classifier: AudioClassifier,
    localizer,
    reporting_window_seconds: float = 30.0,
    classification_window_seconds: float = 0.04,
) -> tuple[FusionNode, Storage]:
    settings = Settings(
        db_path=tmp_path / "hybrid.db",
        snippet_dir=tmp_path / "snippets",
        snippet_retention_seconds=0,
        trigger_rms=0.001,
        trigger_cooldown_seconds=0.0,
        localization_window_seconds=0.04,
        classification_window_seconds=classification_window_seconds,
        max_sensor_buffer_seconds=max(2.0, classification_window_seconds + 0.5),
        fusion_worker_count=1,
        preprocess_enabled=False,
        # Triggered classification is beamformed-only; an "omni" source would
        # leave the orchestrator without a beamformer and classify nothing.
        classification_audio_source="beamformed",
        localization_band_min_hz=300.0,
        localization_band_max_hz=1500.0,
        reporting_window_seconds=reporting_window_seconds,
        site_origin_lat=37.0,
        site_origin_lon=-122.0,
        site_origin_alt_m=0.0,
    )
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.snippet_dir.mkdir(parents=True, exist_ok=True)

    storage = Storage(settings.db_path)
    await storage.initialize()
    fusion = FusionNode(
        settings=settings,
        registry=NodeRegistry(),
        buffer=MultiSensorBuffer(max_duration_seconds=settings.max_sensor_buffer_seconds),
        localizer=localizer,
        classifier=classifier,
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=lambda payload: asyncio.sleep(0, result=None),
        coordinate_frame=LocalCoordinateFrame(origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"),
        zone_matcher=ZoneMatcher(storage=storage),
    )
    await fusion.start()
    return fusion, storage


def _node_spec() -> NodeSpec:
    return NodeSpec(
        id="hybrid-node",
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


async def _ingest(fusion: FusionNode, *, start_time_ns: int, channels: np.ndarray) -> None:
    response = await fusion.ingest(
        IngestFrameRequest(
            node=_node_spec(),
            frame={
                "start_time_ns": start_time_ns,
                "sample_rate_hz": 16000,
                "channels": int(channels.shape[0]),
                "encoding": "pcm16le",
                "samples_b64": encode_pcm16le_b64(channels),
                "sequence": int(start_time_ns % 1000),
            },
        )
    )
    assert response.accepted is True
    assert response.triggered is True


@pytest.mark.asyncio
async def test_localization_bandpass_applies_only_to_localization_branch(tmp_path: Path) -> None:
    localizer = RecordingLocalizer(reference_sensor="hybrid-node:ch0")
    classifier = RecordingClassifier()
    fusion, storage = await _start_fusion(tmp_path, classifier=classifier, localizer=localizer)

    sample_rate_hz = 16000
    n = 1024
    t = np.arange(n, dtype=np.float32) / sample_rate_hz
    signal = (0.1 * np.sin(2.0 * np.pi * 700.0 * t) + 0.4 * np.sin(2.0 * np.pi * 6000.0 * t)).astype(np.float32)
    channels = np.stack([signal, signal, signal, signal]).astype(np.float32)
    await _ingest(fusion, start_time_ns=1_739_810_600_000_000_000, channels=channels)

    await asyncio.sleep(0.25)
    assert localizer.recorded_frequency_hz is not None
    assert 500.0 <= localizer.recorded_frequency_hz <= 900.0
    assert classifier.recorded_frequencies_hz
    assert all(freq >= 5000.0 for freq in classifier.recorded_frequencies_hz)

    await fusion.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_reporting_window_prefers_localized_detection_over_same_label_omni(tmp_path: Path) -> None:
    localizer = RecordingLocalizer(reference_sensor="hybrid-node:ch1")
    classifier = RecordingClassifier(label_for_positive="robin", label_for_negative="robin")
    fusion, storage = await _start_fusion(tmp_path, classifier=classifier, localizer=localizer)

    base = np.full(1024, 0.25, dtype=np.float32)
    channels = np.stack([base, base * -0.5, base * 0.2, base * 0.1]).astype(np.float32)
    await _ingest(fusion, start_time_ns=1_739_810_610_000_000_000, channels=channels)

    await asyncio.sleep(0.25)
    detections = await storage.list_detections(limit=10)
    assert len(detections) == 1
    detection = detections[0]
    assert detection["reporting_modality"] == "localized"
    # The omni branch no longer exists in triggered work: the evidence map
    # carries only the localized (beamformed) branch.
    evidence = detection["feature_summary"]["branch_evidence"]
    assert set(evidence.keys()) == {"localized"}
    assert evidence["localized"]["suppressed"] is False

    await fusion.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_reporting_window_emits_no_omni_branch_detection(tmp_path: Path) -> None:
    """The old hybrid contract emitted a second raw-omni detection per event;
    post-dedupe only the beamformed localized detection remains."""
    localizer = RecordingLocalizer(reference_sensor="hybrid-node:ch1")
    classifier = RecordingClassifier(label_for_positive="robin", label_for_negative="wren")
    fusion, storage = await _start_fusion(tmp_path, classifier=classifier, localizer=localizer)

    positive = np.full(1024, 0.3, dtype=np.float32)
    negative = np.full(1024, -0.3, dtype=np.float32)
    channels = np.stack([positive, negative, positive * 0.1, positive * 0.05]).astype(np.float32)
    await _ingest(fusion, start_time_ns=1_739_810_620_000_000_000, channels=channels)

    await asyncio.sleep(0.25)
    detections = await storage.list_detections(limit=10)
    assert len(detections) == 1
    assert detections[0]["reporting_modality"] == "localized"
    assert classifier.recorded_frequencies_hz  # exactly one beamformed inference
    assert len(classifier.recorded_frequencies_hz) == 1

    await fusion.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_localization_miss_yields_no_detection(tmp_path: Path) -> None:
    """A failed solve used to fall back to a raw-omni detection; post-dedupe the
    candidate is dropped and continuous omni scanning owns ambient coverage."""
    localizer = RecordingLocalizer(fail=True)
    classifier = RecordingClassifier(label_for_positive="warbler", label_for_negative="warbler")
    fusion, storage = await _start_fusion(tmp_path, classifier=classifier, localizer=localizer)

    channels = np.full((4, 1024), 0.2, dtype=np.float32)
    await _ingest(fusion, start_time_ns=1_739_810_630_000_000_000, channels=channels)

    await asyncio.sleep(0.25)
    assert await storage.list_detections(limit=10) == []
    status = await fusion.status()
    assert status["metrics"]["localization_failures"] == 1
    assert status["metrics"]["classification_drops_by_reason"].get("empty_classification") == 1

    await fusion.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_localized_coyote_detection_triggers_alert(tmp_path: Path) -> None:
    localizer = RecordingLocalizer(reference_sensor="hybrid-node:ch0")
    classifier = RecordingClassifier(label_for_positive="coyote", label_for_negative="coyote")
    fusion, storage = await _start_fusion(tmp_path, classifier=classifier, localizer=localizer)

    channels = np.full((4, 1024), 0.25, dtype=np.float32)
    await _ingest(fusion, start_time_ns=1_739_810_640_000_000_000, channels=channels)

    await asyncio.sleep(0.25)
    alerts = await storage.list_alerts(limit=20)
    assert len(alerts) == 2

    await fusion.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_unlocalized_coyote_audio_does_not_alert(tmp_path: Path) -> None:
    """Without a solve there is no detection at all now (no omni fallback), so
    nothing can alert."""
    localizer = RecordingLocalizer(fail=True)
    classifier = RecordingClassifier(label_for_positive="coyote", label_for_negative="coyote")
    fusion, storage = await _start_fusion(tmp_path, classifier=classifier, localizer=localizer)

    channels = np.full((4, 1024), 0.25, dtype=np.float32)
    await _ingest(fusion, start_time_ns=1_739_810_650_000_000_000, channels=channels)

    await asyncio.sleep(0.25)
    assert await storage.list_detections(limit=10) == []
    alerts = await storage.list_alerts(limit=20)
    assert alerts == []

    await fusion.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_coyote_alert_cooldown_avoids_repeat_flood(tmp_path: Path) -> None:
    localizer = RecordingLocalizer(reference_sensor="hybrid-node:ch0")
    classifier = RecordingClassifier(label_for_positive="coyote", label_for_negative="coyote")
    fusion, storage = await _start_fusion(
        tmp_path,
        classifier=classifier,
        localizer=localizer,
        reporting_window_seconds=0.05,
    )

    channels = np.full((4, 1024), 0.25, dtype=np.float32)
    await _ingest(fusion, start_time_ns=1_739_810_660_000_000_000, channels=channels)
    await _ingest(fusion, start_time_ns=1_739_810_660_100_000_000, channels=channels)

    await asyncio.sleep(0.35)
    detections = await storage.list_detections(limit=20)
    assert len(detections) >= 2
    alerts = await storage.list_alerts(limit=20)
    assert len(alerts) == 2

    await fusion.stop()
    await storage.close()
