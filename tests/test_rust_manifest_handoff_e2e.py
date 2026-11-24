from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.hybrid_settings import hybrid_production_settings
from minimappr.api.rust_dsp_manifests import LocalizedClassifierRenderRequest
from minimappr.classifiers.base import AudioClassifier, ClassificationResult
from minimappr.main import app
from minimappr.core.audio_buffer import AudioCoverageStats, MultiSensorBuffer
from minimappr.core.fusion_node import FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization import LocalizationEngine
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import GeoPoint, NodeSpec, NodeType
from minimappr.sim.binary_ingest_encoder import (
    BinaryIngestFrame,
    BinaryIngestNodeHeader,
    encode_binary_ingest_payload,
)
from minimappr.storage.db import Storage
from tests.helpers import SIRITH_TETRA_SENSOR_OFFSETS_M, prepend_noise_padding, synthesize_delayed_array_channels
from tests.rust_sidecar_e2e_helpers import (
    FAKE_CLASSIFIER_CONFIDENCE,
    FAKE_CLASSIFIER_LABEL,
    configure_full_app_with_rust_sidecar,
    fake_sidecar_classifier_command_json,
    http_json,
    pick_free_tcp_port,
    post_binary_ingest,
    post_store_forward_ingest,
    running_rust_sidecar,
    store_forward_payload,
    wait_for_detection_from_node,
    wait_for_live_dsp_results,
    write_fake_sidecar_classifier_helper,
)


TEST_SOURCE_POSITION_M = (0.35, 0.18, 0.07)
TEST_NODE_GEO = GeoPoint(lat=44.987, lon=-93.258, alt_m=281.5)


class _ConstantBirdClassifier(AudioClassifier):
    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        del samples, sample_rate_hz
        return ClassificationResult(
            label="sparrow",
            confidence=0.93,
            scores={"sparrow": 0.93},
            features={},
        )


class _UnexpectedClassifier(AudioClassifier):
    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        del samples, sample_rate_hz
        raise AssertionError("Python classifier should not run when Rust classification is authoritative")


def _binary_ingest_payload(
    *,
    node_id: str,
    samples: np.ndarray,
    start_time_ns: int,
    sequence: int,
    sample_rate_hz: int,
) -> bytes:
    """One single-frame tetra upload, as the legacy firmware header describes it."""
    node = BinaryIngestNodeHeader(
        node_id=node_id,
        node_type=NodeType.SIRITH_TETRA,
        sensor_offsets_m=list(SIRITH_TETRA_SENSOR_OFFSETS_M),
        position_geo=TEST_NODE_GEO,
        hardware="sirith-test-hardware",
        firmware="sirith-test-firmware",
        gps_signal="gps_locked",
        position_source="test",
    )
    frame = BinaryIngestFrame(
        channels_first_samples=samples,
        start_time_ns=start_time_ns,
        sample_rate_hz=sample_rate_hz,
        start_sample_index=0,
        sequence=sequence,
        tor_ns=start_time_ns + 1_000_000,
    )
    return encode_binary_ingest_payload(node, [frame])


def _node_spec(node_id: str) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        node_type=NodeType.SIRITH_TETRA,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=list(SIRITH_TETRA_SENSOR_OFFSETS_M),
        capabilities=["audio", "array_localization"],
        metadata={},
    )


def _padded_chirp_and_tone_signal(sample_rate_hz: int, *, noise_seed: int) -> np.ndarray:
    duration_seconds = 0.8
    timestamps_s = np.linspace(0.0, duration_seconds, int(round(duration_seconds * sample_rate_hz)), endpoint=False)
    chirp = np.sin(2.0 * np.pi * (700.0 * timestamps_s + 850.0 * np.square(timestamps_s)))
    tone = 0.35 * np.sin(2.0 * np.pi * 1700.0 * timestamps_s)
    mono_signal = (0.65 * chirp + tone).astype(np.float32)
    mono_signal *= np.hanning(mono_signal.size).astype(np.float32)
    return prepend_noise_padding(
        mono_signal,
        pad_samples=max(1, int(round(0.25 * sample_rate_hz))),
        noise_rms=0.002,
        seed=noise_seed,
    )


def _synthesized_localizable_channels(sample_rate_hz: int) -> np.ndarray:
    return synthesize_delayed_array_channels(
        _padded_chirp_and_tone_signal(sample_rate_hz, noise_seed=sample_rate_hz),
        sample_rate_hz,
        source_position_m=TEST_SOURCE_POSITION_M,
    )


def _synthesized_mixed_cluster_channels(sample_rate_hz: int) -> tuple[np.ndarray, np.ndarray]:
    padded_signal = _padded_chirp_and_tone_signal(sample_rate_hz, noise_seed=sample_rate_hz + 101)
    tetra_channels = synthesize_delayed_array_channels(
        padded_signal,
        sample_rate_hz,
        source_position_m=TEST_SOURCE_POSITION_M,
    )
    point_channel = synthesize_delayed_array_channels(
        padded_signal,
        sample_rate_hz,
        source_position_m=TEST_SOURCE_POSITION_M,
        sensor_offsets_m=((0.8, 0.6, 0.4),),
    )
    return tetra_channels, point_channel


@pytest.mark.asyncio
async def test_fusion_prefers_authoritative_rust_classification_over_python_reclassify(
    tmp_path: Path,
) -> None:
    settings = hybrid_production_settings(
        db_path=tmp_path / "authoritative-rust.db",
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
        classifier=_UnexpectedClassifier(),
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=lambda payload: asyncio.sleep(0, result=None),
        coordinate_frame=LocalCoordinateFrame(origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"),
        zone_matcher=ZoneMatcher(storage=storage),
    )
    await fusion.start()
    try:
        render_audio = np.clip(
            np.random.default_rng(59).normal(0.0, 0.25, size=24_000),
            -1.0,
            0.9999695,
        ).astype(np.float32)
        render_end_ns = time.time_ns()
        render_start_ns = render_end_ns - 1_500_000_000
        await fusion.ingest_localized_render(
            payload=LocalizedClassifierRenderRequest(
                manifest_id="manifest-rust-authoritative-1",
                node=_node_spec("sirith-rust-authoritative"),
                event_time_ns=render_end_ns,
                sample_rate_hz=16_000,
                decoded_audio=render_audio,
                localization_position_m=TEST_SOURCE_POSITION_M,
                localization_confidence=0.88,
                localization_gdop=1.4,
                localization_method="srp_phat",
                source_type="raw_sensor",
                source_observation_ids=[],
                environment={"temperature_c": 20.0, "humidity_fraction": 0.5},
                render_kind="birdnet_hybrid_spatial_blend",
                render_start_ns=render_start_ns,
                render_end_ns=render_end_ns,
                audio_quality=AudioCoverageStats(
                    sample_count=24_000,
                    covered_samples=20_000,
                    missing_samples=4_000,
                    coverage_ratio=20_000 / 24_000,
                    missing_ratio=4_000 / 24_000,
                    max_gap_samples=1_600,
                    max_gap_seconds=0.1,
                    warning=True,
                    degraded=True,
                ),
                authoritative_classification=ClassificationResult(
                    label="winter wren",
                    confidence=0.91,
                    scores={"winter wren": 0.91, "sparrow": 0.04},
                    features={"rust_classifier_source_node": "sirith-rust-authoritative"},
                ),
            )
        )

        detections = await storage.list_detections(limit=10)
        assert len(detections) == 1
        detection = detections[0]
        assert detection["label"] == "winter wren"
        assert detection["confidence"] == pytest.approx(0.88)
        assert detection["label_confidence"] == pytest.approx(0.91)
        assert detection["classifier_scores"]["winter wren"] == pytest.approx(0.91)
        assert detection["feature_summary"]["classification_path"] == "rust_manifest"
        assert detection["feature_summary"]["rust_classification_authoritative"] is True
        assert detection["feature_summary"]["rust_manifest_id"] == "manifest-rust-authoritative-1"
        assert detection["feature_summary"]["rust_render_kind"] == "birdnet_hybrid_spatial_blend"
        assert detection["feature_summary"]["rust_render_start_ns"] == render_start_ns
        assert detection["feature_summary"]["rust_render_end_ns"] == render_end_ns
        audio_quality = detection["feature_summary"]["audio_quality"]
        assert audio_quality["source_window_type"] == "rust_classifier_render"
        assert audio_quality["sample_count"] == 24_000
        assert audio_quality["missing_samples"] == 4_000
        assert audio_quality["missing_ratio"] == pytest.approx(4_000 / 24_000)
        assert audio_quality["degraded"] is True
    finally:
        await fusion.stop()
        await storage.close()


@pytest.mark.asyncio
async def test_fusion_classifies_non_authoritative_rust_render_immediately_in_production(
    tmp_path: Path,
) -> None:
    settings = hybrid_production_settings(
        db_path=tmp_path / "non-authoritative-rust.db",
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
        classifier=_ConstantBirdClassifier(),
        tracker=TrackManager(settings),
        storage=storage,
        live_callback=lambda payload: asyncio.sleep(0, result=None),
        coordinate_frame=LocalCoordinateFrame(origin=GeoPoint(lat=37.0, lon=-122.0, alt_m=0.0), mode="flat"),
        zone_matcher=ZoneMatcher(storage=storage),
    )
    await fusion.start()
    try:
        render_audio = np.clip(
            np.random.default_rng(61).normal(0.0, 0.25, size=24_000),
            -1.0,
            0.9999695,
        ).astype(np.float32)
        await fusion.ingest_localized_render(
            payload=LocalizedClassifierRenderRequest(
                manifest_id="manifest-rust-python-classify-1",
                node=_node_spec("sirith-rust-python-classify"),
                event_time_ns=time.time_ns(),
                sample_rate_hz=16_000,
                decoded_audio=render_audio,
                localization_position_m=TEST_SOURCE_POSITION_M,
                localization_confidence=0.77,
                localization_gdop=1.8,
                localization_method="srp_phat",
                source_type="raw_sensor",
                source_observation_ids=[],
                environment={"temperature_c": 20.0, "humidity_fraction": 0.5},
                render_kind="birdnet_omni_fallback",
            )
        )

        detections = await storage.list_detections(limit=10)
        assert len(detections) == 1
        detection = detections[0]
        assert detection["label"] == "sparrow"
        assert detection["reporting_modality"] == "omni"
        assert tuple(detection["position_m"]) == pytest.approx(
            _node_spec("sirith-rust-python-classify").position_m
        )
        assert detection["feature_summary"]["rust_manifest_id"] == "manifest-rust-python-classify-1"
        assert detection["feature_summary"]["classification_path"] == "omni"
        status = await fusion.status()
        assert status["queue"]["classification_depth"] == 0
        assert status["metrics"]["birdnet_chunk_dispatches_suppressed"] == 0
    finally:
        await fusion.stop()
        await storage.close()


@pytest.mark.asyncio
async def test_live_rust_sidecar_emits_authoritative_classification_fields(monkeypatch, tmp_path: Path) -> None:
    sample_rate_hz = 16_000
    node_id = "sirith-rust-authoritative-live"
    helper_path = write_fake_sidecar_classifier_helper(tmp_path)
    sidecar_port = pick_free_tcp_port()
    configure_full_app_with_rust_sidecar(
        monkeypatch,
        tmp_path,
        sidecar_port=sidecar_port,
        extra_env={
            "MINIMAPPR_SIDECAR_CLASSIFIER_COMMAND_JSON": fake_sidecar_classifier_command_json(
                helper_path, sys.executable
            )
        },
    )
    # Classification window / buffer inherit the hybrid defaults (30s/32s) from
    # configure_full_app_with_rust_sidecar: with birdnet + chunked dispatch enabled,
    # the chunk overlap (2.0s) must stay below the classification window.
    payload = _binary_ingest_payload(
        node_id=node_id,
        samples=_synthesized_localizable_channels(sample_rate_hz),
        start_time_ns=time.time_ns(),
        sequence=7,
        sample_rate_hz=sample_rate_hz,
    )

    with TestClient(app) as client:
        post_binary_ingest(sidecar_port, payload)

        detection = wait_for_detection_from_node(client, node_id=node_id)
        assert detection["label"] == FAKE_CLASSIFIER_LABEL
        assert detection["label_confidence"] == pytest.approx(FAKE_CLASSIFIER_CONFIDENCE)
        assert detection["feature_summary"]["classification_path"] == "rust_manifest"
        assert detection["feature_summary"]["rust_classification_authoritative"] is True
        assert detection["feature_summary"]["rust_manifest_id"]


@pytest.mark.asyncio
async def test_rust_memory_only_sidecar_does_not_persist_raw_replay(tmp_path: Path) -> None:
    sample_rate_hz = 16_000
    node_id = "sirith-rust-handoff"
    start_time_ns = 1_739_970_000_000_000_000
    payload = _binary_ingest_payload(
        node_id=node_id,
        samples=_synthesized_localizable_channels(sample_rate_hz),
        start_time_ns=start_time_ns,
        sequence=1,
        sample_rate_hz=sample_rate_hz,
    )

    with running_rust_sidecar(
        tmp_path,
        extra_env={"MINIMAPPR_SIDECAR_MEMORY_ONLY_LIVE_PATH": "true"},
    ) as (port, spool_dir):
        post_binary_ingest(port, payload)

        health = http_json(port, "/healthz")
        assert health["backend"]["storage_mode"] == "memory_only_live_path"
        time.sleep(0.2)
        journal_dir = spool_dir / "journal"
        streams_dir = journal_dir / "streams"
        assert not list(streams_dir.rglob("*.index.jsonl"))
        assert not list(streams_dir.rglob("*.bin"))
        assert not list((journal_dir / "manifests").rglob("*.json"))


def test_http_app_in_production_mode_consumes_memory_only_rust_sidecar(monkeypatch, tmp_path: Path) -> None:
    sidecar_port = pick_free_tcp_port()
    spool_dir = configure_full_app_with_rust_sidecar(monkeypatch, tmp_path, sidecar_port=sidecar_port)
    monkeypatch.setattr(
        "minimappr.main._build_runtime_classifier", lambda settings: _ConstantBirdClassifier()
    )
    sample_rate_hz = 16_000
    node_id = "sirith-rust-http-app"
    start_time_ns = time.time_ns()
    payload = _binary_ingest_payload(
        node_id=node_id,
        samples=_synthesized_localizable_channels(sample_rate_hz),
        start_time_ns=start_time_ns,
        sequence=2,
        sample_rate_hz=sample_rate_hz,
    )

    with TestClient(app) as client:
        diagnostics_response = client.get("/api/v1/system/diagnostics")
        assert diagnostics_response.status_code == 200
        diagnostics = diagnostics_response.json()
        assert diagnostics["sidecar"]["enabled"] is True
        assert diagnostics["sidecar"]["status"] == "running"
        assert diagnostics["sidecar"]["health"]["status"] == "ok"
        assert diagnostics["sidecar"]["health"]["backend"]["storage_mode"] == "memory_only_live_path"

        blocked_response = client.post(
            "/api/v1/ingest/binary",
            content=payload,
            headers={"content-type": "application/octet-stream"},
        )
        assert blocked_response.status_code == 410
        assert "sidecar" in blocked_response.json()["detail"]

        post_binary_ingest(sidecar_port, payload)

        detection = wait_for_detection_from_node(client, node_id=node_id)
        assert detection["label"] == "sparrow"
        assert detection["source_node_id"] == node_id
        assert detection["reporting_modality"] == "localized"
        # Default solver re-homes the position solve onto Python's Cartesian
        # TDOA solver using the sidecar's pairwise TDOAs + bearing.
        assert detection["feature_summary"]["localization_method"] == "python_cartesian_rust_tdoa"
        assert detection["feature_summary"]["rust_manifest_id"]
        assert str(detection["feature_summary"]["rust_render_kind"]).startswith("birdnet_")

        journal_dir = spool_dir / "journal"
        streams_dir = journal_dir / "streams"
        assert not list(streams_dir.rglob("*.index.jsonl"))
        assert not list(streams_dir.rglob("*.bin"))


def test_live_rust_sidecar_localizes_mixed_cluster_store_forward_payloads(tmp_path: Path) -> None:
    sample_rate_hz = 16_000
    start_time_ns = time.time_ns()
    tetra_channels, point_channel = _synthesized_mixed_cluster_channels(sample_rate_hz)
    cluster_id = "cluster-mixed-live"
    cluster_sensor_positions = {
        "tetra-node:ch0": [float(x) for x in SIRITH_TETRA_SENSOR_OFFSETS_M[0]],
        "tetra-node:ch1": [float(x) for x in SIRITH_TETRA_SENSOR_OFFSETS_M[1]],
        "tetra-node:ch2": [float(x) for x in SIRITH_TETRA_SENSOR_OFFSETS_M[2]],
        "tetra-node:ch3": [float(x) for x in SIRITH_TETRA_SENSOR_OFFSETS_M[3]],
        "point-node:ch0": [0.8, 0.6, 0.4],
    }

    tetra_payload = store_forward_payload(
        node_id="tetra-node",
        node_type="sirith_tetra",
        sensor_offsets_m=[[float(axis) for axis in offset] for offset in SIRITH_TETRA_SENSOR_OFFSETS_M],
        channels=tetra_channels,
        start_time_ns=start_time_ns,
        sequence=1,
        cluster_id=cluster_id,
        cluster_sensor_positions=cluster_sensor_positions,
        sample_rate_hz=sample_rate_hz,
    )
    point_payload = store_forward_payload(
        node_id="point-node",
        node_type="point",
        sensor_offsets_m=[[0.0, 0.0, 0.0]],
        channels=point_channel,
        start_time_ns=start_time_ns,
        sequence=2,
        cluster_id=cluster_id,
        cluster_sensor_positions=cluster_sensor_positions,
        sample_rate_hz=sample_rate_hz,
    )

    with running_rust_sidecar(
        tmp_path,
        extra_env={
            "MINIMAPPR_CLASSIFICATION_WINDOW_SECONDS": "15.0",
            "MINIMAPPR_CLASSIFIER_RENDER_MIN_INTERVAL_SECONDS": "0.0",
            "MINIMAPPR_DSP_LOCALIZATION_CADENCE_MS": "0",
        },
    ) as (port, _spool_dir):
        post_store_forward_ingest(port, tetra_payload)
        post_store_forward_ingest(port, point_payload)

        manifests = wait_for_live_dsp_results(
            port,
            description="mixed-cluster localization and classifier render",
            predicate=lambda payloads: any(
                payload.get("manifest_type") == "localization_result"
                and payload.get("cluster_id") == cluster_id
                and isinstance(payload.get("localization"), dict)
                and len(payload["localization"].get("pair_tdoas") or []) == 10
                and str(payload["localization"].get("resolved_algorithm")) == "srp_phat"
                for payload in payloads
            ) and any(
                payload.get("manifest_type") == "classifier_render"
                and payload.get("cluster_id") == cluster_id
                and isinstance(payload.get("classifier_render"), dict)
                and int(payload["classifier_render"].get("source_channel_count") or 0) == 5
                for payload in payloads
            ),
        )

        localization_manifest = next(
            payload
            for payload in manifests
            if payload.get("manifest_type") == "localization_result"
            and payload.get("cluster_id") == cluster_id
            and isinstance(payload.get("localization"), dict)
            and len(payload["localization"].get("pair_tdoas") or []) == 10
        )
        classifier_manifest = next(
            payload
            for payload in manifests
            if payload.get("manifest_type") == "classifier_render"
            and payload.get("cluster_id") == cluster_id
            and isinstance(payload.get("classifier_render"), dict)
            and int(payload["classifier_render"].get("source_channel_count") or 0) == 5
        )

        assert localization_manifest["cluster_sensor_positions"] is not None
        assert len(localization_manifest["cluster_sensor_positions"]) == 5
        assert localization_manifest["localization"]["resolved_algorithm"] == "srp_phat"
        assert len(localization_manifest["localization"]["pair_tdoas"]) == 10
        assert classifier_manifest["classifier_render"]["source_channel_count"] == 5
