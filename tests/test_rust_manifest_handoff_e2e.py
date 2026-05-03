from __future__ import annotations

import asyncio
import functools
import json
import os
import socket
import sqlite3
import struct
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from minimappr.api.rust_dsp_manifests import LocalizedClassifierRenderRequest
from minimappr.api.spool_consumer import IngestSpoolConfig, IngestSpoolConsumer
from minimappr.api.transports import HttpIngestTransport
from minimappr.classifiers.base import AudioClassifier, ClassificationResult
from minimappr.config import Settings
from minimappr.main import app
from minimappr.core.audio_buffer import AudioCoverageStats, MultiSensorBuffer
from minimappr.core.fusion_node import FusionNode
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization import LocalizationEngine
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.tracking import TrackManager
from minimappr.core.zones import ZoneMatcher
from minimappr.models import GeoPoint, NodeSpec, NodeType
from minimappr.storage.db import Storage
from tests.helpers import SIRITH_TETRA_SENSOR_OFFSETS_M, prepend_noise_padding, synthesize_delayed_array_channels


MINIMAPPR_ROOT = Path(__file__).resolve().parents[1]
SIDECAR_BINARY_PATH = MINIMAPPR_ROOT / "minimappr-ingest-sidecar" / "target" / "debug" / "minimappr-ingest-sidecar"
TEST_SOURCE_POSITION_M = (0.35, 0.18, 0.07)


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


class _RecordingHttpIngestTransport(HttpIngestTransport):
    def __init__(self, fusion_node: FusionNode) -> None:
        super().__init__(fusion_node)
        self.binary_deliveries = 0
        self.localized_render_deliveries = 0

    async def deliver_binary(self, payload) -> object:
        self.binary_deliveries += 1
        return await super().deliver_binary(payload)

    async def deliver_localized_render(self, payload) -> None:
        self.localized_render_deliveries += 1
        await super().deliver_localized_render(payload)


def _binary_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    assert len(encoded) <= 255
    return struct.pack("<B", len(encoded)) + encoded


def _binary_node(*, node_id: str) -> bytes:
    payload = bytearray()
    payload += _binary_string(node_id)
    payload += struct.pack("<B", 0)
    payload += struct.pack("<fff", 0.0, 0.0, 0.0)
    payload += struct.pack("<B", 0)
    payload += struct.pack("<B", len(SIRITH_TETRA_SENSOR_OFFSETS_M))
    for offset_x, offset_y, offset_z in SIRITH_TETRA_SENSOR_OFFSETS_M:
        payload += struct.pack("<fff", offset_x, offset_y, offset_z)
    capabilities = ["audio", "array_localization"]
    payload += struct.pack("<B", len(capabilities))
    for capability in capabilities:
        payload += _binary_string(capability)
    payload += _binary_string("sirith-test-hardware")
    payload += _binary_string("sirith-test-firmware")
    payload += _binary_string("gps_locked")
    payload += _binary_string("test")
    payload += struct.pack("<I", 1)
    return bytes(payload)


def _node_spec(node_id: str) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        node_type=NodeType.SIRITH_TETRA,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=list(SIRITH_TETRA_SENSOR_OFFSETS_M),
        capabilities=["audio", "array_localization"],
        metadata={},
    )


def _binary_frame(
    samples: np.ndarray,
    *,
    start_time_ns: int,
    sequence: int,
    start_sample_index: int,
    sample_rate_hz: int,
) -> bytes:
    channels, samples_per_channel = samples.shape
    end_sample_index = start_sample_index + samples_per_channel
    end_time_ns = start_time_ns + int(round(samples_per_channel / sample_rate_hz * 1_000_000_000))
    pcm = np.clip(samples.T, -1.0, 0.9999695)
    pcm16 = (pcm * 32768.0).astype("<i2").tobytes()
    payload = bytearray()
    payload += struct.pack(
        "<QQQQIBQQQB",
        start_time_ns,
        end_time_ns,
        start_sample_index,
        end_sample_index,
        sample_rate_hz,
        channels,
        sequence,
        start_time_ns,
        start_time_ns + 1_000_000,
        0,
    )
    payload += struct.pack("<B", 0)
    payload += struct.pack("<B", 0)
    payload += struct.pack("<I", samples_per_channel)
    payload += pcm16
    return bytes(payload)


def _binary_ingest_payload(*, node_id: str, frames: list[bytes]) -> bytes:
    payload = bytearray()
    payload += b"MMB1"
    payload += struct.pack("<BBH", 1, 0, len(frames))
    payload += _binary_node(node_id=node_id)
    for frame in frames:
        payload += frame
    return bytes(payload)


def _pick_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_json(
    port: int,
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> dict | list:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        method=method,
        headers=headers or {},
    )
    with urllib.request.urlopen(request, timeout=5.0) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload) if payload else {}


def _sidecar_output(process: subprocess.Popen[str]) -> str:
    if process.stdout is None:
        return ""
    try:
        return process.stdout.read()
    except Exception:
        return ""


@functools.lru_cache(maxsize=1)
def _ensure_sidecar_binary() -> Path:
    sidecar_dir = MINIMAPPR_ROOT / "minimappr-ingest-sidecar"
    subprocess.run(
        ["cargo", "build", "--quiet", "--bin", "minimappr-ingest-sidecar"],
        cwd=str(sidecar_dir),
        check=True,
    )
    return SIDECAR_BINARY_PATH


def _wait_for_sidecar_ready(process: subprocess.Popen[str], *, port: int, timeout_s: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                "Rust sidecar exited before becoming ready.\n"
                f"Captured output:\n{_sidecar_output(process)}"
            )
        try:
            payload = _http_json(port, "/healthz")
            if payload.get("status") == "ok":
                return
        except Exception as exc:  # pragma: no cover - transient startup race
            last_error = exc
        time.sleep(0.05)
    raise AssertionError(
        "Timed out waiting for the Rust sidecar health endpoint.\n"
        f"Last error: {last_error}\n"
        f"Captured output:\n{_sidecar_output(process)}"
    )


@contextmanager
def _running_rust_sidecar(tmp_path: Path, *, extra_env: dict[str, str] | None = None):
    binary_path = _ensure_sidecar_binary()
    if not binary_path.exists():
        pytest.skip(f"Rust sidecar binary not found at {binary_path}")

    port = _pick_free_tcp_port()
    spool_dir = tmp_path / "sidecar_spool"
    env = {
        **os.environ,
        "MINIMAPPR_SIDECAR_PORT": str(port),
        "MINIMAPPR_INGEST_SPOOL_DIR": str(spool_dir),
        "MINIMAPPR_SIDECAR_STORAGE_MODE": "journal",
        "MINIMAPPR_SIDECAR_ALLOW_NON_TMPFS_JOURNAL": "true",
        "MINIMAPPR_RUNTIME_PROFILE": "birdnet_hybrid_production",
        "MINIMAPPR_LOCALIZATION_SRP_GRID_RESOLUTION_M": "0.05",
        "MINIMAPPR_LOCALIZATION_SEARCH_PADDING_M": "0.3",
        "MINIMAPPR_DEFAULT_TEMPERATURE_C": "20.0",
        "MINIMAPPR_DEFAULT_HUMIDITY": "0.5",
        "MINIMAPPR_SIDECAR_TOTAL_JOURNAL_BUDGET_BYTES": str(32 * 1024 * 1024),
        "MINIMAPPR_SIDECAR_ADMISSION_RESERVE_BYTES": str(4 * 1024 * 1024),
        "MINIMAPPR_SIDECAR_DERIVED_CACHE_BUDGET_BYTES": str(16 * 1024 * 1024),
        "MINIMAPPR_SIDECAR_DERIVED_CACHE_ADMISSION_RESERVE_BYTES": str(2 * 1024 * 1024),
    }
    if extra_env is not None:
        env.update(extra_env)
    process = subprocess.Popen(
        [str(binary_path)],
        cwd=str(MINIMAPPR_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_sidecar_ready(process, port=port)
        yield port, spool_dir
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)


def _wait_for_manifest_pair(spool_dir: Path, *, timeout_s: float = 15.0) -> list[dict]:
    deadline = time.monotonic() + timeout_s
    manifest_dir = spool_dir / "journal" / "manifests"
    last_manifest_types: list[str] = []
    while time.monotonic() < deadline:
        payloads: list[dict] = []
        # Manifests live in pending/ or consumed/ subdirectories.
        if manifest_dir.exists():
            for path in sorted(manifest_dir.rglob("*.json")):
                payloads.append(json.loads(path.read_text(encoding="utf-8")))
        last_manifest_types = [str(payload.get("manifest_type")) for payload in payloads]
        manifest_types = set(last_manifest_types)
        if {"localization_result", "classifier_render"}.issubset(manifest_types):
            return payloads
        time.sleep(0.05)
    raise AssertionError(
        "Timed out waiting for Rust DSP manifest pair. "
        f"Last manifest types: {last_manifest_types}"
    )


def _write_fake_sidecar_classifier_helper(tmp_path: Path) -> Path:
    helper_path = tmp_path / "fake_sidecar_classifier_helper.py"
    helper_path.write_text(
        """
from __future__ import annotations

import json
import sys

for line in sys.stdin:
    if not line.strip():
        continue
    request = json.loads(line)
    response = {
        "request_id": request["request_id"],
        "label": "winter wren",
        "label_confidence": 0.91,
        "scores": {
            "winter wren": 0.91,
            "sparrow": 0.04,
        },
    }
    sys.stdout.write(json.dumps(response) + "\\n")
    sys.stdout.flush()
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return helper_path


def _configure_http_app_sidecar_env(monkeypatch, tmp_path: Path, *, sidecar_port: int) -> Path:
    binary_path = _ensure_sidecar_binary()
    db_path = tmp_path / "http_app_sidecar.db"
    snippet_dir = tmp_path / "snippets"
    artifact_dir = tmp_path / "artifacts"
    spool_dir = tmp_path / "spool"
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(db_path))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(snippet_dir))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("MINIMAPPR_INGEST_SPOOL_DIR", str(spool_dir))
    monkeypatch.setenv("MINIMAPPR_INGEST_STORAGE_MODE", "journal")
    monkeypatch.setenv("MINIMAPPR_DIRECT_INGEST_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_ENABLED", "true")
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_BINARY_PATH", str(binary_path))
    monkeypatch.setenv("MINIMAPPR_SIDECAR_PORT", str(sidecar_port))
    monkeypatch.setenv("MINIMAPPR_SIDECAR_ALLOW_NON_TMPFS_JOURNAL", "true")
    monkeypatch.setenv("MINIMAPPR_RUNTIME_PROFILE", "birdnet_hybrid_production")
    monkeypatch.setenv("MINIMAPPR_LOCALIZATION_SRP_GRID_RESOLUTION_M", "0.05")
    monkeypatch.setenv("MINIMAPPR_LOCALIZATION_SEARCH_PADDING_M", "0.3")
    monkeypatch.setenv("MINIMAPPR_DEFAULT_TEMPERATURE_C", "20.0")
    monkeypatch.setenv("MINIMAPPR_DEFAULT_HUMIDITY", "0.5")
    monkeypatch.setenv("MINIMAPPR_INGEST_SPOOL_POLL_INTERVAL_SECONDS", "0.05")
    monkeypatch.setenv("MINIMAPPR_TRIGGER_RMS", "0.000001")
    monkeypatch.setenv("MINIMAPPR_TRIGGER_COOLDOWN_SECONDS", "0")
    monkeypatch.setenv("MINIMAPPR_LOCALIZATION_WINDOW_SECONDS", "0.02")
    monkeypatch.setenv("MINIMAPPR_FUSION_WORKER_COUNT", "1")
    monkeypatch.setenv("MINIMAPPR_SNIPPET_RETENTION_SECONDS", "0")
    monkeypatch.setenv("MINIMAPPR_REPORTING_WINDOW_SECONDS", "1.0")
    monkeypatch.setenv("MINIMAPPR_SIDECAR_TOTAL_JOURNAL_BUDGET_BYTES", str(32 * 1024 * 1024))
    monkeypatch.setenv("MINIMAPPR_SIDECAR_ADMISSION_RESERVE_BYTES", str(4 * 1024 * 1024))
    monkeypatch.setenv("MINIMAPPR_SIDECAR_DERIVED_CACHE_BUDGET_BYTES", str(16 * 1024 * 1024))
    monkeypatch.setenv("MINIMAPPR_SIDECAR_DERIVED_CACHE_ADMISSION_RESERVE_BYTES", str(2 * 1024 * 1024))
    monkeypatch.setenv("MINIMAPPR_SIDECAR_CLASSIFIER_COMMAND_JSON", "")
    return spool_dir


def _wait_for_detection_from_node(client: TestClient, *, node_id: str, timeout_s: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        response = client.get("/api/v1/detections", params={"limit": 20})
        assert response.status_code == 200
        detections = response.json()
        for detection in detections:
            if detection.get("source_node_id") == node_id:
                return detection
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for detection from node {node_id}")


def _load_first_journal_entry(spool_dir: Path) -> dict:
    index_paths = sorted((spool_dir / "journal" / "streams").glob("*/segments/*.index.jsonl"))
    for index_path in index_paths:
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                return json.loads(line)

    # Channel-only ingest does not persist segment index files; reconstruct a
    # minimal entry from seg-mem source handles in manifests.
    manifest_dir = spool_dir / "journal" / "manifests"
    manifest_paths = sorted(manifest_dir.rglob("*.json")) if manifest_dir.exists() else []
    for manifest_path in manifest_paths:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_handles = payload.get("source_handles") or []
        if not source_handles:
            continue
        first_handle = source_handles[0]
        segment_id = str(first_handle.get("segment_id") or "")
        stream_key = str(first_handle.get("stream_key") or "")
        if not stream_key or not segment_id.startswith("seg-mem-"):
            continue
        parts = segment_id.split("-")
        if len(parts) < 4:
            continue
        try:
            journal_epoch = int(parts[2])
            journal_sequence = int(parts[3])
        except ValueError:
            continue
        return {
            "stream_key": stream_key,
            "journal_epoch": journal_epoch,
            "journal_sequence": journal_sequence,
        }

    raise AssertionError("expected at least one journal entry")


def _load_cursor_row(spool_dir: Path, stream_key: str, *, consumer_name: str = "python-ingest") -> sqlite3.Row:
    connection = sqlite3.connect(spool_dir / "journal" / "consumer_state.sqlite3")
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            """
            SELECT journal_epoch, last_fully_processed_journal_sequence
            FROM consumer_cursors
            WHERE consumer_name = ? AND stream_key = ?
            """,
            (consumer_name, stream_key),
        ).fetchone()
        assert row is not None
        return row
    finally:
        connection.close()


def _synthesized_localizable_channels(sample_rate_hz: int) -> np.ndarray:
    duration_seconds = 0.8
    timestamps_s = np.linspace(0.0, duration_seconds, int(round(duration_seconds * sample_rate_hz)), endpoint=False)
    chirp = np.sin(2.0 * np.pi * (700.0 * timestamps_s + 850.0 * np.square(timestamps_s)))
    tone = 0.35 * np.sin(2.0 * np.pi * 1700.0 * timestamps_s)
    mono_signal = (0.65 * chirp + tone).astype(np.float32)
    mono_signal *= np.hanning(mono_signal.size).astype(np.float32)
    padded_signal = prepend_noise_padding(
        mono_signal,
        pad_samples=max(1, int(round(0.25 * sample_rate_hz))),
        noise_rms=0.002,
        seed=sample_rate_hz,
    )
    return synthesize_delayed_array_channels(
        padded_signal,
        sample_rate_hz,
        source_position_m=TEST_SOURCE_POSITION_M,
    )


@pytest.mark.asyncio
async def test_fusion_prefers_authoritative_rust_classification_over_python_reclassify(
    tmp_path: Path,
) -> None:
    settings = Settings(
        runtime_profile="birdnet_hybrid_production",
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
        await fusion.ingest_localized_render(
            payload=LocalizedClassifierRenderRequest(
                manifest_id="manifest-rust-authoritative-1",
                node=_node_spec("sirith-rust-authoritative"),
                event_time_ns=time.time_ns(),
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
    settings = Settings(
        runtime_profile="birdnet_hybrid_production",
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
        assert detection["feature_summary"]["rust_manifest_id"] == "manifest-rust-python-classify-1"
        assert detection["feature_summary"]["classification_path"] == "omni"
        status = await fusion.status()
        assert status["queue"]["classification_depth"] == 0
        assert status["metrics"]["birdnet_chunk_dispatches_suppressed"] == 0
    finally:
        await fusion.stop()
        await storage.close()


@pytest.mark.asyncio
async def test_live_rust_sidecar_emits_authoritative_classification_fields(tmp_path: Path) -> None:
    sample_rate_hz = 16_000
    node_id = "sirith-rust-authoritative-live"
    helper_path = _write_fake_sidecar_classifier_helper(tmp_path)
    payload = _binary_ingest_payload(
        node_id=node_id,
        frames=[
            _binary_frame(
                _synthesized_localizable_channels(sample_rate_hz),
                start_time_ns=time.time_ns(),
                sequence=7,
                start_sample_index=0,
                sample_rate_hz=sample_rate_hz,
            )
        ],
    )

    with _running_rust_sidecar(
        tmp_path,
        extra_env={
            "MINIMAPPR_SIDECAR_CLASSIFIER_COMMAND_JSON": json.dumps(
                [sys.executable, str(helper_path)]
            ),
            "MINIMAPPR_CLASSIFICATION_WINDOW_SECONDS": "2.0",
            "MINIMAPPR_MAX_SENSOR_BUFFER_SECONDS": "4.0",
        },
    ) as (port, spool_dir):
        response = _http_json(
            port,
            "/api/v1/ingest/binary",
            method="POST",
            body=payload,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert response["accepted"] is True

        manifest_payloads = _wait_for_manifest_pair(spool_dir)
        classifier_manifest = next(
            manifest_payload
            for manifest_payload in manifest_payloads
            if manifest_payload["manifest_type"] == "classifier_render"
        )
        assert classifier_manifest["birdnet"]["label"] == "winter wren"
        assert classifier_manifest["birdnet"]["label_confidence"] == pytest.approx(0.91)
        assert classifier_manifest["birdnet"]["scores"]["sparrow"] == pytest.approx(0.04)

        settings = Settings(
            runtime_profile="birdnet_hybrid_production",
            db_path=tmp_path / "authoritative-live.db",
            snippet_dir=tmp_path / "snippets-live",
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
            transport = _RecordingHttpIngestTransport(fusion)
            consumer = IngestSpoolConsumer(
                config=IngestSpoolConfig(
                    spool_dir=spool_dir,
                    ready_ttl_seconds=60.0,
                    failed_ttl_seconds=86_400.0,
                    tmp_ttl_seconds=300.0,
                    poll_interval_seconds=0.05,
                    worker_count=1,
                    storage_mode="journal",
                    runtime_profile="birdnet_hybrid_production",
                ),
                ingest_transport=transport,
            )

            summary = await consumer.run_once(now_ns=time.time_ns())

            assert summary.processed == 1
            assert transport.localized_render_deliveries == 1

            detections = await storage.list_detections(limit=10)
            assert len(detections) == 1
            detection = detections[0]
            assert detection["label"] == "winter wren"
            assert detection["label_confidence"] == pytest.approx(0.91)
            assert detection["feature_summary"]["classification_path"] == "rust_manifest"
            assert detection["feature_summary"]["rust_classification_authoritative"] is True
        finally:
            await fusion.stop()
            await storage.close()


@pytest.mark.asyncio
async def test_rust_manifest_handoff_reaches_python_without_raw_replay(tmp_path: Path) -> None:
    sample_rate_hz = 16_000
    node_id = "sirith-rust-handoff"
    start_time_ns = 1_739_970_000_000_000_000
    payload = _binary_ingest_payload(
        node_id=node_id,
        frames=[
            _binary_frame(
                _synthesized_localizable_channels(sample_rate_hz),
                start_time_ns=start_time_ns,
                sequence=1,
                start_sample_index=0,
                sample_rate_hz=sample_rate_hz,
            )
        ],
    )

    with _running_rust_sidecar(tmp_path) as (port, spool_dir):
        response = _http_json(
            port,
            "/api/v1/ingest/binary",
            method="POST",
            body=payload,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert response["accepted"] is True

        manifest_payloads = _wait_for_manifest_pair(spool_dir)
        assert {payload["manifest_type"] for payload in manifest_payloads} >= {
            "localization_result",
            "classifier_render",
        }

        journal_entry = _load_first_journal_entry(spool_dir)
        stream_key = str(journal_entry["stream_key"])
        journal_sequence = int(journal_entry["journal_sequence"])

        settings = Settings(
            runtime_profile="birdnet_hybrid_production",
            db_path=tmp_path / "handoff.db",
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
            transport = _RecordingHttpIngestTransport(fusion)
            consumer = IngestSpoolConsumer(
                config=IngestSpoolConfig(
                    spool_dir=spool_dir,
                    ready_ttl_seconds=60.0,
                    failed_ttl_seconds=86_400.0,
                    tmp_ttl_seconds=300.0,
                    poll_interval_seconds=0.05,
                    worker_count=1,
                    storage_mode="journal",
                    runtime_profile="birdnet_hybrid_production",
                ),
                ingest_transport=transport,
            )

            summary = await consumer.run_once(now_ns=time.time_ns())

            assert summary.processed == 1
            assert transport.binary_deliveries == 0
            assert transport.localized_render_deliveries == 1

            detections = await storage.list_detections(limit=10)
            assert len(detections) == 1
            detection = detections[0]
            assert detection["label"] == "sparrow"
            assert detection["source_node_id"] == node_id
            assert detection["reporting_modality"] == "localized"
            assert detection["feature_summary"]["localization_method"] == "srp_phat"
            assert str(detection["feature_summary"]["rust_render_kind"]).startswith("birdnet_")

            cursor_row = _load_cursor_row(spool_dir, stream_key)
            assert int(cursor_row["journal_epoch"]) == 1
            assert int(cursor_row["last_fully_processed_journal_sequence"]) == journal_sequence
            assert list((spool_dir / "processing").glob("*")) == []
        finally:
            await fusion.stop()
            await storage.close()


def test_http_app_in_production_mode_consumes_rust_sidecar_journal(monkeypatch, tmp_path: Path) -> None:
    sidecar_port = _pick_free_tcp_port()
    spool_dir = _configure_http_app_sidecar_env(monkeypatch, tmp_path, sidecar_port=sidecar_port)
    monkeypatch.setattr("minimappr.main.create_classifier", lambda settings: _ConstantBirdClassifier())
    sample_rate_hz = 16_000
    node_id = "sirith-rust-http-app"
    start_time_ns = time.time_ns()
    payload = _binary_ingest_payload(
        node_id=node_id,
        frames=[
            _binary_frame(
                _synthesized_localizable_channels(sample_rate_hz),
                start_time_ns=start_time_ns,
                sequence=2,
                start_sample_index=0,
                sample_rate_hz=sample_rate_hz,
            )
        ],
    )

    with TestClient(app) as client:
        diagnostics_response = client.get("/api/v1/system/diagnostics")
        assert diagnostics_response.status_code == 200
        diagnostics = diagnostics_response.json()
        assert diagnostics["sidecar"]["enabled"] is True
        assert diagnostics["sidecar"]["status"] == "running"
        assert diagnostics["sidecar"]["health"]["status"] == "ok"
        assert diagnostics["sidecar"]["health"]["backend"]["storage_mode"] == "journal"

        blocked_response = client.post(
            "/api/v1/ingest/binary",
            content=payload,
            headers={"content-type": "application/octet-stream"},
        )
        assert blocked_response.status_code == 410
        assert "sidecar" in blocked_response.json()["detail"]

        sidecar_response = _http_json(
            sidecar_port,
            "/api/v1/ingest/binary",
            method="POST",
            body=payload,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert sidecar_response["accepted"] is True

        detection = _wait_for_detection_from_node(client, node_id=node_id)
        assert detection["label"] == "sparrow"
        assert detection["source_node_id"] == node_id
        assert detection["reporting_modality"] == "localized"
        assert detection["feature_summary"]["localization_method"] == "srp_phat"
        assert detection["feature_summary"]["rust_manifest_id"]
        assert str(detection["feature_summary"]["rust_render_kind"]).startswith("birdnet_")

        journal_dir = spool_dir / "journal"
        assert journal_dir.exists()
        assert (journal_dir / "consumer_state.sqlite3").exists()
