from __future__ import annotations

import json
import hashlib
import sqlite3
import struct
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from minimappr.api.binary_ingest import parse_binary_ingest_payload
from minimappr.api.rust_dsp_manifests import LocalizedClassifierRenderRequest
from minimappr.api.spool_consumer import IngestSpoolConfig, IngestSpoolConsumer
from minimappr.main import app


def _binary_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    assert len(encoded) <= 255
    return struct.pack("<B", len(encoded)) + encoded


def _binary_node(*, node_id: str = "binary-node-1", sensor_count: int = 1) -> bytes:
    payload = bytearray()
    payload += _binary_string(node_id)
    payload += struct.pack("<B", 0)
    payload += struct.pack("<fff", 0.0, 0.0, 0.0)
    payload += struct.pack("<B", 0)
    payload += struct.pack("<B", sensor_count)
    for _ in range(sensor_count):
        payload += struct.pack("<fff", 0.0, 0.0, 0.0)
    capabilities = ["audio", "gps_optional"]
    payload += struct.pack("<B", len(capabilities))
    for capability in capabilities:
        payload += _binary_string(capability)
    payload += _binary_string("test-hardware")
    payload += _binary_string("test-firmware")
    payload += _binary_string("gps_locked")
    payload += _binary_string("test")
    payload += struct.pack("<I", 3)
    return bytes(payload)


def _binary_frame(
    samples: np.ndarray,
    *,
    start_time_ns: int,
    sequence: int,
    start_sample_index: int,
    sample_rate_hz: int = 16000,
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
    payload += struct.pack("<B", 1)
    payload += struct.pack("<B", 1)
    payload += struct.pack("<I", 123)
    payload += struct.pack("<I", 2)
    payload += struct.pack("<q", -17)
    payload += struct.pack("<d", 0.25)
    payload += struct.pack("<Q", sequence)
    payload += struct.pack("<Q", 0)
    payload += struct.pack("<Q", 0)
    payload += struct.pack("<Q", 0)
    payload += struct.pack("<I", 1)
    payload += struct.pack("<Q", 0)
    payload += struct.pack("<i", 200)
    payload += struct.pack("<Q", 2500)
    payload += struct.pack("<B", 6)
    payload += struct.pack("<i", -4)
    payload += struct.pack("<I", 2)
    payload += struct.pack("<Q", 11)
    payload += struct.pack("<Q", 7)
    payload += struct.pack("<Q", 3)
    payload += struct.pack("<Q", 5)
    payload += struct.pack("<B", 0)
    payload += struct.pack("<I", samples_per_channel)
    payload += pcm16
    return bytes(payload)


def _binary_ingest_payload(
    frames: list[bytes],
    *,
    sort_by_toa: bool = False,
    node_id: str = "binary-node-1",
    sensor_count: int = 1,
) -> bytes:
    payload = bytearray()
    payload += b"MMB1"
    payload += struct.pack("<BBH", 1, 1 if sort_by_toa else 0, len(frames))
    payload += _binary_node(node_id=node_id, sensor_count=sensor_count)
    for frame in frames:
        payload += frame
    return bytes(payload)


def _configure_env(monkeypatch, tmp_path: Path) -> Path:
    db_path = tmp_path / "http_api.db"
    snippet_dir = tmp_path / "snippets"
    artifact_dir = tmp_path / "artifacts"
    spool_dir = tmp_path / "spool"
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(db_path))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(snippet_dir))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("MINIMAPPR_INGEST_SPOOL_DIR", str(spool_dir))
    monkeypatch.setenv("MINIMAPPR_INGEST_STORAGE_MODE", "journal")
    monkeypatch.setenv("MINIMAPPR_DIRECT_INGEST_ENABLED", "true")
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_INGEST_SPOOL_POLL_INTERVAL_SECONDS", "0.05")
    monkeypatch.setenv("MINIMAPPR_TRIGGER_RMS", "0.000001")
    monkeypatch.setenv("MINIMAPPR_TRIGGER_COOLDOWN_SECONDS", "0")
    monkeypatch.setenv("MINIMAPPR_LOCALIZATION_WINDOW_SECONDS", "0.02")
    monkeypatch.setenv("MINIMAPPR_FUSION_WORKER_COUNT", "1")
    monkeypatch.setenv("MINIMAPPR_SNIPPET_RETENTION_SECONDS", "0")
    monkeypatch.setenv("MINIMAPPR_REPORTING_WINDOW_SECONDS", "1.0")
    return spool_dir


def _journal_sync_source(time_quality: str | None) -> str | None:
    if time_quality in {"gps_locked", "gps_holdover"}:
        return "gps"
    if time_quality in {"ntp_disciplined", "ntp_sync"}:
        return "ntp"
    if time_quality == "free_running":
        return "free_running"
    return time_quality


def _journal_channel_layout(channel_count: int | None) -> str | None:
    if channel_count is None:
        return None
    if channel_count == 1:
        return "mono"
    if channel_count == 2:
        return "stereo"
    if channel_count == 4:
        return "tetrahedral"
    return f"{channel_count}ch_interleaved"


def _write_journal_item(
    spool_dir: Path,
    *,
    endpoint: str,
    body: bytes,
    received_ns: int,
    journal_id: str,
    journal_sequence: int,
    journal_epoch: int = 1,
    stream_key: str = "binary-node-1__audio_main__test",
    segment_id: str = "seg-test-1",
    integrity_hash: str | None = None,
) -> None:
    segments_dir = spool_dir / "journal" / "streams" / stream_key / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    segment_path = segments_dir / f"{segment_id}.bin"
    segment_path.write_bytes(body)

    payload_integrity_hash = integrity_hash or hashlib.sha256(body).hexdigest()
    journal_entry = {
        "observation_id": f"obs-{journal_epoch:020d}-{journal_sequence:020d}-{stream_key}",
        "journal_id": journal_id,
        "journal_epoch": journal_epoch,
        "journal_sequence": journal_sequence,
        "node_id": stream_key.split("__", 1)[0],
        "stream_id": "audio_main",
        "stream_key": stream_key,
        "sensor_type": "audio",
        "source_type": "raw_sensor",
        "transport": "http_binary",
        "toa_ns": None,
        "tor_ns": None,
        "ingest_received_ns": received_ns,
        "time_quality": None,
        "clock_domain": None,
        "sync_source": None,
        "clock_correction_ns": None,
        "clock_drift_ppm": None,
        "sample_rate_hz": None,
        "channel_count": None,
        "channel_layout": None,
        "sample_index_start": None,
        "sample_count": None,
        "geometry_version": None,
        "orientation_version": None,
        "calibration_version": None,
        "retention_hint": "ephemeral",
        "payload_codec": "binary_mmb1_pcm16le",
        "integrity_hash": payload_integrity_hash,
        "endpoint": endpoint,
        "content_type": "application/octet-stream",
        "segment_id": segment_id,
        "segment_path": str(segment_path),
        "payload_offset_bytes": 0,
        "payload_length_bytes": len(body),
        "body_offset_bytes": 0,
        "body_length_bytes": len(body),
        "received_ns": received_ns,
    }
    if endpoint == "/api/v1/ingest/binary":
        try:
            parsed_payload = parse_binary_ingest_payload(body)
        except ValueError:
            parsed_payload = None
        if parsed_payload is not None:
            first_frame = parsed_payload.buffered_frames[0].frame
            channel_count = max(int(first_frame.channels), len(parsed_payload.node.sensor_offsets_m))
            sample_count = sum(int(buffered.frame.samples_per_channel) for buffered in parsed_payload.buffered_frames)
            time_quality = first_frame.time_quality.value if first_frame.time_quality is not None else None
            journal_entry.update(
                {
                    "node_id": parsed_payload.node.id,
                    "toa_ns": first_frame.toa_ns,
                    "tor_ns": first_frame.tor_ns,
                    "time_quality": time_quality,
                    "clock_domain": "utc",
                    "sync_source": _journal_sync_source(time_quality),
                    "clock_correction_ns": first_frame.timing_diagnostics.get("pps_phase_error_ns"),
                    "clock_drift_ppm": first_frame.timing_diagnostics.get("estimated_ppm"),
                    "sample_rate_hz": first_frame.sample_rate_hz,
                    "channel_count": channel_count,
                    "channel_layout": _journal_channel_layout(channel_count),
                    "sample_index_start": first_frame.start_sample_index,
                    "sample_count": sample_count,
                }
            )

    (segments_dir / f"{segment_id}.index.jsonl").write_text(
        json.dumps(journal_entry) + "\n",
        encoding="utf-8",
    )


class _RecordingIngestTransport:
    def __init__(self) -> None:
        self.binary_payloads: list[object] = []
        self.store_forward_payloads: list[object] = []
        self.localized_render_payloads: list[LocalizedClassifierRenderRequest] = []

    async def deliver_binary(self, payload: object) -> None:
        self.binary_payloads.append(payload)

    async def deliver_store_forward(self, payload: object) -> None:
        self.store_forward_payloads.append(payload)

    async def deliver_localized_render(self, payload: LocalizedClassifierRenderRequest) -> None:
        self.localized_render_payloads.append(payload)


def _journal_consumer(
    spool_dir: Path,
    transport: _RecordingIngestTransport,
    *,
    runtime_profile: str = "default",
) -> IngestSpoolConsumer:
    return IngestSpoolConsumer(
        config=IngestSpoolConfig(
            spool_dir=spool_dir,
            ready_ttl_seconds=60.0,
            failed_ttl_seconds=86_400.0,
            tmp_ttl_seconds=300.0,
            poll_interval_seconds=0.05,
            worker_count=1,
            storage_mode="journal",
            runtime_profile=runtime_profile,
        ),
        ingest_transport=transport,
    )


def _binary_body() -> bytes:
    samples = np.random.default_rng(43).normal(0.0, 0.35, size=(1, 512)).astype(np.float32)
    start_time_ns = time.time_ns()
    return _binary_ingest_payload(
        [
            _binary_frame(
                samples,
                start_time_ns=start_time_ns,
                sequence=1010,
                start_sample_index=0,
            )
        ]
    )


def _write_rust_dsp_manifest_pair(
    spool_dir: Path,
    *,
    stream_key: str,
    segment_id: str,
    payload_length_bytes: int,
    derived_pcm16: bytes,
    created_ns: int,
    journal_epoch: int = 1,
    birdnet_label: str | None = None,
    birdnet_label_confidence: float | None = None,
    birdnet_scores: dict[str, float] | None = None,
) -> None:
    manifest_dir = spool_dir / "journal" / "manifests" / "pending"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    derived_dir = spool_dir / "journal" / "derived-cache"
    derived_dir.mkdir(parents=True, exist_ok=True)
    derived_path = derived_dir / "render-1.bin"
    derived_path.write_bytes(derived_pcm16)

    source_handle = {
        "journal_epoch": journal_epoch,
        "segment_id": segment_id,
        "stream_key": stream_key,
        "payload_offset_bytes": 0,
        "payload_length_bytes": payload_length_bytes,
        "sample_index_start": 0,
        "sample_count": 512,
        "integrity_hash": "",
    }
    classifier_render_payload = {
        "render_id": "render-1",
        "render_kind": "birdnet_hybrid_spatial_blend",
        "sample_rate_hz": 48000,
        "channels": 1,
        "sample_count": len(derived_pcm16) // 2,
        "sample_format": "pcm16le",
        "effective_spatial_band": [1000.0, 3400.0],
        "source_channel_count": 4,
        "fallback_reason": None,
    }
    (manifest_dir / "manifest-localization.json").write_text(
        json.dumps(
            {
                "manifest_id": "manifest-localization",
                "manifest_type": "localization_result",
                "created_ns": created_ns,
                "source_handles": [source_handle],
                "derived_handle": None,
                "localization": {
                    "attempted_algorithm": "srp_phat",
                    "resolved_algorithm": "srp_phat",
                    "steering_direction": [0.0, 1.0, 0.0],
                    "position_m": [1.0, 2.0, 0.0],
                    "confidence": 0.82,
                    "residual_rms_seconds": 0.0001,
                    "sound_speed_mps": 343.2,
                    "effective_band_hz": [300.0, 3500.0],
                    "pair_tdoas": [],
                },
                "classifier_render": classifier_render_payload,
                "birdnet": None,
                "coverage_stats": None,
                "promotion_ready": False,
            }
        ),
        encoding="utf-8",
    )
    (manifest_dir / "manifest-render.json").write_text(
        json.dumps(
            {
                "manifest_id": "manifest-render",
                "manifest_type": "classifier_render",
                "created_ns": created_ns,
                "source_handles": [source_handle],
                "derived_handle": {
                    "journal_epoch": 0,
                    "segment_id": "render-1",
                    "stream_key": stream_key,
                    "payload_offset_bytes": 0,
                    "payload_length_bytes": len(derived_pcm16),
                    "sample_index_start": None,
                    "sample_count": len(derived_pcm16) // 2,
                    "integrity_hash": "",
                    "segment_path": str(derived_path),
                },
                "localization": None,
                "classifier_render": classifier_render_payload,
                "birdnet": {
                    "steering_solution": "srp_phat:0.0,1.0,0.0",
                    "classifier_source_node": stream_key,
                    "spatial_blend_mode": "birdnet_hybrid_spatial_blend",
                    "effective_spatial_band": [1000.0, 3400.0],
                    "confidence": 0.82,
                    "fallback_reason": None,
                    "label": birdnet_label,
                    "label_confidence": birdnet_label_confidence,
                    "scores": birdnet_scores,
                },
                "coverage_stats": None,
                "promotion_ready": True,
            }
        ),
        encoding="utf-8",
    )


def _load_cursor_row(spool_dir: Path, *, consumer_name: str = "python-ingest") -> sqlite3.Row:
    connection = sqlite3.connect(spool_dir / "journal" / "consumer_state.sqlite3")
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            """
            SELECT stream_key, journal_epoch, last_fully_processed_journal_sequence
            FROM consumer_cursors
            WHERE consumer_name = ?
            """,
            (consumer_name,),
        ).fetchone()
        assert row is not None
        return row
    finally:
        connection.close()


def _load_exception_row(
    spool_dir: Path,
    journal_id: str,
    *,
    consumer_name: str = "python-ingest",
) -> sqlite3.Row:
    connection = sqlite3.connect(spool_dir / "journal" / "consumer_state.sqlite3")
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            """
            SELECT journal_id, status
            FROM consumer_exceptions
            WHERE consumer_name = ? AND journal_id = ?
            """,
            (consumer_name, journal_id),
        ).fetchone()
        assert row is not None
        return row
    finally:
        connection.close()


def test_http_app_processes_journal_binary_payload(monkeypatch, tmp_path: Path) -> None:
    spool_dir = _configure_env(monkeypatch, tmp_path)
    _write_journal_item(
        spool_dir,
        endpoint="/api/v1/ingest/binary",
        body=_binary_body(),
        received_ns=time.time_ns(),
        journal_id="jrnl-00000000000000000001",
        journal_sequence=1,
    )

    with TestClient(app) as client:
        deadline = time.monotonic() + 2.0
        nodes: list[dict] = []
        while time.monotonic() < deadline:
            nodes_response = client.get("/api/v1/nodes", params={"limit": 10})
            assert nodes_response.status_code == 200
            nodes = nodes_response.json()
            if any(row["id"] == "binary-node-1" for row in nodes):
                break
            time.sleep(0.05)

        assert any(row["id"] == "binary-node-1" for row in nodes)
        cursor_row = _load_cursor_row(spool_dir)
        assert int(cursor_row["journal_epoch"]) == 1
        assert int(cursor_row["last_fully_processed_journal_sequence"]) == 1


@pytest.mark.asyncio
async def test_journal_consumer_processes_each_entry_once(tmp_path: Path) -> None:
    spool_dir = tmp_path / "spool"
    transport = _RecordingIngestTransport()
    consumer = _journal_consumer(spool_dir, transport)
    _write_journal_item(
        spool_dir,
        endpoint="/api/v1/ingest/binary",
        body=_binary_body(),
        received_ns=time.time_ns(),
        journal_id="jrnl-00000000000000000001",
        journal_sequence=1,
    )

    first_summary = await consumer.run_once()
    second_summary = await consumer.run_once()

    assert first_summary.processed == 1
    assert second_summary.processed == 0
    assert len(transport.binary_payloads) == 1
    cursor_row = _load_cursor_row(spool_dir)
    assert int(cursor_row["journal_epoch"]) == 1
    assert int(cursor_row["last_fully_processed_journal_sequence"]) == 1
    assert list((spool_dir / "processing").glob("*")) == []


@pytest.mark.asyncio
async def test_journal_consumer_marks_failed_entry_terminally(tmp_path: Path) -> None:
    spool_dir = tmp_path / "spool"
    transport = _RecordingIngestTransport()
    consumer = _journal_consumer(spool_dir, transport)
    _write_journal_item(
        spool_dir,
        endpoint="/api/v1/ingest/binary",
        body=b"BAD!",
        received_ns=time.time_ns(),
        journal_id="jrnl-00000000000000000002",
        journal_sequence=2,
    )

    first_summary = await consumer.run_once()
    second_summary = await consumer.run_once()

    assert first_summary.failed == 1
    assert second_summary.failed == 0
    cursor_row = _load_cursor_row(spool_dir)
    assert int(cursor_row["last_fully_processed_journal_sequence"]) == 2
    exception_row = _load_exception_row(spool_dir, "jrnl-00000000000000000002")
    assert str(exception_row["status"]) == "failed"
    failed_files = sorted(path.name for path in (spool_dir / "failed").glob("*"))
    assert failed_files == ["jrnl-00000000000000000002.journal.json"]
    assert transport.binary_payloads == []


@pytest.mark.asyncio
async def test_journal_consumer_rejects_hash_mismatch(tmp_path: Path) -> None:
    spool_dir = tmp_path / "spool"
    transport = _RecordingIngestTransport()
    consumer = _journal_consumer(spool_dir, transport)
    _write_journal_item(
        spool_dir,
        endpoint="/api/v1/ingest/binary",
        body=_binary_body(),
        received_ns=time.time_ns(),
        journal_id="jrnl-00000000000000000003",
        journal_sequence=3,
        integrity_hash="0" * 64,
    )

    summary = await consumer.run_once()

    assert summary.failed == 1
    assert transport.binary_payloads == []


@pytest.mark.asyncio
async def test_journal_consumer_prefers_rust_dsp_manifest_pair(tmp_path: Path) -> None:
    spool_dir = tmp_path / "spool"
    transport = _RecordingIngestTransport()
    consumer = _journal_consumer(
        spool_dir,
        transport,
        runtime_profile="birdnet_hybrid_production",
    )
    now_ns = time.time_ns()
    stream_key = "sirith-array__audio_main__test"
    raw_samples = np.random.default_rng(55).normal(0.0, 0.25, size=(4, 512)).astype(np.float32)
    raw_body = _binary_ingest_payload(
        [
            _binary_frame(
                raw_samples,
                start_time_ns=now_ns - 300_000_000,
                sequence=41,
                start_sample_index=0,
                sample_rate_hz=16_000,
            )
        ],
        node_id="sirith-array",
        sensor_count=4,
    )
    _write_journal_item(
        spool_dir,
        endpoint="/api/v1/ingest/binary",
        body=raw_body,
        received_ns=now_ns - 200_000_000,
        journal_id="jrnl-00000000000000000041",
        journal_sequence=41,
        stream_key=stream_key,
        segment_id="seg-rust-manifest-1",
    )
    render_audio = np.clip(np.random.default_rng(56).normal(0.0, 0.3, size=48_000), -1.0, 0.9999695)
    render_pcm16 = (render_audio * 32768.0).astype("<i2").tobytes()
    _write_rust_dsp_manifest_pair(
        spool_dir,
        stream_key=stream_key,
        segment_id="seg-rust-manifest-1",
        payload_length_bytes=len(raw_body),
        derived_pcm16=render_pcm16,
        created_ns=now_ns - 100_000_000,
    )

    summary = await consumer.run_once(now_ns=now_ns)

    assert summary.processed == 1
    assert transport.binary_payloads == []
    assert transport.store_forward_payloads == []
    assert len(transport.localized_render_payloads) == 1
    localized_payload = transport.localized_render_payloads[0]
    assert localized_payload.manifest_id == "manifest-localization"
    assert localized_payload.render_kind == "birdnet_hybrid_spatial_blend"
    assert localized_payload.localization_method == "srp_phat"
    cursor_row = _load_cursor_row(spool_dir)
    assert int(cursor_row["journal_epoch"]) == 1
    assert int(cursor_row["last_fully_processed_journal_sequence"]) == 41
    assert list((spool_dir / "processing").glob("*")) == []


@pytest.mark.asyncio
async def test_journal_consumer_loads_authoritative_rust_classification(tmp_path: Path) -> None:
    spool_dir = tmp_path / "spool"
    transport = _RecordingIngestTransport()
    consumer = _journal_consumer(
        spool_dir,
        transport,
        runtime_profile="birdnet_hybrid_production",
    )

    now_ns = time.time_ns()
    stream_key = "sirith-array__audio_main__rust-authoritative"
    raw_samples = np.random.default_rng(57).normal(0.0, 0.25, size=(4, 512)).astype(np.float32)
    raw_body = _binary_ingest_payload(
        [
            _binary_frame(
                raw_samples,
                start_time_ns=now_ns - 300_000_000,
                sequence=42,
                start_sample_index=0,
                sample_rate_hz=16_000,
            )
        ],
        node_id="sirith-array",
        sensor_count=4,
    )
    _write_journal_item(
        spool_dir,
        endpoint="/api/v1/ingest/binary",
        body=raw_body,
        received_ns=now_ns - 200_000_000,
        journal_id="jrnl-00000000000000000042",
        journal_sequence=42,
        stream_key=stream_key,
        segment_id="seg-rust-manifest-2",
    )
    render_audio = np.clip(np.random.default_rng(58).normal(0.0, 0.3, size=48_000), -1.0, 0.9999695)
    render_pcm16 = (render_audio * 32768.0).astype("<i2").tobytes()
    _write_rust_dsp_manifest_pair(
        spool_dir,
        stream_key=stream_key,
        segment_id="seg-rust-manifest-2",
        payload_length_bytes=len(raw_body),
        derived_pcm16=render_pcm16,
        created_ns=now_ns - 100_000_000,
        birdnet_label="winter wren",
        birdnet_label_confidence=0.91,
        birdnet_scores={"winter wren": 0.91, "song sparrow": 0.08},
    )

    summary = await consumer.run_once(now_ns=now_ns)

    assert summary.processed == 1
    assert len(transport.localized_render_payloads) == 1
    localized_payload = transport.localized_render_payloads[0]
    assert localized_payload.authoritative_classification is not None
    assert localized_payload.authoritative_classification.label == "winter wren"
    assert localized_payload.authoritative_classification.confidence == pytest.approx(0.91)
    assert localized_payload.authoritative_classification.scores["song sparrow"] == pytest.approx(0.08)
