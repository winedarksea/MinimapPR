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


def _binary_ingest_payload(frames: list[bytes], *, sort_by_toa: bool = False) -> bytes:
    payload = bytearray()
    payload += b"MMB1"
    payload += struct.pack("<BBH", 1, 1 if sort_by_toa else 0, len(frames))
    payload += _binary_node()
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
    monkeypatch.setenv("MINIMAPPR_INGEST_SPOOL_POLL_INTERVAL_SECONDS", "0.05")
    monkeypatch.setenv("MINIMAPPR_TRIGGER_RMS", "0.000001")
    monkeypatch.setenv("MINIMAPPR_TRIGGER_COOLDOWN_SECONDS", "0")
    monkeypatch.setenv("MINIMAPPR_LOCALIZATION_WINDOW_SECONDS", "0.02")
    monkeypatch.setenv("MINIMAPPR_FUSION_WORKER_COUNT", "1")
    monkeypatch.setenv("MINIMAPPR_SNIPPET_RETENTION_SECONDS", "0")
    monkeypatch.setenv("MINIMAPPR_REPORTING_WINDOW_SECONDS", "1.0")
    return spool_dir


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
    (segments_dir / f"{segment_id}.bin").write_bytes(body)
    (segments_dir / f"{segment_id}.index.jsonl").write_text(
        "{"
        f'"journal_id":"{journal_id}",' 
        f'"journal_epoch":{journal_epoch},'
        f'"journal_sequence":{journal_sequence},'
        f'"stream_key":"{stream_key}",' 
        f'"segment_id":"{segment_id}",' 
        f'"endpoint":"{endpoint}",' 
        '"content_type":"application/octet-stream",'
        f'"ingest_received_ns":{received_ns},'
        f'"received_ns":{received_ns},'
        '"payload_offset_bytes":0,'
        f'"payload_length_bytes":{len(body)},'
        '"body_offset_bytes":0,'
        f'"body_length_bytes":{len(body)},'
        f'"integrity_hash":"{integrity_hash or hashlib.sha256(body).hexdigest()}"'
        "}\n",
        encoding="utf-8",
    )


class _RecordingIngestTransport:
    def __init__(self) -> None:
        self.binary_payloads: list[object] = []
        self.store_forward_payloads: list[object] = []

    async def deliver_binary(self, payload: object) -> None:
        self.binary_payloads.append(payload)

    async def deliver_store_forward(self, payload: object) -> None:
        self.store_forward_payloads.append(payload)


def _journal_consumer(spool_dir: Path, transport: _RecordingIngestTransport) -> IngestSpoolConsumer:
    return IngestSpoolConsumer(
        config=IngestSpoolConfig(
            spool_dir=spool_dir,
            ready_ttl_seconds=60.0,
            failed_ttl_seconds=86_400.0,
            tmp_ttl_seconds=300.0,
            poll_interval_seconds=0.05,
            worker_count=1,
            storage_mode="journal",
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
    exception_row = _load_exception_row(spool_dir, "jrnl-00000000000000000003")
    assert str(exception_row["status"]) == "failed"
