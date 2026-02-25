from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from minimappr.main import app
from minimappr.utils.audio import encode_pcm16le_b64


def _configure_env(monkeypatch, tmp_path: Path, *, snippet_retention_seconds: int) -> Path:
    db_path = tmp_path / "http_api.db"
    snippet_dir = tmp_path / "snippets"
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(db_path))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(snippet_dir))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("MINIMAPPR_TRIGGER_RMS", "0.000001")
    monkeypatch.setenv("MINIMAPPR_TRIGGER_COOLDOWN_SECONDS", "0")
    monkeypatch.setenv("MINIMAPPR_LOCALIZATION_WINDOW_SECONDS", "0.02")
    monkeypatch.setenv("MINIMAPPR_FUSION_WORKER_COUNT", "1")
    monkeypatch.setenv("MINIMAPPR_SNIPPET_RETENTION_SECONDS", str(snippet_retention_seconds))
    return db_path


def _ingest_single_frame(client: TestClient, *, start_time_ns: int) -> None:
    samples = np.random.default_rng(1234).normal(0.0, 0.5, size=(1, 1024)).astype(np.float32)
    payload = {
        "node": {
            "id": "http-node-1",
            "node_type": "point",
            "position_m": [0.0, 0.0, 0.0],
            "sensor_offsets_m": [[0.0, 0.0, 0.0]],
            "capabilities": ["audio"],
            "metadata": {},
            "properties": {},
        },
        "frame": {
            "start_time_ns": start_time_ns,
            "sample_rate_hz": 16000,
            "channels": 1,
            "encoding": "pcm16le",
            "samples_b64": encode_pcm16le_b64(samples),
            "sequence": 1,
            "source_type": "raw_sensor",
        },
    }
    response = client.post("/api/v1/ingest/frame", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] is True


def _wait_for_detections(client: TestClient, *, timeout_s: float = 2.0) -> list[dict]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        response = client.get("/api/v1/detections", params={"limit": 10})
        assert response.status_code == 200
        detections = response.json()
        if detections:
            return detections
        time.sleep(0.05)
    return []


def test_http_ingest_and_cop_status(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=0)

    with TestClient(app) as client:
        _ingest_single_frame(client, start_time_ns=time.time_ns())
        detections = _wait_for_detections(client)
        assert detections

        nodes_response = client.get("/api/v1/nodes", params={"limit": 1})
        assert nodes_response.status_code == 200
        nodes = nodes_response.json()
        assert len(nodes) <= 1

        cop_response = client.get("/api/v1/cop/status")
        assert cop_response.status_code == 200
        cop = cop_response.json()
        assert cop["active_nodes"] >= 1
        assert cop["active_tracks"] >= 0


def test_detection_audio_rejects_paths_outside_snippet_root(monkeypatch, tmp_path: Path) -> None:
    db_path = _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=3600)

    with TestClient(app) as client:
        _ingest_single_frame(client, start_time_ns=time.time_ns())
        detections = _wait_for_detections(client)
        assert detections
        detection_id = detections[0]["id"]

        rogue_snippet = tmp_path / "outside.wav"
        rogue_snippet.write_bytes(b"RIFF0000WAVEfmt ")
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE detections SET snippet_path = ?, snippet_expires_ns = ? WHERE id = ?",
                (str(rogue_snippet), int(time.time_ns() + 1_000_000_000), detection_id),
            )
            conn.commit()

        response = client.get(f"/api/v1/detections/{detection_id}/audio")
        assert response.status_code == 403
        assert "outside snippet directory" in response.text
