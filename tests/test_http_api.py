from __future__ import annotations

import io
import sqlite3
import time
import wave
from pathlib import Path

import numpy as np
import pytest
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


def _ingest_single_frame(
    client: TestClient,
    *,
    start_time_ns: int,
    metadata: dict | None = None,
    environment: dict | None = None,
) -> dict:
    samples = np.random.default_rng(1234).normal(0.0, 0.5, size=(1, 1024)).astype(np.float32)
    payload = {
        "node": {
            "id": "http-node-1",
            "node_type": "point",
            "position_m": [0.0, 0.0, 0.0],
            "sensor_offsets_m": [[0.0, 0.0, 0.0]],
            "capabilities": ["audio"],
            "metadata": metadata or {},
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
    if environment is not None:
        payload["environment"] = environment
    response = client.post("/api/v1/ingest/frame", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] is True
    return body


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


def test_http_ingest_duplicate_frame_is_idempotent(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=0)
    start_time_ns = time.time_ns()
    samples = np.random.default_rng(55).normal(0.0, 0.45, size=(1, 1024)).astype(np.float32)
    payload = {
        "node": {
            "id": "http-node-duplicate",
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
            "sequence": 11,
            "source_type": "raw_sensor",
        },
    }

    with TestClient(app) as client:
        first = client.post("/api/v1/ingest/frame", json=payload)
        assert first.status_code == 200
        first_body = first.json()
        assert first_body["duplicate"] is False

        second = client.post("/api/v1/ingest/frame", json=payload)
        assert second.status_code == 200
        second_body = second.json()
        assert second_body["accepted"] is True
        assert second_body["duplicate"] is True
        assert second_body["triggered"] is False


def test_http_store_forward_deduplicates_and_preserves_last_seen(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=0)
    latest_live_start_ns = time.time_ns()
    older_start_ns = latest_live_start_ns - 4_000_000_000
    oldest_start_ns = latest_live_start_ns - 5_000_000_000
    oldest_toa_ns = oldest_start_ns - 1_000_000
    older_toa_ns = older_start_ns - 1_000_000

    with TestClient(app) as client:
        _ingest_single_frame(client, start_time_ns=latest_live_start_ns)

        buffered_payload = {
            "node": {
                "id": "http-node-1",
                "node_type": "point",
                "position_m": [0.0, 0.0, 0.0],
                "sensor_offsets_m": [[0.0, 0.0, 0.0]],
                "capabilities": ["audio"],
                "metadata": {},
                "properties": {},
            },
            "sort_by_toa": True,
            "buffered_frames": [
                {
                    "frame": {
                        "start_time_ns": older_start_ns,
                        "sample_rate_hz": 16000,
                        "channels": 1,
                        "encoding": "pcm16le",
                        "samples_b64": encode_pcm16le_b64(
                            np.random.default_rng(1).normal(0.0, 0.4, size=(1, 1024)).astype(np.float32)
                        ),
                        "sequence": 102,
                        "source_type": "raw_sensor",
                        "toa_ns": older_toa_ns,
                    }
                },
                {
                    "frame": {
                        "start_time_ns": oldest_start_ns,
                        "sample_rate_hz": 16000,
                        "channels": 1,
                        "encoding": "pcm16le",
                        "samples_b64": encode_pcm16le_b64(
                            np.random.default_rng(2).normal(0.0, 0.42, size=(1, 1024)).astype(np.float32)
                        ),
                        "sequence": 101,
                        "source_type": "raw_sensor",
                        "toa_ns": oldest_toa_ns,
                    }
                },
                {
                    "frame": {
                        "start_time_ns": oldest_start_ns,
                        "sample_rate_hz": 16000,
                        "channels": 1,
                        "encoding": "pcm16le",
                        "samples_b64": encode_pcm16le_b64(
                            np.random.default_rng(2).normal(0.0, 0.42, size=(1, 1024)).astype(np.float32)
                        ),
                        "sequence": 101,
                        "source_type": "raw_sensor",
                        "toa_ns": oldest_toa_ns,
                    }
                },
            ],
        }

        response = client.post("/api/v1/ingest/store-forward", json=buffered_payload)
        assert response.status_code == 200
        body = response.json()
        assert body["accepted"] is True
        assert body["total_frames"] == 3
        assert body["accepted_frames"] == 3
        assert body["duplicate_frames"] == 1
        assert body["rejected_frames"] == 0
        assert body["results"][0]["start_time_ns"] == oldest_start_ns
        assert sum(1 for row in body["results"] if row["duplicate"]) == 1

        nodes_response = client.get("/api/v1/nodes", params={"limit": 10})
        assert nodes_response.status_code == 200
        nodes = nodes_response.json()
        node = next(row for row in nodes if row["id"] == "http-node-1")
        assert node["last_seen_ns"] >= latest_live_start_ns


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


def test_environment_ingest_from_node_metadata(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=0)
    monkeypatch.setenv("MINIMAPPR_DEFAULT_TEMPERATURE_C", "7.5")
    monkeypatch.setenv("MINIMAPPR_DEFAULT_HUMIDITY", "0.41")

    with TestClient(app) as client:
        _ingest_single_frame(
            client,
            start_time_ns=time.time_ns(),
            metadata={"temperature_c": 29.25, "temperature_source": "board_sensor"},
        )

        rows_response = client.get("/api/v1/environment", params={"limit": 10})
        assert rows_response.status_code == 200
        rows = rows_response.json()
        assert rows
        assert rows[0]["node_id"] == "http-node-1"
        assert rows[0]["temperature_c"] == pytest.approx(29.25, abs=1e-3)

        current_response = client.get("/api/v1/environment/current")
        assert current_response.status_code == 200
        current = current_response.json()
        assert current["temperature_c"] == pytest.approx(29.25, abs=1e-3)
        assert current["metadata"]["source"] == "live"


def test_environment_current_falls_back_when_no_live_temperature(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=0)
    monkeypatch.setenv("MINIMAPPR_DEFAULT_TEMPERATURE_C", "12.0")
    monkeypatch.setenv("MINIMAPPR_DEFAULT_HUMIDITY", "0.35")

    with TestClient(app) as client:
        _ingest_single_frame(
            client,
            start_time_ns=time.time_ns(),
            environment={"pressure_pa": 101420.0, "source": "pressure_only"},
        )

        current_response = client.get("/api/v1/environment/current")
        assert current_response.status_code == 200
        current = current_response.json()
        assert current["temperature_c"] == pytest.approx(12.0, abs=1e-6)
        assert current["humidity_fraction"] == pytest.approx(0.35, abs=1e-6)
        assert current["pressure_pa"] == pytest.approx(101420.0, abs=1e-3)
        assert current["metadata"]["source"] == "static_fallback"


def test_http_ingest_rejects_invalid_base64_payload(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=0)
    payload = {
        "node": {
            "id": "http-node-bad-b64",
            "node_type": "point",
            "position_m": [0.0, 0.0, 0.0],
            "sensor_offsets_m": [[0.0, 0.0, 0.0]],
            "capabilities": ["audio"],
            "metadata": {},
            "properties": {},
        },
        "frame": {
            "start_time_ns": time.time_ns(),
            "sample_rate_hz": 16000,
            "channels": 1,
            "encoding": "pcm16le",
            "samples_b64": "%%%not_base64%%%",
            "sequence": 1,
            "source_type": "raw_sensor",
        },
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/ingest/frame", json=payload)
        assert response.status_code == 400
        assert "Invalid base64 audio payload" in response.text


def test_http_ingest_rejects_odd_byte_length_pcm(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=0)
    payload = {
        "node": {
            "id": "http-node-odd-pcm",
            "node_type": "point",
            "position_m": [0.0, 0.0, 0.0],
            "sensor_offsets_m": [[0.0, 0.0, 0.0]],
            "capabilities": ["audio"],
            "metadata": {},
            "properties": {},
        },
        "frame": {
            "start_time_ns": time.time_ns(),
            "sample_rate_hz": 16000,
            "channels": 1,
            "encoding": "pcm16le",
            "samples_b64": "AA==",  # one byte payload, invalid for pcm16le
            "sequence": 1,
            "source_type": "raw_sensor",
        },
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/ingest/frame", json=payload)
        assert response.status_code == 400
        assert "byte length must be even" in response.text


def test_soundscape_render_endpoint_returns_bformat_and_surround(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=3600)

    with TestClient(app) as client:
        _ingest_single_frame(client, start_time_ns=time.time_ns())
        detections = _wait_for_detections(client)
        assert detections

        bformat_response = client.get("/api/v1/soundscape/render", params={"limit": 16, "render_format": "bformat"})
        assert bformat_response.status_code == 200
        assert bformat_response.headers["content-type"].startswith("audio/wav")
        assert int(bformat_response.headers["x-minimappr-rendered-sources"]) >= 1
        with wave.open(io.BytesIO(bformat_response.content), "rb") as wav:
            assert wav.getnchannels() == 4
            assert wav.getframerate() == 16000

        surround_response = client.get(
            "/api/v1/soundscape/render",
            params={"limit": 16, "render_format": "surround_5_1"},
        )
        assert surround_response.status_code == 200
        with wave.open(io.BytesIO(surround_response.content), "rb") as wav:
            assert wav.getnchannels() == 6
            assert wav.getframerate() == 16000


def test_soundscape_render_returns_404_without_snippets(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=0)

    with TestClient(app) as client:
        _ingest_single_frame(client, start_time_ns=time.time_ns())
        _ = _wait_for_detections(client)
        response = client.get("/api/v1/soundscape/render", params={"limit": 16})
        assert response.status_code == 404
        assert "No compatible detection snippets" in response.text
