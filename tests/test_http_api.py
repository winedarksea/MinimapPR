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

        fusion_response = client.get("/api/v1/fusion/status")
        assert fusion_response.status_code == 200
        fusion = fusion_response.json()
        assert fusion["started"] is True
        assert "metrics" in fusion
        assert isinstance(fusion["metrics"].get("last_localization_algorithm"), str)


def test_spa_refresh_fallback_serves_index_for_frontend_routes(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=0)
    frontend_dir = tmp_path / "frontend"
    frontend_dir.mkdir(parents=True, exist_ok=True)
    (frontend_dir / "index.html").write_text("<html><body>spa-ok</body></html>", encoding="utf-8")
    monkeypatch.setattr("minimappr.main.frontend_dir", frontend_dir)

    with TestClient(app) as client:
        browser_refresh = client.get("/analysis/labels")
        assert browser_refresh.status_code == 200
        assert "spa-ok" in browser_refresh.text

        api_not_found = client.get("/api/v1/does-not-exist")
        assert api_not_found.status_code == 404
        assert api_not_found.json() == {"detail": "Not Found"}


def test_debug_endpoints_expose_runtime_and_event_provenance(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=3600)

    with TestClient(app) as client:
        ingest = _ingest_single_frame(client, start_time_ns=time.time_ns())
        assert ingest["queued_event_id"] is not None

        detections = _wait_for_detections(client)
        assert detections
        detection = detections[0]
        assert detection["reporting_modality"] in {"localized", "omni"}
        assert detection["report_window_start_ns"] is not None
        assert detection["report_window_end_ns"] is not None

        config_response = client.get("/api/v1/debug/config")
        assert config_response.status_code == 200
        config_body = config_response.json()
        assert config_body["runtime"]["classifier"]["requested_backend"] == "yamnet"
        assert "yamnet_input_target_rms" in config_body["thresholds"]
        assert "yamnet_max_input_gain" in config_body["thresholds"]
        assert "python_version" in config_body["runtime"]

        selftest_response = client.get("/api/v1/debug/selftest")
        assert selftest_response.status_code == 200
        selftest_body = selftest_response.json()
        assert selftest_body["summary"]["total"] >= 1
        assert any(check["name"] == "fusion_workers_running" for check in selftest_body["checks"])

        event_response = client.get(f"/api/v1/debug/event/{detection['event_id']}")
        assert event_response.status_code == 200
        event_body = event_response.json()
        assert event_body["event_id"] == detection["event_id"]
        assert event_body["classification"]["label"] == detection["label"]
        assert event_body["reporting"]["modality"] == detection["reporting_modality"]
        assert event_body["reporting"]["report_window_start_ns"] == detection["report_window_start_ns"]
        assert event_body["reporting"]["report_window_end_ns"] == detection["report_window_end_ns"]
        assert event_body["selection"]["selected_sensor_ids"] == detection["source_sensors"]
        assert event_body["provenance"]["source_observation_ids"] == detection["source_observation_ids"]
        assert len(event_body["ingest"]["observations"]) == len(detection["source_observation_ids"])
        tracking_updates = event_body["tracking"]["updates"]
        if tracking_updates:
            assert any(update.get("detection_id") == detection["id"] for update in tracking_updates)


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


def test_detection_audio_endpoint_supports_explicit_download(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=3600)

    with TestClient(app) as client:
        _ingest_single_frame(client, start_time_ns=time.time_ns())
        detections = _wait_for_detections(client)
        assert detections
        detection_id = detections[0]["id"]

        inline_response = client.get(f"/api/v1/detections/{detection_id}/audio")
        assert inline_response.status_code == 200
        assert inline_response.headers["content-type"].startswith("audio/wav")
        inline_disposition = inline_response.headers.get("content-disposition", "")
        assert "inline" in inline_disposition
        assert detection_id in inline_disposition

        download_response = client.get(
            f"/api/v1/detections/{detection_id}/audio",
            params={"download": True},
        )
        assert download_response.status_code == 200
        assert download_response.headers["content-type"].startswith("audio/wav")
        download_disposition = download_response.headers.get("content-disposition", "")
        assert "attachment" in download_disposition
        assert detection_id in download_disposition


def test_track_audio_endpoint_returns_latest_detection_snippet(monkeypatch, tmp_path: Path) -> None:
    db_path = _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=3600)

    with TestClient(app) as client:
        _ingest_single_frame(client, start_time_ns=time.time_ns())
        detections = _wait_for_detections(client)
        assert detections
        detection_id = detections[0]["id"]
        track_id = detections[0].get("track_id")
        if not isinstance(track_id, str) or not track_id:
            track_id = f"trk-test-{detection_id}"
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE detections SET track_id = ? WHERE id = ?",
                    (track_id, detection_id),
                )
                conn.commit()

        response = client.get(f"/api/v1/tracks/{track_id}/audio")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/wav")
        disposition = response.headers.get("content-disposition", "")
        assert "inline" in disposition
        assert track_id in disposition

        download_response = client.get(f"/api/v1/tracks/{track_id}/audio", params={"download": True})
        assert download_response.status_code == 200
        download_disposition = download_response.headers.get("content-disposition", "")
        assert "attachment" in download_disposition
        assert "track_" in download_disposition
        assert "detection_" in download_disposition


def test_config_exposes_updated_detection_threshold_and_cop_defaults(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=0)

    with TestClient(app) as client:
        response = client.get("/api/v1/config")
        assert response.status_code == 200
        body = response.json()
        assert abs(float(body["detection_min_confidence"]) - 0.4) < 1e-9
        assert body["cop"]["detections_max_items"] == 150
        assert body["cop"]["tracks_max_items"] == 150
        assert abs(float(body["cop"]["detections_max_age_seconds"]) - 86_400.0) < 1e-9
        assert abs(float(body["cop"]["tracks_max_age_seconds"]) - 86_400.0) < 1e-9


def test_cop_detections_and_tracks_apply_configured_caps(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=0)
    monkeypatch.setenv("MINIMAPPR_COP_DETECTIONS_MAX_ITEMS", "7")
    monkeypatch.setenv("MINIMAPPR_COP_TRACKS_MAX_ITEMS", "9")
    monkeypatch.setenv("MINIMAPPR_COP_DETECTIONS_MAX_AGE_SECONDS", "123")
    monkeypatch.setenv("MINIMAPPR_COP_TRACKS_MAX_AGE_SECONDS", "456")

    with TestClient(app) as client:
        state = app.state
        captured = {}

        original_list_detections = state.storage.list_detections
        original_list_tracks = state.storage.list_tracks

        async def _capture_list_detections(*args, **kwargs):
            captured["detections_limit"] = kwargs.get("limit")
            captured["detections_since_ns"] = kwargs.get("since_ns")
            return await original_list_detections(*args, **kwargs)

        async def _capture_list_tracks(*args, **kwargs):
            captured["tracks_limit"] = kwargs.get("limit")
            captured["tracks_since_ns"] = kwargs.get("since_ns")
            return await original_list_tracks(*args, **kwargs)

        state.storage.list_detections = _capture_list_detections
        state.storage.list_tracks = _capture_list_tracks
        try:
            before_detections_ns = time.time_ns()
            detections_response = client.get("/api/v1/detections", params={"limit": 1000})
            after_detections_ns = time.time_ns()
            assert detections_response.status_code == 200
            assert captured["detections_limit"] == 7
            expected_detection_cutoff_start = before_detections_ns - int(123 * 1_000_000_000)
            expected_detection_cutoff_end = after_detections_ns - int(123 * 1_000_000_000)
            assert expected_detection_cutoff_start <= int(captured["detections_since_ns"]) <= expected_detection_cutoff_end

            before_tracks_ns = time.time_ns()
            tracks_response = client.get("/api/v1/tracks", params={"limit": 1000})
            after_tracks_ns = time.time_ns()
            assert tracks_response.status_code == 200
            assert captured["tracks_limit"] == 9
            expected_track_cutoff_start = before_tracks_ns - int(456 * 1_000_000_000)
            expected_track_cutoff_end = after_tracks_ns - int(456 * 1_000_000_000)
            assert expected_track_cutoff_start <= int(captured["tracks_since_ns"]) <= expected_track_cutoff_end
        finally:
            state.storage.list_detections = original_list_detections
            state.storage.list_tracks = original_list_tracks


def test_node_recent_audio_endpoint_returns_wav(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=0)

    with TestClient(app) as client:
        _ingest_single_frame(client, start_time_ns=time.time_ns())

        response = client.get("/api/v1/nodes/http-node-1/audio/recent", params={"seconds": 10})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/wav")
        with wave.open(io.BytesIO(response.content), "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getframerate() == 16000
            assert wav.getnframes() > 0


def test_node_recent_audio_endpoint_can_render_single_channel(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=0)

    channels = np.vstack(
        [
            np.linspace(-0.75, 0.75, 1024, dtype=np.float32),
            (0.35 * np.sin(np.linspace(0.0, 12.0 * np.pi, 1024, dtype=np.float32))).astype(np.float32),
        ]
    )
    payload = {
        "node": {
            "id": "http-node-stereo",
            "node_type": "point",
            "position_m": [0.0, 0.0, 0.0],
            "sensor_offsets_m": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            "capabilities": ["audio"],
            "metadata": {},
            "properties": {},
        },
        "frame": {
            "start_time_ns": time.time_ns(),
            "sample_rate_hz": 16000,
            "channels": 2,
            "encoding": "pcm16le",
            "samples_b64": encode_pcm16le_b64(channels),
            "sequence": 1,
            "source_type": "raw_sensor",
        },
    }

    def _read_pcm_mono(wav_bytes: bytes) -> np.ndarray:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getframerate() == 16000
            return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")

    with TestClient(app) as client:
        response = client.post("/api/v1/ingest/frame", json=payload)
        assert response.status_code == 200

        mix_response = client.get("/api/v1/nodes/http-node-stereo/audio/recent", params={"seconds": 10})
        channel0_response = client.get(
            "/api/v1/nodes/http-node-stereo/audio/recent",
            params={"seconds": 10, "channel": 0},
        )
        channel1_response = client.get(
            "/api/v1/nodes/http-node-stereo/audio/recent",
            params={"seconds": 10, "channel": 1},
        )

        assert mix_response.status_code == 200
        assert mix_response.headers["x-minimappr-rendered-channel"] == "mix"
        assert mix_response.headers["x-minimappr-render-mode"] == "auto_mix"
        assert channel0_response.status_code == 200
        assert channel0_response.headers["x-minimappr-rendered-channel"] == "0"
        assert channel0_response.headers["x-minimappr-render-mode"] == "single_channel"
        assert channel1_response.status_code == 200
        assert channel1_response.headers["x-minimappr-rendered-channel"] == "1"
        assert channel1_response.headers["x-minimappr-render-mode"] == "single_channel"

        mix_pcm = _read_pcm_mono(mix_response.content)
        channel0_pcm = _read_pcm_mono(channel0_response.content)
        channel1_pcm = _read_pcm_mono(channel1_response.content)
        assert mix_pcm.size > 0
        assert channel0_pcm.size == mix_pcm.size
        assert channel1_pcm.size == mix_pcm.size
        assert not np.array_equal(channel0_pcm, channel1_pcm)
        assert not np.array_equal(mix_pcm, channel0_pcm)
        assert not np.array_equal(mix_pcm, channel1_pcm)

        missing_channel = client.get(
            "/api/v1/nodes/http-node-stereo/audio/recent",
            params={"seconds": 10, "channel": 2},
        )
        assert missing_channel.status_code == 404


def test_node_recent_audio_endpoint_defaults_array_nodes_to_first_channel(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=0)

    channels = np.vstack(
        [
            np.linspace(-0.9, 0.9, 1024, dtype=np.float32),
            (0.25 * np.sin(np.linspace(0.0, 8.0 * np.pi, 1024, dtype=np.float32))).astype(np.float32),
            (0.2 * np.cos(np.linspace(0.0, 11.0 * np.pi, 1024, dtype=np.float32))).astype(np.float32),
            np.full(1024, 0.05, dtype=np.float32),
        ]
    )
    payload = {
        "node": {
            "id": "http-node-array",
            "node_type": "sirith_tetra",
            "position_m": [0.0, 0.0, 0.0],
            "sensor_offsets_m": [
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [0.0, 0.05, 0.0],
                [0.0, 0.0, 0.05],
            ],
            "capabilities": ["audio", "array_localization"],
            "metadata": {},
            "properties": {},
        },
        "frame": {
            "start_time_ns": time.time_ns(),
            "sample_rate_hz": 16000,
            "channels": 4,
            "encoding": "pcm16le",
            "samples_b64": encode_pcm16le_b64(channels),
            "sequence": 1,
            "source_type": "raw_sensor",
        },
    }

    def _read_pcm_channels(wav_bytes: bytes) -> np.ndarray:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            channel_count = wav.getnchannels()
            assert wav.getframerate() == 16000
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        return pcm.reshape(-1, channel_count).T

    with TestClient(app) as client:
        response = client.post("/api/v1/ingest/frame", json=payload)
        assert response.status_code == 200

        auto_response = client.get("/api/v1/nodes/http-node-array/audio/recent", params={"seconds": 10})
        mix_response = client.get(
            "/api/v1/nodes/http-node-array/audio/recent",
            params={"seconds": 10, "render": "mix"},
        )
        multichannel_response = client.get(
            "/api/v1/nodes/http-node-array/audio/recent",
            params={"seconds": 10, "render": "multichannel"},
        )
        channel0_response = client.get(
            "/api/v1/nodes/http-node-array/audio/recent",
            params={"seconds": 10, "channel": 0},
        )

        assert auto_response.status_code == 200
        assert auto_response.headers["x-minimappr-rendered-channel"] == "0"
        assert auto_response.headers["x-minimappr-render-mode"] == "auto_first_channel"
        assert mix_response.status_code == 200
        assert mix_response.headers["x-minimappr-rendered-channel"] == "mix"
        assert mix_response.headers["x-minimappr-render-mode"] == "mix"
        assert multichannel_response.status_code == 200
        assert multichannel_response.headers["x-minimappr-rendered-channel"] == "all"
        assert multichannel_response.headers["x-minimappr-render-mode"] == "multichannel"
        assert channel0_response.status_code == 200

        auto_pcm = _read_pcm_channels(auto_response.content)
        mix_pcm = _read_pcm_channels(mix_response.content)
        multichannel_pcm = _read_pcm_channels(multichannel_response.content)
        channel0_pcm = _read_pcm_channels(channel0_response.content)

        assert auto_pcm.shape[0] == 1
        assert mix_pcm.shape[0] == 1
        assert multichannel_pcm.shape[0] == 4
        assert np.array_equal(auto_pcm, channel0_pcm)
        assert not np.array_equal(auto_pcm, mix_pcm)


def test_node_recent_audio_endpoint_rejects_unknown_node(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=0)

    with TestClient(app) as client:
        response = client.get("/api/v1/nodes/unknown-node/audio/recent")
        assert response.status_code == 404


def test_node_recent_audio_endpoint_rejects_stale_audio(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=0)

    with TestClient(app) as client:
        stale_start_time_ns = time.time_ns() - 60_000_000_000
        _ingest_single_frame(client, start_time_ns=stale_start_time_ns)

        response = client.get("/api/v1/nodes/http-node-1/audio/recent")
        assert response.status_code == 404
        assert "No recent audio available" in response.text


def test_node_recent_audio_endpoint_validates_seconds(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=0)

    with TestClient(app) as client:
        _ingest_single_frame(client, start_time_ns=time.time_ns())

        response = client.get("/api/v1/nodes/http-node-1/audio/recent", params={"seconds": 0.5})
        assert response.status_code == 422


def test_nodes_include_audio_debug_summary(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=0)

    with TestClient(app) as client:
        _ingest_single_frame(client, start_time_ns=time.time_ns())

        response = client.get("/api/v1/nodes", params={"limit": 10})
        assert response.status_code == 200
        rows = response.json()
        node = next(row for row in rows if row["id"] == "http-node-1")
        audio_debug = node.get("audio_debug")
        assert isinstance(audio_debug, dict)
        assert audio_debug["status"] in {"recent", "stale", "no_audio"}
        assert int(audio_debug["sensor_count"]) >= 1


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


def test_analytics_daily_returns_matrix_shape(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=0)

    with TestClient(app) as client:
        # Ingest a frame so at least one detection exists to bucket.
        _ingest_single_frame(client, start_time_ns=time.time_ns())
        _ = _wait_for_detections(client)

        # Rolling 24h default.
        response = client.get("/api/v1/analytics/daily", params={"tz": "UTC"})
        assert response.status_code == 200
        body = response.json()
        assert body["mode"] == "rolling"
        assert body["tz"] == "UTC"
        assert body["hours"] == 24
        assert len(body["bucket_starts"]) == 24
        assert len(body["bucket_totals"]) == 24
        # counts is a list of rows (one per label); each row must match bucket width.
        for row in body["counts"]:
            assert len(row) == 24
        # Label totals match per-row sums.
        for idx, row in enumerate(body["counts"]):
            assert sum(row) == body["label_totals"][idx]

        # Calendar mode (today in UTC).
        from datetime import datetime, timezone
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        cal_response = client.get(
            "/api/v1/analytics/daily", params={"date": today, "tz": "UTC"}
        )
        assert cal_response.status_code == 200
        cal_body = cal_response.json()
        assert cal_body["mode"] == "calendar"
        assert cal_body["hours"] == 24
        assert len(cal_body["bucket_starts"]) == 24

        # Bad tz → 400.
        bad = client.get("/api/v1/analytics/daily", params={"tz": "Not/A_Zone"})
        assert bad.status_code == 400


def test_analytics_labels_and_detail(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=0)

    with TestClient(app) as client:
        _ingest_single_frame(client, start_time_ns=time.time_ns())
        dets = _wait_for_detections(client)
        assert dets
        label = dets[0].get("label") or "unknown"

        summary = client.get("/api/v1/analytics/labels", params={"window": "24h"})
        assert summary.status_code == 200
        body = summary.json()
        assert body["window"] == "24h"
        assert isinstance(body["labels"], list)
        assert any(row["label"] == label for row in body["labels"])

        detail = client.get(
            f"/api/v1/analytics/labels/{label}",
            params={"window": "24h", "recent_limit": 10},
        )
        assert detail.status_code == 200
        detail_body = detail.json()
        assert detail_body["label"] == label
        assert len(detail_body["hour_histogram"]) == 24
        assert len(detail_body["dow_histogram"]) == 7
        assert len(detail_body["month_histogram"]) == 12
        assert detail_body["total"] >= 1
        assert detail_body["recent"]


def test_analytics_heatmap_returns_binned_points(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, snippet_retention_seconds=0)

    with TestClient(app) as client:
        _ingest_single_frame(
            client,
            start_time_ns=time.time_ns(),
            metadata={"gps": {"lat": 44.987, "lon": -93.258, "alt_m": 0.0}},
        )
        _ = _wait_for_detections(client)

        response = client.get(
            "/api/v1/analytics/heatmap",
            params={"window": "24h", "bin": 0.001, "max_bins": 100},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["window"] == "24h"
        assert abs(body["bin_deg"] - 0.001) < 1e-9
        assert isinstance(body["bins"], list)
        # Each bin has lat/lon/weight; weight >= 1.
        for b in body["bins"]:
            assert "lat" in b and "lon" in b and "weight" in b
            assert b["weight"] >= 1
