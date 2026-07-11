"""Calibration ground-truth API round-trip + bundle export/load tests."""

from __future__ import annotations

import asyncio
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from minimappr.calibration.bundle import (
    build_ground_truth_payload,
    load_bundle,
    write_bundle_zip,
)
from minimappr.calibration.pipeline import write_multichannel_wav
from minimappr.core.capture_session import CaptureSessionRecord, CaptureState
from minimappr.main import app
from minimappr.storage.db import Storage

SESSION_ID = "cal-sess-1"


def _configure_env(monkeypatch, tmp_path: Path) -> Path:
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(db_path))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")
    return db_path


def _write_fake_artifact_dir(artifact_dir: Path) -> Path:
    """Create a minimal calibration artifact dir (manifest + one node WAV)."""
    audio_dir = artifact_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    samples = np.zeros((4, 1600), dtype=np.float32)
    samples[0, :] = 0.25
    write_multichannel_wav(audio_dir / "node-a.wav", samples, 16_000)
    manifest = {
        "schema_version": 1,
        "kind": "minimappr_calibration_bundle",
        "session_id": SESSION_ID,
        "created_utc": "2026-07-10T00:00:00Z",
        "time_window": {"start_ns": 0, "end_ns": 100_000_000},
        "site": {"origin": {"lat": 45.0, "lon": -93.0, "alt_m": 250.0}, "coordinate_mode": "flat"},
        "environment": {
            "temperature_c": 21.0,
            "humidity_fraction": 0.5,
            "speed_of_sound_mps": 344.6,
            "source": "fallback",
        },
        "nodes": [
            {
                "node_id": "node-a",
                "audio_file": "audio/node-a.wav",
                "sample_rate_hz": 16_000,
                "audio_start_time_ns": 0,
                "channel_sensor_ids": [f"node-a:ch{i}" for i in range(4)],
                "position_geo": {"lat": 45.0, "lon": -93.0, "alt_m": 250.0},
                "position_m": [0.0, 0.0, 0.0],
                "sensor_offsets_m": [[0.0, 0.0, 0.0]] * 4,
                "orientation": {},
                "sync_diag": {},
            }
        ],
        "reference_audio": None,
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    (artifact_dir / "detections.json").write_text(
        json.dumps({"schema_version": 1, "detections": []})
    )
    return manifest_path


def _seed_calibration_session(db_path: Path, artifact_dir: Path) -> None:
    manifest_path = _write_fake_artifact_dir(artifact_dir)

    async def _seed() -> None:
        storage = Storage(db_path)
        await storage.initialize()
        record = CaptureSessionRecord(
            session_id=SESSION_ID,
            state=CaptureState.COMPLETED,
            stream_key="calibration",
            range_lease_id=None,
            start_time_ns=time.time_ns() - 60_000_000_000,
            end_time_ns=time.time_ns(),
            first_frame_pts_ns=None,
            work_dir=artifact_dir,
            video_path=None,
            ambix_path=None,
            iamf_path=None,
            youtube_path=None,
            error=None,
            capture_kind="calibration",
            calibration_manifest_path=manifest_path,
        )
        await storage.upsert_capture_session(record)
        await storage.close()

    asyncio.run(_seed())


def _event_body(**overrides) -> dict:
    body = {
        "label": "drone",
        "label_category": "drone",
        "lat": 45.0005,
        "lon": -93.0004,
        "alt_m": 280.0,
        "start_ns": 1_000,
        "end_ns": 5_000,
        "notes": "DJI Mini hover at 30 m AGL",
    }
    body.update(overrides)
    return body


class TestCalibrationGroundTruthApi:
    def test_round_trip_and_bundle(self, monkeypatch, tmp_path):
        db_path = _configure_env(monkeypatch, tmp_path)
        artifact_dir = tmp_path / "artifacts" / f"{SESSION_ID}_calibration"
        _seed_calibration_session(db_path, artifact_dir)

        with TestClient(app) as client:
            # 404 for non-calibration/unknown sessions
            resp = client.post(
                "/api/v1/calibration/nope/ground-truth", json=_event_body()
            )
            assert resp.status_code == 404

            # add
            resp = client.post(
                f"/api/v1/calibration/{SESSION_ID}/ground-truth", json=_event_body()
            )
            assert resp.status_code == 200, resp.text
            event = resp.json()
            assert event["label"] == "drone"
            assert event["geometry_kind"] == "static"
            event_id = event["event_id"]

            # invalid window rejected
            resp = client.post(
                f"/api/v1/calibration/{SESSION_ID}/ground-truth",
                json=_event_body(start_ns=10, end_ns=5),
            )
            assert resp.status_code == 422

            # list
            resp = client.get(f"/api/v1/calibration/{SESSION_ID}/ground-truth")
            assert [e["event_id"] for e in resp.json()] == [event_id]

            # patch
            resp = client.patch(
                f"/api/v1/calibration/ground-truth/{event_id}",
                json={"label": "gunshot", "label_category": "gunshot"},
            )
            assert resp.status_code == 200
            assert resp.json()["label"] == "gunshot"

            # bundle export
            resp = client.get(f"/api/v1/calibration/{SESSION_ID}/bundle")
            assert resp.status_code == 200
            bundle_path = tmp_path / "bundle.zip"
            bundle_path.write_bytes(resp.content)

            # delete
            resp = client.delete(f"/api/v1/calibration/ground-truth/{event_id}")
            assert resp.status_code == 204
            resp = client.delete(f"/api/v1/calibration/ground-truth/{event_id}")
            assert resp.status_code == 404

        bundle = load_bundle(bundle_path)
        assert bundle.manifest["session_id"] == SESSION_ID
        assert len(bundle.events) == 1
        assert bundle.events[0]["label"] == "gunshot"
        assert bundle.events[0]["geometry"]["type"] == "static"
        assert bundle.expectations is None
        channels, sample_rate_hz = bundle.node_audio("node-a")
        assert channels.shape == (4, 1600)
        assert sample_rate_hz == 16_000
        assert np.allclose(channels[0], 0.25, atol=0.01)
        with zipfile.ZipFile(bundle_path) as zf:
            assert "reference_audio/README.txt" in zf.namelist()


def test_write_and_load_bundle_pure(tmp_path):
    artifact_dir = tmp_path / "artifact"
    _write_fake_artifact_dir(artifact_dir)
    rows = [
        {
            "id": "cgt-1",
            "session_id": SESSION_ID,
            "label": "drone",
            "label_category": "drone",
            "geometry_kind": "static",
            "lat": 45.0,
            "lon": -93.0,
            "alt_m": 260.0,
            "start_ns": 0,
            "end_ns": 100,
            "notes": None,
        }
    ]
    payload = build_ground_truth_payload(rows)
    out = write_bundle_zip(
        artifact_dir,
        payload,
        tmp_path / "b.zip",
        expectations={"schema_version": 1, "classification": {"min_label_accuracy": 0.5}},
    )
    bundle = load_bundle(out)
    assert bundle.expectations["classification"]["min_label_accuracy"] == 0.5
    assert bundle.events[0]["event_id"] == "cgt-1"


def test_load_bundle_rejects_unknown_geometry(tmp_path):
    artifact_dir = tmp_path / "artifact"
    _write_fake_artifact_dir(artifact_dir)
    payload = {
        "schema_version": 1,
        "events": [
            {
                "event_id": "e1",
                "label": "drone",
                "geometry": {"type": "trajectory", "waypoints": []},
                "start_ns": 0,
                "end_ns": 1,
            }
        ],
    }
    out = write_bundle_zip(artifact_dir, payload, tmp_path / "b.zip")
    try:
        load_bundle(out)
        raise AssertionError("expected ValueError for trajectory geometry")
    except ValueError as exc:
        assert "geometry" in str(exc)
