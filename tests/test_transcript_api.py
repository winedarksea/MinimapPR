from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from minimappr.main import app


def _configure_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_FUSION_WORKER_COUNT", "1")


def test_get_transcript_returns_the_full_persisted_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    transcript_text = "This is a deliberately long transcript that must be returned without truncation."

    with TestClient(app) as client:
        asyncio.run(client.app.state.storage.insert_transcript(
            transcript_id="txt-review-1",
            node_id="node-a",
            sensor_id="node-a:ch0",
            start_ns=1_700_000_000_000_000_000,
            end_ns=1_700_000_005_000_000_000,
            text=transcript_text,
            model="moonshine",
            trigger_confidence=0.92,
            audio_path=str(tmp_path / "speech.wav"),
            detection_id="det-origin-1",
            created_ns=1_700_000_006_000_000_000,
        ))

        response = client.get("/api/v1/transcripts/txt-review-1")

    assert response.status_code == 200
    assert response.json() == {
        "id": "txt-review-1",
        "node_id": "node-a",
        "sensor_id": "node-a:ch0",
        "start_ns": 1_700_000_000_000_000_000,
        "end_ns": 1_700_000_005_000_000_000,
        "text": transcript_text,
        "model": "moonshine",
        "trigger_confidence": 0.92,
        "audio_path": str(tmp_path / "speech.wav"),
        "detection_id": "det-origin-1",
        "created_ns": 1_700_000_006_000_000_000,
    }


def test_get_transcript_returns_not_found_for_unknown_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)

    with TestClient(app) as client:
        response = client.get("/api/v1/transcripts/missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "Transcript not found"
