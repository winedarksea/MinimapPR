from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from minimappr.main import app
from minimappr.storage.db import Storage


def _configure_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(db_path))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_FUSION_WORKER_COUNT", "1")
    return db_path


def _seed_transcript(db_path: Path, *, transcript_id: str, text: str, audio_path: str) -> None:
    # Seed out-of-band on its own loop BEFORE the TestClient starts: driving the
    # app's aiosqlite storage from a fresh asyncio.run inside the client's
    # lifespan deadlocks (the connection lives on the lifespan's loop).
    async def _seed() -> None:
        storage = Storage(db_path)
        await storage.initialize()
        await storage.insert_transcript(
            transcript_id=transcript_id,
            node_id="node-a",
            sensor_id="node-a:ch0",
            start_ns=1_700_000_000_000_000_000,
            end_ns=1_700_000_005_000_000_000,
            text=text,
            model="moonshine",
            trigger_confidence=0.92,
            audio_path=audio_path,
            # No detections row exists in this seeded DB; the column has a
            # foreign key to detections(id), so link nothing.
            detection_id=None,
            created_ns=1_700_000_006_000_000_000,
        )
        await storage.close()

    asyncio.run(_seed())


def test_get_transcript_returns_the_full_persisted_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = _configure_env(monkeypatch, tmp_path)
    transcript_text = "This is a deliberately long transcript that must be returned without truncation."
    audio_path = str(tmp_path / "speech.wav")
    _seed_transcript(db_path, transcript_id="txt-review-1", text=transcript_text, audio_path=audio_path)

    with TestClient(app) as client:
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
        "audio_path": audio_path,
        "detection_id": None,
        "created_ns": 1_700_000_006_000_000_000,
    }


def test_get_transcript_returns_not_found_for_unknown_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)

    with TestClient(app) as client:
        response = client.get("/api/v1/transcripts/missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "Transcript not found"
