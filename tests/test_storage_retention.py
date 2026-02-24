from __future__ import annotations

from pathlib import Path

import pytest

from minimappr.models import DetectionEvent
from minimappr.storage.db import Storage


@pytest.mark.asyncio
async def test_storage_retention_cleanup_removes_expired_records(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "retention.db")
    await storage.initialize()

    now_ns = 2_000_000_000_000_000_000
    old_ns = now_ns - 10_000_000_000

    snippet_file = tmp_path / "old_snippet.wav"
    snippet_file.write_bytes(b"wav")

    detection = DetectionEvent(
        id="det-old",
        timestamp_ns=old_ns,
        position_m=(0.0, 0.0, 0.0),
        confidence=0.7,
        gdop=1.8,
        label="bird_like",
        label_category="wildlife",
        label_confidence=0.8,
        reference_sensor="node-1:ch0",
        snippet_path=str(snippet_file),
    )
    await storage.insert_detection(
        detection=detection,
        snippet_path=str(snippet_file),
        snippet_expires_ns=old_ns + 1_000_000_000,
        retention_tier="short",
    )
    await storage.insert_ping(
        timestamp_ns=old_ns,
        ping_type="acoustic",
        label="bird_like",
        label_id=None,
        spl_db=55.0,
        position_m=(0.0, 0.0, 0.0),
        position_geo=None,
        source_detection_id="det-old",
        source_observation_id=None,
        source_track_id=None,
        retention_tier="short",
        metadata={},
    )

    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"artifact")
    await storage.insert_large_artifact(
        artifact_type="spectrogram",
        path=str(artifact),
        retention_tier="experiment",
        source_detection_id="det-old",
        source_track_id=None,
        created_ns=old_ns,
        expires_ns=old_ns + 1_000_000_000,
        metadata={},
    )

    summary = await storage.cleanup_retention(
        now_ns=now_ns,
        tier_ttls_seconds={"short": 1, "experiment": 1},
    )
    assert summary["detections"] >= 1
    assert summary["pings"] >= 1
    assert summary["large_artifacts"] >= 1
    assert snippet_file.exists() is False
    assert artifact.exists() is False
    assert await storage.list_detections(limit=10) == []
    assert await storage.list_pings(limit=10) == []

    await storage.close()
