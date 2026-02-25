from __future__ import annotations

from pathlib import Path

import pytest

from minimappr.models import DetectionEvent, NodeSpec, NodeType, TrackState
from minimappr.storage.db import Storage


@pytest.mark.asyncio
async def test_storage_retention_cleanup_removes_expired_records(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "retention.db")
    await storage.initialize()

    now_ns = 2_000_000_000_000_000_000
    old_ns = now_ns - 10_000_000_000

    await storage.upsert_node(
        NodeSpec(
            id="node-1",
            node_type=NodeType.POINT,
            position_m=(0.0, 0.0, 0.0),
            sensor_offsets_m=[(0.0, 0.0, 0.0)],
            capabilities=["audio"],
        ),
        last_seen_ns=old_ns,
    )

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
    await storage.insert_environment(
        node_id="node-1",
        timestamp_ns=old_ns,
        temperature_c=20.0,
        pressure_pa=101325.0,
        humidity_fraction=0.5,
        wind_speed_mps=None,
        wind_dir_deg=None,
        solar_lux=None,
        metadata={"source": "test"},
    )

    track = TrackState(
        id="trk-old",
        first_seen_ns=old_ns,
        last_seen_ns=old_ns,
        position_m=(0.0, 0.0, 0.0),
        velocity_mps=(0.0, 0.0, 0.0),
        label="bird_like",
        label_category="wildlife",
        confidence=0.8,
        update_count=2,
        status="confirmed",
        tqi=0.6,
    )
    await storage.upsert_track(track)
    await storage.insert_track_update(
        track=track,
        timestamp_ns=old_ns,
        event_id="evt-old",
        update_type="detection",
        detection_id="det-old",
        observation_ids=["obs-old"],
        metadata={"from_test": True},
    )
    dropped_track = TrackState(
        id="trk-dropped-old",
        first_seen_ns=old_ns,
        last_seen_ns=old_ns,
        position_m=(3.0, 2.0, 1.0),
        velocity_mps=(0.0, 0.0, 0.0),
        label="ambient",
        label_category="unknown",
        confidence=0.3,
        update_count=1,
        status="dropped",
        tqi=0.1,
    )
    await storage.upsert_track(dropped_track)
    await storage.insert_alert(
        timestamp_ns=old_ns,
        rule_id="rule-test",
        detection_id="det-old",
        track_id="trk-old",
        destination="log",
        priority="normal",
        status="sent",
        payload={"message": "old"},
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
    protected_artifact = tmp_path / "protected.bin"
    protected_artifact.write_bytes(b"protected")
    await storage.insert_large_artifact(
        artifact_type="audit",
        path=str(protected_artifact),
        retention_tier="permanent",
        source_detection_id=None,
        source_track_id=None,
        created_ns=old_ns,
        expires_ns=None,
        metadata={},
    )

    summary = await storage.cleanup_retention(
        now_ns=now_ns,
        tier_ttls_seconds={"short": 1, "experiment": 1},
        operational_ttls_seconds={
            "track_updates": 1,
            "alerts": 1,
            "environment": 1,
            "dropped_tracks": 1,
        },
    )
    assert summary["detections"] >= 1
    assert summary["pings"] >= 1
    assert summary["large_artifacts"] >= 1
    assert summary["track_updates"] >= 1
    assert summary["alerts"] >= 1
    assert summary["environment"] >= 1
    assert summary["dropped_tracks"] >= 1
    assert snippet_file.exists() is False
    assert artifact.exists() is False
    assert protected_artifact.exists() is True
    assert await storage.list_detections(limit=10) == []
    assert await storage.list_pings(limit=10) == []
    assert await storage.list_alerts(limit=10) == []
    assert await storage.list_environment(limit=10) == []
    tracks = await storage.list_tracks(limit=20)
    assert all(track["id"] != "trk-dropped-old" for track in tracks)

    await storage.close()
