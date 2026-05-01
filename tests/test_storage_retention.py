from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from minimappr.cleanup_policy import CleanupDefaults, CleanupPolicy, RetentionActions, RetentionMatchCriteria, RetentionRule
from minimappr.models import DetectionEvent, NodeSpec, NodeType, TrackState
from minimappr.storage.db import Storage


@pytest.mark.asyncio
async def test_storage_recovery_removes_empty_wal_and_stale_shm(tmp_path: Path) -> None:
    db_path = tmp_path / "stale.db"
    db_path.write_bytes(b"SQLite format 3\x00" + bytes(100))
    wal_path = Path(f"{db_path}-wal")
    shm_path = Path(f"{db_path}-shm")
    wal_path.write_bytes(b"")
    shm_path.write_bytes(b"stale-shm")

    storage = Storage(db_path)

    assert await storage._recover_empty_wal_sidecars() is True
    assert wal_path.exists() is False
    assert shm_path.exists() is False


@pytest.mark.asyncio
async def test_storage_recovery_preserves_nonempty_wal(tmp_path: Path) -> None:
    db_path = tmp_path / "active.db"
    db_path.write_bytes(b"SQLite format 3\x00" + bytes(100))
    wal_path = Path(f"{db_path}-wal")
    shm_path = Path(f"{db_path}-shm")
    wal_path.write_bytes(b"pending-wal")
    shm_path.write_bytes(b"active-shm")

    storage = Storage(db_path)

    assert await storage._recover_empty_wal_sidecars() is False
    assert wal_path.read_bytes() == b"pending-wal"
    assert shm_path.read_bytes() == b"active-shm"


@pytest.mark.asyncio
async def test_storage_initialize_retries_schema_after_disk_io_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = Storage(tmp_path / "retry.db")
    call_counts: dict[str, int] = {
        "open": 0,
        "configure": 0,
        "schema": 0,
        "recover": 0,
    }

    async def _open_connection_stub() -> None:
        call_counts["open"] += 1

    async def _configure_connection_stub() -> None:
        call_counts["configure"] += 1

    async def _initialize_schema_and_migrations_stub() -> None:
        call_counts["schema"] += 1
        if call_counts["schema"] == 1:
            raise sqlite3.OperationalError("disk I/O error")

    async def _recover_empty_wal_sidecars_stub() -> bool:
        call_counts["recover"] += 1
        return True

    monkeypatch.setattr(storage, "_open_connection", _open_connection_stub)
    monkeypatch.setattr(storage, "_configure_connection", _configure_connection_stub)
    monkeypatch.setattr(storage, "_initialize_schema_and_migrations", _initialize_schema_and_migrations_stub)
    monkeypatch.setattr(storage, "_recover_empty_wal_sidecars", _recover_empty_wal_sidecars_stub)

    await storage.initialize()

    assert call_counts == {
        "open": 2,
        "configure": 2,
        "schema": 2,
        "recover": 1,
    }


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


@pytest.mark.asyncio
async def test_storage_concurrent_read_write_stress(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "concurrency.db")
    await storage.initialize()

    base_ns = 2_100_000_000_000_000_000
    node = NodeSpec(
        id="node-concurrent",
        node_type=NodeType.POINT,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
    )
    await storage.upsert_node(node, last_seen_ns=base_ns)

    async def writer(worker_id: int) -> None:
        for index in range(24):
            timestamp_ns = base_ns + (worker_id * 1_000_000_000) + (index * 10_000_000)
            await storage.upsert_node(node, last_seen_ns=timestamp_ns)
            await storage.insert_detection(
                detection=DetectionEvent(
                    id=f"det-{worker_id}-{index}",
                    timestamp_ns=timestamp_ns,
                    position_m=(float(worker_id), float(index % 5), 0.0),
                    confidence=0.65,
                    gdop=1.2,
                    label="test_signal",
                    label_category="unknown",
                    label_confidence=0.65,
                    reference_sensor="node-concurrent:ch0",
                ),
                snippet_path=None,
                snippet_expires_ns=None,
                retention_tier="short",
            )
            await asyncio.sleep(0)

    async def reader() -> None:
        for _ in range(64):
            detections = await storage.list_detections(limit=256)
            nodes = await storage.list_nodes(limit=16)
            assert nodes
            assert all("id" in row for row in detections)
            await asyncio.sleep(0)

    await asyncio.gather(
        writer(0),
        writer(1),
        writer(2),
        reader(),
        reader(),
    )

    detections = await storage.list_detections(limit=512)
    assert len(detections) >= 72

    await storage.close()


@pytest.mark.asyncio
async def test_policy_cleanup_respects_label_overrides_and_keeps_metadata(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "policy-cleanup.db")
    await storage.initialize()

    now_ns = 5_000_000_000_000_000_000
    old_ns = now_ns - 200_000_000_000

    await storage.upsert_node(
        NodeSpec(
            id="cleanup-node",
            node_type=NodeType.POINT,
            position_m=(0.0, 0.0, 0.0),
            sensor_offsets_m=[(0.0, 0.0, 0.0)],
            capabilities=["audio"],
        ),
        last_seen_ns=old_ns,
    )

    old_common_snippet = tmp_path / "common.wav"
    old_common_snippet.write_bytes(b"wav")
    old_rare_snippet = tmp_path / "rare.wav"
    old_rare_snippet.write_bytes(b"wav")
    missing_snippet_path = tmp_path / "missing.wav"

    await storage.insert_detection(
        detection=DetectionEvent(
            id="det-common",
            timestamp_ns=old_ns,
            position_m=(0.0, 0.0, 0.0),
            confidence=0.6,
            gdop=1.5,
            label="common_bird",
            label_category="wildlife",
            label_confidence=0.7,
            reference_sensor="cleanup-node:ch0",
        ),
        snippet_path=str(old_common_snippet),
        snippet_expires_ns=None,
        retention_tier="short",
    )
    await storage.insert_detection(
        detection=DetectionEvent(
            id="det-rare",
            timestamp_ns=old_ns,
            position_m=(0.0, 0.0, 0.0),
            confidence=0.7,
            gdop=1.4,
            label="rare_bird_species",
            label_category="wildlife",
            label_confidence=0.9,
            reference_sensor="cleanup-node:ch0",
        ),
        snippet_path=str(old_rare_snippet),
        snippet_expires_ns=None,
        retention_tier="short",
    )
    await storage.insert_detection(
        detection=DetectionEvent(
            id="det-missing",
            timestamp_ns=old_ns,
            position_m=(0.0, 0.0, 0.0),
            confidence=0.4,
            gdop=2.1,
            label="common_bird",
            label_category="wildlife",
            label_confidence=0.4,
            reference_sensor="cleanup-node:ch0",
        ),
        snippet_path=str(missing_snippet_path),
        snippet_expires_ns=None,
        retention_tier="short",
    )
    await storage.insert_detection(
        detection=DetectionEvent(
            id="det-permanent",
            timestamp_ns=old_ns,
            position_m=(0.0, 0.0, 0.0),
            confidence=0.95,
            gdop=1.0,
            label="gunshot",
            label_category="security",
            label_confidence=0.98,
            reference_sensor="cleanup-node:ch0",
        ),
        snippet_path=None,
        snippet_expires_ns=None,
        retention_tier="permanent",
    )

    old_artifact = tmp_path / "common.bin"
    old_artifact.write_bytes(b"artifact")
    rare_artifact = tmp_path / "rare.bin"
    rare_artifact.write_bytes(b"artifact")
    permanent_artifact = tmp_path / "permanent.bin"
    permanent_artifact.write_bytes(b"artifact")

    common_artifact_id = await storage.insert_large_artifact(
        artifact_type="spectrogram",
        path=str(old_artifact),
        retention_tier="experiment",
        source_detection_id="det-common",
        source_track_id=None,
        created_ns=old_ns,
        expires_ns=None,
        metadata={},
    )
    rare_artifact_id = await storage.insert_large_artifact(
        artifact_type="spectrogram",
        path=str(rare_artifact),
        retention_tier="experiment",
        source_detection_id="det-rare",
        source_track_id=None,
        created_ns=old_ns,
        expires_ns=None,
        metadata={},
    )
    permanent_artifact_id = await storage.insert_large_artifact(
        artifact_type="audit",
        path=str(permanent_artifact),
        retention_tier="permanent",
        source_detection_id="det-permanent",
        source_track_id=None,
        created_ns=old_ns,
        expires_ns=None,
        metadata={},
    )

    policy = CleanupPolicy.from_dict(
        {
            "version": 1,
            "defaults": {
                "snippet_max_age_seconds": 60,
                "artifact_max_age_seconds": 60,
            },
            "rules": [
                {
                    "id": "rare-bird-preserve-artifacts",
                    "priority": 10,
                    "match": {
                        "label": "rare_bird_species",
                    },
                    "actions": {
                        "snippet_max_age_seconds": 500,
                        "keep_artifacts": True,
                    },
                }
            ],
        },
        default_snippet_max_age_seconds=60,
        default_artifact_max_age_seconds=60,
    )

    summary = await storage.cleanup_policy_managed_files(now_ns=now_ns, policy=policy)
    assert summary == {
        "snippet_files_deleted": 1,
        "snippet_records_cleared": 2,
        "artifact_files_deleted": 1,
        "artifact_rows_deleted": 1,
    }
    assert old_common_snippet.exists() is False
    assert old_rare_snippet.exists() is True
    assert old_artifact.exists() is False
    assert rare_artifact.exists() is True
    assert permanent_artifact.exists() is True

    detections = {row["id"]: row for row in await storage.list_detections(limit=10)}
    assert detections["det-common"]["snippet_path"] is None
    assert detections["det-missing"]["snippet_path"] is None
    assert detections["det-rare"]["snippet_path"] == str(old_rare_snippet)
    assert detections["det-common"]["label"] == "common_bird"
    artifact_rows = await (
        await storage._require_db().execute("SELECT id, path FROM large_artifacts ORDER BY id")
    ).fetchall()
    artifact_ids = {row["id"] for row in artifact_rows}
    assert artifact_ids == {rare_artifact_id, permanent_artifact_id}
    assert common_artifact_id not in artifact_ids

    await storage.close()


@pytest.mark.asyncio
async def test_policy_cleanup_dry_run_does_not_mutate_storage(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "policy-dry-run.db")
    await storage.initialize()

    now_ns = 6_000_000_000_000_000_000
    old_ns = now_ns - 120_000_000_000
    await storage.upsert_node(
        NodeSpec(
            id="dry-run-node",
            node_type=NodeType.POINT,
            position_m=(0.0, 0.0, 0.0),
            sensor_offsets_m=[(0.0, 0.0, 0.0)],
            capabilities=["audio"],
        ),
        last_seen_ns=old_ns,
    )

    snippet_file = tmp_path / "dry-run.wav"
    snippet_file.write_bytes(b"wav")
    await storage.insert_detection(
        detection=DetectionEvent(
            id="det-dry-run",
            timestamp_ns=old_ns,
            position_m=(0.0, 0.0, 0.0),
            confidence=0.6,
            gdop=1.2,
            label="common_sound",
            label_category="unknown",
            label_confidence=0.6,
            reference_sensor="dry-run-node:ch0",
        ),
        snippet_path=str(snippet_file),
        snippet_expires_ns=None,
        retention_tier="short",
    )

    policy = CleanupPolicy(
        version=1,
        defaults=CleanupDefaults(
            snippet_max_age_seconds=60,
            artifact_max_age_seconds=60,
        ),
    )
    summary = await storage.cleanup_policy_managed_files(now_ns=now_ns, policy=policy, dry_run=True)
    assert summary["snippet_files_deleted"] == 1
    assert summary["snippet_records_cleared"] == 1
    assert snippet_file.exists() is True
    detections = {row["id"]: row for row in await storage.list_detections(limit=10)}
    assert detections["det-dry-run"]["snippet_path"] == str(snippet_file)

    await storage.close()


def test_cleanup_policy_rule_priority_and_legacy_keep_labels_compatibility() -> None:
    policy = CleanupPolicy(
        version=1,
        defaults=CleanupDefaults(
            snippet_max_age_seconds=60,
            artifact_max_age_seconds=60,
        ),
        rules=[
            RetentionRule(
                rule_id="all-birds",
                priority=10,
                match=RetentionMatchCriteria(label="bird"),
                actions=RetentionActions(keep_snippets=True),
            ),
            RetentionRule(
                rule_id="specific-bird-override",
                priority=20,
                match=RetentionMatchCriteria(label="bird"),
                actions=RetentionActions(keep_snippets=False, snippet_max_age_seconds=600),
            ),
        ],
    )
    assert policy.snippet_max_age_seconds_for_label("bird") == 600

    legacy_policy = CleanupPolicy.from_dict(
        {
            "default_snippet_max_age_seconds": 90,
            "default_artifact_max_age_seconds": 45,
            "keep_labels": {
                "gunshot": {
                    "keep_snippets": True,
                    "keep_artifacts": True,
                }
            },
        },
        default_snippet_max_age_seconds=90,
        default_artifact_max_age_seconds=45,
    )
    assert legacy_policy.snippet_max_age_seconds_for_label("gunshot") is None
    assert legacy_policy.artifact_max_age_seconds_for_label("gunshot") is None
    assert legacy_policy.snippet_max_age_seconds_for_label("ambient") == 90
