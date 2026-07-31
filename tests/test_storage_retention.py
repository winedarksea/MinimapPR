from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from minimappr.cleanup_policy import CleanupDefaults, CleanupPolicy, RetentionActions, RetentionMatchCriteria, RetentionRule
from minimappr.cleanup_service import CleanupService
from minimappr.config import Settings
from minimappr.models import DetectionEvent, NodeSpec, NodeType, TrackState
from minimappr.storage.db import Storage


@pytest.mark.asyncio
async def test_storage_initialization_removes_legacy_live_ingest_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy-live-ingest.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE ingested_frames (frame_key TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE node_audio_summaries (node_id TEXT PRIMARY KEY)")

    storage = Storage(db_path)
    await storage.initialize()
    try:
        rows = await (
            await storage._require_db().execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('ingested_frames', 'node_audio_summaries')"
            )
        ).fetchall()
        assert rows == []
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_training_dataset_survives_partial_cleanup_and_is_removed_by_full_cleanup(tmp_path: Path) -> None:
    training_dir = tmp_path / "training"
    training_dir.mkdir()
    (training_dir / "det-1.wav").write_bytes(b"wav")
    (training_dir / "det-1.json").write_text("{}")
    settings = Settings(
        db_path=tmp_path / "training-cleanup.db",
        snippet_dir=tmp_path / "snippets",
        large_artifact_dir=tmp_path / "artifacts",
        training_dataset_dir=training_dir,
        retention_policy_path=tmp_path / "missing-policy.json",
    )
    storage = Storage(settings.db_path)
    await storage.initialize()
    service = CleanupService(settings=settings, storage=storage)

    await service.run_partial_cleanup(now_ns=time.time_ns())
    assert (training_dir / "det-1.wav").exists()

    summary = await service.run_full_cleanup()
    assert summary["training_dataset_dir"]["removed"] is True
    assert not training_dir.exists()


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
        received_level_db=55.0,
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
    await storage.insert_bit_report(
        report_id="bit-old",
        node_id="node-1",
        report_type="startup",
        overall_status="ok",
        timestamp_ns=old_ns,
        received_ns=old_ns,
        results_json=json.dumps([{"check": "pps", "status": "ok"}]),
        failure_codes_json=json.dumps([]),
        firmware_version="1.0.0",
        uptime_seconds=3.5,
        metadata_json=json.dumps({"source": "test"}),
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
            "bit_reports": 1,
            "pings": 1,
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
    assert summary["bit_reports"] >= 1
    assert snippet_file.exists() is False
    assert artifact.exists() is False
    assert protected_artifact.exists() is True
    assert await storage.list_detections(limit=10) == []
    assert await storage.list_pings(limit=10) == []
    assert await storage.list_alerts(limit=10) == []
    assert await storage.list_environment(limit=10) == []
    assert await storage.list_bit_reports(limit=10) == []
    tracks = await storage.list_tracks(limit=20)
    assert all(track["id"] != "trk-dropped-old" for track in tracks)

    await storage.close()


@pytest.mark.asyncio
async def test_housekeeping_cycle_runs_sqlite_maintenance_and_retention_indexes(tmp_path: Path) -> None:
    db_path = tmp_path / "maintenance.db"
    storage = Storage(db_path)
    await storage.initialize()

    now_ns = 2_200_000_000_000_000_000
    old_ns = now_ns - 20_000_000_000
    await storage.upsert_node(
        NodeSpec(
            id="node-maint",
            node_type=NodeType.POINT,
            position_m=(0.0, 0.0, 0.0),
            sensor_offsets_m=[(0.0, 0.0, 0.0)],
            capabilities=["audio"],
        ),
        last_seen_ns=old_ns,
    )

    payload_json = json.dumps({"payload": "x" * 4096})
    for index in range(48):
        await storage.insert_bit_report(
            report_id=f"bit-bulk-{index}",
            node_id="node-maint",
            report_type="health",
            overall_status="ok",
            timestamp_ns=old_ns - index,
            received_ns=old_ns - index,
            results_json=payload_json,
            failure_codes_json=json.dumps([]),
            firmware_version="1.0.0",
            uptime_seconds=float(index),
            metadata_json=payload_json,
        )

    settings = Settings(
        db_path=db_path,
        snippet_dir=tmp_path / "snippets",
        large_artifact_dir=tmp_path / "artifacts",
        retention_policy_path=tmp_path / "cleanup-policy.json",
        retention_bit_reports_seconds=1,
        retention_pings_seconds=1,
        retention_track_updates_seconds=1,
        retention_alerts_seconds=1,
        retention_environment_seconds=1,
        retention_dropped_tracks_seconds=1,
    )
    service = CleanupService(settings=settings, storage=storage)

    db = storage._require_db()
    index_rows = await (
        await db.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'index'
            AND name IN (
                'ix_detections_retention_tier_created',
                'ix_large_artifacts_retention_tier_expires',
                'ix_track_updates_retention_tier_created'
            )
            ORDER BY name
            """
        )
    ).fetchall()
    assert [row[0] for row in index_rows] == [
        "ix_detections_retention_tier_created",
        "ix_large_artifacts_retention_tier_expires",
        "ix_track_updates_retention_tier_created",
    ]

    auto_vacuum_row = await (await db.execute("PRAGMA auto_vacuum;")).fetchone()
    assert int(auto_vacuum_row[0]) == 2

    retention_summary = await storage.cleanup_retention(
        now_ns=now_ns,
        tier_ttls_seconds={"short": 1, "experiment": 1},
        operational_ttls_seconds={"bit_reports": 1},
    )
    assert retention_summary["bit_reports"] == 48

    freelist_after_delete_row = await (await db.execute("PRAGMA freelist_count;")).fetchone()
    freelist_after_delete = int(freelist_after_delete_row[0])
    assert freelist_after_delete > 0

    maintenance_summary = await service.run_housekeeping_cycle(
        now_ns=now_ns,
        force_sqlite_maintenance=True,
    )
    sqlite_maintenance = maintenance_summary["sqlite_maintenance"]
    assert maintenance_summary["sqlite_maintenance_due"] is True
    assert sqlite_maintenance["freelist_before"] == freelist_after_delete
    assert sqlite_maintenance["freelist_after"] < freelist_after_delete

    await storage.close()


@pytest.mark.asyncio
async def test_housekeeping_cycle_throttles_sqlite_maintenance(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "maintenance-throttle.db")
    await storage.initialize()

    settings = Settings(
        db_path=tmp_path / "maintenance-throttle.db",
        snippet_dir=tmp_path / "snippets",
        large_artifact_dir=tmp_path / "artifacts",
        retention_policy_path=tmp_path / "cleanup-policy.json",
        sqlite_maintenance_interval_seconds=3600.0,
    )
    service = CleanupService(settings=settings, storage=storage)

    first = await service.run_housekeeping_cycle(now_ns=1_000_000_000)
    assert first["sqlite_maintenance_due"] is True
    assert first["sqlite_maintenance"] is not None

    second = await service.run_housekeeping_cycle(now_ns=2_000_000_000)
    assert second["sqlite_maintenance_due"] is False
    assert second["sqlite_maintenance"] is None

    third = await service.run_housekeeping_cycle(now_ns=3_601_000_000_000)
    assert third["sqlite_maintenance_due"] is True
    assert third["sqlite_maintenance"] is not None

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


def _track_for_status(track_id: str, status: str, last_seen_ns: int) -> TrackState:
    return TrackState(
        id=track_id,
        first_seen_ns=last_seen_ns - 1_000_000_000,
        last_seen_ns=last_seen_ns,
        position_m=(1.0, 2.0, 3.0),
        velocity_mps=(0.0, 0.0, 0.0),
        label="unknown",
        label_category="unknown",
        confidence=0.5,
        update_count=3,
        status=status,
        tqi=0.5,
    )


@pytest.mark.asyncio
async def test_list_tracks_filters_by_status(tmp_path: Path) -> None:
    """Dropped tracks must be excludable in the query, not after it.

    A live deployment served 149 dropped tracks out of 150 because status was only
    ever filtered on the federation merge path — with federation disabled nothing
    filtered at all.
    """
    storage = Storage(tmp_path / "track-status.db")
    await storage.initialize()
    try:
        now_ns = time.time_ns()
        await storage.upsert_track(_track_for_status("trk-confirmed", "confirmed", now_ns))
        await storage.upsert_track(_track_for_status("trk-coasting", "coasting", now_ns - 1))
        await storage.upsert_track(_track_for_status("trk-tentative", "tentative", now_ns - 2))
        await storage.upsert_track(_track_for_status("trk-dropped", "dropped", now_ns - 3))

        assert len(await storage.list_tracks(limit=10)) == 4

        active = await storage.list_tracks(
            limit=10, statuses=["tentative", "confirmed", "coasting"]
        )
        assert {row["id"] for row in active} == {
            "trk-confirmed",
            "trk-coasting",
            "trk-tentative",
        }

        dropped = await storage.list_tracks(limit=10, statuses=["dropped"])
        assert [row["id"] for row in dropped] == ["trk-dropped"]
        # An empty status set selects nothing rather than degrading to "all".
        assert await storage.list_tracks(limit=10, statuses=[]) == []
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_list_tracks_status_filter_does_not_lose_active_to_limit(tmp_path: Path) -> None:
    """Filtering in SQL keeps dropped rows from consuming the LIMIT.

    With post-query filtering, newer dropped tracks fill the limit and an older
    active track vanishes from the result entirely.
    """
    storage = Storage(tmp_path / "track-status-limit.db")
    await storage.initialize()
    try:
        now_ns = time.time_ns()
        for index in range(5):
            await storage.upsert_track(
                _track_for_status(f"trk-dropped-{index}", "dropped", now_ns - index)
            )
        await storage.upsert_track(_track_for_status("trk-active", "confirmed", now_ns - 99))

        rows = await storage.list_tracks(
            limit=3, statuses=["tentative", "confirmed", "coasting"]
        )
        assert [row["id"] for row in rows] == ["trk-active"]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_spl_db_column_migrates_to_received_level_db(tmp_path: Path) -> None:
    """The rename must carry existing rows across, not orphan them.

    ``_ensure_columns`` only ADDs missing columns, so without an explicit rename an
    upgraded deployment would get an empty ``received_level_db`` alongside a
    populated ``spl_db`` and silently lose every recorded level.
    """
    db_path = tmp_path / "spl-rename.db"
    storage = Storage(db_path)
    await storage.initialize()
    await storage.close()

    # Simulate the deployed schema: column still under its legacy name, with data.
    with sqlite3.connect(db_path) as connection:
        connection.execute("ALTER TABLE detections RENAME COLUMN received_level_db TO spl_db")
        connection.execute("ALTER TABLE pings RENAME COLUMN received_level_db TO spl_db")
        connection.execute(
            "INSERT INTO detections (id, event_id, source_type, source_node_id, timestamp_ns,"
            " toa_ns, tor_ns, time_quality, stale_ns, reporting_modality, x, y, z,"
            " confidence, gdop, label, label_confidence, spl_db, reference_sensor,"
            " source_sensors_json, tdoa_json, classifier_scores_json, feature_summary_json,"
            " retention_tier)"
            " VALUES ('det-legacy','det-legacy','raw_sensor','node-a',1,1,1,'gps_locked',2,"
            " 'localized',0,0,0,0.72,1.0,'blue jay',0.45,-43.445955,'node-a:ch0',"
            " '[]','{}','{}','{}','standard')"
        )
        connection.commit()

    storage = Storage(db_path)
    await storage.initialize()
    try:
        for table in ("detections", "pings"):
            columns = await storage._table_columns(table)
            assert "received_level_db" in columns
            assert "spl_db" not in columns

        rows = await (
            await storage._require_db().execute(
                "SELECT received_level_db FROM detections WHERE id = 'det-legacy'"
            )
        ).fetchall()
        assert rows[0]["received_level_db"] == pytest.approx(-43.445955)
    finally:
        await storage.close()

    # Re-running initialize on an already-migrated DB must be a no-op.
    storage = Storage(db_path)
    await storage.initialize()
    await storage.close()


def test_detection_exposes_legacy_spl_db_alias() -> None:
    """Readers pinned to the old key keep working after the rename."""
    detection = DetectionEvent(
        id="det-alias",
        timestamp_ns=1,
        position_m=(0.0, 0.0, 0.0),
        confidence=0.7,
        gdop=1.8,
        label="blue jay",
        label_category="wildlife",
        label_confidence=0.8,
        reference_sensor="node-1:ch0",
        received_level_db=-43.4,
    )
    assert detection.received_level_db == pytest.approx(-43.4)
    assert detection.spl_db == pytest.approx(-43.4)

    dumped = detection.model_dump(mode="json")
    assert dumped["received_level_db"] == pytest.approx(-43.4)
    assert dumped["spl_db"] == pytest.approx(-43.4)
