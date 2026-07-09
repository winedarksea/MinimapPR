from __future__ import annotations

import time
from pathlib import Path

import pytest

from minimappr.models import NodeOverrides, NodeSpec, NodeType
from minimappr.storage.db import Storage


@pytest.mark.asyncio
async def test_node_override_read_is_effective_and_ingest_preserves_pin(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "nodes.db")
    await storage.initialize()
    spec = NodeSpec(
        id="node-1",
        node_type=NodeType.POINT,
        position_m=(1.0, 2.0, 3.0),
        capabilities=["audio"],
    )
    await storage.upsert_node(spec, last_seen_ns=10)
    await storage.set_node_overrides(
        "node-1",
        NodeOverrides(position_m=(9.0, 8.0, 7.0), updated_ns=time.time_ns()),
    )
    await storage.upsert_node(spec.model_copy(update={"position_m": (4.0, 5.0, 6.0)}), last_seen_ns=20)

    row = await storage.get_node_by_id("node-1")

    assert row is not None
    assert row["position_m"] == [9.0, 8.0, 7.0]
    assert row["reported_position_m"] == [4.0, 5.0, 6.0]
    assert row["overrides"]["position_m"] == [9.0, 8.0, 7.0]


@pytest.mark.asyncio
async def test_legacy_effector_migration_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    storage = Storage(db_path)
    await storage.initialize()
    await storage._require_db().execute(
        """
        CREATE TABLE IF NOT EXISTS effectors (
            id TEXT PRIMARY KEY,
            effector_type TEXT NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            z REAL NOT NULL,
            lat REAL,
            lon REAL,
            alt REAL,
            yaw_deg REAL NOT NULL DEFAULT 0,
            pitch_deg REAL NOT NULL DEFAULT 0,
            capabilities_json TEXT NOT NULL DEFAULT '[]',
            transport_json TEXT NOT NULL DEFAULT '{}',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            properties_json TEXT NOT NULL DEFAULT '{}',
            last_seen_ns INTEGER NOT NULL
        )
        """
    )
    await storage._require_db().execute(
        """
        INSERT INTO effectors (
            id, effector_type, x, y, z, lat, lon, alt, yaw_deg, pitch_deg,
            capabilities_json, transport_json, metadata_json, properties_json, last_seen_ns
        )
        VALUES (
            'cam-legacy', 'camera_ptz', 1.0, 2.0, 3.0, NULL, NULL, NULL, 45.0, -5.0,
            '["snapshot"]', '{"host":"10.0.0.10"}', '{"model":"legacy"}',
            '{"safety":{"require_arm_for_slew":true}}', 123
        )
        """
    )
    await storage._require_db().commit()
    await storage._migrate_effectors_into_nodes()
    await storage._migrate_effectors_into_nodes()

    row = await storage.get_node_by_id("cam-legacy")

    assert row is not None
    assert row["capabilities"] == ["ptz_camera"]
    assert row["transport"] == {"host": "10.0.0.10"}
    assert row["safety"]["require_arm_for_slew"] is True
    assert await storage._table_exists("effectors_backup_v1")
    await storage.close()


@pytest.mark.asyncio
async def test_initialize_does_not_resurrect_legacy_effectors_table(tmp_path: Path) -> None:
    """A second full initialize() must not recreate the legacy effectors tables.

    The schema DDL used to `CREATE TABLE IF NOT EXISTS effectors`, which resurrected
    an empty legacy table on every boot after the migration renamed it to *_backup_v1,
    re-arming the migration each startup. The DDL no longer creates it.
    """
    db_path = tmp_path / "legacy.db"
    storage = Storage(db_path)
    await storage.initialize()
    await storage._require_db().execute(
        """
        CREATE TABLE IF NOT EXISTS effectors (
            id TEXT PRIMARY KEY,
            effector_type TEXT NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            z REAL NOT NULL,
            lat REAL,
            lon REAL,
            alt REAL,
            yaw_deg REAL NOT NULL DEFAULT 0,
            pitch_deg REAL NOT NULL DEFAULT 0,
            capabilities_json TEXT NOT NULL DEFAULT '[]',
            transport_json TEXT NOT NULL DEFAULT '{}',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            properties_json TEXT NOT NULL DEFAULT '{}',
            last_seen_ns INTEGER NOT NULL
        )
        """
    )
    await storage._require_db().execute(
        """
        INSERT INTO effectors (
            id, effector_type, x, y, z, lat, lon, alt, yaw_deg, pitch_deg,
            capabilities_json, transport_json, metadata_json, properties_json, last_seen_ns
        )
        VALUES (
            'cam-legacy', 'camera_ptz', 1.0, 2.0, 3.0, NULL, NULL, NULL, 45.0, -5.0,
            '[]', '{"host":"10.0.0.10"}', '{}', '{}', 123
        )
        """
    )
    await storage._require_db().commit()
    # First migration renames effectors -> effectors_backup_v1.
    await storage._migrate_effectors_into_nodes()
    await storage.close()

    # Simulate a fresh boot against the same file: initialize() must NOT recreate
    # the legacy table, and the migration must find nothing to do.
    storage2 = Storage(db_path)
    await storage2.initialize()
    assert not await storage2._table_exists("effectors")
    assert not await storage2._table_exists("effector_artifacts")
    assert await storage2._table_exists("effectors_backup_v1")
    row = await storage2.get_node_by_id("cam-legacy")
    assert row is not None
    await storage2.close()
