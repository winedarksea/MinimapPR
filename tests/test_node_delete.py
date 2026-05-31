"""Tests for node deletion: Storage.delete_node cascade + DELETE /api/v1/nodes/{id}."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from minimappr.main import app
from minimappr.models import NodeSpec, NodeType
from minimappr.storage.db import Storage


def _node(node_id: str) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        node_type=NodeType.POINT,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
    )


@pytest.mark.asyncio
async def test_delete_node_removes_node_and_keyed_records(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "node-delete.db")
    await storage.initialize()
    now_ns = time.time_ns()

    await storage.upsert_node(_node("node-del"), last_seen_ns=now_ns)
    await storage.insert_observation(
        node_id="node-del",
        sensor_id="node-del:ch0",
        sensor_type="audio",
        source_type="live",
        toa_ns=now_ns,
        tor_ns=now_ns,
        time_quality="gps_locked",
        sample_rate_hz=16000,
        channel_index=0,
        frame_sequence=1,
    )
    await storage.insert_environment(
        node_id="node-del",
        timestamp_ns=now_ns,
        temperature_c=20.0,
        pressure_pa=101325.0,
        humidity_fraction=0.5,
        wind_speed_mps=None,
        wind_dir_deg=None,
        solar_lux=None,
        metadata={"source": "test"},
    )
    await storage.insert_bit_report(
        report_id="bit-del",
        node_id="node-del",
        report_type="power_on",
        overall_status="pass",
        timestamp_ns=now_ns,
        received_ns=now_ns,
        results_json="[]",
        failure_codes_json="[]",
        firmware_version="dev",
        uptime_seconds=1.0,
        metadata_json="{}",
    )
    await storage.register_ingested_frame(
        node_id="node-del",
        frame_sequence=1,
        start_time_ns=now_ns,
        utc_end_ns=now_ns + 1_000_000,
        start_sample_index=0,
        end_sample_index=16000,
        toa_ns=now_ns,
        tor_ns=now_ns,
        source_type="live",
    )

    deleted = await storage.delete_node("node-del")
    assert deleted is True

    assert await storage.get_node_by_id("node-del") is None
    assert await storage.list_environment(node_id="node-del") == []
    assert await storage.list_bit_reports("node-del") == []
    # ingested_frames has no FK cascade; delete_node clears it explicitly.
    assert (
        await storage.has_ingested_frame(
            node_id="node-del",
            boot_session="",
            frame_sequence=1,
            start_time_ns=now_ns,
            utc_end_ns=now_ns + 1_000_000,
            start_sample_index=0,
            end_sample_index=16000,
            source_type="live",
            time_quality="",
            tor_ns=now_ns,
        )
        is False
    )

    # Deleting an unknown node reports no rows removed.
    assert await storage.delete_node("does-not-exist") is False


def _configure_env(monkeypatch, tmp_path: Path) -> Path:
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(db_path))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")
    return db_path


def _seed_node(db_path: Path, node_id: str, last_seen_ns: int) -> None:
    """Write a node row to the DB file on its own loop, before the app starts.

    Done out-of-band (rather than ``asyncio.run`` against the running app's
    storage) so we never touch the app's aiosqlite connection from a foreign
    event loop, which deadlocks.
    """

    async def _seed() -> None:
        storage = Storage(db_path)
        await storage.initialize()
        await storage.upsert_node(_node(node_id), last_seen_ns=last_seen_ns)
        await storage.close()

    asyncio.run(_seed())


class TestDeleteNodeEndpoint:
    def test_deletes_offline_node(self, monkeypatch, tmp_path):
        db_path = _configure_env(monkeypatch, tmp_path)
        stale_ns = time.time_ns() - 120 * 1_000_000_000  # well past 45s offline
        _seed_node(db_path, "stale-node", stale_ns)
        with TestClient(app) as client:
            resp = client.delete("/api/v1/nodes/stale-node")
            assert resp.status_code == 200
            assert resp.json() == {"ok": True, "node_id": "stale-node"}
            # Gone from the list too.
            listed = {n["id"] for n in client.get("/api/v1/nodes").json()}
            assert "stale-node" not in listed

    def test_rejects_active_node(self, monkeypatch, tmp_path):
        db_path = _configure_env(monkeypatch, tmp_path)
        _seed_node(db_path, "live-node", time.time_ns())
        with TestClient(app) as client:
            resp = client.delete("/api/v1/nodes/live-node")
            assert resp.status_code == 409
            # The node is left intact.
            listed = {n["id"] for n in client.get("/api/v1/nodes").json()}
            assert "live-node" in listed

    def test_unknown_node_returns_404(self, monkeypatch, tmp_path):
        _configure_env(monkeypatch, tmp_path)
        with TestClient(app) as client:
            resp = client.delete("/api/v1/nodes/ghost-node")
            assert resp.status_code == 404
