"""Effector API endpoint tests: register -> list -> aim -> snapshot -> artifact.

Follows the TestClient + separate-loop DB seeding pattern from
test_node_delete.py (asyncio.run against the app's storage from a foreign
event loop deadlocks; seed before TestClient starts instead).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from minimappr.core.effectors import registry as registry_module
from minimappr.core.effectors.base import EffectorCapabilities, EffectorCommand, ExecutionResult
from minimappr.main import app
from minimappr.models import TrackState
from minimappr.storage.db import Storage


class _MockDriver:
    def __init__(self) -> None:
        self.executed: list[EffectorCommand] = []
        self.armed = False

    async def connect(self) -> None:
        return None

    async def get_capabilities(self) -> EffectorCapabilities:
        return EffectorCapabilities(movement_strategies=["AbsoluteMove"], selected_movement_strategy="AbsoluteMove")

    async def arm(self, *, zone_id: str | None = None) -> bool:
        self.armed = True
        return True

    async def disarm(self) -> bool:
        self.armed = False
        return True

    async def execute(self, command: EffectorCommand) -> ExecutionResult:
        self.executed.append(command)
        return ExecutionResult(status="COMPLETED")

    async def get_status(self) -> dict:
        return {"state": "idle", "armed": self.armed, "pan_deg": 0.0, "tilt_deg": 0.0}

    async def snapshot(self, *, dest_path: Path) -> Path:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")
        return dest_path


def _configure_env(monkeypatch, tmp_path: Path) -> Path:
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(db_path))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_EFFECTOR_SNAPSHOT_DIR", str(tmp_path / "effector_snapshots"))
    # Disabled here so endpoint-flow tests can issue back-to-back aim calls;
    # the rate-limit interlock itself is covered by test_effector_manager.py.
    monkeypatch.setenv("MINIMAPPR_EFFECTOR_MIN_SLEW_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")
    return db_path


def _seed_track(db_path: Path, track_id: str) -> None:
    async def _seed() -> None:
        storage = Storage(db_path)
        await storage.initialize()
        await storage.upsert_track(
            TrackState(
                id=track_id,
                first_seen_ns=time.time_ns(),
                last_seen_ns=time.time_ns(),
                position_m=(8.0, 6.0, 0.0),
                label="human",
                status="confirmed",
            )
        )
        await storage.close()

    asyncio.run(_seed())


def _register_payload(effector_id: str) -> dict:
    return {
        "id": effector_id,
        "effector_type": "camera_ptz",
        "position_m": [0.0, 0.0, 2.0],
        "orientation": {"yaw_deg": 0.0, "pitch_deg": 0.0},
        "capabilities": ["ptz", "snapshot"],
        "transport": {
            "host": "192.168.1.50",
            "port": 80,
            "username": "admin",
            "password": "super-secret",
        },
        "metadata": {"model": "Reolink RLC-823A"},
    }


@pytest.fixture(autouse=True)
def _mock_driver(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: _MockDriver())


class TestEffectorEndpoints:
    def test_empty_registry_returns_empty_list(self, monkeypatch, tmp_path):
        _configure_env(monkeypatch, tmp_path)
        with TestClient(app) as client:
            resp = client.get("/api/v1/effectors")
            assert resp.status_code == 200
            assert resp.json() == []

    def test_register_list_aim_snapshot_artifact_flow(self, monkeypatch, tmp_path):
        db_path = _configure_env(monkeypatch, tmp_path)
        _seed_track(db_path, "trk-1")

        with TestClient(app) as client:
            register_resp = client.post("/api/v1/effectors", json=_register_payload("cam-1"))
            assert register_resp.status_code == 201
            registered = register_resp.json()
            assert registered["id"] == "cam-1"
            assert registered["transport"] == {}

            list_resp = client.get("/api/v1/effectors")
            assert list_resp.status_code == 200
            effectors = list_resp.json()
            assert len(effectors) == 1
            assert effectors[0]["status"]["state"] == "idle"

            status_resp = client.get("/api/v1/effectors/cam-1/status")
            assert status_resp.status_code == 200
            assert status_resp.json()["capabilities"]["selected_movement_strategy"] == "AbsoluteMove"

            aim_resp = client.post("/api/v1/effectors/cam-1/aim", json={"track_id": "trk-1"})
            assert aim_resp.status_code == 200
            assert aim_resp.json()["status"] == "COMPLETED"

            aim_target_resp = client.post(
                "/api/v1/effectors/cam-1/aim", json={"target": [1.0, 2.0, 0.0]}
            )
            assert aim_target_resp.status_code == 200

            snapshot_resp = client.post(
                "/api/v1/effectors/cam-1/snapshot", json={"track_id": "trk-1"}
            )
            assert snapshot_resp.status_code == 200
            body = snapshot_resp.json()
            assert body["status"] == "COMPLETED"
            artifact_id = body["artifact_ids"][0]

            artifact_resp = client.get(f"/api/v1/effector-artifacts/{artifact_id}")
            assert artifact_resp.status_code == 200
            assert artifact_resp.headers["content-type"] == "image/jpeg"

            live_resp = client.get("/api/v1/effectors/cam-1/snapshot.jpg")
            assert live_resp.status_code == 200

            delete_resp = client.delete("/api/v1/effectors/cam-1")
            assert delete_resp.status_code == 200
            assert client.get("/api/v1/effectors").json() == []

    def test_aim_unknown_track_returns_404(self, monkeypatch, tmp_path):
        _configure_env(monkeypatch, tmp_path)
        with TestClient(app) as client:
            client.post("/api/v1/effectors", json=_register_payload("cam-1"))
            resp = client.post("/api/v1/effectors/cam-1/aim", json={"track_id": "does-not-exist"})
            assert resp.status_code == 404

    def test_aim_unknown_effector_returns_404(self, monkeypatch, tmp_path):
        _configure_env(monkeypatch, tmp_path)
        with TestClient(app) as client:
            resp = client.post("/api/v1/effectors/ghost/aim", json={"target": [1.0, 1.0, 0.0]})
            assert resp.status_code == 404

    def test_delete_unknown_effector_returns_404(self, monkeypatch, tmp_path):
        _configure_env(monkeypatch, tmp_path)
        with TestClient(app) as client:
            resp = client.delete("/api/v1/effectors/ghost")
            assert resp.status_code == 404

    def test_credentials_never_appear_in_any_response(self, monkeypatch, tmp_path):
        _configure_env(monkeypatch, tmp_path)
        with TestClient(app) as client:
            register_resp = client.post("/api/v1/effectors", json=_register_payload("cam-1"))
            list_resp = client.get("/api/v1/effectors")
            status_resp = client.get("/api/v1/effectors/cam-1/status")

            for resp in (register_resp, list_resp, status_resp):
                body_text = resp.text
                assert "super-secret" not in body_text
                assert "192.168.1.50" not in body_text

    def test_arm_disarm_and_require_arm_safety_flow(self, monkeypatch, tmp_path):
        _configure_env(monkeypatch, tmp_path)
        with TestClient(app) as client:
            register_resp = client.post("/api/v1/effectors", json=_register_payload("cam-1"))
            assert register_resp.status_code == 201

            default_safety = client.get("/api/v1/effectors/cam-1/safety")
            assert default_safety.status_code == 200
            assert default_safety.json()["require_arm_for_slew"] is False

            patch_resp = client.patch(
                "/api/v1/effectors/cam-1/safety",
                json={
                    "require_arm_for_slew": True,
                    "min_slew_interval_seconds": 0,
                    "no_go_zone_ids": [],
                },
            )
            assert patch_resp.status_code == 200
            assert patch_resp.json()["require_arm_for_slew"] is True

            blocked = client.post("/api/v1/effectors/cam-1/aim", json={"target": [1.0, 2.0, 0.0]})
            assert blocked.status_code == 409
            assert blocked.json()["failure_class"] == "interlock:disarmed"

            arm_resp = client.post("/api/v1/effectors/cam-1/arm", json={})
            assert arm_resp.status_code == 200
            assert arm_resp.json()["status"] == "COMPLETED"
            assert client.get("/api/v1/effectors/cam-1/status").json()["status"]["armed"] is True

            allowed = client.post("/api/v1/effectors/cam-1/aim", json={"target": [1.0, 2.0, 0.0]})
            assert allowed.status_code == 200

            disarm_resp = client.post("/api/v1/effectors/cam-1/disarm")
            assert disarm_resp.status_code == 200
            assert client.get("/api/v1/effectors/cam-1/status").json()["status"]["armed"] is False

    def test_safety_unknown_effector_returns_404(self, monkeypatch, tmp_path):
        _configure_env(monkeypatch, tmp_path)
        with TestClient(app) as client:
            assert client.get("/api/v1/effectors/ghost/safety").status_code == 404
            assert client.patch(
                "/api/v1/effectors/ghost/safety",
                json={
                    "require_arm_for_slew": False,
                    "min_slew_interval_seconds": None,
                    "no_go_zone_ids": [],
                },
            ).status_code == 404
            assert client.post("/api/v1/effectors/ghost/arm", json={}).status_code == 404
            assert client.post("/api/v1/effectors/ghost/disarm").status_code == 404
