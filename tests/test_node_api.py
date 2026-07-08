from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from minimappr.core.effectors import registry as registry_module
from minimappr.core.effectors.base import EffectorCapabilities, EffectorCommand, ExecutionResult
from minimappr.main import app


class _MockDriver:
    async def connect(self) -> None:
        return None

    async def get_capabilities(self) -> EffectorCapabilities:
        return EffectorCapabilities(movement_strategies=["AbsoluteMove"], selected_movement_strategy="AbsoluteMove")

    async def arm(self, *, zone_id: str | None = None) -> bool:
        return True

    async def disarm(self) -> bool:
        return True

    async def execute(self, command: EffectorCommand) -> ExecutionResult:
        return ExecutionResult(status="COMPLETED")

    async def get_status(self) -> dict:
        return {"state": "idle", "armed": False}

    async def snapshot(self, *, dest_path: Path) -> Path:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")
        return dest_path


def _configure_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_EFFECTOR_SNAPSHOT_DIR", str(tmp_path / "effector_snapshots"))
    monkeypatch.setenv("MINIMAPPR_EFFECTOR_MIN_SLEW_INTERVAL_SECONDS", "0")
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")


@pytest.fixture(autouse=True)
def _mock_driver(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: _MockDriver())


def test_node_registration_rejects_unknown_capability(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/nodes",
            json={
                "id": "node-bad",
                "node_type": "point",
                "position_m": [0.0, 0.0, 0.0],
                "capabilities": ["bogus"],
            },
        )
        assert resp.status_code == 422


def test_node_patch_override_lifecycle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        payload = {
            "id": "node-audio",
            "node_type": "point",
            "position_m": [1.0, 2.0, 3.0],
            "capabilities": ["audio"],
        }
        assert client.post("/api/v1/nodes", json=payload).status_code == 201

        patch_resp = client.patch(
            "/api/v1/nodes/node-audio",
            json={"overrides": {"position_m": [9.0, 8.0, 7.0]}},
        )
        assert patch_resp.status_code == 200
        patched = patch_resp.json()
        assert patched["position_m"] == [9.0, 8.0, 7.0]
        assert patched["reported_position_m"] == [1.0, 2.0, 3.0]

        detail = client.get("/api/v1/nodes/node-audio").json()
        assert detail["overrides"]["position_m"] == [9.0, 8.0, 7.0]

        clear_resp = client.patch(
            "/api/v1/nodes/node-audio",
            json={"clear_overrides": ["position_m"]},
        )
        assert clear_resp.status_code == 200
        assert clear_resp.json()["position_m"] == [1.0, 2.0, 3.0]


def test_node_effector_routes_require_ptz_capability(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        audio_resp = client.post(
            "/api/v1/nodes",
            json={
                "id": "node-audio",
                "node_type": "point",
                "position_m": [0.0, 0.0, 0.0],
                "capabilities": ["audio"],
            },
        )
        assert audio_resp.status_code == 201
        assert client.get("/api/v1/nodes/node-audio/effector/status").status_code == 404

        camera_resp = client.post(
            "/api/v1/nodes",
            json={
                "id": "cam-node",
                "node_type": "point",
                "position_m": [0.0, 0.0, 2.0],
                "capabilities": ["ptz_camera"],
                "transport": {"host": "192.168.1.50"},
            },
        )
        assert camera_resp.status_code == 201
        status_resp = client.get("/api/v1/nodes/cam-node/effector/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"]["state"] == "idle"


def test_legacy_effector_routes_are_removed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        assert client.get("/api/v1/effectors").status_code == 404
        assert client.post("/api/v1/effectors", json={}).status_code in {404, 405}
        assert client.get("/api/v1/effector-artifacts/missing").status_code == 404
