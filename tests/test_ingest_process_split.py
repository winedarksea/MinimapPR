from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi.testclient import TestClient

from minimappr import __main__ as minimappr_cli
from minimappr.config import Settings
from minimappr.core.capture_session import CaptureSessionRecord, CaptureState
from minimappr.main import app
from minimappr.models import NodeSpec, NodeType
from minimappr.storage.db import Storage


def test_settings_prefers_canonical_ingest_port(monkeypatch) -> None:
    monkeypatch.setenv("MINIMAPPR_INGEST_PORT", "19091")
    monkeypatch.setenv("MINIMAPPR_SIDECAR_PORT", "18081")

    settings = Settings.from_env()

    assert settings.ingest_port == 19091
    assert settings.ingest_sidecar_port == 19091
    assert settings.ingest_base_url == "http://127.0.0.1:19091"


def test_settings_falls_back_to_legacy_sidecar_port(monkeypatch) -> None:
    monkeypatch.delenv("MINIMAPPR_INGEST_PORT", raising=False)
    monkeypatch.setenv("MINIMAPPR_SIDECAR_PORT", "18081")

    settings = Settings.from_env()

    assert settings.ingest_port == 18081
    assert settings.ingest_sidecar_port == 18081
    assert settings.ingest_base_url == "http://127.0.0.1:18081"


def test_settings_accepts_explicit_ingest_base_url(monkeypatch) -> None:
    monkeypatch.setenv("MINIMAPPR_INGEST_PORT", "19091")
    monkeypatch.setenv("MINIMAPPR_INGEST_BASE_URL", "http://minimap-ingest.local:19091")

    settings = Settings.from_env()

    assert settings.ingest_base_url == "http://minimap-ingest.local:19091"


def test_split_api_role_hides_ingest_endpoints(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MINIMAPPR_PROCESS_ROLE", "api")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_FEDERATION_ENABLED", "false")

    with TestClient(app) as client:
        response = client.post("/api/v1/ingest/binary", content=b"")
        health = client.get("/health")

    assert response.status_code == 404
    assert health.status_code == 200
    assert health.json()["process_role"] == "api"


def test_split_api_role_lists_nodes_from_storage_without_live_registry(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "api-nodes.db"

    async def prepare_node() -> None:
        storage = Storage(db_path)
        await storage.initialize()
        try:
            await storage.upsert_node(
                NodeSpec(
                    id="api-node-1",
                    node_type=NodeType.SIRITH_TETRA,
                    position_m=(1.0, 2.0, 3.0),
                    sensor_offsets_m=[
                        (0.0, 0.0, 0.0),
                        (0.1, 0.0, 0.0),
                        (0.0, 0.1, 0.0),
                        (0.0, 0.0, 0.1),
                    ],
                    capabilities=["audio"],
                ),
                last_seen_ns=time.time_ns(),
            )
        finally:
            await storage.close()

    asyncio.run(prepare_node())
    monkeypatch.setenv("MINIMAPPR_PROCESS_ROLE", "api")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(db_path))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_FEDERATION_ENABLED", "false")

    with TestClient(app) as client:
        response = client.get("/api/v1/nodes")
        tracks_response = client.get("/api/v1/tracks")
        recent_audio_response = client.get("/api/v1/nodes/api-node-1/audio/recent")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["id"] == "api-node-1"
    assert body[0]["audio_debug"]["status"] == "external_ingest_process"
    assert body[0]["audio_debug"]["sensor_count"] == 4
    assert tracks_response.status_code == 200
    assert recent_audio_response.status_code == 404


def test_python_ingest_role_exposes_ingest_endpoints_and_hides_ui(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MINIMAPPR_PROCESS_ROLE", "ingest")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")
    monkeypatch.setenv("MINIMAPPR_DIRECT_INGEST_ENABLED", "true")
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "ingest.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_CLASSIFIER", "heuristic")
    monkeypatch.setenv("MINIMAPPR_MODEL_CHAIN_CONFIG_PATH", str(tmp_path / "missing-model-chain.json"))
    monkeypatch.setenv("MINIMAPPR_FEDERATION_ENABLED", "false")

    with TestClient(app) as client:
        ui_response = client.get("/api/v1/nodes")
        ingest_response = client.post("/api/v1/ingest/binary", content=b"")

    assert ui_response.status_code == 404
    assert ingest_response.status_code == 400


def test_ingest_role_rejects_rust_backend() -> None:
    try:
        Settings(process_role="ingest", ingest_backend="rust")
    except ValueError as exc:
        assert "MINIMAPPR_PROCESS_ROLE=ingest" in str(exc)
    else:
        raise AssertionError("Expected ingest role with rust backend to be rejected")


def test_capture_start_uses_configured_ingest_base_url(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MINIMAPPR_PROCESS_ROLE", "api")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "rust")
    monkeypatch.setenv("MINIMAPPR_INGEST_STORAGE_MODE", "journal")
    monkeypatch.setenv("MINIMAPPR_DIRECT_INGEST_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_INGEST_PORT", "19091")
    monkeypatch.setenv("MINIMAPPR_INGEST_BASE_URL", "http://127.0.0.1:19091")
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "capture.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    observed: dict[str, str] = {}

    async def fake_start(self, request):
        observed["sidecar_url"] = request.sidecar_url
        return CaptureSessionRecord(
            session_id="session-1",
            state=CaptureState.RECORDING,
            stream_key=request.stream_key,
            range_lease_id="lease-1",
            start_time_ns=123,
            end_time_ns=None,
            first_frame_pts_ns=None,
            work_dir=request.work_dir / "session-1",
            video_path=None,
            iamf_path=None,
            youtube_path=None,
            error=None,
        )

    monkeypatch.setattr("minimappr.core.capture_session.CaptureSessionManager.start", fake_start)

    with TestClient(app) as client:
        response = client.post("/api/v1/capture/start", json={"stream_key": "node-a"})

    assert response.status_code == 200
    assert observed["sidecar_url"] == "http://127.0.0.1:19091"


def test_capture_start_unavailable_for_python_ingest(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MINIMAPPR_PROCESS_ROLE", "api")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "capture-python.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))

    with TestClient(app) as client:
        response = client.post("/api/v1/capture/start", json={"stream_key": "node-a"})

    assert response.status_code == 503
    assert "Rust ingest journal mode" in response.json()["detail"]


def test_plain_minimappr_supervises_python_ingest_when_ingest_port_is_explicit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    class _FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakePopen:
        returncode = None

        def __init__(self, argv, *, env):
            observed["argv"] = argv
            observed["child_env"] = env
            self.terminated = False
            self.killed = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            observed["terminated"] = True
            self.returncode = 0

        def kill(self):
            self.killed = True
            observed["killed"] = True
            self.returncode = -9

        def wait(self, timeout=None):
            observed["wait_timeout"] = timeout
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    def fake_uvicorn_run(app_name: str, *, host: str, port: int, **kwargs):
        observed["app"] = app_name
        observed["host"] = host
        observed["port"] = port
        observed["parent_role"] = __import__("os").environ.get("MINIMAPPR_PROCESS_ROLE")

    monkeypatch.setenv("MINIMAPPR_HOST", "127.0.0.1")
    monkeypatch.setenv("MINIMAPPR_PORT", "8080")
    monkeypatch.setenv("MINIMAPPR_INGEST_PORT", "8081")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "supervised.db"))
    monkeypatch.setattr("minimappr.__main__.subprocess.Popen", _FakePopen)
    monkeypatch.setattr("minimappr.__main__.urllib.request.urlopen", lambda url, timeout: _FakeResponse())
    monkeypatch.setattr("minimappr.__main__.uvicorn.run", fake_uvicorn_run)

    minimappr_cli.main([])

    child_env = observed["child_env"]
    assert isinstance(child_env, dict)
    assert observed["app"] == "minimappr.main:app"
    assert observed["host"] == "127.0.0.1"
    assert observed["port"] == 8080
    assert observed["parent_role"] == "api"
    assert child_env["MINIMAPPR_PROCESS_ROLE"] == "ingest"
    assert child_env["MINIMAPPR_INGEST_PORT"] == "8081"
    assert child_env["MINIMAPPR_DIRECT_INGEST_ENABLED"] == "true"
    assert observed["terminated"] is True


def test_plain_minimappr_does_not_supervise_when_ingest_port_is_not_explicit(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_uvicorn_run(app_name: str, *, host: str, port: int, **kwargs):
        observed["app"] = app_name
        observed["host"] = host
        observed["port"] = port

    monkeypatch.delenv("MINIMAPPR_INGEST_PORT", raising=False)
    monkeypatch.setenv("MINIMAPPR_HOST", "127.0.0.1")
    monkeypatch.setenv("MINIMAPPR_PORT", "9090")
    monkeypatch.setattr("minimappr.__main__.uvicorn.run", fake_uvicorn_run)

    minimappr_cli.main([])

    assert observed == {
        "app": "minimappr.main:app",
        "host": "127.0.0.1",
        "port": 9090,
    }
