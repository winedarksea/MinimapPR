from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from minimappr import __main__ as minimappr_cli
from minimappr.config import Settings
from minimappr.core.ingest import _buffer_timestamps_for_frame
from minimappr.core.capture_session import CaptureSessionRecord, CaptureState
from minimappr.main import app
from minimappr.models import NodeSpec, NodeType, TimeQuality
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


def test_gps_buffer_timestamps_ignore_receipt_jitter() -> None:
    frame_start_ns = 1_000_000_000
    frame_end_ns = 1_080_000_000
    delayed_receipt_ns = frame_start_ns + 150_000_000

    start_ns, end_ns, used_receipt_time = _buffer_timestamps_for_frame(
        frame_start_time_ns=frame_start_ns,
        frame_end_time_ns=frame_end_ns,
        sample_count=1280,
        sample_rate_hz=16_000,
        time_quality=TimeQuality.GPS_LOCKED,
        server_received_ns=delayed_receipt_ns,
        allow_receipt_time_fallback=True,
    )

    assert start_ns == frame_start_ns
    assert end_ns == frame_end_ns
    assert used_receipt_time is False


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
            await storage.upsert_node_audio_summary(
                node_id="api-node-1",
                summary={
                    "sensor_count": 4,
                    "active_sensor_count": 4,
                    "sample_rate_hz": 16000,
                    "last_sample_time_ns": time.time_ns(),
                    "age_seconds": 0.0,
                    "rms": 0.031,
                    "recent_coverage_ratio": 1.0,
                    "recent_missing_ratio": 0.0,
                    "recent_max_gap_seconds": 0.0,
                    "max_buffer_samples": 160000,
                    "max_buffer_seconds": 10.0,
                    "status": "live_ingest_process",
                },
                updated_ns=time.time_ns(),
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
    assert body[0]["audio_debug"]["status"] == "recent"
    assert body[0]["audio_debug"]["sensor_count"] == 4
    assert body[0]["audio_debug"]["active_sensor_count"] == 4
    assert body[0]["audio_debug"]["sample_rate_hz"] == 16000
    assert body[0]["audio_debug"]["rms"] == 0.031
    assert body[0]["audio_debug"]["recent_coverage_ratio"] == 1.0
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
    monkeypatch.setenv("MINIMAPPR_PROCESS_ROLE", "combined")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "rust")
    monkeypatch.setenv("MINIMAPPR_INGEST_STORAGE_MODE", "journal")
    monkeypatch.setenv("MINIMAPPR_SIDECAR_MEMORY_ONLY_LIVE_PATH", "false")
    monkeypatch.setenv("MINIMAPPR_DIRECT_INGEST_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_INGEST_PORT", "19091")
    monkeypatch.setenv("MINIMAPPR_INGEST_BASE_URL", "http://127.0.0.1:19091")
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "capture.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    observed: dict[str, object] = {}

    async def fake_start(self, request):
        observed["sidecar_url"] = request.sidecar_url
        observed["has_buffer"] = request.multi_sensor_buffer is not None
        observed["channel_sensor_ids"] = request.channel_sensor_ids
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
            ambix_path=None,
            iamf_path=None,
            youtube_path=None,
            error=None,
        )

    monkeypatch.setattr("minimappr.core.capture_session.CaptureSessionManager.start", fake_start)

    with TestClient(app) as client:
        response = client.post("/api/v1/capture/start", json={"stream_key": "node-a"})

    assert response.status_code == 200
    assert observed["sidecar_url"] is None
    assert observed["has_buffer"] is True
    assert observed["channel_sensor_ids"] == [
        "node-a:ch0",
        "node-a:ch1",
        "node-a:ch2",
        "node-a:ch3",
    ]


def test_capture_start_unavailable_for_python_ingest(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MINIMAPPR_PROCESS_ROLE", "api")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "capture-python.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))

    with TestClient(app) as client:
        response = client.post("/api/v1/capture/start", json={"stream_key": "node-a"})

    assert response.status_code == 503
    assert "combined process role" in response.json()["detail"]


def test_capture_start_proxies_to_python_ingest_worker_in_split_mode(
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

        def read(self):
            return json.dumps(
                {
                    "session_id": "session-python-split",
                    "state": "recording",
                    "stream_key": "node-a",
                    "range_lease_id": None,
                    "start_time_ns": 123,
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        observed["url"] = request.full_url
        observed["method"] = request.get_method()
        observed["timeout"] = timeout
        observed["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setenv("MINIMAPPR_PROCESS_ROLE", "api")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")
    monkeypatch.setenv("MINIMAPPR_INGEST_PORT", "19091")
    monkeypatch.setenv("MINIMAPPR_PORT", "18080")
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "capture-python-proxy.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr("minimappr.main.urllib.request.urlopen", fake_urlopen)

    with TestClient(app) as client:
        response = client.post("/api/v1/capture/start", json={"stream_key": "node-a"})

    assert response.status_code == 200
    assert response.json()["session_id"] == "session-python-split"
    assert observed["url"] == "http://127.0.0.1:19091/api/v1/capture/start"
    assert observed["method"] == "POST"
    assert observed["body"]["stream_key"] == "node-a"


def test_capture_start_unavailable_for_rust_api_role_without_live_buffer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MINIMAPPR_PROCESS_ROLE", "api")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "rust")
    monkeypatch.setenv("MINIMAPPR_INGEST_STORAGE_MODE", "journal")
    monkeypatch.setenv("MINIMAPPR_SIDECAR_MEMORY_ONLY_LIVE_PATH", "true")
    monkeypatch.setenv("MINIMAPPR_DIRECT_INGEST_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "capture-memory-only.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))

    with TestClient(app) as client:
        response = client.post("/api/v1/capture/start", json={"stream_key": "node-a"})

    assert response.status_code == 503
    assert "in-memory live buffer" in response.json()["detail"]


def test_recordings_start_unavailable_for_rust_api_role_without_live_buffer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MINIMAPPR_PROCESS_ROLE", "api")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "rust")
    monkeypatch.setenv("MINIMAPPR_INGEST_STORAGE_MODE", "journal")
    monkeypatch.setenv("MINIMAPPR_SIDECAR_MEMORY_ONLY_LIVE_PATH", "true")
    monkeypatch.setenv("MINIMAPPR_DIRECT_INGEST_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "recordings-rust-api.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/recordings",
            json={"listener_node_id": "node-a", "include_iamf": True},
        )

    assert response.status_code == 503
    assert "in-memory live buffer" in response.json()["detail"]


def test_recordings_start_uses_live_buffer_for_combined_rust_ingest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MINIMAPPR_PROCESS_ROLE", "combined")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "rust")
    monkeypatch.setenv("MINIMAPPR_DIRECT_INGEST_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "recordings-rust-combined.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    observed: dict[str, object] = {}

    async def fake_start(self, request):
        observed["sidecar_url"] = request.sidecar_url
        observed["has_buffer"] = request.multi_sensor_buffer is not None
        observed["channel_sensor_ids"] = request.channel_sensor_ids
        observed["include_iamf"] = request.include_iamf
        observed["record_video"] = request.record_video
        return CaptureSessionRecord(
            session_id="recording-rust-live",
            state=CaptureState.RECORDING,
            stream_key=request.stream_key,
            range_lease_id=None,
            start_time_ns=123,
            end_time_ns=None,
            first_frame_pts_ns=None,
            work_dir=request.work_dir / "recording-rust-live",
            video_path=None,
            ambix_path=None,
            iamf_path=None,
            youtube_path=None,
            error=None,
            include_iamf=request.include_iamf,
            include_video=request.record_video,
        )

    monkeypatch.setattr("minimappr.core.capture_session.CaptureSessionManager.start", fake_start)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/recordings",
            json={
                "listener_node_id": "node-a",
                "include_iamf": True,
                "include_video": False,
            },
        )

    assert response.status_code == 200
    assert observed["sidecar_url"] is None
    assert observed["has_buffer"] is True
    assert observed["channel_sensor_ids"] == [
        "node-a:ch0",
        "node-a:ch1",
        "node-a:ch2",
        "node-a:ch3",
    ]
    assert observed["include_iamf"] is True
    assert observed["record_video"] is False


def test_capture_start_uses_live_buffer_for_python_ingest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MINIMAPPR_PROCESS_ROLE", "combined")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")
    monkeypatch.setenv("MINIMAPPR_DIRECT_INGEST_ENABLED", "true")
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "capture-python-live.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_CLASSIFIER", "heuristic")
    monkeypatch.setenv("MINIMAPPR_MODEL_CHAIN_CONFIG_PATH", str(tmp_path / "missing-model-chain.json"))
    monkeypatch.setenv("MINIMAPPR_FEDERATION_ENABLED", "false")
    observed: dict[str, object] = {}

    async def fake_start(self, request):
        observed["sidecar_url"] = request.sidecar_url
        observed["channel_sensor_ids"] = request.channel_sensor_ids
        observed["has_buffer"] = request.multi_sensor_buffer is not None
        return CaptureSessionRecord(
            session_id="session-python-live",
            state=CaptureState.RECORDING,
            stream_key=request.stream_key,
            range_lease_id=None,
            start_time_ns=123,
            end_time_ns=None,
            first_frame_pts_ns=None,
            work_dir=request.work_dir / "session-python-live",
            video_path=None,
            ambix_path=None,
            iamf_path=None,
            youtube_path=None,
            error=None,
        )

    monkeypatch.setattr("minimappr.core.capture_session.CaptureSessionManager.start", fake_start)

    with TestClient(app) as client:
        response = client.post("/api/v1/capture/start", json={"stream_key": "node-a"})

    assert response.status_code == 200
    assert observed["sidecar_url"] is None
    assert observed["has_buffer"] is True
    assert observed["channel_sensor_ids"] == [
        "node-a:ch0",
        "node-a:ch1",
        "node-a:ch2",
        "node-a:ch3",
    ]


def test_recording_stop_is_idempotent_after_worker_completed_session(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MINIMAPPR_PROCESS_ROLE", "combined")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")
    monkeypatch.setenv("MINIMAPPR_DIRECT_INGEST_ENABLED", "true")
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "recording-stop-completed.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_CLASSIFIER", "heuristic")
    monkeypatch.setenv("MINIMAPPR_MODEL_CHAIN_CONFIG_PATH", str(tmp_path / "missing-model-chain.json"))
    monkeypatch.setenv("MINIMAPPR_FEDERATION_ENABLED", "false")

    completed_record = CaptureSessionRecord(
        session_id="session-completed",
        state=CaptureState.COMPLETED,
        stream_key="node-a",
        range_lease_id=None,
        start_time_ns=123,
        end_time_ns=456,
        first_frame_pts_ns=123,
        work_dir=tmp_path / "session-completed",
        video_path=None,
        ambix_path=None,
        iamf_path=None,
        youtube_path=None,
        error=None,
    )

    async def fake_stop(self, session_id, sidecar_url=""):
        del self, session_id, sidecar_url
        raise ValueError("session session-completed is not recording (state=CaptureState.COMPLETED)")

    def fake_get(self, session_id):
        del self
        return completed_record if session_id == "session-completed" else None

    monkeypatch.setattr("minimappr.core.capture_session.CaptureSessionManager.stop", fake_stop)
    monkeypatch.setattr("minimappr.core.capture_session.CaptureSessionManager.get", fake_get)

    with TestClient(app) as client:
        response = client.patch("/api/v1/recordings/session-completed/stop")

    assert response.status_code == 200
    assert response.json()["status"] == "completed"


def test_recording_delete_removes_capture_session_row_and_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "recording-delete.db"
    artifact_dir = tmp_path / "artifacts"
    work_dir = tmp_path / "captures" / "session-delete"
    artifact_dir.mkdir()
    work_dir.mkdir(parents=True)
    ambix_path = artifact_dir / "session-delete_ambix.wav"
    iamf_path = artifact_dir / "session-delete_audio.iamf"
    visual_path = artifact_dir / "session-delete_visual.mp4"
    ambix_path.write_bytes(b"wav")
    iamf_path.write_bytes(b"iamf")
    visual_path.write_bytes(b"mp4")

    async def prepare_recording() -> None:
        storage = Storage(db_path)
        await storage.initialize()
        try:
            record = CaptureSessionRecord(
                session_id="session-delete",
                state=CaptureState.COMPLETED,
                stream_key="node-a",
                range_lease_id=None,
                start_time_ns=123,
                end_time_ns=456,
                first_frame_pts_ns=123,
                work_dir=work_dir,
                video_path=None,
                ambix_path=ambix_path,
                iamf_path=iamf_path,
                visual_path=visual_path,
                youtube_path=None,
                error=None,
            )
            await storage.upsert_capture_session(record)
            await storage.insert_large_artifact_for_session(
                session_id=record.session_id,
                artifact_type="iamf_video",
                ambix_path=str(ambix_path),
                iamf_path=str(iamf_path),
                visual_path=str(visual_path),
                youtube_path=None,
                created_ns=time.time_ns(),
            )
        finally:
            await storage.close()

    async def fetch_remaining_counts() -> tuple[int, int]:
        storage = Storage(db_path)
        await storage.initialize()
        try:
            db = storage._require_db()
            capture_count = (await (await db.execute("SELECT COUNT(*) FROM capture_sessions")).fetchone())[0]
            artifact_count = (await (await db.execute("SELECT COUNT(*) FROM large_artifacts")).fetchone())[0]
            return int(capture_count), int(artifact_count)
        finally:
            await storage.close()

    asyncio.run(prepare_recording())

    monkeypatch.setenv("MINIMAPPR_PROCESS_ROLE", "combined")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")
    monkeypatch.setenv("MINIMAPPR_DIRECT_INGEST_ENABLED", "true")
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(db_path))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("MINIMAPPR_CLASSIFIER", "heuristic")
    monkeypatch.setenv("MINIMAPPR_MODEL_CHAIN_CONFIG_PATH", str(tmp_path / "missing-model-chain.json"))
    monkeypatch.setenv("MINIMAPPR_FEDERATION_ENABLED", "false")

    with TestClient(app) as client:
        response = client.delete("/api/v1/recordings/session-delete")

    assert response.status_code == 204
    assert ambix_path.exists() is False
    assert iamf_path.exists() is False
    assert visual_path.exists() is False
    assert work_dir.exists() is False
    assert asyncio.run(fetch_remaining_counts()) == (0, 0)


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


def test_split_api_role_clusters_endpoint_is_available(monkeypatch, tmp_path: Path) -> None:
    # Regression: the API-only lifespan must bind cluster_registry; otherwise the
    # cluster CRUD handlers raise KeyError on app.state and return HTTP 500.
    monkeypatch.setenv("MINIMAPPR_PROCESS_ROLE", "api")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "api-clusters.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_FEDERATION_ENABLED", "false")

    with TestClient(app) as client:
        response = client.get("/api/v1/clusters")

    assert response.status_code == 200
    assert response.json() == []


def test_split_api_role_environment_current_reflects_stored_reading(
    monkeypatch, tmp_path: Path
) -> None:
    # Regression: in split mode environment ingest is proxied to the worker, so
    # the API process's provider stays empty and /environment/current reports
    # static_fallback. The live DB poll loop must hydrate it from storage.
    db_path = tmp_path / "api-env.db"

    async def seed_environment() -> None:
        storage = Storage(db_path)
        await storage.initialize()
        try:
            await storage.upsert_node(
                NodeSpec(
                    id="env-node-1",
                    node_type=NodeType.SIRITH_TETRA,
                    position_m=(1.0, 2.0, 3.0),
                    sensor_offsets_m=[
                        (0.0, 0.0, 0.0),
                        (0.1, 0.0, 0.0),
                        (0.0, 0.1, 0.0),
                        (0.0, 0.0, 0.1),
                    ],
                    capabilities=["audio", "temperature", "humidity"],
                ),
                last_seen_ns=time.time_ns(),
            )
            await storage.insert_environment(
                node_id="env-node-1",
                timestamp_ns=time.time_ns(),
                temperature_c=27.9,
                pressure_pa=None,
                humidity_fraction=0.55,
                wind_speed_mps=None,
                wind_dir_deg=None,
                solar_lux=None,
                metadata={"source": "sht45"},
            )
        finally:
            await storage.close()

    asyncio.run(seed_environment())
    monkeypatch.setenv("MINIMAPPR_PROCESS_ROLE", "api")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(db_path))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_FEDERATION_ENABLED", "false")

    with TestClient(app) as client:
        body = {}
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            response = client.get("/api/v1/environment/current")
            assert response.status_code == 200
            body = response.json()
            if body.get("metadata", {}).get("source") != "static_fallback":
                break
            time.sleep(0.25)

    assert body["metadata"]["source"] != "static_fallback"
    assert body["temperature_c"] == pytest.approx(27.9, abs=1e-3)
    # Real temperature feeds the speed-of-sound estimate (~349 m/s at 27.9 C),
    # not the 20 C static-fallback value (~344 m/s).
    assert body["speed_of_sound_mps"] > 347.0


def test_split_api_role_surfaces_runner_stats_in_audio_debug(monkeypatch, tmp_path: Path) -> None:
    # Firmware NodeRunner counters persisted with the audio summary must reach
    # /api/v1/nodes so publish-queue/Wi-Fi health is monitorable in split mode.
    db_path = tmp_path / "api-runner.db"

    async def prepare_node() -> None:
        storage = Storage(db_path)
        await storage.initialize()
        try:
            await storage.upsert_node(
                NodeSpec(
                    id="runner-node",
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
            await storage.upsert_node_audio_summary(
                node_id="runner-node",
                summary={
                    "sensor_count": 4,
                    "active_sensor_count": 4,
                    "sample_rate_hz": 16000,
                    "last_sample_time_ns": time.time_ns(),
                    "age_seconds": 0.0,
                    "rms": 0.02,
                    "recent_coverage_ratio": 0.9,
                    "recent_missing_ratio": 0.1,
                    "recent_max_gap_seconds": 0.3,
                    "max_buffer_samples": 160000,
                    "max_buffer_seconds": 10.0,
                    "status": "live_ingest_process",
                    "runner_stats": {
                        "runner_queue_overflows": 11,
                        "runner_frames_dropped": 4,
                        "runner_publish_wifi_down_failures": 6,
                    },
                },
                updated_ns=time.time_ns(),
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

    assert response.status_code == 200
    node = response.json()[0]
    runner_stats = node["audio_debug"]["runner_stats"]
    assert runner_stats["runner_queue_overflows"] == 11
    assert runner_stats["runner_frames_dropped"] == 4
    assert runner_stats["runner_publish_wifi_down_failures"] == 6
