"""Tests for GET /api/v1/pipeline/nodes and PATCH /api/v1/pipeline/nodes/{node_id}/audio."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from minimappr.main import app


def _configure_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")


# ---------------------------------------------------------------------------
# GET /api/v1/pipeline/nodes
# ---------------------------------------------------------------------------

class TestGetPipelineNodes:
    def test_returns_200_with_python_pipeline(self, monkeypatch, tmp_path):
        _configure_env(monkeypatch, tmp_path)
        with TestClient(app) as client:
            resp = client.get("/api/v1/pipeline/nodes")
            assert resp.status_code == 200
            body = resp.json()
            assert body["active_pipeline"] == "python"
            assert isinstance(body["nodes"], list)

    def test_response_shape(self, monkeypatch, tmp_path):
        _configure_env(monkeypatch, tmp_path)
        with TestClient(app) as client:
            resp = client.get("/api/v1/pipeline/nodes")
            assert resp.status_code == 200
            body = resp.json()
            assert "active_pipeline" in body
            assert "nodes" in body
            # pipeline_seconds_behind_realtime is optional
            assert "pipeline_seconds_behind_realtime" in body

    def test_empty_nodes_before_ingest(self, monkeypatch, tmp_path):
        _configure_env(monkeypatch, tmp_path)
        with TestClient(app) as client:
            resp = client.get("/api/v1/pipeline/nodes")
            assert resp.status_code == 200
            # No frames have been ingested, so no nodes tracked yet.
            assert resp.json()["nodes"] == []


# ---------------------------------------------------------------------------
# PATCH /api/v1/pipeline/nodes/{node_id}/audio
# ---------------------------------------------------------------------------

class TestPatchNodeAudio:
    def test_stores_override(self, monkeypatch, tmp_path):
        _configure_env(monkeypatch, tmp_path)
        with TestClient(app) as client:
            payload = {
                "mic_gains_db": [3.0, 3.0, 3.0, 3.0],
                "hp_hz": 80.0,
                "lp_hz": 8000.0,
                "smoothing": "ema_50ms",
            }
            resp = client.patch("/api/v1/pipeline/nodes/test-node-1/audio", json=payload)
            assert resp.status_code == 200
            body = resp.json()
            assert body["node_id"] == "test-node-1"
            assert body["override"]["mic_gains_db"] == [3.0, 3.0, 3.0, 3.0]
            assert body["override"]["hp_hz"] == 80.0
            assert body["override"]["lp_hz"] == 8000.0
            assert body["override"]["smoothing"] == "ema_50ms"

    def test_stored_override_visible_in_get(self, monkeypatch, tmp_path):
        _configure_env(monkeypatch, tmp_path)
        with TestClient(app) as client:
            # PATCH an override for a node.
            payload = {"hp_hz": 120.0, "lp_hz": 6000.0}
            patch_resp = client.patch("/api/v1/pipeline/nodes/node-xyz/audio", json=payload)
            assert patch_resp.status_code == 200

            # GET should now include this node even without ingest frames.
            get_resp = client.get("/api/v1/pipeline/nodes")
            assert get_resp.status_code == 200
            nodes = get_resp.json()["nodes"]
            node_ids = [n["node_id"] for n in nodes]
            assert "node-xyz" in node_ids

            node = next(n for n in nodes if n["node_id"] == "node-xyz")
            # Mics should reflect the override values.
            assert len(node["mics"]) >= 1
            assert abs(node["mics"][0]["hp_hz"] - 120.0) < 1e-6
            assert abs(node["mics"][0]["lp_hz"] - 6000.0) < 1e-6

    def test_partial_override_keeps_defaults(self, monkeypatch, tmp_path):
        _configure_env(monkeypatch, tmp_path)
        with TestClient(app) as client:
            # Only set HP — LP should fall back to global default.
            resp = client.patch("/api/v1/pipeline/nodes/partial-node/audio", json={"hp_hz": 200.0})
            assert resp.status_code == 200
            assert resp.json()["override"]["hp_hz"] == 200.0

    def test_stage_override_round_trips_canonical_shape(self, monkeypatch, tmp_path):
        _configure_env(monkeypatch, tmp_path)
        with TestClient(app) as client:
            resp = client.patch(
                "/api/v1/pipeline/nodes/stage-node/audio",
                json={"stages": [{"type": "highpass", "cutoff_hz": 120.0}]},
            )
            nodes = client.get("/api/v1/pipeline/nodes").json()["nodes"]

        assert resp.status_code == 200
        assert resp.json()["override"] == {
            "stages": [{"type": "highpass", "cutoff_hz": 120.0, "order": 4}]
        }
        stage_node = next(node for node in nodes if node["node_id"] == "stage-node")
        assert stage_node["audio_override"] == {
            "mic_gains_db": None,
            "hp_hz": None,
            "lp_hz": None,
            "smoothing": None,
            "stages": [{"type": "highpass", "cutoff_hz": 120.0, "order": 4}],
        }

    def test_invalid_stage_rejected(self, monkeypatch, tmp_path):
        _configure_env(monkeypatch, tmp_path)
        with TestClient(app) as client:
            resp = client.patch(
                "/api/v1/pipeline/nodes/bad-stage/audio",
                json={"stages": [{"type": "bandpass", "low_hz": 500.0, "high_hz": 400.0}]},
            )

        assert resp.status_code == 422
        assert "high_hz" in resp.json()["detail"]

    def test_legacy_patch_does_not_replace_existing_stages(self, monkeypatch, tmp_path):
        _configure_env(monkeypatch, tmp_path)
        with TestClient(app) as client:
            first = client.patch(
                "/api/v1/pipeline/nodes/stage-authoritative/audio",
                json={"stages": [{"type": "gain", "db": 6.0}]},
            )
            second = client.patch(
                "/api/v1/pipeline/nodes/stage-authoritative/audio",
                json={"hp_hz": 200.0},
            )

        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json()["override"] == first.json()["override"]

    def test_invalid_gain_db_rejected(self, monkeypatch, tmp_path):
        _configure_env(monkeypatch, tmp_path)
        with TestClient(app) as client:
            resp = client.patch(
                "/api/v1/pipeline/nodes/bad-node/audio",
                json={"mic_gains_db": [999.0]},
            )
            assert resp.status_code == 422

    def test_negative_hp_rejected(self, monkeypatch, tmp_path):
        _configure_env(monkeypatch, tmp_path)
        with TestClient(app) as client:
            resp = client.patch(
                "/api/v1/pipeline/nodes/bad-node/audio",
                json={"hp_hz": -10.0},
            )
            assert resp.status_code == 422

    def test_smoothing_stored(self, monkeypatch, tmp_path):
        _configure_env(monkeypatch, tmp_path)
        with TestClient(app) as client:
            resp = client.patch(
                "/api/v1/pipeline/nodes/sm-node/audio",
                json={"smoothing": "ema_200ms"},
            )
            assert resp.status_code == 200
            assert resp.json()["override"]["smoothing"] == "ema_200ms"

    def test_python_pipeline_flagged_in_response(self, monkeypatch, tmp_path):
        _configure_env(monkeypatch, tmp_path)
        with TestClient(app) as client:
            resp = client.patch(
                "/api/v1/pipeline/nodes/flag-node/audio",
                json={"hp_hz": 50.0},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert "rust_sidecar_active" in body
            assert body["rust_sidecar_active"] is False

    def test_stage_override_forwarded_to_rust_sidecar(self, monkeypatch, tmp_path):
        _configure_env(monkeypatch, tmp_path)
        monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "rust")
        captured: dict[str, object] = {}

        monkeypatch.setattr("minimappr.main._ingest_sidecar_is_running", lambda state: True)

        def fake_fetch_json_from_sidecar(base_url: str, endpoint_path: str, payload: dict):
            captured["base_url"] = base_url
            captured["endpoint_path"] = endpoint_path
            captured["payload"] = payload
            return {"applied": True}

        monkeypatch.setattr("minimappr.main._fetch_json_from_sidecar", fake_fetch_json_from_sidecar)

        with TestClient(app) as client:
            resp = client.patch(
                "/api/v1/pipeline/nodes/rust-stage-node/audio",
                json={"stages": [{"type": "gain", "db": 6.0}]},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["rust_sidecar_active"] is True
        assert body["rust_sidecar_forwarded"] is True
        assert captured["endpoint_path"] == "/api/v1/dsp/config"
        assert captured["payload"] == {
            "node_id": "rust-stage-node",
            "stages": [{"type": "gain", "db": 6.0}],
        }
