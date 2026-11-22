"""PATCH /api/v1/config for the BirdNET+ V3.0 keys (all restart-required).

A bad V3 value must be refused at PATCH time: once persisted it would either
stop the next startup (Settings ValueError) or silently drop BirdNET from
routing (missing model / species list).
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from minimappr.main import app


def _configure_env(monkeypatch, tmp_path: Path, *, with_model_files: bool) -> None:
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_FUSION_WORKER_COUNT", "1")
    model_path = tmp_path / "v3.onnx"
    labels_path = tmp_path / "v3_labels.csv"
    if with_model_files:
        model_path.write_bytes(b"placeholder")
        labels_path.write_text("ix\n", encoding="utf-8")
    monkeypatch.setenv("MINIMAPPR_BIRDNET_V3_MODEL_PATH", str(model_path))
    monkeypatch.setenv("MINIMAPPR_BIRDNET_V3_LABELS_PATH", str(labels_path))


def test_switch_to_v3_without_model_files_is_rejected(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, with_model_files=False)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"birdnet_model_version": "3.0"})
        assert resp.status_code == 422
        assert any("fetch_birdnet_v3" in error for error in resp.json()["detail"])
        assert client.get("/api/v1/config").json()["birdnet_model_version"] == "2.4"


def test_switch_to_v3_with_model_files_is_restart_required(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, with_model_files=True)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"birdnet_model_version": " 3.0 "})
        assert resp.status_code == 200
        body = resp.json()
        assert body["birdnet_model_version"] == "3.0"
        assert "birdnet_model_version" in body["restart_required"]


def test_unknown_model_version_is_rejected(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, with_model_files=True)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"birdnet_model_version": "2.5"})
        assert resp.status_code == 422


def test_missing_species_list_is_rejected(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, with_model_files=True)
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/config", json={"birdnet_v3_species_list_path": str(tmp_path / "absent.txt")}
        )
        assert resp.status_code == 422

        species_list = tmp_path / "region.txt"
        species_list.write_text("Haemorhous mexicanus\n", encoding="utf-8")
        resp = client.patch("/api/v1/config", json={"birdnet_v3_species_list_path": str(species_list)})
        assert resp.status_code == 200
        assert "birdnet_v3_species_list_path" in resp.json()["restart_required"]


def test_window_cross_field_rules_use_merged_values(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, with_model_files=True)
    with TestClient(app) as client:
        # Default max window is 6 s: a 7 s minimum alone violates min <= max.
        assert client.patch("/api/v1/config", json={"birdnet_v3_min_window_seconds": 7.0}).status_code == 422
        assert client.patch("/api/v1/config", json={"birdnet_v3_window_overlap_seconds": 6.0}).status_code == 422
        resp = client.patch(
            "/api/v1/config",
            json={"birdnet_v3_max_window_seconds": 10.0, "birdnet_v3_min_window_seconds": 7.0},
        )
        assert resp.status_code == 200
        assert "birdnet_v3_min_window_seconds" in resp.json()["restart_required"]
