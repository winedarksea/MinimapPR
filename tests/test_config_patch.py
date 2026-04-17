"""Tests for PATCH /api/v1/config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from minimappr.main import app


def _configure_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_TRIGGER_RMS", "0.015")
    monkeypatch.setenv("MINIMAPPR_FUSION_WORKER_COUNT", "1")


def test_patch_single_float_field(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"trigger_rms": 0.05})
        assert resp.status_code == 200
        body = resp.json()
        assert abs(body["trigger_rms"] - 0.05) < 1e-9

        # Verify GET reflects the change
        get_resp = client.get("/api/v1/config")
        assert get_resp.status_code == 200
        assert abs(get_resp.json()["trigger_rms"] - 0.05) < 1e-9


def test_patch_bool_field(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"preprocess_enabled": False})
        assert resp.status_code == 200
        assert resp.json()["preprocess_enabled"] is False

        resp2 = client.patch("/api/v1/config", json={"preprocess_enabled": True})
        assert resp2.status_code == 200
        assert resp2.json()["preprocess_enabled"] is True


def test_patch_enum_localization_algorithm(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"localization_algorithm": "srp_phat"})
        assert resp.status_code == 200
        assert resp.json()["localization_algorithm"] == "srp_phat"


def test_patch_enum_classifier_backend(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"classifier_backend": "heuristic"})
        assert resp.status_code == 200
        assert resp.json()["classifier_backend"] == "heuristic"


def test_patch_multiple_fields(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/config",
            json={
                "trigger_rms": 0.02,
                "trigger_cooldown_seconds": 1.5,
                "yamnet_min_confidence": 0.4,
                "fusion_worker_count": 2,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert abs(body["trigger_rms"] - 0.02) < 1e-9
        assert abs(body["trigger_cooldown_seconds"] - 1.5) < 1e-9
        assert abs(body["yamnet_min_confidence"] - 0.4) < 1e-9
        assert body["fusion_worker_count"] == 2


def test_patch_unknown_key_rejected(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"unknown_field": 42})
        assert resp.status_code == 422


def test_patch_read_only_key_rejected(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        for read_only_key in ("retention", "site_origin", "federation", "db_path"):
            resp = client.patch("/api/v1/config", json={read_only_key: "anything"})
            assert resp.status_code == 422, f"{read_only_key} should be rejected"


def test_patch_negative_trigger_rms_rejected(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"trigger_rms": -0.01})
        assert resp.status_code == 422


def test_patch_zero_trigger_rms_rejected(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"trigger_rms": 0.0})
        assert resp.status_code == 422


def test_patch_yamnet_confidence_out_of_range(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"yamnet_min_confidence": 1.5})
        assert resp.status_code == 422

        resp2 = client.patch("/api/v1/config", json={"yamnet_min_confidence": -0.1})
        assert resp2.status_code == 422


def test_patch_invalid_localization_algorithm(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"localization_algorithm": "magic_beams"})
        assert resp.status_code == 422


def test_patch_invalid_tracking_filter(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"tracking_filter": "particle"})
        assert resp.status_code == 422


def test_patch_fusion_worker_count_zero_rejected(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"fusion_worker_count": 0})
        assert resp.status_code == 422


def test_patch_beamformer_type_das_alias(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"beamformer_type": "das"})
        assert resp.status_code == 200
        # 'das' normalises to 'delay_and_sum'
        assert resp.json()["beamformer_type"] == "delay_and_sum"


def test_patch_coordinate_mode(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"coordinate_mode": "geodetic"})
        assert resp.status_code == 200
        assert resp.json()["coordinate_mode"] == "geodetic"


def test_patch_invalid_coordinate_mode(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"coordinate_mode": "spherical"})
        assert resp.status_code == 422


def test_patch_broadcasts_config_updated_event(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/live") as ws:
            # Consume the subscription_ack sent on connect
            ws.receive_json()

            client.patch("/api/v1/config", json={"trigger_rms": 0.03})

            # The next event should be config_updated
            event = ws.receive_json()
            assert event["type"] == "config_updated"
            assert abs(event["config"]["trigger_rms"] - 0.03) < 1e-9


def test_patch_type_coercion_int_as_float(monkeypatch, tmp_path: Path) -> None:
    """Sending an integer for a float field should be accepted."""
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"trigger_rms": 1})
        assert resp.status_code == 200
        assert abs(resp.json()["trigger_rms"] - 1.0) < 1e-9


def test_patch_bool_non_bool_rejected(monkeypatch, tmp_path: Path) -> None:
    """Sending a string for a bool field should be rejected."""
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"preprocess_enabled": "yes"})
        assert resp.status_code == 422
