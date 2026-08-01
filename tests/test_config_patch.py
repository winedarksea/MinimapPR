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


def test_patch_classifier_routing_toggles(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/config",
            json={"birdnet_enabled": False, "stt_trigger_min_confidence": 0.7},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["birdnet_enabled"] is False
        assert body["stt_trigger_min_confidence"] == 0.7
        # removed key rejected
        resp = client.patch("/api/v1/config", json={"classifier_backend": "heuristic"})
        assert resp.status_code == 422


def test_patch_multiple_fields(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/config",
            json={
                "trigger_rms": 0.02,
                "trigger_cooldown_seconds": 1.5,
                "yamnet_min_confidence": 0.4,
                "detection_min_confidence": 0.55,
                "fusion_worker_count": 2,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert abs(body["trigger_rms"] - 0.02) < 1e-9
        assert abs(body["trigger_cooldown_seconds"] - 1.5) < 1e-9
        assert abs(body["yamnet_min_confidence"] - 0.4) < 1e-9
        assert abs(body["detection_min_confidence"] - 0.55) < 1e-9
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


def test_patch_detection_confidence_out_of_range(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"detection_min_confidence": 1.5})
        assert resp.status_code == 422

        resp2 = client.patch("/api/v1/config", json={"detection_min_confidence": -0.1})
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


def test_patch_beamformer_type_band_split_das(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"beamformer_type": "band_split_das"})
        assert resp.status_code == 200
        assert resp.json()["beamformer_type"] == "band_split_das"


def test_patch_omni_scan_min_rms(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"omni_scan_min_rms": 0.002})
        assert resp.status_code == 200
        assert resp.json()["omni_scan_min_rms"] == 0.002


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


def test_get_config_exposes_canonical_cleanup_keys(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        body = client.get("/api/v1/config").json()

    assert body["localization_max_tau_seconds"] == body["localization_max_tau_s"]
    assert body["drop_on_backpressure"] == body["fusion_drop_on_backpressure"]
    assert body["classifier_stage_timeout_seconds"] == body["classification_stage_timeout_seconds"]


def test_patch_hass_config_redacts_secrets(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/config",
            json={
                "hass_enabled": True,
                "hass_base_url": "http://homeassistant.local:8123",
                "hass_token": "secret-token",
                "hass_mqtt_host": "mqtt.local",
                "hass_mqtt_port": 1884,
                "hass_mqtt_password": "broker-secret",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["hass"]["enabled"] is True
        assert body["hass"]["base_url"] == "http://homeassistant.local:8123"
        assert body["hass"]["token"] == "***"
        assert body["hass"]["mqtt_password"] == "***"
        assert body["hass"]["mqtt_host"] == "mqtt.local"
        assert body["hass"]["mqtt_port"] == 1884

        get_body = client.get("/api/v1/config").json()
        assert get_body["hass"]["token"] == "***"
        assert get_body["hass"]["mqtt_password"] == "***"


def test_patch_hass_secret_placeholder_is_ignored(monkeypatch, tmp_path: Path) -> None:
    """Echoing the "***" redaction back must not destroy the stored secret."""
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        client.patch(
            "/api/v1/config",
            json={
                "hass_enabled": True,
                "hass_mqtt_host": "mqtt.local",
                "hass_token": "real-token",
                "hass_mqtt_password": "real-password",
            },
        ).raise_for_status()

        # A UI that round-trips the whole redacted block.
        resp = client.patch(
            "/api/v1/config",
            json={"hass_token": "***", "hass_mqtt_password": "***", "hass_mqtt_port": 1885},
        )
        assert resp.status_code == 200
        assert resp.json()["hass"]["mqtt_port"] == 1885

    settings = app.state.settings
    assert settings.hass_token == "real-token"
    assert settings.hass_mqtt_password == "real-password"


def test_patch_hass_enabled_requires_a_broker_host(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"hass_enabled": True})
        assert resp.status_code == 422
        assert any("hass_mqtt_host" in item for item in resp.json()["detail"])


@pytest.mark.parametrize(
    "patch_body",
    [
        {"hass_mqtt_port": 0},
        {"hass_mqtt_port": 70000},
        {"hass_mqtt_keepalive_seconds": 0},
        {"hass_publish_interval_seconds": 0.5},
        {"hass_publish_min_interval_seconds": -1.0},
        {"hass_reconcile_interval_seconds": 0.0},
        {"hass_queue_size": 0},
        {"hass_reconnect_backoff_initial_seconds": 0.0},
        {"hass_reconnect_backoff_initial_seconds": 10.0, "hass_reconnect_backoff_max_seconds": 5.0},
        {"hass_detection_off_delay_seconds": 0},
        {"hass_track_slot_count": -1},
        {"hass_track_slot_count": 65},
        {"hass_zone_spl_window_seconds": 0.0},
        {"hass_discovery_prefix": "a/b"},
        {"hass_base_topic": "a+b"},
        {"hass_base_topic": ""},
        {"hass_device_id": "a#"},
    ],
)
def test_patch_hass_numeric_and_topic_validation(monkeypatch, tmp_path: Path, patch_body: dict) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json=patch_body)
        assert resp.status_code == 422, f"{patch_body} should have been rejected"


def test_patch_hass_entity_toggles_round_trip(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/config",
            json={
                "hass_publish_zone_occupancy": False,
                "hass_publish_track_slots": True,
                "hass_track_slot_count": 4,
                "hass_base_topic": "site_a",
            },
        )
        assert resp.status_code == 200
        hass = resp.json()["hass"]
        assert hass["publish_zone_occupancy"] is False
        assert hass["publish_track_slots"] is True
        assert hass["track_slot_count"] == 4
        assert hass["base_topic"] == "site_a"


# ---------------------------------------------------------------------------
# Per-node audio overrides — round-trip via /api/v1/pipeline/nodes/{id}/audio
# ---------------------------------------------------------------------------


def test_node_audio_override_does_not_affect_global_config(monkeypatch, tmp_path: Path) -> None:
    """Patching a node's audio override must not alter the global config keys."""
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        before = client.get("/api/v1/config").json()
        client.patch(
            "/api/v1/pipeline/nodes/node-a/audio",
            json={"hp_hz": 300.0, "lp_hz": 5000.0, "mic_gains_db": [6.0]},
        )
        after = client.get("/api/v1/config").json()

    assert abs(before["audio_highpass_hz"] - after["audio_highpass_hz"]) < 1e-9
    assert abs(before["audio_lowpass_hz"] - after["audio_lowpass_hz"]) < 1e-9


def test_global_audio_config_patch_does_not_clear_node_overrides(monkeypatch, tmp_path: Path) -> None:
    """Patching global audio settings must not wipe existing per-node overrides."""
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        client.patch("/api/v1/pipeline/nodes/node-b/audio", json={"hp_hz": 200.0})
        client.patch("/api/v1/config", json={"audio_highpass_hz": 75.0})
        pipeline = client.get("/api/v1/pipeline/nodes").json()

    node_ids = [n["node_id"] for n in pipeline["nodes"]]
    assert "node-b" in node_ids, "Per-node override must survive a global config PATCH"


def test_node_override_hp_reflected_in_pipeline_view_independent_of_global(
    monkeypatch, tmp_path: Path
) -> None:
    """Per-node HP override is returned in the pipeline view even when global HP differs."""
    _configure_env(monkeypatch, tmp_path)
    monkeypatch.setenv("MINIMAPPR_AUDIO_HIGHPASS_HZ", "50")
    with TestClient(app) as client:
        client.patch("/api/v1/pipeline/nodes/node-c/audio", json={"hp_hz": 400.0})
        pipeline = client.get("/api/v1/pipeline/nodes").json()

    nodes_by_id = {n["node_id"]: n for n in pipeline["nodes"]}
    node_c = nodes_by_id.get("node-c")
    assert node_c is not None
    assert len(node_c["mics"]) >= 1
    assert abs(node_c["mics"][0]["hp_hz"] - 400.0) < 1e-6


def test_multiple_node_overrides_are_independent(monkeypatch, tmp_path: Path) -> None:
    """Separate nodes each get their own override values."""
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        client.patch("/api/v1/pipeline/nodes/node-x/audio", json={"hp_hz": 100.0})
        client.patch("/api/v1/pipeline/nodes/node-y/audio", json={"hp_hz": 800.0})
        pipeline = client.get("/api/v1/pipeline/nodes").json()

    nodes_by_id = {n["node_id"]: n for n in pipeline["nodes"]}
    assert abs(nodes_by_id["node-x"]["mics"][0]["hp_hz"] - 100.0) < 1e-6
    assert abs(nodes_by_id["node-y"]["mics"][0]["hp_hz"] - 800.0) < 1e-6


def test_patch_reports_restart_required_for_startup_snapshot_keys(
    monkeypatch, tmp_path: Path
) -> None:
    """A key baked into the startup LocalizationConfig snapshot cannot hot-apply.

    Reporting success with an empty ``restart_required`` is how a config change
    appears to do nothing and then lands all at once on the next restart.
    """
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/config",
            json={"trigger_rms": 0.05, "trigger_cooldown_seconds": 5.0},
        )
        assert resp.status_code == 200
        restart_required = resp.json()["restart_required"]

    assert "trigger_rms" in restart_required
    assert "trigger_cooldown_seconds" in restart_required


def test_patch_reports_restart_required_for_fusion_worker_count(
    monkeypatch, tmp_path: Path
) -> None:
    """Worker tasks are created in FusionNode.start(); a live patch cannot add any."""
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"fusion_worker_count": 2})
        assert resp.status_code == 200
        assert resp.json()["fusion_worker_count"] == 2
        assert "fusion_worker_count" in resp.json()["restart_required"]


def test_patch_hot_appliable_key_is_not_restart_required(monkeypatch, tmp_path: Path) -> None:
    """Retention knobs are re-read from Settings, so they must not be flagged."""
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"retention_yamnet_audio_seconds": 12345})
        assert resp.status_code == 200
        assert resp.json()["restart_required"] == []


def test_patch_reports_pipeline_in_separate_process_flag(monkeypatch, tmp_path: Path) -> None:
    """Combined-role deployments own their pipeline, so the flag stays false."""
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch("/api/v1/config", json={"trigger_rms": 0.02})
        assert resp.status_code == 200
        assert resp.json()["pipeline_in_separate_process"] is False


def test_patch_classification_backpressure_keys(monkeypatch, tmp_path: Path) -> None:
    """The 2026-08-01 live-box lag review exposed these as env-only; they must be
    PATCHable (restart-required) so a wedged deployment can be tuned in place."""
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.patch(
            "/api/v1/config",
            json={
                "fusion_classification_queue_size": 32,
                "classification_window_seconds": 15.0,
                "drop_on_backpressure": True,
                "fusion_backpressure_drop_policy": "OLDEST",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["fusion_classification_queue_size"] == 32
        assert body["classification_window_seconds"] == 15.0
        # policy normalises to lowercase
        assert body["fusion_backpressure_drop_policy"] == "oldest"
        for key in (
            "fusion_classification_queue_size",
            "classification_window_seconds",
            "drop_on_backpressure",
            "fusion_backpressure_drop_policy",
        ):
            assert key in body["restart_required"], key


def test_patch_classification_backpressure_keys_validation(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        assert client.patch("/api/v1/config", json={"fusion_classification_queue_size": 0}).status_code == 422
        assert client.patch("/api/v1/config", json={"classification_window_seconds": 0.0}).status_code == 422
        assert client.patch("/api/v1/config", json={"fusion_backpressure_drop_policy": "sideways"}).status_code == 422
