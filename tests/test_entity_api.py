"""Field Mode W3-entity: GET /api/v1/entities, /ws/live entity events, config registration."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from minimappr.core.fusion_node import DetectionProduct
from minimappr.main import app
from minimappr.models import DetectionEvent
from tests.bird_entity_helpers import ROBIN, observation, seconds, track_state

MAX_WS_MESSAGES_BEFORE_ENTITY = 20


def _configure_env(monkeypatch, tmp_path: Path, *, entity_enabled: bool) -> None:
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")
    monkeypatch.setenv("MINIMAPPR_ENTITY_REPORTING_ENABLED", "true" if entity_enabled else "false")
    monkeypatch.setenv("MINIMAPPR_ENTITY_NOTIFICATION_GRACE_SECONDS", "0")


def test_entities_endpoint_disabled_by_default(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, entity_enabled=False)
    with TestClient(app) as client:
        body = client.get("/api/v1/entities").json()
        assert body["enabled"] is False and body["entities"] == []
        config = client.get("/api/v1/config").json()
        assert config["entity_reporting_enabled"] is False
        assert config["entity_notification_grace_seconds"] == 0.0


def test_entities_endpoint_lists_spatial_and_heard_nearby(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, entity_enabled=True)
    with TestClient(app) as client:
        registry = app.state.fusion_node.entity_reporting.registry
        registry.observe(observation(at_s=0, track=track_state()))
        registry.observe(observation(at_s=1, label="house wren", mode="node_only", node_id="node-far"))
        body = client.get("/api/v1/entities").json()
        assert body["enabled"] is True
        by_state = {entity["state"]: entity for entity in body["entities"]}
        assert set(by_state) == {"pinpointed", "heard_nearby"}
        heard = by_state["heard_nearby"]
        assert heard["position_m"] is None and heard["position_geo"] is None
        assert heard["reporting_node_id"] == "node-far"
        pinpointed = by_state["pinpointed"]
        assert pinpointed["species_top_label"] == ROBIN
        assert pinpointed["position_geo"]["lat"] != 0.0
        assert pinpointed["track_id"] == "trk-1"
        assert pinpointed["first_heard_ns"] == seconds(0)


def test_entity_event_reaches_ws_live(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, entity_enabled=True)
    with TestClient(app) as client, client.websocket_connect("/ws/live") as websocket:
        detection = DetectionEvent(
            id="det-ws", timestamp_ns=seconds(0), source_node_id="node-a", reporting_modality="localized",
            position_m=(6.0, 12.0, 2.0), confidence=0.8, gdop=1.0, label=ROBIN, label_category="wildlife",
            label_confidence=0.8, reference_sensor="m0",
        )
        product = DetectionProduct(detection=detection, track=None, suppressed_by_zone=False,
                                   suppression_reasons=[], pipeline_item_id="det-ws",
                                   event_time_ns=detection.timestamp_ns)
        client.portal.call(app.state.fusion_node._process_rules_and_delivery, product)
        for _ in range(MAX_WS_MESSAGES_BEFORE_ENTITY):
            message = websocket.receive_json()
            if message.get("type") == "entity":
                break
        else:
            raise AssertionError("no entity event on /ws/live")
        assert message["event"] == "created"
        assert message["entity"]["state"] == "pinpointed"
        assert message["entity"]["canonical_detection_id"] == "det-ws"


def test_entity_patch_validation_and_restart_required(monkeypatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path, entity_enabled=False)
    with TestClient(app) as client:
        bad = client.patch("/api/v1/config", json={"entity_absorption_window_seconds": -1})
        assert bad.status_code in (400, 422)
        good = client.patch("/api/v1/config", json={"entity_notification_grace_seconds": 3.0})
        assert good.status_code == 200
        assert "entity_notification_grace_seconds" in good.json().get("restart_required", [])
