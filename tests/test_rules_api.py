from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from minimappr.core.rules import ConfigRuleEngine
from minimappr.main import app
from minimappr.models import DetectionEvent


def _configure_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    rules_path = tmp_path / "rules.json"
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_RULES_CONFIG_PATH", str(rules_path))
    monkeypatch.setenv("MINIMAPPR_FUSION_WORKER_COUNT", "1")
    return rules_path


def _sample_rule(rule_id: str = "test_security") -> dict:
    return {
        "id": rule_id,
        "enabled": True,
        "scope": "detection",
        "when": {
            "label_categories": ["security"],
            "min_confidence": 0.4,
        },
        "actions": [
            {
                "type": "alert",
                "destination": "cop",
                "priority": "high",
                "payload": {"message": "Security detection"},
            }
        ],
        "cooldown_seconds": 0.0,
    }


def _security_detection(confidence: float = 0.9) -> DetectionEvent:
    return DetectionEvent(
        id="det-001",
        timestamp_ns=1_700_000_000_000_000_000,
        position_m=(0.0, 0.0, 0.0),
        confidence=confidence,
        gdop=1.2,
        label="gunshot",
        label_category="security",
        label_confidence=confidence,
        reference_sensor="node-a:ch0",
    )


def test_get_rules_returns_defaults_when_file_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rules_path = _configure_env(monkeypatch, tmp_path)

    with TestClient(app) as client:
        response = client.get("/api/v1/rules")

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "default"
    assert body["path"] == str(rules_path)
    assert body["rules"]


def test_get_rules_returns_defaults_when_file_invalid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rules_path = _configure_env(monkeypatch, tmp_path)
    rules_path.parent.mkdir(parents=True, exist_ok=True)
    rules_path.write_text("{not valid json", encoding="utf-8")

    with TestClient(app) as client:
        response = client.get("/api/v1/rules")

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "default"
    assert body["rules"]


def test_put_rules_validates_and_round_trips(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rules_path = _configure_env(monkeypatch, tmp_path)

    with TestClient(app) as client:
        bad_response = client.put("/api/v1/rules", json={"rules": [{"id": "", "actions": []}]})
        assert bad_response.status_code == 422

        rule = _sample_rule()
        put_response = client.put("/api/v1/rules", json={"rules": [rule]})
        assert put_response.status_code == 200
        put_body = put_response.json()
        assert put_body["source"] == "file"
        assert put_body["rules"][0]["id"] == "test_security"
        assert put_body["rules"][0]["actions"][0]["payload"]["message"] == "Security detection"

        get_response = client.get("/api/v1/rules")
        assert get_response.status_code == 200
        assert get_response.json() == put_body

    written = json.loads(rules_path.read_text(encoding="utf-8"))
    assert written["rules"][0]["id"] == "test_security"


@pytest.mark.asyncio
async def test_put_rules_hot_reloads_existing_engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rules_path = _configure_env(monkeypatch, tmp_path)
    rules_path.write_text(json.dumps({"rules": [_sample_rule("initial_rule")]}), encoding="utf-8")
    # TTL 0 so the engine re-stats the file on every evaluate; the production
    # default throttles the stat to ~1s, which would make this test flaky.
    engine = ConfigRuleEngine(rules_path, reload_ttl_seconds=0.0)

    first = await engine.evaluate(detection=_security_detection(), track=None)
    assert first[0].rule_id == "initial_rule"
    assert first[0].matched is True

    with TestClient(app) as client:
        response = client.put("/api/v1/rules", json={"rules": [_sample_rule("updated_rule")]})
        assert response.status_code == 200

    second = await engine.evaluate(detection=_security_detection(), track=None)
    assert second[0].rule_id == "updated_rule"
    assert second[0].matched is True


def test_put_rules_rejects_bad_confidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    rule = _sample_rule()
    rule["when"]["min_confidence"] = 2.0

    with TestClient(app) as client:
        response = client.put("/api/v1/rules", json={"rules": [rule]})

    assert response.status_code == 422


def test_rules_api_round_trips_transcript_rule(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _configure_env(monkeypatch, tmp_path)
    transcript_rule = {
        "id": "spoken_help",
        "enabled": True,
        "scope": "transcript",
        "when": {"transcript_contains": ["help me"]},
        "actions": [{"type": "alert", "destination": "cop", "priority": "critical"}],
        "cooldown_seconds": 0.0,
    }

    with TestClient(app) as client:
        response = client.put("/api/v1/rules", json={"rules": [transcript_rule]})
        assert response.status_code == 200
        assert response.json()["rules"][0]["scope"] == "transcript"
        assert response.json()["rules"][0]["when"]["transcript_contains"] == ["help me"]
