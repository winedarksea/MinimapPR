"""Tests for GET /api/v1/config/structured and config-group coverage."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from minimappr.core.config_groups import (
    CONFIG_STAGE_GROUPS,
    UNGROUPED_KEYS,
    group_flat_config,
)
from minimappr.main import app


def _configure_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")


def _flat_config(monkeypatch, tmp_path) -> dict:
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.get("/api/v1/config")
        assert resp.status_code == 200
        return resp.json()


def test_every_flat_key_in_exactly_one_group_or_ungrouped(monkeypatch, tmp_path):
    flat = _flat_config(monkeypatch, tmp_path)
    group_keys: list[str] = []
    for _id, _title, _stage, keys in CONFIG_STAGE_GROUPS:
        group_keys.extend(keys)
    # No key appears in two groups.
    assert len(group_keys) == len(set(group_keys)), "config key duplicated across groups"
    grouped = set(group_keys)
    for key in flat:
        placements = (1 if key in grouped else 0) + (1 if key in UNGROUPED_KEYS else 0)
        assert placements == 1, f"config key {key!r} must be in exactly one group or UNGROUPED_KEYS (got {placements})"


def test_no_group_key_missing_from_flat_config(monkeypatch, tmp_path):
    flat = _flat_config(monkeypatch, tmp_path)
    for _id, _title, _stage, keys in CONFIG_STAGE_GROUPS:
        for key in keys:
            assert key in flat, f"group references nonexistent flat key {key!r}"


def test_structured_endpoint_shape(monkeypatch, tmp_path):
    _configure_env(monkeypatch, tmp_path)
    with TestClient(app) as client:
        resp = client.get("/api/v1/config/structured")
        assert resp.status_code == 200
        body = resp.json()
        assert "groups" in body and "ungrouped" in body
        group_ids = {g["id"] for g in body["groups"]}
        assert "localization" in group_ids
        assert "classification" in group_ids
        # The "hass" block lives in its own group, not under rules_alerts; the id
        # matches the /settings/integrations route that edits it.
        assert "integrations" in group_ids
        integrations = next(g for g in body["groups"] if g["id"] == "integrations")
        assert [entry["key"] for entry in integrations["entries"]] == ["hass"]
        # ungrouped only holds the explicit meta keys.
        for key in body["ungrouped"]:
            assert key in UNGROUPED_KEYS


def test_group_flat_config_pure():
    flat = {"trigger_rms": 0.01, "beamformer_type": "das", "classifier_backends_available": []}
    out = group_flat_config(flat)
    gates = next(g for g in out["groups"] if g["id"] == "gates")
    assert {"key": "trigger_rms", "value": 0.01} in gates["entries"]
    assert "classifier_backends_available" in out["ungrouped"]


def test_shipped_stt_defaults_do_not_trip_the_buffer_clamp(monkeypatch, caplog) -> None:
    """Settings.from_env() must agree with the dataclass defaults.

    ``from_env`` supplies a value for every field, so its hard-coded fallback --
    not the dataclass default -- is what a real deployment gets. The two had
    drifted (27.0 vs 30.0), so every boot logged a WARNING and silently clamped
    the STT window even though the shipped default was already correct.
    """
    import logging

    from minimappr.config import Settings

    for var in (
        "MINIMAPPR_STT_MAX_UTTERANCE_SECONDS",
        "MINIMAPPR_STT_PRE_ROLL_SECONDS",
        "MINIMAPPR_STT_HANGOVER_SECONDS",
        "MINIMAPPR_MAX_SENSOR_BUFFER_SECONDS",
    ):
        monkeypatch.delenv(var, raising=False)

    with caplog.at_level(logging.WARNING, logger="minimappr.config"):
        settings = Settings.from_env()

    assert settings.stt_max_utterance_seconds == Settings().stt_max_utterance_seconds
    span = (
        settings.stt_pre_roll_seconds
        + settings.stt_max_utterance_seconds
        + settings.stt_hangover_seconds
    )
    assert span <= settings.max_sensor_buffer_seconds
    assert not [r for r in caplog.records if "clamping stt_max_utterance_seconds" in r.message]
