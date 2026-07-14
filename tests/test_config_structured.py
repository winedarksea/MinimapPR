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
        # ungrouped only holds the explicit meta keys.
        for key in body["ungrouped"]:
            assert key in UNGROUPED_KEYS


def test_group_flat_config_pure():
    flat = {"trigger_rms": 0.01, "beamformer_type": "das", "classifier_backends_available": []}
    out = group_flat_config(flat)
    gates = next(g for g in out["groups"] if g["id"] == "gates")
    assert {"key": "trigger_rms", "value": 0.01} in gates["entries"]
    assert "classifier_backends_available" in out["ungrouped"]
