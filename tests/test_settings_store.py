"""Tests for the persistent YAML settings-override store."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from minimappr.config import Settings
from minimappr.settings_store import (
    CONFIG_PATCH_ALLOWLIST,
    allowlisted_overrides,
    config_overrides_path,
    load_overrides,
    save_overrides,
)


def test_load_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_overrides(tmp_path / "nope.yml") == {}


def test_roundtrip_save_load(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    save_overrides(path, {"birdnet_enabled": False, "trigger_rms": 0.02})
    assert load_overrides(path) == {"birdnet_enabled": False, "trigger_rms": 0.02}


def test_malformed_yaml_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text("::: not : valid : yaml :::\n- [", encoding="utf-8")
    assert load_overrides(path) == {}


def test_non_dict_document_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    assert load_overrides(path) == {}


def test_allowlist_filters_unknown_keys() -> None:
    kept = allowlisted_overrides({"birdnet_enabled": False, "bogus_key": 1})
    assert kept == {"birdnet_enabled": False}


def test_save_drops_non_allowlisted(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    save_overrides(path, {"birdnet_enabled": False, "db_path": "/evil"})
    assert load_overrides(path) == {"birdnet_enabled": False}


def test_config_overrides_path_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MINIMAPPR_CONFIG_PATH", raising=False)
    assert config_overrides_path() == Path("data/config.yml")
    assert config_overrides_path(tmp_path / "explicit.yml") == tmp_path / "explicit.yml"
    monkeypatch.setenv("MINIMAPPR_CONFIG_PATH", str(tmp_path / "env.yml"))
    assert config_overrides_path() == tmp_path / "env.yml"


def test_precedence_default_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # default
    monkeypatch.delenv("MINIMAPPR_BIRDNET_ENABLED", raising=False)
    path = tmp_path / "config.yml"
    monkeypatch.setenv("MINIMAPPR_CONFIG_PATH", str(path))
    assert Settings.from_env().birdnet_enabled is True

    # env overrides default
    monkeypatch.setenv("MINIMAPPR_BIRDNET_ENABLED", "false")
    assert Settings.from_env().birdnet_enabled is False

    # file override wins over env
    save_overrides(path, {"birdnet_enabled": True})
    assert Settings.from_env().birdnet_enabled is True


def test_removed_classifier_env_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIMAPPR_CLASSIFIER", "yamnet")
    with pytest.raises(ValueError, match="MINIMAPPR_CLASSIFIER is removed"):
        Settings.from_env()
    monkeypatch.delenv("MINIMAPPR_CLASSIFIER")
    monkeypatch.setenv("MINIMAPPR_MODEL_CHAIN_CONFIG_PATH", "x.json")
    with pytest.raises(ValueError, match="MINIMAPPR_MODEL_CHAIN_CONFIG_PATH is removed"):
        Settings.from_env()


def test_classification_audio_source_in_allowlist() -> None:
    assert "classification_audio_source" in CONFIG_PATCH_ALLOWLIST
    assert "min_localization_confidence" in CONFIG_PATCH_ALLOWLIST
