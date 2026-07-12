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
    save_overrides(path, {"classifier_backend": "birdnet", "trigger_rms": 0.02})
    assert load_overrides(path) == {"classifier_backend": "birdnet", "trigger_rms": 0.02}


def test_malformed_yaml_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text("::: not : valid : yaml :::\n- [", encoding="utf-8")
    assert load_overrides(path) == {}


def test_non_dict_document_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    assert load_overrides(path) == {}


def test_allowlist_filters_unknown_keys() -> None:
    kept = allowlisted_overrides({"classifier_backend": "birdnet", "bogus_key": 1})
    assert kept == {"classifier_backend": "birdnet"}


def test_save_drops_non_allowlisted(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    save_overrides(path, {"classifier_backend": "birdnet", "db_path": "/evil"})
    assert load_overrides(path) == {"classifier_backend": "birdnet"}


def test_config_overrides_path_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MINIMAPPR_CONFIG_PATH", raising=False)
    assert config_overrides_path() == Path("data/config.yml")
    assert config_overrides_path(tmp_path / "explicit.yml") == tmp_path / "explicit.yml"
    monkeypatch.setenv("MINIMAPPR_CONFIG_PATH", str(tmp_path / "env.yml"))
    assert config_overrides_path() == tmp_path / "env.yml"


def test_precedence_default_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # default
    monkeypatch.delenv("MINIMAPPR_CLASSIFIER", raising=False)
    path = tmp_path / "config.yml"
    monkeypatch.setenv("MINIMAPPR_CONFIG_PATH", str(path))
    assert Settings.from_env().classifier_backend == "auto"

    # env overrides default
    monkeypatch.setenv("MINIMAPPR_CLASSIFIER", "yamnet")
    assert Settings.from_env().classifier_backend == "yamnet"

    # file override wins over env
    save_overrides(path, {"classifier_backend": "heuristic"})
    assert Settings.from_env().classifier_backend == "heuristic"


def test_classification_audio_source_in_allowlist() -> None:
    assert "classification_audio_source" in CONFIG_PATCH_ALLOWLIST
    assert "min_localization_confidence" in CONFIG_PATCH_ALLOWLIST
