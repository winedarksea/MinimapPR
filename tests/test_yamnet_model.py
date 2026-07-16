"""Tests for the project-bundled YAMNet SavedModel asset contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from minimappr.classifiers.yamnet_model import (
    validate_yamnet_model_asset,
    yamnet_model_asset_directory,
)


def test_bundled_yamnet_asset_contains_the_complete_saved_model() -> None:
    asset_directory = validate_yamnet_model_asset()
    assert asset_directory == yamnet_model_asset_directory()


def test_validation_identifies_missing_yamnet_asset_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="saved_model.pb"):
        validate_yamnet_model_asset(tmp_path)
