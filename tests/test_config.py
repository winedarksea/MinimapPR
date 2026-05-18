from __future__ import annotations

import os

import pytest

from minimappr.config import ClassifierConfig, FusionConfig, LocalizationConfig, Settings


def _clear_minimappr_env(monkeypatch) -> None:
    for key in list(os.environ):
        if key.startswith("MINIMAPPR_"):
            monkeypatch.delenv(key, raising=False)


def test_settings_from_env_populates_subconfigs_with_cleanup_defaults(monkeypatch, tmp_path) -> None:
    _clear_minimappr_env(monkeypatch)
    monkeypatch.setenv("MINIMAPPR_FEDERATION_PEERS_CONFIG_PATH", str(tmp_path / "missing-peers.json"))

    settings = Settings.from_env()

    localization = settings.localization_config()
    classifier = settings.classifier_config()
    fusion = settings.fusion_config()

    assert isinstance(localization, LocalizationConfig)
    assert isinstance(classifier, ClassifierConfig)
    assert isinstance(fusion, FusionConfig)
    assert settings.localization_max_tau_seconds == 0.02
    assert localization.localization_max_tau_seconds == 0.02
    assert settings.classifier_stage_timeout_seconds == 30.0
    assert classifier.stage_timeout_seconds == 30.0
    assert fusion.event_queue_size == 512
    assert fusion.localization_queue_size == 1024
    assert fusion.classification_queue_size == 1024
    assert fusion.rules_queue_size == 512


def test_settings_from_env_accepts_legacy_cleanup_env_keys(monkeypatch, tmp_path) -> None:
    _clear_minimappr_env(monkeypatch)
    monkeypatch.setenv("MINIMAPPR_FEDERATION_PEERS_CONFIG_PATH", str(tmp_path / "missing-peers.json"))
    monkeypatch.setenv("MINIMAPPR_LOCALIZATION_MAX_TAU_S", "0.031")
    monkeypatch.setenv("MINIMAPPR_CLASSIFICATION_STAGE_TIMEOUT_SECONDS", "0.75")
    monkeypatch.setenv("MINIMAPPR_FUSION_DROP_ON_BACKPRESSURE", "false")

    settings = Settings.from_env()

    assert settings.localization_max_tau_seconds == pytest.approx(0.031)
    assert settings.localization_max_tau_s == pytest.approx(0.031)
    assert settings.classifier_stage_timeout_seconds == pytest.approx(0.75)
    assert settings.classification_stage_timeout_seconds == pytest.approx(0.75)
    assert settings.drop_on_backpressure is False
    assert settings.fusion_drop_on_backpressure is False


def test_settings_cleanup_alias_properties_track_canonical_fields() -> None:
    settings = Settings(
        localization_max_tau_seconds=0.02,
        classifier_stage_timeout_seconds=0.5,
        drop_on_backpressure=True,
    )

    settings.localization_max_tau_s = 0.04
    settings.classification_stage_timeout_seconds = 0.25
    settings.fusion_drop_on_backpressure = False

    assert settings.localization_max_tau_seconds == pytest.approx(0.04)
    assert settings.classifier_stage_timeout_seconds == pytest.approx(0.25)
    assert settings.drop_on_backpressure is False


def test_settings_reject_zero_fusion_queue_size() -> None:
    with pytest.raises(ValueError, match="fusion_event_queue_size must be >= 1"):
        Settings(fusion_event_queue_size=0)