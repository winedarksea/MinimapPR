from __future__ import annotations

from pathlib import Path

from minimappr.config import Settings


def test_birdnet_omni_testing_profile_sets_direct_birdnet_defaults() -> None:
    settings = Settings(runtime_profile="birdnet_omni_testing")

    assert settings.runtime_profile == "birdnet_omni_testing"
    assert settings.classifier_backend == "birdnet"
    assert settings.beamformed_classification_enabled is False
    assert settings.skip_localization_for_classification is True
    assert settings.classification_window_seconds == 30.0
    assert settings.max_sensor_buffer_seconds >= 32.0


def test_birdnet_hybrid_production_profile_sets_hybrid_defaults() -> None:
    settings = Settings(runtime_profile="birdnet_hybrid_production")

    assert settings.runtime_profile == "birdnet_hybrid_production"
    assert settings.classifier_backend == "birdnet"
    assert settings.localization_algorithm == "srp_phat"
    assert settings.localization_strategy == "fixed"
    assert settings.beamformed_classification_enabled is False
    assert settings.skip_localization_for_classification is False
    assert settings.classification_window_seconds == 30.0
    assert settings.max_sensor_buffer_seconds >= 32.0
    assert settings.localization_band_min_hz == 300.0
    assert settings.localization_band_max_hz == 3500.0
    assert settings.reporting_window_seconds == 30.0
    assert settings.rules_config_path == Path("data/rules_birdnet_hybrid_production.json")


def test_birdnet_hybrid_production_respects_explicit_rules_path_override() -> None:
    explicit_rules_path = Path("data/custom_rules.json")
    settings = Settings(
        runtime_profile="birdnet_hybrid_production",
        rules_config_path=explicit_rules_path,
    )

    assert settings.rules_config_path == explicit_rules_path


def test_classification_window_defaults_to_localization_window() -> None:
    settings = Settings(localization_window_seconds=0.12, classification_window_seconds=0.0)

    assert settings.classification_window_seconds == 0.12
