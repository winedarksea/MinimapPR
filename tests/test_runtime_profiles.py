from __future__ import annotations

from minimappr.config import Settings


def test_birdnet_omni_testing_profile_sets_direct_birdnet_defaults() -> None:
    settings = Settings(runtime_profile="birdnet_omni_testing")

    assert settings.runtime_profile == "birdnet_omni_testing"
    assert settings.classifier_backend == "birdnet"
    assert settings.beamformed_classification_enabled is False
    assert settings.skip_localization_for_classification is True
    assert settings.classification_window_seconds == 30.0
    assert settings.max_sensor_buffer_seconds >= 32.0


def test_classification_window_defaults_to_localization_window() -> None:
    settings = Settings(localization_window_seconds=0.12, classification_window_seconds=0.0)

    assert settings.classification_window_seconds == 0.12
