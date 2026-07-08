from __future__ import annotations

import os

import pytest

from minimappr.config import (
    ClassifierConfig,
    FusionConfig,
    IngestSidecarProcessConfig,
    IngestSidecarStartupConfig,
    LocalizationConfig,
    Settings,
)


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
    sidecar_process = settings.ingest_sidecar_process_config()
    sidecar_startup = settings.ingest_sidecar_startup_config()

    assert isinstance(localization, LocalizationConfig)
    assert isinstance(classifier, ClassifierConfig)
    assert isinstance(fusion, FusionConfig)
    assert isinstance(sidecar_process, IngestSidecarProcessConfig)
    assert isinstance(sidecar_startup, IngestSidecarStartupConfig)
    assert settings.localization_max_tau_seconds == 0.02
    assert localization.localization_max_tau_seconds == 0.02
    assert settings.classifier_stage_timeout_seconds == 30.0
    assert classifier.stage_timeout_seconds == 30.0
    assert fusion.event_queue_size == 512
    assert fusion.localization_queue_size == 1024
    assert fusion.classification_queue_size == 1024
    assert fusion.rules_queue_size == 512
    assert sidecar_process.sidecar_port == 8081
    assert sidecar_process.storage_mode == "spool"
    assert sidecar_process.memory_only_live_path is True
    assert sidecar_startup.ready_timeout_seconds == pytest.approx(5.0)
    assert sidecar_startup.ready_poll_interval_seconds == pytest.approx(0.1)
    assert sidecar_startup.healthcheck_timeout_seconds == pytest.approx(0.5)
    assert settings.ingest_max_concurrent == 64
    assert settings.iamf_ambi_profile == "parametric_v2"


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


def test_settings_from_env_populates_sidecar_startup_timeouts(monkeypatch, tmp_path) -> None:
    _clear_minimappr_env(monkeypatch)
    monkeypatch.setenv("MINIMAPPR_FEDERATION_PEERS_CONFIG_PATH", str(tmp_path / "missing-peers.json"))
    monkeypatch.setenv("MINIMAPPR_SIDECAR_READY_TIMEOUT_SECONDS", "7.5")
    monkeypatch.setenv("MINIMAPPR_SIDECAR_READY_POLL_INTERVAL_SECONDS", "0.2")
    monkeypatch.setenv("MINIMAPPR_SIDECAR_HEALTHCHECK_TIMEOUT_SECONDS", "1.25")

    settings = Settings.from_env()
    sidecar_startup = settings.ingest_sidecar_startup_config()

    assert sidecar_startup.ready_timeout_seconds == pytest.approx(7.5)
    assert sidecar_startup.ready_poll_interval_seconds == pytest.approx(0.2)
    assert sidecar_startup.healthcheck_timeout_seconds == pytest.approx(1.25)


def test_settings_from_env_reads_ingest_max_concurrent(monkeypatch, tmp_path) -> None:
    _clear_minimappr_env(monkeypatch)
    monkeypatch.setenv("MINIMAPPR_FEDERATION_PEERS_CONFIG_PATH", str(tmp_path / "missing-peers.json"))
    monkeypatch.setenv("MINIMAPPR_INGEST_MAX_CONCURRENT", "7")

    settings = Settings.from_env()

    assert settings.ingest_max_concurrent == 7


def test_settings_from_env_reads_iamf_ambi_profile(monkeypatch, tmp_path) -> None:
    _clear_minimappr_env(monkeypatch)
    monkeypatch.setenv("MINIMAPPR_FEDERATION_PEERS_CONFIG_PATH", str(tmp_path / "missing-peers.json"))
    monkeypatch.setenv("MINIMAPPR_IAMF_AMBI_PROFILE", "linear_v1")

    settings = Settings.from_env()

    assert settings.iamf_ambi_profile == "linear_v1"


def test_settings_rejects_unknown_iamf_ambi_profile() -> None:
    with pytest.raises(ValueError, match="MINIMAPPR_IAMF_AMBI_PROFILE"):
        Settings(iamf_ambi_profile="not-a-profile")


def test_settings_reject_non_positive_ingest_max_concurrent() -> None:
    with pytest.raises(ValueError, match="MINIMAPPR_INGEST_MAX_CONCURRENT"):
        Settings(ingest_max_concurrent=0)


def test_settings_reject_invalid_far_field_localization_config() -> None:
    with pytest.raises(ValueError, match="FAR_FIELD_DEFAULT_RANGE"):
        Settings(localization_far_field_default_range_m=0.0)

    # Phase 2 cross-validation: the range knobs must form a coherent ladder, so a
    # far-field-max below the default (which would silently clamp the seed range) is
    # now rejected rather than accepted.
    with pytest.raises(ValueError, match="FAR_FIELD_MAX_RANGE_M must be >="):
        Settings(
            localization_far_field_default_range_m=100.0,
            localization_far_field_max_range_m=50.0,
        )

    # A coherent ladder (default <= far-max <= sanity-gate) is accepted.
    settings = Settings(
        localization_far_field_default_range_m=50.0,
        localization_far_field_max_range_m=800.0,
        localization_max_range_m=900.0,
    )
    assert settings.localization_far_field_max_range_m == 800.0

    with pytest.raises(ValueError, match="MUSIC_AZ_STEP"):
        Settings(localization_music_azimuth_step_deg=90.0)

    with pytest.raises(ValueError, match="MUSIC_EL_STEP"):
        Settings(localization_music_elevation_step_deg=90.0)
