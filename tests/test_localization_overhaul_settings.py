"""Localization-overhaul Settings keys: defaults, validation, env, sidecar handoff."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minimappr.config import Settings
from minimappr.core.config_groups import GROUPED_CONFIG_KEYS
from minimappr.core.localization_overhaul_config import (
    LOCALIZATION_OVERHAUL_SETTING_KEYS,
    RUST_FORWARDED_LOCALIZATION_SETTING_KEYS,
    TRACKING_MODE_DEFAULT_CATEGORY_KEY,
    sidecar_environment_variable_name,
)
from minimappr.ingest_sidecar_runtime import sidecar_localization_solve_environment
from minimappr.models import TimeQuality


def test_defaults_match_the_shared_contract() -> None:
    settings = Settings()
    assert settings.localization_max_solve_sensors == 16
    assert settings.localization_min_pair_baseline_m == pytest.approx(0.02)
    assert settings.localization_cross_node_max_timeline_residual_seconds == pytest.approx(0.010)
    assert settings.localization_timeline_residual_window_frames == 16
    assert settings.localization_timeline_residual_percentile == pytest.approx(50.0)
    assert settings.localization_timeline_residual_over_limit_clear_ratio == pytest.approx(0.8)
    assert settings.localization_clock_unknown_uncertainty_seconds == pytest.approx(0.005)
    assert settings.localization_timestamp_step_threshold_seconds == pytest.approx(0.002)
    assert settings.localization_timestamp_step_recovery_holdoff_seconds == pytest.approx(30.0)
    assert settings.localization_trigger_noise_floor_relative_enabled is True
    # The sidecar trigger defaults to the Python-ingest relative trigger's bar.
    assert settings.localization_trigger_noise_floor_relative_db == settings.trigger_noise_floor_relative_snr_db
    assert settings.localization_sensor_registry_stale_seconds == pytest.approx(120.0)
    assert settings.sidecar_geometry_push_estimate_move_threshold_m == pytest.approx(0.5)
    assert settings.sidecar_geometry_reconcile_interval_seconds == pytest.approx(30.0)
    assert settings.localized_handoff_duplicate_merge_max_separation_m == pytest.approx(15.0)
    assert settings.tracking_mode_by_category == {
        "wildlife": "stationary_hold",
        TRACKING_MODE_DEFAULT_CATEGORY_KEY: "continuous_motion",
    }
    assert settings.position_observation_queue_max_items == 1024
    assert settings.position_observation_unclassified_track_policy == "start_hidden_tentative"
    assert settings.point_network_clip_classification_enabled is True


def test_clock_class_defaults_cover_every_time_quality() -> None:
    uncertainty_by_class = Settings().localization_clock_class_uncertainty_seconds
    # Legacy ntp_sync is converted to ntp_disciplined on load, so it needs no entry.
    assert set(uncertainty_by_class) == {quality.value for quality in TimeQuality if quality is not TimeQuality.NTP_SYNC}
    assert uncertainty_by_class["gps_locked"] < uncertainty_by_class["ntp_disciplined"]


def test_dict_defaults_are_not_shared_between_instances() -> None:
    first, second = Settings(), Settings()
    first.tracking_mode_by_category["vehicle"] = "none"
    assert "vehicle" not in second.tracking_mode_by_category


def test_from_env_matches_dataclass_defaults(monkeypatch) -> None:
    for key in LOCALIZATION_OVERHAUL_SETTING_KEYS:
        monkeypatch.delenv(sidecar_environment_variable_name(key), raising=False)
    from_env = Settings.from_env()
    defaults = Settings()
    for key in LOCALIZATION_OVERHAUL_SETTING_KEYS:
        assert getattr(from_env, key) == getattr(defaults, key), key


def test_from_env_parses_json_dict_and_scalars(monkeypatch) -> None:
    monkeypatch.setenv(
        "MINIMAPPR_LOCALIZATION_CLOCK_CLASS_UNCERTAINTY_SECONDS", json.dumps({"gps_locked": 2e-6})
    )
    monkeypatch.setenv("MINIMAPPR_LOCALIZATION_MAX_SOLVE_SENSORS", "8")
    monkeypatch.setenv("MINIMAPPR_TRACKING_MODE_BY_CATEGORY", json.dumps({"default": "none"}))
    settings = Settings.from_env()
    assert settings.localization_clock_class_uncertainty_seconds == {"gps_locked": 2e-6}
    assert settings.localization_max_solve_sensors == 8
    assert settings.tracking_mode_by_category == {"default": "none"}


@pytest.mark.parametrize(
    ("overrides", "message_fragment"),
    [
        ({"localization_max_solve_sensors": 1}, "localization_max_solve_sensors"),
        ({"localization_min_pair_baseline_m": 0.0}, "localization_min_pair_baseline_m"),
        ({"localization_clock_class_uncertainty_seconds": {"sundial": 1.0}}, "unknown time_quality"),
        ({"localization_clock_class_uncertainty_seconds": {"gps_locked": -1.0}}, "gps_locked"),
        ({"localization_timestamp_step_recovery_holdoff_seconds": -1.0}, "holdoff"),
        ({"localization_timeline_residual_window_frames": 0}, "residual_window_frames"),
        ({"localization_timeline_residual_percentile": 101.0}, "residual_percentile"),
        ({"localization_timeline_residual_over_limit_clear_ratio": 0.0}, "clear_ratio"),
        ({"localization_timeline_residual_over_limit_clear_ratio": 1.5}, "clear_ratio"),
        ({"tracking_mode_by_category": {"wildlife": "stationary_hold"}}, "'default' entry"),
        ({"tracking_mode_by_category": {"default": "hover"}}, "tracking_mode_by_category"),
        ({"position_observation_queue_max_items": 0}, "position_observation_queue_max_items"),
        ({"position_observation_unclassified_track_policy": "maybe"}, "unclassified_track_policy"),
    ],
)
def test_invalid_values_are_rejected(overrides, message_fragment) -> None:
    with pytest.raises(ValueError, match=message_fragment):
        Settings(**overrides)


def test_every_new_key_is_grouped_for_the_structured_config() -> None:
    for key in LOCALIZATION_OVERHAUL_SETTING_KEYS:
        assert key in GROUPED_CONFIG_KEYS, key


def test_rust_forwarded_keys_reach_the_sidecar_env() -> None:
    settings = Settings(
        localization_max_solve_sensors=9,
        localization_trigger_noise_floor_relative_enabled=False,
    )
    environment = sidecar_localization_solve_environment(settings)
    assert set(environment) == {
        sidecar_environment_variable_name(key) for key in RUST_FORWARDED_LOCALIZATION_SETTING_KEYS
    }
    assert environment["MINIMAPPR_LOCALIZATION_MAX_SOLVE_SENSORS"] == "9"
    assert environment["MINIMAPPR_LOCALIZATION_TRIGGER_NOISE_FLOOR_RELATIVE_ENABLED"] == "false"
    assert json.loads(environment["MINIMAPPR_LOCALIZATION_CLOCK_CLASS_UNCERTAINTY_SECONDS"]) == (
        settings.localization_clock_class_uncertainty_seconds
    )


def test_partial_settings_double_falls_back_to_dataclass_defaults() -> None:
    environment = sidecar_localization_solve_environment(SimpleNamespace())
    assert environment["MINIMAPPR_LOCALIZATION_MIN_PAIR_BASELINE_M"] == str(Settings().localization_min_pair_baseline_m)
