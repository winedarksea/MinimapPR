"""Settings coverage for the Home Assistant MQTT bridge configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from minimappr.config import Settings


def test_hass_defaults_are_dormant() -> None:
    settings = Settings()
    assert settings.hass_enabled is False
    assert settings.hass_mqtt_host == ""
    assert settings.hass_mqtt_port == 1883
    assert settings.hass_discovery_prefix == "homeassistant"
    assert settings.hass_base_topic == "minimappr"
    assert settings.hass_device_id == "minimappr"
    assert settings.hass_device_name == "MinimapPR"
    assert settings.hass_publish_interval_seconds == 5.0
    assert settings.hass_track_slot_count == 8
    assert settings.hass_publish_track_slots is False, "track slots must ship disabled"
    assert settings.hass_detection_classes == ("security", "human", "vehicle", "wildlife")
    assert settings.hass_discovery_ledger_path == Path("data/hass_discovery_ledger.json")


def test_hass_config_is_disabled_without_a_host() -> None:
    config = Settings().hass_config()
    assert config.enabled is False
    assert config.mqtt_host == ""


def test_hass_config_derives_enabled_from_flag_and_host() -> None:
    config = Settings(hass_enabled=True, hass_mqtt_host=" broker.local ").hass_config()
    assert config.enabled is True
    assert config.mqtt_host == "broker.local", "host is stripped once, in hass_config()"


def test_hass_config_disabled_when_flag_off_even_with_host() -> None:
    assert Settings(hass_mqtt_host="broker.local").hass_config().enabled is False


def test_hass_config_carries_every_field_through() -> None:
    settings = Settings(
        hass_enabled=True,
        hass_mqtt_host="broker.local",
        hass_mqtt_port=8883,
        hass_mqtt_username="mm",
        hass_mqtt_password="secret",
        hass_mqtt_client_id="mm-1",
        hass_mqtt_keepalive_seconds=30,
        hass_mqtt_tls_enabled=True,
        hass_mqtt_tls_insecure=True,
        hass_discovery_prefix="ha",
        hass_base_topic="mmpr",
        hass_device_id="site_a",
        hass_device_name="Site A",
        hass_publish_interval_seconds=2.5,
        hass_publish_min_interval_seconds=0.5,
        hass_reconcile_interval_seconds=30.0,
        hass_queue_size=10,
        hass_reconnect_backoff_initial_seconds=2.0,
        hass_reconnect_backoff_max_seconds=20.0,
        hass_detection_off_delay_seconds=12,
        hass_detection_classes=("gunshot",),
        hass_track_slot_count=4,
        hass_zone_spl_window_seconds=15.0,
        hass_discovery_ledger_path=Path("/tmp/ledger.json"),
        hass_publish_zone_occupancy=False,
        hass_publish_zone_spl=False,
        hass_publish_detection_classes=False,
        hass_publish_node_status=False,
        hass_publish_system_health=False,
        hass_publish_events=False,
        hass_publish_track_slots=True,
    )
    config = settings.hass_config()
    assert (config.mqtt_port, config.mqtt_username, config.mqtt_password) == (8883, "mm", "secret")
    assert (config.mqtt_client_id, config.mqtt_keepalive_seconds) == ("mm-1", 30)
    assert config.mqtt_tls_enabled and config.mqtt_tls_insecure
    assert (config.discovery_prefix, config.base_topic) == ("ha", "mmpr")
    assert (config.device_id, config.device_name) == ("site_a", "Site A")
    assert config.publish_interval_seconds == 2.5
    assert config.publish_min_interval_seconds == 0.5
    assert config.reconcile_interval_seconds == 30.0
    assert config.queue_size == 10
    assert config.reconnect_backoff_initial_seconds == 2.0
    assert config.reconnect_backoff_max_seconds == 20.0
    assert config.detection_off_delay_seconds == 12
    assert config.detection_classes == ("gunshot",)
    assert config.track_slot_count == 4
    assert config.zone_spl_window_seconds == 15.0
    assert config.discovery_ledger_path == Path("/tmp/ledger.json")
    assert not any(
        (
            config.publish_zone_occupancy,
            config.publish_zone_spl,
            config.publish_detection_classes,
            config.publish_node_status,
            config.publish_system_health,
            config.publish_events,
        )
    )
    assert config.publish_track_slots is True
    assert config.version, "device block sw_version must not be blank"


# -- validation --------------------------------------------------------------


def test_enabled_without_a_host_is_a_config_error() -> None:
    with pytest.raises(ValueError, match="MINIMAPPR_HASS_MQTT_HOST"):
        Settings(hass_enabled=True)


def test_enabled_with_whitespace_only_host_is_a_config_error() -> None:
    with pytest.raises(ValueError, match="MINIMAPPR_HASS_MQTT_HOST"):
        Settings(hass_enabled=True, hass_mqtt_host="   ")


@pytest.mark.parametrize(
    ("kwargs", "expected_env"),
    [
        ({"hass_mqtt_port": 0}, "MINIMAPPR_HASS_MQTT_PORT"),
        ({"hass_mqtt_port": 70000}, "MINIMAPPR_HASS_MQTT_PORT"),
        ({"hass_mqtt_keepalive_seconds": 0}, "MINIMAPPR_HASS_MQTT_KEEPALIVE_SECONDS"),
        ({"hass_publish_interval_seconds": 0.5}, "MINIMAPPR_HASS_PUBLISH_INTERVAL_SECONDS"),
        ({"hass_publish_min_interval_seconds": -1.0}, "MINIMAPPR_HASS_PUBLISH_MIN_INTERVAL_SECONDS"),
        ({"hass_reconcile_interval_seconds": 0.0}, "MINIMAPPR_HASS_RECONCILE_INTERVAL_SECONDS"),
        ({"hass_queue_size": 0}, "MINIMAPPR_HASS_QUEUE_SIZE"),
        (
            {"hass_reconnect_backoff_initial_seconds": 0.0},
            "MINIMAPPR_HASS_RECONNECT_BACKOFF_INITIAL_SECONDS",
        ),
        (
            {"hass_reconnect_backoff_initial_seconds": 10.0, "hass_reconnect_backoff_max_seconds": 5.0},
            "MINIMAPPR_HASS_RECONNECT_BACKOFF_MAX_SECONDS",
        ),
        ({"hass_detection_off_delay_seconds": 0}, "MINIMAPPR_HASS_DETECTION_OFF_DELAY_SECONDS"),
        ({"hass_track_slot_count": -1}, "MINIMAPPR_HASS_TRACK_SLOT_COUNT"),
        ({"hass_track_slot_count": 65}, "MINIMAPPR_HASS_TRACK_SLOT_COUNT"),
        ({"hass_zone_spl_window_seconds": 0.0}, "MINIMAPPR_HASS_ZONE_SPL_WINDOW_SECONDS"),
        ({"hass_discovery_prefix": "a/b"}, "MINIMAPPR_HASS_DISCOVERY_PREFIX"),
        ({"hass_base_topic": "a+b"}, "MINIMAPPR_HASS_BASE_TOPIC"),
        ({"hass_base_topic": ""}, "MINIMAPPR_HASS_BASE_TOPIC"),
        ({"hass_device_id": "a#"}, "MINIMAPPR_HASS_DEVICE_ID"),
    ],
)
def test_each_validation_error_names_its_env_var(kwargs: dict, expected_env: str) -> None:
    with pytest.raises(ValueError, match=expected_env):
        Settings(**kwargs)


def test_track_slot_count_zero_is_allowed() -> None:
    """0 slots is a legitimate way to keep the pool but publish nothing."""
    assert Settings(hass_track_slot_count=0).hass_config().track_slot_count == 0


# -- env ---------------------------------------------------------------------


def test_every_hass_env_var_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {
        "MINIMAPPR_HASS_ENABLED": "1",
        "MINIMAPPR_HASS_BASE_URL": "http://ha.local:8123",
        "MINIMAPPR_HASS_TOKEN": "tok",
        "MINIMAPPR_HASS_MQTT_HOST": "broker.local",
        "MINIMAPPR_HASS_MQTT_PORT": "8883",
        "MINIMAPPR_HASS_MQTT_USERNAME": "mm",
        "MINIMAPPR_HASS_MQTT_PASSWORD": "pw",
        "MINIMAPPR_HASS_MQTT_CLIENT_ID": "mm-1",
        "MINIMAPPR_HASS_MQTT_KEEPALIVE_SECONDS": "30",
        "MINIMAPPR_HASS_MQTT_TLS_ENABLED": "1",
        "MINIMAPPR_HASS_MQTT_TLS_INSECURE": "1",
        "MINIMAPPR_HASS_DISCOVERY_PREFIX": "ha",
        "MINIMAPPR_HASS_BASE_TOPIC": "mmpr",
        "MINIMAPPR_HASS_DEVICE_ID": "site_a",
        "MINIMAPPR_HASS_DEVICE_NAME": "Site A",
        "MINIMAPPR_HASS_PUBLISH_INTERVAL_SECONDS": "2.5",
        "MINIMAPPR_HASS_PUBLISH_MIN_INTERVAL_SECONDS": "0.25",
        "MINIMAPPR_HASS_RECONCILE_INTERVAL_SECONDS": "30",
        "MINIMAPPR_HASS_QUEUE_SIZE": "42",
        "MINIMAPPR_HASS_RECONNECT_BACKOFF_INITIAL_SECONDS": "2",
        "MINIMAPPR_HASS_RECONNECT_BACKOFF_MAX_SECONDS": "20",
        "MINIMAPPR_HASS_DETECTION_OFF_DELAY_SECONDS": "12",
        "MINIMAPPR_HASS_DETECTION_CLASSES": "gunshot,speech",
        "MINIMAPPR_HASS_TRACK_SLOT_COUNT": "4",
        "MINIMAPPR_HASS_ZONE_SPL_WINDOW_SECONDS": "15",
        "MINIMAPPR_HASS_DISCOVERY_LEDGER_PATH": "/tmp/led.json",
        "MINIMAPPR_HASS_PUBLISH_ZONE_OCCUPANCY": "0",
        "MINIMAPPR_HASS_PUBLISH_ZONE_SPL": "0",
        "MINIMAPPR_HASS_PUBLISH_DETECTION_CLASSES": "0",
        "MINIMAPPR_HASS_PUBLISH_NODE_STATUS": "0",
        "MINIMAPPR_HASS_PUBLISH_SYSTEM_HEALTH": "0",
        "MINIMAPPR_HASS_PUBLISH_EVENTS": "0",
        "MINIMAPPR_HASS_PUBLISH_TRACK_SLOTS": "1",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    settings = Settings.from_env()

    assert settings.hass_enabled is True
    assert settings.hass_base_url == "http://ha.local:8123"
    assert settings.hass_token == "tok"
    assert settings.hass_mqtt_host == "broker.local"
    assert settings.hass_mqtt_port == 8883
    assert settings.hass_mqtt_username == "mm"
    assert settings.hass_mqtt_password == "pw"
    assert settings.hass_mqtt_client_id == "mm-1"
    assert settings.hass_mqtt_keepalive_seconds == 30
    assert settings.hass_mqtt_tls_enabled is True
    assert settings.hass_mqtt_tls_insecure is True
    assert settings.hass_discovery_prefix == "ha"
    assert settings.hass_base_topic == "mmpr"
    assert settings.hass_device_id == "site_a"
    assert settings.hass_device_name == "Site A"
    assert settings.hass_publish_interval_seconds == 2.5
    assert settings.hass_publish_min_interval_seconds == 0.25
    assert settings.hass_reconcile_interval_seconds == 30.0
    assert settings.hass_queue_size == 42
    assert settings.hass_reconnect_backoff_initial_seconds == 2.0
    assert settings.hass_reconnect_backoff_max_seconds == 20.0
    assert settings.hass_detection_off_delay_seconds == 12
    assert settings.hass_detection_classes == ("gunshot", "speech")
    assert settings.hass_track_slot_count == 4
    assert settings.hass_zone_spl_window_seconds == 15.0
    assert settings.hass_discovery_ledger_path == Path("/tmp/led.json")
    assert settings.hass_publish_track_slots is True
    assert not settings.hass_publish_zone_occupancy
    assert not settings.hass_publish_events


def test_from_env_leaves_hass_dormant_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(Settings.__dataclass_fields__):
        monkeypatch.delenv(f"MINIMAPPR_{key.upper()}", raising=False)
    settings = Settings.from_env()
    assert settings.hass_config().enabled is False
