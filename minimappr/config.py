"""Runtime configuration for MinimapPR."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    return float(raw) if raw is not None else default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    return int(raw) if raw is not None else default


def _env_str(key: str, default: str) -> str:
    raw = os.getenv(key)
    return raw if raw is not None else default


@dataclass(slots=True)
class LocalizationConfig:
    trigger_rms: float
    trigger_cooldown_seconds: float
    localization_window_seconds: float
    max_sensor_buffer_seconds: float
    default_temperature_c: float
    default_humidity: float
    audio_highpass_hz: float
    audio_lowpass_hz: float
    preprocess_enabled: bool
    min_sensors_for_3d: int
    min_sensors_for_2d: int


@dataclass(slots=True)
class TrackingConfig:
    association_distance_m: float
    track_stale_seconds: float
    tracking_filter: str
    kalman_process_noise: float
    kalman_measurement_noise: float
    kalman_initial_position_variance: float
    kalman_initial_velocity_variance: float


@dataclass(slots=True)
class ClassifierConfig:
    backend: str
    yamnet_min_confidence: float


@dataclass(slots=True)
class StorageConfig:
    db_path: Path
    snippet_dir: Path
    snippet_retention_seconds: int
    retention_ephemeral_seconds: int
    retention_short_seconds: int
    retention_long_seconds: int
    retention_experiment_seconds: int
    large_artifact_dir: Path


@dataclass(slots=True)
class FusionConfig:
    worker_count: int
    event_queue_size: int
    localization_queue_size: int
    classification_queue_size: int
    rules_queue_size: int
    drop_on_backpressure: bool
    offline_replay_mode: bool


@dataclass(slots=True)
class RulesConfig:
    rules_config_path: Path
    taxonomy_config_path: Path
    model_chain_config_path: Path


@dataclass(slots=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 8080
    db_path: Path = Path("data/minimappr.db")
    snippet_dir: Path = Path("data/snippets")
    snippet_retention_seconds: int = 3600
    rules_config_path: Path = Path("data/rules.json")
    taxonomy_config_path: Path = Path("data/taxonomy.json")
    model_chain_config_path: Path = Path("data/model_chain.json")
    large_artifact_dir: Path = Path("data/artifacts")

    trigger_rms: float = 0.015
    trigger_cooldown_seconds: float = 0.8
    localization_window_seconds: float = 0.08
    max_sensor_buffer_seconds: float = 8.0
    preprocess_enabled: bool = True
    audio_highpass_hz: float = 50.0
    audio_lowpass_hz: float = 0.0
    min_sensors_for_3d: int = 4
    min_sensors_for_2d: int = 3

    default_temperature_c: float = 20.0
    default_humidity: float = 0.5
    site_origin_lat: float = 37.7749
    site_origin_lon: float = -122.4194
    site_origin_alt_m: float = 0.0
    coordinate_mode: str = "flat"

    classifier_backend: str = "heuristic"
    yamnet_min_confidence: float = 0.25

    association_distance_m: float = 8.0
    track_stale_seconds: float = 20.0
    tracking_filter: str = "linear"
    kalman_process_noise: float = 2.0
    kalman_measurement_noise: float = 1.5
    kalman_initial_position_variance: float = 4.0
    kalman_initial_velocity_variance: float = 16.0

    fusion_worker_count: int = 1
    fusion_event_queue_size: int = 256
    fusion_localization_queue_size: int = 256
    fusion_classification_queue_size: int = 256
    fusion_rules_queue_size: int = 256
    fusion_drop_on_backpressure: bool = True
    fusion_offline_replay_mode: bool = False

    cleanup_interval_seconds: float = 15.0
    node_degraded_after_seconds: float = 15.0
    node_offline_after_seconds: float = 45.0
    event_stale_seconds: float = 30.0
    retention_ephemeral_seconds: int = 900
    retention_short_seconds: int = 86_400
    retention_long_seconds: int = 2_592_000
    retention_experiment_seconds: int = 21_600

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            host=_env_str("MINIMAPPR_HOST", "0.0.0.0"),
            port=_env_int("MINIMAPPR_PORT", 8080),
            db_path=Path(_env_str("MINIMAPPR_DB_PATH", "data/minimappr.db")),
            snippet_dir=Path(_env_str("MINIMAPPR_SNIPPET_DIR", "data/snippets")),
            snippet_retention_seconds=_env_int("MINIMAPPR_SNIPPET_RETENTION_SECONDS", 3600),
            rules_config_path=Path(_env_str("MINIMAPPR_RULES_CONFIG_PATH", "data/rules.json")),
            taxonomy_config_path=Path(_env_str("MINIMAPPR_TAXONOMY_CONFIG_PATH", "data/taxonomy.json")),
            model_chain_config_path=Path(_env_str("MINIMAPPR_MODEL_CHAIN_CONFIG_PATH", "data/model_chain.json")),
            large_artifact_dir=Path(_env_str("MINIMAPPR_LARGE_ARTIFACT_DIR", "data/artifacts")),
            trigger_rms=_env_float("MINIMAPPR_TRIGGER_RMS", 0.015),
            trigger_cooldown_seconds=_env_float("MINIMAPPR_TRIGGER_COOLDOWN_SECONDS", 0.8),
            localization_window_seconds=_env_float("MINIMAPPR_LOCALIZATION_WINDOW_SECONDS", 0.08),
            max_sensor_buffer_seconds=_env_float("MINIMAPPR_MAX_SENSOR_BUFFER_SECONDS", 8.0),
            preprocess_enabled=_env_bool("MINIMAPPR_PREPROCESS_ENABLED", True),
            audio_highpass_hz=_env_float("MINIMAPPR_AUDIO_HIGHPASS_HZ", 50.0),
            audio_lowpass_hz=_env_float("MINIMAPPR_AUDIO_LOWPASS_HZ", 0.0),
            min_sensors_for_3d=_env_int("MINIMAPPR_MIN_SENSORS_FOR_3D", 4),
            min_sensors_for_2d=_env_int("MINIMAPPR_MIN_SENSORS_FOR_2D", 3),
            default_temperature_c=_env_float("MINIMAPPR_DEFAULT_TEMPERATURE_C", 20.0),
            default_humidity=_env_float("MINIMAPPR_DEFAULT_HUMIDITY", 0.5),
            site_origin_lat=_env_float("MINIMAPPR_SITE_ORIGIN_LAT", 37.7749),
            site_origin_lon=_env_float("MINIMAPPR_SITE_ORIGIN_LON", -122.4194),
            site_origin_alt_m=_env_float("MINIMAPPR_SITE_ORIGIN_ALT_M", 0.0),
            coordinate_mode=_env_str("MINIMAPPR_COORDINATE_MODE", "flat"),
            classifier_backend=_env_str("MINIMAPPR_CLASSIFIER", "heuristic"),
            yamnet_min_confidence=_env_float("MINIMAPPR_YAMNET_MIN_CONFIDENCE", 0.25),
            association_distance_m=_env_float("MINIMAPPR_ASSOCIATION_DISTANCE_M", 8.0),
            track_stale_seconds=_env_float("MINIMAPPR_TRACK_STALE_SECONDS", 20.0),
            tracking_filter=_env_str("MINIMAPPR_TRACKING_FILTER", "linear"),
            kalman_process_noise=_env_float("MINIMAPPR_KALMAN_PROCESS_NOISE", 2.0),
            kalman_measurement_noise=_env_float("MINIMAPPR_KALMAN_MEASUREMENT_NOISE", 1.5),
            kalman_initial_position_variance=_env_float(
                "MINIMAPPR_KALMAN_INITIAL_POSITION_VARIANCE",
                4.0,
            ),
            kalman_initial_velocity_variance=_env_float(
                "MINIMAPPR_KALMAN_INITIAL_VELOCITY_VARIANCE",
                16.0,
            ),
            fusion_worker_count=_env_int("MINIMAPPR_FUSION_WORKER_COUNT", 1),
            fusion_event_queue_size=_env_int("MINIMAPPR_FUSION_EVENT_QUEUE_SIZE", 256),
            fusion_localization_queue_size=_env_int("MINIMAPPR_FUSION_LOCALIZATION_QUEUE_SIZE", 256),
            fusion_classification_queue_size=_env_int("MINIMAPPR_FUSION_CLASSIFICATION_QUEUE_SIZE", 256),
            fusion_rules_queue_size=_env_int("MINIMAPPR_FUSION_RULES_QUEUE_SIZE", 256),
            fusion_drop_on_backpressure=_env_bool("MINIMAPPR_FUSION_DROP_ON_BACKPRESSURE", True),
            fusion_offline_replay_mode=_env_bool("MINIMAPPR_FUSION_OFFLINE_REPLAY_MODE", False),
            cleanup_interval_seconds=_env_float("MINIMAPPR_CLEANUP_INTERVAL_SECONDS", 15.0),
            node_degraded_after_seconds=_env_float("MINIMAPPR_NODE_DEGRADED_AFTER_SECONDS", 15.0),
            node_offline_after_seconds=_env_float("MINIMAPPR_NODE_OFFLINE_AFTER_SECONDS", 45.0),
            event_stale_seconds=_env_float("MINIMAPPR_EVENT_STALE_SECONDS", 30.0),
            retention_ephemeral_seconds=_env_int("MINIMAPPR_RETENTION_EPHEMERAL_SECONDS", 900),
            retention_short_seconds=_env_int("MINIMAPPR_RETENTION_SHORT_SECONDS", 86_400),
            retention_long_seconds=_env_int("MINIMAPPR_RETENTION_LONG_SECONDS", 2_592_000),
            retention_experiment_seconds=_env_int("MINIMAPPR_RETENTION_EXPERIMENT_SECONDS", 21_600),
        )
        settings.coordinate_mode = settings.coordinate_mode.strip().lower()
        if settings.coordinate_mode not in {"flat", "geodetic"}:
            raise ValueError("MINIMAPPR_COORDINATE_MODE must be 'flat' or 'geodetic'")
        if settings.node_degraded_after_seconds <= 0.0:
            raise ValueError("MINIMAPPR_NODE_DEGRADED_AFTER_SECONDS must be > 0")
        if settings.node_offline_after_seconds <= settings.node_degraded_after_seconds:
            raise ValueError("MINIMAPPR_NODE_OFFLINE_AFTER_SECONDS must be > degraded threshold")
        if settings.event_stale_seconds <= 0.0:
            raise ValueError("MINIMAPPR_EVENT_STALE_SECONDS must be > 0")
        if settings.min_sensors_for_2d < 2:
            raise ValueError("MINIMAPPR_MIN_SENSORS_FOR_2D must be >= 2")
        if settings.min_sensors_for_3d < settings.min_sensors_for_2d:
            raise ValueError("MINIMAPPR_MIN_SENSORS_FOR_3D must be >= MINIMAPPR_MIN_SENSORS_FOR_2D")
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        settings.snippet_dir.mkdir(parents=True, exist_ok=True)
        settings.large_artifact_dir.mkdir(parents=True, exist_ok=True)
        return settings

    def localization_config(self) -> LocalizationConfig:
        return LocalizationConfig(
            trigger_rms=self.trigger_rms,
            trigger_cooldown_seconds=self.trigger_cooldown_seconds,
            localization_window_seconds=self.localization_window_seconds,
            max_sensor_buffer_seconds=self.max_sensor_buffer_seconds,
            default_temperature_c=self.default_temperature_c,
            default_humidity=self.default_humidity,
            audio_highpass_hz=self.audio_highpass_hz,
            audio_lowpass_hz=self.audio_lowpass_hz,
            preprocess_enabled=self.preprocess_enabled,
            min_sensors_for_3d=self.min_sensors_for_3d,
            min_sensors_for_2d=self.min_sensors_for_2d,
        )

    def tracking_config(self) -> TrackingConfig:
        return TrackingConfig(
            association_distance_m=self.association_distance_m,
            track_stale_seconds=self.track_stale_seconds,
            tracking_filter=self.tracking_filter,
            kalman_process_noise=self.kalman_process_noise,
            kalman_measurement_noise=self.kalman_measurement_noise,
            kalman_initial_position_variance=self.kalman_initial_position_variance,
            kalman_initial_velocity_variance=self.kalman_initial_velocity_variance,
        )

    def classifier_config(self) -> ClassifierConfig:
        return ClassifierConfig(
            backend=self.classifier_backend,
            yamnet_min_confidence=self.yamnet_min_confidence,
        )

    def storage_config(self) -> StorageConfig:
        return StorageConfig(
            db_path=self.db_path,
            snippet_dir=self.snippet_dir,
            snippet_retention_seconds=self.snippet_retention_seconds,
            retention_ephemeral_seconds=self.retention_ephemeral_seconds,
            retention_short_seconds=self.retention_short_seconds,
            retention_long_seconds=self.retention_long_seconds,
            retention_experiment_seconds=self.retention_experiment_seconds,
            large_artifact_dir=self.large_artifact_dir,
        )

    def fusion_config(self) -> FusionConfig:
        return FusionConfig(
            worker_count=self.fusion_worker_count,
            event_queue_size=self.fusion_event_queue_size,
            localization_queue_size=self.fusion_localization_queue_size,
            classification_queue_size=self.fusion_classification_queue_size,
            rules_queue_size=self.fusion_rules_queue_size,
            drop_on_backpressure=self.fusion_drop_on_backpressure,
            offline_replay_mode=self.fusion_offline_replay_mode,
        )

    def rules_config(self) -> RulesConfig:
        return RulesConfig(
            rules_config_path=self.rules_config_path,
            taxonomy_config_path=self.taxonomy_config_path,
            model_chain_config_path=self.model_chain_config_path,
        )
