"""Runtime configuration for MinimapPR."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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
class Settings:
    host: str = "0.0.0.0"
    port: int = 8080
    db_path: Path = Path("data/minimappr.db")
    snippet_dir: Path = Path("data/snippets")
    snippet_retention_seconds: int = 3600

    trigger_rms: float = 0.015
    trigger_cooldown_seconds: float = 0.8
    localization_window_seconds: float = 0.08
    max_sensor_buffer_seconds: float = 8.0

    default_temperature_c: float = 20.0
    default_humidity: float = 0.5

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

    cleanup_interval_seconds: float = 15.0

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            host=_env_str("MINIMAPPR_HOST", "0.0.0.0"),
            port=_env_int("MINIMAPPR_PORT", 8080),
            db_path=Path(_env_str("MINIMAPPR_DB_PATH", "data/minimappr.db")),
            snippet_dir=Path(_env_str("MINIMAPPR_SNIPPET_DIR", "data/snippets")),
            snippet_retention_seconds=_env_int("MINIMAPPR_SNIPPET_RETENTION_SECONDS", 3600),
            trigger_rms=_env_float("MINIMAPPR_TRIGGER_RMS", 0.015),
            trigger_cooldown_seconds=_env_float("MINIMAPPR_TRIGGER_COOLDOWN_SECONDS", 0.8),
            localization_window_seconds=_env_float("MINIMAPPR_LOCALIZATION_WINDOW_SECONDS", 0.08),
            max_sensor_buffer_seconds=_env_float("MINIMAPPR_MAX_SENSOR_BUFFER_SECONDS", 8.0),
            default_temperature_c=_env_float("MINIMAPPR_DEFAULT_TEMPERATURE_C", 20.0),
            default_humidity=_env_float("MINIMAPPR_DEFAULT_HUMIDITY", 0.5),
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
            cleanup_interval_seconds=_env_float("MINIMAPPR_CLEANUP_INTERVAL_SECONDS", 15.0),
        )
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        settings.snippet_dir.mkdir(parents=True, exist_ok=True)
        return settings
