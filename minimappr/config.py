"""Runtime configuration for MinimapPR."""

from __future__ import annotations

import json
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
    localization_max_tau_s: float
    localization_algorithm: str
    localization_strategy: str
    localization_srp_grid_resolution_m: float
    localization_search_padding_m: float
    localization_music_azimuth_step_deg: float
    localization_music_elevation_step_deg: float
    localization_subspace_freq_min_hz: float
    localization_subspace_freq_max_hz: float
    localization_refine_confidence_threshold: float
    localization_tight_array_aperture_m: float
    beamformed_classification_enabled: bool
    beamformer_type: str
    beamformed_classification_min_sensor_count: int
    beamformed_classification_confidence_margin: float
    mvdr_diagonal_loading: float


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
class FederationPeerConfig:
    peer_id: str
    base_url: str
    api_key: str | None = None
    enabled: bool = True


@dataclass(slots=True)
class FederationConfig:
    enabled: bool
    server_id: str
    peers: tuple[FederationPeerConfig, ...]
    publish_interval_seconds: float
    heartbeat_interval_seconds: float
    link_timeout_seconds: float
    request_timeout_seconds: float
    track_ttl_seconds: float
    deconflict_mahalanobis_gate: float
    tqi_hysteresis: float
    auth_token: str | None = None


def _coerce_peer_enabled(value: object, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _normalize_base_url(value: str) -> str:
    text = value.strip()
    while text.endswith("/"):
        text = text[:-1]
    return text


def _load_federation_peers(*, raw_json: str | None, config_path: Path) -> tuple[FederationPeerConfig, ...]:
    text: str | None = None
    if raw_json is not None and raw_json.strip():
        text = raw_json
    elif config_path.exists():
        text = config_path.read_text(encoding="utf-8")
    if not text:
        return ()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        source = "MINIMAPPR_FEDERATION_PEERS_JSON" if raw_json else str(config_path)
        raise ValueError(f"Invalid federation peer JSON from {source}: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError("Federation peer config must be a JSON list")

    peers: list[FederationPeerConfig] = []
    seen_ids: set[str] = set()
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Federation peer entry at index {idx} must be an object")
        peer_id = str(item.get("peer_id", "")).strip()
        base_url = _normalize_base_url(str(item.get("base_url", "")).strip())
        if not peer_id:
            raise ValueError(f"Federation peer entry at index {idx} is missing peer_id")
        if not base_url:
            raise ValueError(f"Federation peer '{peer_id}' is missing base_url")
        if peer_id in seen_ids:
            raise ValueError(f"Federation peer_id '{peer_id}' is duplicated")
        seen_ids.add(peer_id)
        api_key_raw = item.get("api_key")
        api_key = str(api_key_raw).strip() if api_key_raw is not None else None
        if api_key == "":
            api_key = None
        peers.append(
            FederationPeerConfig(
                peer_id=peer_id,
                base_url=base_url,
                api_key=api_key,
                enabled=_coerce_peer_enabled(item.get("enabled"), True),
            )
        )
    return tuple(peers)


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
    localization_max_tau_s: float = 0.02
    localization_algorithm: str = "gcc_phat"
    localization_strategy: str = "fixed"
    localization_srp_grid_resolution_m: float = 0.5
    localization_search_padding_m: float = 2.0
    localization_music_azimuth_step_deg: float = 6.0
    localization_music_elevation_step_deg: float = 8.0
    localization_subspace_freq_min_hz: float = 300.0
    localization_subspace_freq_max_hz: float = 3500.0
    localization_refine_confidence_threshold: float = 0.45
    localization_tight_array_aperture_m: float = 0.35
    beamformed_classification_enabled: bool = False
    beamformer_type: str = "delay_and_sum"
    beamformed_classification_min_sensor_count: int = 2
    beamformed_classification_confidence_margin: float = 0.0
    mvdr_diagonal_loading: float = 1e-3

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
    federation_enabled: bool = False
    federation_server_id: str = "srv-local"
    federation_peers_config_path: Path = Path("data/federation_peers.json")
    federation_peers: tuple[FederationPeerConfig, ...] = ()
    federation_publish_interval_seconds: float = 1.0
    federation_heartbeat_interval_seconds: float = 2.0
    federation_link_timeout_seconds: float = 8.0
    federation_request_timeout_seconds: float = 2.5
    federation_track_ttl_seconds: float = 20.0
    federation_deconflict_mahalanobis_gate: float = 4.5
    federation_tqi_hysteresis: float = 0.05
    federation_auth_token: str = ""

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
            localization_max_tau_s=_env_float("MINIMAPPR_LOCALIZATION_MAX_TAU_S", 0.02),
            localization_algorithm=_env_str("MINIMAPPR_LOCALIZATION_ALGORITHM", "gcc_phat"),
            localization_strategy=_env_str("MINIMAPPR_LOCALIZATION_STRATEGY", "fixed"),
            localization_srp_grid_resolution_m=_env_float("MINIMAPPR_LOCALIZATION_SRP_GRID_RESOLUTION_M", 0.5),
            localization_search_padding_m=_env_float("MINIMAPPR_LOCALIZATION_SEARCH_PADDING_M", 2.0),
            localization_music_azimuth_step_deg=_env_float("MINIMAPPR_LOCALIZATION_MUSIC_AZ_STEP_DEG", 6.0),
            localization_music_elevation_step_deg=_env_float("MINIMAPPR_LOCALIZATION_MUSIC_EL_STEP_DEG", 8.0),
            localization_subspace_freq_min_hz=_env_float("MINIMAPPR_LOCALIZATION_SUBSPACE_FREQ_MIN_HZ", 300.0),
            localization_subspace_freq_max_hz=_env_float("MINIMAPPR_LOCALIZATION_SUBSPACE_FREQ_MAX_HZ", 3500.0),
            localization_refine_confidence_threshold=_env_float(
                "MINIMAPPR_LOCALIZATION_REFINE_CONFIDENCE_THRESHOLD",
                0.45,
            ),
            localization_tight_array_aperture_m=_env_float("MINIMAPPR_LOCALIZATION_TIGHT_ARRAY_APERTURE_M", 0.35),
            beamformed_classification_enabled=_env_bool("MINIMAPPR_BEAMFORMED_CLASSIFICATION_ENABLED", False),
            beamformer_type=_env_str("MINIMAPPR_BEAMFORMER_TYPE", "delay_and_sum"),
            beamformed_classification_min_sensor_count=_env_int(
                "MINIMAPPR_BEAMFORMED_CLASSIFICATION_MIN_SENSOR_COUNT",
                2,
            ),
            beamformed_classification_confidence_margin=_env_float(
                "MINIMAPPR_BEAMFORMED_CLASSIFICATION_CONFIDENCE_MARGIN",
                0.0,
            ),
            mvdr_diagonal_loading=_env_float("MINIMAPPR_MVDR_DIAGONAL_LOADING", 1e-3),
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
            federation_enabled=_env_bool("MINIMAPPR_FEDERATION_ENABLED", False),
            federation_server_id=_env_str("MINIMAPPR_FEDERATION_SERVER_ID", "srv-local"),
            federation_peers_config_path=Path(
                _env_str("MINIMAPPR_FEDERATION_PEERS_CONFIG_PATH", "data/federation_peers.json")
            ),
            federation_publish_interval_seconds=_env_float("MINIMAPPR_FEDERATION_PUBLISH_INTERVAL_SECONDS", 1.0),
            federation_heartbeat_interval_seconds=_env_float("MINIMAPPR_FEDERATION_HEARTBEAT_INTERVAL_SECONDS", 2.0),
            federation_link_timeout_seconds=_env_float("MINIMAPPR_FEDERATION_LINK_TIMEOUT_SECONDS", 8.0),
            federation_request_timeout_seconds=_env_float("MINIMAPPR_FEDERATION_REQUEST_TIMEOUT_SECONDS", 2.5),
            federation_track_ttl_seconds=_env_float("MINIMAPPR_FEDERATION_TRACK_TTL_SECONDS", 20.0),
            federation_deconflict_mahalanobis_gate=_env_float(
                "MINIMAPPR_FEDERATION_DECONFLICT_MAHALANOBIS_GATE",
                4.5,
            ),
            federation_tqi_hysteresis=_env_float("MINIMAPPR_FEDERATION_TQI_HYSTERESIS", 0.05),
            federation_auth_token=_env_str("MINIMAPPR_FEDERATION_AUTH_TOKEN", ""),
        )
        settings.federation_peers_config_path.parent.mkdir(parents=True, exist_ok=True)
        settings.federation_peers = _load_federation_peers(
            raw_json=os.getenv("MINIMAPPR_FEDERATION_PEERS_JSON"),
            config_path=settings.federation_peers_config_path,
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
        if settings.localization_max_tau_s <= 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_MAX_TAU_S must be > 0")
        settings.localization_algorithm = settings.localization_algorithm.strip().lower()
        if settings.localization_algorithm not in {"gcc_phat", "srp_phat", "music", "esprit"}:
            raise ValueError("MINIMAPPR_LOCALIZATION_ALGORITHM must be one of gcc_phat/srp_phat/music/esprit")
        settings.localization_strategy = settings.localization_strategy.strip().lower()
        if settings.localization_strategy not in {"fixed", "geometry_aware", "cascade"}:
            raise ValueError("MINIMAPPR_LOCALIZATION_STRATEGY must be fixed, geometry_aware, or cascade")
        if settings.localization_srp_grid_resolution_m <= 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_SRP_GRID_RESOLUTION_M must be > 0")
        if settings.localization_search_padding_m <= 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_SEARCH_PADDING_M must be > 0")
        if settings.localization_subspace_freq_min_hz <= 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_SUBSPACE_FREQ_MIN_HZ must be > 0")
        if settings.localization_subspace_freq_max_hz <= settings.localization_subspace_freq_min_hz:
            raise ValueError("MINIMAPPR_LOCALIZATION_SUBSPACE_FREQ_MAX_HZ must be > MIN frequency")
        if settings.localization_refine_confidence_threshold < 0.0 or settings.localization_refine_confidence_threshold > 1.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_REFINE_CONFIDENCE_THRESHOLD must be in [0,1]")
        if settings.localization_tight_array_aperture_m <= 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_TIGHT_ARRAY_APERTURE_M must be > 0")
        settings.beamformer_type = settings.beamformer_type.strip().lower()
        if settings.beamformer_type not in {"delay_and_sum", "mvdr"}:
            raise ValueError("MINIMAPPR_BEAMFORMER_TYPE must be delay_and_sum or mvdr")
        if settings.beamformed_classification_min_sensor_count < 1:
            raise ValueError("MINIMAPPR_BEAMFORMED_CLASSIFICATION_MIN_SENSOR_COUNT must be >= 1")
        if settings.beamformed_classification_confidence_margin < 0.0:
            raise ValueError("MINIMAPPR_BEAMFORMED_CLASSIFICATION_CONFIDENCE_MARGIN must be >= 0")
        if settings.mvdr_diagonal_loading <= 0.0:
            raise ValueError("MINIMAPPR_MVDR_DIAGONAL_LOADING must be > 0")
        settings.federation_server_id = settings.federation_server_id.strip()
        if not settings.federation_server_id:
            raise ValueError("MINIMAPPR_FEDERATION_SERVER_ID must not be empty")
        if settings.federation_publish_interval_seconds <= 0.0:
            raise ValueError("MINIMAPPR_FEDERATION_PUBLISH_INTERVAL_SECONDS must be > 0")
        if settings.federation_heartbeat_interval_seconds <= 0.0:
            raise ValueError("MINIMAPPR_FEDERATION_HEARTBEAT_INTERVAL_SECONDS must be > 0")
        if settings.federation_link_timeout_seconds <= 0.0:
            raise ValueError("MINIMAPPR_FEDERATION_LINK_TIMEOUT_SECONDS must be > 0")
        if settings.federation_request_timeout_seconds <= 0.0:
            raise ValueError("MINIMAPPR_FEDERATION_REQUEST_TIMEOUT_SECONDS must be > 0")
        if settings.federation_track_ttl_seconds <= 0.0:
            raise ValueError("MINIMAPPR_FEDERATION_TRACK_TTL_SECONDS must be > 0")
        if settings.federation_deconflict_mahalanobis_gate <= 0.0:
            raise ValueError("MINIMAPPR_FEDERATION_DECONFLICT_MAHALANOBIS_GATE must be > 0")
        if settings.federation_tqi_hysteresis < 0.0:
            raise ValueError("MINIMAPPR_FEDERATION_TQI_HYSTERESIS must be >= 0")
        for peer in settings.federation_peers:
            if peer.peer_id == settings.federation_server_id:
                raise ValueError("Federation peer_id cannot match MINIMAPPR_FEDERATION_SERVER_ID")
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
            localization_max_tau_s=self.localization_max_tau_s,
            localization_algorithm=self.localization_algorithm,
            localization_strategy=self.localization_strategy,
            localization_srp_grid_resolution_m=self.localization_srp_grid_resolution_m,
            localization_search_padding_m=self.localization_search_padding_m,
            localization_music_azimuth_step_deg=self.localization_music_azimuth_step_deg,
            localization_music_elevation_step_deg=self.localization_music_elevation_step_deg,
            localization_subspace_freq_min_hz=self.localization_subspace_freq_min_hz,
            localization_subspace_freq_max_hz=self.localization_subspace_freq_max_hz,
            localization_refine_confidence_threshold=self.localization_refine_confidence_threshold,
            localization_tight_array_aperture_m=self.localization_tight_array_aperture_m,
            beamformed_classification_enabled=self.beamformed_classification_enabled,
            beamformer_type=self.beamformer_type,
            beamformed_classification_min_sensor_count=self.beamformed_classification_min_sensor_count,
            beamformed_classification_confidence_margin=self.beamformed_classification_confidence_margin,
            mvdr_diagonal_loading=self.mvdr_diagonal_loading,
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

    def federation_config(self) -> FederationConfig:
        token = self.federation_auth_token.strip()
        return FederationConfig(
            enabled=self.federation_enabled and bool(self.federation_peers),
            server_id=self.federation_server_id,
            peers=self.federation_peers,
            publish_interval_seconds=self.federation_publish_interval_seconds,
            heartbeat_interval_seconds=self.federation_heartbeat_interval_seconds,
            link_timeout_seconds=self.federation_link_timeout_seconds,
            request_timeout_seconds=self.federation_request_timeout_seconds,
            track_ttl_seconds=self.federation_track_ttl_seconds,
            deconflict_mahalanobis_gate=self.federation_deconflict_mahalanobis_gate,
            tqi_hysteresis=self.federation_tqi_hysteresis,
            auth_token=token or None,
        )
