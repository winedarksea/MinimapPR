"""Runtime configuration for MinimapPR."""

from __future__ import annotations

import json
import math
import os
import platform
from dataclasses import dataclass
from pathlib import Path


DEFAULT_RULES_CONFIG_PATH = Path("data/rules.json")
DEFAULT_BIRDNET_HYBRID_RULES_CONFIG_PATH = Path("data/rules_birdnet_hybrid_production.json")


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


def _env_list(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(key)
    if raw is None:
        return default
    text = raw.strip()
    if not text:
        return default

    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            values = [str(item).strip() for item in parsed if str(item).strip()]
            return tuple(values) if values else default

    values = [item.strip() for item in text.split(",") if item.strip()]
    return tuple(values) if values else default


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
    ingest_gain_multiplier: float
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
    classifier_diagonal_loading_scale: float
    pre_classification_highpass_hz: float
    pre_classification_lowpass_hz: float
    gcc_phat_interp_factor: int
    classification_window_seconds: float = 30.0
    localization_band_min_hz: float = 0.0
    localization_band_max_hz: float = 0.0
    skip_localization_for_classification: bool = False
    cluster_aware_localization: bool = False
    wavelength_gating_enabled: bool = True
    wavelength_penalty_floor: float = 0.25


@dataclass(slots=True)
class TrackingConfig:
    association_distance_m: float
    track_stale_seconds: float
    tracking_filter: str
    kalman_process_noise: float
    kalman_measurement_noise: float
    kalman_initial_position_variance: float
    kalman_initial_velocity_variance: float
    linear_position_alpha: float
    linear_velocity_alpha: float
    tqi_weight_confidence: float
    tqi_weight_corroboration: float
    tqi_weight_recency: float
    tqi_weight_sensor: float
    track_drop_multiplier: float
    track_reap_multiplier: float


@dataclass(slots=True)
class ClassifierConfig:
    backend: str
    stage_timeout_seconds: float
    yamnet_min_confidence: float
    yamnet_input_target_rms: float
    yamnet_max_input_gain: float
    birdnet_trigger_min_confidence: float
    birdnet_geo_min_confidence: float
    heuristic_ambient_rms_threshold: float
    heuristic_impulse_crest_threshold: float
    heuristic_impulse_bandwidth_threshold_hz: float
    heuristic_bird_centroid_min_hz: float
    heuristic_bird_zcr_min: float
    heuristic_speech_centroid_min_hz: float
    heuristic_speech_centroid_max_hz: float
    heuristic_speech_zcr_min: float
    heuristic_speech_zcr_max: float
    heuristic_speech_flatness_max: float
    heuristic_machine_centroid_max_hz: float
    heuristic_machine_flatness_max: float
    heuristic_unknown_min_score: float
    heuristic_unknown_score: float


@dataclass(slots=True)
class StorageConfig:
    db_path: Path
    snippet_dir: Path
    snippet_retention_seconds: int
    retention_policy_path: Path
    retention_ephemeral_seconds: int
    retention_short_seconds: int
    retention_long_seconds: int
    retention_experiment_seconds: int
    retention_ingested_frames_seconds: int
    retention_bit_reports_seconds: int
    retention_pings_seconds: int
    retention_track_updates_seconds: int
    retention_alerts_seconds: int
    retention_environment_seconds: int
    retention_dropped_tracks_seconds: int
    large_artifact_dir: Path


@dataclass(slots=True)
class FusionConfig:
    worker_count: int
    event_queue_size: int
    localization_queue_size: int
    classification_queue_size: int
    rules_queue_size: int
    birdnet_chunked_dispatch_enabled: bool
    birdnet_chunk_overlap_seconds: float
    birdnet_chunk_max_retries_per_chunk: int
    birdnet_chunk_min_retry_progress_seconds: float
    birdnet_chunk_retry_on_classifier_error: bool
    drop_on_backpressure: bool
    offline_replay_mode: bool
    sensor_energy_threshold_multiplier: float
    fallback_localization_confidence: float
    reporting_window_seconds: float
    taxonomy_refresh_interval_seconds: float
    retention_permanent_labels: tuple[str, ...]
    retention_long_security_confidence: float


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
    deconflict_use_3d: bool = False
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
    runtime_profile: str = "default"
    process_role: str = "combined"
    host: str = "0.0.0.0"
    port: int = 8080
    ingest_host: str = "0.0.0.0"
    ingest_port: int = 8081
    ingest_base_url: str = ""
    ingest_backend: str = "python"
    db_path: Path = Path("data/minimappr.db")
    snippet_dir: Path = Path("data/snippets")
    snippet_retention_seconds: int = 3600
    ingest_spool_dir: Path = Path("data/spool")
    ingest_spool_ready_ttl_seconds: float = 60.0
    ingest_spool_failed_ttl_seconds: float = 86_400.0
    ingest_spool_tmp_ttl_seconds: float = 300.0
    ingest_spool_poll_interval_seconds: float = 0.05
    ingest_spool_worker_count: int = 1
    ingest_storage_mode: str = "spool"
    ingest_consumer_name: str = "python-ingest"
    direct_ingest_enabled: bool = True
    ingest_sidecar_enabled: bool = True
    ingest_sidecar_binary_path: Path = Path("dist/minimappr-ingest-sidecar")
    ingest_sidecar_port: int = 8081
    ingest_sidecar_memory_only_live_path: bool = True
    ingest_sidecar_total_journal_budget_bytes: int = 268_435_456
    ingest_sidecar_admission_reserve_bytes: int = 16_777_216
    ingest_sidecar_allow_non_tmpfs_journal: bool | None = None
    persist_observations_on_ingest: bool | None = None
    retention_policy_path: Path = Path("data/retention_policy.json")
    rules_config_path: Path = DEFAULT_RULES_CONFIG_PATH
    taxonomy_config_path: Path = Path("data/taxonomy.json")
    model_chain_config_path: Path = Path("data/model_chain.json")
    large_artifact_dir: Path = Path("data/artifacts")
    capture_final_tracks_settle_seconds: float = 30.0
    cors_allow_origins: tuple[str, ...] = ("http://localhost:8080", "http://127.0.0.1:8080")
    cors_allow_credentials: bool = False

    trigger_rms: float = 0.015
    trigger_cooldown_seconds: float = 0.8
    localization_window_seconds: float = 0.08
    classification_window_seconds: float = 30.0
    max_sensor_buffer_seconds: float = 32.0
    preprocess_enabled: bool = True
    ingest_gain_multiplier: float = 1.0
    audio_highpass_hz: float = 50.0
    audio_lowpass_hz: float = 0.0
    min_sensors_for_3d: int = 4
    min_sensors_for_2d: int = 3
    localization_max_tau_s: float = 0.02
    localization_algorithm: str = "gcc_phat"
    localization_strategy: str = "fixed"
    localization_band_min_hz: float = 0.0
    localization_band_max_hz: float = 0.0
    localization_srp_grid_resolution_m: float = 0.5
    localization_search_padding_m: float = 2.0
    localization_music_azimuth_step_deg: float = 6.0
    localization_music_elevation_step_deg: float = 8.0
    localization_subspace_freq_min_hz: float = 300.0
    localization_subspace_freq_max_hz: float = 3500.0
    localization_refine_confidence_threshold: float = 0.45
    localization_tight_array_aperture_m: float = 0.35
    wavelength_gating_enabled: bool = True
    wavelength_penalty_floor: float = 0.25
    skip_localization_for_classification: bool = False
    cluster_aware_localization: bool = False
    beamformed_classification_enabled: bool = False
    beamformer_type: str = "delay_and_sum"
    beamformed_classification_min_sensor_count: int = 2
    beamformed_classification_confidence_margin: float = 0.0
    mvdr_diagonal_loading: float = 1e-3
    classifier_diagonal_loading_scale: float = 10.0
    classification_stage_timeout_seconds: float = 30.0
    pre_classification_highpass_hz: float = 0.0
    pre_classification_lowpass_hz: float = 0.0
    gcc_phat_interp_factor: int = 4

    default_temperature_c: float = 20.0
    default_humidity: float = 0.5
    environment_reading_max_age_seconds: float = 300.0
    site_origin_source: str = "auto"
    site_origin_reconcile_delay_seconds: float = 30.0
    site_origin_lat: float = 44.98698840878797
    site_origin_lon: float = -93.2579197515542
    site_origin_alt_m: float = 0.0
    coordinate_mode: str = "flat"

    classifier_backend: str = "yamnet"
    yamnet_min_confidence: float = 0.25
    yamnet_input_target_rms: float = 0.10
    yamnet_max_input_gain: float = 32.0
    birdnet_trigger_min_confidence: float = 0.40
    birdnet_geo_min_confidence: float = 0.03
    detection_min_confidence: float = 0.4
    cop_detections_max_items: int = 150
    cop_tracks_max_items: int = 150
    cop_detections_max_age_seconds: float = 86_400.0
    cop_tracks_max_age_seconds: float = 86_400.0
    heuristic_ambient_rms_threshold: float = 0.01
    heuristic_impulse_crest_threshold: float = 10.0
    heuristic_impulse_bandwidth_threshold_hz: float = 1200.0
    heuristic_bird_centroid_min_hz: float = 2200.0
    heuristic_bird_zcr_min: float = 0.12
    heuristic_speech_centroid_min_hz: float = 200.0
    heuristic_speech_centroid_max_hz: float = 2200.0
    heuristic_speech_zcr_min: float = 0.04
    heuristic_speech_zcr_max: float = 0.2
    heuristic_speech_flatness_max: float = 0.75
    heuristic_machine_centroid_max_hz: float = 450.0
    heuristic_machine_flatness_max: float = 0.55
    heuristic_unknown_min_score: float = 0.2
    heuristic_unknown_score: float = 0.6

    association_distance_m: float = 8.0
    track_stale_seconds: float = 20.0
    tracking_filter: str = "linear"
    kalman_process_noise: float = 2.0
    kalman_measurement_noise: float = 1.5
    kalman_initial_position_variance: float = 4.0
    kalman_initial_velocity_variance: float = 16.0
    linear_position_alpha: float = 0.4
    linear_velocity_alpha: float = 0.5
    tqi_weight_confidence: float = 0.3
    tqi_weight_corroboration: float = 0.3
    tqi_weight_recency: float = 0.2
    tqi_weight_sensor: float = 0.2
    track_drop_multiplier: float = 3.0
    track_reap_multiplier: float = 5.0

    fusion_worker_count: int = 1
    fusion_event_queue_size: int = 256
    fusion_localization_queue_size: int = 256
    fusion_classification_queue_size: int = 256
    fusion_rules_queue_size: int = 256
    birdnet_chunked_dispatch_enabled: bool = False
    birdnet_chunk_overlap_seconds: float = 2.0
    birdnet_chunk_max_retries_per_chunk: int = 1
    birdnet_chunk_min_retry_progress_seconds: float = 8.0
    birdnet_chunk_retry_on_classifier_error: bool = False
    fusion_drop_on_backpressure: bool = True
    fusion_offline_replay_mode: bool = False
    sensor_energy_threshold_multiplier: float = 0.45
    fallback_localization_confidence: float = 0.25
    reporting_window_seconds: float = 30.0
    taxonomy_refresh_interval_seconds: float = 10.0
    retention_permanent_labels: tuple[str, ...] = ("gunshot", "explosion", "artillery", "fusillade")
    retention_long_security_confidence: float = 0.6

    cleanup_interval_seconds: float = 15.0
    node_degraded_after_seconds: float = 15.0
    node_offline_after_seconds: float = 45.0
    event_stale_seconds: float = 30.0
    retention_ephemeral_seconds: int = 900
    retention_short_seconds: int = 86_400
    retention_long_seconds: int = 2_592_000
    retention_experiment_seconds: int = 21_600
    retention_ingested_frames_seconds: int = 86_400
    retention_bit_reports_seconds: int = 604_800
    retention_pings_seconds: int = 86_400
    retention_track_updates_seconds: int = 604_800
    retention_alerts_seconds: int = 2_592_000
    retention_environment_seconds: int = 604_800
    retention_dropped_tracks_seconds: int = 604_800
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
    federation_deconflict_use_3d: bool = False
    federation_auth_token: str = ""

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)
        self.snippet_dir = Path(self.snippet_dir)
        self.ingest_spool_dir = Path(self.ingest_spool_dir)
        self.retention_policy_path = Path(self.retention_policy_path)
        self.rules_config_path = Path(self.rules_config_path)
        self.taxonomy_config_path = Path(self.taxonomy_config_path)
        self.model_chain_config_path = Path(self.model_chain_config_path)
        self.large_artifact_dir = Path(self.large_artifact_dir)
        self.federation_peers_config_path = Path(self.federation_peers_config_path)

        self.coordinate_mode = self.coordinate_mode.strip().lower()
        if self.coordinate_mode not in {"flat", "geodetic"}:
            raise ValueError("MINIMAPPR_COORDINATE_MODE must be 'flat' or 'geodetic'")
        self.runtime_profile = self.runtime_profile.strip().lower() or "default"
        self.process_role = self.process_role.strip().lower() or "combined"
        if self.process_role not in {"combined", "api", "ingest"}:
            raise ValueError("MINIMAPPR_PROCESS_ROLE must be one of combined/api/ingest")
        self.ingest_backend = self.ingest_backend.strip().lower() or "python"
        if self.ingest_backend not in {"python", "rust"}:
            raise ValueError("MINIMAPPR_INGEST_BACKEND must be one of python/rust")
        if self.ingest_port <= 0 or self.ingest_port > 65535:
            raise ValueError("MINIMAPPR_INGEST_PORT must be in [1, 65535]")
        if not self.ingest_base_url:
            self.ingest_base_url = f"http://127.0.0.1:{self.ingest_port}"
        self.ingest_sidecar_port = self.ingest_port
        self._apply_runtime_profile()
        if self.ingest_sidecar_allow_non_tmpfs_journal is None:
            self.ingest_sidecar_allow_non_tmpfs_journal = platform.system() != "Linux"
        if self.persist_observations_on_ingest is None:
            # BirdNET production favors real-time ingest and contiguous detection
            # snippets over dense raw-observation provenance at ingest time.
            # Persisting one observation row per sensor per frame amplifies DB I/O
            # and can starve HTTP ingest under sustained edge publish load.
            self.persist_observations_on_ingest = self.runtime_profile != "birdnet_hybrid_production"

        if self.node_degraded_after_seconds <= 0.0:
            raise ValueError("MINIMAPPR_NODE_DEGRADED_AFTER_SECONDS must be > 0")
        if self.node_offline_after_seconds <= self.node_degraded_after_seconds:
            raise ValueError("MINIMAPPR_NODE_OFFLINE_AFTER_SECONDS must be > degraded threshold")
        if self.event_stale_seconds <= 0.0:
            raise ValueError("MINIMAPPR_EVENT_STALE_SECONDS must be > 0")
        if self.cleanup_interval_seconds <= 0.0:
            raise ValueError("MINIMAPPR_CLEANUP_INTERVAL_SECONDS must be > 0")
        if self.capture_final_tracks_settle_seconds < 0.0:
            raise ValueError("MINIMAPPR_CAPTURE_FINAL_TRACKS_SETTLE_SECONDS must be >= 0")
        if self.ingest_spool_ready_ttl_seconds < 0.0:
            raise ValueError("MINIMAPPR_INGEST_SPOOL_READY_TTL_SECONDS must be >= 0")
        if self.ingest_spool_failed_ttl_seconds < 0.0:
            raise ValueError("MINIMAPPR_INGEST_SPOOL_FAILED_TTL_SECONDS must be >= 0")
        if self.ingest_spool_tmp_ttl_seconds < 0.0:
            raise ValueError("MINIMAPPR_INGEST_SPOOL_TMP_TTL_SECONDS must be >= 0")
        if self.ingest_spool_poll_interval_seconds <= 0.0:
            raise ValueError("MINIMAPPR_INGEST_SPOOL_POLL_INTERVAL_SECONDS must be > 0")
        if self.ingest_spool_worker_count < 1:
            raise ValueError("MINIMAPPR_INGEST_SPOOL_WORKER_COUNT must be >= 1")
        if not self.ingest_consumer_name.strip():
            raise ValueError("MINIMAPPR_INGEST_CONSUMER_NAME must not be blank")
        if self.process_role == "ingest" and self.ingest_backend == "rust":
            raise ValueError("MINIMAPPR_PROCESS_ROLE=ingest is only valid with MINIMAPPR_INGEST_BACKEND=python")
        if self.process_role == "api" and self.direct_ingest_enabled:
            self.direct_ingest_enabled = False
        if self.ingest_backend == "rust" and self.direct_ingest_enabled and self.process_role != "combined":
            raise ValueError("Rust ingest backend cannot run with direct Python ingest enabled in split mode")
        if self.ingest_sidecar_total_journal_budget_bytes < 0:
            raise ValueError("MINIMAPPR_SIDECAR_TOTAL_JOURNAL_BUDGET_BYTES must be >= 0")
        if self.ingest_sidecar_admission_reserve_bytes < 0:
            raise ValueError("MINIMAPPR_SIDECAR_ADMISSION_RESERVE_BYTES must be >= 0")
        if (
            self.ingest_sidecar_total_journal_budget_bytes > 0
            and self.ingest_sidecar_admission_reserve_bytes >= self.ingest_sidecar_total_journal_budget_bytes
        ):
            raise ValueError(
                "MINIMAPPR_SIDECAR_ADMISSION_RESERVE_BYTES must be smaller than "
                "MINIMAPPR_SIDECAR_TOTAL_JOURNAL_BUDGET_BYTES"
            )
        if self.min_sensors_for_2d < 2:
            raise ValueError("MINIMAPPR_MIN_SENSORS_FOR_2D must be >= 2")
        if self.min_sensors_for_3d < self.min_sensors_for_2d:
            raise ValueError("MINIMAPPR_MIN_SENSORS_FOR_3D must be >= MINIMAPPR_MIN_SENSORS_FOR_2D")
        if self.environment_reading_max_age_seconds < 0.0:
            raise ValueError("MINIMAPPR_ENVIRONMENT_READING_MAX_AGE_SECONDS must be >= 0")
        self.site_origin_source = self.site_origin_source.strip().lower() or "auto"
        if self.site_origin_source not in {"auto", "manual"}:
            raise ValueError("MINIMAPPR_SITE_ORIGIN_SOURCE must be 'auto' or 'manual'")
        if self.site_origin_reconcile_delay_seconds < 0.0:
            raise ValueError("MINIMAPPR_SITE_ORIGIN_RECONCILE_DELAY_SECONDS must be >= 0")
        if not math.isfinite(self.ingest_gain_multiplier) or self.ingest_gain_multiplier <= 0.0:
            raise ValueError("MINIMAPPR_INGEST_GAIN_MULTIPLIER must be finite and > 0")

        if self.localization_max_tau_s <= 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_MAX_TAU_S must be > 0")
        if self.localization_window_seconds <= 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_WINDOW_SECONDS must be > 0")
        if self.classification_window_seconds <= 0.0:
            self.classification_window_seconds = self.localization_window_seconds
        if self.classification_window_seconds < self.localization_window_seconds:
            raise ValueError(
                "MINIMAPPR_CLASSIFICATION_WINDOW_SECONDS must be >= MINIMAPPR_LOCALIZATION_WINDOW_SECONDS"
            )
        if self.max_sensor_buffer_seconds < self.classification_window_seconds:
            raise ValueError(
                "MINIMAPPR_MAX_SENSOR_BUFFER_SECONDS must be >= MINIMAPPR_CLASSIFICATION_WINDOW_SECONDS"
            )
        self.localization_algorithm = self.localization_algorithm.strip().lower()
        if self.localization_algorithm not in {"gcc_phat", "srp_phat", "music", "esprit"}:
            raise ValueError("MINIMAPPR_LOCALIZATION_ALGORITHM must be one of gcc_phat/srp_phat/music/esprit")
        self.localization_strategy = self.localization_strategy.strip().lower()
        if self.localization_strategy not in {"fixed", "geometry_aware", "cascade"}:
            raise ValueError("MINIMAPPR_LOCALIZATION_STRATEGY must be fixed, geometry_aware, or cascade")
        if self.localization_band_min_hz < 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_BAND_MIN_HZ must be >= 0")
        if self.localization_band_max_hz < 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_BAND_MAX_HZ must be >= 0")
        if self.localization_band_max_hz > 0.0 and self.localization_band_max_hz <= self.localization_band_min_hz:
            raise ValueError("MINIMAPPR_LOCALIZATION_BAND_MAX_HZ must be > MIN when enabled")
        if self.localization_srp_grid_resolution_m <= 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_SRP_GRID_RESOLUTION_M must be > 0")
        if self.localization_search_padding_m <= 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_SEARCH_PADDING_M must be > 0")
        if self.localization_subspace_freq_min_hz <= 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_SUBSPACE_FREQ_MIN_HZ must be > 0")
        if self.localization_subspace_freq_max_hz <= self.localization_subspace_freq_min_hz:
            raise ValueError("MINIMAPPR_LOCALIZATION_SUBSPACE_FREQ_MAX_HZ must be > MIN frequency")
        if self.localization_refine_confidence_threshold < 0.0 or self.localization_refine_confidence_threshold > 1.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_REFINE_CONFIDENCE_THRESHOLD must be in [0,1]")
        if self.localization_tight_array_aperture_m <= 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_TIGHT_ARRAY_APERTURE_M must be > 0")
        if self.wavelength_penalty_floor < 0.0 or self.wavelength_penalty_floor > 1.0:
            raise ValueError("MINIMAPPR_WAVELENGTH_PENALTY_FLOOR must be in [0,1]")
        if self.gcc_phat_interp_factor < 1:
            raise ValueError("MINIMAPPR_GCC_PHAT_INTERP_FACTOR must be >= 1")
        self.beamformer_type = self.beamformer_type.strip().lower()
        if self.beamformer_type == "das":
            # Preserve the more descriptive config name internally while
            # still accepting the historical shorthand.
            self.beamformer_type = "delay_and_sum"
        _valid_beamformers = {"delay_and_sum", "das", "freq_domain_das", "mvdr", "superdirective", "gevd"}
        if self.beamformer_type not in _valid_beamformers:
            raise ValueError(
                f"MINIMAPPR_BEAMFORMER_TYPE must be one of {sorted(_valid_beamformers)}"
            )
        if self.classifier_diagonal_loading_scale < 1.0:
            raise ValueError("MINIMAPPR_CLASSIFIER_DIAGONAL_LOADING_SCALE must be >= 1.0")
        if self.classification_stage_timeout_seconds <= 0.0:
            raise ValueError("MINIMAPPR_CLASSIFICATION_STAGE_TIMEOUT_SECONDS must be > 0")
        if self.pre_classification_highpass_hz < 0.0:
            raise ValueError("MINIMAPPR_PRE_CLASSIFICATION_HIGHPASS_HZ must be >= 0")
        if self.pre_classification_lowpass_hz < 0.0:
            raise ValueError("MINIMAPPR_PRE_CLASSIFICATION_LOWPASS_HZ must be >= 0")
        if self.beamformed_classification_min_sensor_count < 1:
            raise ValueError("MINIMAPPR_BEAMFORMED_CLASSIFICATION_MIN_SENSOR_COUNT must be >= 1")
        if self.beamformed_classification_confidence_margin < 0.0:
            raise ValueError("MINIMAPPR_BEAMFORMED_CLASSIFICATION_CONFIDENCE_MARGIN must be >= 0")
        if self.mvdr_diagonal_loading <= 0.0:
            raise ValueError("MINIMAPPR_MVDR_DIAGONAL_LOADING must be > 0")
        if self.sensor_energy_threshold_multiplier <= 0.0:
            raise ValueError("MINIMAPPR_SENSOR_ENERGY_THRESHOLD_MULTIPLIER must be > 0")
        if self.fallback_localization_confidence < 0.0 or self.fallback_localization_confidence > 1.0:
            raise ValueError("MINIMAPPR_FALLBACK_LOCALIZATION_CONFIDENCE must be in [0,1]")
        if self.reporting_window_seconds <= 0.0:
            raise ValueError("MINIMAPPR_REPORTING_WINDOW_SECONDS must be > 0")
        if self.birdnet_chunk_overlap_seconds < 0.0:
            raise ValueError("MINIMAPPR_BIRDNET_CHUNK_OVERLAP_SECONDS must be >= 0")
        if self.birdnet_chunk_max_retries_per_chunk < 0:
            raise ValueError("MINIMAPPR_BIRDNET_CHUNK_MAX_RETRIES_PER_CHUNK must be >= 0")
        if self.birdnet_chunk_min_retry_progress_seconds < 0.0:
            raise ValueError("MINIMAPPR_BIRDNET_CHUNK_MIN_RETRY_PROGRESS_SECONDS must be >= 0")
        if (
            self.birdnet_chunked_dispatch_enabled
            and self.classifier_backend == "birdnet"
            and self.birdnet_chunk_overlap_seconds >= self.classification_window_seconds
        ):
            raise ValueError(
                "MINIMAPPR_BIRDNET_CHUNK_OVERLAP_SECONDS must be < MINIMAPPR_CLASSIFICATION_WINDOW_SECONDS"
            )
        if (
            self.birdnet_chunked_dispatch_enabled
            and self.classifier_backend == "birdnet"
            and self.birdnet_chunk_overlap_seconds > 2.0
        ):
            raise ValueError(
                "MINIMAPPR_BIRDNET_CHUNK_OVERLAP_SECONDS must be <= 2.0 for BirdNET chunked dispatch"
            )
        if self.taxonomy_refresh_interval_seconds <= 0.0:
            raise ValueError("MINIMAPPR_TAXONOMY_REFRESH_INTERVAL_SECONDS must be > 0")
        if self.retention_long_security_confidence < 0.0 or self.retention_long_security_confidence > 1.0:
            raise ValueError("MINIMAPPR_RETENTION_LONG_SECURITY_CONFIDENCE must be in [0,1]")
        if self.yamnet_min_confidence < 0.0 or self.yamnet_min_confidence > 1.0:
            raise ValueError("MINIMAPPR_YAMNET_MIN_CONFIDENCE must be in [0,1]")
        if not math.isfinite(self.yamnet_input_target_rms) or self.yamnet_input_target_rms <= 0.0:
            raise ValueError("MINIMAPPR_YAMNET_INPUT_TARGET_RMS must be finite and > 0")
        if not math.isfinite(self.yamnet_max_input_gain) or self.yamnet_max_input_gain <= 0.0:
            raise ValueError("MINIMAPPR_YAMNET_MAX_INPUT_GAIN must be finite and > 0")
        if self.birdnet_trigger_min_confidence < 0.0 or self.birdnet_trigger_min_confidence > 1.0:
            raise ValueError("MINIMAPPR_BIRDNET_TRIGGER_MIN_CONFIDENCE must be in [0,1]")
        if self.birdnet_geo_min_confidence < 0.0 or self.birdnet_geo_min_confidence > 1.0:
            raise ValueError("MINIMAPPR_BIRDNET_GEO_MIN_CONFIDENCE must be in [0,1]")
        if self.detection_min_confidence < 0.0 or self.detection_min_confidence > 1.0:
            raise ValueError("MINIMAPPR_DETECTION_MIN_CONFIDENCE must be in [0,1]")
        if self.cop_detections_max_items < 1:
            raise ValueError("MINIMAPPR_COP_DETECTIONS_MAX_ITEMS must be >= 1")
        if self.cop_tracks_max_items < 1:
            raise ValueError("MINIMAPPR_COP_TRACKS_MAX_ITEMS must be >= 1")
        if self.cop_detections_max_age_seconds <= 0.0:
            raise ValueError("MINIMAPPR_COP_DETECTIONS_MAX_AGE_SECONDS must be > 0")
        if self.cop_tracks_max_age_seconds <= 0.0:
            raise ValueError("MINIMAPPR_COP_TRACKS_MAX_AGE_SECONDS must be > 0")

        self.cors_allow_origins = tuple(origin.strip() for origin in self.cors_allow_origins if origin.strip())
        if not self.cors_allow_origins:
            raise ValueError("MINIMAPPR_CORS_ALLOW_ORIGINS must include at least one origin")
        if "*" in self.cors_allow_origins and self.cors_allow_credentials:
            raise ValueError("CORS allow_credentials cannot be true when allow_origins includes '*'")

        self.retention_permanent_labels = tuple(
            sorted({label.strip().lower() for label in self.retention_permanent_labels if label.strip()})
        )
        if not self.retention_permanent_labels:
            raise ValueError("MINIMAPPR_RETENTION_PERMANENT_LABELS must include at least one label")
        for field_name in (
            "retention_ephemeral_seconds",
            "retention_short_seconds",
            "retention_long_seconds",
            "retention_experiment_seconds",
            "retention_ingested_frames_seconds",
            "retention_bit_reports_seconds",
            "retention_pings_seconds",
            "retention_track_updates_seconds",
            "retention_alerts_seconds",
            "retention_environment_seconds",
            "retention_dropped_tracks_seconds",
        ):
            value = getattr(self, field_name)
            if value < -1:
                raise ValueError(f"{field_name} must be >= -1")

        if self.kalman_process_noise < 0.0:
            raise ValueError("kalman_process_noise must be >= 0 (MINIMAPPR_KALMAN_PROCESS_NOISE)")
        if self.kalman_measurement_noise <= 0.0:
            raise ValueError("kalman_measurement_noise must be > 0 (MINIMAPPR_KALMAN_MEASUREMENT_NOISE)")
        if self.kalman_initial_position_variance <= 0.0:
            raise ValueError("kalman_initial_position_variance must be > 0 (MINIMAPPR_KALMAN_INITIAL_POSITION_VARIANCE)")
        if self.kalman_initial_velocity_variance <= 0.0:
            raise ValueError("kalman_initial_velocity_variance must be > 0 (MINIMAPPR_KALMAN_INITIAL_VELOCITY_VARIANCE)")
        if not (0.0 <= self.linear_position_alpha <= 1.0):
            raise ValueError("MINIMAPPR_LINEAR_POSITION_ALPHA must be in [0,1]")
        if not (0.0 <= self.linear_velocity_alpha <= 1.0):
            raise ValueError("MINIMAPPR_LINEAR_VELOCITY_ALPHA must be in [0,1]")
        if self.track_drop_multiplier <= 1.0:
            raise ValueError("MINIMAPPR_TRACK_DROP_MULTIPLIER must be > 1")
        if self.track_reap_multiplier <= self.track_drop_multiplier:
            raise ValueError("MINIMAPPR_TRACK_REAP_MULTIPLIER must be > MINIMAPPR_TRACK_DROP_MULTIPLIER")
        tqi_weights = (
            self.tqi_weight_confidence,
            self.tqi_weight_corroboration,
            self.tqi_weight_recency,
            self.tqi_weight_sensor,
        )
        if any(weight < 0.0 for weight in tqi_weights):
            raise ValueError("TQI weights must be >= 0")
        if sum(tqi_weights) <= 0.0:
            raise ValueError("TQI weights must sum to > 0")

        self.federation_server_id = self.federation_server_id.strip()
        if not self.federation_server_id:
            raise ValueError("MINIMAPPR_FEDERATION_SERVER_ID must not be empty")
        if self.federation_publish_interval_seconds <= 0.0:
            raise ValueError("MINIMAPPR_FEDERATION_PUBLISH_INTERVAL_SECONDS must be > 0")
        if self.federation_heartbeat_interval_seconds <= 0.0:
            raise ValueError("MINIMAPPR_FEDERATION_HEARTBEAT_INTERVAL_SECONDS must be > 0")
        if self.federation_link_timeout_seconds <= 0.0:
            raise ValueError("MINIMAPPR_FEDERATION_LINK_TIMEOUT_SECONDS must be > 0")
        if self.federation_request_timeout_seconds <= 0.0:
            raise ValueError("MINIMAPPR_FEDERATION_REQUEST_TIMEOUT_SECONDS must be > 0")
        if self.federation_track_ttl_seconds <= 0.0:
            raise ValueError("MINIMAPPR_FEDERATION_TRACK_TTL_SECONDS must be > 0")
        if self.federation_deconflict_mahalanobis_gate <= 0.0:
            raise ValueError("MINIMAPPR_FEDERATION_DECONFLICT_MAHALANOBIS_GATE must be > 0")
        if self.federation_tqi_hysteresis < 0.0:
            raise ValueError("MINIMAPPR_FEDERATION_TQI_HYSTERESIS must be >= 0")
        for peer in self.federation_peers:
            if peer.peer_id == self.federation_server_id:
                raise ValueError("Federation peer_id cannot match MINIMAPPR_FEDERATION_SERVER_ID")

    @classmethod
    def from_env(cls) -> "Settings":
        peers_config_path = Path(_env_str("MINIMAPPR_FEDERATION_PEERS_CONFIG_PATH", "data/federation_peers.json"))
        peers = _load_federation_peers(
            raw_json=os.getenv("MINIMAPPR_FEDERATION_PEERS_JSON"),
            config_path=peers_config_path,
        )
        allow_non_tmpfs_journal_raw = os.getenv("MINIMAPPR_SIDECAR_ALLOW_NON_TMPFS_JOURNAL")
        persist_observations_on_ingest_raw = os.getenv("MINIMAPPR_PERSIST_OBSERVATIONS_ON_INGEST")
        ingest_storage_mode = _env_str("MINIMAPPR_INGEST_STORAGE_MODE", "spool").strip().lower()
        if ingest_storage_mode not in {"spool", "journal"}:
            raise ValueError(
                "MINIMAPPR_INGEST_STORAGE_MODE must be one of {'spool', 'journal'}"
            )
        ingest_port = _env_int(
            "MINIMAPPR_INGEST_PORT",
            _env_int("MINIMAPPR_SIDECAR_PORT", 8081),
        )
        ingest_backend_raw = os.getenv("MINIMAPPR_INGEST_BACKEND")
        if ingest_backend_raw is None:
            # Preserve the historical sidecar deployment toggle while adding
            # the explicit backend selector for split-process operation.
            ingest_backend = (
                "rust"
                if (
                    _env_bool("MINIMAPPR_INGEST_SIDECAR_ENABLED", True)
                    and not _env_bool("MINIMAPPR_DIRECT_INGEST_ENABLED", True)
                )
                else "python"
            )
        else:
            ingest_backend = ingest_backend_raw
        return cls(
            runtime_profile=_env_str("MINIMAPPR_RUNTIME_PROFILE", "default"),
            process_role=_env_str("MINIMAPPR_PROCESS_ROLE", "combined"),
            host=_env_str("MINIMAPPR_HOST", "0.0.0.0"),
            port=_env_int("MINIMAPPR_PORT", 8080),
            ingest_host=_env_str("MINIMAPPR_INGEST_HOST", "0.0.0.0"),
            ingest_port=ingest_port,
            ingest_base_url=_env_str("MINIMAPPR_INGEST_BASE_URL", ""),
            ingest_backend=ingest_backend,
            db_path=Path(_env_str("MINIMAPPR_DB_PATH", "data/minimappr.db")),
            snippet_dir=Path(_env_str("MINIMAPPR_SNIPPET_DIR", "data/snippets")),
            snippet_retention_seconds=_env_int("MINIMAPPR_SNIPPET_RETENTION_SECONDS", 3600),
            ingest_spool_dir=Path(_env_str("MINIMAPPR_INGEST_SPOOL_DIR", "data/spool")),
            ingest_spool_ready_ttl_seconds=_env_float("MINIMAPPR_INGEST_SPOOL_READY_TTL_SECONDS", 60.0),
            ingest_spool_failed_ttl_seconds=_env_float("MINIMAPPR_INGEST_SPOOL_FAILED_TTL_SECONDS", 86_400.0),
            ingest_spool_tmp_ttl_seconds=_env_float("MINIMAPPR_INGEST_SPOOL_TMP_TTL_SECONDS", 300.0),
            ingest_spool_poll_interval_seconds=_env_float("MINIMAPPR_INGEST_SPOOL_POLL_INTERVAL_SECONDS", 0.05),
            ingest_spool_worker_count=_env_int("MINIMAPPR_INGEST_SPOOL_WORKER_COUNT", 1),
            ingest_storage_mode=ingest_storage_mode,
            ingest_consumer_name=_env_str("MINIMAPPR_INGEST_CONSUMER_NAME", "python-ingest"),
            direct_ingest_enabled=_env_bool("MINIMAPPR_DIRECT_INGEST_ENABLED", True),
            ingest_sidecar_enabled=_env_bool("MINIMAPPR_INGEST_SIDECAR_ENABLED", True),
            ingest_sidecar_binary_path=Path(
                _env_str("MINIMAPPR_INGEST_SIDECAR_BINARY_PATH", "dist/minimappr-ingest-sidecar")
            ),
            ingest_sidecar_port=ingest_port,
            ingest_sidecar_memory_only_live_path=_env_bool(
                "MINIMAPPR_SIDECAR_MEMORY_ONLY_LIVE_PATH",
                True,
            ),
            ingest_sidecar_total_journal_budget_bytes=_env_int(
                "MINIMAPPR_SIDECAR_TOTAL_JOURNAL_BUDGET_BYTES", 268_435_456
            ),
            ingest_sidecar_admission_reserve_bytes=_env_int(
                "MINIMAPPR_SIDECAR_ADMISSION_RESERVE_BYTES", 16_777_216
            ),
            ingest_sidecar_allow_non_tmpfs_journal=(
                _env_bool("MINIMAPPR_SIDECAR_ALLOW_NON_TMPFS_JOURNAL", False)
                if allow_non_tmpfs_journal_raw is not None
                else None
            ),
            persist_observations_on_ingest=(
                _env_bool("MINIMAPPR_PERSIST_OBSERVATIONS_ON_INGEST", True)
                if persist_observations_on_ingest_raw is not None
                else None
            ),
            retention_policy_path=Path(_env_str("MINIMAPPR_RETENTION_POLICY_PATH", "data/retention_policy.json")),
            rules_config_path=Path(_env_str("MINIMAPPR_RULES_CONFIG_PATH", "data/rules.json")),
            taxonomy_config_path=Path(_env_str("MINIMAPPR_TAXONOMY_CONFIG_PATH", "data/taxonomy.json")),
            model_chain_config_path=Path(_env_str("MINIMAPPR_MODEL_CHAIN_CONFIG_PATH", "data/model_chain.json")),
            large_artifact_dir=Path(_env_str("MINIMAPPR_LARGE_ARTIFACT_DIR", "data/artifacts")),
            capture_final_tracks_settle_seconds=_env_float(
                "MINIMAPPR_CAPTURE_FINAL_TRACKS_SETTLE_SECONDS",
                30.0,
            ),
            cors_allow_origins=_env_list(
                "MINIMAPPR_CORS_ALLOW_ORIGINS",
                ("http://localhost:8080", "http://127.0.0.1:8080"),
            ),
            cors_allow_credentials=_env_bool("MINIMAPPR_CORS_ALLOW_CREDENTIALS", False),
            trigger_rms=_env_float("MINIMAPPR_TRIGGER_RMS", 0.001),
            trigger_cooldown_seconds=_env_float("MINIMAPPR_TRIGGER_COOLDOWN_SECONDS", 0.8),
            localization_window_seconds=_env_float("MINIMAPPR_LOCALIZATION_WINDOW_SECONDS", 0.08),
            classification_window_seconds=_env_float("MINIMAPPR_CLASSIFICATION_WINDOW_SECONDS", 30.0),
            max_sensor_buffer_seconds=_env_float("MINIMAPPR_MAX_SENSOR_BUFFER_SECONDS", 32.0),
            preprocess_enabled=_env_bool("MINIMAPPR_PREPROCESS_ENABLED", True),
            ingest_gain_multiplier=_env_float("MINIMAPPR_INGEST_GAIN_MULTIPLIER", 4.0),
            audio_highpass_hz=_env_float("MINIMAPPR_AUDIO_HIGHPASS_HZ", 50.0),
            audio_lowpass_hz=_env_float("MINIMAPPR_AUDIO_LOWPASS_HZ", 0.0),
            min_sensors_for_3d=_env_int("MINIMAPPR_MIN_SENSORS_FOR_3D", 4),
            min_sensors_for_2d=_env_int("MINIMAPPR_MIN_SENSORS_FOR_2D", 3),
            localization_max_tau_s=_env_float("MINIMAPPR_LOCALIZATION_MAX_TAU_S", 0.02),
            localization_algorithm=_env_str("MINIMAPPR_LOCALIZATION_ALGORITHM", "gcc_phat"),
            localization_strategy=_env_str("MINIMAPPR_LOCALIZATION_STRATEGY", "geometry_aware"),
            localization_band_min_hz=_env_float("MINIMAPPR_LOCALIZATION_BAND_MIN_HZ", 0.0),
            localization_band_max_hz=_env_float("MINIMAPPR_LOCALIZATION_BAND_MAX_HZ", 0.0),
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
            wavelength_gating_enabled=_env_bool("MINIMAPPR_WAVELENGTH_GATING_ENABLED", True),
            wavelength_penalty_floor=_env_float("MINIMAPPR_WAVELENGTH_PENALTY_FLOOR", 0.25),
            skip_localization_for_classification=_env_bool(
                "MINIMAPPR_SKIP_LOCALIZATION_FOR_CLASSIFICATION",
                False,
            ),
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
            classifier_diagonal_loading_scale=_env_float("MINIMAPPR_CLASSIFIER_DIAGONAL_LOADING_SCALE", 10.0),
            classification_stage_timeout_seconds=_env_float("MINIMAPPR_CLASSIFICATION_STAGE_TIMEOUT_SECONDS", 30.0),
            pre_classification_highpass_hz=_env_float("MINIMAPPR_PRE_CLASSIFICATION_HIGHPASS_HZ", 0.0),
            pre_classification_lowpass_hz=_env_float("MINIMAPPR_PRE_CLASSIFICATION_LOWPASS_HZ", 0.0),
            gcc_phat_interp_factor=_env_int("MINIMAPPR_GCC_PHAT_INTERP_FACTOR", 4),
            default_temperature_c=_env_float("MINIMAPPR_DEFAULT_TEMPERATURE_C", 20.0),
            default_humidity=_env_float("MINIMAPPR_DEFAULT_HUMIDITY", 0.5),
            environment_reading_max_age_seconds=_env_float("MINIMAPPR_ENVIRONMENT_READING_MAX_AGE_SECONDS", 300.0),
            site_origin_source=_env_str("MINIMAPPR_SITE_ORIGIN_SOURCE", "auto"),
            site_origin_reconcile_delay_seconds=_env_float(
                "MINIMAPPR_SITE_ORIGIN_RECONCILE_DELAY_SECONDS",
                30.0,
            ),
            site_origin_lat=_env_float("MINIMAPPR_SITE_ORIGIN_LAT", 44.98698840878797),
            site_origin_lon=_env_float("MINIMAPPR_SITE_ORIGIN_LON", -93.2579197515542),
            site_origin_alt_m=_env_float("MINIMAPPR_SITE_ORIGIN_ALT_M", 0.0),
            coordinate_mode=_env_str("MINIMAPPR_COORDINATE_MODE", "flat"),
            classifier_backend=_env_str("MINIMAPPR_CLASSIFIER", "yamnet"),
            yamnet_min_confidence=_env_float("MINIMAPPR_YAMNET_MIN_CONFIDENCE", 0.25),
            yamnet_input_target_rms=_env_float("MINIMAPPR_YAMNET_INPUT_TARGET_RMS", 0.10),
            yamnet_max_input_gain=_env_float("MINIMAPPR_YAMNET_MAX_INPUT_GAIN", 32.0),
            birdnet_trigger_min_confidence=_env_float("MINIMAPPR_BIRDNET_TRIGGER_MIN_CONFIDENCE", 0.40),
            birdnet_geo_min_confidence=_env_float("MINIMAPPR_BIRDNET_GEO_MIN_CONFIDENCE", 0.03),
            detection_min_confidence=_env_float("MINIMAPPR_DETECTION_MIN_CONFIDENCE", 0.4),
            cop_detections_max_items=_env_int("MINIMAPPR_COP_DETECTIONS_MAX_ITEMS", 150),
            cop_tracks_max_items=_env_int("MINIMAPPR_COP_TRACKS_MAX_ITEMS", 150),
            cop_detections_max_age_seconds=_env_float("MINIMAPPR_COP_DETECTIONS_MAX_AGE_SECONDS", 86_400.0),
            cop_tracks_max_age_seconds=_env_float("MINIMAPPR_COP_TRACKS_MAX_AGE_SECONDS", 86_400.0),
            heuristic_ambient_rms_threshold=_env_float("MINIMAPPR_HEURISTIC_AMBIENT_RMS_THRESHOLD", 0.01),
            heuristic_impulse_crest_threshold=_env_float("MINIMAPPR_HEURISTIC_IMPULSE_CREST_THRESHOLD", 10.0),
            heuristic_impulse_bandwidth_threshold_hz=_env_float(
                "MINIMAPPR_HEURISTIC_IMPULSE_BANDWIDTH_THRESHOLD_HZ",
                1200.0,
            ),
            heuristic_bird_centroid_min_hz=_env_float("MINIMAPPR_HEURISTIC_BIRD_CENTROID_MIN_HZ", 2200.0),
            heuristic_bird_zcr_min=_env_float("MINIMAPPR_HEURISTIC_BIRD_ZCR_MIN", 0.12),
            heuristic_speech_centroid_min_hz=_env_float("MINIMAPPR_HEURISTIC_SPEECH_CENTROID_MIN_HZ", 200.0),
            heuristic_speech_centroid_max_hz=_env_float("MINIMAPPR_HEURISTIC_SPEECH_CENTROID_MAX_HZ", 2200.0),
            heuristic_speech_zcr_min=_env_float("MINIMAPPR_HEURISTIC_SPEECH_ZCR_MIN", 0.04),
            heuristic_speech_zcr_max=_env_float("MINIMAPPR_HEURISTIC_SPEECH_ZCR_MAX", 0.2),
            heuristic_speech_flatness_max=_env_float("MINIMAPPR_HEURISTIC_SPEECH_FLATNESS_MAX", 0.75),
            heuristic_machine_centroid_max_hz=_env_float("MINIMAPPR_HEURISTIC_MACHINE_CENTROID_MAX_HZ", 450.0),
            heuristic_machine_flatness_max=_env_float("MINIMAPPR_HEURISTIC_MACHINE_FLATNESS_MAX", 0.55),
            heuristic_unknown_min_score=_env_float("MINIMAPPR_HEURISTIC_UNKNOWN_MIN_SCORE", 0.2),
            heuristic_unknown_score=_env_float("MINIMAPPR_HEURISTIC_UNKNOWN_SCORE", 0.6),
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
            linear_position_alpha=_env_float("MINIMAPPR_LINEAR_POSITION_ALPHA", 0.4),
            linear_velocity_alpha=_env_float("MINIMAPPR_LINEAR_VELOCITY_ALPHA", 0.5),
            tqi_weight_confidence=_env_float("MINIMAPPR_TQI_WEIGHT_CONFIDENCE", 0.3),
            tqi_weight_corroboration=_env_float("MINIMAPPR_TQI_WEIGHT_CORROBORATION", 0.3),
            tqi_weight_recency=_env_float("MINIMAPPR_TQI_WEIGHT_RECENCY", 0.2),
            tqi_weight_sensor=_env_float("MINIMAPPR_TQI_WEIGHT_SENSOR", 0.2),
            track_drop_multiplier=_env_float("MINIMAPPR_TRACK_DROP_MULTIPLIER", 3.0),
            track_reap_multiplier=_env_float("MINIMAPPR_TRACK_REAP_MULTIPLIER", 5.0),
            fusion_worker_count=_env_int("MINIMAPPR_FUSION_WORKER_COUNT", 1),
            fusion_event_queue_size=_env_int("MINIMAPPR_FUSION_EVENT_QUEUE_SIZE", 256),
            fusion_localization_queue_size=_env_int("MINIMAPPR_FUSION_LOCALIZATION_QUEUE_SIZE", 256),
            fusion_classification_queue_size=_env_int("MINIMAPPR_FUSION_CLASSIFICATION_QUEUE_SIZE", 256),
            fusion_rules_queue_size=_env_int("MINIMAPPR_FUSION_RULES_QUEUE_SIZE", 256),
            birdnet_chunked_dispatch_enabled=_env_bool("MINIMAPPR_BIRDNET_CHUNKED_DISPATCH_ENABLED", False),
            birdnet_chunk_overlap_seconds=_env_float("MINIMAPPR_BIRDNET_CHUNK_OVERLAP_SECONDS", 2.0),
            birdnet_chunk_max_retries_per_chunk=_env_int("MINIMAPPR_BIRDNET_CHUNK_MAX_RETRIES_PER_CHUNK", 1),
            birdnet_chunk_min_retry_progress_seconds=_env_float(
                "MINIMAPPR_BIRDNET_CHUNK_MIN_RETRY_PROGRESS_SECONDS",
                8.0,
            ),
            birdnet_chunk_retry_on_classifier_error=_env_bool(
                "MINIMAPPR_BIRDNET_CHUNK_RETRY_ON_CLASSIFIER_ERROR",
                False,
            ),
            cluster_aware_localization=_env_bool("MINIMAPPR_CLUSTER_AWARE_LOCALIZATION", False),
            fusion_drop_on_backpressure=_env_bool("MINIMAPPR_FUSION_DROP_ON_BACKPRESSURE", True),
            fusion_offline_replay_mode=_env_bool("MINIMAPPR_FUSION_OFFLINE_REPLAY_MODE", False),
            sensor_energy_threshold_multiplier=_env_float("MINIMAPPR_SENSOR_ENERGY_THRESHOLD_MULTIPLIER", 0.45),
            fallback_localization_confidence=_env_float("MINIMAPPR_FALLBACK_LOCALIZATION_CONFIDENCE", 0.25),
            reporting_window_seconds=_env_float("MINIMAPPR_REPORTING_WINDOW_SECONDS", 30.0),
            taxonomy_refresh_interval_seconds=_env_float("MINIMAPPR_TAXONOMY_REFRESH_INTERVAL_SECONDS", 10.0),
            retention_permanent_labels=_env_list(
                "MINIMAPPR_RETENTION_PERMANENT_LABELS",
                ("gunshot", "explosion", "artillery", "fusillade"),
            ),
            retention_long_security_confidence=_env_float("MINIMAPPR_RETENTION_LONG_SECURITY_CONFIDENCE", 0.6),
            cleanup_interval_seconds=_env_float("MINIMAPPR_CLEANUP_INTERVAL_SECONDS", 15.0),
            node_degraded_after_seconds=_env_float("MINIMAPPR_NODE_DEGRADED_AFTER_SECONDS", 15.0),
            node_offline_after_seconds=_env_float("MINIMAPPR_NODE_OFFLINE_AFTER_SECONDS", 45.0),
            event_stale_seconds=_env_float("MINIMAPPR_EVENT_STALE_SECONDS", 30.0),
            retention_ephemeral_seconds=_env_int("MINIMAPPR_RETENTION_EPHEMERAL_SECONDS", 900),
            retention_short_seconds=_env_int("MINIMAPPR_RETENTION_SHORT_SECONDS", 86_400),
            retention_long_seconds=_env_int("MINIMAPPR_RETENTION_LONG_SECONDS", 2_592_000),
            retention_experiment_seconds=_env_int("MINIMAPPR_RETENTION_EXPERIMENT_SECONDS", 21_600),
            retention_ingested_frames_seconds=_env_int("MINIMAPPR_RETENTION_INGESTED_FRAMES_SECONDS", 86_400),
            retention_bit_reports_seconds=_env_int("MINIMAPPR_RETENTION_BIT_REPORTS_SECONDS", 604_800),
            retention_pings_seconds=_env_int("MINIMAPPR_RETENTION_PINGS_SECONDS", 86_400),
            retention_track_updates_seconds=_env_int("MINIMAPPR_RETENTION_TRACK_UPDATES_SECONDS", 604_800),
            retention_alerts_seconds=_env_int("MINIMAPPR_RETENTION_ALERTS_SECONDS", 2_592_000),
            retention_environment_seconds=_env_int("MINIMAPPR_RETENTION_ENVIRONMENT_SECONDS", 604_800),
            retention_dropped_tracks_seconds=_env_int("MINIMAPPR_RETENTION_DROPPED_TRACKS_SECONDS", 604_800),
            federation_enabled=_env_bool("MINIMAPPR_FEDERATION_ENABLED", False),
            federation_server_id=_env_str("MINIMAPPR_FEDERATION_SERVER_ID", "srv-local"),
            federation_peers_config_path=peers_config_path,
            federation_peers=peers,
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
            federation_deconflict_use_3d=_env_bool("MINIMAPPR_FEDERATION_DECONFLICT_USE_3D", False),
            federation_auth_token=_env_str("MINIMAPPR_FEDERATION_AUTH_TOKEN", ""),
        )

    def localization_config(self) -> LocalizationConfig:
        return LocalizationConfig(
            trigger_rms=self.trigger_rms,
            trigger_cooldown_seconds=self.trigger_cooldown_seconds,
            localization_window_seconds=self.localization_window_seconds,
            classification_window_seconds=self.classification_window_seconds,
            max_sensor_buffer_seconds=self.max_sensor_buffer_seconds,
            default_temperature_c=self.default_temperature_c,
            default_humidity=self.default_humidity,
            audio_highpass_hz=self.audio_highpass_hz,
            audio_lowpass_hz=self.audio_lowpass_hz,
            preprocess_enabled=self.preprocess_enabled,
            ingest_gain_multiplier=self.ingest_gain_multiplier,
            min_sensors_for_3d=self.min_sensors_for_3d,
            min_sensors_for_2d=self.min_sensors_for_2d,
            localization_max_tau_s=self.localization_max_tau_s,
            localization_algorithm=self.localization_algorithm,
            localization_strategy=self.localization_strategy,
            localization_band_min_hz=self.localization_band_min_hz,
            localization_band_max_hz=self.localization_band_max_hz,
            localization_srp_grid_resolution_m=self.localization_srp_grid_resolution_m,
            localization_search_padding_m=self.localization_search_padding_m,
            localization_music_azimuth_step_deg=self.localization_music_azimuth_step_deg,
            localization_music_elevation_step_deg=self.localization_music_elevation_step_deg,
            localization_subspace_freq_min_hz=self.localization_subspace_freq_min_hz,
            localization_subspace_freq_max_hz=self.localization_subspace_freq_max_hz,
            localization_refine_confidence_threshold=self.localization_refine_confidence_threshold,
            localization_tight_array_aperture_m=self.localization_tight_array_aperture_m,
            wavelength_gating_enabled=self.wavelength_gating_enabled,
            wavelength_penalty_floor=self.wavelength_penalty_floor,
            skip_localization_for_classification=self.skip_localization_for_classification,
            cluster_aware_localization=self.cluster_aware_localization,
            beamformed_classification_enabled=self.beamformed_classification_enabled,
            beamformer_type=self.beamformer_type,
            beamformed_classification_min_sensor_count=self.beamformed_classification_min_sensor_count,
            beamformed_classification_confidence_margin=self.beamformed_classification_confidence_margin,
            mvdr_diagonal_loading=self.mvdr_diagonal_loading,
            classifier_diagonal_loading_scale=self.classifier_diagonal_loading_scale,
            pre_classification_highpass_hz=self.pre_classification_highpass_hz,
            pre_classification_lowpass_hz=self.pre_classification_lowpass_hz,
            gcc_phat_interp_factor=self.gcc_phat_interp_factor,
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
            linear_position_alpha=self.linear_position_alpha,
            linear_velocity_alpha=self.linear_velocity_alpha,
            tqi_weight_confidence=self.tqi_weight_confidence,
            tqi_weight_corroboration=self.tqi_weight_corroboration,
            tqi_weight_recency=self.tqi_weight_recency,
            tqi_weight_sensor=self.tqi_weight_sensor,
            track_drop_multiplier=self.track_drop_multiplier,
            track_reap_multiplier=self.track_reap_multiplier,
        )

    def classifier_config(self) -> ClassifierConfig:
        return ClassifierConfig(
            backend=self.classifier_backend,
            stage_timeout_seconds=self.classification_stage_timeout_seconds,
            yamnet_min_confidence=self.yamnet_min_confidence,
            yamnet_input_target_rms=self.yamnet_input_target_rms,
            yamnet_max_input_gain=self.yamnet_max_input_gain,
            birdnet_trigger_min_confidence=self.birdnet_trigger_min_confidence,
            birdnet_geo_min_confidence=self.birdnet_geo_min_confidence,
            heuristic_ambient_rms_threshold=self.heuristic_ambient_rms_threshold,
            heuristic_impulse_crest_threshold=self.heuristic_impulse_crest_threshold,
            heuristic_impulse_bandwidth_threshold_hz=self.heuristic_impulse_bandwidth_threshold_hz,
            heuristic_bird_centroid_min_hz=self.heuristic_bird_centroid_min_hz,
            heuristic_bird_zcr_min=self.heuristic_bird_zcr_min,
            heuristic_speech_centroid_min_hz=self.heuristic_speech_centroid_min_hz,
            heuristic_speech_centroid_max_hz=self.heuristic_speech_centroid_max_hz,
            heuristic_speech_zcr_min=self.heuristic_speech_zcr_min,
            heuristic_speech_zcr_max=self.heuristic_speech_zcr_max,
            heuristic_speech_flatness_max=self.heuristic_speech_flatness_max,
            heuristic_machine_centroid_max_hz=self.heuristic_machine_centroid_max_hz,
            heuristic_machine_flatness_max=self.heuristic_machine_flatness_max,
            heuristic_unknown_min_score=self.heuristic_unknown_min_score,
            heuristic_unknown_score=self.heuristic_unknown_score,
        )

    def storage_config(self) -> StorageConfig:
        return StorageConfig(
            db_path=self.db_path,
            snippet_dir=self.snippet_dir,
            snippet_retention_seconds=self.snippet_retention_seconds,
            retention_policy_path=self.retention_policy_path,
            retention_ephemeral_seconds=self.retention_ephemeral_seconds,
            retention_short_seconds=self.retention_short_seconds,
            retention_long_seconds=self.retention_long_seconds,
            retention_experiment_seconds=self.retention_experiment_seconds,
            retention_ingested_frames_seconds=self.retention_ingested_frames_seconds,
            retention_bit_reports_seconds=self.retention_bit_reports_seconds,
            retention_pings_seconds=self.retention_pings_seconds,
            retention_track_updates_seconds=self.retention_track_updates_seconds,
            retention_alerts_seconds=self.retention_alerts_seconds,
            retention_environment_seconds=self.retention_environment_seconds,
            retention_dropped_tracks_seconds=self.retention_dropped_tracks_seconds,
            large_artifact_dir=self.large_artifact_dir,
        )

    def fusion_config(self) -> FusionConfig:
        return FusionConfig(
            worker_count=self.fusion_worker_count,
            event_queue_size=self.fusion_event_queue_size,
            localization_queue_size=self.fusion_localization_queue_size,
            classification_queue_size=self.fusion_classification_queue_size,
            rules_queue_size=self.fusion_rules_queue_size,
            birdnet_chunked_dispatch_enabled=self.birdnet_chunked_dispatch_enabled,
            birdnet_chunk_overlap_seconds=self.birdnet_chunk_overlap_seconds,
            birdnet_chunk_max_retries_per_chunk=self.birdnet_chunk_max_retries_per_chunk,
            birdnet_chunk_min_retry_progress_seconds=self.birdnet_chunk_min_retry_progress_seconds,
            birdnet_chunk_retry_on_classifier_error=self.birdnet_chunk_retry_on_classifier_error,
            drop_on_backpressure=self.fusion_drop_on_backpressure,
            offline_replay_mode=self.fusion_offline_replay_mode,
            sensor_energy_threshold_multiplier=self.sensor_energy_threshold_multiplier,
            fallback_localization_confidence=self.fallback_localization_confidence,
            reporting_window_seconds=self.reporting_window_seconds,
            taxonomy_refresh_interval_seconds=self.taxonomy_refresh_interval_seconds,
            retention_permanent_labels=self.retention_permanent_labels,
            retention_long_security_confidence=self.retention_long_security_confidence,
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
            deconflict_use_3d=self.federation_deconflict_use_3d,
            auth_token=token or None,
        )

    def _apply_runtime_profile(self) -> None:
        if self.runtime_profile == "default":
            return
        if self.runtime_profile == "birdnet_omni_testing":
            # BirdNET needs a much longer omni clip than the TDOA/localization path.
            self.classifier_backend = "birdnet"
            self.beamformed_classification_enabled = False
            self.skip_localization_for_classification = True
            self.birdnet_chunked_dispatch_enabled = True
            self.birdnet_chunk_overlap_seconds = min(self.birdnet_chunk_overlap_seconds, 2.0)
            self.classification_window_seconds = max(self.classification_window_seconds, 30.0)
            self.max_sensor_buffer_seconds = max(
                self.max_sensor_buffer_seconds,
                self.classification_window_seconds + 2.0,
            )
            return
        if self.runtime_profile == "birdnet_hybrid_production":
            self.classifier_backend = "birdnet"
            self.localization_algorithm = "srp_phat"
            self.localization_strategy = "fixed"
            self.beamformed_classification_enabled = False
            self.skip_localization_for_classification = False
            self.birdnet_chunked_dispatch_enabled = True
            self.birdnet_chunk_overlap_seconds = min(self.birdnet_chunk_overlap_seconds, 2.0)
            if self.rules_config_path == DEFAULT_RULES_CONFIG_PATH:
                # Wildlife deployments should not page on generic human/security
                # categories unless operators explicitly provide a broader rules file.
                self.rules_config_path = DEFAULT_BIRDNET_HYBRID_RULES_CONFIG_PATH
            self.classification_window_seconds = max(self.classification_window_seconds, 30.0)
            self.max_sensor_buffer_seconds = max(
                self.max_sensor_buffer_seconds,
                self.classification_window_seconds + 2.0,
            )
            self.localization_band_min_hz = 300.0
            self.localization_band_max_hz = 3500.0
            self.reporting_window_seconds = max(self.reporting_window_seconds, 30.0)
            return
        raise ValueError(
            "MINIMAPPR_RUNTIME_PROFILE must be 'default', 'birdnet_omni_testing', or 'birdnet_hybrid_production'"
        )
