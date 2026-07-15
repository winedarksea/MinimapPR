"""Runtime configuration for MinimapPR."""

from __future__ import annotations

import json
import logging
import math
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from minimappr.settings_store import (
    allowlisted_overrides,
    config_overrides_path,
    load_overrides,
)

if TYPE_CHECKING:
    from minimappr.core.effectors.registry import EffectorManagerConfig


_config_logger = logging.getLogger(__name__)

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


def _env_float_alias(keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        raw = os.getenv(key)
        if raw is not None:
            return float(raw)
    return default


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


def _env_vec3_optional(
    key: str, default: tuple[float, float, float] | None
) -> tuple[float, float, float] | None:
    """Parse a ``"x,y,z"`` env var into a Vec3, or ``None`` to disable.

    An empty value (e.g. ``MINIMAPPR_...=``) explicitly disables the setting;
    an unset var keeps ``default``.
    """
    raw = os.getenv(key)
    if raw is None:
        return default
    text = raw.strip()
    if not text:
        return None
    parts = [item.strip() for item in text.split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError(f"{key} must be 'x,y,z'; got {raw!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


_RUNTIME_PROFILE_MIGRATIONS: dict[str, tuple[str, ...]] = {
    "birdnet_hybrid_production": (
        "MINIMAPPR_BIRDNET_ENABLED=true",
        "MINIMAPPR_LOCALIZATION_ALGORITHM=srp_phat",
        "MINIMAPPR_LOCALIZATION_STRATEGY=fixed",
        "MINIMAPPR_CLASSIFICATION_AUDIO_SOURCE=omni",
        "MINIMAPPR_BIRDNET_CHUNKED_DISPATCH_ENABLED=true",
        "MINIMAPPR_BIRDNET_CHUNK_OVERLAP_SECONDS=2.0",
        "MINIMAPPR_CLASSIFICATION_WINDOW_SECONDS=30.0",
        "MINIMAPPR_MAX_SENSOR_BUFFER_SECONDS=32.0",
        "MINIMAPPR_LOCALIZATION_BAND_MIN_HZ=300.0",
        "MINIMAPPR_LOCALIZATION_BAND_MAX_HZ=3500.0",
        "MINIMAPPR_REPORTING_WINDOW_SECONDS=30.0",
        f"MINIMAPPR_RULES_CONFIG_PATH={DEFAULT_BIRDNET_HYBRID_RULES_CONFIG_PATH}",
    ),
    "birdnet_omni_testing": (
        "MINIMAPPR_BIRDNET_ENABLED=true",
        "MINIMAPPR_CLASSIFICATION_AUDIO_SOURCE=omni",
        "MINIMAPPR_SKIP_LOCALIZATION_FOR_CLASSIFICATION=true",
        "MINIMAPPR_BIRDNET_CHUNKED_DISPATCH_ENABLED=true",
        "MINIMAPPR_BIRDNET_CHUNK_OVERLAP_SECONDS=2.0",
        "MINIMAPPR_CLASSIFICATION_WINDOW_SECONDS=30.0",
        "MINIMAPPR_MAX_SENSOR_BUFFER_SECONDS=32.0",
    ),
}


def _reject_removed_classifier_env() -> None:
    """Fail fast if the removed single-backend classifier env vars are set.

    ``MINIMAPPR_CLASSIFIER`` / ``MINIMAPPR_MODEL_CHAIN_CONFIG_PATH`` are gone:
    classification is now always-on per-context routing configured via
    ``data/classifier_routing.json``. Silently ignoring the old vars would
    change which models run on live audio.
    """
    if os.getenv("MINIMAPPR_CLASSIFIER") is not None:
        raise ValueError(
            "MINIMAPPR_CLASSIFIER is removed. Classifiers are now routed per context "
            "via the routing config (data/classifier_routing.json, override with "
            "MINIMAPPR_CLASSIFIER_ROUTING_CONFIG_PATH). Kill switches: "
            "MINIMAPPR_BIRDNET_ENABLED, MINIMAPPR_DRONE_HEAD_ENABLED, "
            "MINIMAPPR_STT_ENABLED, MINIMAPPR_OMNI_SCAN_ENABLED. Unset MINIMAPPR_CLASSIFIER."
        )
    if os.getenv("MINIMAPPR_MODEL_CHAIN_CONFIG_PATH") is not None:
        raise ValueError(
            "MINIMAPPR_MODEL_CHAIN_CONFIG_PATH is removed. Chain stages now live in the "
            "'chains' section of the classifier routing config "
            "(data/classifier_routing.json). Unset MINIMAPPR_MODEL_CHAIN_CONFIG_PATH."
        )


def _reject_runtime_profile_env() -> None:
    """Fail fast if the removed ``MINIMAPPR_RUNTIME_PROFILE`` mode is still set.

    Silently ignoring it would change deployed behavior (the profile used to
    force a set of classifier/localization settings). Raise with the exact
    equivalent env vars so operators can migrate deterministically.
    """
    raw = os.getenv("MINIMAPPR_RUNTIME_PROFILE")
    if raw is None:
        return
    value = raw.strip().lower()
    if value in ("", "default"):
        return
    equivalents = _RUNTIME_PROFILE_MIGRATIONS.get(value)
    if equivalents is None:
        raise ValueError(
            f"MINIMAPPR_RUNTIME_PROFILE is removed (got {raw!r}). Unset it; the "
            "'mode' system no longer exists — configure settings directly."
        )
    listing = "\n  ".join(equivalents)
    raise ValueError(
        f"MINIMAPPR_RUNTIME_PROFILE is removed (got {raw!r}). The 'mode' system "
        "no longer exists. Unset MINIMAPPR_RUNTIME_PROFILE and set the equivalent "
        f"env vars instead:\n  {listing}"
    )


def _resolve_classification_audio_source_env() -> str:
    """Resolve the classification audio source, honoring the deprecated bool.

    Prefers the new ``MINIMAPPR_CLASSIFICATION_AUDIO_SOURCE``; when that is unset
    but the legacy ``MINIMAPPR_BEAMFORMED_CLASSIFICATION_ENABLED`` bool is set,
    map it onto the enum (True -> beamformed, False -> omni) with a deprecation log.
    """
    new_value = os.getenv("MINIMAPPR_CLASSIFICATION_AUDIO_SOURCE")
    if new_value is not None and new_value.strip():
        return new_value.strip().lower()
    legacy = os.getenv("MINIMAPPR_BEAMFORMED_CLASSIFICATION_ENABLED")
    if legacy is not None:
        enabled = legacy.strip().lower() in {"1", "true", "yes", "on"}
        mapped = "beamformed" if enabled else "omni"
        _config_logger.warning(
            "MINIMAPPR_BEAMFORMED_CLASSIFICATION_ENABLED is deprecated; use "
            "MINIMAPPR_CLASSIFICATION_AUDIO_SOURCE. Mapping %r -> %s.",
            legacy,
            mapped,
        )
        return mapped
    return "beamformed"


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
    localization_max_tau_seconds: float
    localization_algorithm: str
    localization_strategy: str
    localization_srp_grid_resolution_m: float
    localization_search_padding_m: float
    localization_music_azimuth_step_deg: float
    localization_music_elevation_step_deg: float
    localization_subspace_freq_min_hz: float
    localization_subspace_freq_max_hz: float
    localization_refine_confidence_threshold: float
    beamformer_type: str
    beamformed_classification_min_sensor_count: int
    beamformed_classification_confidence_margin: float
    mvdr_diagonal_loading: float
    classifier_diagonal_loading_scale: float
    pre_classification_highpass_hz: float
    pre_classification_lowpass_hz: float
    gcc_phat_interp_factor: int
    localization_node_bearing_strength: float = 1.0
    localization_amplitude_ratio_strength: float = 0.15
    # The default seeds unbounded radial search; max is retained but never limits results.
    localization_far_field_default_range_m: float = 50.0
    # Phase 2: envelope extended to 1 km for long-range cross-node localization.
    localization_far_field_max_range_m: float = 1000.0
    # Hard sanity gate (m): localizations whose solved position lies farther than this
    # from the contributing-sensor centroid are dropped before becoming detections/
    # tracks. Guards against unphysical solver blowups (ill-conditioned geometry).
    localization_max_range_m: float = 1200.0
    # Absolute ceiling (m) on per-axis position standard deviation. Under the
    # range-proportional caps (Phase 1b) the effective ceiling scales with range up to
    # this hard cap, so tracks never carry σ beyond a physical 1 km bound.
    localization_max_position_std_m: float = 1000.0
    classification_window_seconds: float = 30.0
    localization_band_min_hz: float = 0.0
    localization_band_max_hz: float = 0.0
    skip_localization_for_classification: bool = False
    # Classification audio source: "beamformed" | "omni" | "nearest_node_omni".
    # Mirrors Settings.classification_audio_source so the fusion node can read it
    # from the localization snapshot.
    classification_audio_source: str = "beamformed"
    # Min localization confidence for nearest-node omni selection (else omni fallback).
    min_localization_confidence: float = 0.20
    # Default ON: a defined cluster is the unit of cross-node localization on both
    # the Python and Rust paths. When a node is not a member of any cluster,
    # `cluster_for_node` returns None and the pipeline keeps its global-sensor
    # pool, so this is behavior-preserving for single-cluster / no-cluster sites.
    cluster_aware_localization: bool = True
    wavelength_gating_enabled: bool = True
    wavelength_penalty_floor: float = 0.25
    # Bound on how long _localize_candidate will wait for the per-node sensor
    # buffers to advance past event_time_ns + window/2 before dropping the
    # candidate as `buffer_lag_timeout`. Was effectively 40 ms under the
    # legacy fixed-grace-sleep retry; 300 ms restores headroom for typical
    # per-sensor ingest jitter (see plan: valiant-launching-whale).
    localization_buffer_wait_max_seconds: float = 0.30
    # Range-proportional covariance caps (Phase 1b). The effective per-axis std
    # ceiling scales with distance from the contributing sensors:
    #   min(max(std_range_factor * range_m, position_std_floor_m), max_position_std_m)
    # std_range_factor <= 0 disables the range term (legacy fixed clamp at ceiling).
    localization_std_range_factor: float = 1.0
    localization_position_std_floor_m: float = 30.0
    # Amplitude/SNR-informed range prior (Phase 1c). Substitutes the projection
    # distance for unobservable-range modes only (never overrides range_refined).
    # Ships disabled; enable after a per-node gain_offset_db field check.
    localization_amplitude_range_prior_enabled: bool = False
    localization_amplitude_reference_level_db: float = 100.0
    localization_amplitude_prior_min_range_m: float = 5.0
    localization_amplitude_prior_max_range_m: float = 1000.0
    localization_amplitude_prior_std_factor: float = 2.0

    @property
    def localization_max_tau_s(self) -> float:
        return self.localization_max_tau_seconds

    @localization_max_tau_s.setter
    def localization_max_tau_s(self, value: float) -> None:
        self.localization_max_tau_seconds = value

    @property
    def beamformed_classification_enabled(self) -> bool:
        return self.classification_audio_source == "beamformed"


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
    # Phase 3: upper bound (m) on the physical association gate radius. Default 32.0
    # preserves the legacy 4×association_distance_m clamp; raise toward
    # ~2×localization_max_position_std_m to allow cross-node cone fusion (two nodes'
    # cones for the same distant source merging into one track).
    association_max_gate_m: float = 32.0
    # Chi-squared gate on the Mahalanobis association score (3 DoF; 9.0 ≈ ~97%).
    association_chi2_gate: float = 9.0


@dataclass(slots=True)
class ClassifierConfig:
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
    training_dataset_dir: Path
    # Precedence: retention_policy_path overrides this default per label.
    snippet_retention_seconds: int
    retention_policy_path: Path
    retention_ephemeral_seconds: int
    retention_short_seconds: int
    retention_long_seconds: int
    retention_experiment_seconds: int
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
    omni_suppression_scope: str
    omni_suppression_max_distance_m: float
    taxonomy_refresh_interval_seconds: float
    retention_permanent_labels: tuple[str, ...]
    retention_long_security_confidence: float
    classification_audio_source: str = "beamformed"
    min_localization_confidence: float = 0.20


@dataclass(slots=True)
class IngestSidecarStartupConfig:
    ready_timeout_seconds: float
    ready_poll_interval_seconds: float
    healthcheck_timeout_seconds: float


@dataclass(slots=True)
class IngestSidecarProcessConfig:
    binary_path: Path
    spool_dir: Path
    consumer_name: str
    ingest_port: int
    sidecar_port: int
    storage_mode: str
    total_journal_budget_bytes: int
    admission_reserve_bytes: int
    allow_non_tmpfs_journal: bool
    memory_only_live_path: bool


@dataclass(slots=True)
class RulesConfig:
    rules_config_path: Path
    taxonomy_config_path: Path


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
    process_role: str = "combined"
    host: str = "0.0.0.0"
    port: int = 8080
    ingest_host: str = "0.0.0.0"
    ingest_port: int = 8081
    ingest_max_concurrent: int = 64
    ingest_request_timeout_seconds: float = 5.0
    ingest_base_url: str = ""
    ingest_backend: str = "python"
    db_path: Path = Path("data/minimappr.db")
    snippet_dir: Path = Path("data/snippets")
    training_dataset_dir: Path = Path("data/training")
    # Precedence: retention_policy_path overrides this default per label.
    snippet_retention_seconds: int = 3600
    # Bulky detection audio is governed independently from the lightweight
    # detection record.  These defaults intentionally mirror the classifier
    # routing members, not the legacy retention tiers.
    retention_yamnet_audio_seconds: int = 259_200
    retention_birdnet_audio_seconds: int = 2_592_000
    retention_drone_audio_seconds: int = 2_592_000
    retention_alert_audio_seconds: int = 2_592_000
    retention_detection_metadata_seconds: int = 63_072_000
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
    ingest_sidecar_ready_timeout_seconds: float = 5.0
    ingest_sidecar_ready_poll_interval_seconds: float = 0.1
    ingest_sidecar_healthcheck_timeout_seconds: float = 0.5
    persist_observations_on_ingest: bool | None = None
    retention_policy_path: Path = Path("data/retention_policy.json")
    rules_config_path: Path = DEFAULT_RULES_CONFIG_PATH
    taxonomy_config_path: Path = Path("data/taxonomy.json")
    large_artifact_dir: Path = Path("data/artifacts")
    map_overlay_dir: Path = Path("data/overlays")
    capture_final_tracks_settle_seconds: float = 30.0
    # Calibration captures mirror raw f32 audio for every registered node in
    # RAM (~0.7 MB/s/node at 44.1 kHz × 4 ch): 2 min × 4 nodes ≈ 340 MB peak.
    # Hard cap enforced at session start rather than silently truncating.
    calibration_max_duration_s: float = 120.0
    iamf_ambi_profile: str = "parametric_v2"
    # Raised-cosine omni blend above the alias cutoff for IAMF object renders
    # so objects keep full bandwidth (contract §7 rung 2 semantics, offline).
    iamf_object_band_split_enabled: bool = True
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
    localization_max_tau_seconds: float = 0.02
    localization_algorithm: str = "gcc_phat"
    localization_strategy: str = "fixed"
    localization_band_min_hz: float = 0.0
    localization_band_max_hz: float = 0.0
    localization_srp_grid_resolution_m: float = 0.5
    localization_search_padding_m: float = 2.0
    localization_far_field_default_range_m: float = 50.0
    localization_far_field_max_range_m: float = 1000.0
    localization_max_range_m: float = 1200.0
    localization_max_position_std_m: float = 1000.0
    localization_std_range_factor: float = 1.0
    localization_position_std_floor_m: float = 30.0
    localization_amplitude_range_prior_enabled: bool = False
    localization_amplitude_reference_level_db: float = 100.0
    localization_amplitude_prior_min_range_m: float = 5.0
    localization_amplitude_prior_max_range_m: float = 1000.0
    localization_amplitude_prior_std_factor: float = 2.0
    # Phase 4: windowed multi-node bearing triangulation (tier b). Ships off.
    multi_node_bearing_fusion_enabled: bool = False
    multi_node_bearing_window_seconds: float = 1.5
    multi_node_bearing_ttl_seconds: float = 4.0
    multi_node_bearing_min_separation_deg: float = 5.0
    multi_node_bearing_max_condition: float = 1e4
    # True enables cross-node TDOA automatically whenever the localization window
    # contains sensors from two or more nodes.  A single-node window has no
    # cross-node pairs, so it retains the normal per-node behavior.
    localization_cross_node_tdoa_enabled: bool = True
    localization_cross_node_max_tau_seconds: float = 0.35
    localization_cross_node_window_seconds: float = 1.0
    localization_cross_node_max_baseline_m: float = 150.0
    localization_cross_node_wait_seconds: float = 0.6
    localization_cross_node_min_sync_weight: float = 0.25
    localization_music_azimuth_step_deg: float = 6.0
    localization_music_elevation_step_deg: float = 8.0
    localization_subspace_freq_min_hz: float = 300.0
    localization_subspace_freq_max_hz: float = 3500.0
    localization_refine_confidence_threshold: float = 0.45
    # Estimator for the single-node tetrahedral (Rust sidecar) path:
    #   "python_cartesian"— feed the sidecar's pairwise TDOAs + bearing into the Python
    #   "rust"            — trust the sidecar's own SRP-PHAT position (legacy behavior)
    localization_single_node_solver: str = "python_cartesian"
    localization_node_bearing_strength: float = 1.0
    localization_amplitude_ratio_strength: float = 0.15
    wavelength_gating_enabled: bool = True
    wavelength_penalty_floor: float = 0.25
    skip_localization_for_classification: bool = False
    cluster_aware_localization: bool = True
    # Classification audio source (replaces the old beamforming on/off bool):
    #   "beamformed"        — DAS / band-split render feeds the classifier (default)
    #   "omni"              — raw loudest-mic / reporting-node omni window
    #   "nearest_node_omni" — raw omni of the sensor/node nearest the localized
    #                         source (falls back to loudest-mic omni when
    #                         localization confidence is below the threshold);
    #                         disables cross-node beamformed late fusion.
    classification_audio_source: str = "beamformed"
    # Minimum localization confidence for nearest_node_omni sensor selection.
    # Previously hardcoded 0.20 in the Rust dsp_worker; now configurable both sides.
    min_localization_confidence: float = 0.20
    beamformer_type: str = "band_split_das"
    beamformed_classification_min_sensor_count: int = 2
    beamformed_classification_confidence_margin: float = 0.0
    beamform_render_highpass_hz: float = 100.0
    beamform_low_crossover_width_hz: float = 100.0
    beamform_high_crossover_width_min_hz: float = 400.0
    beamform_high_crossover_width_fraction: float = 0.15
    mvdr_diagonal_loading: float = 1e-3
    classifier_diagonal_loading_scale: float = 10.0
    classifier_stage_timeout_seconds: float = 30.0
    pre_classification_highpass_hz: float = 0.0
    pre_classification_lowpass_hz: float = 0.0
    gcc_phat_interp_factor: int = 4
    localization_buffer_wait_max_seconds: float = 0.30

    default_temperature_c: float = 20.0
    default_humidity: float = 0.5
    environment_reading_max_age_seconds: float = 300.0
    site_origin_source: str = "auto"
    site_origin_reconcile_delay_seconds: float = 30.0
    site_origin_lat: float = 44.98698840878797
    site_origin_lon: float = -93.2579197515542
    site_origin_alt_m: float = 0.0
    coordinate_mode: str = "flat"

    # Per-context classifier routing (see minimappr/classifiers/routing.py).
    # Replaces the removed single-backend ``classifier_backend`` setting.
    classifier_routing_config_path: Path = Path("data/classifier_routing.json")
    audio_processing_config_path: Path = Path("data/audio_processing.json")
    birdnet_enabled: bool = True
    drone_head_enabled: bool = True
    drone_head_model_path: Path = Path("data/models/drone_head.onnx")
    drone_head_min_confidence: float = 0.5
    stt_enabled: bool = True
    stt_model_id: str = "onnx-community/moonshine-base-ONNX"
    stt_model_cache_dir: Path = Path("data/models/huggingface")
    stt_trigger_min_confidence: float = 0.5
    stt_pre_roll_seconds: float = 3.0
    stt_hangover_seconds: float = 2.0
    stt_max_utterance_seconds: float = 30.0
    speech_audio_dir: Path = Path("data/speech")
    transcript_retention_seconds: float = 604_800.0
    omni_scan_enabled: bool = True
    omni_scan_interval_seconds: float = 30.0
    omni_scan_window_seconds: float = 21.0
    omni_scan_min_rms: float = 0.0
    t3t4_enabled: bool = True
    t3t4_min_confidence: float = 0.5
    t3t4_min_repeats: int = 3
    t3t4_tone_band_low_hz: float = 2800.0
    t3t4_tone_band_high_hz: float = 3500.0
    t3t4_tolerance: float = 0.18
    t3t4_hysteresis_hi_ratio: float = 5.0
    t3t4_hysteresis_lo_ratio: float = 2.5
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
    association_max_gate_m: float = 32.0
    association_chi2_gate: float = 9.0
    track_stale_seconds: float = 20.0
    tracking_filter: str = "kalman"
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
    fusion_event_queue_size: int = 512
    fusion_localization_queue_size: int = 1024
    fusion_classification_queue_size: int = 1024
    fusion_rules_queue_size: int = 512
    birdnet_chunked_dispatch_enabled: bool = False
    birdnet_chunk_overlap_seconds: float = 2.0
    birdnet_chunk_max_retries_per_chunk: int = 1
    birdnet_chunk_min_retry_progress_seconds: float = 8.0
    birdnet_chunk_retry_on_classifier_error: bool = False
    drop_on_backpressure: bool = True
    fusion_offline_replay_mode: bool = False
    sensor_energy_threshold_multiplier: float = 0.45
    fallback_localization_confidence: float = 0.25
    # Cross-node audio for classification (BEAMFORMED_RENDER_CONTRACT Phase 6).
    # Off by default: each event's extra classifier invocations cost real CPU.
    cross_node_beam_enabled: bool = False
    cross_node_beam_max_range_m: float = 75.0
    cross_node_beam_max_nodes: int = 3
    reporting_window_seconds: float = 30.0
    # Scope of the omni→localized suppression check: "site" consults detections
    # from every node in the reporting window; "node" keeps the legacy per-node
    # check. Insert/upgrade paths always stay per-node keyed.
    omni_suppression_scope: str = "site"
    # Escape hatch: skip site-wide suppression when the localized detection is
    # farther than this from the omni-reporting node (0 = disabled).
    omni_suppression_max_distance_m: float = 0.0
    taxonomy_refresh_interval_seconds: float = 10.0
    retention_permanent_labels: tuple[str, ...] = ("gunshot", "explosion", "artillery", "fusillade")
    retention_long_security_confidence: float = 0.6

    cleanup_interval_seconds: float = 15.0
    sqlite_maintenance_interval_seconds: float = 3600.0
    node_degraded_after_seconds: float = 15.0
    node_offline_after_seconds: float = 45.0
    # Per-node GPS position Kalman filter (1-D applied independently to each ENU axis).
    # Q: process noise (m²/frame). At ~4 ingest frames/s, 0.5 m²/frame → ~1 m²/s uncertainty
    #    growth; time constant ≈ R/Q ≈ 50 frames ≈ 12 s. Raise Q to track deliberate moves faster.
    # R: measurement noise (m²). Consumer GNSS ~5 m 1-sigma → R = 25.
    # init_p: initial variance (m²). First fix snaps to raw measurement, subsequent frames blend.
    node_position_kalman_q: float = 0.5
    # Process noise for nodes that report mobility == "stationary". These never move,
    # so Q should be ~0 to average out GNSS noise over many fixes (steady-state gain
    # collapses toward 0). The default mobile Q (0.5) lets the estimate chase ~5-10 m
    # 2-D GPS noise, which corrupts inter-node geometry.
    node_position_kalman_q_stationary: float = 0.001
    node_position_kalman_r: float = 25.0
    node_position_kalman_init_p: float = 100.0
    # Per-axis GNSS jump gate (m). Once a node's position estimate is initialized, a raw
    # fix that deviates by more than this on any ENU axis is treated as an outlier and
    # skipped, so a single bad 2-D fix cannot yank a stationary node's position.
    node_position_gps_gate_m: float = 5.0
    # Stationary GPS KDE.  Reservoir sampling retains a representative long-term
    # distribution without persisting a per-fix location history.
    node_position_kde_bandwidth_m: float = 2.5
    node_position_kde_reservoir_capacity: int = 2048
    node_position_kde_warmup_fixes: int = 30
    node_position_kde_recompute_seconds: float = 30.0
    node_position_kde_checkpoint_seconds: float = 60.0
    node_position_kde_acceptance_radius_m: float = 100.0
    # Local position stamped onto binary-ingest frames from legacy firmware that
    # reports neither position_geo nor position_m (pre static-fallback-geo
    # descriptor builds). Lets those nodes register instead of 400-ing on every
    # frame. Set MINIMAPPR_LEGACY_INGEST_FALLBACK_POSITION_M="" to disable.
    legacy_ingest_fallback_position_m: tuple[float, float, float] | None = (0.0, 0.0, 0.0)
    event_stale_seconds: float = 30.0
    retention_ephemeral_seconds: int = 900
    retention_short_seconds: int = 86_400
    retention_long_seconds: int = 2_592_000
    retention_experiment_seconds: int = 21_600
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
    hass_enabled: bool = False
    hass_base_url: str = ""
    hass_token: str = ""
    hass_mqtt_host: str = ""
    hass_mqtt_port: int = 1883

    # Effector subsystem kill-switch only — the real gate is the `effectors` DB
    # table being empty. Defaults True so UI-driven onboarding needs no config edit.
    effectors_enabled: bool = True
    effector_snapshot_dir: Path = Path("data/effector_snapshots")
    effector_slew_dwell_seconds: float = 10.0
    effector_min_slew_interval_seconds: float = 3.0
    effector_status_poll_interval_seconds: float = 5.0

    # BLE-device-as-track subsystem. A background loop periodically trilaterates
    # BLE observations and feeds a dedicated TrackManager so BLE devices show up
    # as first-class tracks. Gating knobs default looser than the acoustic
    # associator because RSSI positions are coarse and jittery.
    ble_tracking_enabled: bool = True
    ble_tracking_period_s: float = 2.0
    ble_track_association_distance_m: float = 12.0
    ble_track_max_gate_m: float = 40.0

    node_audio_overrides: dict = field(default_factory=dict)

    # Persisted UI-set overrides file (sparse YAML). See settings_store.py.
    config_overrides_path: Path = Path("data/config.yml")

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)
        self.snippet_dir = Path(self.snippet_dir)
        self.ingest_spool_dir = Path(self.ingest_spool_dir)
        self.retention_policy_path = Path(self.retention_policy_path)
        self.rules_config_path = Path(self.rules_config_path)
        self.taxonomy_config_path = Path(self.taxonomy_config_path)
        self.classifier_routing_config_path = Path(self.classifier_routing_config_path)
        self.audio_processing_config_path = Path(self.audio_processing_config_path)
        self.drone_head_model_path = Path(self.drone_head_model_path)
        self.speech_audio_dir = Path(self.speech_audio_dir)
        self.effector_snapshot_dir = Path(self.effector_snapshot_dir)
        self.large_artifact_dir = Path(self.large_artifact_dir)
        self.map_overlay_dir = Path(self.map_overlay_dir)
        self.federation_peers_config_path = Path(self.federation_peers_config_path)
        self.config_overrides_path = Path(self.config_overrides_path)

        self.coordinate_mode = self.coordinate_mode.strip().lower()
        if self.coordinate_mode not in {"flat", "geodetic"}:
            raise ValueError("MINIMAPPR_COORDINATE_MODE must be 'flat' or 'geodetic'")
        self.classification_audio_source = self.classification_audio_source.strip().lower() or "beamformed"
        if self.classification_audio_source not in {"beamformed", "omni", "nearest_node_omni"}:
            raise ValueError(
                "MINIMAPPR_CLASSIFICATION_AUDIO_SOURCE must be one of "
                "beamformed/omni/nearest_node_omni"
            )
        if not (0.0 <= self.min_localization_confidence <= 1.0):
            raise ValueError("MINIMAPPR_MIN_LOCALIZATION_CONFIDENCE must be in [0,1]")
        self.process_role = self.process_role.strip().lower() or "combined"
        if self.process_role not in {"combined", "api", "ingest"}:
            raise ValueError("MINIMAPPR_PROCESS_ROLE must be one of combined/api/ingest")
        self.ingest_backend = self.ingest_backend.strip().lower() or "python"
        if self.ingest_backend not in {"python", "rust"}:
            raise ValueError("MINIMAPPR_INGEST_BACKEND must be one of python/rust")
        if self.ingest_port <= 0 or self.ingest_port > 65535:
            raise ValueError("MINIMAPPR_INGEST_PORT must be in [1, 65535]")
        if self.ingest_max_concurrent < 1:
            raise ValueError("MINIMAPPR_INGEST_MAX_CONCURRENT must be >= 1")
        if self.ingest_request_timeout_seconds <= 0.0:
            raise ValueError("MINIMAPPR_INGEST_REQUEST_TIMEOUT_SECONDS must be > 0")
        if not self.ingest_base_url:
            self.ingest_base_url = f"http://127.0.0.1:{self.ingest_port}"
        self.ingest_sidecar_port = self.ingest_port
        if self.ingest_sidecar_allow_non_tmpfs_journal is None:
            self.ingest_sidecar_allow_non_tmpfs_journal = platform.system() != "Linux"
        if self.persist_observations_on_ingest is None:
            # production favors real-time ingest and contiguous detection
            # snippets over dense raw-observation provenance at ingest time.
            # Persisting one observation row per sensor per frame amplifies DB I/O
            # and can starve HTTP ingest under sustained edge publish load.
            self.persist_observations_on_ingest = False

        if self.node_degraded_after_seconds <= 0.0:
            raise ValueError("MINIMAPPR_NODE_DEGRADED_AFTER_SECONDS must be > 0")
        if self.node_offline_after_seconds <= self.node_degraded_after_seconds:
            raise ValueError("MINIMAPPR_NODE_OFFLINE_AFTER_SECONDS must be > degraded threshold")
        if self.event_stale_seconds <= 0.0:
            raise ValueError("MINIMAPPR_EVENT_STALE_SECONDS must be > 0")
        if self.cleanup_interval_seconds <= 0.0:
            raise ValueError("MINIMAPPR_CLEANUP_INTERVAL_SECONDS must be > 0")
        if self.sqlite_maintenance_interval_seconds <= 0.0:
            raise ValueError("MINIMAPPR_SQLITE_MAINTENANCE_INTERVAL_SECONDS must be > 0")
        if self.capture_final_tracks_settle_seconds < 0.0:
            raise ValueError("MINIMAPPR_CAPTURE_FINAL_TRACKS_SETTLE_SECONDS must be >= 0")
        from minimappr.spatial_audio.profiles import PROFILES

        if self.iamf_ambi_profile not in PROFILES:
            known_profiles = ", ".join(sorted(PROFILES))
            raise ValueError(
                "MINIMAPPR_IAMF_AMBI_PROFILE must be one of "
                f"{known_profiles}; got {self.iamf_ambi_profile!r}"
            )
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
        if self.ingest_sidecar_ready_timeout_seconds <= 0.0:
            raise ValueError("MINIMAPPR_SIDECAR_READY_TIMEOUT_SECONDS must be > 0")
        if self.ingest_sidecar_ready_poll_interval_seconds <= 0.0:
            raise ValueError("MINIMAPPR_SIDECAR_READY_POLL_INTERVAL_SECONDS must be > 0")
        if self.ingest_sidecar_healthcheck_timeout_seconds <= 0.0:
            raise ValueError("MINIMAPPR_SIDECAR_HEALTHCHECK_TIMEOUT_SECONDS must be > 0")
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

        if self.localization_max_tau_seconds <= 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_MAX_TAU_SECONDS must be > 0")
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
        if self.localization_far_field_default_range_m <= 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_FAR_FIELD_DEFAULT_RANGE_M must be > 0")
        # Phase 2 envelope cross-validation: the range knobs must form a coherent
        # ladder default ≤ far-field-max ≤ sanity-gate, and the covariance ceiling
        # and range-proportional caps must be physical.
        if self.localization_far_field_max_range_m < self.localization_far_field_default_range_m:
            raise ValueError(
                "MINIMAPPR_LOCALIZATION_FAR_FIELD_MAX_RANGE_M must be >= FAR_FIELD_DEFAULT_RANGE_M"
            )
        if self.localization_max_range_m <= 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_MAX_RANGE_M must be > 0")
        if self.localization_max_range_m < self.localization_far_field_max_range_m:
            raise ValueError(
                "MINIMAPPR_LOCALIZATION_MAX_RANGE_M must be >= FAR_FIELD_MAX_RANGE_M"
            )
        if self.localization_max_position_std_m <= 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_MAX_POSITION_STD_M must be > 0")
        if self.localization_std_range_factor < 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_STD_RANGE_FACTOR must be >= 0")
        if self.localization_position_std_floor_m < 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_POSITION_STD_FLOOR_M must be >= 0")
        if self.localization_position_std_floor_m > self.localization_max_position_std_m:
            raise ValueError(
                "MINIMAPPR_LOCALIZATION_POSITION_STD_FLOOR_M must be <= MAX_POSITION_STD_M"
            )
        if self.localization_amplitude_prior_min_range_m < 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_AMPLITUDE_PRIOR_MIN_RANGE_M must be >= 0")
        if self.localization_amplitude_prior_max_range_m < self.localization_amplitude_prior_min_range_m:
            raise ValueError(
                "MINIMAPPR_LOCALIZATION_AMPLITUDE_PRIOR_MAX_RANGE_M must be >= MIN_RANGE_M"
            )
        if self.localization_amplitude_prior_std_factor < 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_AMPLITUDE_PRIOR_STD_FACTOR must be >= 0")
        if self.localization_subspace_freq_min_hz <= 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_SUBSPACE_FREQ_MIN_HZ must be > 0")
        if self.localization_subspace_freq_max_hz <= self.localization_subspace_freq_min_hz:
            raise ValueError("MINIMAPPR_LOCALIZATION_SUBSPACE_FREQ_MAX_HZ must be > MIN frequency")
        if self.localization_music_azimuth_step_deg <= 0.0 or self.localization_music_azimuth_step_deg > 45.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_MUSIC_AZ_STEP_DEG must be in (0,45]")
        if self.localization_music_elevation_step_deg <= 0.0 or self.localization_music_elevation_step_deg > 45.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_MUSIC_EL_STEP_DEG must be in (0,45]")
        if self.localization_refine_confidence_threshold < 0.0 or self.localization_refine_confidence_threshold > 1.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_REFINE_CONFIDENCE_THRESHOLD must be in [0,1]")
        if self.localization_single_node_solver not in {"rust", "python_cartesian"}:
            raise ValueError(
                "MINIMAPPR_LOCALIZATION_SINGLE_NODE_SOLVER must be 'rust' or 'python_cartesian'"
            )
        if self.localization_node_bearing_strength < 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_NODE_BEARING_STRENGTH must be >= 0")
        if self.localization_amplitude_ratio_strength < 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_AMPLITUDE_RATIO_STRENGTH must be >= 0")
        if self.wavelength_penalty_floor < 0.0 or self.wavelength_penalty_floor > 1.0:
            raise ValueError("MINIMAPPR_WAVELENGTH_PENALTY_FLOOR must be in [0,1]")
        if self.gcc_phat_interp_factor < 1:
            raise ValueError("MINIMAPPR_GCC_PHAT_INTERP_FACTOR must be >= 1")
        if self.localization_buffer_wait_max_seconds < 0.0:
            raise ValueError("MINIMAPPR_LOCALIZATION_BUFFER_WAIT_MAX_SECONDS must be >= 0")
        self.beamformer_type = self.beamformer_type.strip().lower()
        if self.beamformer_type == "das":
            # Preserve the more descriptive config name internally while
            # still accepting the historical shorthand.
            self.beamformer_type = "delay_and_sum"
        if self.beamformer_type == "band_split":
            self.beamformer_type = "band_split_das"
        _valid_beamformers = {
            "delay_and_sum",
            "das",
            "freq_domain_das",
            "band_split_das",
            "band_split",
            "mvdr",
            "superdirective",
            "gevd",
        }
        if self.beamformer_type not in _valid_beamformers:
            raise ValueError(
                f"MINIMAPPR_BEAMFORMER_TYPE must be one of {sorted(_valid_beamformers)}"
            )
        if self.classifier_diagonal_loading_scale < 1.0:
            raise ValueError("MINIMAPPR_CLASSIFIER_DIAGONAL_LOADING_SCALE must be >= 1.0")
        if self.classifier_stage_timeout_seconds <= 0.0:
            raise ValueError("MINIMAPPR_CLASSIFIER_STAGE_TIMEOUT_SECONDS must be > 0")
        if not (0.0 <= self.drone_head_min_confidence <= 1.0):
            raise ValueError("MINIMAPPR_DRONE_HEAD_MIN_CONFIDENCE must be in [0, 1]")
        if not (0.0 <= self.stt_trigger_min_confidence <= 1.0):
            raise ValueError("MINIMAPPR_STT_TRIGGER_MIN_CONFIDENCE must be in [0, 1]")
        if not self.stt_model_id.strip():
            raise ValueError("MINIMAPPR_STT_MODEL_ID must not be empty")
        if self.stt_pre_roll_seconds < 0.0:
            raise ValueError("MINIMAPPR_STT_PRE_ROLL_SECONDS must be >= 0")
        if self.stt_hangover_seconds < 0.0:
            raise ValueError("MINIMAPPR_STT_HANGOVER_SECONDS must be >= 0")
        if self.stt_max_utterance_seconds <= 0.0:
            raise ValueError("MINIMAPPR_STT_MAX_UTTERANCE_SECONDS must be > 0")
        if self.transcript_retention_seconds <= 0.0:
            raise ValueError("MINIMAPPR_TRANSCRIPT_RETENTION_SECONDS must be > 0")
        for field_name in (
            "retention_yamnet_audio_seconds",
            "retention_birdnet_audio_seconds",
            "retention_drone_audio_seconds",
            "retention_alert_audio_seconds",
            "retention_detection_metadata_seconds",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be > 0")
        if self.omni_scan_interval_seconds <= 0.0:
            raise ValueError("MINIMAPPR_OMNI_SCAN_INTERVAL_SECONDS must be > 0")
        if self.omni_scan_window_seconds <= 0.0:
            raise ValueError("MINIMAPPR_OMNI_SCAN_WINDOW_SECONDS must be > 0")
        if self.omni_scan_min_rms < 0.0:
            raise ValueError("MINIMAPPR_OMNI_SCAN_MIN_RMS must be >= 0")
        if self.t3t4_min_repeats < 2:
            raise ValueError("MINIMAPPR_T3T4_MIN_REPEATS must be >= 2")
        if not 0.0 <= self.t3t4_min_confidence <= 1.0:
            raise ValueError("MINIMAPPR_T3T4_MIN_CONFIDENCE must be in [0, 1]")
        if self.t3t4_tone_band_low_hz <= 0.0 or self.t3t4_tone_band_high_hz <= self.t3t4_tone_band_low_hz:
            raise ValueError("MINIMAPPR_T3T4_TONE_BAND_HIGH_HZ must be > MINIMAPPR_T3T4_TONE_BAND_LOW_HZ > 0")
        if not 0.0 < self.t3t4_tolerance <= 1.0:
            raise ValueError("MINIMAPPR_T3T4_TOLERANCE must be in (0, 1]")
        if self.t3t4_hysteresis_hi_ratio < 1.0:
            raise ValueError("MINIMAPPR_T3T4_HYSTERESIS_HI_RATIO must be >= 1")
        if not 1.0 <= self.t3t4_hysteresis_lo_ratio <= self.t3t4_hysteresis_hi_ratio:
            raise ValueError(
                "MINIMAPPR_T3T4_HYSTERESIS_LO_RATIO must be in [1, MINIMAPPR_T3T4_HYSTERESIS_HI_RATIO]"
            )
        # An utterance capture needs pre-roll + utterance + hangover of audio to
        # still be resident in the ring buffers when it closes. Clamp (rather
        # than raise: the defaults 3+30+2 slightly exceed the 32s buffer).
        stt_span = self.stt_pre_roll_seconds + self.stt_max_utterance_seconds + self.stt_hangover_seconds
        if self.stt_enabled and stt_span > self.max_sensor_buffer_seconds:
            clamped = max(
                1.0,
                self.max_sensor_buffer_seconds
                - self.stt_pre_roll_seconds
                - self.stt_hangover_seconds,
            )
            logging.getLogger(__name__).warning(
                "STT pre_roll+max_utterance+hangover (%.1fs) exceeds max_sensor_buffer_seconds "
                "(%.1fs); clamping stt_max_utterance_seconds %.1f -> %.1f",
                stt_span,
                self.max_sensor_buffer_seconds,
                self.stt_max_utterance_seconds,
                clamped,
            )
            self.stt_max_utterance_seconds = clamped
        for field_name in (
            "fusion_event_queue_size",
            "fusion_localization_queue_size",
            "fusion_classification_queue_size",
            "fusion_rules_queue_size",
        ):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be >= 1")
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
        if self.beamform_render_highpass_hz < 0.0:
            raise ValueError("MINIMAPPR_BEAMFORM_RENDER_HIGHPASS_HZ must be >= 0")
        if self.beamform_low_crossover_width_hz < 0.0:
            raise ValueError("MINIMAPPR_BEAMFORM_LOW_CROSSOVER_WIDTH_HZ must be >= 0")
        if self.beamform_high_crossover_width_min_hz < 0.0:
            raise ValueError("MINIMAPPR_BEAMFORM_HIGH_CROSSOVER_WIDTH_MIN_HZ must be >= 0")
        if not 0.0 <= self.beamform_high_crossover_width_fraction < 1.0:
            raise ValueError("MINIMAPPR_BEAMFORM_HIGH_CROSSOVER_WIDTH_FRACTION must be in [0,1)")
        if self.sensor_energy_threshold_multiplier <= 0.0:
            raise ValueError("MINIMAPPR_SENSOR_ENERGY_THRESHOLD_MULTIPLIER must be > 0")
        if self.fallback_localization_confidence < 0.0 or self.fallback_localization_confidence > 1.0:
            raise ValueError("MINIMAPPR_FALLBACK_LOCALIZATION_CONFIDENCE must be in [0,1]")
        if self.reporting_window_seconds <= 0.0:
            raise ValueError("MINIMAPPR_REPORTING_WINDOW_SECONDS must be > 0")
        if self.cross_node_beam_max_range_m <= 0.0:
            raise ValueError("MINIMAPPR_CROSS_NODE_BEAM_MAX_RANGE_M must be > 0")
        if self.cross_node_beam_max_nodes < 1:
            raise ValueError("MINIMAPPR_CROSS_NODE_BEAM_MAX_NODES must be >= 1")
        self.omni_suppression_scope = self.omni_suppression_scope.strip().lower()
        if self.omni_suppression_scope not in {"site", "node"}:
            raise ValueError("MINIMAPPR_OMNI_SUPPRESSION_SCOPE must be 'site' or 'node'")
        if self.omni_suppression_max_distance_m < 0.0:
            raise ValueError("MINIMAPPR_OMNI_SUPPRESSION_MAX_DISTANCE_M must be >= 0")
        if self.birdnet_chunk_overlap_seconds < 0.0:
            raise ValueError("MINIMAPPR_BIRDNET_CHUNK_OVERLAP_SECONDS must be >= 0")
        if self.birdnet_chunk_max_retries_per_chunk < 0:
            raise ValueError("MINIMAPPR_BIRDNET_CHUNK_MAX_RETRIES_PER_CHUNK must be >= 0")
        if self.birdnet_chunk_min_retry_progress_seconds < 0.0:
            raise ValueError("MINIMAPPR_BIRDNET_CHUNK_MIN_RETRY_PROGRESS_SECONDS must be >= 0")
        if (
            self.birdnet_chunked_dispatch_enabled
            and self.birdnet_enabled
            and self.birdnet_chunk_overlap_seconds >= self.classification_window_seconds
        ):
            raise ValueError(
                "MINIMAPPR_BIRDNET_CHUNK_OVERLAP_SECONDS must be < MINIMAPPR_CLASSIFICATION_WINDOW_SECONDS"
            )
        if (
            self.birdnet_chunked_dispatch_enabled
            and self.birdnet_enabled
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

        if self.effector_slew_dwell_seconds < 0.0:
            raise ValueError("MINIMAPPR_EFFECTOR_SLEW_DWELL_SECONDS must be >= 0")
        if self.effector_min_slew_interval_seconds < 0.0:
            raise ValueError("MINIMAPPR_EFFECTOR_MIN_SLEW_INTERVAL_SECONDS must be >= 0")
        if self.effector_status_poll_interval_seconds <= 0.0:
            raise ValueError("MINIMAPPR_EFFECTOR_STATUS_POLL_INTERVAL_SECONDS must be > 0")
        if self.hass_mqtt_port < 1 or self.hass_mqtt_port > 65535:
            raise ValueError("MINIMAPPR_HASS_MQTT_PORT must be in [1, 65535]")

    @property
    def beamformed_classification_enabled(self) -> bool:
        """Back-compat derived flag: True only when beamformed audio feeds the
        classifier. ``omni`` / ``nearest_node_omni`` sources disable beamforming."""
        return self.classification_audio_source == "beamformed"

    @classmethod
    def from_env(cls) -> "Settings":
        _reject_runtime_profile_env()
        _reject_removed_classifier_env()
        classification_audio_source = _resolve_classification_audio_source_env()
        peers_config_path = Path(_env_str("MINIMAPPR_FEDERATION_PEERS_CONFIG_PATH", "data/federation_peers.json"))
        peers = _load_federation_peers(
            raw_json=os.getenv("MINIMAPPR_FEDERATION_PEERS_JSON"),
            config_path=peers_config_path,
        )
        allow_non_tmpfs_journal_raw = os.getenv("MINIMAPPR_SIDECAR_ALLOW_NON_TMPFS_JOURNAL")
        persist_observations_on_ingest_raw = os.getenv("MINIMAPPR_PERSIST_OBSERVATIONS_ON_INGEST")
        legacy_ingest_gain_raw = os.getenv("MINIMAPPR_INGEST_GAIN_MULTIPLIER")
        if legacy_ingest_gain_raw is not None:
            _config_logger.warning(
                "MINIMAPPR_INGEST_GAIN_MULTIPLIER is deprecated; express calibrated fixed "
                "gain in the sensor ingest profile or per-node stages. The explicit legacy "
                "value remains active during migration."
            )
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
        kwargs: dict[str, object] = dict(
            process_role=_env_str("MINIMAPPR_PROCESS_ROLE", "combined"),
            host=_env_str("MINIMAPPR_HOST", "0.0.0.0"),
            port=_env_int("MINIMAPPR_PORT", 8080),
            ingest_host=_env_str("MINIMAPPR_INGEST_HOST", "0.0.0.0"),
            ingest_port=ingest_port,
            ingest_max_concurrent=_env_int("MINIMAPPR_INGEST_MAX_CONCURRENT", 64),
            ingest_request_timeout_seconds=_env_float("MINIMAPPR_INGEST_REQUEST_TIMEOUT_SECONDS", 5.0),
            ingest_base_url=_env_str("MINIMAPPR_INGEST_BASE_URL", ""),
            ingest_backend=ingest_backend,
            db_path=Path(_env_str("MINIMAPPR_DB_PATH", "data/minimappr.db")),
            snippet_dir=Path(_env_str("MINIMAPPR_SNIPPET_DIR", "data/snippets")),
            training_dataset_dir=Path(_env_str("MINIMAPPR_TRAINING_DATASET_DIR", "data/training")),
            snippet_retention_seconds=_env_int("MINIMAPPR_SNIPPET_RETENTION_SECONDS", 3600),
            retention_yamnet_audio_seconds=_env_int("MINIMAPPR_RETENTION_YAMNET_AUDIO_SECONDS", 259_200),
            retention_birdnet_audio_seconds=_env_int("MINIMAPPR_RETENTION_BIRDNET_AUDIO_SECONDS", 2_592_000),
            retention_drone_audio_seconds=_env_int("MINIMAPPR_RETENTION_DRONE_AUDIO_SECONDS", 2_592_000),
            retention_alert_audio_seconds=_env_int("MINIMAPPR_RETENTION_ALERT_AUDIO_SECONDS", 2_592_000),
            retention_detection_metadata_seconds=_env_int("MINIMAPPR_RETENTION_DETECTION_METADATA_SECONDS", 63_072_000),
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
            ingest_sidecar_ready_timeout_seconds=_env_float(
                "MINIMAPPR_SIDECAR_READY_TIMEOUT_SECONDS",
                5.0,
            ),
            ingest_sidecar_ready_poll_interval_seconds=_env_float(
                "MINIMAPPR_SIDECAR_READY_POLL_INTERVAL_SECONDS",
                0.1,
            ),
            ingest_sidecar_healthcheck_timeout_seconds=_env_float(
                "MINIMAPPR_SIDECAR_HEALTHCHECK_TIMEOUT_SECONDS",
                0.5,
            ),
            persist_observations_on_ingest=(
                _env_bool("MINIMAPPR_PERSIST_OBSERVATIONS_ON_INGEST", True)
                if persist_observations_on_ingest_raw is not None
                else None
            ),
            retention_policy_path=Path(_env_str("MINIMAPPR_RETENTION_POLICY_PATH", "data/retention_policy.json")),
            rules_config_path=Path(_env_str("MINIMAPPR_RULES_CONFIG_PATH", "data/rules.json")),
            taxonomy_config_path=Path(_env_str("MINIMAPPR_TAXONOMY_CONFIG_PATH", "data/taxonomy.json")),
            large_artifact_dir=Path(_env_str("MINIMAPPR_LARGE_ARTIFACT_DIR", "data/artifacts")),
            map_overlay_dir=Path(_env_str("MINIMAPPR_MAP_OVERLAY_DIR", "data/overlays")),
            capture_final_tracks_settle_seconds=_env_float(
                "MINIMAPPR_CAPTURE_FINAL_TRACKS_SETTLE_SECONDS",
                30.0,
            ),
            iamf_ambi_profile=_env_str("MINIMAPPR_IAMF_AMBI_PROFILE", "parametric_v2"),
            iamf_object_band_split_enabled=_env_bool("MINIMAPPR_IAMF_OBJECT_BAND_SPLIT_ENABLED", True),
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
            ingest_gain_multiplier=_env_float("MINIMAPPR_INGEST_GAIN_MULTIPLIER", 1.0),
            audio_highpass_hz=_env_float("MINIMAPPR_AUDIO_HIGHPASS_HZ", 50.0),
            audio_lowpass_hz=_env_float("MINIMAPPR_AUDIO_LOWPASS_HZ", 0.0),
            min_sensors_for_3d=_env_int("MINIMAPPR_MIN_SENSORS_FOR_3D", 4),
            min_sensors_for_2d=_env_int("MINIMAPPR_MIN_SENSORS_FOR_2D", 3),
            localization_max_tau_seconds=_env_float_alias(
                ("MINIMAPPR_LOCALIZATION_MAX_TAU_SECONDS", "MINIMAPPR_LOCALIZATION_MAX_TAU_S"),
                0.02,
            ),
            localization_algorithm=_env_str("MINIMAPPR_LOCALIZATION_ALGORITHM", "gcc_phat"),
            localization_strategy=_env_str("MINIMAPPR_LOCALIZATION_STRATEGY", "geometry_aware"),
            localization_band_min_hz=_env_float("MINIMAPPR_LOCALIZATION_BAND_MIN_HZ", 0.0),
            localization_band_max_hz=_env_float("MINIMAPPR_LOCALIZATION_BAND_MAX_HZ", 0.0),
            localization_srp_grid_resolution_m=_env_float("MINIMAPPR_LOCALIZATION_SRP_GRID_RESOLUTION_M", 0.5),
            localization_search_padding_m=_env_float("MINIMAPPR_LOCALIZATION_SEARCH_PADDING_M", 2.0),
            localization_far_field_default_range_m=_env_float(
                "MINIMAPPR_LOCALIZATION_FAR_FIELD_DEFAULT_RANGE_M",
                50.0,
            ),
            localization_far_field_max_range_m=_env_float(
                "MINIMAPPR_LOCALIZATION_FAR_FIELD_MAX_RANGE_M",
                1000.0,
            ),
            localization_max_range_m=_env_float("MINIMAPPR_LOCALIZATION_MAX_RANGE_M", 1200.0),
            localization_max_position_std_m=_env_float(
                "MINIMAPPR_LOCALIZATION_MAX_POSITION_STD_M",
                1000.0,
            ),
            localization_std_range_factor=_env_float(
                "MINIMAPPR_LOCALIZATION_STD_RANGE_FACTOR",
                1.0,
            ),
            localization_position_std_floor_m=_env_float(
                "MINIMAPPR_LOCALIZATION_POSITION_STD_FLOOR_M",
                30.0,
            ),
            localization_amplitude_range_prior_enabled=_env_bool(
                "MINIMAPPR_LOCALIZATION_AMPLITUDE_RANGE_PRIOR_ENABLED",
                False,
            ),
            localization_amplitude_reference_level_db=_env_float(
                "MINIMAPPR_LOCALIZATION_AMPLITUDE_REFERENCE_LEVEL_DB",
                100.0,
            ),
            localization_amplitude_prior_min_range_m=_env_float(
                "MINIMAPPR_LOCALIZATION_AMPLITUDE_PRIOR_MIN_RANGE_M",
                5.0,
            ),
            localization_amplitude_prior_max_range_m=_env_float(
                "MINIMAPPR_LOCALIZATION_AMPLITUDE_PRIOR_MAX_RANGE_M",
                1000.0,
            ),
            localization_amplitude_prior_std_factor=_env_float(
                "MINIMAPPR_LOCALIZATION_AMPLITUDE_PRIOR_STD_FACTOR",
                2.0,
            ),
            multi_node_bearing_fusion_enabled=_env_bool(
                "MINIMAPPR_MULTI_NODE_BEARING_FUSION_ENABLED",
                False,
            ),
            multi_node_bearing_window_seconds=_env_float(
                "MINIMAPPR_MULTI_NODE_BEARING_WINDOW_SECONDS",
                1.5,
            ),
            multi_node_bearing_ttl_seconds=_env_float(
                "MINIMAPPR_MULTI_NODE_BEARING_TTL_SECONDS",
                4.0,
            ),
            multi_node_bearing_min_separation_deg=_env_float(
                "MINIMAPPR_MULTI_NODE_BEARING_MIN_SEPARATION_DEG",
                5.0,
            ),
            multi_node_bearing_max_condition=_env_float(
                "MINIMAPPR_MULTI_NODE_BEARING_MAX_CONDITION",
                1e4,
            ),
            localization_cross_node_tdoa_enabled=_env_bool(
                "MINIMAPPR_LOCALIZATION_CROSS_NODE_TDOA_ENABLED",
                True,
            ),
            localization_cross_node_max_tau_seconds=_env_float(
                "MINIMAPPR_LOCALIZATION_CROSS_NODE_MAX_TAU_SECONDS",
                0.35,
            ),
            localization_cross_node_window_seconds=_env_float(
                "MINIMAPPR_LOCALIZATION_CROSS_NODE_WINDOW_SECONDS",
                1.0,
            ),
            localization_cross_node_max_baseline_m=_env_float(
                "MINIMAPPR_LOCALIZATION_CROSS_NODE_MAX_BASELINE_M",
                150.0,
            ),
            localization_cross_node_wait_seconds=_env_float(
                "MINIMAPPR_LOCALIZATION_CROSS_NODE_WAIT_SECONDS",
                0.6,
            ),
            localization_cross_node_min_sync_weight=_env_float(
                "MINIMAPPR_LOCALIZATION_CROSS_NODE_MIN_SYNC_WEIGHT",
                0.25,
            ),
            localization_music_azimuth_step_deg=_env_float("MINIMAPPR_LOCALIZATION_MUSIC_AZ_STEP_DEG", 6.0),
            localization_music_elevation_step_deg=_env_float("MINIMAPPR_LOCALIZATION_MUSIC_EL_STEP_DEG", 8.0),
            localization_subspace_freq_min_hz=_env_float("MINIMAPPR_LOCALIZATION_SUBSPACE_FREQ_MIN_HZ", 300.0),
            localization_subspace_freq_max_hz=_env_float("MINIMAPPR_LOCALIZATION_SUBSPACE_FREQ_MAX_HZ", 3500.0),
            localization_refine_confidence_threshold=_env_float(
                "MINIMAPPR_LOCALIZATION_REFINE_CONFIDENCE_THRESHOLD",
                0.45,
            ),
            localization_single_node_solver=_env_str(
                "MINIMAPPR_LOCALIZATION_SINGLE_NODE_SOLVER", "python_cartesian"
            ).strip().lower(),
            localization_node_bearing_strength=_env_float(
                "MINIMAPPR_LOCALIZATION_NODE_BEARING_STRENGTH",
                1.0,
            ),
            localization_amplitude_ratio_strength=_env_float(
                "MINIMAPPR_LOCALIZATION_AMPLITUDE_RATIO_STRENGTH",
                0.15,
            ),
            wavelength_gating_enabled=_env_bool("MINIMAPPR_WAVELENGTH_GATING_ENABLED", True),
            wavelength_penalty_floor=_env_float("MINIMAPPR_WAVELENGTH_PENALTY_FLOOR", 0.25),
            skip_localization_for_classification=_env_bool(
                "MINIMAPPR_SKIP_LOCALIZATION_FOR_CLASSIFICATION",
                False,
            ),
            classification_audio_source=classification_audio_source,
            min_localization_confidence=_env_float("MINIMAPPR_MIN_LOCALIZATION_CONFIDENCE", 0.20),
            beamformer_type=_env_str("MINIMAPPR_BEAMFORMER_TYPE", "band_split_das"),
            beamform_render_highpass_hz=_env_float("MINIMAPPR_BEAMFORM_RENDER_HIGHPASS_HZ", 100.0),
            beamform_low_crossover_width_hz=_env_float("MINIMAPPR_BEAMFORM_LOW_CROSSOVER_WIDTH_HZ", 100.0),
            beamform_high_crossover_width_min_hz=_env_float(
                "MINIMAPPR_BEAMFORM_HIGH_CROSSOVER_WIDTH_MIN_HZ",
                400.0,
            ),
            beamform_high_crossover_width_fraction=_env_float(
                "MINIMAPPR_BEAMFORM_HIGH_CROSSOVER_WIDTH_FRACTION",
                0.15,
            ),
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
            classifier_stage_timeout_seconds=_env_float_alias(
                (
                    "MINIMAPPR_CLASSIFIER_STAGE_TIMEOUT_SECONDS",
                    "MINIMAPPR_CLASSIFICATION_STAGE_TIMEOUT_SECONDS",
                ),
                30.0,
            ),
            pre_classification_highpass_hz=_env_float("MINIMAPPR_PRE_CLASSIFICATION_HIGHPASS_HZ", 0.0),
            pre_classification_lowpass_hz=_env_float("MINIMAPPR_PRE_CLASSIFICATION_LOWPASS_HZ", 0.0),
            gcc_phat_interp_factor=_env_int("MINIMAPPR_GCC_PHAT_INTERP_FACTOR", 4),
            localization_buffer_wait_max_seconds=_env_float(
                "MINIMAPPR_LOCALIZATION_BUFFER_WAIT_MAX_SECONDS", 0.30
            ),
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
            classifier_routing_config_path=Path(
                _env_str("MINIMAPPR_CLASSIFIER_ROUTING_CONFIG_PATH", "data/classifier_routing.json")
            ),
            audio_processing_config_path=Path(
                _env_str("MINIMAPPR_AUDIO_PROCESSING_CONFIG_PATH", "data/audio_processing.json")
            ),
            birdnet_enabled=_env_bool("MINIMAPPR_BIRDNET_ENABLED", True),
            drone_head_enabled=_env_bool("MINIMAPPR_DRONE_HEAD_ENABLED", True),
            drone_head_model_path=Path(
                _env_str("MINIMAPPR_DRONE_HEAD_MODEL_PATH", "data/models/drone_head.onnx")
            ),
            drone_head_min_confidence=_env_float("MINIMAPPR_DRONE_HEAD_MIN_CONFIDENCE", 0.5),
            stt_enabled=_env_bool("MINIMAPPR_STT_ENABLED", True),
            stt_model_id=_env_str("MINIMAPPR_STT_MODEL_ID", "onnx-community/moonshine-base-ONNX"),
            stt_model_cache_dir=Path(
                _env_str("MINIMAPPR_STT_MODEL_CACHE_DIR", "data/models/huggingface")
            ),
            stt_trigger_min_confidence=_env_float("MINIMAPPR_STT_TRIGGER_MIN_CONFIDENCE", 0.5),
            stt_pre_roll_seconds=_env_float("MINIMAPPR_STT_PRE_ROLL_SECONDS", 3.0),
            stt_hangover_seconds=_env_float("MINIMAPPR_STT_HANGOVER_SECONDS", 2.0),
            stt_max_utterance_seconds=_env_float("MINIMAPPR_STT_MAX_UTTERANCE_SECONDS", 30.0),
            speech_audio_dir=Path(_env_str("MINIMAPPR_SPEECH_AUDIO_DIR", "data/speech")),
            transcript_retention_seconds=_env_float(
                "MINIMAPPR_TRANSCRIPT_RETENTION_SECONDS", 604_800.0
            ),
            omni_scan_enabled=_env_bool("MINIMAPPR_OMNI_SCAN_ENABLED", True),
            omni_scan_interval_seconds=_env_float("MINIMAPPR_OMNI_SCAN_INTERVAL_SECONDS", 30.0),
            omni_scan_window_seconds=_env_float("MINIMAPPR_OMNI_SCAN_WINDOW_SECONDS", 21.0),
            omni_scan_min_rms=_env_float("MINIMAPPR_OMNI_SCAN_MIN_RMS", 0.0),
            t3t4_enabled=_env_bool("MINIMAPPR_T3T4_ENABLED", True),
            t3t4_min_confidence=_env_float("MINIMAPPR_T3T4_MIN_CONFIDENCE", 0.5),
            t3t4_min_repeats=_env_int("MINIMAPPR_T3T4_MIN_REPEATS", 3),
            t3t4_tone_band_low_hz=_env_float("MINIMAPPR_T3T4_TONE_BAND_LOW_HZ", 2800.0),
            t3t4_tone_band_high_hz=_env_float("MINIMAPPR_T3T4_TONE_BAND_HIGH_HZ", 3500.0),
            t3t4_tolerance=_env_float("MINIMAPPR_T3T4_TOLERANCE", 0.18),
            t3t4_hysteresis_hi_ratio=_env_float("MINIMAPPR_T3T4_HYSTERESIS_HI_RATIO", 5.0),
            t3t4_hysteresis_lo_ratio=_env_float("MINIMAPPR_T3T4_HYSTERESIS_LO_RATIO", 2.5),
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
            association_max_gate_m=_env_float("MINIMAPPR_ASSOCIATION_MAX_GATE_M", 32.0),
            association_chi2_gate=_env_float("MINIMAPPR_ASSOCIATION_CHI2_GATE", 9.0),
            track_stale_seconds=_env_float("MINIMAPPR_TRACK_STALE_SECONDS", 20.0),
            tracking_filter=_env_str("MINIMAPPR_TRACKING_FILTER", "kalman"),
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
            fusion_event_queue_size=_env_int("MINIMAPPR_FUSION_EVENT_QUEUE_SIZE", 512),
            fusion_localization_queue_size=_env_int("MINIMAPPR_FUSION_LOCALIZATION_QUEUE_SIZE", 1024),
            fusion_classification_queue_size=_env_int("MINIMAPPR_FUSION_CLASSIFICATION_QUEUE_SIZE", 1024),
            fusion_rules_queue_size=_env_int("MINIMAPPR_FUSION_RULES_QUEUE_SIZE", 512),
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
            cluster_aware_localization=_env_bool("MINIMAPPR_CLUSTER_AWARE_LOCALIZATION", True),
            drop_on_backpressure=_env_bool("MINIMAPPR_FUSION_DROP_ON_BACKPRESSURE", True),
            fusion_offline_replay_mode=_env_bool("MINIMAPPR_FUSION_OFFLINE_REPLAY_MODE", False),
            sensor_energy_threshold_multiplier=_env_float("MINIMAPPR_SENSOR_ENERGY_THRESHOLD_MULTIPLIER", 0.45),
            fallback_localization_confidence=_env_float("MINIMAPPR_FALLBACK_LOCALIZATION_CONFIDENCE", 0.25),
            cross_node_beam_enabled=_env_bool("MINIMAPPR_CROSS_NODE_BEAM_ENABLED", False),
            cross_node_beam_max_range_m=_env_float("MINIMAPPR_CROSS_NODE_BEAM_MAX_RANGE_M", 75.0),
            cross_node_beam_max_nodes=_env_int("MINIMAPPR_CROSS_NODE_BEAM_MAX_NODES", 3),
            reporting_window_seconds=_env_float("MINIMAPPR_REPORTING_WINDOW_SECONDS", 30.0),
            omni_suppression_scope=_env_str("MINIMAPPR_OMNI_SUPPRESSION_SCOPE", "site"),
            omni_suppression_max_distance_m=_env_float("MINIMAPPR_OMNI_SUPPRESSION_MAX_DISTANCE_M", 0.0),
            taxonomy_refresh_interval_seconds=_env_float("MINIMAPPR_TAXONOMY_REFRESH_INTERVAL_SECONDS", 10.0),
            retention_permanent_labels=_env_list(
                "MINIMAPPR_RETENTION_PERMANENT_LABELS",
                ("gunshot", "explosion", "artillery", "fusillade"),
            ),
            retention_long_security_confidence=_env_float("MINIMAPPR_RETENTION_LONG_SECURITY_CONFIDENCE", 0.6),
            cleanup_interval_seconds=_env_float("MINIMAPPR_CLEANUP_INTERVAL_SECONDS", 15.0),
            sqlite_maintenance_interval_seconds=_env_float(
                "MINIMAPPR_SQLITE_MAINTENANCE_INTERVAL_SECONDS", 3600.0
            ),
            node_degraded_after_seconds=_env_float("MINIMAPPR_NODE_DEGRADED_AFTER_SECONDS", 15.0),
            node_offline_after_seconds=_env_float("MINIMAPPR_NODE_OFFLINE_AFTER_SECONDS", 45.0),
            legacy_ingest_fallback_position_m=_env_vec3_optional(
                "MINIMAPPR_LEGACY_INGEST_FALLBACK_POSITION_M", (0.0, 0.0, 0.0)
            ),
            event_stale_seconds=_env_float("MINIMAPPR_EVENT_STALE_SECONDS", 30.0),
            retention_ephemeral_seconds=_env_int("MINIMAPPR_RETENTION_EPHEMERAL_SECONDS", 900),
            retention_short_seconds=_env_int("MINIMAPPR_RETENTION_SHORT_SECONDS", 86_400),
            retention_long_seconds=_env_int("MINIMAPPR_RETENTION_LONG_SECONDS", 2_592_000),
            retention_experiment_seconds=_env_int("MINIMAPPR_RETENTION_EXPERIMENT_SECONDS", 21_600),
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
            hass_enabled=_env_bool("MINIMAPPR_HASS_ENABLED", False),
            hass_base_url=_env_str("MINIMAPPR_HASS_BASE_URL", ""),
            hass_token=_env_str("MINIMAPPR_HASS_TOKEN", ""),
            hass_mqtt_host=_env_str("MINIMAPPR_HASS_MQTT_HOST", ""),
            hass_mqtt_port=_env_int("MINIMAPPR_HASS_MQTT_PORT", 1883),
            effectors_enabled=_env_bool("MINIMAPPR_EFFECTORS_ENABLED", True),
            effector_snapshot_dir=Path(_env_str("MINIMAPPR_EFFECTOR_SNAPSHOT_DIR", "data/effector_snapshots")),
            effector_slew_dwell_seconds=_env_float("MINIMAPPR_EFFECTOR_SLEW_DWELL_SECONDS", 10.0),
            effector_min_slew_interval_seconds=_env_float("MINIMAPPR_EFFECTOR_MIN_SLEW_INTERVAL_SECONDS", 3.0),
            effector_status_poll_interval_seconds=_env_float(
                "MINIMAPPR_EFFECTOR_STATUS_POLL_INTERVAL_SECONDS", 5.0
            ),
            ble_tracking_enabled=_env_bool("MINIMAPPR_BLE_TRACKING_ENABLED", True),
            ble_tracking_period_s=_env_float("MINIMAPPR_BLE_TRACKING_PERIOD_S", 2.0),
            ble_track_association_distance_m=_env_float(
                "MINIMAPPR_BLE_TRACK_ASSOCIATION_DISTANCE_M", 12.0
            ),
            ble_track_max_gate_m=_env_float("MINIMAPPR_BLE_TRACK_MAX_GATE_M", 40.0),
        )

        overrides_path = config_overrides_path(os.getenv("MINIMAPPR_CONFIG_PATH"))
        kwargs["config_overrides_path"] = overrides_path

        # Overlay persisted UI-set overrides (highest precedence). Restricted to
        # the allowlist; each applied key is logged so file-over-env precedence is
        # never silent. cls(**kwargs) below re-runs __post_init__ validation on the
        # merged result.
        file_overrides = allowlisted_overrides(load_overrides(overrides_path))
        for key, value in file_overrides.items():
            kwargs[key] = value
            _config_logger.info("Applying persisted config override: %s=%r", key, value)

        return cls(**kwargs)

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
            localization_max_tau_seconds=self.localization_max_tau_seconds,
            localization_algorithm=self.localization_algorithm,
            localization_strategy=self.localization_strategy,
            localization_band_min_hz=self.localization_band_min_hz,
            localization_band_max_hz=self.localization_band_max_hz,
            localization_srp_grid_resolution_m=self.localization_srp_grid_resolution_m,
            localization_search_padding_m=self.localization_search_padding_m,
            localization_far_field_default_range_m=self.localization_far_field_default_range_m,
            localization_far_field_max_range_m=self.localization_far_field_max_range_m,
            localization_max_range_m=self.localization_max_range_m,
            localization_max_position_std_m=self.localization_max_position_std_m,
            localization_std_range_factor=self.localization_std_range_factor,
            localization_position_std_floor_m=self.localization_position_std_floor_m,
            localization_amplitude_range_prior_enabled=self.localization_amplitude_range_prior_enabled,
            localization_amplitude_reference_level_db=self.localization_amplitude_reference_level_db,
            localization_amplitude_prior_min_range_m=self.localization_amplitude_prior_min_range_m,
            localization_amplitude_prior_max_range_m=self.localization_amplitude_prior_max_range_m,
            localization_amplitude_prior_std_factor=self.localization_amplitude_prior_std_factor,
            localization_music_azimuth_step_deg=self.localization_music_azimuth_step_deg,
            localization_music_elevation_step_deg=self.localization_music_elevation_step_deg,
            localization_subspace_freq_min_hz=self.localization_subspace_freq_min_hz,
            localization_subspace_freq_max_hz=self.localization_subspace_freq_max_hz,
            localization_refine_confidence_threshold=self.localization_refine_confidence_threshold,
            localization_node_bearing_strength=self.localization_node_bearing_strength,
            localization_amplitude_ratio_strength=self.localization_amplitude_ratio_strength,
            wavelength_gating_enabled=self.wavelength_gating_enabled,
            wavelength_penalty_floor=self.wavelength_penalty_floor,
            skip_localization_for_classification=self.skip_localization_for_classification,
            classification_audio_source=self.classification_audio_source,
            min_localization_confidence=self.min_localization_confidence,
            cluster_aware_localization=self.cluster_aware_localization,
            beamformer_type=self.beamformer_type,
            beamformed_classification_min_sensor_count=self.beamformed_classification_min_sensor_count,
            beamformed_classification_confidence_margin=self.beamformed_classification_confidence_margin,
            mvdr_diagonal_loading=self.mvdr_diagonal_loading,
            classifier_diagonal_loading_scale=self.classifier_diagonal_loading_scale,
            pre_classification_highpass_hz=self.pre_classification_highpass_hz,
            pre_classification_lowpass_hz=self.pre_classification_lowpass_hz,
            gcc_phat_interp_factor=self.gcc_phat_interp_factor,
            localization_buffer_wait_max_seconds=self.localization_buffer_wait_max_seconds,
        )

    def tracking_config(self) -> TrackingConfig:
        return TrackingConfig(
            association_distance_m=self.association_distance_m,
            association_max_gate_m=self.association_max_gate_m,
            association_chi2_gate=self.association_chi2_gate,
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

    def ble_tracking_config(self) -> TrackingConfig:
        """Tracking config for the dedicated BLE TrackManager.

        Mirrors ``tracking_config`` but with looser association gating so coarse,
        jittery RSSI positions don't spawn a churn of short-lived tracks.
        """
        cfg = self.tracking_config()
        cfg.association_distance_m = self.ble_track_association_distance_m
        cfg.association_max_gate_m = self.ble_track_max_gate_m
        return cfg

    def classifier_config(self) -> ClassifierConfig:
        return ClassifierConfig(
            stage_timeout_seconds=self.classifier_stage_timeout_seconds,
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
            training_dataset_dir=self.training_dataset_dir,
            snippet_retention_seconds=self.snippet_retention_seconds,
            retention_policy_path=self.retention_policy_path,
            retention_ephemeral_seconds=self.retention_ephemeral_seconds,
            retention_short_seconds=self.retention_short_seconds,
            retention_long_seconds=self.retention_long_seconds,
            retention_experiment_seconds=self.retention_experiment_seconds,
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
            drop_on_backpressure=self.drop_on_backpressure,
            offline_replay_mode=self.fusion_offline_replay_mode,
            sensor_energy_threshold_multiplier=self.sensor_energy_threshold_multiplier,
            fallback_localization_confidence=self.fallback_localization_confidence,
            reporting_window_seconds=self.reporting_window_seconds,
            omni_suppression_scope=self.omni_suppression_scope,
            omni_suppression_max_distance_m=self.omni_suppression_max_distance_m,
            taxonomy_refresh_interval_seconds=self.taxonomy_refresh_interval_seconds,
            retention_permanent_labels=self.retention_permanent_labels,
            retention_long_security_confidence=self.retention_long_security_confidence,
            classification_audio_source=self.classification_audio_source,
            min_localization_confidence=self.min_localization_confidence,
        )

    def ingest_sidecar_startup_config(self) -> IngestSidecarStartupConfig:
        return IngestSidecarStartupConfig(
            ready_timeout_seconds=self.ingest_sidecar_ready_timeout_seconds,
            ready_poll_interval_seconds=self.ingest_sidecar_ready_poll_interval_seconds,
            healthcheck_timeout_seconds=self.ingest_sidecar_healthcheck_timeout_seconds,
        )

    def ingest_sidecar_process_config(self) -> IngestSidecarProcessConfig:
        return IngestSidecarProcessConfig(
            binary_path=self.ingest_sidecar_binary_path,
            spool_dir=self.ingest_spool_dir,
            consumer_name=self.ingest_consumer_name,
            ingest_port=self.ingest_port,
            sidecar_port=self.ingest_sidecar_port,
            storage_mode=self.ingest_storage_mode,
            total_journal_budget_bytes=self.ingest_sidecar_total_journal_budget_bytes,
            admission_reserve_bytes=self.ingest_sidecar_admission_reserve_bytes,
            allow_non_tmpfs_journal=bool(self.ingest_sidecar_allow_non_tmpfs_journal),
            memory_only_live_path=self.ingest_sidecar_memory_only_live_path,
        )

    @property
    def localization_max_tau_s(self) -> float:
        return self.localization_max_tau_seconds

    @localization_max_tau_s.setter
    def localization_max_tau_s(self, value: float) -> None:
        self.localization_max_tau_seconds = value

    @property
    def classification_stage_timeout_seconds(self) -> float:
        return self.classifier_stage_timeout_seconds

    @classification_stage_timeout_seconds.setter
    def classification_stage_timeout_seconds(self, value: float) -> None:
        self.classifier_stage_timeout_seconds = value

    @property
    def fusion_drop_on_backpressure(self) -> bool:
        return self.drop_on_backpressure

    @fusion_drop_on_backpressure.setter
    def fusion_drop_on_backpressure(self, value: bool) -> None:
        self.drop_on_backpressure = value

    def rules_config(self) -> RulesConfig:
        return RulesConfig(
            rules_config_path=self.rules_config_path,
            taxonomy_config_path=self.taxonomy_config_path,
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

    def effector_manager_config(self) -> EffectorManagerConfig:
        from minimappr.core.effectors.registry import EffectorManagerConfig

        return EffectorManagerConfig(
            snapshot_dir=self.effector_snapshot_dir,
            min_slew_interval_seconds=self.effector_min_slew_interval_seconds,
            status_poll_interval_seconds=self.effector_status_poll_interval_seconds,
            slew_dwell_seconds=self.effector_slew_dwell_seconds,
        )
