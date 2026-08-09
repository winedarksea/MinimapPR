"""Persistent sparse settings overrides (YAML ``config.yml``).

Settings changed through the UI (``PATCH /api/v1/config``) must survive a
restart. This module owns the persisted-overrides file: a sparse YAML mapping
of allowlisted keys to values that overlays the dataclass defaults + env vars
at ``Settings.from_env()`` time.

Precedence (lowest to highest): dataclass defaults -> env vars -> persisted
overrides. A UI-saved value therefore wins over an env var, which is the
approved behavior — operators are warned via a startup log of each applied
override key and via ``persisted_override_keys`` in ``GET /api/v1/config``.

Resilience mirrors ``rules.py``: a missing file yields ``{}``; malformed YAML
or a non-dict document warns and yields ``{}`` rather than crashing startup.
Writes are atomic (temp file + ``os.replace``), mirroring the rules PUT path.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_OVERRIDES_PATH = Path("data/config.yml")

# The single source of truth for which settings may be patched at runtime and
# persisted. ``main.py`` imports this so the HTTP allowlist and the persisted
# overrides can never drift apart. Values are the coercion target type.
CONFIG_PATCH_ALLOWLIST: dict[str, type] = {
    "trigger_rms": float,
    "trigger_cooldown_seconds": float,
    "localization_window_seconds": float,
    "preprocess_enabled": bool,
    "audio_highpass_hz": float,
    "audio_lowpass_hz": float,
    "localization_algorithm": str,
    "localization_strategy": str,
    "localization_band_min_hz": float,
    "localization_band_max_hz": float,
    "classification_audio_source": str,
    "birdnet_enabled": bool,
    "drone_head_enabled": bool,
    "drone_head_min_confidence": float,
    "drone_head_min_frame_fraction": float,
    "drone_head_ambient_margin": float,
    "drone_head_min_mean_confidence": float,
    "stt_enabled": bool,
    "stt_trigger_min_confidence": float,
    "transcript_retention_seconds": float,
    "retention_yamnet_audio_seconds": int,
    "retention_birdnet_audio_seconds": int,
    "retention_drone_audio_seconds": int,
    "retention_alert_audio_seconds": int,
    "retention_detection_metadata_seconds": int,
    "omni_scan_enabled": bool,
    "omni_scan_interval_seconds": float,
    "omni_scan_window_seconds": float,
    "omni_scan_min_rms": float,
    "min_localization_confidence": float,
    "localization_min_reportable_confidence": float,
    "localization_max_reportable_gdop": float,
    "skip_localization_for_classification": bool,
    "yamnet_min_confidence": float,
    # Restart-required: the classifier is built once at startup with this gain.
    "yamnet_max_input_gain": float,
    # Noise-floor texture gate. Live-tunable on purpose: the rollout ships in
    # annotate-only mode (confidence_factor = 1.0) and flips to demotion via
    # PATCH once the flagged population has been reviewed.
    "classification_texture_gate_enabled": bool,
    "classification_texture_gate_contrast_db": float,
    "classification_texture_gate_flatness_min": float,
    "classification_texture_gate_confidence_factor": float,
    # Classification-stage admission ordering. Restart-required (the queue is
    # built in FusionNode.__init__), but allowlisted so the weighting can be
    # tuned from a deployment without editing env/YAML by hand.
    "classification_priority_enabled": bool,
    "classification_priority_track_radius_m": float,
    "classification_priority_track_cache_seconds": float,
    "classification_priority_buckets": int,
    "classification_priority_track_weight": float,
    "classification_priority_confidence_weight": float,
    "classification_priority_tier_weight": float,
    "classification_priority_signal_weight": float,
    "classification_priority_corroboration_weight": float,
    "detection_min_confidence": float,
    "cop_detections_max_items": int,
    "cop_tracks_max_items": int,
    "cop_detections_max_age_seconds": float,
    "cop_tracks_max_age_seconds": float,
    "beamformer_type": str,
    "birdnet_chunked_dispatch_enabled": bool,
    "birdnet_trigger_min_confidence": float,
    "birdnet_geo_min_confidence": float,
    # Restart-required: the classifier (and its session pool) is built once at
    # startup. pool_size=1 serializes every fusion worker through one BirdNET
    # session (2026-08-01 live-box throughput root cause).
    "birdnet_pool_size": int,
    # Restart-required alongside pool_size: the classifier's predict_session()
    # is built once at startup, so overlap can't be re-tuned without a restart.
    "birdnet_session_overlap_seconds": float,
    # Restart-required alongside pool_size: the classifier is built at startup.
    "birdnet_batch_max_wait_seconds": float,
    "birdnet_batch_max_size": int,
    "tracking_filter": str,
    "fusion_worker_count": int,
    "coordinate_mode": str,
    # Home Assistant MQTT bridge. `hass_detection_classes` (a tuple) and
    # `hass_discovery_ledger_path` are deliberately absent: they are env/YAML-level
    # knobs, not UI fields, and a ledger path change would orphan the old ledger.
    "hass_enabled": bool,
    "hass_base_url": str,
    "hass_token": str,
    "hass_mqtt_host": str,
    "hass_mqtt_port": int,
    "hass_mqtt_username": str,
    "hass_mqtt_password": str,
    "hass_mqtt_client_id": str,
    "hass_mqtt_keepalive_seconds": int,
    "hass_mqtt_tls_enabled": bool,
    "hass_mqtt_tls_insecure": bool,
    "hass_discovery_prefix": str,
    "hass_base_topic": str,
    "hass_device_id": str,
    "hass_device_name": str,
    "hass_publish_interval_seconds": float,
    "hass_publish_min_interval_seconds": float,
    "hass_reconcile_interval_seconds": float,
    "hass_queue_size": int,
    "hass_reconnect_backoff_initial_seconds": float,
    "hass_reconnect_backoff_max_seconds": float,
    "hass_detection_off_delay_seconds": int,
    "hass_track_slot_count": int,
    "hass_zone_spl_window_seconds": float,
    "hass_publish_zone_occupancy": bool,
    "hass_publish_zone_spl": bool,
    "hass_publish_detection_classes": bool,
    "hass_publish_node_status": bool,
    "hass_publish_system_health": bool,
    "hass_publish_events": bool,
    "hass_publish_track_slots": bool,
    # Track association / Kalman "greediness" knobs (see core/track_associators.py,
    # core/track_filters.py). Previously env-var-only (MINIMAPPR_*); exposed here so
    # they can be tuned from a live deployment without a process restart+redeploy.
    "association_distance_m": float,
    "association_max_gate_m": float,
    "association_chi2_gate": float,
    "kalman_process_noise": float,
    "kalman_measurement_noise": float,
    "track_stale_seconds": float,
    # Track-continuity overhaul: per-category lifecycle windows, per-category
    # Kalman q, Kalman coast guards, class-aware association and dormant
    # reacquisition. All live-tunable — the two semantic changes
    # (association_category_gate_enabled, dormant_reacquire_enabled) double as
    # kill switches since the new behaviour ships on by default.
    "track_stale_seconds_wildlife": float,
    "track_stale_seconds_vehicle": float,
    "track_stale_seconds_human": float,
    "track_stale_seconds_security": float,
    "kalman_process_noise_wildlife": float,
    "kalman_process_noise_vehicle": float,
    "kalman_process_noise_human": float,
    "kalman_process_noise_security": float,
    "kalman_max_coast_process_seconds": float,
    "kalman_coast_velocity_half_life_seconds": float,
    "association_category_gate_enabled": bool,
    "association_fingerprint_weight": float,
    "track_fingerprint_alpha": float,
    "track_fingerprint_top_k": int,
    "dormant_reacquire_enabled": bool,
    "dormant_ttl_seconds": float,
    "dormant_reacquire_radius_m": float,
    "dormant_fingerprint_min_similarity": float,
    "dormant_confidence_half_life_seconds": float,
    "dormant_max_records": int,
    # DOA/TDOA solve blend + cross-node bearing-fusion "greediness" knobs (see
    # core/cartesian_tdoa.py, core/multi_node_bearing_fusion.py).
    "localization_node_bearing_strength": float,
    # The enable flag itself was oddly absent while every tuning knob below was
    # PATCHable; a 2-node site could tune bearing fusion but never turn it on.
    "multi_node_bearing_fusion_enabled": bool,
    "multi_node_bearing_window_seconds": float,
    "multi_node_bearing_min_separation_deg": float,
    "multi_node_bearing_ttl_seconds": float,
    "multi_node_bearing_max_condition": float,
    # Phase 5 cross-node sensor admission (see FusionNode._admit_cross_node_sensors).
    "localization_cross_node_admission_enabled": bool,
    "localization_cross_node_relative_energy_floor": float,
    # Classification-lane backpressure knobs (see core/fusion_node.py). All
    # restart-required — the queues and the FusionConfig snapshot are built at
    # FusionNode.start(). Exposed after the 2026-08-01 live-box review found the
    # 1024-deep classification queue was itself the 16-minute-lag mechanism and
    # none of these could be tuned without an env change + redeploy.
    "fusion_classification_queue_size": int,
    "classification_window_seconds": float,
    "drop_on_backpressure": bool,
    "fusion_backpressure_drop_policy": str,
    # Pre-render report-window gate (0 = off): skip the beamform render +
    # inference once a (node, reporting window) already emitted this many
    # localized detections. The storage-level dedupe made the same call after
    # the render — ~89% of classification-lane CPU on the 2026-08-01 live box.
    "fusion_report_window_localized_emission_cap": int,
}


def config_overrides_path(configured: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the overrides file path.

    Precedence: explicit ``configured`` argument -> ``MINIMAPPR_CONFIG_PATH``
    env var -> ``DEFAULT_CONFIG_OVERRIDES_PATH``.
    """
    if configured is not None and str(configured).strip():
        return Path(configured)
    env = os.getenv("MINIMAPPR_CONFIG_PATH")
    if env is not None and env.strip():
        return Path(env)
    return DEFAULT_CONFIG_OVERRIDES_PATH


def load_overrides(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return the persisted overrides mapping, or ``{}`` on any problem.

    Missing file -> ``{}``. Malformed YAML or a non-dict document -> a warning
    and ``{}`` (matches ``rules.py`` resilience; never crash startup).
    """
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Unable to read config overrides %s: %s", file_path, exc)
        return {}
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        logger.warning(
            "Config overrides %s is not a mapping (got %s); ignoring.",
            file_path,
            type(raw).__name__,
        )
        return {}
    return {str(key): value for key, value in raw.items()}


def allowlisted_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """Restrict a raw overrides mapping to allowlisted keys.

    Unknown keys are dropped with a warning rather than raising: an operator's
    stale hand-edited file should not brick startup.
    """
    kept: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in CONFIG_PATCH_ALLOWLIST:
            kept[key] = value
        else:
            logger.warning("Ignoring non-allowlisted config override key: %s", key)
    return kept


def save_overrides(path: str | os.PathLike[str], overrides: dict[str, Any]) -> None:
    """Atomically persist the overrides mapping to ``path`` as YAML.

    Restricted to allowlisted keys. Written to a temp file then ``os.replace``d
    into place so a concurrent reader never sees a partial document.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = allowlisted_overrides(overrides)
    tmp_path = file_path.with_name(f".{file_path.name}.tmp")
    tmp_path.write_text(
        yaml.safe_dump(payload, default_flow_style=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp_path, file_path)


__all__ = [
    "CONFIG_PATCH_ALLOWLIST",
    "DEFAULT_CONFIG_OVERRIDES_PATH",
    "allowlisted_overrides",
    "config_overrides_path",
    "load_overrides",
    "save_overrides",
]
