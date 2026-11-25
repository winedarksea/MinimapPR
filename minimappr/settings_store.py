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
    "localization_cross_node_band_max_hz": float,
    "localization_cross_node_window_taper_fraction": float,
    "localization_cross_node_phat_beta": float,
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
    # ── BirdNET+ V3.0 (Field Mode W1-birdnet-v3) ── restart-required: the
    # classifier is built once at startup. Model/labels paths stay env-only,
    # like drone_head_model_path.
    "birdnet_model_version": str,
    "birdnet_v3_species_list_path": str,
    "birdnet_v3_min_window_seconds": float,
    "birdnet_v3_max_window_seconds": float,
    "birdnet_v3_window_overlap_seconds": float,
    "birdnet_v3_max_nonfinite_sample_fraction": float,
    "birdnet_v3_pad_noise_floor_percentile": float,
    "birdnet_v3_num_threads": int,
    # ── end BirdNET+ V3.0 ──
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
    # --- Field Mode W2-tracker --- restart-required: TrackManager and the
    # classifier read these once at startup.
    # Per-category tracking mode (core/tracking_mode_config.py); the only
    # dict-valued patchable key, validated as a whole mapping.
    "tracking_mode_by_category": dict,
    "tracking_perched_position_diffusion_m2_per_second": float,
    "tracking_perched_relocation_snap_distance_m": float,
    "tracking_perched_relocation_gate_max_m": float,
    "tracking_perched_relocation_min_silence_seconds": float,
    "tracking_perched_relocation_ghost_seconds": float,
    "tracking_species_log_odds_enabled": bool,
    "tracking_species_log_odds_label_categories": str,
    "tracking_species_log_odds_prior_probability": float,
    "tracking_species_log_odds_per_update_cap": float,
    "tracking_species_log_odds_total_cap": float,
    "tracking_species_log_odds_evidence_top_k": int,
    "tracking_species_display_min_belief": float,
    "tracking_species_belief_exposed_top_n": int,
    "tracking_partial_constraint_binding_enabled": bool,
    "tracking_partial_binding_window_seconds": float,
    "tracking_partial_binding_bearing_sigma_deg": float,
    "tracking_partial_binding_hyperbola_sigma_m": float,
    "tracking_partial_binding_snr_level_margin_db": float,
    "tracking_partial_binding_max_normalized_residual": float,
    "tracking_partial_binding_species_top_k": int,
    "tracking_partial_binding_soft_update_gain": float,
    "tracking_partial_binding_ambiguity_score_ratio": float,
    "birdnet_two_threshold_enabled": bool,
    "birdnet_localize_to_track_min_confidence": float,
    # --- end Field Mode W2-tracker ---
    # --- Field Mode W3-birdnet-cutover --- restart-required: the classifier,
    # rule engine, retention policy and TrackManager read these at startup.
    "birdnet_v3_trigger_min_confidence": float,
    "birdnet_v3_localize_to_track_min_confidence": float,
    "birdnet_v3_species_display_min_belief": float,
    "birdnet_v3_label_migration_enabled": bool,
    "birdnet_two_threshold_require_tracker_companions": bool,
    # --- end Field Mode W3-birdnet-cutover ---
    # --- Field Mode W3-entity --- restart-required: the entity coordinator is
    # built with FusionNode at startup.
    "entity_reporting_enabled": bool,
    "entity_reporting_label_categories": str,
    "entity_absorption_window_seconds": float,
    "entity_absorption_max_fix_distance_m": float,
    "entity_absorption_max_node_distance_m": float,
    "entity_absorption_max_bearing_difference_deg": float,
    "entity_pinpointed_max_horizontal_uncertainty_m": float,
    "entity_spatial_state_hold_seconds": float,
    "entity_expiry_seconds": float,
    "entity_bearing_default_half_width_deg": float,
    "entity_bearing_min_half_width_deg": float,
    "entity_bearing_max_half_width_deg": float,
    "entity_uncertainty_sigma_multiplier": float,
    "entity_notification_grace_seconds": float,
    "entity_maintenance_interval_seconds": float,
    "entity_max_tracked_entities": int,
    # --- end Field Mode W3-entity ---
    # --- Field Mode S-fixes tracking --- restart-required (TrackManager / entity coordinator).
    "tracking_species_association_gate_enabled": bool,
    "tracking_species_association_confident_probability": float,
    "tracking_species_association_detection_claim_probability": float,
    "tracking_perched_concurrent_perch_guard_enabled": bool,
    "tracking_perched_concurrent_activity_seconds": float,
    "entity_absorption_min_species_belief": float,
    "entity_absorption_covariance_gate_enabled": bool,
    "entity_absorption_chi2_gate": float,
    "entity_absorption_min_fix_distance_m": float,
    "entity_concurrent_fix_window_seconds": float,
    "entity_bearing_evidence_absorption_enabled": bool,
    "entity_canonical_row_fused_position_enabled": bool,
    # --- end Field Mode S-fixes tracking ---
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
    # --- coverage race fix ---
    "localization_coverage_wait_all_streaming_nodes_enabled": bool,
    "localization_coverage_stall_budget_seconds": float,
    "localization_coverage_total_cap_seconds": float,
    # --- Field Mode W2-onset-srp ---
    "localization_onset_clustering_enabled": float,
    "onset_analysis_pre_trigger_seconds": float,
    "onset_analysis_post_trigger_seconds": float,
    "onset_buffer_wait_max_seconds": float,
    "onset_subband_count": int,
    "onset_hop_seconds": float,
    "onset_band_min_hz": float,
    "onset_band_max_hz": float,
    "onset_flux_threshold_mad_multiplier": float,
    "onset_min_separation_seconds": float,
    "onset_max_onsets_per_window": int,
    "onset_frame_seconds": float,
    "onset_frame_pre_roll_seconds": float,
    "onset_active_band_snr_db": float,
    "onset_narrowband_bandwidth_hz": float,
    "onset_srp_azimuth_step_deg": float,
    "onset_srp_elevation_step_deg": float,
    "onset_srp_grid_min_range_m": float,
    "onset_srp_grid_max_range_m": float,
    "onset_srp_grid_ring_count": int,
    "onset_srp_peaks_per_frame": int,
    "onset_srp_peak_min_relative_height": float,
    "onset_cluster_radius_m": float,
    "onset_cluster_range_fraction": float,
    "onset_cluster_angle_deg": float,
    "onset_cluster_min_weight_fraction": float,
    "onset_cluster_max_count": int,
    "onset_echo_min_lag_seconds": float,
    "onset_echo_max_lag_seconds": float,
    "onset_echo_min_independent_fraction": float,
    "onset_level_consistency_margin_db": float,
    "onset_persistence_min_independent_onsets": int,
    "onset_dedupe_ttl_seconds": float,
    # --- Field Mode S-fixes onset ---
    "onset_frame_max_intra_node_weight_share": float,
    "onset_srp_node_grid_ring_count": int,  # numpy counts reject floats (geomspace)
    "onset_srp_node_grid_min_range_m": float,
    "onset_srp_node_grid_max_range_m": float,
    "onset_srp_node_grid_azimuth_step_deg": float,
    "onset_srp_node_grid_elevation_step_deg": float,
    "onset_ground_prior_enabled": bool,
    "onset_node_height_above_ground_m": float,
    "onset_source_max_height_above_ground_m": float,
    "onset_planar_layout_max_flatness_ratio": float,
    "onset_height_mirror_min_score_ratio": float,
    "onset_range_profile_enabled": bool,
    "onset_range_profile_min_range_m": float,
    "onset_range_profile_max_range_m": float,
    "onset_range_profile_sample_count": int,  # numpy counts reject floats (geomspace/linspace)
    "onset_range_profile_cone_half_angle_deg": float,
    "onset_range_profile_cone_step_deg": float,
    "onset_range_profile_support_ratio": float,
    "onset_range_observable_max_interval_ratio": float,
    "onset_range_observable_interval_floor_aperture_fraction": float,
    "onset_near_node_level_tolerance_db": float,
    "onset_same_source_max_tdoa_difference_seconds": float,
    # --- end Field Mode S-fixes onset ---
    # --- end Field Mode W2-onset-srp ---
    # --- Field Mode W3-attribution --- restart-required: FusionNode builds the
    # onset stage and the attributor once at construction.
    "onset_attribution_enabled": bool,
    "onset_attribution_audible_snr_db": float,
    "onset_attribution_dominance_margin_db": float,
    "onset_attribution_spreading_tolerance_db": float,
    "onset_attribution_loco_enabled": bool,
    "onset_attribution_loco_max_reruns_per_window": int,
    "onset_attribution_loco_min_confidence": float,
    "onset_attribution_loco_label_margin": float,
    "onset_attribution_loco_collapse_ratio": float,
    "onset_attribution_loco_stft_frame_seconds": float,
    "onset_attribution_loco_noise_floor_percentile": float,
    "onset_attribution_loco_mask_active_snr_db": float,
    "onset_attribution_loco_mask_release_seconds": float,
    "onset_attribution_loco_mask_max_seconds": float,
    "onset_attribution_loco_mask_band_padding_hz": float,
    "onset_attribution_promote_band_distinct_tentative": bool,
    "onset_window_label_reconciliation_enabled": bool,
    "onset_window_label_reconciliation_timeout_seconds": float,
    "localization_collinear_mirror_max_aspect_ratio": float,
    "trigger_noise_floor_relative_enabled": bool,
    "trigger_noise_floor_relative_snr_db": float,
    "trigger_noise_floor_rise_seconds": float,
    "trigger_noise_floor_fall_seconds": float,
    "trigger_noise_floor_warmup_seconds": float,
    "trigger_noise_floor_minimum_rms": float,
    "trigger_noise_floor_fluctuation_multiplier": float,
    # --- end Field Mode W3-attribution ---
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
    # --- localization overhaul ---
    # Live: the sidecar geometry publisher re-reads both on every cycle.
    "sidecar_geometry_push_estimate_move_threshold_m": float,
    "sidecar_geometry_reconcile_interval_seconds": float,
    # Sidecar-restart-required: forwarded to the sidecar as launch env. The
    # dict-valued clock-class map is not patchable (no dict coercion here).
    "localization_max_solve_sensors": int,
    "localization_min_pair_baseline_m": float,
    "localization_cross_node_max_timeline_residual_seconds": float,
    "localization_timeline_residual_window_frames": int,
    "localization_timeline_residual_percentile": float,
    "localization_timeline_residual_over_limit_clear_ratio": float,
    "localization_clock_unknown_uncertainty_seconds": float,
    "localization_timestamp_step_threshold_seconds": float,
    "localization_timestamp_step_recovery_holdoff_seconds": float,
    "localization_timeline_step_reanchor_enabled": bool,
    "localization_timestamp_step_confirm_frames": int,
    "localization_timeline_drift_slew_fraction": float,
    "localization_trigger_noise_floor_relative_enabled": bool,
    "localization_trigger_noise_floor_relative_db": float,
    "localization_solver_floor_relative_db": float,
    "localization_sensor_registry_stale_seconds": float,
    "localization_coverage_delivery_margin_seconds": float,
    "localization_event_span_clock_margin_sigma": float,
    "localization_event_span_max_cross_node_lag_seconds": float,
    "localization_coverage_max_pending_per_buffer": int,
    # --- end localization overhaul ---
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
