"""Stage-grouped projection of the flat ``GET /api/v1/config`` surface.

``GET /api/v1/config/structured`` is an additive, read-only regrouping of the
existing flat config into the same stage vocabulary the pipeline DAG uses
(ingest / preprocess / gates / localization / beamform / classification /
tracking / fusion / rules_alerts) plus non-stage groups (storage_retention,
federation, system). The flat GET/PATCH ``/api/v1/config`` is unchanged.

``PipelineParam.config_key`` and structured ``entries[].key`` share the flat
top-level key namespace, so DAG deep links resolve to ``/settings/config#{id}``.

Every flat config key MUST live in exactly one group or in ``UNGROUPED_KEYS``;
``tests/test_config_structured.py`` enforces this so new Settings keys cannot be
added without also being placed in the structured view.
"""

from __future__ import annotations

from ..models import PipelineStageKind

# (group_id, title, stage_kind | None, keys) — group_id matches DAG column ids
# where a stage exists, so `/settings/config#{group_id}` anchors line up.
CONFIG_STAGE_GROUPS: tuple[tuple[str, str, PipelineStageKind | None, tuple[str, ...]], ...] = (
    (
        "ingest",
        "Ingest",
        PipelineStageKind.SOURCE,
        ("ingest_backend", "ingest_base_url", "ingest_host", "ingest_port", "process_role"),
    ),
    (
        "preprocess",
        "Preprocess",
        PipelineStageKind.PREPROCESS,
        ("preprocess_enabled", "audio_highpass_hz", "audio_lowpass_hz"),
    ),
    (
        "gates",
        "Gates",
        PipelineStageKind.GATE,
        (
            "trigger_rms",
            # --- Field Mode W3-attribution --- noise-floor-relative trigger
            "trigger_noise_floor_relative_enabled",
            "trigger_noise_floor_relative_snr_db",
            "trigger_noise_floor_rise_seconds",
            "trigger_noise_floor_fall_seconds",
            "trigger_noise_floor_warmup_seconds",
            "trigger_noise_floor_minimum_rms",
            "trigger_noise_floor_fluctuation_multiplier",
            # --- end Field Mode W3-attribution ---
            "trigger_cooldown_seconds",
            "omni_scan_enabled",
            "omni_scan_interval_seconds",
            "omni_scan_window_seconds",
            "omni_scan_min_rms",
            # --- localization overhaul --- Rust sidecar trigger
            "localization_trigger_noise_floor_relative_enabled",
            "localization_trigger_noise_floor_relative_db",
        ),
    ),
    (
        "localization",
        "Localization",
        PipelineStageKind.LOCALIZATION,
        (
            "localization_algorithm",
            "localization_strategy",
            "localization_window_seconds",
            "localization_max_tau_s",
            "localization_max_tau_seconds",
            "localization_band_min_hz",
            "localization_band_max_hz",
            "localization_cross_node_band_max_hz",
            "localization_cross_node_window_taper_fraction",
            "localization_cross_node_phat_beta",
            "localization_srp_grid_resolution_m",
            "localization_search_padding_m",
            "localization_music_azimuth_step_deg",
            "localization_music_elevation_step_deg",
            "localization_subspace_freq_min_hz",
            "localization_subspace_freq_max_hz",
            "localization_refine_confidence_threshold",
            "localization_min_reportable_confidence",
            "localization_max_reportable_gdop",
            "min_sensors_for_2d",
            "min_sensors_for_3d",
            "min_localization_confidence",
            "skip_localization_for_classification",
            "coordinate_mode",
            "localization_node_bearing_strength",
            "multi_node_bearing_fusion_enabled",
            "multi_node_bearing_window_seconds",
            "multi_node_bearing_min_separation_deg",
            "multi_node_bearing_ttl_seconds",
            "multi_node_bearing_max_condition",
            "localization_cross_node_admission_enabled",
            "localization_cross_node_relative_energy_floor",
            # --- coverage race fix ---
            "localization_coverage_wait_all_streaming_nodes_enabled",
            "localization_coverage_stall_budget_seconds",
            "localization_coverage_total_cap_seconds",
            # --- Field Mode W2-onset-srp ---
            "localization_onset_clustering_enabled",
            "onset_analysis_pre_trigger_seconds",
            "onset_analysis_post_trigger_seconds",
            "onset_buffer_wait_max_seconds",
            "onset_subband_count",
            "onset_hop_seconds",
            "onset_band_min_hz",
            "onset_band_max_hz",
            "onset_flux_threshold_mad_multiplier",
            "onset_min_separation_seconds",
            "onset_max_onsets_per_window",
            "onset_frame_seconds",
            "onset_frame_pre_roll_seconds",
            "onset_active_band_snr_db",
            "onset_narrowband_bandwidth_hz",
            "onset_srp_azimuth_step_deg",
            "onset_srp_elevation_step_deg",
            "onset_srp_grid_min_range_m",
            "onset_srp_grid_max_range_m",
            "onset_srp_grid_ring_count",
            "onset_srp_peaks_per_frame",
            "onset_srp_peak_min_relative_height",
            "onset_cluster_radius_m",
            "onset_cluster_range_fraction",
            "onset_cluster_angle_deg",
            "onset_cluster_min_weight_fraction",
            "onset_cluster_max_count",
            "onset_echo_min_lag_seconds",
            "onset_echo_max_lag_seconds",
            "onset_echo_min_independent_fraction",
            "onset_level_consistency_margin_db",
            "onset_persistence_min_independent_onsets",
            "onset_dedupe_ttl_seconds",
            # --- Field Mode S-fixes onset ---
            "onset_frame_max_intra_node_weight_share",
            "onset_srp_node_grid_ring_count",
            "onset_srp_node_grid_min_range_m",
            "onset_srp_node_grid_max_range_m",
            "onset_srp_node_grid_azimuth_step_deg",
            "onset_srp_node_grid_elevation_step_deg",
            "onset_ground_prior_enabled",
            "onset_node_height_above_ground_m",
            "onset_source_max_height_above_ground_m",
            "onset_planar_layout_max_flatness_ratio",
            "onset_height_mirror_min_score_ratio",
            "onset_range_profile_enabled",
            "onset_range_profile_min_range_m",
            "onset_range_profile_max_range_m",
            "onset_range_profile_sample_count",
            "onset_range_profile_cone_half_angle_deg",
            "onset_range_profile_cone_step_deg",
            "onset_range_profile_support_ratio",
            "onset_range_observable_max_interval_ratio",
            "onset_range_observable_interval_floor_aperture_fraction",
            "onset_near_node_level_tolerance_db",
            "onset_same_source_max_tdoa_difference_seconds",
            "localization_collinear_mirror_max_aspect_ratio",
            # --- end Field Mode S-fixes onset ---
            # --- end Field Mode W2-onset-srp ---
            # --- Field Mode W3-attribution ---
            "onset_attribution_enabled",
            "onset_attribution_audible_snr_db",
            "onset_attribution_dominance_margin_db",
            "onset_attribution_spreading_tolerance_db",
            "onset_attribution_loco_enabled",
            "onset_attribution_loco_max_reruns_per_window",
            "onset_attribution_loco_min_confidence",
            "onset_attribution_loco_label_margin",
            "onset_attribution_loco_collapse_ratio",
            "onset_attribution_loco_stft_frame_seconds",
            "onset_attribution_loco_noise_floor_percentile",
            "onset_attribution_loco_mask_active_snr_db",
            "onset_attribution_loco_mask_release_seconds",
            "onset_attribution_loco_mask_max_seconds",
            "onset_attribution_loco_mask_band_padding_hz",
            "onset_attribution_promote_band_distinct_tentative",
            "onset_window_label_reconciliation_enabled",
            "onset_window_label_reconciliation_timeout_seconds",
            # --- end Field Mode W3-attribution ---
            # --- localization overhaul --- solve geometry, pair admission, clock evidence
            "localization_max_solve_sensors",
            "localization_min_pair_baseline_m",
            "localization_cross_node_max_timeline_residual_seconds",
            "localization_timeline_residual_window_frames",
            "localization_timeline_residual_percentile",
            "localization_timeline_residual_over_limit_clear_ratio",
            "localization_clock_class_uncertainty_seconds",
            "localization_clock_unknown_uncertainty_seconds",
            "localization_timestamp_step_threshold_seconds",
            "localization_timestamp_step_recovery_holdoff_seconds",
            "localization_timeline_step_reanchor_enabled",
            "localization_timestamp_step_confirm_frames",
            "localization_timeline_drift_slew_fraction",
            "localization_solver_floor_relative_db",
            "localization_sensor_registry_stale_seconds",
            "sidecar_geometry_push_estimate_move_threshold_m",
            "sidecar_geometry_reconcile_interval_seconds",
        ),
    ),
    (
        "beamform",
        "Beamform",
        PipelineStageKind.BEAMFORM,
        (
            "beamformer_type",
            "beamformed_classification_min_sensor_count",
            "beamformed_classification_confidence_margin",
            "mvdr_diagonal_loading",
        ),
    ),
    (
        "classification",
        "Classification",
        PipelineStageKind.CLASSIFIER,
        (
            "classification_audio_source",
            "classification_window_seconds",
            "classifier_routing_config_path",
            "classifier_stage_timeout_seconds",
            "classification_stage_timeout_seconds",
            "birdnet_enabled",
            "birdnet_chunked_dispatch_enabled",
            "birdnet_trigger_min_confidence",
            "birdnet_geo_min_confidence",
            "birdnet_pool_size",
            "birdnet_session_overlap_seconds",
            "birdnet_batch_max_wait_seconds",
            "birdnet_batch_max_size",
            "birdnet_direct_inference",
            "birdnet_direct_num_threads",
            # ── BirdNET+ V3.0 (Field Mode W1-birdnet-v3) ──
            "birdnet_model_version",
            "birdnet_v3_model_path",
            "birdnet_v3_labels_path",
            "birdnet_v3_species_list_path",
            "birdnet_v3_min_window_seconds",
            "birdnet_v3_max_window_seconds",
            "birdnet_v3_window_overlap_seconds",
            "birdnet_v3_max_nonfinite_sample_fraction",
            "birdnet_v3_pad_noise_floor_percentile",
            "birdnet_v3_num_threads",
            # ── end BirdNET+ V3.0 ──
            "drone_head_enabled",
            "drone_head_model_path",
            "drone_head_min_confidence",
            "drone_head_min_frame_fraction",
            "drone_head_ambient_margin",
            "drone_head_min_mean_confidence",
            "stt_enabled",
            "stt_model_id",
            "stt_model_cache_dir",
            "stt_trigger_min_confidence",
            "yamnet_min_confidence",
            "yamnet_input_target_rms",
            "yamnet_max_input_gain",
            "classification_texture_gate_enabled",
            "classification_texture_gate_contrast_db",
            "classification_texture_gate_flatness_min",
            "classification_texture_gate_confidence_factor",
            "classification_priority_enabled",
            "classification_priority_track_radius_m",
            "classification_priority_track_cache_seconds",
            "classification_priority_buckets",
            "classification_priority_track_weight",
            "classification_priority_confidence_weight",
            "classification_priority_tier_weight",
            "classification_priority_signal_weight",
            "classification_priority_corroboration_weight",
            "detection_min_confidence",
            "taxonomy_config_path",
            "point_network_clip_classification_enabled",  # localization overhaul
        ),
    ),
    (
        "tracking",
        "Tracking",
        PipelineStageKind.TRACKING,
        (
            "tracking_filter",
            "association_distance_m",
            "association_max_gate_m",
            "association_chi2_gate",
            "kalman_process_noise",
            "kalman_measurement_noise",
            "track_stale_seconds",
            "track_stale_seconds_wildlife",
            "track_stale_seconds_vehicle",
            "track_stale_seconds_human",
            "track_stale_seconds_security",
            "kalman_process_noise_wildlife",
            "kalman_process_noise_vehicle",
            "kalman_process_noise_human",
            "kalman_process_noise_security",
            "kalman_max_coast_process_seconds",
            "kalman_coast_velocity_half_life_seconds",
            "association_category_gate_enabled",
            "association_fingerprint_weight",
            "track_fingerprint_alpha",
            "track_fingerprint_top_k",
            "dormant_reacquire_enabled",
            "dormant_ttl_seconds",
            "dormant_reacquire_radius_m",
            "dormant_fingerprint_min_similarity",
            "dormant_confidence_half_life_seconds",
            "dormant_max_records",
            # --- Field Mode W2-tracker ---
            "tracking_perched_position_diffusion_m2_per_second",
            "tracking_perched_relocation_snap_distance_m",
            "tracking_perched_relocation_gate_max_m",
            "tracking_perched_relocation_min_silence_seconds",
            "tracking_perched_relocation_ghost_seconds",
            "tracking_species_log_odds_enabled",
            "tracking_species_log_odds_label_categories",
            "tracking_species_log_odds_prior_probability",
            "tracking_species_log_odds_per_update_cap",
            "tracking_species_log_odds_total_cap",
            "tracking_species_log_odds_evidence_top_k",
            "tracking_species_display_min_belief",
            "tracking_species_belief_exposed_top_n",
            "tracking_partial_constraint_binding_enabled",
            "tracking_partial_binding_window_seconds",
            "tracking_partial_binding_bearing_sigma_deg",
            "tracking_partial_binding_hyperbola_sigma_m",
            "tracking_partial_binding_snr_level_margin_db",
            "tracking_partial_binding_max_normalized_residual",
            "tracking_partial_binding_species_top_k",
            "tracking_partial_binding_soft_update_gain",
            "tracking_partial_binding_ambiguity_score_ratio",
            "birdnet_two_threshold_enabled",
            "birdnet_localize_to_track_min_confidence",
            # --- end Field Mode W2-tracker ---
            # --- Field Mode S-fixes tracking ---
            "tracking_species_association_gate_enabled",
            "tracking_species_association_confident_probability",
            "tracking_species_association_detection_claim_probability",
            "tracking_perched_concurrent_perch_guard_enabled",
            "tracking_perched_concurrent_activity_seconds",
            # --- end Field Mode S-fixes tracking ---
            # --- Field Mode W3-birdnet-cutover ---
            "birdnet_v3_trigger_min_confidence",
            "birdnet_v3_localize_to_track_min_confidence",
            "birdnet_v3_species_display_min_belief",
            "birdnet_v3_label_migration_enabled",
            "birdnet_two_threshold_require_tracker_companions",
            # --- end Field Mode W3-birdnet-cutover ---
            # --- localization overhaul --- one position observation per solve
            "tracking_mode_by_category",
            "position_observation_queue_max_items",
            "position_observation_unclassified_track_policy",
            "position_observation_solve_ledger_max_entries",
            "position_observation_solve_id_memory_max_entries",
            "position_observation_held_fix_max_wait_seconds",
        ),
    ),
    (
        "fusion",
        "Fusion Pipeline",
        None,
        (
            "fusion_worker_count",
            "fusion_event_queue_size",
            "fusion_localization_queue_size",
            "fusion_classification_queue_size",
            "fusion_rules_queue_size",
            "drop_on_backpressure",
            "fusion_drop_on_backpressure",
            "fusion_backpressure_drop_policy",
            "fusion_report_window_localized_emission_cap",
            "fusion_offline_replay_mode",
            "localized_handoff_duplicate_merge_max_separation_m",  # localization overhaul
        ),
    ),
    # --- Field Mode W3-entity ---
    (
        "bird_entities",
        "Bird Entities",
        None,
        (
            "entity_reporting_enabled",
            "entity_reporting_label_categories",
            "entity_absorption_window_seconds",
            "entity_absorption_max_fix_distance_m",
            "entity_absorption_max_node_distance_m",
            "entity_absorption_max_bearing_difference_deg",
            "entity_pinpointed_max_horizontal_uncertainty_m",
            "entity_spatial_state_hold_seconds",
            "entity_expiry_seconds",
            "entity_bearing_default_half_width_deg",
            "entity_bearing_min_half_width_deg",
            "entity_bearing_max_half_width_deg",
            "entity_uncertainty_sigma_multiplier",
            "entity_notification_grace_seconds",
            "entity_maintenance_interval_seconds",
            "entity_max_tracked_entities",
            # --- Field Mode S-fixes tracking ---
            "entity_absorption_min_species_belief",
            "entity_absorption_covariance_gate_enabled",
            "entity_absorption_chi2_gate",
            "entity_absorption_min_fix_distance_m",
            "entity_concurrent_fix_window_seconds",
            "entity_bearing_evidence_absorption_enabled",
            "entity_canonical_row_fused_position_enabled",
            # --- end Field Mode S-fixes tracking ---
        ),
    ),
    # --- end Field Mode W3-entity ---
    (
        "rules_alerts",
        "Rules & Alerts",
        PipelineStageKind.RULES,
        ("rules_config_path",),
    ),
    (
        "storage_retention",
        "Storage & Retention",
        None,
        (
            "snippet_retention_seconds",
            "retention",
            "retention_yamnet_audio_seconds",
            "retention_birdnet_audio_seconds",
            "retention_drone_audio_seconds",
            "retention_alert_audio_seconds",
            "retention_detection_metadata_seconds",
            "retention_policy_path",
            "transcript_retention_seconds",
            "capture_final_tracks_settle_seconds",
        ),
    ),
    (
        "federation",
        "Federation",
        None,
        ("federation",),
    ),
    (
        # The "hass" block moved here out of rules_alerts once the MQTT bridge
        # became real: it is an integration in its own right, not a rules knob,
        # and this group id lines up with the /settings/integrations page.
        "integrations",
        "Integrations",
        None,
        ("hass",),
    ),
    (
        "system",
        "System & Site",
        None,
        (
            "cop",
            "default_temperature_c",
            "default_humidity",
            "environment_reading_max_age_seconds",
            "node_degraded_after_seconds",
            "node_offline_after_seconds",
            "site_origin",
        ),
    ),
)

# Read-only meta keys that carry no editable value (backend probe lists, override
# bookkeeping). Explicitly allowlisted so the coverage test stays exhaustive.
UNGROUPED_KEYS: frozenset[str] = frozenset(
    {
        "classifier_backends_available",
        "persisted_override_keys",
    }
)

# Union of every flat key that belongs to a stage group. Used by the pipeline
# graph builder to decide whether a `config_key` is deep-linkable.
GROUPED_CONFIG_KEYS: frozenset[str] = frozenset(
    key for (_id, _title, _stage, keys) in CONFIG_STAGE_GROUPS for key in keys
)

EXPOSED_CONFIG_KEYS: frozenset[str] = GROUPED_CONFIG_KEYS | UNGROUPED_KEYS


def group_flat_config(flat: dict) -> dict:
    """Regroup a flat ``get_config`` dict into stage groups + ungrouped.

    Returns ``{groups: [...], ungrouped: {...}, persisted_override_keys: [...]}``.
    Unknown keys (not in any group and not in UNGROUPED_KEYS) fall into
    ``ungrouped`` so the projection never silently drops a key.
    """
    groups_out = []
    consumed: set[str] = set()
    for group_id, title, stage, keys in CONFIG_STAGE_GROUPS:
        entries = []
        for key in keys:
            if key in flat:
                entries.append({"key": key, "value": flat[key]})
                consumed.add(key)
        groups_out.append(
            {
                "id": group_id,
                "title": title,
                "stage": stage.value if stage is not None else None,
                "entries": entries,
            }
        )
    ungrouped = {
        key: value
        for key, value in flat.items()
        if key not in consumed
    }
    return {
        "groups": groups_out,
        "ungrouped": ungrouped,
        "persisted_override_keys": flat.get("persisted_override_keys", []),
    }
