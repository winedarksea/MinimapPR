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
            "trigger_cooldown_seconds",
            "omni_scan_enabled",
            "omni_scan_interval_seconds",
            "omni_scan_window_seconds",
            "omni_scan_min_rms",
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
        ),
    ),
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
