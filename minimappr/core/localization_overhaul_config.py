"""Settings keys added by the localization overhaul: defaults, names, validation.

``Settings`` (config.py) declares the fields; this module owns their default
values, allowed vocabularies and validation so config.py carries no new
operational constants. Keys in ``RUST_FORWARDED_LOCALIZATION_SETTING_KEYS`` are
passed to the ingest sidecar as ``MINIMAPPR_<KEY_UPPER>`` env vars (dict values
as JSON) by ``ingest_sidecar_runtime.py``; the Rust side reads the same names.
"""

from __future__ import annotations

import json
import math

from minimappr.models import TimeQuality

# --- Rust-forwarded defaults ---------------------------------------------------
# Upper bound on sensors in one solve; bounds SRP cost as clusters grow.
DEFAULT_LOCALIZATION_MAX_SOLVE_SENSORS = 16
# Pairs shorter than this carry almost no TDOA information at audio rates.
DEFAULT_LOCALIZATION_MIN_PAIR_BASELINE_M = 0.02
# A node whose measured timeline residual exceeds this is known to be wrong
# (not merely uncertain), so its cross-node pairs are dropped.
DEFAULT_LOCALIZATION_CROSS_NODE_MAX_TIMELINE_RESIDUAL_SECONDS = 0.010
# Applied when a node reports no recognisable time_quality. Conservative but
# not exclusionary: weak clocks widen the error, they never drop a pair.
DEFAULT_LOCALIZATION_CLOCK_UNKNOWN_UNCERTAINTY_SECONDS = 0.005
DEFAULT_LOCALIZATION_TIMESTAMP_STEP_THRESHOLD_SECONDS = 0.002
DEFAULT_LOCALIZATION_TIMESTAMP_STEP_RECOVERY_HOLDOFF_SECONDS = 30.0
DEFAULT_LOCALIZATION_TRIGGER_NOISE_FLOOR_RELATIVE_ENABLED = True
# Shared with the Python-ingest ``trigger_noise_floor_relative_snr_db`` so both
# ingest paths admit the same events by default.
DEFAULT_NOISE_FLOOR_RELATIVE_TRIGGER_SNR_DB = 8.0
# The relative trigger only admits windows >= its SNR above the floor, so a
# solver floor 3 dB above the noise floor never rejects an admitted faint event
# while still rejecting dead or muted channels.
DEFAULT_LOCALIZATION_SOLVER_FLOOR_RELATIVE_DB = 3.0
DEFAULT_LOCALIZATION_SENSOR_REGISTRY_STALE_SECONDS = 120.0

# Per-class clock uncertainty. A disciplined GPS/PPS clock is ~1 us; holdover
# drifts away from that (the reported holdover error wins when larger); NTP is
# ms-class; an undisciplined clock gets a wide (50 ms) uncertainty: weak clocks
# widen the error rather than being excluded (only the measured residual gates).
_CLOCK_CLASS_UNCERTAINTY_SECONDS_BY_TIME_QUALITY: dict[TimeQuality, float] = {
    TimeQuality.GPS_LOCKED: 1.0e-6,
    TimeQuality.GPS_HOLDOVER: 1.0e-4,
    TimeQuality.NTP_DISCIPLINED: 2.0e-3,
    TimeQuality.NTP_SYNC: 2.0e-3,
    TimeQuality.FREE_RUNNING: 5.0e-2,
}


def default_localization_clock_class_uncertainty_seconds() -> dict[str, float]:
    return {quality.value: seconds for quality, seconds in _CLOCK_CLASS_UNCERTAINTY_SECONDS_BY_TIME_QUALITY.items()}


# --- Python-only defaults ------------------------------------------------------
DEFAULT_SIDECAR_GEOMETRY_PUSH_ESTIMATE_MOVE_THRESHOLD_M = 0.5
DEFAULT_SIDECAR_GEOMETRY_RECONCILE_INTERVAL_SECONDS = 30.0
DEFAULT_LOCALIZED_HANDOFF_DUPLICATE_MERGE_MAX_SEPARATION_M = 15.0
DEFAULT_POSITION_OBSERVATION_QUEUE_MAX_ITEMS = 1024
DEFAULT_POINT_NETWORK_CLIP_CLASSIFICATION_ENABLED = True

TRACKING_MODE_CONTINUOUS_MOTION = "continuous_motion"
TRACKING_MODE_STATIONARY_HOLD = "stationary_hold"
TRACKING_MODE_NONE = "none"
TRACKING_MODES = frozenset({TRACKING_MODE_CONTINUOUS_MOTION, TRACKING_MODE_STATIONARY_HOLD, TRACKING_MODE_NONE})
# The mode for every label category not listed explicitly.
TRACKING_MODE_DEFAULT_CATEGORY_KEY = "default"


def default_tracking_mode_by_category() -> dict[str, str]:
    return {
        "wildlife": TRACKING_MODE_STATIONARY_HOLD,
        TRACKING_MODE_DEFAULT_CATEGORY_KEY: TRACKING_MODE_CONTINUOUS_MOTION,
    }


# An unclassified fix may start a track that stays hidden until classified, or
# must wait for a classification before any track starts.
POSITION_OBSERVATION_POLICY_START_HIDDEN_TENTATIVE = "start_hidden_tentative"
POSITION_OBSERVATION_POLICY_REQUIRE_CLASSIFICATION = "require_classification"
POSITION_OBSERVATION_UNCLASSIFIED_TRACK_POLICIES = frozenset(
    {POSITION_OBSERVATION_POLICY_START_HIDDEN_TENTATIVE, POSITION_OBSERVATION_POLICY_REQUIRE_CLASSIFICATION}
)
DEFAULT_POSITION_OBSERVATION_UNCLASSIFIED_TRACK_POLICY = POSITION_OBSERVATION_POLICY_START_HIDDEN_TENTATIVE

# --- Key registries ------------------------------------------------------------
RUST_FORWARDED_LOCALIZATION_SETTING_KEYS: tuple[str, ...] = (
    "localization_max_solve_sensors",
    "localization_min_pair_baseline_m",
    "localization_cross_node_max_timeline_residual_seconds",
    "localization_clock_class_uncertainty_seconds",
    "localization_clock_unknown_uncertainty_seconds",
    "localization_timestamp_step_threshold_seconds",
    "localization_timestamp_step_recovery_holdoff_seconds",
    "localization_trigger_noise_floor_relative_enabled",
    "localization_trigger_noise_floor_relative_db",
    "localization_solver_floor_relative_db",
    "localization_sensor_registry_stale_seconds",
)
PYTHON_ONLY_LOCALIZATION_OVERHAUL_SETTING_KEYS: tuple[str, ...] = (
    "sidecar_geometry_push_estimate_move_threshold_m",
    "sidecar_geometry_reconcile_interval_seconds",
    "localized_handoff_duplicate_merge_max_separation_m",
    "tracking_mode_by_category",
    "position_observation_queue_max_items",
    "position_observation_unclassified_track_policy",
    "point_network_clip_classification_enabled",
)
LOCALIZATION_OVERHAUL_SETTING_KEYS: tuple[str, ...] = (
    RUST_FORWARDED_LOCALIZATION_SETTING_KEYS + PYTHON_ONLY_LOCALIZATION_OVERHAUL_SETTING_KEYS
)


def localization_overhaul_flat_config_entries(settings) -> dict[str, object]:
    """Entries for the flat ``GET /api/v1/config`` surface (dicts copied)."""
    entries: dict[str, object] = {}
    for setting_key in LOCALIZATION_OVERHAUL_SETTING_KEYS:
        value = getattr(settings, setting_key)
        entries[setting_key] = dict(value) if isinstance(value, dict) else value
    return entries


def sidecar_environment_variable_name(setting_key: str) -> str:
    return f"MINIMAPPR_{setting_key.upper()}"


def sidecar_environment_value(value: object) -> str:
    """Render one setting for the sidecar env: bools lowercase, dicts JSON."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, dict):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return str(value)


def _is_positive_finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0.0


def localization_overhaul_setting_errors(settings) -> list[str]:
    errors: list[str] = []
    if settings.localization_max_solve_sensors < 2:
        errors.append("localization_max_solve_sensors must be >= 2 (one pair)")
    for key in (
        "localization_min_pair_baseline_m",
        "localization_cross_node_max_timeline_residual_seconds",
        "localization_clock_unknown_uncertainty_seconds",
        "localization_timestamp_step_threshold_seconds",
        "localization_sensor_registry_stale_seconds",
        "sidecar_geometry_push_estimate_move_threshold_m",
        "sidecar_geometry_reconcile_interval_seconds",
        "localized_handoff_duplicate_merge_max_separation_m",
    ):
        if not _is_positive_finite(getattr(settings, key)):
            errors.append(f"{key} must be a finite value > 0")
    if not (
        math.isfinite(settings.localization_timestamp_step_recovery_holdoff_seconds)
        and settings.localization_timestamp_step_recovery_holdoff_seconds >= 0.0
    ):
        errors.append("localization_timestamp_step_recovery_holdoff_seconds must be a finite value >= 0")
    if not _is_positive_finite(settings.localization_trigger_noise_floor_relative_db):
        errors.append("localization_trigger_noise_floor_relative_db must be a finite value > 0")
    if not math.isfinite(settings.localization_solver_floor_relative_db):
        errors.append("localization_solver_floor_relative_db must be finite")
    if settings.position_observation_queue_max_items < 1:
        errors.append("position_observation_queue_max_items must be >= 1")
    errors.extend(_clock_class_uncertainty_errors(settings.localization_clock_class_uncertainty_seconds))
    errors.extend(_tracking_mode_by_category_errors(settings.tracking_mode_by_category))
    if settings.position_observation_unclassified_track_policy not in POSITION_OBSERVATION_UNCLASSIFIED_TRACK_POLICIES:
        errors.append(
            "position_observation_unclassified_track_policy must be one of "
            f"{sorted(POSITION_OBSERVATION_UNCLASSIFIED_TRACK_POLICIES)}"
        )
    return errors


def _clock_class_uncertainty_errors(uncertainty_by_class: object) -> list[str]:
    if not isinstance(uncertainty_by_class, dict):
        return ["localization_clock_class_uncertainty_seconds must be a mapping of time_quality -> seconds"]
    known_classes = {quality.value for quality in TimeQuality}
    errors: list[str] = []
    for clock_class, seconds in uncertainty_by_class.items():
        if clock_class not in known_classes:
            errors.append(
                f"localization_clock_class_uncertainty_seconds has unknown time_quality {clock_class!r}; "
                f"expected one of {sorted(known_classes)}"
            )
        if not _is_positive_finite(seconds):
            errors.append(f"localization_clock_class_uncertainty_seconds[{clock_class!r}] must be a finite value > 0")
    return errors


def _tracking_mode_by_category_errors(mode_by_category: object) -> list[str]:
    if not isinstance(mode_by_category, dict):
        return ["tracking_mode_by_category must be a mapping of label category -> tracking mode"]
    errors: list[str] = []
    if TRACKING_MODE_DEFAULT_CATEGORY_KEY not in mode_by_category:
        errors.append(f"tracking_mode_by_category must include a {TRACKING_MODE_DEFAULT_CATEGORY_KEY!r} entry")
    for category, mode in mode_by_category.items():
        if mode not in TRACKING_MODES:
            errors.append(
                f"tracking_mode_by_category[{category!r}] must be one of {sorted(TRACKING_MODES)}; got {mode!r}"
            )
    return errors
