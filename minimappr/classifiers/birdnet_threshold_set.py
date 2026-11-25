"""Per-model-version BirdNET threshold set + the two-threshold tracker constraint.

v2.4 and V3.0 confidences are on different scales, so each model version owns
its own threshold set; ``birdnet_model_version`` picks which one is live.
The v2.4 set is the long-standing keys (``birdnet_trigger_min_confidence``,
``birdnet_localize_to_track_min_confidence``,
``tracking_species_display_min_belief``); the V3 set is the
``birdnet_v3_*`` keys whose defaults were derived by quantile matching
(``scripts/calibrate_birdnet_v3_thresholds.py``, see
``docs/field_mode_progress/birdnet_cutover.md``). Consumers read
``Settings.birdnet_threshold_set()`` instead of the raw keys, so switching the
version swaps every threshold at once.

Constraint (W2-tracker measurements): two-threshold BirdNET without the
perched filter and species log-odds lands weak, confused detections on the
wrong constant-velocity track (swap rate 0.27 vs 0). It is warned about at
startup, and becomes a validation error with
``birdnet_two_threshold_require_tracker_companions``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from minimappr.core.tracking_mode_config import TrackingModeConfig

logger = logging.getLogger(__name__)

BIRDNET_V3_MODEL_VERSION = "3.0"
BIRDNET_V24_MODEL_VERSION = "2.4"
# ``features["model"]`` written by the V3 backend (mirrors
# birdnet_v3.BIRDNET_V3_MODEL_NAME without importing onnxruntime here).
BIRDNET_V3_MODEL_FEATURE_VALUE = "birdnet_v3"
UNIT_INTERVAL_V3_THRESHOLD_KEYS: tuple[str, ...] = (
    "birdnet_v3_trigger_min_confidence",
    "birdnet_v3_localize_to_track_min_confidence",
    "birdnet_v3_species_display_min_belief",
)
# Weak detections need the wildlife tracker to hold perches and fuse species
# belief, or they land on the wrong track.
TWO_THRESHOLD_WILDLIFE_CATEGORY = "wildlife"
TWO_THRESHOLD_STATIONARY_HOLD_COMPANION = "tracking_mode_by_category[wildlife]=stationary_hold"
TWO_THRESHOLD_SPECIES_LOG_ODDS_COMPANION = "tracking_species_log_odds_enabled"


@dataclass(frozen=True, slots=True)
class BirdNETThresholdSet:
    """The thresholds that move with the BirdNET model version."""

    model_version: str
    # Display / trigger floor: results below are "unknown" (or, with two
    # thresholds, below_display_threshold). Also the retention confident bucket.
    trigger_min_confidence: float
    # Two-threshold low floor: results down to here feed track belief.
    localize_to_track_min_confidence: float
    # Track species belief needed before a sub-display-only track is shown.
    species_display_min_belief: float


def birdnet_threshold_set_for(values) -> BirdNETThresholdSet:
    """Threshold set for ``values`` (Settings or a merged PATCH view)."""
    version = str(getattr(values, "birdnet_model_version", "2.4")).strip()
    if version == BIRDNET_V3_MODEL_VERSION:
        return BirdNETThresholdSet(
            model_version=version,
            trigger_min_confidence=float(values.birdnet_v3_trigger_min_confidence),
            localize_to_track_min_confidence=float(values.birdnet_v3_localize_to_track_min_confidence),
            species_display_min_belief=float(values.birdnet_v3_species_display_min_belief),
        )
    return BirdNETThresholdSet(
        model_version=version,
        trigger_min_confidence=float(values.birdnet_trigger_min_confidence),
        localize_to_track_min_confidence=float(values.birdnet_localize_to_track_min_confidence),
        species_display_min_belief=float(values.tracking_species_display_min_belief),
    )


def birdnet_model_version_for_classification_features(features: dict | None) -> str:
    """Which BirdNET model scored a stored/assembled detection ("2.4" or "3.0").

    Retention re-judges every stored row on each cleanup pass, so a
    confidence-gated BirdNET rule must use the row's own scale, not the
    currently configured model: switching to V3 must not re-bucket (and
    delete) rows v2.4 scored, nor the reverse. The composite classifier
    prefixes non-primary member features with ``<member_id>:``; rows without
    V3 provenance (incl. every pre-V3 row) are v2.4.
    """
    if not features:
        return BIRDNET_V24_MODEL_VERSION
    winner_member = features.get("winner_member")
    model_name = features.get(f"{winner_member}:model") if winner_member else None
    if model_name is None:
        model_name = features.get("model")
    if str(model_name or "").strip().lower() == BIRDNET_V3_MODEL_FEATURE_VALUE:
        return BIRDNET_V3_MODEL_VERSION
    return BIRDNET_V24_MODEL_VERSION


def two_threshold_missing_companions(values) -> list[str]:
    """Companion flags that are off while two-threshold BirdNET is on."""
    if not bool(getattr(values, "birdnet_two_threshold_enabled", False)):
        return []
    missing: list[str] = []
    if not _wildlife_holds_station(values):
        missing.append(TWO_THRESHOLD_STATIONARY_HOLD_COMPANION)
    if not bool(getattr(values, TWO_THRESHOLD_SPECIES_LOG_ODDS_COMPANION, False)):
        missing.append(TWO_THRESHOLD_SPECIES_LOG_ODDS_COMPANION)
    return missing


def _wildlife_holds_station(values) -> bool:
    try:
        modes = TrackingModeConfig.from_mapping(getattr(values, "tracking_mode_by_category", None))
    except ValueError:
        # An invalid mapping is reported by its own validation.
        return True
    return modes.holds_station(TWO_THRESHOLD_WILDLIFE_CATEGORY)


def two_threshold_companion_message(missing: list[str]) -> str:
    return (
        "birdnet_two_threshold_enabled without "
        + " and ".join(missing)
        + ": weak detections land on the wrong track (W2-tracker: swap rate 0.27 vs 0 with both on)"
    )


def birdnet_cutover_setting_errors(values) -> list[str]:
    """Validation errors for the W3-birdnet-cutover keys."""
    errors: list[str] = []
    for key in UNIT_INTERVAL_V3_THRESHOLD_KEYS:
        if not 0.0 <= float(getattr(values, key)) <= 1.0:
            errors.append(f"{key}: must be in [0, 1]")
    if bool(getattr(values, "birdnet_two_threshold_require_tracker_companions", False)):
        missing = two_threshold_missing_companions(values)
        if missing:
            errors.append(two_threshold_companion_message(missing))
    return errors


def warn_on_two_threshold_without_companions(values) -> None:
    missing = two_threshold_missing_companions(values)
    if missing:
        logger.warning("%s; enable them or set birdnet_two_threshold_enabled=false", two_threshold_companion_message(missing))
