"""Field Mode W3-entity: bird-entity reporting sub-config and validation.

Built by ``Settings.entity_reporting_config()``. Every knob of the entity state
machine, the per-entity absorption window and the notification grace window
lives here so ``bird_entity_*`` modules carry no operational constants.
"""

from __future__ import annotations

from dataclasses import dataclass

ENTITY_REPORTING_SETTING_KEY_PREFIX = "entity_"


@dataclass(frozen=True, slots=True)
class EntityReportingConfig:
    enabled: bool
    # Detections of these categories go through entities; others keep the
    # legacy bucketed ReportingFusionPolicy.
    label_categories: frozenset[str]
    # Sliding per-entity window: a detection absorbs into an entity last heard
    # within this many seconds (measured from the entity, not a fixed bucket).
    absorption_window_seconds: float
    # A fix within this distance of a spatial entity's position is the same bird.
    absorption_max_fix_distance_m: float
    # An unpositioned detection absorbs when its reporting node is within this
    # distance of the entity (stands in for the per-node SNR-profile fit).
    absorption_max_node_distance_m: float
    # A bearing-only detection absorbs when its bearing from the reporting node
    # is within this angle of the bearing to the entity.
    absorption_max_bearing_difference_deg: float
    # Pinpointed while the 1-sigma horizontal major axis stays within this.
    pinpointed_max_horizontal_uncertainty_m: float
    # Without a spatial update for this long, a spatial entity steps back one state.
    spatial_state_hold_seconds: float
    # Not heard for this long -> the entity expires.
    expiry_seconds: float
    bearing_default_half_width_deg: float
    bearing_min_half_width_deg: float
    bearing_max_half_width_deg: float
    # Bearing fans and range bands are drawn at this many sigma.
    uncertainty_sigma_multiplier: float
    # Rules on a new unlocalized entity wait this long for a fix (0 = fire at once).
    notification_grace_seconds: float
    maintenance_interval_seconds: float
    max_tracked_entities: int
    # --- Field Mode S-fixes tracking ---
    # A label absorbs into an entity only when it is the entity's top species or
    # its belief is at least this (prior-level secondary scores are not "believed").
    absorption_min_species_belief: float = 0.3
    # A fix absorbs into a spatial entity when its horizontal Mahalanobis distance
    # (fix + entity covariance) is within the chi-square gate (2 dof; 9.21 = 99 %),
    # or it lies within the min distance (jitter floor for overconfident covariances).
    # Without a fix covariance the fixed ``absorption_max_fix_distance_m`` applies.
    absorption_covariance_gate_enabled: bool = True
    absorption_chi2_gate: float = 9.21
    absorption_min_fix_distance_m: float = 3.0
    # Two fixes of one species this close in time and inconsistent beyond their
    # covariance are two birds (one bird cannot sing from both perches).
    concurrent_fix_window_seconds: float = 2.0
    # A positioned fix whose own horizontal uncertainty exceeds the pinpoint limit
    # (or a bearing-projected range) is bearing evidence: it absorbs into a
    # better-located entity without moving it or rewriting its row.
    bearing_evidence_absorption_enabled: bool = True
    # The canonical row carries the entity's fused (tracked) position, not the raw fix.
    canonical_row_fused_position_enabled: bool = True
    # --- end Field Mode S-fixes tracking ---

    def applies_to(self, label_category: str | None) -> bool:
        return self.enabled and (label_category or "").strip().lower() in self.label_categories


def parse_label_categories(raw: str) -> frozenset[str]:
    return frozenset(part.strip().lower() for part in str(raw).split(",") if part.strip())


def entity_reporting_setting_errors(values) -> list[str]:
    """Errors for the W3-entity keys on *values* (Settings or a merged view)."""
    errors: list[str] = []
    for key in (
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
        "entity_maintenance_interval_seconds",
        "entity_absorption_chi2_gate",  # Field Mode S-fixes tracking
    ):
        if float(getattr(values, key)) <= 0.0:
            errors.append(f"{key}: must be > 0")
    if float(values.entity_notification_grace_seconds) < 0.0:
        errors.append("entity_notification_grace_seconds: must be >= 0")
    # --- Field Mode S-fixes tracking ---
    for key in ("entity_absorption_min_fix_distance_m", "entity_concurrent_fix_window_seconds"):
        if float(getattr(values, key)) < 0.0:
            errors.append(f"{key}: must be >= 0")
    if not 0.0 <= float(values.entity_absorption_min_species_belief) <= 1.0:
        errors.append("entity_absorption_min_species_belief: must be in [0, 1]")
    # --- end Field Mode S-fixes tracking ---
    if int(values.entity_max_tracked_entities) < 1:
        errors.append("entity_max_tracked_entities: must be >= 1")
    if float(values.entity_bearing_min_half_width_deg) > float(values.entity_bearing_max_half_width_deg):
        errors.append("entity_bearing_min_half_width_deg: must be <= entity_bearing_max_half_width_deg")
    if float(values.entity_expiry_seconds) < float(values.entity_absorption_window_seconds):
        errors.append("entity_expiry_seconds: must be >= entity_absorption_window_seconds")
    if not parse_label_categories(values.entity_reporting_label_categories):
        errors.append("entity_reporting_label_categories: needs at least one category")
    return errors


def entity_reporting_config_from_settings(settings) -> EntityReportingConfig:
    return EntityReportingConfig(
        enabled=bool(settings.entity_reporting_enabled),
        label_categories=parse_label_categories(settings.entity_reporting_label_categories),
        absorption_window_seconds=float(settings.entity_absorption_window_seconds),
        absorption_max_fix_distance_m=float(settings.entity_absorption_max_fix_distance_m),
        absorption_max_node_distance_m=float(settings.entity_absorption_max_node_distance_m),
        absorption_max_bearing_difference_deg=float(settings.entity_absorption_max_bearing_difference_deg),
        pinpointed_max_horizontal_uncertainty_m=float(settings.entity_pinpointed_max_horizontal_uncertainty_m),
        spatial_state_hold_seconds=float(settings.entity_spatial_state_hold_seconds),
        expiry_seconds=float(settings.entity_expiry_seconds),
        bearing_default_half_width_deg=float(settings.entity_bearing_default_half_width_deg),
        bearing_min_half_width_deg=float(settings.entity_bearing_min_half_width_deg),
        bearing_max_half_width_deg=float(settings.entity_bearing_max_half_width_deg),
        uncertainty_sigma_multiplier=float(settings.entity_uncertainty_sigma_multiplier),
        notification_grace_seconds=float(settings.entity_notification_grace_seconds),
        maintenance_interval_seconds=float(settings.entity_maintenance_interval_seconds),
        max_tracked_entities=int(settings.entity_max_tracked_entities),
        # --- Field Mode S-fixes tracking ---
        absorption_min_species_belief=float(settings.entity_absorption_min_species_belief),
        absorption_covariance_gate_enabled=bool(settings.entity_absorption_covariance_gate_enabled),
        absorption_chi2_gate=float(settings.entity_absorption_chi2_gate),
        absorption_min_fix_distance_m=float(settings.entity_absorption_min_fix_distance_m),
        concurrent_fix_window_seconds=float(settings.entity_concurrent_fix_window_seconds),
        bearing_evidence_absorption_enabled=bool(settings.entity_bearing_evidence_absorption_enabled),
        canonical_row_fused_position_enabled=bool(settings.entity_canonical_row_fused_position_enabled),
        # --- end Field Mode S-fixes tracking ---
    )
