"""Audio modality vs spatial status of a Rust render, and the solve-time bearing.

A render's *audio* modality says what the clip is: the sidecar falls back to an
omni render when its solve is below ``min_localization_confidence`` (or for
another fallback reason). Its *spatial* status says whether the solve places
the source. The two used to be one flag, so an omni clip threw away a
supported srp_phat bearing that the event-only manifest of the same solve
still carried. Now an omni-audio render whose solve is a positioned srp_phat
result with a steering direction gets the same re-solve, gates and tracker path
as a localized render, provided the sidecar's solve confidence reaches
``omni_audio_bearing_min_localization_confidence`` (below it the render stays
omni at the node, as before, and records no solve-time bearing: promoting every
sub-threshold steering direction put noise-only solves onto tracks). The
solve-ID ledger keeps a promoted solve counted once. Fallbacks
with no solve (``single_point_node``, ``insufficient_channels``, coverage) and
a degraded solve-contract violation stay omni at the node.

``solve_time_bearing`` states the solve's ray for the detection: origin at the
contributing-sensor centroid (where the solver projects from), compass
direction and elevation toward the gated position, half width from the
covariance's tangential spread (the entity fan's formula and limits), and the
radial status.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np

from minimappr.api.rust_solve_contract_degradation import SOLVE_CONTRACT_VIOLATION_FALLBACK_REASON
from minimappr.core.bird_entity_model import bearing_estimate_toward
from minimappr.models import UNOBSERVABLE_RANGE_DISPLAY_MODES, DetectionSolveBearing

if TYPE_CHECKING:
    from minimappr.api.rust_dsp_manifests import LocalizedClassifierRenderRequest
    from minimappr.core.geo import LocalCoordinateFrame

ReportingModality = Literal["localized", "omni"]

BIRDNET_OMNI_FALLBACK_RENDER_KIND = "birdnet_omni_fallback"
# The only sidecar solver whose result carries a steering direction.
QUALIFIED_BEARING_LOCALIZATION_METHOD = "srp_phat"
LOCALIZATION_SOLVE_BEARING_FEATURE_KEY = "localization_solve_bearing"
CLASSIFICATION_AUDIO_MODALITY_FEATURE_KEY = "classification_audio_modality"
# A bearing-projected range is a placeholder distance along a supported ray.
BEARING_PROJECTED_RANGE_MODE = "range_bearing_projected"
# Below this the position sits on the centroid and states no direction.
MINIMUM_BEARING_RAY_LENGTH_M = 1e-6


def render_audio_modality(payload: "LocalizedClassifierRenderRequest") -> ReportingModality:
    """What the render's clip is: omni when the sidecar fell back, for any reason."""
    if payload.render_kind == BIRDNET_OMNI_FALLBACK_RENDER_KIND or payload.fallback_reason is not None:
        return "omni"
    return payload.reporting_modality


def solve_carries_qualified_bearing(payload: "LocalizedClassifierRenderRequest") -> bool:
    """A positioned srp_phat solve with a steering direction and provenance.

    Provenance is required so the render joins its event-only solve by
    ``solve_id`` and the solve contributes tracker evidence once.
    """
    return (
        payload.localization_solve is not None
        and payload.fallback_reason != SOLVE_CONTRACT_VIOLATION_FALLBACK_REASON
        and payload.localization_method == QUALIFIED_BEARING_LOCALIZATION_METHOD
        and payload.localization_steering_direction is not None
    )


def omni_audio_bearing_below_confidence_floor(
    payload: "LocalizedClassifierRenderRequest", *, min_localization_confidence: float
) -> bool:
    """An omni clip whose qualified solve is too weak to promote: the sidecar's
    solve confidence (the value its own render threshold compares) is below the floor."""
    return (
        render_audio_modality(payload) == "omni"
        and solve_carries_qualified_bearing(payload)
        and payload.localization_confidence < min_localization_confidence
    )


def handoff_spatial_modality(
    payload: "LocalizedClassifierRenderRequest", *, omni_audio_bearing_min_localization_confidence: float
) -> ReportingModality:
    """Spatial status entering the gates: an omni clip keeps a qualified solve
    whose confidence reaches the floor."""
    if render_audio_modality(payload) == "localized":
        return "localized"
    if not solve_carries_qualified_bearing(payload):
        return "omni"
    below_floor = omni_audio_bearing_below_confidence_floor(
        payload, min_localization_confidence=omni_audio_bearing_min_localization_confidence
    )
    return "omni" if below_floor else "localized"


def solve_time_bearing(
    *,
    reporting_node_id: str,
    contributing_centroid_m: np.ndarray,
    position_m,
    covariance_m2: list[list[float]] | None,
    range_projection_mode: str | None,
    coordinate_frame: "LocalCoordinateFrame",
    settings,
) -> DetectionSolveBearing | None:
    """The gated solve's ray from the contributing centroid; ``None`` when the
    position sits on the centroid (no direction)."""
    origin = np.asarray(contributing_centroid_m, dtype=np.float64)[:3]
    target = np.asarray(position_m, dtype=np.float64)[:3]
    offset = target - origin
    if not np.all(np.isfinite(offset)) or float(np.linalg.norm(offset)) < MINIMUM_BEARING_RAY_LENGTH_M:
        return None
    range_observable = (
        range_projection_mode not in UNOBSERVABLE_RANGE_DISPLAY_MODES
        and range_projection_mode != BEARING_PROJECTED_RANGE_MODE
    )
    origin_m = tuple(float(value) for value in origin)
    estimate = bearing_estimate_toward(
        origin_node_id=reporting_node_id,
        origin_position_m=origin_m,
        target_position_m=tuple(float(value) for value in target),
        covariance_m2=covariance_m2,
        range_observable=range_observable,
        default_half_width_deg=settings.entity_bearing_default_half_width_deg,
        min_half_width_deg=settings.entity_bearing_min_half_width_deg,
        max_half_width_deg=settings.entity_bearing_max_half_width_deg,
        sigma_multiplier=settings.entity_uncertainty_sigma_multiplier,
    )
    horizontal_m = math.hypot(float(offset[0]), float(offset[1]))
    return DetectionSolveBearing(
        origin_node_id=estimate.origin_node_id,
        origin_position_m=estimate.origin_position_m,
        origin_geo=coordinate_frame.local_to_geo(origin_m),
        bearing_deg=estimate.bearing_deg,
        half_width_deg=estimate.half_width_deg,
        range_min_m=estimate.range_min_m,
        range_max_m=estimate.range_max_m,
        elevation_deg=math.degrees(math.atan2(float(offset[2]), horizontal_m)),
        range_projection_mode=range_projection_mode,
        range_observable=range_observable,
    )
