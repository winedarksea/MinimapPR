"""Localization resolution and reportability gating for Rust classifier renders.

Split out of ``localized_render_handoff.py`` to keep each file focused: this
module turns a render payload plus its solve geometry into a gated position
estimate; the handoff module owns registration, classification and reporting.

All functions read FusionNode state through ``host`` at call time (settings,
metrics and the shared gate helpers), so they behave exactly like the inline
code they replaced, including when tests monkeypatch those helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from minimappr.api.rust_dsp_manifests import LocalizedClassifierRenderRequest
from minimappr.core.ambi_atob import alias_cutoff_from_positions
from minimappr.core.amplitude_range import amplitude_range_prior_m
from minimappr.core.localization_uncertainty import (
    apply_frequency_covariance_scaling,
    covariance_to_nested_list,
)
from minimappr.core.range_projection import apply_unobservable_range_haircut, normalize_range_mode
from minimappr.core.rust_tdoa_solve import solve_localization_from_rust_tdoas

if TYPE_CHECKING:
    from minimappr.core.fusion_node import FusionNode
    from minimappr.core.localized_render_handoff_hooks import HandoffSolveGeometry
    from minimappr.core.node_position_covariance import NodePositionUncertainty

LOCALIZATION_METHOD_PYTHON_CARTESIAN_RUST_TDOA = "python_cartesian_rust_tdoa"
LOCALIZATION_REJECTED_OUT_OF_RANGE_REASON = "position_out_of_range"


@dataclass(slots=True)
class HandoffLocalizationEstimate:
    """One render's position estimate as it moves through resolution and gating.

    ``range_projection_mode`` holds the raw solver/sidecar mode until
    ``gate_handoff_localization`` canonicalizes it.
    """

    position_m: Any
    confidence: float
    gdop: float
    position_covariance_m2: list[list[float]] | None
    range_observability: float | None
    residual_rms_seconds: float | None
    range_projection_mode: str | None
    method: str

    @classmethod
    def from_render_payload(cls, payload: LocalizedClassifierRenderRequest) -> "HandoffLocalizationEstimate":
        return cls(
            position_m=payload.localization_position_m,
            confidence=payload.localization_confidence,
            gdop=payload.localization_gdop,
            position_covariance_m2=payload.localization_position_covariance_m2,
            range_observability=payload.localization_range_observability,
            residual_rms_seconds=payload.localization_residual_rms_seconds,
            range_projection_mode=payload.localization_range_projection_mode,
            method=payload.localization_method,
        )


@dataclass(slots=True)
class HandoffGatingOutcome:
    reporting_modality: str
    rejected_out_of_range: bool
    rejected_low_quality_reason: str | None
    capability_tier: str
    contributing_centroid_m: np.ndarray | None


def resolve_default_handoff_estimate(
    host: "FusionNode",
    payload: LocalizedClassifierRenderRequest,
    geometry: "HandoffSolveGeometry",
    reporting_modality: str,
) -> HandoffLocalizationEstimate:
    """Trust the sidecar's estimate, or re-home the solve to Python.

    With ``localization_single_node_solver == "python_cartesian"`` the sidecar's
    pairwise TDOAs + bearing are re-solved by the shared multi-node Cartesian
    estimator. Either way the canonical range mode + haircut apply later.
    """
    estimate = HandoffLocalizationEstimate.from_render_payload(payload)
    if not (
        python_cartesian_resolve_requested(host, payload, reporting_modality)
        and payload.localization_pair_tdoas
    ):
        return estimate
    solved = python_cartesian_estimate_from_rust_tdoas(
        host,
        payload,
        ordered_sensor_ids=geometry.ordered_pair_sensor_ids(),
        sensor_positions=geometry.ordered_pair_sensor_positions(),
        pair_tdoas=payload.localization_pair_tdoas,
        steering_direction=payload.localization_steering_direction,
    )
    return estimate if solved is None else solved


def solve_timing_sample_rate_hz(payload: LocalizedClassifierRenderRequest) -> int:
    """The rate the solve's TDOAs were measured at. A render's own
    ``sample_rate_hz`` is its audio's, so an event and its covering render
    would otherwise give one solve two timing-noise covariances."""
    solve = payload.localization_solve
    return solve.provenance.solve_sample_rate_hz if solve is not None else payload.sample_rate_hz


def resolve_handoff_estimate(
    host: "FusionNode",
    solution_resolver: Any,
    payload: LocalizedClassifierRenderRequest,
    geometry: "HandoffSolveGeometry",
    reporting_modality: str,
) -> HandoffLocalizationEstimate:
    """The solution seam's estimate, else the default one. Synchronous and free
    of event-loop state, so the handoff may run it in a worker thread."""
    estimate = solution_resolver.resolve_handoff_solution(
        payload=payload, geometry=geometry, reporting_modality=reporting_modality
    )
    if estimate is None:
        estimate = resolve_default_handoff_estimate(host, payload, geometry, reporting_modality)
    return estimate


def python_cartesian_resolve_requested(
    host: "FusionNode",
    payload: LocalizedClassifierRenderRequest,
    reporting_modality: str,
) -> bool:
    return (
        host.settings.localization_single_node_solver == "python_cartesian"
        and reporting_modality == "localized"
        and payload.localization_sound_speed_mps is not None
    )


def python_cartesian_estimate_from_rust_tdoas(
    host: "FusionNode",
    payload: LocalizedClassifierRenderRequest,
    *,
    ordered_sensor_ids: list[str],
    sensor_positions: dict[str, Any],
    pair_tdoas: list[dict],
    steering_direction: tuple[float, float, float] | None,
    node_position_uncertainty: "NodePositionUncertainty | None" = None,
) -> HandoffLocalizationEstimate | None:
    """Python Cartesian solve of the sidecar's pair TDOAs; ``None`` when unsolvable.

    ``ordered_sensor_ids`` must be the complete list the pair indices index,
    with a site position for every entry.
    """
    # Amplitude/SNR-informed range prior: only affects unobservable-range
    # projection; a range_refined solve ignores far_field_initial_range_m.
    far_field_range_m = host.settings.localization_far_field_default_range_m
    far_field_prior_radial_std_m: float | None = None
    cfg = host.localization_config
    if cfg.localization_amplitude_range_prior_enabled and payload.localization_received_level_dbfs is not None:
        prior_range_m, prior_clamped = amplitude_range_prior_m(
            payload.localization_received_level_dbfs,
            reference_source_level_db=cfg.localization_amplitude_reference_level_db,
            min_range_m=cfg.localization_amplitude_prior_min_range_m,
            max_range_m=cfg.localization_amplitude_prior_max_range_m,
        )
        far_field_range_m = prior_range_m
        far_field_prior_radial_std_m = cfg.localization_amplitude_prior_std_factor * prior_range_m
        host._metrics.localization_amplitude_prior_used_count += 1
        if prior_clamped:
            host._metrics.localization_amplitude_prior_clamped_count += 1
    solved = solve_localization_from_rust_tdoas(
        sensor_positions=sensor_positions,
        ordered_sensor_ids=ordered_sensor_ids,
        pair_tdoas=pair_tdoas,
        steering_direction=steering_direction,
        sound_speed_mps=payload.localization_sound_speed_mps,
        sample_rate_hz=solve_timing_sample_rate_hz(payload),
        interpolation_factor=host.settings.gcc_phat_interp_factor,
        far_field_default_range_m=far_field_range_m,
        far_field_prior_radial_std_m=far_field_prior_radial_std_m,
        # The radial search stops at the envelope instead of diverging to 10^9 m.
        far_field_max_range_m=host.settings.localization_far_field_max_range_m,
        bearing_strength=cfg.localization_node_bearing_strength,
        bearing_observed_obs_floor=getattr(cfg, "localization_obs_factor_bearing_floor", None),
        node_position_uncertainty=node_position_uncertainty,
    )
    if solved is None:
        host._metrics.localization_single_node_python_fallback_count += 1
        return None
    host._metrics.localization_single_node_python_solved_count += 1
    return HandoffLocalizationEstimate(
        position_m=solved.position_m,
        confidence=solved.confidence,
        gdop=solved.gdop,
        position_covariance_m2=solved.position_covariance_m2,
        range_observability=solved.range_observability,
        residual_rms_seconds=solved.residual_rms_seconds,
        range_projection_mode=solved.range_projection_mode,
        method=LOCALIZATION_METHOD_PYTHON_CARTESIAN_RUST_TDOA,
    )


def gate_handoff_localization(
    host: "FusionNode",
    *,
    payload: LocalizedClassifierRenderRequest,
    geometry: "HandoffSolveGeometry",
    estimate: HandoffLocalizationEstimate,
    reporting_modality: str,
    source_node_id: str,
    records_rejections: bool = True,
) -> HandoffGatingOutcome:
    """Apply the same reportability gates the solver path applies.

    This ingress carries precomputed positions from the Rust sidecar and is the
    path production actually uses, so it must gate as strictly as
    ``_build_localization_branch``. Mutates ``estimate`` in place.
    ``records_rejections`` False gates without counting (a covering render
    repeating a solve its event already gated and counted).
    """
    selected_positions = geometry.selected_positions
    contributing_centroid_m = (
        np.mean(
            np.vstack([np.asarray(selected_positions[sid], dtype=np.float64) for sid in selected_positions]),
            axis=0,
        )
        if selected_positions
        else None
    )
    rejected_out_of_range = reporting_modality == "localized" and not host._localization_position_is_reportable(
        estimate.position_m,
        contributing_centroid_m,
    )
    if rejected_out_of_range:
        # Keep the classification as an omni observation at its node rather than
        # letting a bogus geographic position into a localized track.
        if records_rejections:
            host._metrics.localization_rejected_out_of_range_count += 1
            host._record_silent_drop(stage="localization", reason=LOCALIZATION_REJECTED_OUT_OF_RANGE_REASON)
        reporting_modality = "omni"

    # Angular resolution degrades when the dominant frequency falls well below the
    # array's spatial-aliasing cutoff, so widen the covariance perpendicular to the
    # bearing (mirrors the multi-node dispatcher). Skipped when the sidecar omits
    # the dominant frequency.
    if (
        estimate.position_covariance_m2 is not None
        and payload.localization_dominant_frequency_hz is not None
        and payload.localization_sound_speed_mps is not None
    ):
        positions = np.vstack(
            [np.asarray(selected_positions[sid], dtype=np.float64) for sid in sorted(selected_positions)]
        )
        alias_cutoff_hz = alias_cutoff_from_positions(positions, c_sound=payload.localization_sound_speed_mps)
        centroid_m = np.mean(positions, axis=0)
        bearing_vec = np.asarray(estimate.position_m, dtype=np.float64) - centroid_m
        scaled_cov = apply_frequency_covariance_scaling(
            covariance_m2=np.asarray(estimate.position_covariance_m2, dtype=np.float64),
            bearing_unit_vec=bearing_vec,
            dominant_frequency_hz=payload.localization_dominant_frequency_hz,
            alias_cutoff_hz=alias_cutoff_hz,
        )
        estimate.position_covariance_m2 = covariance_to_nested_list(scaled_cov)

    # Canonicalize the range mode and apply the path-agnostic haircut so a
    # far-field single-node estimate cannot pass through over-confident.
    raw_range_projection_mode = estimate.range_projection_mode
    estimate.range_projection_mode = normalize_range_mode(raw_range_projection_mode)
    if records_rejections:
        host._record_range_projection_metrics(raw_range_projection_mode)
    estimate.confidence, estimate.range_observability = apply_unobservable_range_haircut(
        mode=estimate.range_projection_mode,
        confidence=estimate.confidence,
        range_observability=estimate.range_observability,
    )
    # Judged on the post-haircut confidence, like the solver path.
    rejected_low_quality_reason: str | None = None
    if reporting_modality == "localized":
        rejected_low_quality_reason = host._localization_quality_is_reportable(
            confidence=estimate.confidence,
            gdop=estimate.gdop,
        )
        if rejected_low_quality_reason is not None:
            if records_rejections:
                host._metrics.localization_rejected_low_quality_count += 1
                host._record_silent_drop(
                    stage="localization",
                    reason=rejected_low_quality_reason,
                    source_node_id=source_node_id,
                    range_projection_mode=estimate.range_projection_mode,
                    localization_confidence=estimate.confidence,
                    gdop=estimate.gdop,
                )
            reporting_modality = "omni"
    # Resolved before the covariance clamp so the clamp knows whether altitude
    # was actually observable on this solve.
    capability_tier = (
        "full_3d"
        if reporting_modality == "localized"
        and len(geometry.selected_sensor_ids) >= host.settings.min_sensors_for_3d
        else "classification_only"
    )
    estimate.position_covariance_m2 = host._clamp_localization_covariance(
        estimate.position_covariance_m2,
        position_m=estimate.position_m,
        contributing_centroid_m=contributing_centroid_m,
        capability_tier=capability_tier,
        range_projection_mode=estimate.range_projection_mode,
    )
    return HandoffGatingOutcome(
        reporting_modality=reporting_modality,
        rejected_out_of_range=rejected_out_of_range,
        rejected_low_quality_reason=rejected_low_quality_reason,
        capability_tier=capability_tier,
        contributing_centroid_m=contributing_centroid_m,
    )
