"""Per-solve helpers of the render handoff (``core/localized_render_handoff.py``).

Kept apart so the handoff module stays about flow: audio validation, the
``rust_*`` provenance features stamped on a render's classification, and the
omni re-positioning at the reporting node.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from minimappr.api.rust_dsp_manifests import LocalizedClassifierRenderRequest
from minimappr.core.localized_render_handoff_hooks import HandoffSolveGeometry
from minimappr.core.localized_render_localization import (
    LOCALIZATION_REJECTED_OUT_OF_RANGE_REASON,
    HandoffGatingOutcome,
    HandoffLocalizationEstimate,
)
from minimappr.models import NodeSpec

if TYPE_CHECKING:
    from minimappr.core.fusion_node import FusionNode

OMNI_RENDER_LOCALIZATION_METHOD = "rust_classifier_render_fallback"


def validated_mono_render_audio(payload: LocalizedClassifierRenderRequest) -> np.ndarray:
    normalized_audio = np.asarray(payload.decoded_audio, dtype=np.float32)
    if normalized_audio.ndim != 1 or normalized_audio.size == 0:
        raise ValueError("Localized render audio must be a non-empty mono signal")
    if payload.sample_rate_hz <= 0:
        raise ValueError("Localized render sample_rate_hz must be positive")
    return normalized_audio


def rust_render_features(
    payload: LocalizedClassifierRenderRequest,
    gating: HandoffGatingOutcome,
    geometry: HandoffSolveGeometry,
) -> dict[str, Any]:
    rust_extra_features: dict[str, Any] = {"rust_manifest_id": payload.manifest_id}
    if payload.render_kind is not None:
        rust_extra_features["rust_render_kind"] = payload.render_kind
    if payload.render_start_ns is not None:
        rust_extra_features["rust_render_start_ns"] = int(payload.render_start_ns)
    if payload.render_end_ns is not None:
        rust_extra_features["rust_render_end_ns"] = int(payload.render_end_ns)
    if payload.fallback_reason is not None:
        rust_extra_features["rust_fallback_reason"] = payload.fallback_reason
    if geometry.rejected_reason is not None:
        rust_extra_features["localization_rejected_reason"] = geometry.rejected_reason
    elif gating.rejected_out_of_range:
        rust_extra_features["localization_rejected_reason"] = LOCALIZATION_REJECTED_OUT_OF_RANGE_REASON
    elif gating.rejected_low_quality_reason is not None:
        rust_extra_features["localization_rejected_reason"] = gating.rejected_low_quality_reason
    return rust_extra_features


def replace_estimate_with_node_position(
    host: "FusionNode",
    estimate: HandoffLocalizationEstimate,
    node: NodeSpec,
) -> None:
    """Position an omni detection at its source node; it cannot create a track.

    ``node`` is the registry spec, so an operator position override wins over
    the packet's reported position. A render that arrived "localized" can be
    degraded to omni by the gates, so it needs the same node-position guard as
    a render that arrived omni. The raw sidecar localization survives only as
    provenance in the features.
    """
    if node.position_m is None:
        raise ValueError("Omni classifier render requires node.position_m")
    estimate.position_m = tuple(float(value) for value in node.position_m)
    estimate.confidence = host.fusion_config.fallback_localization_confidence
    estimate.gdop = float("inf")
    estimate.position_covariance_m2 = None
    estimate.range_observability = None
    estimate.residual_rms_seconds = None
    estimate.range_projection_mode = None
    estimate.method = OMNI_RENDER_LOCALIZATION_METHOD


async def classify_render_audio(orchestrator, payload: LocalizedClassifierRenderRequest, normalized_audio: np.ndarray):
    """The render's own classification: the sidecar's authoritative label, else the mono render."""
    if payload.authoritative_classification is not None:
        classified = await orchestrator.adopt_authoritative_classification(
            classification=payload.authoritative_classification,
            event_time_ns=payload.event_time_ns,
            classification_signal=normalized_audio,
            sample_rate_hz=payload.sample_rate_hz,
        )
        classified.classification.features["rust_classification_authoritative"] = True
        return classified
    return await orchestrator.classify_omni_only(
        reference_signal=normalized_audio,
        sample_rate_hz=payload.sample_rate_hz,
        event_time_ns=payload.event_time_ns,
    )
