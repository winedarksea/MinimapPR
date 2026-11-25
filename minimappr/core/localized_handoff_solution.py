"""Position estimate for a Rust render whose solve carries provenance.

Solution seam implementation (plan item P2). Given the geometry
``ManifestSolveGeometryResolver`` rebuilt from the solve, either:

- ``rust`` solver: adopt the sidecar's own estimate, mapped from the solve
  frame into the site frame (a no-op except for node-body solves); or
- ``python_cartesian``: re-solve the admitted pair TDOAs on the solve's full
  ordered sensor list and snapshot positions. The id map is complete by
  construction, so a pair can never be read against the wrong microphones.

Legacy renders (no provenance) and rejected geometry return ``None`` so the
handoff keeps its default path / omni handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from minimappr.core.localized_render_localization import (
    HandoffLocalizationEstimate,
    python_cartesian_estimate_from_rust_tdoas,
    python_cartesian_resolve_requested,
)

if TYPE_CHECKING:
    from minimappr.api.rust_dsp_manifests import LocalizedClassifierRenderRequest
    from minimappr.core.fusion_node import FusionNode
    from minimappr.core.localized_render_handoff_hooks import HandoffSolveGeometry


class SnapshotGeometrySolutionResolver:
    def __init__(self, host: "FusionNode") -> None:
        self._host = host

    def resolve_handoff_solution(
        self,
        *,
        payload: "LocalizedClassifierRenderRequest",
        geometry: "HandoffSolveGeometry",
        reporting_modality: str,
    ) -> HandoffLocalizationEstimate | None:
        solve = payload.localization_solve
        if solve is None or geometry.rejected_reason is not None:
            return None
        sidecar_estimate = _sidecar_estimate_in_site_frame(payload, geometry)
        if not (python_cartesian_resolve_requested(self._host, payload, reporting_modality) and solve.admitted_pair_tdoas):
            return sidecar_estimate
        steering_direction = payload.localization_steering_direction
        if steering_direction is not None and geometry.solve_frame_to_site is not None:
            steering_direction = geometry.solve_frame_to_site.direction(steering_direction)
        resolved = python_cartesian_estimate_from_rust_tdoas(
            self._host,
            payload,
            ordered_sensor_ids=geometry.ordered_pair_sensor_ids(),
            sensor_positions=geometry.ordered_pair_sensor_positions(),
            pair_tdoas=[
                {
                    "ch_a": pair.ch_a,
                    "ch_b": pair.ch_b,
                    "lag_seconds": pair.lag_seconds,
                    "confidence": pair.confidence,
                    "timing_uncertainty_seconds": pair.timing_uncertainty_seconds,
                }
                for pair in solve.admitted_pair_tdoas
            ],
            steering_direction=steering_direction,
        )
        return sidecar_estimate if resolved is None else resolved


def _sidecar_estimate_in_site_frame(
    payload: "LocalizedClassifierRenderRequest",
    geometry: "HandoffSolveGeometry",
) -> HandoffLocalizationEstimate:
    estimate = HandoffLocalizationEstimate.from_render_payload(payload)
    transform = geometry.solve_frame_to_site
    if transform is None:
        return estimate
    estimate.position_m = transform.position_m(estimate.position_m)
    if estimate.position_covariance_m2 is not None:
        estimate.position_covariance_m2 = transform.covariance_m2(estimate.position_covariance_m2)
    return estimate
