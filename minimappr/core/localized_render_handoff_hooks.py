"""Collaborator seams for the localization overhaul (plan items P2 and P5).

FusionNode builds its seam collaborators through
``build_localization_seam_collaborators`` and never constructs them itself, so
later work plugs real implementations in here without editing
``fusion_node.py``. The P2 seams (geometry, solution, provenance) are live and
act only on renders whose solve carries provenance; a legacy render keeps the
pre-overhaul path. The observation sink is the P5 ``PositionObservationPipeline``
(one tracker update per solve).

Seams and their contracts:

- ``HandoffGeometryResolver`` (P2): may replace the registry-derived solve
  geometry with the manifest's ordered sensor ids and solve-time positions.
  ``None`` keeps the registry geometry.
- ``HandoffSolutionResolver`` (P2): may supply the position estimate for a
  render (e.g. a re-solve on the manifest geometry). ``None`` keeps the default
  path (sidecar estimate, or the ``python_cartesian`` re-solve). Whatever it
  returns still passes the handoff's reportability gates.
- ``LocalizationProvenanceRecorder`` (P2): records per-solve provenance once
  the localized product exists. It may add keys to
  ``localized_product.extra_classification_features``; those reach the stored
  detection's classification features.
- ``PositionObservationSink`` (P5): receives exactly one call per solve, after
  localization quality checks and before the classification queue/reporting,
  so no classification-side suppression can starve the tracker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from minimappr.api.rust_dsp_manifests import LocalizedClassifierRenderRequest
    from minimappr.core.fusion_node import FusionNode
    from minimappr.core.fusion_stage_carriers import LocalizedCandidate
    from minimappr.core.localized_handoff_geometry import SolveFrameToSiteTransform
    from minimappr.core.localized_render_localization import HandoffLocalizationEstimate
    from minimappr.models import NodeSpec

# Where a solve came from; lets the observation sink apply origin-specific rules
# (e.g. only event-triggered solves may start a provisional track).
SOLVE_ORIGIN_RUST_CLASSIFIER_RENDER = "rust_classifier_render"
SOLVE_ORIGIN_PYTHON_INGEST = "python_ingest"
# An event-only ``localization_result`` (no render): position only, never classified.
SOLVE_ORIGIN_RUST_EVENT_LOCALIZATION = "rust_event_localization"


@dataclass(frozen=True, slots=True)
class HandoffSolveGeometry:
    """Sensors a render's solve used, in solve order, with their site positions.

    ``selected_*`` are the contributing sensors: selection, centroid, range
    gate and capability tier follow them. ``pair_index_sensor_ids`` /
    ``pair_index_positions`` are the solve's full ordered sensor list (pair
    channel ``i`` is entry ``i``) when it differs from the selection; ``None``
    means the selection itself is that list (legacy registry geometry).
    ``solve_frame_to_site`` maps the sidecar's solve-frame outputs (position,
    covariance, steering) into the site frame; ``None`` means they already are.
    ``rejected_reason`` set means the solve cannot be placed in the site frame
    and the render must be reported omni with that reason.
    """

    selected_sensor_ids: list[str]
    selected_positions: dict[str, Any]
    reference_sensor: str
    pair_index_sensor_ids: list[str] | None = None
    pair_index_positions: dict[str, Any] | None = None
    solve_frame_to_site: "SolveFrameToSiteTransform | None" = None
    rejected_reason: str | None = None

    def ordered_pair_sensor_ids(self) -> list[str]:
        return self.selected_sensor_ids if self.pair_index_sensor_ids is None else self.pair_index_sensor_ids

    def ordered_pair_sensor_positions(self) -> dict[str, Any]:
        return self.selected_positions if self.pair_index_positions is None else self.pair_index_positions


class HandoffGeometryResolver(Protocol):
    def resolve_handoff_geometry(
        self,
        *,
        payload: "LocalizedClassifierRenderRequest",
        node: "NodeSpec",
        registry_geometry: HandoffSolveGeometry,
    ) -> HandoffSolveGeometry | None: ...


class HandoffSolutionResolver(Protocol):
    def resolve_handoff_solution(
        self,
        *,
        payload: "LocalizedClassifierRenderRequest",
        geometry: HandoffSolveGeometry,
        reporting_modality: str,
    ) -> "HandoffLocalizationEstimate | None": ...


class LocalizationProvenanceRecorder(Protocol):
    def record_handoff_provenance(
        self,
        *,
        payload: "LocalizedClassifierRenderRequest",
        geometry: HandoffSolveGeometry,
        localized_product: "LocalizedCandidate",
        reporting_modality: str,
    ) -> None: ...


class PositionObservationSink(Protocol):
    async def observe_localized_solve(
        self,
        *,
        localized_product: "LocalizedCandidate",
        reporting_modality: str,
        solve_origin: str,
    ) -> None: ...


@dataclass(slots=True)
class LocalizationSeamCollaborators:
    geometry_resolver: HandoffGeometryResolver
    solution_resolver: HandoffSolutionResolver
    provenance_recorder: LocalizationProvenanceRecorder
    position_observation_sink: PositionObservationSink


def build_localization_seam_collaborators(host: "FusionNode") -> LocalizationSeamCollaborators:
    """Seam factory FusionNode calls once at construction.

    ``host`` is passed so real implementations can reach settings, the tracker
    and the coordinate frame (read per call, so an origin rebind applies).
    """
    # Imported here: these modules import this one for the seam carriers.
    from minimappr.core.localization_provenance import (
        ManifestLocalizationProvenanceRecorder,
        site_frame_id_for_coordinate_frame,
    )
    from minimappr.core.localized_handoff_geometry import ManifestSolveGeometryResolver
    from minimappr.core.localized_handoff_solution import SnapshotGeometrySolutionResolver
    from minimappr.core.position_observation_config import position_observation_config_from_settings
    from minimappr.core.position_observation_pipeline import PositionObservationPipeline

    return LocalizationSeamCollaborators(
        geometry_resolver=ManifestSolveGeometryResolver(
            site_frame_id_provider=lambda: site_frame_id_for_coordinate_frame(host.coordinate_frame)
        ),
        solution_resolver=SnapshotGeometrySolutionResolver(host),
        provenance_recorder=ManifestLocalizationProvenanceRecorder(),
        position_observation_sink=PositionObservationPipeline(
            config=position_observation_config_from_settings(host.settings),
            tracker_provider=lambda: host.tracker,
            metrics_provider=lambda: host._metrics,
        ),
    )
