"""Collaborator seams for the localization overhaul (plan items P2 and P5).

FusionNode builds its seam collaborators through
``build_localization_seam_collaborators`` and never constructs them itself, so
later work plugs real implementations in here without editing
``fusion_node.py``. Every default below is a no-op: with these defaults the
Rust render handoff and the Python ingest path behave exactly as before.

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
    from minimappr.core.localized_render_localization import HandoffLocalizationEstimate
    from minimappr.models import NodeSpec

# Where a solve came from; lets the observation sink apply origin-specific rules
# (e.g. only event-triggered solves may start a provisional track).
SOLVE_ORIGIN_RUST_CLASSIFIER_RENDER = "rust_classifier_render"
SOLVE_ORIGIN_PYTHON_INGEST = "python_ingest"


@dataclass(frozen=True, slots=True)
class HandoffSolveGeometry:
    """Sensors a render's solve used, in solve order, with their positions."""

    selected_sensor_ids: list[str]
    selected_positions: dict[str, Any]
    reference_sensor: str


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


class RegistryGeometryResolver:
    """Default: the registry geometry stands."""

    def resolve_handoff_geometry(self, *, payload, node, registry_geometry) -> HandoffSolveGeometry | None:
        return None


class DefaultPathSolutionResolver:
    """Default: the handoff's built-in estimate path stands."""

    def resolve_handoff_solution(self, *, payload, geometry, reporting_modality):
        return None


class NoOpLocalizationProvenanceRecorder:
    def record_handoff_provenance(self, *, payload, geometry, localized_product, reporting_modality) -> None:
        return None


class NoOpPositionObservationSink:
    async def observe_localized_solve(self, *, localized_product, reporting_modality, solve_origin) -> None:
        return None


@dataclass(slots=True)
class LocalizationSeamCollaborators:
    geometry_resolver: HandoffGeometryResolver
    solution_resolver: HandoffSolutionResolver
    provenance_recorder: LocalizationProvenanceRecorder
    position_observation_sink: PositionObservationSink


def build_localization_seam_collaborators(host: "FusionNode") -> LocalizationSeamCollaborators:
    """Seam factory FusionNode calls once at construction.

    ``host`` is passed so real implementations can reach settings, the tracker
    and the coordinate frame; the no-op defaults ignore it.
    """
    del host
    return LocalizationSeamCollaborators(
        geometry_resolver=RegistryGeometryResolver(),
        solution_resolver=DefaultPathSolutionResolver(),
        provenance_recorder=NoOpLocalizationProvenanceRecorder(),
        position_observation_sink=NoOpPositionObservationSink(),
    )
