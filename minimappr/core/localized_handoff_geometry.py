"""Solve geometry for a Rust classifier render, taken from the solve itself.

The sidecar publishes the ordered sensor list and the exact positions a solve
used (``solve_provenance.sensors``). Every pair index in the render indexes
that list, so the handoff geometry is rebuilt from it — never from Python's
current registry, and never by mapping cluster indices onto the reporting
node's own channels (which picked the wrong microphones depending on node
arrival order).

Frame acceptance, by ``coordinate_frame.kind``:

- ``site_enu``: accepted only when ``frame_id`` equals Python's
  ``site_frame_id_for`` of its current origin + projection; positions are used
  as given.
- ``sidecar_automatic_enu``: rejected; the sidecar chose its own origin and
  projection, so its positions are not site positions.
- ``explicit_cluster_positions``: accepted as given; they are the
  operator-declared cluster positions the node itself announced in site metres.
- ``node_body``: a node-local solve in the reporting node's body frame; placed
  with the shared rule world = node_position + R(body->world) x body, using the
  registry's (operator-overridden) node spec.

A rejected solve keeps the registry geometry (for the omni report at the node)
and carries the reason.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from minimappr.api.rust_localization_solve_contract import CoordinateFrameKind, RustLocalizationSolve
from minimappr.core.localized_render_handoff_hooks import HandoffSolveGeometry
from minimappr.spatial_audio.geometry import orientation_rotation_matrix

if TYPE_CHECKING:
    from minimappr.api.rust_dsp_manifests import LocalizedClassifierRenderRequest
    from minimappr.models import NodeSpec

SOLVE_REJECTED_COORDINATE_FRAME_MISMATCH = "solve_coordinate_frame_mismatch"
SOLVE_REJECTED_SIDECAR_AUTOMATIC_FRAME = "solve_coordinate_frame_sidecar_automatic"
SOLVE_REJECTED_NODE_BODY_SENSORS_OFF_REPORTING_NODE = "solve_node_body_sensors_off_reporting_node"
SOLVE_REJECTED_NO_CONTRIBUTING_SENSORS = "solve_has_no_contributing_sensors"
# Legacy manifest (no provenance) whose pair indices run past the reporting
# node's channels: a cluster solve whose sensor order Python cannot know.
SOLVE_REJECTED_LEGACY_SENSOR_MAP_UNAVAILABLE = "solve_sensor_map_unavailable"


@dataclass(frozen=True, slots=True)
class SolveFrameToSiteTransform:
    """Rigid body -> site transform: site = rotation @ body + translation."""

    rotation: np.ndarray
    translation_m: np.ndarray

    @classmethod
    def for_node_body(cls, node: "NodeSpec") -> "SolveFrameToSiteTransform":
        if node.position_m is None:
            raise ValueError(f"Node {node.id!r} has no position_m to place its body-frame solve")
        return cls(
            rotation=orientation_rotation_matrix(node.orientation),
            translation_m=np.asarray(node.position_m, dtype=np.float64),
        )

    def position_m(self, body_position_m) -> tuple[float, float, float]:
        site = self.rotation @ np.asarray(body_position_m, dtype=np.float64) + self.translation_m
        return (float(site[0]), float(site[1]), float(site[2]))

    def direction(self, body_direction) -> tuple[float, float, float]:
        site = self.rotation @ np.asarray(body_direction, dtype=np.float64)
        return (float(site[0]), float(site[1]), float(site[2]))

    def covariance_m2(self, body_covariance_m2) -> list[list[float]]:
        site = self.rotation @ np.asarray(body_covariance_m2, dtype=np.float64) @ self.rotation.T
        return [[float(cell) for cell in row] for row in site]


def legacy_pair_indices_exceed_registry_sensors(
    payload: "LocalizedClassifierRenderRequest",
    registry_geometry: HandoffSolveGeometry,
) -> bool:
    """True when a provenance-less render's pairs index past the reporter's channels."""
    if payload.localization_solve is not None or not payload.localization_pair_tdoas:
        return False
    sensor_count = len(registry_geometry.selected_sensor_ids)
    return any(
        not (0 <= int(pair["ch_a"]) < sensor_count and 0 <= int(pair["ch_b"]) < sensor_count)
        for pair in payload.localization_pair_tdoas
    )


class ManifestSolveGeometryResolver:
    """Geometry seam: the solve's own sensor list and snapshot positions.

    ``site_frame_id_provider`` is read per render so a re-anchored origin
    (``FusionNode.rebind_runtime_dependencies``) takes effect immediately.
    """

    def __init__(self, site_frame_id_provider: Callable[[], str]) -> None:
        self._site_frame_id_provider = site_frame_id_provider

    def resolve_handoff_geometry(
        self,
        *,
        payload: "LocalizedClassifierRenderRequest",
        node: "NodeSpec",
        registry_geometry: HandoffSolveGeometry,
    ) -> HandoffSolveGeometry | None:
        solve = payload.localization_solve
        if solve is None:
            return None
        rejected_reason = self._frame_rejection_reason(solve, node)
        if rejected_reason is not None:
            return replace(registry_geometry, rejected_reason=rejected_reason)
        return _geometry_from_solve(solve, node)

    def _frame_rejection_reason(self, solve: RustLocalizationSolve, node: "NodeSpec") -> str | None:
        frame = solve.provenance.coordinate_frame
        if frame.kind is CoordinateFrameKind.SITE_ENU:
            return None if frame.frame_id == self._site_frame_id_provider() else SOLVE_REJECTED_COORDINATE_FRAME_MISMATCH
        if frame.kind is CoordinateFrameKind.SIDECAR_AUTOMATIC_ENU:
            return SOLVE_REJECTED_SIDECAR_AUTOMATIC_FRAME
        if frame.kind is CoordinateFrameKind.NODE_BODY and any(
            sensor.node_id != node.id for sensor in solve.provenance.sensors
        ):
            return SOLVE_REJECTED_NODE_BODY_SENSORS_OFF_REPORTING_NODE
        if not solve.contributing_sensor_indices():
            return SOLVE_REJECTED_NO_CONTRIBUTING_SENSORS
        return None


def _geometry_from_solve(solve: RustLocalizationSolve, node: "NodeSpec") -> HandoffSolveGeometry:
    provenance = solve.provenance
    solve_frame_to_site = (
        SolveFrameToSiteTransform.for_node_body(node)
        if provenance.coordinate_frame.kind is CoordinateFrameKind.NODE_BODY
        else None
    )
    pair_index_sensor_ids = provenance.ordered_sensor_ids
    pair_index_positions = {
        sensor.sensor_id: np.asarray(
            sensor.position_m if solve_frame_to_site is None else solve_frame_to_site.position_m(sensor.position_m),
            dtype=np.float64,
        )
        for sensor in provenance.sensors
    }
    contributing_sensor_ids = [pair_index_sensor_ids[index] for index in solve.contributing_sensor_indices()]
    # The render's audio-quality and speech-capture keys name a reporter sensor
    # when one contributed; otherwise the solve's first contributor.
    reference_sensor = next(
        (
            provenance.sensors[index].sensor_id
            for index in solve.contributing_sensor_indices()
            if provenance.sensors[index].node_id == provenance.reporting_node_id
        ),
        contributing_sensor_ids[0],
    )
    return HandoffSolveGeometry(
        selected_sensor_ids=contributing_sensor_ids,
        selected_positions={sensor_id: pair_index_positions[sensor_id] for sensor_id in contributing_sensor_ids},
        reference_sensor=reference_sensor,
        pair_index_sensor_ids=pair_index_sensor_ids,
        pair_index_positions=pair_index_positions,
        solve_frame_to_site=solve_frame_to_site,
    )
