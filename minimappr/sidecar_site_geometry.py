"""Build the site-geometry snapshot Python pushes to the Rust sidecar.

Wire shape: ``SiteGeometryPush`` in ``docs/localization_solve_contract.md``
section 10. The snapshot is the solve geometry Rust uses ahead of raw packet GPS:

- operator position overrides (``operator_override``), with the entered
  accuracy as ``position_std_m``;
- the filtered KDE estimate of every stationary node without an override
  (``stationary_estimate``), so per-packet GPS jitter never becomes solve
  geometry;
- an orientation-only entry (no ``position_m``) for a node whose operator
  pinned only its mount orientation: Rust keeps its raw GPS position and
  applies this rotation, which the packet itself never carries.

Inputs are storage rows (node rows and estimator checkpoints), not in-memory
estimator state, so the same builder serves the combined process and the
split-mode API process that owns the sidecar but not the estimator.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization_provenance import (
    contract_projection_for_coordinate_frame_mode,
    site_frame_id_for_coordinate_frame,
)
from minimappr.core.node_position_estimator import checkpointed_stationary_estimate
from minimappr.core.node_registry import operator_override_position_m
from minimappr.core.sensor_world_position import body_to_world_quaternion_wxyz
from minimappr.models import GeoPoint, NodeOrientation, NodeOverrides

logger = logging.getLogger(__name__)

PLACEMENT_SOURCE_OPERATOR_OVERRIDE = "operator_override"
PLACEMENT_SOURCE_STATIONARY_ESTIMATE = "stationary_estimate"


# Accuracy assumed for an operator-entered position with no stated accuracy: a
# tape- or laser-measured placement is good to about a decimetre. Operators
# with a survey-grade or phone-GPS placement enter position_std_m explicitly.
DEFAULT_OPERATOR_PLACEMENT_POSITION_STD_M = 0.1


def contract_projection_for(coordinate_frame: LocalCoordinateFrame) -> str:
    """Contract projection name of the frame Python actually uses."""
    return str(contract_projection_for_coordinate_frame_mode(coordinate_frame.mode))


@dataclass(frozen=True, slots=True)
class SiteNodePlacement:
    # None (with position_std_m None) = orientation-only placement.
    position_m: tuple[float, float, float] | None
    position_std_m: float | None
    placement_source: str
    body_to_world_quaternion_wxyz: tuple[float, float, float, float] | None

    def to_wire(self) -> dict[str, object]:
        wire: dict[str, object] = {"placement_source": self.placement_source}
        if self.position_m is not None:
            wire["position_m"] = list(self.position_m)
            wire["position_std_m"] = self.position_std_m
        # Key omitted means identity in the contract.
        if self.body_to_world_quaternion_wxyz is not None:
            wire["body_to_world_quaternion_wxyz"] = list(self.body_to_world_quaternion_wxyz)
        return wire


@dataclass(frozen=True, slots=True)
class SiteGeometrySnapshot:
    """One full geometry snapshot, without its revision (assigned on adoption)."""

    site_frame_id: str
    site_origin: GeoPoint
    projection: str
    node_placements: Mapping[str, SiteNodePlacement]

    def to_push_body(self, geometry_push_revision: int) -> dict[str, object]:
        return {
            "site_frame_id": self.site_frame_id,
            "geometry_push_revision": geometry_push_revision,
            "site_origin": {
                "lat": self.site_origin.lat,
                "lon": self.site_origin.lon,
                "alt_m": self.site_origin.alt_m,
            },
            "projection": self.projection,
            "node_placements": {
                node_id: self.node_placements[node_id].to_wire() for node_id in sorted(self.node_placements)
            },
        }

    def differs_beyond_estimate_drift(
        self,
        previous: "SiteGeometrySnapshot | None",
        *,
        estimate_move_threshold_m: float,
    ) -> bool:
        """Whether this snapshot should replace ``previous`` in the sidecar.

        Any change of frame, node set, placement source, operator placement or
        orientation counts. A stationary estimate counts only once it has moved
        more than the threshold: the KDE re-evaluates continually, and pushing
        every centimetre of re-estimation would churn the sidecar's layout
        revision without improving any solve.
        """
        if previous is None:
            return True
        if (
            self.site_frame_id != previous.site_frame_id
            or self.projection != previous.projection
            or self.site_origin != previous.site_origin
            or set(self.node_placements) != set(previous.node_placements)
        ):
            return True
        for node_id, placement in self.node_placements.items():
            previous_placement = previous.node_placements[node_id]
            if (
                placement.placement_source != previous_placement.placement_source
                or placement.body_to_world_quaternion_wxyz != previous_placement.body_to_world_quaternion_wxyz
            ):
                return True
            if placement.placement_source == PLACEMENT_SOURCE_OPERATOR_OVERRIDE or placement.position_m is None:
                if placement != previous_placement:
                    return True
                continue
            moved_m = float(
                np.linalg.norm(np.subtract(placement.position_m, previous_placement.position_m))
            )
            if moved_m > estimate_move_threshold_m:
                return True
        return False


def _orientation_quaternion(node_row: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    # The row's orientation is already effective (override, else reported); an
    # empty dict means no orientation is known, which the contract spells as identity.
    raw_orientation = node_row.get("orientation")
    if not isinstance(raw_orientation, dict) or not raw_orientation:
        return None
    return body_to_world_quaternion_wxyz(NodeOrientation.model_validate(raw_orientation))


def _operator_placement(
    node_row: Mapping[str, Any],
    coordinate_frame: LocalCoordinateFrame,
) -> SiteNodePlacement | None:
    raw_overrides = node_row.get("overrides")
    if not isinstance(raw_overrides, dict) or not raw_overrides:
        return None
    overrides = NodeOverrides.model_validate(raw_overrides)
    position_m = operator_override_position_m(overrides, coordinate_frame)
    if position_m is None:
        return None
    position_std_m = (
        overrides.position_std_m
        if overrides.position_std_m is not None
        else DEFAULT_OPERATOR_PLACEMENT_POSITION_STD_M
    )
    return SiteNodePlacement(
        position_m=position_m,
        position_std_m=float(position_std_m),
        placement_source=PLACEMENT_SOURCE_OPERATOR_OVERRIDE,
        body_to_world_quaternion_wxyz=_orientation_quaternion(node_row),
    )


def _stationary_estimate_placement(
    node_row: Mapping[str, Any],
    estimator_checkpoint: object,
    coordinate_frame: LocalCoordinateFrame,
) -> SiteNodePlacement | None:
    # A node switched to mobile or Kalman keeps its stale KDE checkpoint in
    # storage; that geometry no longer describes where the node is.
    if node_row.get("mobility") != "stationary" or node_row.get("effective_position_filter") != "kde":
        return None
    estimate = checkpointed_stationary_estimate(estimator_checkpoint)
    if estimate is None:
        return None
    position_m = coordinate_frame.geo_to_local(estimate.estimate_geo)
    if not all(math.isfinite(axis) for axis in position_m):
        return None
    return SiteNodePlacement(
        position_m=position_m,
        position_std_m=estimate.horizontal_std_m,
        placement_source=PLACEMENT_SOURCE_STATIONARY_ESTIMATE,
        body_to_world_quaternion_wxyz=_orientation_quaternion(node_row),
    )


def _orientation_only_placement(node_row: Mapping[str, Any]) -> SiteNodePlacement | None:
    """An operator orientation override on a node with no pushed position."""
    raw_overrides = node_row.get("overrides")
    if not isinstance(raw_overrides, dict) or not raw_overrides:
        return None
    overrides = NodeOverrides.model_validate(raw_overrides)
    if overrides.orientation is None:
        return None
    return SiteNodePlacement(
        position_m=None,
        position_std_m=None,
        placement_source=PLACEMENT_SOURCE_OPERATOR_OVERRIDE,
        body_to_world_quaternion_wxyz=body_to_world_quaternion_wxyz(overrides.orientation),
    )


def _placement_is_finite(placement: SiteNodePlacement) -> bool:
    # Rust rejects a push holding any non-finite number, and at launch that
    # refusal stops the sidecar, so one bad node must not reach the body.
    numbers = [
        *(placement.position_m or ()),
        *(() if placement.position_std_m is None else (placement.position_std_m,)),
        *(placement.body_to_world_quaternion_wxyz or ()),
    ]
    return all(math.isfinite(value) for value in numbers)


def build_site_geometry_snapshot(
    *,
    node_rows: list[Mapping[str, Any]],
    estimator_states_by_node_id: Mapping[str, Mapping[str, Any]],
    coordinate_frame: LocalCoordinateFrame,
) -> SiteGeometrySnapshot:
    """Snapshot from storage rows (``Storage.list_nodes`` and
    ``Storage.list_node_position_estimator_states``) in the current site frame.

    Nodes with neither an override nor a settled stationary estimate are left
    out: Rust then uses their raw GPS, or keeps them node-local when untrusted.
    An orientation override alone gives an orientation-only entry.
    """
    placements: dict[str, SiteNodePlacement] = {}
    for node_row in node_rows:
        node_id = str(node_row["id"])
        try:
            placement = _operator_placement(node_row, coordinate_frame)
            if placement is None:
                estimator_row = estimator_states_by_node_id.get(node_id) or {}
                placement = _stationary_estimate_placement(node_row, estimator_row.get("state"), coordinate_frame)
            if placement is None:
                placement = _orientation_only_placement(node_row)
        except ValueError as exc:
            # One malformed stored row must not withhold every other node's
            # geometry; the node simply stays on raw GPS until it is fixed.
            logger.warning("Site geometry: skipping node %s with unusable placement data: %s", node_id, exc)
            continue
        if placement is None:
            continue
        if not _placement_is_finite(placement):
            logger.warning("Site geometry: skipping node %s with a non-finite placement: %s", node_id, placement)
            continue
        placements[node_id] = placement
    return SiteGeometrySnapshot(
        site_frame_id=site_frame_id_for_coordinate_frame(coordinate_frame),
        site_origin=coordinate_frame.origin,
        projection=contract_projection_for(coordinate_frame),
        node_placements=MappingProxyType(placements),
    )
