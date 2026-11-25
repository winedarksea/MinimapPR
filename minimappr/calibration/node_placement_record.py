"""The survey record a calibration bundle keeps for each node.

A bundle's node position must be the geometry the live solve used, with its
provenance and accuracy, or a field capture cannot be scored honestly (the
field capture protocol, ``docs/field_capture_protocol.md``). Precedence is the
sidecar's (``docs/localization_solve_contract.md`` section 3.5): a pushed
placement (operator override, else settled stationary estimate) from the same
builder that feeds the sidecar, then the node's reported GPS fix, else the
firmware fallback position.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from minimappr.api.rust_localization_solve_contract import SensorPositionSource
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.sensor_world_position import body_to_world_quaternion_wxyz
from minimappr.core.site_origin import node_has_trusted_gps_position
from minimappr.models import GeoPoint, NodeOrientation
from minimappr.sidecar_site_geometry import build_site_geometry_snapshot

_IDENTITY_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class CalibrationNodePlacementRecord:
    position_m: list[float] | None
    position_geo: dict[str, float] | None
    position_source: str | None
    position_std_m: float | None
    # Written whenever the node row states an orientation (identity spelled
    # out), so "identity" and "never recorded" stay distinguishable.
    body_to_world_quaternion_wxyz: list[float] | None

    def manifest_fields(self) -> dict[str, Any]:
        return {
            "position_geo": self.position_geo,
            "position_m": self.position_m,
            "position_source": self.position_source,
            "position_std_m": self.position_std_m,
            "body_to_world_quaternion_wxyz": self.body_to_world_quaternion_wxyz,
        }


def _geo_payload(point: GeoPoint) -> dict[str, float]:
    return {"lat": float(point.lat), "lon": float(point.lon), "alt_m": float(point.alt_m)}


def _row_quaternion(node_row: Mapping[str, Any]) -> list[float] | None:
    raw_orientation = node_row.get("orientation")
    if not isinstance(raw_orientation, dict) or not raw_orientation:
        return None
    quaternion = body_to_world_quaternion_wxyz(NodeOrientation.model_validate(raw_orientation))
    return list(quaternion or _IDENTITY_QUATERNION_WXYZ)


def _reported_position(
    node_row: Mapping[str, Any], coordinate_frame: LocalCoordinateFrame
) -> tuple[list[float] | None, dict[str, float] | None]:
    geo = node_row.get("position_geo")
    if isinstance(geo, dict) and geo.get("lat") is not None and geo.get("lon") is not None:
        point = GeoPoint(lat=float(geo["lat"]), lon=float(geo["lon"]), alt_m=float(geo.get("alt_m") or 0.0))
        return [float(axis) for axis in coordinate_frame.geo_to_local(point)], _geo_payload(point)
    raw_local = node_row.get("position_m")
    if isinstance(raw_local, (list, tuple)) and len(raw_local) >= 3:
        position_m = [float(raw_local[0]), float(raw_local[1]), float(raw_local[2])]
        return position_m, _geo_payload(coordinate_frame.local_to_geo(tuple(position_m)))
    return None, None


def calibration_node_placement_record(
    node_row: Mapping[str, Any] | None,
    estimator_state_row: Mapping[str, Any] | None,
    coordinate_frame: LocalCoordinateFrame,
) -> CalibrationNodePlacementRecord:
    """``estimator_state_row`` is the node's ``list_node_position_estimator_states`` entry."""
    if node_row is None:
        return CalibrationNodePlacementRecord(None, None, None, None, None)
    node_id = str(node_row["id"])
    snapshot = build_site_geometry_snapshot(
        node_rows=[node_row],
        estimator_states_by_node_id={node_id: estimator_state_row} if estimator_state_row else {},
        coordinate_frame=coordinate_frame,
    )
    placement = snapshot.node_placements.get(node_id)
    quaternion = _row_quaternion(node_row)
    if placement is not None and placement.position_m is not None:
        position_m = [float(axis) for axis in placement.position_m]
        return CalibrationNodePlacementRecord(
            position_m=position_m,
            position_geo=_geo_payload(coordinate_frame.local_to_geo(tuple(position_m))),
            position_source=placement.placement_source,
            position_std_m=placement.position_std_m,
            body_to_world_quaternion_wxyz=quaternion,
        )
    position_m, position_geo = _reported_position(node_row, coordinate_frame)
    position_source = (
        SensorPositionSource.REPORTED_GPS.value
        if position_geo is not None and node_has_trusted_gps_position(node_row.get("metadata"))
        else SensorPositionSource.FALLBACK_STATIC.value
    )
    return CalibrationNodePlacementRecord(
        position_m=position_m,
        position_geo=position_geo,
        position_source=position_source if position_m is not None else None,
        position_std_m=None,
        body_to_world_quaternion_wxyz=quaternion,
    )
