"""Shared rotation rule: world = node + R(body->world) x offset, in the registry and the push."""

from __future__ import annotations

import math

import numpy as np
import pytest

from minimappr.core.cross_node_beamform import world_to_node_local
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.sensor_world_position import (
    body_to_world_quaternion_wxyz,
    rotation_matrix_from_quaternion_wxyz,
    sensor_world_position_m,
)
from minimappr.models import GeoPoint, NodeOrientation, NodeSpec, NodeType
from minimappr.spatial_audio.geometry import orientation_rotation_matrix
from tests.site_geometry_fixtures import SITE_ORIGIN

ORIENTATION_CASES = [
    NodeOrientation(yaw_deg=35.0),
    NodeOrientation(pitch_deg=-20.0),
    NodeOrientation(roll_deg=170.0),
    NodeOrientation(yaw_deg=-120.0, pitch_deg=45.0, roll_deg=-30.0),
    NodeOrientation(yaw_deg=359.0, pitch_deg=89.0, roll_deg=179.0),
]


@pytest.mark.parametrize("orientation", ORIENTATION_CASES)
def test_quaternion_is_exactly_the_orientation_rotation_matrix(orientation: NodeOrientation) -> None:
    quaternion = body_to_world_quaternion_wxyz(orientation)

    assert quaternion is not None
    assert math.isclose(sum(component * component for component in quaternion), 1.0, rel_tol=1e-12)
    assert quaternion[0] >= 0.0
    np.testing.assert_allclose(
        rotation_matrix_from_quaternion_wxyz(quaternion), orientation_rotation_matrix(orientation), atol=1e-12
    )


def test_pure_yaw_quaternion_matches_the_contract_doc_example() -> None:
    psi = math.radians(35.0)
    assert body_to_world_quaternion_wxyz(NodeOrientation(yaw_deg=35.0)) == pytest.approx(
        (math.cos(psi / 2.0), 0.0, 0.0, math.sin(psi / 2.0))
    )


def test_identity_orientation_has_no_quaternion() -> None:
    assert body_to_world_quaternion_wxyz(NodeOrientation()) is None
    assert body_to_world_quaternion_wxyz(None) is None


@pytest.mark.parametrize("orientation", ORIENTATION_CASES)
def test_world_position_inverts_the_cross_node_beamform_convention(orientation: NodeOrientation) -> None:
    node_position_m = np.asarray([37.0, -22.0, 1.5])
    body_offset_m = (0.0177, -0.0177, 0.004)

    world = sensor_world_position_m(node_position_m, orientation, body_offset_m)

    assert world_to_node_local(tuple(world), node_position_m, orientation) == pytest.approx(body_offset_m, abs=1e-12)


@pytest.mark.asyncio
async def test_registry_rotates_sensor_offsets_by_node_orientation() -> None:
    registry = NodeRegistry()
    spec = NodeSpec(
        id="yawed-planar",
        node_type=NodeType.ARRAY,
        position_m=(10.0, 20.0, 1.0),
        sensor_offsets_m=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        orientation=NodeOrientation(yaw_deg=90.0),
        capabilities=["audio"],
    )

    await registry.upsert(spec, last_seen_ns=1)
    positions = await registry.sensor_positions()

    # Rz(+90 deg): body +x -> world +y (north), body +y -> world -x (west).
    np.testing.assert_allclose(positions["yawed-planar:ch0"], [10.0, 21.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(positions["yawed-planar:ch1"], [9.0, 20.0, 1.0], atol=1e-12)


@pytest.mark.asyncio
async def test_registry_applies_an_orientation_override_to_sensor_positions() -> None:
    registry = NodeRegistry()
    await registry.load_overrides({"n": {"orientation": {"yaw_deg": 180.0}}})
    spec = NodeSpec(
        id="n",
        node_type=NodeType.POINT,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=[(1.0, 0.0, 0.0)],
        capabilities=["audio"],
    )

    await registry.upsert(spec, last_seen_ns=1)

    np.testing.assert_allclose((await registry.sensor_positions())["n:ch0"], [-1.0, 0.0, 0.0], atol=1e-12)


@pytest.mark.asyncio
async def test_registry_maps_a_geo_only_override_to_site_position() -> None:
    frame = LocalCoordinateFrame(origin=SITE_ORIGIN, mode="flat")
    registry = NodeRegistry(coordinate_frame=frame)
    pinned = GeoPoint(lat=SITE_ORIGIN.lat + 0.0002, lon=SITE_ORIGIN.lon - 0.0003, alt_m=SITE_ORIGIN.alt_m + 2.0)
    await registry.load_overrides({"pinned": {"position_geo": pinned.model_dump(mode="json")}})
    reported_gps_spec = NodeSpec(
        id="pinned",
        node_type=NodeType.POINT,
        position_m=(500.0, 500.0, 0.0),
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
    )

    runtime = await registry.upsert(reported_gps_spec, last_seen_ns=1)

    assert runtime.spec.position_m == pytest.approx(frame.geo_to_local(pinned))
    assert runtime.spec.position_geo == pinned
    np.testing.assert_allclose(
        (await registry.sensor_positions())["pinned:ch0"], frame.geo_to_local(pinned), atol=1e-9
    )


@pytest.mark.asyncio
async def test_registry_geo_override_follows_a_site_origin_rebind() -> None:
    registry = NodeRegistry(coordinate_frame=LocalCoordinateFrame(origin=SITE_ORIGIN, mode="flat"))
    await registry.load_overrides({"pinned": {"position_geo": SITE_ORIGIN.model_dump(mode="json")}})
    shifted_origin = GeoPoint(lat=SITE_ORIGIN.lat - 0.001, lon=SITE_ORIGIN.lon, alt_m=SITE_ORIGIN.alt_m)
    shifted_frame = LocalCoordinateFrame(origin=shifted_origin, mode="flat")
    spec = NodeSpec(id="pinned", node_type=NodeType.POINT, position_m=(0.0, 0.0, 0.0), capabilities=["audio"])

    registry.set_coordinate_frame(shifted_frame)
    runtime = await registry.upsert(spec, last_seen_ns=1)

    assert runtime.spec.position_m == pytest.approx(shifted_frame.geo_to_local(SITE_ORIGIN))
    assert runtime.spec.position_m[1] > 100.0


@pytest.mark.asyncio
async def test_registry_local_override_wins_over_geo_override() -> None:
    registry = NodeRegistry(coordinate_frame=LocalCoordinateFrame(origin=SITE_ORIGIN, mode="flat"))
    await registry.load_overrides(
        {"both": {"position_m": [1.0, 2.0, 3.0], "position_geo": SITE_ORIGIN.model_dump(mode="json")}}
    )
    spec = NodeSpec(id="both", node_type=NodeType.POINT, position_m=(9.0, 9.0, 9.0), capabilities=["audio"])

    runtime = await registry.upsert(spec, last_seen_ns=1)

    assert runtime.spec.position_m == (1.0, 2.0, 3.0)


@pytest.mark.asyncio
async def test_registry_refuses_a_geo_override_without_a_site_frame() -> None:
    registry = NodeRegistry()
    await registry.load_overrides({"pinned": {"position_geo": SITE_ORIGIN.model_dump(mode="json")}})
    spec = NodeSpec(id="pinned", node_type=NodeType.POINT, position_m=(0.0, 0.0, 0.0), capabilities=["audio"])

    with pytest.raises(ValueError, match="coordinate frame"):
        await registry.upsert(spec, last_seen_ns=1)
