"""Parity: Rust site geometry (localization overhaul R2) vs the Python reference.

The Rust side runs through the sidecar's hidden ``site-geometry-oracle``; the
expected values are computed live from:

- ``minimappr.core.geo.LocalCoordinateFrame`` (``flat`` and ``geodetic`` modes)
  for ``minimappr_core::site_frame_projection`` and for the positions the
  sidecar's ``SiteGeometryState`` hands to solves (pushed frame and automatic
  first-GPS-fix frame);
- ``minimappr.spatial_audio.geometry.orientation_rotation_matrix`` for
  ``minimappr_core::sensor_world_position`` (world = node + R @ body offset,
  R carried on the wire as a body -> world quaternion ``[w, x, y, z]``).

Tolerances are the contract budget, not the achieved agreement (nanometres:
both sides evaluate the same f64 expressions): 1 mm within 5 km of the origin,
1 cm out to 50 km; rotations 1e-12.
"""

from __future__ import annotations

import json
import math
import subprocess

import numpy as np
import pytest

from minimappr.core.geo import LocalCoordinateFrame
from minimappr.models import GeoPoint, NodeOrientation
from minimappr.spatial_audio.geometry import orientation_rotation_matrix
from tests.rust_sidecar_binary import build_rust_sidecar_binary, cargo_is_available

pytestmark = pytest.mark.skipif(not cargo_is_available(), reason="cargo is required to build the Rust sidecar oracle")

NEAR_RANGE_M = 5_000.0
NEAR_TOLERANCE_M = 1.0e-3
FAR_RANGE_M = 50_000.0
FAR_TOLERANCE_M = 1.0e-2
ROTATION_TOLERANCE = 1.0e-12
PROJECTION_MODE_BY_WIRE_NAME = {"flat_wgs84_series": "flat", "geodetic_ecef_enu": "geodetic"}
SITE_ORIGINS = [
    GeoPoint(lat=45.0, lon=-93.0, alt_m=250.0),
    GeoPoint(lat=-33.86, lon=151.21, alt_m=40.0),
    GeoPoint(lat=64.13, lon=-21.9, alt_m=15.0),
    GeoPoint(lat=0.5, lon=10.0, alt_m=-20.0),
]
RANDOM_SEED = 20261124


def _run_oracle(cases: list[dict]) -> list[dict]:
    completed = subprocess.run(
        [str(build_rust_sidecar_binary()), "site-geometry-oracle"],
        input=json.dumps({"cases": cases}),
        capture_output=True,
        text=True,
        check=True,
    )
    results = json.loads(completed.stdout)["results"]
    assert len(results) == len(cases)
    return results


def _origin_json(origin: GeoPoint) -> dict:
    return {"lat": origin.lat, "lon": origin.lon, "alt_m": origin.alt_m}


def _local_points_m(rng: np.random.Generator, max_horizontal_m: float, count: int) -> np.ndarray:
    horizontal = rng.uniform(-max_horizontal_m, max_horizontal_m, size=(count, 2))
    vertical = rng.uniform(-100.0, 300.0, size=(count, 1))
    return np.hstack([horizontal, vertical])


def _geodetic_points(origin: GeoPoint, local_points_m: np.ndarray) -> list[list[float]]:
    # Generated through the flat inverse only to spread points around the
    # origin; both projections are then compared on the same geodetic input.
    frame = LocalCoordinateFrame(origin=origin, mode="flat")
    points = [frame.local_to_geo(tuple(point)) for point in local_points_m]
    return [[point.lat, point.lon, point.alt_m] for point in points]


@pytest.mark.parametrize("wire_projection", sorted(PROJECTION_MODE_BY_WIRE_NAME))
def test_projection_matches_local_coordinate_frame(wire_projection: str) -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    cases, expected, tolerances = [], [], []
    for origin in SITE_ORIGINS:
        frame = LocalCoordinateFrame(origin=origin, mode=PROJECTION_MODE_BY_WIRE_NAME[wire_projection])
        for max_range_m, tolerance_m in ((NEAR_RANGE_M, NEAR_TOLERANCE_M), (FAR_RANGE_M, FAR_TOLERANCE_M)):
            points = _geodetic_points(origin, _local_points_m(rng, max_range_m, count=64))
            cases.append({"kind": "projection", "origin": _origin_json(origin), "projection": wire_projection, "points": points})
            expected.append([frame.geo_to_local(GeoPoint(lat=lat, lon=lon, alt_m=alt)) for lat, lon, alt in points])
            tolerances.append(tolerance_m)

    for case, result, python_enu, tolerance_m in zip(cases, _run_oracle(cases), expected, tolerances):
        error_m = np.abs(np.asarray(result["enu_m"]) - np.asarray(python_enu)).max()
        assert error_m <= tolerance_m, f"{wire_projection} at {case['origin']}: max error {error_m:.3e} m"


def _python_matrix_from_quaternion(quaternion_wxyz: list[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion_wxyz) / np.linalg.norm(quaternion_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_sensor_world_position_follows_orientation_rotation_matrix() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    orientations = [None, NodeOrientation(yaw_deg=90.0), NodeOrientation(yaw_deg=-135.0, pitch_deg=10.0, roll_deg=-5.0)]
    orientations += [
        NodeOrientation(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll)
        for yaw, pitch, roll in zip(rng.uniform(-360, 360, 16), rng.uniform(-90, 90, 16), rng.uniform(-180, 180, 16))
    ]
    cases = []
    for orientation in orientations:
        case = {
            "kind": "sensor_world_position",
            "node_position_m": rng.uniform(-500.0, 500.0, 3).tolist(),
            "body_offsets_m": rng.uniform(-0.1, 0.1, (5, 3)).tolist(),
        }
        if orientation is not None:
            case["orientation"] = orientation.model_dump()
        cases.append(case)

    for orientation, case, result in zip(orientations, cases, _run_oracle(cases)):
        rotation = orientation_rotation_matrix(orientation)
        expected = np.asarray(case["node_position_m"]) + np.asarray(case["body_offsets_m"]) @ rotation.T
        np.testing.assert_allclose(result["world_positions_m"], expected, rtol=0.0, atol=ROTATION_TOLERANCE * 1e3)
        if orientation is None:
            assert result["body_to_world_quaternion_wxyz"] is None
            continue
        np.testing.assert_allclose(
            _python_matrix_from_quaternion(result["body_to_world_quaternion_wxyz"]), rotation, rtol=0.0, atol=ROTATION_TOLERANCE
        )

    pure_yaw = cases[1]
    half_yaw = math.radians(90.0) / 2.0
    np.testing.assert_allclose(
        _run_oracle([pure_yaw])[0]["body_to_world_quaternion_wxyz"],
        [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
        rtol=0.0,
        atol=ROTATION_TOLERANCE,
    )


def _fix(point: GeoPoint, position_source: str = "reported_gps") -> dict:
    return {"lat": point.lat, "lon": point.lon, "alt_m": point.alt_m, "position_source": position_source}


def test_resolved_placements_match_python_frames() -> None:
    origin = SITE_ORIGINS[0]
    flat_frame = LocalCoordinateFrame(origin=origin, mode="flat")
    gps_b = flat_frame.local_to_geo((1_800.0, -950.0, 6.0))
    gps_a = flat_frame.local_to_geo((-40.0, 25.0, 1.0))
    static_c = flat_frame.local_to_geo((6.0, 0.0, 2.0))
    push = {
        "site_frame_id": "parity-site",
        "geometry_push_revision": 4,
        "site_origin": _origin_json(origin),
        "projection": "geodetic_ecef_enu",
        "node_placements": {
            "a": {"position_m": [1.0, 2.0, 1.5], "position_std_m": 0.05, "placement_source": "operator_override"},
        },
    }
    nodes = [
        {"node_id": "a", "fix": _fix(gps_a)},
        {"node_id": "b", "fix": _fix(gps_b)},
        {"node_id": "c", "fix": _fix(static_c, "fallback_static")},
        {"node_id": "d"},
    ]
    pushed, automatic = _run_oracle(
        [
            {"kind": "resolve_placements", "push": push, "nodes": nodes},
            # No push: b is the first trusted fix, so it anchors the flat automatic frame.
            {"kind": "resolve_placements", "nodes": nodes[1:]},
        ]
    )

    geodetic_frame = LocalCoordinateFrame(origin=origin, mode="geodetic")
    assert pushed["coordinate_frame"] == {
        "frame_id": "parity-site",
        "kind": "site_enu",
        "origin": _origin_json(origin),
        "projection": "geodetic_ecef_enu",
    }
    placed_a, gps_placed_b, untrusted_c, untrusted_d = pushed["resolutions"]
    assert placed_a == {"position_m": [1.0, 2.0, 1.5], "position_source": "operator_override"}
    assert gps_placed_b["position_source"] == "reported_gps"
    np.testing.assert_allclose(gps_placed_b["position_m"], geodetic_frame.geo_to_local(gps_b), rtol=0.0, atol=NEAR_TOLERANCE_M)
    assert untrusted_c == untrusted_d == {"untrusted_reason": "node_position_untrusted_no_override"}

    assert automatic["coordinate_frame"]["kind"] == "sidecar_automatic_enu"
    assert automatic["coordinate_frame"]["frame_id"] is None
    assert automatic["coordinate_frame"]["projection"] == "flat_wgs84_series"
    automatic_frame = LocalCoordinateFrame(origin=gps_b, mode="flat")
    np.testing.assert_allclose(automatic["resolutions"][0]["position_m"], [0.0, 0.0, 0.0], rtol=0.0, atol=NEAR_TOLERANCE_M)
    assert automatic["resolutions"][1] == {"untrusted_reason": "node_position_untrusted_no_override"}
    np.testing.assert_allclose(
        automatic_frame.geo_to_local(gps_b), automatic["resolutions"][0]["position_m"], rtol=0.0, atol=NEAR_TOLERANCE_M
    )
