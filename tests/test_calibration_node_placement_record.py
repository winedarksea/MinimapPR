"""The survey record calibration capture writes per node (plan item E9).

These are the fields ``minimappr/calibration/bundle_validation.py`` checks
against the field capture protocol: position source, accuracy and an explicit
mount quaternion, chosen with the sidecar's placement precedence.
"""

from __future__ import annotations

import numpy as np

from minimappr.calibration.node_placement_record import calibration_node_placement_record
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.sensor_world_position import body_to_world_quaternion_wxyz
from minimappr.models import GeoPoint, NodeOrientation

IDENTITY_ORIENTATION = {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}

SITE_FRAME = LocalCoordinateFrame(origin=GeoPoint(lat=45.0, lon=-93.0, alt_m=250.0), mode="flat")
NODE_POSITION_M = (12.0, -4.0, 1.5)


def _node_row(**fields) -> dict:
    geo = SITE_FRAME.local_to_geo(NODE_POSITION_M)
    row = {
        "id": "node-a",
        "position_geo": {"lat": geo.lat, "lon": geo.lon, "alt_m": geo.alt_m},
        "position_m": list(NODE_POSITION_M),
        "orientation": {},
        "overrides": {},
        "metadata": {},
        "mobility": "stationary",
        "effective_position_filter": "kde",
    }
    row.update(fields)
    return row


def test_placement_record_operator_override_keeps_its_stated_accuracy() -> None:
    record = calibration_node_placement_record(
        _node_row(overrides={"position_m": [1.0, 2.0, 3.0], "position_std_m": 0.3}), None, SITE_FRAME
    )
    assert (record.position_source, record.position_std_m) == ("operator_override", 0.3)
    assert record.position_m == [1.0, 2.0, 3.0]
    assert np.allclose(SITE_FRAME.geo_to_local(GeoPoint(**record.position_geo)), [1.0, 2.0, 3.0], atol=1e-3)


def test_placement_record_sources_without_a_pushed_placement() -> None:
    gps = calibration_node_placement_record(_node_row(metadata={"gps": {"position_source": "gps", "signal": "fix_3d"}}), None, SITE_FRAME)
    assert (gps.position_source, gps.position_std_m) == ("reported_gps", None)
    assert np.allclose(gps.position_m, NODE_POSITION_M, atol=1e-3)
    fallback = calibration_node_placement_record(_node_row(), None, SITE_FRAME)
    assert fallback.position_source == "fallback_static"
    unknown = calibration_node_placement_record(None, None, SITE_FRAME)
    assert unknown.position_source is None and unknown.position_m is None


def test_placement_record_spells_out_identity_orientation() -> None:
    assert calibration_node_placement_record(_node_row(), None, SITE_FRAME).body_to_world_quaternion_wxyz is None
    identity = calibration_node_placement_record(_node_row(orientation=dict(IDENTITY_ORIENTATION)), None, SITE_FRAME)
    assert identity.body_to_world_quaternion_wxyz == [1.0, 0.0, 0.0, 0.0]
    yawed_orientation = {"yaw_deg": 30.0, "pitch_deg": 0.0, "roll_deg": 0.0}
    yawed = calibration_node_placement_record(_node_row(orientation=yawed_orientation), None, SITE_FRAME)
    assert yawed.body_to_world_quaternion_wxyz == list(body_to_world_quaternion_wxyz(NodeOrientation(**yawed_orientation)))
