"""Pure pan/tilt slew math: camera pose + target Vec3 -> (pan_deg, tilt_deg).

Works entirely in local Cartesian meters (east, north, up) — the same frame
``LocalCoordinateFrame`` produces for nodes and tracks — so no geo round-trip
is required. Driver- and network-free; fully unit-testable with known
geometries.
"""

from __future__ import annotations

import math

from minimappr.models import NodeOrientation, Vec3


def compute_pan_tilt(
    camera_pos: Vec3,
    camera_orientation: NodeOrientation,
    target_pos: Vec3,
) -> tuple[float, float]:
    """Return (pan_deg, tilt_deg) to aim from camera_pos at target_pos.

    pan_deg is clockwise-from-north bearing to the target, corrected by the
    camera's registered home yaw (i.e. 0 = dead-center of the camera's home
    view, positive = clockwise/right). tilt_deg is elevation angle above the
    camera's horizontal, corrected by the camera's home pitch (0 = level with
    the home view, positive = up).
    """
    dx = target_pos[0] - camera_pos[0]
    dy = target_pos[1] - camera_pos[1]
    dz = target_pos[2] - camera_pos[2]

    bearing_deg = math.degrees(math.atan2(dx, dy)) % 360.0
    pan_deg = ((bearing_deg - camera_orientation.yaw_deg + 180.0) % 360.0) - 180.0

    horizontal_range_m = math.hypot(dx, dy)
    elevation_deg = math.degrees(math.atan2(dz, horizontal_range_m)) if horizontal_range_m > 1e-9 else (
        90.0 if dz > 0 else (-90.0 if dz < 0 else 0.0)
    )
    tilt_deg = elevation_deg - camera_orientation.pitch_deg
    tilt_deg = max(-90.0, min(90.0, tilt_deg))

    return pan_deg, tilt_deg
