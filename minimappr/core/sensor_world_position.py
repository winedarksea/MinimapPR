"""The shared sensor-placement rule: world = node position + R(body->world) x offset.

R is ``orientation_rotation_matrix`` (Rz(yaw) Ry(pitch) Rx(roll), world = R x
body), the convention ``core/cross_node_beamform.py`` already inverts when it
steers into a node's mic frame. The Rust sidecar implements the same rule in
``minimappr-core/src/sensor_world_position.rs`` from the quaternion pushed in
the site geometry, so both sides place every microphone identically.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from minimappr.models import NodeOrientation, Vec3
from minimappr.spatial_audio.geometry import orientation_rotation_matrix


def node_orientation_is_identity(orientation: NodeOrientation | None) -> bool:
    if orientation is None:
        return True
    return orientation.yaw_deg == 0.0 and orientation.pitch_deg == 0.0 and orientation.roll_deg == 0.0


def sensor_world_position_m(
    node_position_m: Vec3 | NDArray[np.float64],
    orientation: NodeOrientation | None,
    body_offset_m: Vec3 | NDArray[np.float64],
) -> NDArray[np.float64]:
    rotation = orientation_rotation_matrix(orientation)
    return np.asarray(node_position_m, dtype=np.float64) + rotation @ np.asarray(body_offset_m, dtype=np.float64)


def body_to_world_quaternion_wxyz(orientation: NodeOrientation | None) -> tuple[float, float, float, float] | None:
    """Unit quaternion (w, x, y, z) of ``orientation_rotation_matrix``; None for identity.

    Composed as q = qz(yaw) * qy(pitch) * qx(roll) (Hamilton product), the
    quaternion form of Rz Ry Rx. Canonicalised to w >= 0 so equal rotations
    serialise identically and a re-push of unchanged geometry is byte-stable.
    """
    if node_orientation_is_identity(orientation):
        return None
    half_yaw = math.radians(orientation.yaw_deg) / 2.0
    half_pitch = math.radians(orientation.pitch_deg) / 2.0
    half_roll = math.radians(orientation.roll_deg) / 2.0
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = cy * sp * cr + sy * cp * sr
    z = sy * cp * cr - cy * sp * sr
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    sign = -1.0 if w < 0.0 else 1.0
    return (sign * w / norm, sign * x / norm, sign * y / norm, sign * z / norm)


def rotation_matrix_from_quaternion_wxyz(quaternion_wxyz: tuple[float, float, float, float]) -> NDArray[np.float64]:
    """Inverse of ``body_to_world_quaternion_wxyz``, for parity checks against Rust."""
    w, x, y, z = quaternion_wxyz
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
