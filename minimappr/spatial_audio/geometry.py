"""Shared geometry helpers for Sirith tetrahedral ambisonics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


SIRITH_MIC_POSITIONS_M: NDArray[np.float64] = np.array(
    [
        [0.0, 0.050, 0.0],
        [0.0433, 0.025, 0.0],
        [0.0, 0.0, 0.0],
        [0.02165, 0.025, 0.04082],
    ],
    dtype=np.float64,
)

SPEED_OF_SOUND_MPS: float = 343.2
ALIAS_CUTOFF_HZ: float = SPEED_OF_SOUND_MPS / (2.0 * 0.05)


@dataclass(frozen=True, slots=True)
class NodeOrientation:
    """Yaw/pitch/roll rotation from node-local mic offsets into site coordinates."""

    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0


def centroid_corrected_positions(
    positions: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    centroid = np.asarray(positions, dtype=np.float64).mean(axis=0)
    return np.asarray(positions, dtype=np.float64) - centroid, centroid


def max_baseline_m(positions: NDArray[np.float64]) -> float:
    n = int(positions.shape[0])
    if n < 2:
        return 0.0
    max_distance_m = 0.0
    for first_index in range(n):
        for second_index in range(first_index + 1, n):
            distance_m = float(
                np.linalg.norm(positions[first_index] - positions[second_index])
            )
            max_distance_m = max(max_distance_m, distance_m)
    return max_distance_m


def alias_cutoff_from_positions(
    positions: NDArray[np.float64],
    c_sound: float = SPEED_OF_SOUND_MPS,
) -> float:
    baseline_m = max_baseline_m(np.asarray(positions, dtype=np.float64))
    if baseline_m < 1e-6:
        return ALIAS_CUTOFF_HZ
    return float(c_sound) / (2.0 * baseline_m)


def foa_geometry_suitable(
    positions: NDArray[np.float64],
    max_baseline_m: float,
) -> tuple[bool, str]:
    n_mics = int(positions.shape[0])
    if n_mics < 4:
        return False, f"FOA requires >=4 mics; array has {n_mics}"
    baseline_m = max_baseline_m_from_array(np.asarray(positions, dtype=np.float64))
    if baseline_m > max_baseline_m:
        return False, (
            f"array baseline {baseline_m:.3f} m exceeds FOA limit "
            f"{max_baseline_m:.3f} m"
        )
    return True, "ok"


def max_baseline_m_from_array(positions: NDArray[np.float64]) -> float:
    return max_baseline_m(positions)


def orientation_rotation_matrix(
    orientation: NodeOrientation | object | None,
) -> NDArray[np.float64]:
    """Return a ZYX yaw/pitch/roll rotation matrix.

    Yaw is clockwise from local north in the existing MinimapPR convention. The
    local coordinate frame is east/north/up, so yaw is a rotation about +Z.
    """
    yaw_deg = float(getattr(orientation, "yaw_deg", 0.0) if orientation else 0.0)
    pitch_deg = float(getattr(orientation, "pitch_deg", 0.0) if orientation else 0.0)
    roll_deg = float(getattr(orientation, "roll_deg", 0.0) if orientation else 0.0)

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return (rz @ ry @ rx).astype(np.float64)


def rotate_positions(
    positions: NDArray[np.float64],
    orientation: NodeOrientation | object | None,
) -> NDArray[np.float64]:
    rotation = orientation_rotation_matrix(orientation)
    return np.asarray(positions, dtype=np.float64) @ rotation.T
