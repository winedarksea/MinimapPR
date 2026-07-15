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


def min_pair_spacing_m(positions: NDArray[np.float64]) -> float:
    """Smallest non-degenerate pairwise mic spacing.

    Unlike :func:`max_baseline_m` (which controls the *widest* baseline and thus
    the *lowest* spatial-alias frequency), the closest mic pair sets the *highest*
    frequency at which per-pair processing (GCC-PHAT on that pair) is still free
    of spatial aliasing. For the planar array the 25 mm corner-to-center /
    corner-to-adjacent-corner pairs raise the usable band well above the 50 mm
    diagonal.
    """
    arr = np.asarray(positions, dtype=np.float64)
    n = int(arr.shape[0])
    if n < 2:
        return 0.0
    min_distance_m = math.inf
    for first_index in range(n):
        for second_index in range(first_index + 1, n):
            distance_m = float(np.linalg.norm(arr[first_index] - arr[second_index]))
            if distance_m > 1e-6:
                min_distance_m = min(min_distance_m, distance_m)
    return 0.0 if math.isinf(min_distance_m) else min_distance_m


def alias_cutoff_from_positions(
    positions: NDArray[np.float64],
    c_sound: float = SPEED_OF_SOUND_MPS,
    mode: str = "max_baseline",
) -> float:
    """Spatial-alias cutoff frequency (Hz).

    ``mode="max_baseline"`` (default) uses the widest mic pair — the conservative,
    whole-array cutoff that keeps tetra behaviour unchanged. ``mode="min_pair"``
    uses the closest mic pair, which is the correct bound where processing is done
    per mic-pair (e.g. planar's narrow 25 mm pairs → ~6.9 kHz vs 3.43 kHz for the
    50 mm diagonal). Opt in explicitly; callers that want whole-array behaviour
    should not pass ``mode``.
    """
    arr = np.asarray(positions, dtype=np.float64)
    if mode == "min_pair":
        spacing_m = min_pair_spacing_m(arr)
    else:
        spacing_m = max_baseline_m(arr)
    if spacing_m < 1e-6:
        return ALIAS_CUTOFF_HZ
    return float(c_sound) / (2.0 * spacing_m)


def array_out_of_plane_extent_m(positions: NDArray[np.float64]) -> float:
    """Extent of the array along its thinnest principal axis (metres).

    ~0 for a coplanar array (all mics in one plane, e.g. the 5-mic planar node),
    non-trivial for a genuinely 3D array like the tetrahedron. Computed as the
    smallest singular value spread of the centroid-referenced positions.
    """
    arr = np.asarray(positions, dtype=np.float64)
    if int(arr.shape[0]) < 3:
        return 0.0
    centered = arr - arr.mean(axis=0)
    # Singular values are proportional to the spread along each principal axis.
    singular_values = np.linalg.svd(centered, compute_uv=False)
    return float(singular_values[-1]) if singular_values.size >= 3 else 0.0


def is_coplanar(
    positions: NDArray[np.float64],
    tolerance_m: float = 1e-3,
) -> bool:
    """True if every mic lies within ``tolerance_m`` of a common plane."""
    return array_out_of_plane_extent_m(positions) <= tolerance_m


def reflect_position_into_half_space(
    position_m: NDArray[np.float64],
    plane_z_m: float,
    half_space: str | None,
) -> NDArray[np.float64]:
    """Mirror ``position_m`` across the array's own z-plane if it landed on the
    physically-impossible side of a coplanar array's half-space constraint (D7).

    A coplanar array's TDOA measurements are mirror-symmetric across its own
    plane, so an unconstrained solve is equally satisfied by the true position
    and its reflection; ``half_space`` (``"upper"``/``"lower"``/``None``)
    disambiguates which side is physically valid. Only the z-axis case (a
    horizontal array plane) is implemented — a pitched/rolled planar node
    needs the array's rotated normal, not the raw z-axis (tracked as a
    follow-up, see plan risk #5).
    """
    position = np.asarray(position_m, dtype=np.float64).reshape(3).copy()
    if half_space == "upper" and position[2] < plane_z_m:
        position[2] = (2.0 * plane_z_m) - position[2]
    elif half_space == "lower" and position[2] > plane_z_m:
        position[2] = (2.0 * plane_z_m) - position[2]
    return position


def reflect_covariance_into_half_space(
    covariance_m2: NDArray[np.float64],
    was_reflected: bool,
) -> NDArray[np.float64]:
    """Apply the corresponding z-reflection to a 3x3 position covariance.

    A z-mirror is the linear map J = diag(1, 1, -1); covariance transforms as
    J @ cov @ J^T, which only flips the sign of the xz/yz cross terms (the
    diagonal and xy term are invariant). No-op if the position wasn't reflected.
    """
    cov = np.asarray(covariance_m2, dtype=np.float64).copy()
    if not was_reflected or cov.shape != (3, 3):
        return cov
    cov[0, 2] *= -1.0
    cov[2, 0] *= -1.0
    cov[1, 2] *= -1.0
    cov[2, 1] *= -1.0
    return cov


def foa_geometry_suitable(
    positions: NDArray[np.float64],
    max_baseline_m: float,
    coplanar_tolerance_m: float = 1e-3,
) -> tuple[bool, str]:
    n_mics = int(positions.shape[0])
    if n_mics < 4:
        return False, f"FOA requires >=4 mics; array has {n_mics}"
    # A coplanar array cannot resolve a non-degenerate Z (W/X/Y are fine, but the
    # vertical FOA channel would be synthesised from noise). Planar nodes emit 2D
    # FOA (Z=0) tagged as such; half-space-synthesised Z is deferred.
    if is_coplanar(np.asarray(positions, dtype=np.float64), coplanar_tolerance_m):
        return False, (
            "array is coplanar (out-of-plane extent "
            f"{array_out_of_plane_extent_m(positions):.4f} m <= "
            f"{coplanar_tolerance_m:.4f} m); full 3D FOA Z channel is degenerate"
        )
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
