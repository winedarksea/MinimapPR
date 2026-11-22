"""Ground / canopy source prior and node-plane geometry for onset-SRP (Field Mode S-fixes).

Why: in a layout whose node centroids are (nearly) coplanar, cross-node TDOAs
are symmetric about the node plane, so a source and its mirror image fit them
equally well. Only the compact arrays' weak elevation evidence separates them,
and the ground-reflection image sits on the mirror side. Onset-SRP's grid
(elevation down to -30 deg) and its unconstrained LM therefore flipped birds
below ground (countersinging: z -2.0 m for a bird at 4.0 m).

The prior: the ground lies ``node_height_above_ground_m`` below the node plane
(fitted through the node centroids, so a hillside layout tilts it), and a
source sits between the ground and ``source_max_height_above_ground_m`` above
it. It decides between mirror twins after LM, masks the range profile, and
bounds LM in non-planar layouts (no twin exists there). ``is_planar`` says whether the mirror ambiguity exists at all (it always
does for <= 3 nodes: any three points are coplanar).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UP = np.array([0.0, 0.0, 1.0])
# Below this singular-value ratio the node centroids are collinear (or a single
# point) and define no plane; the node plane then falls back to horizontal.
COLLINEAR_SINGULAR_VALUE_RATIO = 1.0e-3
# Nodes needed before a fitted plane can show out-of-plane spread.
MIN_NODES_FOR_NON_PLANAR_LAYOUT = 4


@dataclass(frozen=True, slots=True)
class NodePlaneGroundPrior:
    """Node plane (unit normal pointing up) plus the source height slab above ground."""

    plane_point_m: np.ndarray
    plane_normal: np.ndarray
    node_height_above_ground_m: float
    source_max_height_above_ground_m: float
    is_planar: bool
    enabled: bool

    def height_above_ground_m(self, points_m: np.ndarray) -> np.ndarray:
        offsets = np.atleast_2d(np.asarray(points_m, dtype=np.float64)) - self.plane_point_m
        return offsets @ self.plane_normal + self.node_height_above_ground_m

    def admissible_mask(self, points_m: np.ndarray) -> np.ndarray:
        """True where a candidate lies inside the ground..canopy slab (always True if disabled)."""
        points = np.atleast_2d(np.asarray(points_m, dtype=np.float64))
        if not self.enabled:
            return np.ones(points.shape[0], dtype=bool)
        heights = self.height_above_ground_m(points)
        return (heights >= 0.0) & (heights <= self.source_max_height_above_ground_m)

    def mirror_across_node_plane(self, point_m: np.ndarray) -> np.ndarray:
        point = np.asarray(point_m, dtype=np.float64)
        distance = float((point - self.plane_point_m) @ self.plane_normal)
        return point - 2.0 * distance * self.plane_normal

    def fold_above_plane(self, point_m: np.ndarray) -> np.ndarray:
        point = np.asarray(point_m, dtype=np.float64)
        return self.mirror_across_node_plane(point) if float((point - self.plane_point_m) @ self.plane_normal) < 0.0 else point

    def fold_direction_above_plane(self, direction: np.ndarray) -> np.ndarray:
        direction = np.asarray(direction, dtype=np.float64)
        along_normal = float(direction @ self.plane_normal)
        return direction - 2.0 * along_normal * self.plane_normal if along_normal < 0.0 else direction

    def vertical_bounds_m(self, point_m: np.ndarray) -> tuple[float, float] | None:
        """(z_min, z_max) of the slab under ``point_m``'s horizontal position, for LM bounds."""
        if not self.enabled or abs(float(self.plane_normal[2])) <= COLLINEAR_SINGULAR_VALUE_RATIO:
            return None
        point = np.asarray(point_m, dtype=np.float64)
        # z where the node plane passes under (x, y); heights are measured along the normal.
        horizontal = point[:2] - self.plane_point_m[:2]
        plane_z = self.plane_point_m[2] - float(horizontal @ self.plane_normal[:2]) / float(self.plane_normal[2])
        vertical_per_normal_m = 1.0 / float(self.plane_normal[2])
        ground_z = plane_z - self.node_height_above_ground_m * vertical_per_normal_m
        return ground_z, ground_z + self.source_max_height_above_ground_m * vertical_per_normal_m


def build_node_plane_ground_prior(
    node_centroids_m: list[np.ndarray],
    *,
    enabled: bool,
    node_height_above_ground_m: float,
    source_max_height_above_ground_m: float,
    planar_layout_max_flatness_ratio: float,
) -> NodePlaneGroundPrior:
    centroids = np.vstack([np.asarray(c, dtype=np.float64) for c in node_centroids_m])
    plane_point = centroids.mean(axis=0)
    normal = UP.copy()
    is_planar = True
    if centroids.shape[0] >= 3:
        _, singular_values, right_vectors = np.linalg.svd(centroids - plane_point)
        if singular_values[1] > COLLINEAR_SINGULAR_VALUE_RATIO * max(float(singular_values[0]), 1e-12):
            normal = right_vectors[2] if right_vectors[2][2] >= 0.0 else -right_vectors[2]
        flatness = float(singular_values[2] / max(float(singular_values[0]), 1e-12)) if singular_values.size > 2 else 0.0
        is_planar = (
            centroids.shape[0] < MIN_NODES_FOR_NON_PLANAR_LAYOUT or flatness < planar_layout_max_flatness_ratio
        )
    return NodePlaneGroundPrior(
        plane_point_m=plane_point,
        plane_normal=normal / float(np.linalg.norm(normal)),
        node_height_above_ground_m=float(node_height_above_ground_m),
        source_max_height_above_ground_m=float(source_max_height_above_ground_m),
        is_planar=is_planar,
        enabled=bool(enabled),
    )


def node_near_field_points(
    node_centres_m: list[np.ndarray],
    *,
    min_range_m: float,
    max_range_m: float,
    ring_count: int,
    azimuth_step_deg: float,
    elevation_step_deg: float,
) -> np.ndarray:
    """Polar shells around every node.

    The layout-centred grid samples a source 2 m from one node at ~10 deg steps as
    seen from that node, so the node's own (range-blind) compact pairs never line
    up there and the far end of its bearing ray wins the frame. Shells centred on
    each node sample its near field at the node's own angular resolution.
    """
    if ring_count < 1 or not node_centres_m:
        return np.zeros((0, 3))
    azimuths = np.radians(np.arange(0.0, 360.0, azimuth_step_deg))
    elevations = np.radians(np.arange(-90.0, 90.0 + 1e-9, elevation_step_deg))
    azimuth_grid, elevation_grid = np.meshgrid(azimuths, elevations)
    directions = np.column_stack(
        [
            (np.cos(elevation_grid) * np.cos(azimuth_grid)).ravel(),
            (np.cos(elevation_grid) * np.sin(azimuth_grid)).ravel(),
            np.sin(elevation_grid).ravel(),
        ]
    )
    rings = np.geomspace(min_range_m, max_range_m, ring_count) if ring_count > 1 else np.asarray([min_range_m])
    shell = (rings[:, None, None] * directions[None, :, :]).reshape(-1, 3)
    return np.vstack([np.asarray(centre, dtype=np.float64) + shell for centre in node_centres_m])
