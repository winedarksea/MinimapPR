"""Node geometry helpers."""

from __future__ import annotations

import math


def regular_tetrahedron_offsets_m(edge_m: float = 0.05) -> list[tuple[float, float, float]]:
    a = edge_m
    v1 = (0.0, 0.0, 0.0)
    v2 = (a, 0.0, 0.0)
    v3 = (0.5 * a, (math.sqrt(3.0) / 2.0) * a, 0.0)
    v4 = (0.5 * a, (math.sqrt(3.0) / 6.0) * a, math.sqrt(2.0 / 3.0) * a)

    centroid = (
        (v1[0] + v2[0] + v3[0] + v4[0]) / 4.0,
        (v1[1] + v2[1] + v3[1] + v4[1]) / 4.0,
        (v1[2] + v2[2] + v3[2] + v4[2]) / 4.0,
    )

    verts = [v1, v2, v3, v4]
    return [
        (vertex[0] - centroid[0], vertex[1] - centroid[1], vertex[2] - centroid[2])
        for vertex in verts
    ]
