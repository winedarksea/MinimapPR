"""Collinear-node mirror guard for onset-SRP fixes (closeout B3).

Port of `node_set_is_collinear` in `minimappr-core/src/mesh/host_entity_tracking.rs`.

Nodes that lie close to one line cannot tell which side of that line a source
is on from arrival times alone: every TDOA fix has an equally good mirror twin
across the line, and the solver's local covariance cannot show it. A rigid
multi-mic array resolves the side from its own geometry, so the guard applies
only when every contributing node is a single-mic (POINT) node. Such a fix is
then reported with the existing "range unobservable" representation
(`range_asymptotic`), which renders as a bearing ray and never creates a track
or pinpoints an entity, instead of a point that may be the mirror image.

`localization_collinear_mirror_max_aspect_ratio` is a ratio of horizontal
standard deviations (minor / major); 0 disables the guard. It is a heuristic: a
thin triangle is near-degenerate rather than truly mirror-ambiguous, which is
still reason enough not to pinpoint.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

# A node with at least this many mic positions is a rigid array that resolves
# left/right of the node line by itself.
MIN_RIGID_ARRAY_SENSOR_COUNT = 2
HORIZONTAL_AXES = 2


def node_set_is_collinear(positions_m: Iterable[Iterable[float]], max_aspect_ratio: float) -> bool:
    """Do these node positions lie close to one line (horizontally)?

    True when the minor/major horizontal spread ratio (of standard deviations)
    is below `max_aspect_ratio`; fewer than two nodes are trivially collinear.
    """
    points = np.asarray([list(p)[:HORIZONTAL_AXES] for p in positions_m], dtype=np.float64)
    if len(points) < 2:
        return True
    deviations = points - points.mean(axis=0)
    xx = float(np.sum(deviations[:, 0] ** 2))
    yy = float(np.sum(deviations[:, 1] ** 2))
    xy = float(np.sum(deviations[:, 0] * deviations[:, 1]))
    half_trace = 0.5 * (xx + yy)
    spread = float(np.sqrt(0.25 * (xx - yy) ** 2 + xy * xy))
    major, minor = half_trace + spread, max(half_trace - spread, 0.0)
    return major <= 0.0 or float(np.sqrt(minor / major)) < max_aspect_ratio


def fix_is_collinear_mirror_ambiguous(
    contributing_node_ids: Iterable[str],
    node_ids_by_sensor: Mapping[str, str],
    sensor_positions: Mapping[str, np.ndarray],
    max_aspect_ratio: float,
) -> bool:
    """Only single-mic nodes contributed, and they lie close to one line."""
    if max_aspect_ratio <= 0.0:
        return False
    sensors_by_node: dict[str, list[str]] = {}
    for sensor_id, node_id in node_ids_by_sensor.items():
        if sensor_id in sensor_positions:
            sensors_by_node.setdefault(node_id, []).append(sensor_id)
    contributing = [node for node in dict.fromkeys(contributing_node_ids) if node in sensors_by_node]
    if not contributing or any(len(sensors_by_node[node]) >= MIN_RIGID_ARRAY_SENSOR_COUNT for node in contributing):
        return False
    node_positions = [np.mean([sensor_positions[s] for s in sensors_by_node[node]], axis=0) for node in contributing]
    return node_set_is_collinear(node_positions, max_aspect_ratio)
