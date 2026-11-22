"""Pick an onset cluster's classification node that the beamformed classifier can use (Field Mode S-fixes).

Why: the trigger path classifies only a beamformed render, and the classifier
refuses renders from fewer than ``beamformed_classification_min_sensor_count``
sensors (no raw-omni fallback by design). Onset-SRP classified from the
cluster's nearest/loudest node, so a localized bird next to a single-mic phone
(POINT node) was always sent to that phone's one channel, and every such
candidate ended as ``empty_classification`` (S gate note). The nearest node that
has enough sensors is used instead; when no node qualifies the cluster is left
unchanged (the old outcome).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from minimappr.core.onset_srp_types import LocalizedOnsetCluster


def classification_capable_cluster(
    cluster: LocalizedOnsetCluster,
    node_ids_by_sensor: dict[str, str],
    sensor_positions: dict[str, np.ndarray],
    min_sensor_count: int,
) -> LocalizedOnsetCluster:
    sensors_by_node: dict[str, list[str]] = {}
    for sensor_id, node_id in node_ids_by_sensor.items():
        if sensor_id in sensor_positions:
            sensors_by_node.setdefault(node_id, []).append(sensor_id)
    if len(sensors_by_node.get(cluster.loudest_node_id, ())) >= min_sensor_count:
        return cluster
    capable = {node: sensors for node, sensors in sensors_by_node.items() if len(sensors) >= min_sensor_count}
    if not capable:
        return cluster
    position = np.asarray(cluster.position_m, dtype=np.float64)

    def distance_m(node_id: str) -> float:
        centroid = np.mean(np.vstack([sensor_positions[s] for s in capable[node_id]]), axis=0)
        return float(np.linalg.norm(centroid - position))

    return dataclasses.replace(cluster, loudest_node_id=min(capable, key=distance_m))
