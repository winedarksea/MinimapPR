"""Room-scale BLE RSSI multilateration helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from minimappr.core.ble_observations import BleObservationRecord


EPSILON = 1e-9


@dataclass(slots=True)
class BleDeviceEstimate:
    mac: str
    position_m: tuple[float, float, float]
    covariance: list[list[float]]
    n_receivers: int
    residual_rms_m: float
    last_seen_ns: int
    contributor_node_ids: list[str]

    def as_api_dict(self) -> dict:
        return {
            "mac": self.mac,
            "position_m": list(self.position_m),
            "covariance": self.covariance,
            "n_receivers": self.n_receivers,
            "residual_rms_m": self.residual_rms_m,
            "last_seen_ns": self.last_seen_ns,
            "contributor_node_ids": self.contributor_node_ids,
        }


def rssi_to_range_m(
    rssi_dbm: float,
    *,
    tx_power_1m_dbm: float = -59.0,
    path_loss_exponent: float = 2.7,
) -> float:
    exponent = max(float(path_loss_exponent), 0.1)
    distance_m = 10.0 ** ((float(tx_power_1m_dbm) - float(rssi_dbm)) / (10.0 * exponent))
    if not math.isfinite(distance_m):
        return 1_000.0
    return float(min(max(distance_m, 0.25), 1_000.0))


def estimate_ble_device_position(
    mac: str,
    observations: list[BleObservationRecord],
    node_positions_m: dict[str, tuple[float, float, float]],
    *,
    now_ns: int,
    max_age_seconds: float = 5.0,
    path_loss_exponent: float = 2.7,
    tx_power_1m_dbm: float = -59.0,
) -> BleDeviceEstimate | None:
    max_age_ns = int(max(max_age_seconds, 0.1) * 1_000_000_000)
    selected: list[tuple[BleObservationRecord, np.ndarray, float]] = []
    seen_nodes: set[str] = set()
    for observation in sorted(observations, key=lambda item: item.recv_ns, reverse=True):
        if observation.node_id in seen_nodes:
            continue
        if now_ns - observation.recv_ns > max_age_ns:
            continue
        position = node_positions_m.get(observation.node_id)
        if position is None:
            continue
        anchor = np.asarray(position, dtype=np.float64)
        if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
            continue
        range_m = rssi_to_range_m(
            observation.rssi_ewma,
            tx_power_1m_dbm=tx_power_1m_dbm,
            path_loss_exponent=path_loss_exponent,
        )
        selected.append((observation, anchor, range_m))
        seen_nodes.add(observation.node_id)

    if len(selected) < 3:
        return None

    anchors_xy = np.vstack([anchor[:2] for _, anchor, _ in selected])
    centered = anchors_xy - np.mean(anchors_xy, axis=0)
    if np.linalg.matrix_rank(centered, tol=1e-6) < 2:
        return None

    ranges = np.asarray([range_m for _, _, range_m in selected], dtype=np.float64)
    weights = 1.0 / np.maximum(ranges, 1.0)
    estimate_xy = np.average(anchors_xy, axis=0, weights=weights)

    normal = np.zeros((2, 2), dtype=np.float64)
    residual_vector = np.zeros(2, dtype=np.float64)
    for _ in range(25):
        normal.fill(0.0)
        residual_vector.fill(0.0)
        for (_, anchor, range_m), weight in zip(selected, weights):
            delta = estimate_xy - anchor[:2]
            predicted = float(np.linalg.norm(delta))
            if predicted < EPSILON:
                continue
            residual = range_m - predicted
            jacobian = delta / predicted
            normal += weight * np.outer(jacobian, jacobian)
            residual_vector += weight * jacobian * residual
        if not np.all(np.isfinite(normal)) or np.linalg.cond(normal) > 1e5:
            return None
        try:
            step = np.linalg.solve(normal, residual_vector)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(step)):
            return None
        estimate_xy += step
        if float(np.linalg.norm(step)) < 1e-3:
            break

    residuals = []
    for _, anchor, range_m in selected:
        residuals.append(float(np.linalg.norm(estimate_xy - anchor[:2]) - range_m))
    residual_rms_m = float(np.sqrt(np.mean(np.square(residuals)))) if residuals else float("inf")
    if not math.isfinite(residual_rms_m):
        return None

    try:
        covariance_xy = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(covariance_xy)):
        return None

    z_m = float(np.mean([anchor[2] for _, anchor, _ in selected]))
    covariance = np.eye(3, dtype=np.float64) * 100.0
    covariance[:2, :2] = covariance_xy
    return BleDeviceEstimate(
        mac=mac,
        position_m=(float(estimate_xy[0]), float(estimate_xy[1]), z_m),
        covariance=covariance.tolist(),
        n_receivers=len(selected),
        residual_rms_m=residual_rms_m,
        last_seen_ns=max(observation.recv_ns for observation, _, _ in selected),
        contributor_node_ids=[observation.node_id for observation, _, _ in selected],
    )
