from __future__ import annotations

import math

import pytest

from minimappr.core.ble_multilateration import estimate_ble_device_position
from minimappr.core.ble_observations import BleObservationRecord


def _rssi_for_range(distance_m: float, *, tx_power_1m_dbm: float = -59.0, n: float = 2.0) -> float:
    return tx_power_1m_dbm - (10.0 * n * math.log10(distance_m))


def _record(node_id: str, position: tuple[float, float, float], target: tuple[float, float, float]) -> BleObservationRecord:
    distance_m = math.dist(position, target)
    rssi = _rssi_for_range(distance_m)
    return BleObservationRecord(
        node_id=node_id,
        mac="aa:bb:cc:dd:ee:ff",
        addr_type=0,
        rssi_last=round(rssi),
        rssi_ewma=rssi,
        rssi_min=round(rssi - 1),
        rssi_max=round(rssi + 1),
        count=4,
        recv_ns=10_000_000_000,
        age_ms=0,
        adv_type=0,
        name=None,
    )


def test_ble_multilateration_recovers_position_in_four_node_square() -> None:
    node_positions = {
        "n1": (0.0, 0.0, 0.0),
        "n2": (10.0, 0.0, 0.0),
        "n3": (0.0, 10.0, 0.0),
        "n4": (10.0, 10.0, 0.0),
    }
    target = (4.0, 6.0, 0.0)
    observations = [
        _record(node_id, position, target)
        for node_id, position in node_positions.items()
    ]

    estimate = estimate_ble_device_position(
        "aa:bb:cc:dd:ee:ff",
        observations,
        node_positions,
        now_ns=10_100_000_000,
        path_loss_exponent=2.0,
    )

    assert estimate is not None
    assert estimate.n_receivers == 4
    assert estimate.position_m[0] == pytest.approx(4.0, abs=0.1)
    assert estimate.position_m[1] == pytest.approx(6.0, abs=0.1)
    assert estimate.residual_rms_m < 0.1


def test_ble_multilateration_rejects_collinear_nodes() -> None:
    node_positions = {
        "n1": (0.0, 0.0, 0.0),
        "n2": (5.0, 0.0, 0.0),
        "n3": (10.0, 0.0, 0.0),
    }
    target = (4.0, 6.0, 0.0)
    observations = [
        _record(node_id, position, target)
        for node_id, position in node_positions.items()
    ]

    estimate = estimate_ble_device_position(
        "aa:bb:cc:dd:ee:ff",
        observations,
        node_positions,
        now_ns=10_100_000_000,
        path_loss_exponent=2.0,
    )

    assert estimate is None
