from __future__ import annotations

import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from minimappr.main import app


def _configure_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MINIMAPPR_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MINIMAPPR_SNIPPET_DIR", str(tmp_path / "snippets"))
    monkeypatch.setenv("MINIMAPPR_LARGE_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MINIMAPPR_INGEST_SIDECAR_ENABLED", "false")
    monkeypatch.setenv("MINIMAPPR_INGEST_BACKEND", "python")


def _rssi_for_range(distance_m: float, *, tx_power_1m_dbm: float = -59.0, n: float = 2.7) -> float:
    return tx_power_1m_dbm - (10.0 * n * math.log10(distance_m))


def _register_node(client: TestClient, node_id: str, position_m: list[float]) -> None:
    response = client.post(
        "/api/v1/nodes",
        json={
            "id": node_id,
            "node_type": "point",
            "position_m": position_m,
            "capabilities": [],
        },
    )
    assert response.status_code == 201


def _ble_payload(node_id: str, rssi: float) -> dict:
    return {
        "node_id": node_id,
        "boot_count": 1,
        "boot_id": 1234,
        "monotonic_ms": 10_000,
        "observations": [
            {
                "mac": "AA-BB-CC-DD-EE-FF",
                "addr_type": 0,
                "rssi_last": round(rssi),
                "rssi_ewma": rssi,
                "rssi_min": round(rssi - 1),
                "rssi_max": round(rssi + 1),
                "count": 5,
                "age_ms": 40,
                "adv_type": 0,
                "name": "beacon",
            }
        ],
    }


def test_ble_ingest_exposes_raw_node_observations_and_device_estimate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_env(monkeypatch, tmp_path)
    node_positions = {
        "ble-a": [0.0, 0.0, 0.0],
        "ble-b": [10.0, 0.0, 0.0],
        "ble-c": [0.0, 10.0, 0.0],
    }
    target = [4.0, 5.0, 0.0]

    with TestClient(app) as client:
        for node_id, position in node_positions.items():
            _register_node(client, node_id, position)
            rssi = _rssi_for_range(math.dist(position, target))
            response = client.post("/api/v1/ingest/ble", json=_ble_payload(node_id, rssi))
            assert response.status_code == 200
            assert response.json()["accepted"] == 1

        node_detail = client.get("/api/v1/nodes/ble-a")
        assert node_detail.status_code == 200
        observations = node_detail.json()["ble_observations"]
        assert observations[0]["mac"] == "aa:bb:cc:dd:ee:ff"
        assert observations[0]["name"] == "beacon"

        devices_response = client.get("/api/v1/ble/devices")
        assert devices_response.status_code == 200
        devices = devices_response.json()["devices"]
        assert len(devices) == 1
        device = devices[0]
        assert device["receiver_count"] == 3
        assert len(device["observations"]) == 3
        assert device["estimate"] is not None
        assert device["estimate"]["position_m"][0] == pytest.approx(target[0], abs=0.5)
        assert device["estimate"]["position_m"][1] == pytest.approx(target[1], abs=0.5)
