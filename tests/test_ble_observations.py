from __future__ import annotations

import pytest

from minimappr.core.ble_observations import BleObservationStore, normalize_ble_mac
from minimappr.models import BleObservationIn


def _observation(mac: str = "AA:BB:CC:DD:EE:FF", *, rssi: float = -61.0) -> BleObservationIn:
    return BleObservationIn(
        mac=mac,
        addr_type=0,
        rssi_last=int(round(rssi)),
        rssi_ewma=rssi,
        rssi_min=int(round(rssi - 1)),
        rssi_max=int(round(rssi + 1)),
        count=3,
    )


def test_normalize_ble_mac_accepts_common_formats() -> None:
    assert normalize_ble_mac("AA-BB-CC-DD-EE-FF") == "aa:bb:cc:dd:ee:ff"
    assert normalize_ble_mac("aabbccddeeff") == "aa:bb:cc:dd:ee:ff"


@pytest.mark.asyncio
async def test_ble_observation_store_replaces_per_node_latest_and_prunes_ttl() -> None:
    store = BleObservationStore(ttl_seconds=1.0, max_entries=8)
    await store.ingest(
        node_id="node-a",
        observations=[_observation(rssi=-65.0)],
        recv_ns=1_000_000_000,
    )
    await store.ingest(
        node_id="node-a",
        observations=[_observation(rssi=-62.0)],
        recv_ns=1_100_000_000,
    )
    grouped = await store.latest_by_device(now_ns=1_200_000_000)
    records = grouped["aa:bb:cc:dd:ee:ff"]
    assert len(records) == 1
    assert records[0].rssi_ewma == -62.0

    assert await store.size(now_ns=2_200_000_001) == 0


@pytest.mark.asyncio
async def test_ble_observation_store_evicts_oldest_entries() -> None:
    store = BleObservationStore(ttl_seconds=30.0, max_entries=2)
    for index in range(3):
        await store.ingest(
            node_id=f"node-{index}",
            observations=[_observation(f"00:00:00:00:00:0{index}")],
            recv_ns=1_000_000_000 + index,
        )

    grouped = await store.latest_by_device(now_ns=1_000_000_010)
    assert sorted(grouped) == ["00:00:00:00:00:01", "00:00:00:00:00:02"]
