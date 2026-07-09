"""BLE-device-as-track background estimation tests."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from minimappr.config import Settings
from minimappr.core.ble_observations import BleObservationStore
from minimappr.core.ble_tracking import BleTracker
from minimappr.core.tracking import TrackManager
from minimappr.models import BleObservationIn
from minimappr.storage.db import Storage


def _rssi_for_range(distance_m: float, *, tx_power_1m_dbm: float = -59.0, n: float = 2.7) -> float:
    return tx_power_1m_dbm - (10.0 * n * math.log10(max(distance_m, 0.25)))


def _observation(mac: str, rssi: float, *, name: str | None = "beacon") -> BleObservationIn:
    return BleObservationIn(
        mac=mac,
        addr_type=0,
        rssi_last=int(round(rssi)),
        rssi_ewma=rssi,
        rssi_min=int(round(rssi - 1)),
        rssi_max=int(round(rssi + 1)),
        count=5,
        name=name,
    )


class _RecordingHub:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def broadcast(self, message: dict) -> None:
        self.messages.append(message)


async def _ingest_square(
    store: BleObservationStore,
    node_positions: dict[str, tuple[float, float, float]],
    target: tuple[float, float, float],
    *,
    mac: str,
    recv_ns: int,
) -> None:
    for node_id, position in node_positions.items():
        rssi = _rssi_for_range(math.dist(position, target))
        await store.ingest(
            node_id=node_id,
            observations=[_observation(mac, rssi)],
            recv_ns=recv_ns,
        )


@pytest.mark.asyncio
async def test_ble_tick_creates_persisted_ble_track(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "ble_tracks.db")
    await storage.initialize()
    store = BleObservationStore()
    node_positions = {
        "ble-a": (0.0, 0.0, 0.0),
        "ble-b": (10.0, 0.0, 0.0),
        "ble-c": (0.0, 10.0, 0.0),
        "ble-d": (10.0, 10.0, 0.0),
    }
    target = (4.0, 5.0, 0.0)
    now_ns = 1_000_000_000_000
    await _ingest_square(store, node_positions, target, mac="AA:BB:CC:DD:EE:FF", recv_ns=now_ns)

    tracker = BleTracker(Settings().ble_tracking_config())
    updated = await tracker.run_tick(
        storage=storage,
        observation_store=store,
        node_positions=node_positions,
        now_ns=now_ns,
    )

    assert len(updated) == 1
    track = updated[0]
    assert track.track_kind == "ble"
    assert track.id.startswith("ble-")
    assert track.label == "beacon"
    assert track.capability_tier == "2d"
    assert track.position_m[0] == pytest.approx(target[0], abs=1.0)
    assert track.position_m[1] == pytest.approx(target[1], abs=1.0)

    # Persisted with track_kind read back correctly.
    stored = await storage.list_tracks(limit=10)
    assert len(stored) == 1
    assert stored[0]["track_kind"] == "ble"
    assert stored[0]["id"] == track.id
    await storage.close()


@pytest.mark.asyncio
async def test_ble_tick_broadcasts_updated_track(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "ble_tracks_ws.db")
    await storage.initialize()
    store = BleObservationStore()
    hub = _RecordingHub()
    node_positions = {
        "ble-a": (0.0, 0.0, 0.0),
        "ble-b": (10.0, 0.0, 0.0),
        "ble-c": (0.0, 10.0, 0.0),
    }
    now_ns = 2_000_000_000_000
    await _ingest_square(store, node_positions, (3.0, 4.0, 0.0), mac="11:22:33:44:55:66", recv_ns=now_ns)

    tracker = BleTracker(Settings().ble_tracking_config())
    await tracker.run_tick(
        storage=storage,
        observation_store=store,
        node_positions=node_positions,
        now_ns=now_ns,
        live_hub=hub,
    )

    track_messages = [msg for msg in hub.messages if msg["type"] == "track_updated"]
    assert len(track_messages) == 1
    assert track_messages[0]["track"]["track_kind"] == "ble"
    await storage.close()


@pytest.mark.asyncio
async def test_fewer_than_three_receivers_creates_no_track(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "ble_tracks_few.db")
    await storage.initialize()
    store = BleObservationStore()
    node_positions = {
        "ble-a": (0.0, 0.0, 0.0),
        "ble-b": (10.0, 0.0, 0.0),
    }
    now_ns = 3_000_000_000_000
    await _ingest_square(store, node_positions, (4.0, 5.0, 0.0), mac="AA:BB:CC:DD:EE:FF", recv_ns=now_ns)

    tracker = BleTracker(Settings().ble_tracking_config())
    updated = await tracker.run_tick(
        storage=storage,
        observation_store=store,
        node_positions=node_positions,
        now_ns=now_ns,
    )

    assert updated == []
    assert await storage.list_tracks(limit=10) == []
    await storage.close()


@pytest.mark.asyncio
async def test_track_kind_column_defaults_acoustic_for_legacy_rows(tmp_path: Path) -> None:
    """A row written without track_kind (pre-migration behavior) reads back acoustic."""
    storage = Storage(tmp_path / "ble_tracks_migrate.db")
    await storage.initialize()
    db = storage._require_db()  # type: ignore[attr-defined]
    await db.execute(
        """
        INSERT INTO tracks (
            id, first_seen_ns, last_seen_ns, x, y, z,
            vx, vy, vz, label, confidence, update_count, status
        ) VALUES ('legacy-1', 1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'unknown', 0.5, 1, 'tentative')
        """
    )
    await db.commit()

    rows = await storage.list_tracks(limit=10)
    assert len(rows) == 1
    assert rows[0]["id"] == "legacy-1"
    assert rows[0]["track_kind"] == "acoustic"
    await storage.close()


@pytest.mark.asyncio
async def test_acoustic_and_ble_tracks_never_associate(tmp_path: Path) -> None:
    """A co-located acoustic track and BLE tick yield two distinct tracks."""
    storage = Storage(tmp_path / "ble_tracks_coexist.db")
    await storage.initialize()
    settings = Settings()
    node_positions = {
        "ble-a": (0.0, 0.0, 0.0),
        "ble-b": (10.0, 0.0, 0.0),
        "ble-c": (0.0, 10.0, 0.0),
    }
    target = (4.0, 5.0, 0.0)
    now_ns = 4_000_000_000_000

    # Acoustic track co-located with the BLE device.
    acoustic = TrackManager(settings.tracking_config())
    acoustic_track = await acoustic.update(
        timestamp_ns=now_ns,
        position_m=target,
        label="bird",
        confidence=0.9,
    )
    await storage.upsert_track(acoustic_track)

    store = BleObservationStore()
    await _ingest_square(store, node_positions, target, mac="AA:BB:CC:DD:EE:FF", recv_ns=now_ns)
    tracker = BleTracker(settings.ble_tracking_config())
    updated = await tracker.run_tick(
        storage=storage,
        observation_store=store,
        node_positions=node_positions,
        now_ns=now_ns,
    )

    assert len(updated) == 1
    ble_track = updated[0]
    assert ble_track.id != acoustic_track.id
    assert acoustic_track.track_kind == "acoustic"
    assert ble_track.track_kind == "ble"

    stored = await storage.list_tracks(limit=10)
    kinds = {row["id"]: row["track_kind"] for row in stored}
    assert kinds == {acoustic_track.id: "acoustic", ble_track.id: "ble"}
    await storage.close()
