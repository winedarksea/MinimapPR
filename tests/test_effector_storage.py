"""Storage round-trip tests for `effectors` + `effector_artifacts` tables."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from minimappr.models import EffectorOrientation, EffectorSpec, EffectorType, GeoPoint
from minimappr.storage.db import Storage


def _spec(effector_id: str, **overrides) -> EffectorSpec:
    kwargs = dict(
        id=effector_id,
        effector_type=EffectorType.CAMERA_PTZ,
        position_m=(1.0, 2.0, 3.0),
        orientation=EffectorOrientation(yaw_deg=45.0, pitch_deg=-5.0),
        capabilities=["ptz", "snapshot"],
        transport={"host": "192.168.1.50", "port": 80, "username": "admin", "password": "hunter2"},
        metadata={"model": "Reolink RLC-823A"},
        properties={"note": "backyard"},
    )
    kwargs.update(overrides)
    return EffectorSpec(**kwargs)


@pytest.mark.asyncio
async def test_upsert_and_get_effector_round_trip(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "effectors.db")
    await storage.initialize()
    now_ns = time.time_ns()

    await storage.upsert_effector(_spec("cam-1"), now_ns)
    row = await storage.get_effector_by_id("cam-1")

    assert row is not None
    assert row["id"] == "cam-1"
    assert row["effector_type"] == "camera_ptz"
    assert row["position_m"] == [1.0, 2.0, 3.0]
    assert row["orientation"] == {"yaw_deg": 45.0, "pitch_deg": -5.0}
    assert row["capabilities"] == ["ptz", "snapshot"]
    assert row["transport"] == {
        "host": "192.168.1.50", "port": 80, "username": "admin", "password": "hunter2"
    }
    assert row["metadata"] == {"model": "Reolink RLC-823A"}
    assert row["properties"] == {"note": "backyard"}
    assert row["last_seen_ns"] == now_ns


@pytest.mark.asyncio
async def test_upsert_effector_persists_geo_position(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "effectors.db")
    await storage.initialize()
    spec = _spec("cam-geo", position_m=(0.0, 0.0, 0.0), position_geo=GeoPoint(lat=1.5, lon=2.5, alt_m=10.0))

    await storage.upsert_effector(spec, time.time_ns())
    row = await storage.get_effector_by_id("cam-geo")

    assert row["position_geo"] == {"lat": 1.5, "lon": 2.5, "alt_m": 10.0}


@pytest.mark.asyncio
async def test_upsert_effector_requires_position_m(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "effectors.db")
    await storage.initialize()
    spec = EffectorSpec(
        id="cam-nogeo",
        effector_type=EffectorType.CAMERA_PTZ,
        position_geo=GeoPoint(lat=0.0, lon=0.0, alt_m=0.0),
    )
    with pytest.raises(ValueError):
        await storage.upsert_effector(spec, time.time_ns())


@pytest.mark.asyncio
async def test_upsert_effector_overwrites_existing_row(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "effectors.db")
    await storage.initialize()
    await storage.upsert_effector(_spec("cam-1"), time.time_ns())
    await storage.upsert_effector(_spec("cam-1", position_m=(9.0, 9.0, 9.0)), time.time_ns() + 1)

    rows = await storage.list_effectors()
    assert len(rows) == 1
    assert rows[0]["position_m"] == [9.0, 9.0, 9.0]


@pytest.mark.asyncio
async def test_list_effectors_orders_by_id(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "effectors.db")
    await storage.initialize()
    await storage.upsert_effector(_spec("cam-b"), time.time_ns())
    await storage.upsert_effector(_spec("cam-a"), time.time_ns())

    rows = await storage.list_effectors()
    assert [row["id"] for row in rows] == ["cam-a", "cam-b"]


@pytest.mark.asyncio
async def test_delete_effector(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "effectors.db")
    await storage.initialize()
    await storage.upsert_effector(_spec("cam-1"), time.time_ns())

    assert await storage.delete_effector("cam-1") is True
    assert await storage.get_effector_by_id("cam-1") is None
    assert await storage.delete_effector("cam-1") is False


@pytest.mark.asyncio
async def test_effector_artifact_round_trip(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "effectors.db")
    await storage.initialize()
    await storage.upsert_effector(_spec("cam-1"), time.time_ns())
    now_ns = time.time_ns()

    artifact_id = await storage.insert_effector_artifact(
        effector_id="cam-1",
        track_id="trk-1",
        detection_id=None,
        kind="snapshot",
        path=str(tmp_path / "cam-1-snap.jpg"),
        created_ns=now_ns,
    )

    artifact = await storage.get_effector_artifact(artifact_id)
    assert artifact is not None
    assert artifact["effector_id"] == "cam-1"
    assert artifact["track_id"] == "trk-1"
    assert artifact["kind"] == "snapshot"

    by_effector = await storage.list_effector_artifacts(effector_id="cam-1")
    assert len(by_effector) == 1
    by_track = await storage.list_effector_artifacts(track_id="trk-1")
    assert len(by_track) == 1
    by_unknown_track = await storage.list_effector_artifacts(track_id="does-not-exist")
    assert by_unknown_track == []


@pytest.mark.asyncio
async def test_effector_artifacts_cascade_delete_with_effector(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "effectors.db")
    await storage.initialize()
    await storage.upsert_effector(_spec("cam-1"), time.time_ns())
    artifact_id = await storage.insert_effector_artifact(
        effector_id="cam-1",
        track_id=None,
        detection_id=None,
        path=str(tmp_path / "snap.jpg"),
        created_ns=time.time_ns(),
    )

    await storage.delete_effector("cam-1")

    assert await storage.get_effector_artifact(artifact_id) is None
