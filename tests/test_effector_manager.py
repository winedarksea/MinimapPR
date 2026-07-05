"""EffectorManager: registry CRUD, status broadcast, and the rate-limit
interlock — all against a mock `Effector` driver (no network)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from minimappr.core.effectors import registry as registry_module
from minimappr.core.effectors.base import EffectorCapabilities, EffectorCommand, ExecutionResult
from minimappr.core.effectors.registry import EffectorManager, EffectorManagerConfig
from minimappr.models import EffectorSpec, EffectorType
from minimappr.storage.db import Storage


class _MockDriver:
    def __init__(self) -> None:
        self.executed: list[EffectorCommand] = []
        self.armed = False
        self.snapshots: list[Path] = []
        self.state = "idle"

    async def connect(self) -> None:
        return None

    async def get_capabilities(self) -> EffectorCapabilities:
        return EffectorCapabilities(movement_strategies=["AbsoluteMove"], selected_movement_strategy="AbsoluteMove")

    async def arm(self, *, zone_id: str | None = None) -> bool:
        self.armed = True
        return True

    async def disarm(self) -> bool:
        self.armed = False
        return True

    async def execute(self, command: EffectorCommand) -> ExecutionResult:
        self.executed.append(command)
        return ExecutionResult(status="COMPLETED")

    async def get_status(self) -> dict:
        return {"state": self.state, "armed": self.armed, "pan_deg": 1.0, "tilt_deg": 2.0}

    async def snapshot(self, *, dest_path: Path) -> Path:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"jpeg-bytes")
        self.snapshots.append(dest_path)
        return dest_path


def _spec(effector_id: str) -> EffectorSpec:
    return EffectorSpec(
        id=effector_id,
        effector_type=EffectorType.CAMERA_PTZ,
        position_m=(0.0, 0.0, 0.0),
        transport={"host": "192.168.1.50"},
    )


async def _manager(tmp_path: Path, *, min_slew_interval_seconds: float = 3.0) -> tuple[EffectorManager, Storage, list[dict]]:
    storage = Storage(tmp_path / "effectors.db")
    await storage.initialize()
    events: list[dict] = []

    async def _live_callback(payload: dict) -> None:
        events.append(payload)

    config = EffectorManagerConfig(
        snapshot_dir=tmp_path / "snapshots",
        min_slew_interval_seconds=min_slew_interval_seconds,
        status_poll_interval_seconds=3600.0,
    )
    manager = EffectorManager(storage=storage, live_callback=_live_callback, config=config)
    return manager, storage, events


@pytest.mark.asyncio
async def test_register_persists_and_connects_mock_driver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager, storage, events = await _manager(tmp_path)
    mock_driver = _MockDriver()
    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: mock_driver)

    await manager.register(_spec("cam-1"))

    row = await storage.get_effector_by_id("cam-1")
    assert row is not None
    statuses = await manager.list_status()
    assert len(statuses) == 1
    assert statuses[0].effector_id == "cam-1"
    assert statuses[0].state == "idle"
    assert any(e.get("effector_id") == "cam-1" for e in events)


@pytest.mark.asyncio
async def test_start_loads_effectors_from_db_and_stays_dormant_when_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, storage, _ = await _manager(tmp_path)
    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: _MockDriver())

    await manager.start()
    assert manager.enabled is False
    assert await manager.list_status() == []
    await manager.stop()


@pytest.mark.asyncio
async def test_start_reconnects_persisted_effectors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage = Storage(tmp_path / "effectors.db")
    await storage.initialize()
    await storage.upsert_effector(_spec("cam-1"), time.time_ns())
    await storage.close()

    manager, storage2, _ = await _manager(tmp_path)
    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: _MockDriver())

    await manager.start()
    assert manager.enabled is True
    status = await manager.get_status("cam-1")
    assert status is not None
    assert status.state == "idle"
    await manager.stop()


@pytest.mark.asyncio
async def test_delete_removes_from_registry_and_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager, storage, _ = await _manager(tmp_path)
    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: _MockDriver())
    await manager.register(_spec("cam-1"))

    deleted = await manager.delete("cam-1")
    assert deleted is True
    assert await manager.get_status("cam-1") is None
    assert await storage.get_effector_by_id("cam-1") is None
    assert await manager.delete("cam-1") is False


@pytest.mark.asyncio
async def test_slew_to_target_dispatches_to_driver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager, _, _ = await _manager(tmp_path)
    mock_driver = _MockDriver()
    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: mock_driver)
    await manager.register(_spec("cam-1"))

    result = await manager.slew_to_target("cam-1", (5.0, 5.0, 0.0), track_id="trk-1")

    assert result.status == "COMPLETED"
    assert len(mock_driver.executed) == 1
    assert mock_driver.executed[0].target_position_m == (5.0, 5.0, 0.0)
    assert mock_driver.executed[0].track_id == "trk-1"


@pytest.mark.asyncio
async def test_slew_to_target_unknown_effector_fails(tmp_path: Path) -> None:
    manager, _, _ = await _manager(tmp_path)
    result = await manager.slew_to_target("ghost", (0.0, 0.0, 0.0))
    assert result.status == "FAILED"
    assert result.failure_class == "not_found"


@pytest.mark.asyncio
async def test_rate_limit_interlock_rejects_second_slew_within_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, _, _ = await _manager(tmp_path, min_slew_interval_seconds=60.0)
    mock_driver = _MockDriver()
    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: mock_driver)
    await manager.register(_spec("cam-1"))

    first = await manager.slew_to_target("cam-1", (1.0, 0.0, 0.0))
    assert first.status == "COMPLETED"

    second = await manager.slew_to_target("cam-1", (2.0, 0.0, 0.0))
    assert second.status == "REJECTED"
    assert second.failure_class == "rate_limited"
    # Driver was only actually commanded once — the interlock blocked the second call.
    assert len(mock_driver.executed) == 1


@pytest.mark.asyncio
async def test_rate_limit_interlock_allows_slew_after_interval_elapses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, _, _ = await _manager(tmp_path, min_slew_interval_seconds=0.05)
    mock_driver = _MockDriver()
    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: mock_driver)
    await manager.register(_spec("cam-1"))

    first = await manager.slew_to_target("cam-1", (1.0, 0.0, 0.0))
    assert first.status == "COMPLETED"

    import asyncio
    await asyncio.sleep(0.1)

    second = await manager.slew_to_target("cam-1", (2.0, 0.0, 0.0))
    assert second.status == "COMPLETED"
    assert len(mock_driver.executed) == 2


@pytest.mark.asyncio
async def test_capture_persists_artifact_and_links_track(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager, storage, _ = await _manager(tmp_path)
    mock_driver = _MockDriver()
    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: mock_driver)
    await manager.register(_spec("cam-1"))

    result = await manager.capture("cam-1", track_id="trk-1", detection_id="det-1")

    assert result.status == "COMPLETED"
    assert len(result.result_refs) == 1
    artifact = await storage.get_effector_artifact(result.result_refs[0])
    assert artifact is not None
    assert artifact["track_id"] == "trk-1"
    assert artifact["detection_id"] == "det-1"
    assert len(mock_driver.snapshots) == 1


@pytest.mark.asyncio
async def test_capture_unknown_effector_fails(tmp_path: Path) -> None:
    manager, _, _ = await _manager(tmp_path)
    result = await manager.capture("ghost")
    assert result.status == "FAILED"
    assert result.failure_class == "not_found"


@pytest.mark.asyncio
async def test_capture_reports_failure_when_driver_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager, _, _ = await _manager(tmp_path)

    class _BrokenDriver(_MockDriver):
        async def snapshot(self, *, dest_path: Path) -> Path:
            raise RuntimeError("camera offline")

    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: _BrokenDriver())
    await manager.register(_spec("cam-1"))

    result = await manager.capture("cam-1")
    assert result.status == "FAILED"
    assert result.failure_class == "RuntimeError"


@pytest.mark.asyncio
async def test_register_without_transport_host_leaves_driver_none(tmp_path: Path) -> None:
    manager, _, _ = await _manager(tmp_path)
    spec = EffectorSpec(id="cam-no-host", effector_type=EffectorType.CAMERA_PTZ, position_m=(0.0, 0.0, 0.0))

    await manager.register(spec)

    status = await manager.get_status("cam-no-host")
    assert status is not None
    assert status.state == "offline"
