"""EffectorManager: PTZ-node registry CRUD, status broadcast, and the rate-limit
interlock — all against a mock PTZ driver (no network)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from minimappr.core.effectors import registry as registry_module
from minimappr.core.effectors.base import EffectorCapabilities, EffectorCommand, ExecutionResult
from minimappr.core.effectors.registry import EffectorManager, EffectorManagerConfig
from minimappr.models import NodeCapability, NodeSafetyConfig, NodeSpec, NodeType
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


def _node_spec(node_id: str, *, host: str | None = "192.168.1.50") -> NodeSpec:
    transport = {"host": host} if host is not None else {}
    return NodeSpec(
        id=node_id,
        node_type=NodeType.POINT,
        position_m=(0.0, 0.0, 0.0),
        capabilities=[NodeCapability.PTZ_CAMERA],
        transport=transport,
    )


async def _register_ptz_node(
    manager: EffectorManager,
    storage: Storage,
    node_id: str,
    *,
    host: str | None = "192.168.1.50",
) -> None:
    await storage.insert_node_registration(_node_spec(node_id, host=host), last_seen_ns=time.time_ns())
    row = await storage.get_node_by_id(node_id)
    assert row is not None
    await manager.register_node(row)


async def _manager(
    tmp_path: Path,
    *,
    min_slew_interval_seconds: float = 3.0,
    status_poll_interval_seconds: float = 3600.0,
    slew_dwell_seconds: float = 10.0,
) -> tuple[EffectorManager, Storage, list[dict]]:
    storage = Storage(tmp_path / "effectors.db")
    await storage.initialize()
    events: list[dict] = []

    async def _live_callback(payload: dict) -> None:
        events.append(payload)

    config = EffectorManagerConfig(
        snapshot_dir=tmp_path / "snapshots",
        min_slew_interval_seconds=min_slew_interval_seconds,
        status_poll_interval_seconds=status_poll_interval_seconds,
        slew_dwell_seconds=slew_dwell_seconds,
    )
    manager = EffectorManager(storage=storage, live_callback=_live_callback, config=config)
    return manager, storage, events


@pytest.mark.asyncio
async def test_register_persists_and_connects_mock_driver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager, storage, events = await _manager(tmp_path)
    mock_driver = _MockDriver()
    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: mock_driver)

    await _register_ptz_node(manager, manager._storage, "cam-1")

    row = await storage.get_node_by_id("cam-1")
    assert row is not None
    statuses = await manager.list_status()
    assert len(statuses) == 1
    assert statuses[0].node_id == "cam-1"
    assert statuses[0].state == "idle"
    assert any(e.get("node_id") == "cam-1" for e in events)


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
    await storage.insert_node_registration(_node_spec("cam-1"), last_seen_ns=time.time_ns())
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
async def test_detach_removes_from_runtime_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager, storage, _ = await _manager(tmp_path)
    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: _MockDriver())
    await _register_ptz_node(manager, manager._storage, "cam-1")

    deleted = await manager.detach("cam-1")
    assert deleted is True
    assert await manager.get_status("cam-1") is None
    assert await storage.get_node_by_id("cam-1") is not None
    assert await manager.detach("cam-1") is False


@pytest.mark.asyncio
async def test_slew_to_target_dispatches_to_driver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager, _, _ = await _manager(tmp_path)
    mock_driver = _MockDriver()
    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: mock_driver)
    await _register_ptz_node(manager, manager._storage, "cam-1")

    result = await manager.slew_to_target("cam-1", (5.0, 5.0, 0.0), track_id="trk-1")

    assert result.status == "COMPLETED"
    assert len(mock_driver.executed) == 1
    assert mock_driver.executed[0].target_position_m == (5.0, 5.0, 0.0)
    assert mock_driver.executed[0].track_id == "trk-1"


@pytest.mark.asyncio
async def test_slew_to_target_unknown_ptz_node_fails(tmp_path: Path) -> None:
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
    await _register_ptz_node(manager, manager._storage, "cam-1")

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
    await _register_ptz_node(manager, manager._storage, "cam-1")

    first = await manager.slew_to_target("cam-1", (1.0, 0.0, 0.0))
    assert first.status == "COMPLETED"

    import asyncio
    await asyncio.sleep(0.1)

    second = await manager.slew_to_target("cam-1", (2.0, 0.0, 0.0))
    assert second.status == "COMPLETED"
    assert len(mock_driver.executed) == 2


@pytest.mark.asyncio
async def test_require_arm_safety_interlock_blocks_until_armed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, storage, _ = await _manager(tmp_path, min_slew_interval_seconds=0.0)
    mock_driver = _MockDriver()
    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: mock_driver)
    await _register_ptz_node(manager, manager._storage, "cam-1")

    safety = NodeSafetyConfig(require_arm_for_action=True, min_action_interval_seconds=0.0)
    updated = await manager.update_safety("cam-1", safety)
    assert updated == safety
    row = await storage.get_node_by_id("cam-1")
    assert row is not None
    assert row["safety"]["require_arm_for_action"] is True

    blocked = await manager.slew_to_target("cam-1", (1.0, 0.0, 0.0))
    assert blocked.status == "REJECTED"
    assert blocked.failure_class == "interlock:disarmed"
    assert mock_driver.executed == []

    armed = await manager.arm("cam-1")
    assert armed.status == "COMPLETED"
    allowed = await manager.slew_to_target("cam-1", (1.0, 0.0, 0.0))
    assert allowed.status == "COMPLETED"
    assert len(mock_driver.executed) == 1


@pytest.mark.asyncio
async def test_no_go_zone_safety_interlock_uses_target_zone_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, _, _ = await _manager(tmp_path, min_slew_interval_seconds=0.0)
    mock_driver = _MockDriver()
    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: mock_driver)
    await _register_ptz_node(manager, manager._storage, "cam-1")
    await manager.update_safety(
        "cam-1",
        NodeSafetyConfig(no_go_zone_ids=["no-go"], min_action_interval_seconds=0.0),
    )

    async def _target_zone_resolver(target_pos) -> set[str]:
        return {"no-go"} if target_pos[0] > 0 else set()

    manager.set_target_zone_resolver(_target_zone_resolver)

    blocked = await manager.slew_to_target("cam-1", (1.0, 0.0, 0.0))
    assert blocked.status == "REJECTED"
    assert blocked.failure_class == "interlock:no_go_zone"
    assert mock_driver.executed == []

    allowed = await manager.slew_to_target("cam-1", (-1.0, 0.0, 0.0))
    assert allowed.status == "COMPLETED"
    assert len(mock_driver.executed) == 1


@pytest.mark.asyncio
async def test_capture_persists_artifact_and_links_track(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager, storage, _ = await _manager(tmp_path)
    mock_driver = _MockDriver()
    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: mock_driver)
    await _register_ptz_node(manager, manager._storage, "cam-1")

    result = await manager.capture("cam-1", track_id="trk-1", detection_id="det-1")

    assert result.status == "COMPLETED"
    assert len(result.result_refs) == 1
    artifact = await storage.get_node_artifact(result.result_refs[0])
    assert artifact is not None
    assert artifact["track_id"] == "trk-1"
    assert artifact["detection_id"] == "det-1"
    assert len(mock_driver.snapshots) == 1


@pytest.mark.asyncio
async def test_capture_unknown_ptz_node_fails(tmp_path: Path) -> None:
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
    await _register_ptz_node(manager, manager._storage, "cam-1")

    result = await manager.capture("cam-1")
    assert result.status == "FAILED"
    assert result.failure_class == "RuntimeError"


class _HomingDriver(_MockDriver):
    def __init__(self) -> None:
        super().__init__()
        self.home_calls = 0

    async def go_home(self) -> None:
        self.home_calls += 1


@pytest.mark.asyncio
async def test_slew_records_active_track_id_in_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager, storage, _ = await _manager(tmp_path)
    mock_driver = _MockDriver()
    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: mock_driver)
    await _register_ptz_node(manager, manager._storage, "cam-1")

    await manager.slew_to_target("cam-1", (5.0, 5.0, 0.0), track_id="trk-9")

    status = await manager.get_status("cam-1")
    assert status is not None
    assert status.active_track_id == "trk-9"
    await manager.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_camera_returns_home_after_dwell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    manager, storage, _ = await _manager(tmp_path, slew_dwell_seconds=0.05)
    driver = _HomingDriver()
    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: driver)
    await _register_ptz_node(manager, manager._storage, "cam-1")

    result = await manager.slew_to_target("cam-1", (5.0, 5.0, 0.0), track_id="trk-1")
    assert result.status == "COMPLETED"

    await asyncio.sleep(0.15)

    assert driver.home_calls == 1
    status = await manager.get_status("cam-1")
    assert status is not None
    assert status.active_track_id is None
    await manager.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_home_return_disabled_when_dwell_is_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    manager, storage, _ = await _manager(tmp_path, slew_dwell_seconds=0.0)
    driver = _HomingDriver()
    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: driver)
    await _register_ptz_node(manager, manager._storage, "cam-1")

    await manager.slew_to_target("cam-1", (5.0, 5.0, 0.0))
    await asyncio.sleep(0.1)

    assert driver.home_calls == 0
    await manager.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_new_slew_supersedes_pending_home_return(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    manager, storage, _ = await _manager(
        tmp_path, min_slew_interval_seconds=0.0, slew_dwell_seconds=0.1
    )
    driver = _HomingDriver()
    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: driver)
    await _register_ptz_node(manager, manager._storage, "cam-1")

    await manager.slew_to_target("cam-1", (5.0, 0.0, 0.0), track_id="trk-1")
    await asyncio.sleep(0.05)
    # Second slew inside the dwell window restarts the timer.
    await manager.slew_to_target("cam-1", (0.0, 5.0, 0.0), track_id="trk-2")
    await asyncio.sleep(0.06)

    assert driver.home_calls == 0
    await asyncio.sleep(0.08)
    assert driver.home_calls == 1
    await manager.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_status_reports_error_when_driver_get_status_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FlakyStatusDriver(_MockDriver):
        async def get_status(self) -> dict:
            raise RuntimeError("camera unreachable")

    manager, storage, _ = await _manager(tmp_path)
    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: _FlakyStatusDriver())
    await _register_ptz_node(manager, manager._storage, "cam-1")

    statuses = await manager.list_status()
    assert len(statuses) == 1
    assert statuses[0].state == "error"
    await manager.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_poll_loop_survives_driver_status_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    class _FlakyStatusDriver(_MockDriver):
        async def get_status(self) -> dict:
            raise RuntimeError("camera unreachable")

    manager, storage, events = await _manager(tmp_path, status_poll_interval_seconds=0.02)
    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: _FlakyStatusDriver())
    await _register_ptz_node(manager, manager._storage, "cam-1")

    await asyncio.sleep(0.1)

    assert manager._poll_task is not None
    assert not manager._poll_task.done()
    assert any(e.get("state") == "error" for e in events)
    await manager.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_poll_loop_reconnects_previously_failed_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    attempts = {"count": 0}

    class _EventuallyConnectingDriver(_MockDriver):
        async def connect(self) -> None:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("camera booting")

    driver = _EventuallyConnectingDriver()
    manager, storage, _ = await _manager(tmp_path, status_poll_interval_seconds=0.02)
    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: driver)
    await _register_ptz_node(manager, manager._storage, "cam-1")

    status = await manager.get_status("cam-1")
    assert status is not None
    assert status.state == "offline"

    await asyncio.sleep(0.1)

    status = await manager.get_status("cam-1")
    assert status is not None
    assert status.state == "idle"
    assert attempts["count"] >= 2
    await manager.stop()
    await storage.close()


@pytest.mark.asyncio
async def test_deleting_last_ptz_node_stops_poll_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, storage, _ = await _manager(tmp_path)
    monkeypatch.setattr(registry_module, "_build_driver", lambda spec, *, snapshot_dir: _MockDriver())
    await _register_ptz_node(manager, manager._storage, "cam-1")
    assert manager._poll_task is not None

    await manager.detach("cam-1")

    assert manager._poll_task is None
    await storage.close()


@pytest.mark.asyncio
async def test_register_without_transport_host_leaves_driver_none(tmp_path: Path) -> None:
    manager, _, _ = await _manager(tmp_path)
    await _register_ptz_node(manager, manager._storage, "cam-no-host", host=None)

    status = await manager.get_status("cam-no-host")
    assert status is not None
    assert status.state == "offline"
