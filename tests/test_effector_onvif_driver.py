"""Movement/snapshot fallback selection and execute() paths for OnvifPtzDriver.

Uses mocked ONVIF service objects (no network, no real onvif-zeep-async
dependency required) to exercise capability probing and the
AbsoluteMove -> RelativeMove -> ContinuousMove fallback order.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from minimappr.core.effectors import onvif_ptz
from minimappr.core.effectors.base import EffectorCapabilities, EffectorCommand
from minimappr.core.effectors.onvif_ptz import (
    OnvifPtzDriver,
    OnvifPtzError,
    select_movement_strategy,
    select_snapshot_strategy,
)
from minimappr.models import NodeOrientation


def _driver(tmp_path: Path) -> OnvifPtzDriver:
    return OnvifPtzDriver(
        effector_id="cam-1",
        host="192.168.1.50",
        port=80,
        username="admin",
        password="secret",
        camera_pos=(0.0, 0.0, 2.0),
        camera_orientation=NodeOrientation(),
        snapshot_dir=tmp_path,
        rtsp_url="rtsp://192.168.1.50/stream1",
    )


class _Spaces:
    def __init__(self, *, absolute: bool = False, relative: bool = False, continuous: bool = False) -> None:
        self.AbsolutePanTiltPositionSpace = [object()] if absolute else None
        self.RelativePanTiltTranslationSpace = [object()] if relative else None
        self.ContinuousPanTiltVelocitySpace = [object()] if continuous else None


# ── movement strategy selection ────────────────────────────────────────────

def test_select_movement_strategy_prefers_absolute() -> None:
    supported = {"AbsolutePanTiltPositionSpace", "RelativePanTiltTranslationSpace", "ContinuousPanTiltVelocitySpace"}
    assert select_movement_strategy(supported) == "AbsoluteMove"


def test_select_movement_strategy_falls_back_to_relative() -> None:
    supported = {"RelativePanTiltTranslationSpace", "ContinuousPanTiltVelocitySpace"}
    assert select_movement_strategy(supported) == "RelativeMove"


def test_select_movement_strategy_falls_back_to_continuous() -> None:
    assert select_movement_strategy({"ContinuousPanTiltVelocitySpace"}) == "ContinuousMove"


def test_select_movement_strategy_raises_when_nothing_supported() -> None:
    with pytest.raises(OnvifPtzError):
        select_movement_strategy(set())


def test_select_snapshot_strategy_prefers_snapshot_uri() -> None:
    assert select_snapshot_strategy(snapshot_uri="http://cam/snap.jpg") == "snapshot_uri"


def test_select_snapshot_strategy_falls_back_to_rtsp() -> None:
    assert select_snapshot_strategy(snapshot_uri=None) == "rtsp_frame_grab"


# ── capability probing (mocked PTZ service) ────────────────────────────────

@pytest.mark.asyncio
async def test_probe_capabilities_absolute_move_capable(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    driver._ptz_service = AsyncMock()
    driver._ptz_config_token = "cfg-1"
    options = MagicMock()
    options.Spaces = _Spaces(absolute=True, relative=True, continuous=True)
    driver._ptz_service.GetConfigurationOptions.return_value = options

    caps = await driver._probe_capabilities()

    assert caps.selected_movement_strategy == "AbsoluteMove"


@pytest.mark.asyncio
async def test_probe_capabilities_relative_only(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    driver._ptz_service = AsyncMock()
    driver._ptz_config_token = "cfg-1"
    options = MagicMock()
    options.Spaces = _Spaces(relative=True)
    driver._ptz_service.GetConfigurationOptions.return_value = options

    caps = await driver._probe_capabilities()

    assert caps.selected_movement_strategy == "RelativeMove"


@pytest.mark.asyncio
async def test_probe_capabilities_continuous_only(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    driver._ptz_service = AsyncMock()
    driver._ptz_config_token = "cfg-1"
    options = MagicMock()
    options.Spaces = _Spaces(continuous=True)
    driver._ptz_service.GetConfigurationOptions.return_value = options

    caps = await driver._probe_capabilities()

    assert caps.selected_movement_strategy == "ContinuousMove"


@pytest.mark.asyncio
async def test_probe_capabilities_detects_zoom_space(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    driver._ptz_service = AsyncMock()
    driver._ptz_config_token = "cfg-1"
    options = MagicMock()
    spaces = _Spaces(absolute=True)
    spaces.AbsoluteZoomPositionSpace = [object()]
    options.Spaces = spaces
    driver._ptz_service.GetConfigurationOptions.return_value = options

    caps = await driver._probe_capabilities()

    assert caps.has_zoom is True


@pytest.mark.asyncio
async def test_probe_capabilities_defaults_to_continuous_when_probe_fails(tmp_path: Path) -> None:
    """Reolink and similar consumer firmware often fail GetConfigurationOptions;
    ContinuousMove is the safe universal fallback rather than erroring out."""
    driver = _driver(tmp_path)
    driver._ptz_service = AsyncMock()
    driver._ptz_config_token = "cfg-1"
    driver._ptz_service.GetConfigurationOptions.side_effect = RuntimeError("firmware quirk")

    caps = await driver._probe_capabilities()

    assert caps.selected_movement_strategy == "ContinuousMove"


# ── execute() strategy dispatch ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_uses_absolute_move_when_selected(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    driver._capabilities = EffectorCapabilities(
        movement_strategies=["AbsoluteMove"], selected_movement_strategy="AbsoluteMove"
    )
    driver._media_profile_token = "prof-1"
    driver._ptz_service = AsyncMock()

    result = await driver.execute(EffectorCommand(target_position_m=(10.0, 0.0, 2.0)))

    assert result.status == "COMPLETED"
    driver._ptz_service.AbsoluteMove.assert_awaited_once()
    driver._ptz_service.RelativeMove.assert_not_awaited()
    driver._ptz_service.ContinuousMove.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_uses_relative_move_when_selected(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    driver._capabilities = EffectorCapabilities(
        movement_strategies=["RelativeMove"], selected_movement_strategy="RelativeMove"
    )
    driver._media_profile_token = "prof-1"
    driver._ptz_service = AsyncMock()
    status = MagicMock()
    status.Position.PanTilt.x = 0.0
    status.Position.PanTilt.y = 0.0
    driver._ptz_service.GetStatus.return_value = status

    result = await driver.execute(EffectorCommand(target_position_m=(10.0, 0.0, 2.0)))

    assert result.status == "COMPLETED"
    driver._ptz_service.RelativeMove.assert_awaited_once()
    driver._ptz_service.AbsoluteMove.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_uses_continuous_move_and_stop_when_selected(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    driver._capabilities = EffectorCapabilities(
        movement_strategies=["ContinuousMove"], selected_movement_strategy="ContinuousMove"
    )
    driver._media_profile_token = "prof-1"
    driver._ptz_service = AsyncMock()

    # Small pan offset (~1.1 deg) -> real move required but minimal (fast) dwell.
    result = await driver.execute(EffectorCommand(target_position_m=(0.2, 10.0, 2.0)))

    assert result.status == "COMPLETED"
    driver._ptz_service.ContinuousMove.assert_awaited_once()
    driver._ptz_service.Stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_continuous_move_skips_when_already_on_target(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    driver._capabilities = EffectorCapabilities(
        movement_strategies=["ContinuousMove"], selected_movement_strategy="ContinuousMove"
    )
    driver._media_profile_token = "prof-1"
    driver._ptz_service = AsyncMock()

    # Target straight ahead of home view -> pan=tilt=0 -> nothing to do.
    result = await driver.execute(EffectorCommand(target_position_m=(0.0, 10.0, 2.0)))

    assert result.status == "COMPLETED"
    driver._ptz_service.ContinuousMove.assert_not_awaited()
    driver._ptz_service.Stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_continuous_move_velocity_is_proportional_per_axis(tmp_path: Path) -> None:
    """A pan-only slew must not command any tilt velocity (and vice versa) —
    fixed-magnitude velocities on both axes would overshoot the smaller one."""
    driver = _driver(tmp_path)
    driver._capabilities = EffectorCapabilities(
        movement_strategies=["ContinuousMove"], selected_movement_strategy="ContinuousMove"
    )
    driver._media_profile_token = "prof-1"
    driver._ptz_service = AsyncMock()

    # Pan-only target (~2 deg right, level with the camera).
    result = await driver.execute(EffectorCommand(target_position_m=(0.35, 10.0, 2.0)))

    assert result.status == "COMPLETED"
    velocity = driver._ptz_service.ContinuousMove.await_args.args[0]["Velocity"]["PanTilt"]
    assert velocity["x"] > 0.0
    assert velocity["y"] == 0.0


# ── go_home ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_go_home_uses_onvif_home_preset(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    driver._capabilities = EffectorCapabilities(
        movement_strategies=["AbsoluteMove"], selected_movement_strategy="AbsoluteMove"
    )
    driver._media_profile_token = "prof-1"
    driver._ptz_service = AsyncMock()
    driver._last_pan_deg = 45.0
    driver._last_tilt_deg = 10.0

    await driver.go_home()

    driver._ptz_service.GotoHomePosition.assert_awaited_once()
    driver._ptz_service.AbsoluteMove.assert_not_awaited()
    assert driver._last_pan_deg == 0.0
    assert driver._last_tilt_deg == 0.0
    assert driver._state == "idle"


@pytest.mark.asyncio
async def test_go_home_falls_back_to_slew_when_home_preset_unsupported(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    driver._capabilities = EffectorCapabilities(
        movement_strategies=["AbsoluteMove"], selected_movement_strategy="AbsoluteMove"
    )
    driver._media_profile_token = "prof-1"
    driver._ptz_service = AsyncMock()
    driver._ptz_service.GotoHomePosition.side_effect = RuntimeError("not supported")

    await driver.go_home()

    driver._ptz_service.AbsoluteMove.assert_awaited_once()
    position = driver._ptz_service.AbsoluteMove.await_args.args[0]["Position"]["PanTilt"]
    assert position == {"x": 0.0, "y": 0.0}
    assert driver._state == "idle"


# ── live get_status ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_status_reads_live_onvif_position(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    driver._media_profile_token = "prof-1"
    driver._ptz_service = AsyncMock()
    driver._state = "idle"
    status = MagicMock()
    status.Position.PanTilt.x = 0.5
    status.Position.PanTilt.y = -0.5
    driver._ptz_service.GetStatus.return_value = status

    raw = await driver.get_status()

    assert raw["state"] == "idle"
    assert raw["pan_deg"] == pytest.approx(90.0)
    assert raw["tilt_deg"] == pytest.approx(-45.0)


@pytest.mark.asyncio
async def test_get_status_reports_error_when_camera_unreachable(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    driver._media_profile_token = "prof-1"
    driver._ptz_service = AsyncMock()
    driver._state = "idle"
    driver._ptz_service.GetStatus.side_effect = RuntimeError("timeout")

    raw = await driver.get_status()

    assert raw["state"] == "error"


@pytest.mark.asyncio
async def test_get_status_recovers_from_error_when_camera_answers(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    driver._media_profile_token = "prof-1"
    driver._ptz_service = AsyncMock()
    driver._state = "error"
    status = MagicMock()
    status.Position.PanTilt.x = 0.0
    status.Position.PanTilt.y = 0.0
    driver._ptz_service.GetStatus.return_value = status

    raw = await driver.get_status()

    assert raw["state"] == "idle"
    assert driver._state == "idle"


@pytest.mark.asyncio
async def test_execute_returns_failed_on_driver_exception(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    driver._capabilities = EffectorCapabilities(
        movement_strategies=["AbsoluteMove"], selected_movement_strategy="AbsoluteMove"
    )
    driver._media_profile_token = "prof-1"
    driver._ptz_service = AsyncMock()
    driver._ptz_service.AbsoluteMove.side_effect = RuntimeError("camera unreachable")

    result = await driver.execute(EffectorCommand(target_position_m=(10.0, 0.0, 2.0)))

    assert result.status == "FAILED"
    assert result.failure_class == "RuntimeError"


# ── snapshot fallback ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_snapshot_falls_back_to_rtsp_frame_grab_when_no_snapshot_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = _driver(tmp_path)
    driver._snapshot_uri = None
    calls: dict[str, str] = {}

    async def _fake_grab(rtsp_url: str, output_path: Path, *, timeout_s: float = 8.0) -> None:
        calls["rtsp_url"] = rtsp_url
        output_path.write_bytes(b"fake-jpeg")

    monkeypatch.setattr(onvif_ptz, "rtsp_frame_grab", _fake_grab)

    dest = tmp_path / "out.jpg"
    result_path = await driver.snapshot(dest_path=dest)

    assert calls["rtsp_url"] == "rtsp://192.168.1.50/stream1"
    assert result_path == dest
    assert dest.read_bytes() == b"fake-jpeg"


@pytest.mark.asyncio
async def test_snapshot_falls_back_to_rtsp_when_snapshot_uri_fetch_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = _driver(tmp_path)
    driver._snapshot_uri = "http://192.168.1.50/snap.jpg"

    async def _failing_snapshot_via_uri(dest_path: Path) -> None:
        raise RuntimeError("401 unauthorized")

    monkeypatch.setattr(driver, "_snapshot_via_uri", _failing_snapshot_via_uri)

    calls: dict[str, str] = {}

    async def _fake_grab(rtsp_url: str, output_path: Path, *, timeout_s: float = 8.0) -> None:
        calls["rtsp_url"] = rtsp_url
        output_path.write_bytes(b"fallback-jpeg")

    monkeypatch.setattr(onvif_ptz, "rtsp_frame_grab", _fake_grab)

    dest = tmp_path / "out.jpg"
    await driver.snapshot(dest_path=dest)

    assert calls["rtsp_url"] == "rtsp://192.168.1.50/stream1"
    assert dest.read_bytes() == b"fallback-jpeg"


@pytest.mark.asyncio
async def test_snapshot_raises_when_no_uri_and_no_rtsp_configured(tmp_path: Path) -> None:
    driver = OnvifPtzDriver(
        effector_id="cam-2",
        host="192.168.1.51",
        port=80,
        username="",
        password="",
        camera_pos=(0.0, 0.0, 0.0),
        camera_orientation=NodeOrientation(),
        snapshot_dir=tmp_path,
        rtsp_url=None,
    )
    driver._snapshot_uri = None

    with pytest.raises(OnvifPtzError):
        await driver.snapshot(dest_path=tmp_path / "out.jpg")
