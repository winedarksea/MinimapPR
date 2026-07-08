"""Generic ONVIF Profile S PTZ driver with capability probing and Reolink quirks.

``onvif`` / ``zeep`` are imported lazily so the core runtime carries no hard
dependency on them when no effector is registered (mirrors the optional
YAMNet-classifier import pattern in ``classifiers/factory.py``).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from minimappr.core.effectors.base import EffectorCapabilities, EffectorCommand, ExecutionResult
from minimappr.core.effectors.geometry import compute_pan_tilt
from minimappr.models import NodeOrientation, Vec3

logger = logging.getLogger(__name__)

try:
    from onvif import ONVIFCamera  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional runtime dependency
    ONVIFCamera = None  # type: ignore[assignment,misc]


class OnvifPtzError(RuntimeError):
    pass


# ONVIF PTZ configuration-options space URIs that indicate operation support.
_ABSOLUTE_SPACE_KEY = "AbsolutePanTiltPositionSpace"
_RELATIVE_SPACE_KEY = "RelativePanTiltTranslationSpace"
_CONTINUOUS_SPACE_KEY = "ContinuousPanTiltVelocitySpace"
_ZOOM_SPACE_KEYS = (
    "AbsoluteZoomPositionSpace",
    "RelativeZoomTranslationSpace",
    "ContinuousZoomVelocitySpace",
)

# Angular deltas below this are treated as already-on-target for the timed
# ContinuousMove approximation (its positioning error is far coarser anyway).
_MIN_MOVE_DEG = 0.5


def select_movement_strategy(supported_spaces: set[str]) -> str:
    """Pick the best available PTZ move operation, in ONVIF-reliability order.

    AbsoluteMove is preferred (single-shot, no timing needed). RelativeMove
    is next (delta from GetStatus position). ContinuousMove with a computed
    timed Stop is last resort — several Reolink models reject AbsoluteMove
    outright but always support ContinuousMove.
    """
    if _ABSOLUTE_SPACE_KEY in supported_spaces:
        return "AbsoluteMove"
    if _RELATIVE_SPACE_KEY in supported_spaces:
        return "RelativeMove"
    if _CONTINUOUS_SPACE_KEY in supported_spaces:
        return "ContinuousMove"
    raise OnvifPtzError("PTZ node advertises no known move operation")


def select_snapshot_strategy(*, snapshot_uri: str | None) -> str:
    """GetSnapshotUri HTTP GET is preferred; RTSP frame-grab is the default-capable
    fallback — some Reolink firmwares simply don't serve a snapshot URI."""
    return "snapshot_uri" if snapshot_uri else "rtsp_frame_grab"


async def rtsp_frame_grab(rtsp_url: str, output_path: Path, *, timeout_s: float = 8.0) -> None:
    """Grab a single JPEG frame from an RTSP stream via ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-frames:v", "1",
        "-f", "image2",
        str(output_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise OnvifPtzError(f"RTSP frame-grab timed out after {timeout_s}s") from None
    if proc.returncode != 0 or not output_path.exists():
        raise OnvifPtzError(
            f"RTSP frame-grab failed: {stderr.decode('utf-8', errors='replace')[-500:]}"
        )


class OnvifPtzDriver:
    """Effector Protocol implementation for a generic ONVIF Profile S PTZ camera."""

    def __init__(
        self,
        *,
        effector_id: str,
        host: str,
        port: int,
        username: str,
        password: str,
        camera_pos: Vec3,
        camera_orientation: NodeOrientation,
        snapshot_dir: Path,
        rtsp_url: str | None = None,
        continuous_move_speed: float = 0.5,
        wsdl_dir: str | None = None,
    ) -> None:
        self._effector_id = effector_id
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._camera_pos = camera_pos
        self._camera_orientation = camera_orientation
        self._snapshot_dir = snapshot_dir
        self._rtsp_url = rtsp_url
        self._continuous_move_speed = continuous_move_speed
        self._wsdl_dir = wsdl_dir

        self._camera: Any = None
        self._media_service: Any = None
        self._ptz_service: Any = None
        self._media_profile_token: str | None = None
        self._ptz_config_token: str | None = None
        self._capabilities: EffectorCapabilities | None = None
        self._snapshot_uri: str | None = None
        self._armed = False
        self._state = "offline"
        self._last_pan_deg: float | None = None
        self._last_tilt_deg: float | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        if ONVIFCamera is None:
            raise OnvifPtzError(
                "onvif-zeep-async is not installed; add it to the environment to enable PTZ effectors"
            )
        self._camera = ONVIFCamera(self._host, self._port, self._username, self._password, self._wsdl_dir)
        await self._camera.update_xaddrs()
        self._media_service = await self._camera.create_media_service()
        self._ptz_service = await self._camera.create_ptz_service()

        profiles = await self._media_service.GetProfiles()
        if not profiles:
            raise OnvifPtzError("ONVIF device advertises no media profiles")
        profile = profiles[0]
        self._media_profile_token = profile.token

        configs = await self._ptz_service.GetConfigurations()
        if not configs:
            raise OnvifPtzError("ONVIF device advertises no PTZ configurations")
        self._ptz_config_token = configs[0].token

        self._capabilities = await self._probe_capabilities()
        self._snapshot_uri = await self._probe_snapshot_uri()
        self._state = "idle"

    async def _probe_capabilities(self) -> EffectorCapabilities:
        supported_spaces: set[str] = set()
        has_zoom = False
        try:
            options = await self._ptz_service.GetConfigurationOptions(
                {"ConfigurationToken": self._ptz_config_token}
            )
            spaces = getattr(options, "Spaces", None)
            if spaces is not None:
                if getattr(spaces, "AbsolutePanTiltPositionSpace", None):
                    supported_spaces.add(_ABSOLUTE_SPACE_KEY)
                if getattr(spaces, "RelativePanTiltTranslationSpace", None):
                    supported_spaces.add(_RELATIVE_SPACE_KEY)
                if getattr(spaces, "ContinuousPanTiltVelocitySpace", None):
                    supported_spaces.add(_CONTINUOUS_SPACE_KEY)
                has_zoom = any(getattr(spaces, key, None) for key in _ZOOM_SPACE_KEYS)
        except Exception as exc:  # pragma: no cover - defensive against quirky firmware
            logger.warning("PTZ capability probe failed for %s: %s", self._effector_id, exc)

        if not supported_spaces:
            # Reolink and similar consumer PTZ firmwares are notoriously
            # unreliable about advertising spaces; ContinuousMove is
            # near-universally supported so it is the safe assumption.
            supported_spaces.add(_CONTINUOUS_SPACE_KEY)

        strategy = select_movement_strategy(supported_spaces)
        return EffectorCapabilities(
            movement_strategies=sorted(supported_spaces),
            selected_movement_strategy=strategy,
            snapshot_strategies=[],
            selected_snapshot_strategy=None,
            has_zoom=has_zoom,
        )

    async def _probe_snapshot_uri(self) -> str | None:
        try:
            result = await self._media_service.GetSnapshotUri({"ProfileToken": self._media_profile_token})
            uri = getattr(result, "Uri", None)
            if uri:
                return str(uri)
        except Exception as exc:
            logger.info("GetSnapshotUri unavailable for %s, will use RTSP frame-grab: %s", self._effector_id, exc)
        return None

    async def get_capabilities(self) -> EffectorCapabilities:
        if self._capabilities is None:
            raise OnvifPtzError("driver not connected")
        caps = self._capabilities
        snapshot_strategy = select_snapshot_strategy(snapshot_uri=self._snapshot_uri)
        caps.snapshot_strategies = ["snapshot_uri", "rtsp_frame_grab"]
        caps.selected_snapshot_strategy = snapshot_strategy
        return caps

    async def arm(self, *, zone_id: str | None = None) -> bool:
        del zone_id
        self._armed = True
        return True

    async def disarm(self) -> bool:
        self._armed = False
        return True

    async def execute(self, command: EffectorCommand) -> ExecutionResult:
        async with self._lock:
            if self._capabilities is None:
                return ExecutionResult(status="FAILED", failure_class="not_connected")
            pan_deg, tilt_deg = compute_pan_tilt(
                self._camera_pos, self._camera_orientation, command.target_position_m
            )
            if self._capabilities.selected_movement_strategy is None:
                return ExecutionResult(status="FAILED", failure_class="no_movement_strategy")
            self._state = "slewing"
            try:
                await self._move_via_selected_strategy(pan_deg, tilt_deg)
            except Exception as exc:
                self._state = "error"
                return ExecutionResult(status="FAILED", failure_class=f"{type(exc).__name__}", detail=str(exc))
            self._last_pan_deg = pan_deg
            self._last_tilt_deg = tilt_deg
            self._state = "idle"
            return ExecutionResult(status="COMPLETED")

    async def go_home(self) -> None:
        """Return the camera to its registered home bearing (pan=0, tilt=0).

        Prefers the ONVIF home preset; many consumer PTZ firmwares don't
        implement one, so falling back to a normal slew to the home bearing
        is treated as an equally valid path, not an error.
        """
        async with self._lock:
            if self._capabilities is None:
                raise OnvifPtzError("driver not connected")
            self._state = "slewing"
            try:
                try:
                    await self._ptz_service.GotoHomePosition(
                        {"ProfileToken": self._media_profile_token}
                    )
                except Exception as exc:
                    logger.debug(
                        "GotoHomePosition unavailable for %s (%s); slewing to home bearing",
                        self._effector_id,
                        exc,
                    )
                    await self._move_via_selected_strategy(0.0, 0.0)
            except Exception:
                self._state = "error"
                raise
            self._last_pan_deg = 0.0
            self._last_tilt_deg = 0.0
            self._state = "idle"

    async def _move_via_selected_strategy(self, pan_deg: float, tilt_deg: float) -> None:
        strategy = self._capabilities.selected_movement_strategy if self._capabilities else None
        if strategy == "AbsoluteMove":
            await self._absolute_move(pan_deg, tilt_deg)
        elif strategy == "RelativeMove":
            await self._relative_move(pan_deg, tilt_deg)
        elif strategy == "ContinuousMove":
            await self._continuous_move_timed(pan_deg, tilt_deg)
        else:
            raise OnvifPtzError("no movement strategy selected")

    async def _absolute_move(self, pan_deg: float, tilt_deg: float) -> None:
        await self._ptz_service.AbsoluteMove(
            {
                "ProfileToken": self._media_profile_token,
                "Position": {"PanTilt": {"x": pan_deg / 180.0, "y": tilt_deg / 90.0}},
            }
        )

    async def _relative_move(self, pan_deg: float, tilt_deg: float) -> None:
        status = await self._ptz_service.GetStatus({"ProfileToken": self._media_profile_token})
        current = getattr(status, "Position", None)
        current_pan = getattr(getattr(current, "PanTilt", None), "x", 0.0) or 0.0
        current_tilt = getattr(getattr(current, "PanTilt", None), "y", 0.0) or 0.0
        target_pan = pan_deg / 180.0
        target_tilt = tilt_deg / 90.0
        await self._ptz_service.RelativeMove(
            {
                "ProfileToken": self._media_profile_token,
                "Translation": {
                    "PanTilt": {"x": target_pan - current_pan, "y": target_tilt - current_tilt}
                },
            }
        )

    async def _continuous_move_timed(self, pan_deg: float, tilt_deg: float) -> None:
        """Approximate an absolute slew via velocity move + computed dwell + Stop.

        Duration is estimated from the angular delta and a conservative
        assumed slew rate; this is a rough approximation appropriate for
        cameras that only support ContinuousMove.
        """
        current_pan = self._last_pan_deg or 0.0
        current_tilt = self._last_tilt_deg or 0.0
        pan_delta = pan_deg - current_pan
        tilt_delta = tilt_deg - current_tilt
        dominant_delta = max(abs(pan_delta), abs(tilt_delta))
        if dominant_delta < _MIN_MOVE_DEG:
            return
        assumed_deg_per_s = 30.0 * self._continuous_move_speed
        duration_s = min(5.0, max(0.2, dominant_delta / max(assumed_deg_per_s, 1e-6)))

        # Scale each axis by its share of the dominant delta so both axes
        # arrive together and an axis that is already on target stays put.
        pan_velocity = self._continuous_move_speed * (pan_delta / dominant_delta)
        tilt_velocity = self._continuous_move_speed * (tilt_delta / dominant_delta)

        await self._ptz_service.ContinuousMove(
            {
                "ProfileToken": self._media_profile_token,
                "Velocity": {"PanTilt": {"x": pan_velocity, "y": tilt_velocity}},
            }
        )
        await asyncio.sleep(duration_s)
        await self._ptz_service.Stop({"ProfileToken": self._media_profile_token, "PanTilt": True})

    async def get_status(self) -> dict[str, Any]:
        state = self._state
        if self._ptz_service is not None and state != "slewing":
            try:
                status = await self._ptz_service.GetStatus(
                    {"ProfileToken": self._media_profile_token}
                )
                pan_tilt = getattr(getattr(status, "Position", None), "PanTilt", None)
                x = getattr(pan_tilt, "x", None)
                y = getattr(pan_tilt, "y", None)
                if x is not None:
                    self._last_pan_deg = float(x) * 180.0
                if y is not None:
                    self._last_tilt_deg = float(y) * 90.0
                if state == "error":
                    # Camera is answering again after a failed move/snapshot.
                    self._state = state = "idle"
            except Exception as exc:
                logger.debug("ONVIF GetStatus failed for %s: %s", self._effector_id, exc)
                state = "error"
        return {
            "effector_id": self._effector_id,
            "state": state,
            "armed": self._armed,
            "pan_deg": self._last_pan_deg,
            "tilt_deg": self._last_tilt_deg,
        }

    async def snapshot(self, *, dest_path: Path) -> Path:
        strategy = select_snapshot_strategy(snapshot_uri=self._snapshot_uri)
        if strategy == "snapshot_uri":
            try:
                await self._snapshot_via_uri(dest_path)
                return dest_path
            except Exception as exc:
                logger.warning(
                    "snapshot_uri fetch failed for %s, falling back to RTSP frame-grab: %s",
                    self._effector_id,
                    exc,
                )
        if not self._rtsp_url:
            raise OnvifPtzError("no snapshot URI available and no RTSP URL configured for frame-grab fallback")
        await rtsp_frame_grab(self._rtsp_url, dest_path)
        return dest_path

    async def _snapshot_via_uri(self, dest_path: Path) -> None:
        import httpx

        if not self._snapshot_uri:
            raise OnvifPtzError("no snapshot URI available")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        auth = (self._username, self._password) if self._username else None
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(self._snapshot_uri, auth=auth)
            response.raise_for_status()
            dest_path.write_bytes(response.content)
