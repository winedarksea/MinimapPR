"""EffectorManager: optional PTZ capability driver (lifecycle + status broadcast).

Template: ``FederationCoordinator`` (core/federation.py) — an optional subsystem
that is fully dormant (no background work) when unconfigured, and activates
without a restart once the first PTZ-capable node is registered.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from minimappr.core.effectors.base import Effector, EffectorCommand, ExecutionResult
from minimappr.core.effectors.onvif_ptz import OnvifPtzDriver
from minimappr.interfaces import StorageBackend
from minimappr.models import (
    NodeCapability,
    NodeOrientation,
    NodeSafetyConfig,
    PtzStatus,
    Vec3,
)

logger = logging.getLogger(__name__)

LiveCallback = Callable[[dict], Awaitable[None]]
TargetZoneResolver = Callable[[Vec3], Awaitable[set[str]]]


@dataclass(slots=True)
class PtzNodeSpec:
    id: str
    position_m: Vec3 | None
    orientation: NodeOrientation
    transport: dict[str, Any]
    metadata: dict[str, Any]
    properties: dict[str, Any]
    safety: NodeSafetyConfig


@dataclass(slots=True)
class PtzRuntime:
    spec: PtzNodeSpec
    driver: Effector | None = None
    connect_error: str | None = None
    last_slew_ns: int | None = None
    active_track_id: str | None = None
    home_return_task: asyncio.Task[None] | None = None


@dataclass(slots=True)
class EffectorManagerConfig:
    snapshot_dir: Path
    min_slew_interval_seconds: float = 3.0
    status_poll_interval_seconds: float = 5.0
    slew_dwell_seconds: float = 10.0


class EffectorManager:
    def __init__(
        self,
        *,
        storage: StorageBackend,
        live_callback: LiveCallback | None,
        config: EffectorManagerConfig,
    ) -> None:
        self._storage = storage
        self._live_callback = live_callback
        self._config = config
        self._runtimes: dict[str, PtzRuntime] = {}
        self._lock = asyncio.Lock()
        self._poll_task: asyncio.Task[None] | None = None
        self._target_zone_resolver: TargetZoneResolver | None = None

    @property
    def enabled(self) -> bool:
        return len(self._runtimes) > 0

    async def start(self) -> None:
        nodes = await self._storage.list_nodes()
        for row in nodes:
            if NodeCapability.PTZ_CAMERA.value not in {str(capability) for capability in row.get("capabilities", [])}:
                continue
            spec = _ptz_spec_from_node_row(row)
            await self._connect_and_register_runtime(spec, persist=False)
        self._ensure_poll_task()

    async def stop(self) -> None:
        task = self._poll_task
        self._poll_task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        async with self._lock:
            runtimes = list(self._runtimes.values())
        for runtime in runtimes:
            self._cancel_home_return(runtime)

    async def register_node(self, node: dict[str, Any]) -> PtzNodeSpec:
        spec = _ptz_spec_from_node_row(node)
        await self._connect_and_register_runtime(spec, persist=True)
        self._ensure_poll_task()
        return spec

    def set_target_zone_resolver(self, resolver: TargetZoneResolver | None) -> None:
        self._target_zone_resolver = resolver

    async def detach(self, node_id: str) -> bool:
        async with self._lock:
            runtime = self._runtimes.pop(node_id, None)
            registry_empty = not self._runtimes
        if runtime is not None:
            self._cancel_home_return(runtime)
        await self._broadcast_status(node_id)
        if registry_empty:
            # Back to fully dormant: no PTZ nodes means no background work.
            task = self._poll_task
            self._poll_task = None
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        return runtime is not None

    async def list_status(self) -> list[PtzStatus]:
        results: list[PtzStatus] = []
        async with self._lock:
            runtimes = list(self._runtimes.values())
        for runtime in runtimes:
            results.append(await self._status_for_runtime(runtime))
        return results

    async def get_status(self, node_id: str) -> PtzStatus | None:
        runtime = self._runtimes.get(node_id)
        if runtime is None:
            return None
        return await self._status_for_runtime(runtime)

    async def get_capabilities(self, node_id: str) -> dict[str, Any] | None:
        runtime = self._runtimes.get(node_id)
        if runtime is None or runtime.driver is None:
            return None
        try:
            caps = await runtime.driver.get_capabilities()
        except Exception as exc:
            logger.debug("ptz node %s get_capabilities failed: %s", node_id, exc)
            return None
        return {
            "movement_strategies": caps.movement_strategies,
            "selected_movement_strategy": caps.selected_movement_strategy,
            "snapshot_strategies": caps.snapshot_strategies,
            "selected_snapshot_strategy": caps.selected_snapshot_strategy,
            "has_zoom": caps.has_zoom,
        }

    async def get_safety(self, node_id: str) -> NodeSafetyConfig | None:
        runtime = self._runtimes.get(node_id)
        if runtime is not None:
            return runtime.spec.safety
        row = await self._storage.get_node_by_id(node_id)
        if row is None:
            return None
        return NodeSafetyConfig.model_validate(row.get("safety") or {})

    async def update_safety(
        self,
        node_id: str,
        safety: NodeSafetyConfig,
    ) -> NodeSafetyConfig | None:
        if not await self._storage.update_node_operator_fields(node_id, safety=safety.model_dump(mode="json")):
            return None
        runtime = self._runtimes.get(node_id)
        if runtime is not None:
            runtime.spec.safety = safety
        await self._broadcast_status(node_id)
        return safety

    async def arm(self, node_id: str, *, zone_id: str | None = None) -> ExecutionResult:
        runtime = self._runtimes.get(node_id)
        if runtime is None or runtime.driver is None:
            return ExecutionResult(status="FAILED", failure_class="not_found")
        try:
            accepted = await runtime.driver.arm(zone_id=zone_id)
        except Exception as exc:
            logger.warning("ptz node %s arm failed: %s", node_id, exc)
            return ExecutionResult(status="FAILED", failure_class=type(exc).__name__, detail=str(exc))
        await self._broadcast_status(node_id)
        if not accepted:
            return ExecutionResult(status="REJECTED", failure_class="driver_refused")
        return ExecutionResult(status="COMPLETED")

    async def disarm(self, node_id: str) -> ExecutionResult:
        runtime = self._runtimes.get(node_id)
        if runtime is None or runtime.driver is None:
            return ExecutionResult(status="FAILED", failure_class="not_found")
        try:
            accepted = await runtime.driver.disarm()
        except Exception as exc:
            logger.warning("ptz node %s disarm failed: %s", node_id, exc)
            return ExecutionResult(status="FAILED", failure_class=type(exc).__name__, detail=str(exc))
        await self._broadcast_status(node_id)
        if not accepted:
            return ExecutionResult(status="REJECTED", failure_class="driver_refused")
        return ExecutionResult(status="COMPLETED")

    async def slew_to_target(
        self,
        node_id: str,
        target_pos: Vec3,
        *,
        track_id: str | None = None,
        detection_id: str | None = None,
        execution_id: str | None = None,
    ) -> ExecutionResult:
        runtime = self._runtimes.get(node_id)
        if runtime is None or runtime.driver is None:
            return ExecutionResult(
                status="FAILED",
                execution_id=execution_id or uuid.uuid4().hex,
                failure_class="not_found",
            )

        rejection = await self._check_interlocks(node_id, runtime, target_pos)
        if rejection is not None:
            logger.info("ptz node %s slew rejected by interlock: %s", node_id, rejection)
            return ExecutionResult(
                status="REJECTED",
                execution_id=execution_id or uuid.uuid4().hex,
                failure_class=rejection,
            )

        command = EffectorCommand(target_position_m=target_pos, track_id=track_id, detection_id=detection_id)
        result = await runtime.driver.execute(command)
        if execution_id is not None:
            result.execution_id = execution_id
        if result.status == "COMPLETED":
            runtime.last_slew_ns = time.time_ns()
            runtime.active_track_id = track_id
            self._schedule_home_return(runtime)
        await self._broadcast_status(node_id)
        return result

    async def capture(
        self,
        node_id: str,
        *,
        track_id: str | None = None,
        detection_id: str | None = None,
        execution_id: str | None = None,
    ) -> ExecutionResult:
        runtime = self._runtimes.get(node_id)
        if runtime is None or runtime.driver is None:
            return ExecutionResult(
                status="FAILED",
                execution_id=execution_id or uuid.uuid4().hex,
                failure_class="not_found",
            )
        if not hasattr(runtime.driver, "snapshot"):
            return ExecutionResult(
                status="FAILED",
                execution_id=execution_id or uuid.uuid4().hex,
                failure_class="snapshot_unsupported",
            )

        now_ns = time.time_ns()
        dest_path = self._config.snapshot_dir / f"{node_id}-{now_ns}.jpg"
        try:
            await runtime.driver.snapshot(dest_path=dest_path)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("ptz node %s snapshot failed: %s", node_id, exc)
            return ExecutionResult(
                status="FAILED",
                execution_id=execution_id or uuid.uuid4().hex,
                failure_class=f"{type(exc).__name__}",
                detail=str(exc),
            )

        artifact_id = await self._storage.insert_node_artifact(
            node_id=node_id,
            track_id=track_id,
            detection_id=detection_id,
            kind="snapshot",
            path=str(dest_path),
            created_ns=now_ns,
        )
        await self._broadcast_status(node_id)
        return ExecutionResult(
            status="COMPLETED",
            execution_id=execution_id or uuid.uuid4().hex,
            result_refs=[artifact_id],
        )

    async def snapshot_live(self, node_id: str) -> Path | None:
        """Grab a fresh frame for the snapshot-refresh <img> live view.

        Overwrites a single per-node file rather than persisting an
        artifact record — this is a live-view poll, not an evidence capture
        (see ``capture()`` for the persisted/linked variant).
        """
        runtime = self._runtimes.get(node_id)
        if runtime is None or runtime.driver is None or not hasattr(runtime.driver, "snapshot"):
            return None
        dest_path = self._config.snapshot_dir / f"{node_id}-live.jpg"
        await runtime.driver.snapshot(dest_path=dest_path)  # type: ignore[attr-defined]
        return dest_path

    # ------------------------------------------------------------------

    async def _check_interlocks(
        self,
        node_id: str,
        runtime: PtzRuntime,
        target_pos: Vec3,
    ) -> str | None:
        """Pre-execution safety gate for camera movement commands."""
        safety = runtime.spec.safety
        if safety.require_arm_for_action:
            status = await self._status_for_runtime(runtime)
            if not status.armed:
                return "interlock:disarmed"

        if runtime.last_slew_ns is None:
            rate_limit_rejection = None
        else:
            min_interval_s = (
                safety.min_action_interval_seconds
                if safety.min_action_interval_seconds is not None
                else self._config.min_slew_interval_seconds
            )
            min_interval_ns = int(min_interval_s * 1_000_000_000)
            elapsed_ns = time.time_ns() - runtime.last_slew_ns
            rate_limit_rejection = "rate_limited" if elapsed_ns < min_interval_ns else None
        if rate_limit_rejection is not None:
            return rate_limit_rejection

        if safety.no_go_zone_ids and self._target_zone_resolver is not None:
            target_zone_ids = await self._target_zone_resolver(target_pos)
            no_go_ids = {zone_id.strip().lower() for zone_id in safety.no_go_zone_ids}
            if {zone_id.strip().lower() for zone_id in target_zone_ids} & no_go_ids:
                return "interlock:no_go_zone"
        return None

    def _cancel_home_return(self, runtime: PtzRuntime) -> None:
        task = runtime.home_return_task
        runtime.home_return_task = None
        if task is not None:
            task.cancel()

    def _schedule_home_return(self, runtime: PtzRuntime) -> None:
        """After a configurable dwell on the slewed-to target, return the
        camera to its home bearing. A newer slew supersedes (cancels) any
        pending return; dwell <= 0 disables the behavior."""
        self._cancel_home_return(runtime)
        if self._config.slew_dwell_seconds <= 0.0:
            return
        if runtime.driver is None or not hasattr(runtime.driver, "go_home"):
            return
        runtime.home_return_task = asyncio.create_task(
            self._home_return_after_dwell(runtime),
            name=f"ptz-home-return-{runtime.spec.id}",
        )

    async def _home_return_after_dwell(self, runtime: PtzRuntime) -> None:
        await asyncio.sleep(self._config.slew_dwell_seconds)
        driver = runtime.driver
        if driver is None:
            return
        try:
            await driver.go_home()  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("ptz node %s home return failed: %s", runtime.spec.id, exc)
            return
        runtime.active_track_id = None
        await self._broadcast_status(runtime.spec.id)

    async def _connect_and_register_runtime(self, spec: PtzNodeSpec, *, persist: bool) -> None:
        if persist:
            await self._storage.update_node_operator_fields(
                spec.id,
                transport=spec.transport,
                safety=spec.safety.model_dump(mode="json"),
                metadata=spec.metadata,
            )

        runtime = PtzRuntime(spec=spec)
        driver = _build_driver(spec, snapshot_dir=self._config.snapshot_dir)
        if driver is not None:
            try:
                await driver.connect()
                runtime.driver = driver
            except Exception as exc:
                runtime.connect_error = str(exc)
                logger.warning("ptz node %s failed to connect: %s", spec.id, exc)
        async with self._lock:
            self._runtimes[spec.id] = runtime
        await self._broadcast_status(spec.id)

    def _ensure_poll_task(self) -> None:
        if not self._runtimes:
            return
        if self._poll_task is not None and not self._poll_task.done():
            return
        self._poll_task = asyncio.create_task(self._poll_loop(), name="ptz-status-poll")

    async def _poll_loop(self) -> None:
        while True:
            await asyncio.sleep(self._config.status_poll_interval_seconds)
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                # One bad driver/cycle must never kill the status loop.
                logger.warning("ptz status poll iteration failed", exc_info=True)

    async def _poll_once(self) -> None:
        async with self._lock:
            runtimes = list(self._runtimes.values())
        for runtime in runtimes:
            if runtime.driver is None:
                await self._attempt_reconnect(runtime)
            await self._broadcast_status(runtime.spec.id)

    async def _attempt_reconnect(self, runtime: PtzRuntime) -> None:
        driver = _build_driver(runtime.spec, snapshot_dir=self._config.snapshot_dir)
        if driver is None:
            return
        try:
            await driver.connect()
        except Exception as exc:
            runtime.connect_error = str(exc)
            logger.debug("ptz node %s reconnect attempt failed: %s", runtime.spec.id, exc)
            return
        runtime.driver = driver
        runtime.connect_error = None
        logger.info("ptz node %s connected after retry", runtime.spec.id)

    async def _status_for_runtime(self, runtime: PtzRuntime) -> PtzStatus:
        if runtime.driver is None:
            return PtzStatus(node_id=runtime.spec.id, state="offline", armed=False)
        try:
            raw = await runtime.driver.get_status()
        except Exception as exc:
            logger.debug("ptz node %s get_status failed: %s", runtime.spec.id, exc)
            return PtzStatus(node_id=runtime.spec.id, state="error", armed=False)
        return PtzStatus(
            node_id=runtime.spec.id,
            state=raw.get("state", "offline"),
            pan_deg=raw.get("pan_deg"),
            tilt_deg=raw.get("tilt_deg"),
            zoom=raw.get("zoom"),
            armed=bool(raw.get("armed", False)),
            last_seen_ns=time.time_ns(),
            active_track_id=raw.get("active_track_id") or runtime.active_track_id,
        )

    async def _broadcast_status(self, node_id: str) -> None:
        if self._live_callback is None:
            return
        runtime = self._runtimes.get(node_id)
        if runtime is None:
            payload = {
                "type": "node_capability_status",
                "node_id": node_id,
                "effector_id": node_id,
                "capability": "ptz_camera",
                "status": {"state": "deleted"},
                "state": "deleted",
            }
        else:
            status = await self._status_for_runtime(runtime)
            status_payload = status.model_dump(mode="json")
            payload = {
                "type": "node_capability_status",
                "node_id": node_id,
                "effector_id": node_id,
                "capability": "ptz_camera",
                "status": status_payload,
                **status_payload,
            }
        try:
            await self._live_callback(payload)
        except Exception:
            logger.debug("ptz status broadcast failed for %s", node_id, exc_info=True)


def _ptz_spec_from_node_row(row: dict[str, Any]) -> PtzNodeSpec:
    orientation_row = row.get("orientation") or {}
    return PtzNodeSpec(
        id=row["id"],
        position_m=tuple(row["position_m"]) if row.get("position_m") else None,
        orientation=NodeOrientation.model_validate(orientation_row),
        transport=dict(row.get("transport") or {}),
        metadata=dict(row.get("metadata") or {}),
        properties=dict(row.get("properties") or {}),
        safety=NodeSafetyConfig.model_validate(row.get("safety") or {}),
    )


def _build_driver(spec: PtzNodeSpec, *, snapshot_dir: Path) -> Effector | None:
    transport = spec.transport or {}
    host = transport.get("host")
    if not host:
        logger.warning("ptz node %s has no transport.host; driver not created", spec.id)
        return None
    return OnvifPtzDriver(
        effector_id=spec.id,
        host=host,
        port=int(transport.get("port", 80)),
        username=str(transport.get("username", "")),
        password=str(transport.get("password", "")),
        camera_pos=spec.position_m or (0.0, 0.0, 0.0),
        camera_orientation=spec.orientation,
        snapshot_dir=snapshot_dir,
        rtsp_url=transport.get("rtsp_url"),
    )
