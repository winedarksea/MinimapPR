"""EffectorManager: optional effector subsystem (registry + lifecycle + status broadcast).

Template: ``FederationCoordinator`` (core/federation.py) — an optional subsystem
that is fully dormant (no background work) when unconfigured, and activates
without a restart once the first effector is registered. Here "unconfigured"
means the ``effectors`` DB table is empty.
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
    EffectorOrientation,
    EffectorSafetyConfig,
    EffectorSpec,
    EffectorStatus,
    EffectorType,
    Vec3,
)

logger = logging.getLogger(__name__)

LiveCallback = Callable[[dict], Awaitable[None]]
TargetZoneResolver = Callable[[Vec3], Awaitable[set[str]]]


@dataclass(slots=True)
class EffectorRuntime:
    spec: EffectorSpec
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
        self._runtimes: dict[str, EffectorRuntime] = {}
        self._lock = asyncio.Lock()
        self._poll_task: asyncio.Task[None] | None = None
        self._target_zone_resolver: TargetZoneResolver | None = None

    @property
    def enabled(self) -> bool:
        return len(self._runtimes) > 0

    async def start(self) -> None:
        specs = await self._storage.list_effectors()
        for row in specs:
            spec = _spec_from_row(row)
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

    async def register(self, spec: EffectorSpec) -> EffectorSpec:
        await self._connect_and_register_runtime(spec, persist=True)
        self._ensure_poll_task()
        return spec

    def set_target_zone_resolver(self, resolver: TargetZoneResolver | None) -> None:
        self._target_zone_resolver = resolver

    async def delete(self, effector_id: str) -> bool:
        async with self._lock:
            runtime = self._runtimes.pop(effector_id, None)
            registry_empty = not self._runtimes
        if runtime is not None:
            self._cancel_home_return(runtime)
        deleted = await self._storage.delete_effector(effector_id)
        await self._broadcast_status(effector_id)
        if registry_empty:
            # Back to fully dormant: no effectors means no background work.
            task = self._poll_task
            self._poll_task = None
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        return deleted

    async def list_status(self) -> list[EffectorStatus]:
        results: list[EffectorStatus] = []
        async with self._lock:
            runtimes = list(self._runtimes.values())
        for runtime in runtimes:
            results.append(await self._status_for_runtime(runtime))
        return results

    async def get_status(self, effector_id: str) -> EffectorStatus | None:
        runtime = self._runtimes.get(effector_id)
        if runtime is None:
            return None
        return await self._status_for_runtime(runtime)

    async def get_capabilities(self, effector_id: str) -> dict[str, Any] | None:
        runtime = self._runtimes.get(effector_id)
        if runtime is None or runtime.driver is None:
            return None
        try:
            caps = await runtime.driver.get_capabilities()
        except Exception as exc:
            logger.debug("effector %s get_capabilities failed: %s", effector_id, exc)
            return None
        return {
            "movement_strategies": caps.movement_strategies,
            "selected_movement_strategy": caps.selected_movement_strategy,
            "snapshot_strategies": caps.snapshot_strategies,
            "selected_snapshot_strategy": caps.selected_snapshot_strategy,
            "has_zoom": caps.has_zoom,
        }

    async def get_safety(self, effector_id: str) -> EffectorSafetyConfig | None:
        runtime = self._runtimes.get(effector_id)
        if runtime is not None:
            return _safety_from_spec(runtime.spec)
        row = await self._storage.get_effector_by_id(effector_id)
        if row is None:
            return None
        return _safety_from_properties(row.get("properties", {}))

    async def update_safety(
        self,
        effector_id: str,
        safety: EffectorSafetyConfig,
    ) -> EffectorSafetyConfig | None:
        runtime = self._runtimes.get(effector_id)
        if runtime is None:
            row = await self._storage.get_effector_by_id(effector_id)
            if row is None:
                return None
            spec = _spec_from_row(row)
        else:
            spec = runtime.spec

        properties = dict(spec.properties or {})
        properties["safety"] = safety.model_dump(mode="json")
        updated_spec = spec.model_copy(update={"properties": properties})
        await self._storage.upsert_effector(updated_spec, time.time_ns())
        if runtime is not None:
            runtime.spec = updated_spec
        await self._broadcast_status(effector_id)
        return safety

    async def arm(self, effector_id: str, *, zone_id: str | None = None) -> ExecutionResult:
        runtime = self._runtimes.get(effector_id)
        if runtime is None or runtime.driver is None:
            return ExecutionResult(status="FAILED", failure_class="not_found")
        try:
            accepted = await runtime.driver.arm(zone_id=zone_id)
        except Exception as exc:
            logger.warning("effector %s arm failed: %s", effector_id, exc)
            return ExecutionResult(status="FAILED", failure_class=type(exc).__name__, detail=str(exc))
        await self._broadcast_status(effector_id)
        if not accepted:
            return ExecutionResult(status="REJECTED", failure_class="driver_refused")
        return ExecutionResult(status="COMPLETED")

    async def disarm(self, effector_id: str) -> ExecutionResult:
        runtime = self._runtimes.get(effector_id)
        if runtime is None or runtime.driver is None:
            return ExecutionResult(status="FAILED", failure_class="not_found")
        try:
            accepted = await runtime.driver.disarm()
        except Exception as exc:
            logger.warning("effector %s disarm failed: %s", effector_id, exc)
            return ExecutionResult(status="FAILED", failure_class=type(exc).__name__, detail=str(exc))
        await self._broadcast_status(effector_id)
        if not accepted:
            return ExecutionResult(status="REJECTED", failure_class="driver_refused")
        return ExecutionResult(status="COMPLETED")

    async def slew_to_target(
        self,
        effector_id: str,
        target_pos: Vec3,
        *,
        track_id: str | None = None,
        detection_id: str | None = None,
        execution_id: str | None = None,
    ) -> ExecutionResult:
        runtime = self._runtimes.get(effector_id)
        if runtime is None or runtime.driver is None:
            return ExecutionResult(
                status="FAILED",
                execution_id=execution_id or uuid.uuid4().hex,
                failure_class="not_found",
            )

        rejection = await self._check_interlocks(effector_id, runtime, target_pos)
        if rejection is not None:
            logger.info("effector %s slew rejected by interlock: %s", effector_id, rejection)
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
        await self._broadcast_status(effector_id)
        return result

    async def capture(
        self,
        effector_id: str,
        *,
        track_id: str | None = None,
        detection_id: str | None = None,
        execution_id: str | None = None,
    ) -> ExecutionResult:
        runtime = self._runtimes.get(effector_id)
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
        dest_path = self._config.snapshot_dir / f"{effector_id}-{now_ns}.jpg"
        try:
            await runtime.driver.snapshot(dest_path=dest_path)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("effector %s snapshot failed: %s", effector_id, exc)
            return ExecutionResult(
                status="FAILED",
                execution_id=execution_id or uuid.uuid4().hex,
                failure_class=f"{type(exc).__name__}",
                detail=str(exc),
            )

        artifact_id = await self._storage.insert_effector_artifact(
            effector_id=effector_id,
            track_id=track_id,
            detection_id=detection_id,
            kind="snapshot",
            path=str(dest_path),
            created_ns=now_ns,
        )
        await self._broadcast_status(effector_id)
        return ExecutionResult(
            status="COMPLETED",
            execution_id=execution_id or uuid.uuid4().hex,
            result_refs=[artifact_id],
        )

    async def snapshot_live(self, effector_id: str) -> Path | None:
        """Grab a fresh frame for the snapshot-refresh <img> live view.

        Overwrites a single per-effector file rather than persisting an
        artifact record — this is a live-view poll, not an evidence capture
        (see ``capture()`` for the persisted/linked variant).
        """
        runtime = self._runtimes.get(effector_id)
        if runtime is None or runtime.driver is None or not hasattr(runtime.driver, "snapshot"):
            return None
        dest_path = self._config.snapshot_dir / f"{effector_id}-live.jpg"
        await runtime.driver.snapshot(dest_path=dest_path)  # type: ignore[attr-defined]
        return dest_path

    # ------------------------------------------------------------------

    async def _check_interlocks(
        self,
        effector_id: str,
        runtime: EffectorRuntime,
        target_pos: Vec3,
    ) -> str | None:
        """Pre-execution safety gate for camera movement commands."""
        safety = _safety_from_spec(runtime.spec)
        if safety.require_arm_for_slew:
            status = await self._status_for_runtime(runtime)
            if not status.armed:
                return "interlock:disarmed"

        if runtime.last_slew_ns is None:
            rate_limit_rejection = None
        else:
            min_interval_s = (
                safety.min_slew_interval_seconds
                if safety.min_slew_interval_seconds is not None
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

    def _cancel_home_return(self, runtime: EffectorRuntime) -> None:
        task = runtime.home_return_task
        runtime.home_return_task = None
        if task is not None:
            task.cancel()

    def _schedule_home_return(self, runtime: EffectorRuntime) -> None:
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
            name=f"effector-home-return-{runtime.spec.id}",
        )

    async def _home_return_after_dwell(self, runtime: EffectorRuntime) -> None:
        await asyncio.sleep(self._config.slew_dwell_seconds)
        driver = runtime.driver
        if driver is None:
            return
        try:
            await driver.go_home()  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("effector %s home return failed: %s", runtime.spec.id, exc)
            return
        runtime.active_track_id = None
        await self._broadcast_status(runtime.spec.id)

    async def _connect_and_register_runtime(self, spec: EffectorSpec, *, persist: bool) -> None:
        if persist:
            await self._storage.upsert_effector(spec, time.time_ns())

        runtime = EffectorRuntime(spec=spec)
        driver = _build_driver(spec, snapshot_dir=self._config.snapshot_dir)
        if driver is not None:
            try:
                await driver.connect()
                runtime.driver = driver
            except Exception as exc:
                runtime.connect_error = str(exc)
                logger.warning("effector %s failed to connect: %s", spec.id, exc)
        async with self._lock:
            self._runtimes[spec.id] = runtime
        await self._broadcast_status(spec.id)

    def _ensure_poll_task(self) -> None:
        if not self._runtimes:
            return
        if self._poll_task is not None and not self._poll_task.done():
            return
        self._poll_task = asyncio.create_task(self._poll_loop(), name="effector-status-poll")

    async def _poll_loop(self) -> None:
        while True:
            await asyncio.sleep(self._config.status_poll_interval_seconds)
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                # One bad driver/cycle must never kill the status loop.
                logger.warning("effector status poll iteration failed", exc_info=True)

    async def _poll_once(self) -> None:
        async with self._lock:
            runtimes = list(self._runtimes.values())
        for runtime in runtimes:
            if runtime.driver is None:
                await self._attempt_reconnect(runtime)
            await self._broadcast_status(runtime.spec.id)

    async def _attempt_reconnect(self, runtime: EffectorRuntime) -> None:
        driver = _build_driver(runtime.spec, snapshot_dir=self._config.snapshot_dir)
        if driver is None:
            return
        try:
            await driver.connect()
        except Exception as exc:
            runtime.connect_error = str(exc)
            logger.debug("effector %s reconnect attempt failed: %s", runtime.spec.id, exc)
            return
        runtime.driver = driver
        runtime.connect_error = None
        logger.info("effector %s connected after retry", runtime.spec.id)

    async def _status_for_runtime(self, runtime: EffectorRuntime) -> EffectorStatus:
        if runtime.driver is None:
            return EffectorStatus(effector_id=runtime.spec.id, state="offline", armed=False)
        try:
            raw = await runtime.driver.get_status()
        except Exception as exc:
            logger.debug("effector %s get_status failed: %s", runtime.spec.id, exc)
            return EffectorStatus(effector_id=runtime.spec.id, state="error", armed=False)
        return EffectorStatus(
            effector_id=runtime.spec.id,
            state=raw.get("state", "offline"),
            pan_deg=raw.get("pan_deg"),
            tilt_deg=raw.get("tilt_deg"),
            zoom=raw.get("zoom"),
            armed=bool(raw.get("armed", False)),
            last_seen_ns=time.time_ns(),
            active_track_id=raw.get("active_track_id") or runtime.active_track_id,
        )

    async def _broadcast_status(self, effector_id: str) -> None:
        if self._live_callback is None:
            return
        runtime = self._runtimes.get(effector_id)
        if runtime is None:
            payload = {"type": "effector_status", "effector_id": effector_id, "state": "deleted"}
        else:
            status = await self._status_for_runtime(runtime)
            payload = {"type": "effector_status", **status.model_dump(mode="json")}
        try:
            await self._live_callback(payload)
        except Exception:
            logger.debug("effector status broadcast failed for %s", effector_id, exc_info=True)


def _spec_from_row(row: dict[str, Any]) -> EffectorSpec:
    orientation_row = row.get("orientation") or {}
    return EffectorSpec(
        id=row["id"],
        effector_type=EffectorType(row["effector_type"]),
        position_m=tuple(row["position_m"]) if row.get("position_m") else None,
        position_geo=row.get("position_geo"),
        orientation=EffectorOrientation(
            yaw_deg=orientation_row.get("yaw_deg", 0.0),
            pitch_deg=orientation_row.get("pitch_deg", 0.0),
        ),
        capabilities=row.get("capabilities", []),
        transport=row.get("transport", {}),
        metadata=row.get("metadata", {}),
        properties=row.get("properties", {}),
    )


def _safety_from_spec(spec: EffectorSpec) -> EffectorSafetyConfig:
    return _safety_from_properties(spec.properties)


def _safety_from_properties(properties: dict[str, Any] | None) -> EffectorSafetyConfig:
    raw = properties.get("safety") if isinstance(properties, dict) else None
    if not isinstance(raw, dict):
        raw = {}
    return EffectorSafetyConfig.model_validate(raw)


def _build_driver(spec: EffectorSpec, *, snapshot_dir: Path) -> Effector | None:
    if spec.effector_type != EffectorType.CAMERA_PTZ:
        return None
    transport = spec.transport or {}
    host = transport.get("host")
    if not host:
        logger.warning("effector %s has no transport.host; driver not created", spec.id)
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
