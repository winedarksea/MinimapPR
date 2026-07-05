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
from minimappr.models import EffectorOrientation, EffectorSpec, EffectorStatus, EffectorType, Vec3

logger = logging.getLogger(__name__)

LiveCallback = Callable[[dict], Awaitable[None]]


@dataclass(slots=True)
class EffectorRuntime:
    spec: EffectorSpec
    driver: Effector | None = None
    connect_error: str | None = None
    last_slew_ns: int | None = None


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

    async def register(self, spec: EffectorSpec) -> EffectorSpec:
        await self._connect_and_register_runtime(spec, persist=True)
        self._ensure_poll_task()
        return spec

    async def delete(self, effector_id: str) -> bool:
        async with self._lock:
            self._runtimes.pop(effector_id, None)
        deleted = await self._storage.delete_effector(effector_id)
        await self._broadcast_status(effector_id)
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
        caps = await runtime.driver.get_capabilities()
        return {
            "movement_strategies": caps.movement_strategies,
            "selected_movement_strategy": caps.selected_movement_strategy,
            "snapshot_strategies": caps.snapshot_strategies,
            "selected_snapshot_strategy": caps.selected_snapshot_strategy,
            "has_zoom": caps.has_zoom,
        }

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

        rejection = self._check_interlocks(effector_id, runtime)
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

    def _check_interlocks(self, effector_id: str, runtime: EffectorRuntime) -> str | None:
        """Pre-execution safety gate. v1 implements a single check: per-effector
        minimum slew interval, preventing camera thrashing between tracks. This
        is the seam the Phase 5 interlock engine (master-arm, blue-force, zones)
        later plugs into without touching call sites."""
        del effector_id
        if runtime.last_slew_ns is None:
            return None
        min_interval_ns = int(self._config.min_slew_interval_seconds * 1_000_000_000)
        elapsed_ns = time.time_ns() - runtime.last_slew_ns
        if elapsed_ns < min_interval_ns:
            return "rate_limited"
        return None

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
            async with self._lock:
                effector_ids = list(self._runtimes.keys())
            for effector_id in effector_ids:
                await self._broadcast_status(effector_id)

    async def _status_for_runtime(self, runtime: EffectorRuntime) -> EffectorStatus:
        if runtime.driver is None:
            return EffectorStatus(effector_id=runtime.spec.id, state="offline", armed=False)
        raw = await runtime.driver.get_status()
        return EffectorStatus(
            effector_id=runtime.spec.id,
            state=raw.get("state", "offline"),
            pan_deg=raw.get("pan_deg"),
            tilt_deg=raw.get("tilt_deg"),
            zoom=raw.get("zoom"),
            armed=bool(raw.get("armed", False)),
            last_seen_ns=time.time_ns(),
            active_track_id=raw.get("active_track_id"),
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
