"""Keep the Rust sidecar's site geometry equal to Python's.

One publisher per Python process that owns the managed sidecar (combined, or
the split-mode API process). It adopts snapshots built by
``sidecar_site_geometry.build_site_geometry_snapshot`` under a revision that is
monotonic for the life of the process, hands the adopted body to the sidecar at
launch (``MINIMAPPR_SITE_GEOMETRY_JSON``) and pushes it with
``POST /api/v1/dsp/geometry`` when an explicit trigger fires (node PATCH, node
delete, site-origin rebind) or a stationary estimate drifts past the threshold.
A reconcile tick compares the sidecar's applied revision with the adopted one
and re-pushes on mismatch (for example after a sidecar restart).

Every push runs on the publisher's own task and the HTTP call runs in a worker
thread with a bounded timeout, so neither request handlers nor the ingest path
ever wait on the sidecar.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from minimappr.core.geo import LocalCoordinateFrame
from minimappr.sidecar_json_http import SidecarRequestError, request_sidecar_json
from minimappr.sidecar_site_geometry import SiteGeometrySnapshot, build_site_geometry_snapshot

logger = logging.getLogger(__name__)

SIDECAR_SITE_GEOMETRY_PATH = "/api/v1/dsp/geometry"
SIDECAR_HEALTH_PATH = "/healthz"
APPLIED_REVISION_FIELD = "applied_geometry_push_revision"
# A geometry body is a few KB; anything slower than this means the sidecar is
# wedged, and the reconcile tick retries rather than holding a worker thread.
SIDECAR_GEOMETRY_REQUEST_TIMEOUT_SECONDS = 2.0
HTTP_CONFLICT = 409
STALE_REVISION_ERROR_CODE = "stale_geometry_push_revision"

PUSH_REASON_LAUNCH = "sidecar_launch"
PUSH_REASON_NODE_OVERRIDE_CHANGED = "node_override_changed"
PUSH_REASON_NODE_DELETED = "node_deleted"
PUSH_REASON_POSITION_ESTIMATOR_RESET = "position_estimator_reset"
PUSH_REASON_SITE_ORIGIN_REBOUND = "site_origin_rebound"
PUSH_REASON_ESTIMATE_MOVED = "stationary_estimate_moved"
PUSH_REASON_RECONCILE_MISMATCH = "reconcile_revision_mismatch"


class SiteGeometryStorage(Protocol):
    async def list_nodes(self, limit: int | None = None) -> list[dict]: ...

    async def list_node_position_estimator_states(self) -> dict[str, dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class AppliedSiteGeometry:
    """What the sidecar reports it is using; ``revision`` None means no push applied yet."""

    revision: int | None
    site_frame_id: str | None


class SiteGeometryTransport(Protocol):
    async def push_site_geometry(self, body: dict[str, object]) -> Mapping[str, Any]: ...

    async def fetch_applied_site_geometry(self) -> AppliedSiteGeometry | None:
        """None when the sidecar exposes no applied revision (an older build)."""
        ...


class HttpSiteGeometryTransport:
    """Talks to the sidecar over HTTP from a worker thread."""

    def __init__(self, sidecar_base_url: str, *, timeout_seconds: float = SIDECAR_GEOMETRY_REQUEST_TIMEOUT_SECONDS) -> None:
        self._sidecar_base_url = sidecar_base_url
        self._timeout_seconds = timeout_seconds

    async def push_site_geometry(self, body: dict[str, object]) -> Mapping[str, Any]:
        return await asyncio.to_thread(
            request_sidecar_json,
            self._sidecar_base_url,
            SIDECAR_SITE_GEOMETRY_PATH,
            body=body,
            timeout_seconds=self._timeout_seconds,
        )

    async def fetch_applied_site_geometry(self) -> AppliedSiteGeometry | None:
        try:
            reply = await self._get(SIDECAR_SITE_GEOMETRY_PATH)
        except SidecarRequestError as exc:
            if exc.status_code not in (404, 405):
                raise
            # The applied revision may live on the health payload instead.
            reply = await self._get(SIDECAR_HEALTH_PATH)
        if APPLIED_REVISION_FIELD not in reply:
            return None
        raw_revision = reply.get(APPLIED_REVISION_FIELD)
        # GET /api/v1/dsp/geometry nests the frame id in coordinate_frame
        # (null before any push); an acknowledgement-shaped reply has it top-level.
        coordinate_frame = reply.get("coordinate_frame")
        if isinstance(coordinate_frame, dict):
            raw_frame_id = coordinate_frame.get("frame_id")
        else:
            raw_frame_id = reply.get("site_frame_id")
        return AppliedSiteGeometry(
            revision=int(raw_revision) if raw_revision is not None else None,
            site_frame_id=str(raw_frame_id) if raw_frame_id is not None else None,
        )

    async def _get(self, path: str) -> dict:
        return await asyncio.to_thread(
            request_sidecar_json,
            self._sidecar_base_url,
            path,
            body=None,
            timeout_seconds=self._timeout_seconds,
        )


class SidecarSiteGeometryPublisher:
    def __init__(
        self,
        *,
        storage: SiteGeometryStorage,
        coordinate_frame_provider: Callable[[], LocalCoordinateFrame],
        transport: SiteGeometryTransport,
        sidecar_is_running: Callable[[], bool],
        estimate_move_threshold_m_provider: Callable[[], float],
        reconcile_interval_seconds_provider: Callable[[], float],
    ) -> None:
        self._storage = storage
        self._coordinate_frame_provider = coordinate_frame_provider
        self._transport = transport
        self._sidecar_is_running = sidecar_is_running
        # Providers, not values: both keys are live-patchable via PATCH /api/v1/config.
        self._estimate_move_threshold_m_provider = estimate_move_threshold_m_provider
        self._reconcile_interval_seconds_provider = reconcile_interval_seconds_provider
        self._adopted_snapshot: SiteGeometrySnapshot | None = None
        self._adopted_revision = 0
        self._sidecar_acknowledged_revision: int | None = None
        self._pending_push_reasons: set[str] = set()
        self._wake = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._task: asyncio.Task | None = None
        self._missing_applied_revision_logged = False

    @property
    def adopted_revision(self) -> int:
        return self._adopted_revision

    @property
    def sidecar_acknowledged_revision(self) -> int | None:
        return self._sidecar_acknowledged_revision

    def adopted_push_body(self) -> dict[str, object] | None:
        if self._adopted_snapshot is None:
            return None
        return self._adopted_snapshot.to_push_body(self._adopted_revision)

    async def build_snapshot(self) -> SiteGeometrySnapshot:
        node_rows = await self._storage.list_nodes()
        estimator_states = await self._storage.list_node_position_estimator_states()
        return build_site_geometry_snapshot(
            node_rows=node_rows,
            estimator_states_by_node_id=estimator_states,
            coordinate_frame=self._coordinate_frame_provider(),
        )

    def _adopt(self, snapshot: SiteGeometrySnapshot, *, estimate_drift_counts: bool) -> bool:
        """Adopt ``snapshot`` under a new revision if it changes the geometry."""
        if estimate_drift_counts:
            changed = snapshot != self._adopted_snapshot
        else:
            changed = snapshot.differs_beyond_estimate_drift(
                self._adopted_snapshot,
                estimate_move_threshold_m=self._estimate_move_threshold_m_provider(),
            )
        if changed:
            self._adopted_snapshot = snapshot
            self._adopted_revision += 1
        return changed

    async def launch_geometry_json(self) -> str | None:
        """Body for ``MINIMAPPR_SITE_GEOMETRY_JSON`` at (re)launch.

        A storage failure here must not stop the sidecar from starting, since
        audio continuity outranks geometry: the sidecar then runs on its
        automatic frame until the reconcile tick delivers a push.
        """
        try:
            self._adopt(await self.build_snapshot(), estimate_drift_counts=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Site geometry for sidecar launch could not be built; launching without it: %s", exc)
            return None
        body = self.adopted_push_body()
        logger.info(
            "Sidecar launch carries site geometry revision %d (%d node placements)",
            self._adopted_revision,
            len(body["node_placements"]) if body else 0,
        )
        return json.dumps(body, separators=(",", ":"), allow_nan=False)

    def request_push(self, reason: str) -> None:
        """Ask for a fresh snapshot to be pushed soon; never blocks the caller."""
        self._pending_push_reasons.add(reason)
        self._idle.clear()
        self._wake.set()

    async def wait_until_idle(self) -> None:
        """Resolve once every requested push has been attempted (tests, shutdown)."""
        await self._idle.wait()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="sidecar-site-geometry-publisher")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self._idle.set()

    async def _run(self) -> None:
        while True:
            woke_for_push = False
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=self._reconcile_interval_seconds_provider())
                woke_for_push = True
            self._wake.clear()
            try:
                if woke_for_push and self._pending_push_reasons:
                    reasons, self._pending_push_reasons = self._pending_push_reasons, set()
                    await self.push_current_geometry(reasons=reasons)
                else:
                    await self.reconcile_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("Site geometry publisher cycle failed; retrying next tick")
            finally:
                if not self._pending_push_reasons:
                    self._idle.set()

    async def push_current_geometry(self, *, reasons: set[str]) -> bool:
        """Explicit trigger: rebuild, adopt any change at all, and push."""
        self._adopt(await self.build_snapshot(), estimate_drift_counts=True)
        return await self._push_adopted(reason=",".join(sorted(reasons)))

    async def reconcile_once(self) -> None:
        """Push estimate drift past the threshold, else re-push on revision mismatch."""
        if self._adopt(await self.build_snapshot(), estimate_drift_counts=False):
            await self._push_adopted(reason=PUSH_REASON_ESTIMATE_MOVED)
            return
        if not self._sidecar_is_running() or self._adopted_snapshot is None:
            return
        try:
            applied = await self._transport.fetch_applied_site_geometry()
        except SidecarRequestError as exc:
            logger.warning("Site geometry reconcile: sidecar applied revision unavailable (%s); retrying next tick", exc)
            return
        if applied is None:
            if not self._missing_applied_revision_logged:
                self._missing_applied_revision_logged = True
                logger.warning(
                    "Site geometry reconcile: sidecar reports no %s; cannot verify revision %d, retrying each tick",
                    APPLIED_REVISION_FIELD,
                    self._adopted_revision,
                )
            return
        self._missing_applied_revision_logged = False
        self._sidecar_acknowledged_revision = applied.revision
        # A sidecar that reports no frame id is judged on the revision alone.
        frame_mismatch = (
            applied.site_frame_id is not None and applied.site_frame_id != self._adopted_snapshot.site_frame_id
        )
        if applied.revision != self._adopted_revision or frame_mismatch:
            logger.info(
                "Site geometry reconcile: sidecar applied revision %s (frame %s), Python adopted %d; re-pushing",
                applied.revision,
                applied.site_frame_id,
                self._adopted_revision,
            )
            await self._push_adopted(reason=PUSH_REASON_RECONCILE_MISMATCH)

    def _adopt_revision_above_stale_rejection(self, rejection: SidecarRequestError) -> bool:
        body = rejection.response_body or {}
        if rejection.status_code != HTTP_CONFLICT or body.get("error") != STALE_REVISION_ERROR_CODE:
            return False
        applied_revision = body.get(APPLIED_REVISION_FIELD)
        if not isinstance(applied_revision, int) or applied_revision < self._adopted_revision:
            return False
        logger.info(
            "Sidecar holds geometry revision %d above Python's %d (Python restarted under a running sidecar); "
            "re-pushing as revision %d",
            applied_revision,
            self._adopted_revision,
            applied_revision + 1,
        )
        self._adopted_revision = applied_revision + 1
        return True

    async def _push_adopted(self, *, reason: str) -> bool:
        body = self.adopted_push_body()
        if body is None:
            return False
        if not self._sidecar_is_running():
            # The next launch carries the adopted snapshot in its env instead.
            logger.debug("Site geometry revision %d not pushed (%s): sidecar not running", self._adopted_revision, reason)
            return False
        try:
            acknowledgement = await self._transport.push_site_geometry(body)
        except SidecarRequestError as exc:
            if self._adopt_revision_above_stale_rejection(exc):
                # The sidecar outlived an earlier Python process and holds a
                # higher revision; the same geometry goes again just above it.
                return await self._push_adopted(reason=f"{reason},{STALE_REVISION_ERROR_CODE}")
            logger.warning(
                "Site geometry push of revision %d (%s) failed: %s; reconcile will retry",
                self._adopted_revision,
                reason,
                exc,
            )
            return False
        applied_revision = acknowledgement.get(APPLIED_REVISION_FIELD)
        if applied_revision != self._adopted_revision or acknowledgement.get("site_frame_id") != body["site_frame_id"]:
            logger.warning(
                "Site geometry push of revision %d (%s) acknowledged as revision %r frame %r; reconcile will retry",
                self._adopted_revision,
                reason,
                applied_revision,
                acknowledgement.get("site_frame_id"),
            )
            return False
        self._sidecar_acknowledged_revision = self._adopted_revision
        logger.info(
            "Pushed site geometry revision %d to sidecar (%s, %d node placements)",
            self._adopted_revision,
            reason,
            len(body["node_placements"]),
        )
        return True

