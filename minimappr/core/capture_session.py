"""Capture Session Manager.

Coordinates the full recording lifecycle:
  PENDING → RECORDING → PROCESSING → COMPLETED | FAILED

Responsibilities:
- Issues a StreamRangeLease to the Rust sidecar at session start.
- Spawns the ffmpeg video subprocess via VideoCapture.
- Sends periodic heartbeats to keep the Rust lease alive.
- On stop: enqueues background post-processing so the API returns immediately.
- On FAILED: actively sweeps the session working directory to remove large
  intermediate files to prevent disk exhaustion.

The `end_ns` hard cap is enforced by the Rust sidecar at lease creation time;
Python may not override it.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Optional

import httpx

if TYPE_CHECKING:
    from minimappr.core.audio_buffer import MultiSensorBuffer
    from minimappr.core.iamf_pipeline import IamfPipeline

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_S: float = 10.0
"""Send a heartbeat every 10 s.  Rust GCs leases after 30 s without one."""


class CaptureState(str, enum.Enum):
    PENDING = "pending"
    RECORDING = "recording"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CaptureSessionRecord:
    session_id: str
    state: CaptureState
    stream_key: str
    range_lease_id: Optional[str]
    start_time_ns: Optional[int]
    end_time_ns: Optional[int]
    first_frame_pts_ns: Optional[int]
    work_dir: Path
    video_path: Optional[Path]
    iamf_path: Optional[Path]
    youtube_path: Optional[Path]
    error: Optional[str]
    created_ns: int = field(default_factory=time.time_ns)
    use_python_ingest: bool = False
    channel_sensor_ids: list[str] = field(default_factory=list)


@dataclass
class CaptureStartRequest:
    stream_key: str
    work_dir: Path
    """Directory where intermediate and final files will be written."""
    sidecar_url: Optional[str] = None
    """Base URL of the Rust ingest sidecar. None = Python ingest mode."""
    multi_sensor_buffer: Optional["MultiSensorBuffer"] = None
    """Python ring buffer to pin when sidecar_url is None."""
    channel_sensor_ids: list[str] = field(default_factory=list)
    """Ordered sensor IDs for the 4-channel array (Python ingest mode)."""
    max_duration_s: float = 300.0
    video_source: Optional[str] = None
    libcamera_mode: bool = False
    deployment_profile: str = "auto"


PostProcessCallback = Callable[
    [CaptureSessionRecord], Coroutine[Any, Any, None]
]


class CaptureSessionManager:
    """Manages a set of concurrent capture sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, CaptureSessionRecord] = {}
        self._captures: dict[str, Any] = {}  # session_id → VideoCapture
        self._heartbeat_tasks: dict[str, asyncio.Task[None]] = {}
        self._post_process_callback: Optional[PostProcessCallback] = None
        self._http: Optional[httpx.AsyncClient] = None
        self._python_buffers: dict[str, "MultiSensorBuffer"] = {}  # session_id → buffer

    def set_post_process_callback(self, cb: PostProcessCallback) -> None:
        self._post_process_callback = cb

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=10.0)
        return self._http

    async def start(self, request: CaptureStartRequest) -> CaptureSessionRecord:
        """Create a new session, pin the journal range, start video capture."""
        from minimappr.core.video_capture import VideoCapture, VideoCaptureConfig

        session_id = str(uuid.uuid4())
        work_dir = request.work_dir / session_id
        work_dir.mkdir(parents=True, exist_ok=True)

        now_ns = time.time_ns()
        record = CaptureSessionRecord(
            session_id=session_id,
            state=CaptureState.PENDING,
            stream_key=request.stream_key,
            range_lease_id=None,
            start_time_ns=None,
            end_time_ns=None,
            first_frame_pts_ns=None,
            work_dir=work_dir,
            video_path=None,
            iamf_path=None,
            youtube_path=None,
            error=None,
            created_ns=now_ns,
        )
        self._sessions[session_id] = record

        if request.sidecar_url is None:
            # Python ingest mode: pin the in-memory ring buffer instead of a Rust range lease.
            if request.multi_sensor_buffer is None:
                record.state = CaptureState.FAILED
                record.error = "Python ingest mode requires multi_sensor_buffer"
                logger.error("session %s: %s", session_id, record.error)
                return record
            request.multi_sensor_buffer.pin(session_id, now_ns)
            self._python_buffers[session_id] = request.multi_sensor_buffer
            record.start_time_ns = now_ns
            record.use_python_ingest = True
            record.channel_sensor_ids = list(request.channel_sensor_ids)
            logger.info(
                "session %s started (python ingest): stream=%s channels=%s",
                session_id,
                record.stream_key,
                record.channel_sensor_ids,
            )
        else:
            # Rust sidecar mode: issue a StreamRangeLease.
            try:
                lease_resp = await self._create_range_lease(
                    request.sidecar_url,
                    request.stream_key,
                    now_ns,
                    max_duration_s=request.max_duration_s,
                )
                record.range_lease_id = lease_resp["lease_id"]
                record.start_time_ns = lease_resp["start_ns"]
            except Exception as exc:
                record.state = CaptureState.FAILED
                record.error = f"range lease creation failed: {exc}"
                logger.error("session %s: %s", session_id, record.error)
                return record

            # Begin heartbeat loop (Rust sidecar only).
            self._heartbeat_tasks[session_id] = asyncio.create_task(
                self._heartbeat_loop(session_id, request.sidecar_url),
                name=f"capture_heartbeat_{session_id[:8]}",
            )
            logger.info(
                "session %s started: lease=%s stream=%s",
                session_id,
                record.range_lease_id,
                record.stream_key,
            )

        # Start video capture (both modes).
        video_path = work_dir / "output_raw.mp4"
        cap_cfg = VideoCaptureConfig(
            output_path=video_path,
            video_source=request.video_source,
            libcamera_mode=request.libcamera_mode,
        )
        cap = VideoCapture(cap_cfg)
        try:
            await cap.start()
        except Exception as exc:
            record.state = CaptureState.FAILED
            record.error = f"video capture start failed: {exc}"
            sidecar = request.sidecar_url or ""
            await self._release_range_lease(sidecar, record)
            return record

        record.video_path = video_path
        record.state = CaptureState.RECORDING
        self._captures[session_id] = cap

        return record

    async def stop(self, session_id: str, sidecar_url: str = "") -> CaptureSessionRecord:
        """Stop recording and enqueue background post-processing."""
        record = self._sessions.get(session_id)
        if record is None:
            raise KeyError(f"session {session_id} not found")
        if record.state != CaptureState.RECORDING:
            raise ValueError(f"session {session_id} is not recording (state={record.state})")

        # Cancel heartbeat task (Rust path only; Python mode never creates one).
        ht = self._heartbeat_tasks.pop(session_id, None)
        if ht is not None:
            ht.cancel()

        # Stop video capture.
        cap = self._captures.pop(session_id, None)
        if cap is not None:
            try:
                result = await cap.stop()
                record.first_frame_pts_ns = result.first_frame_pts_ns
            except Exception as exc:
                logger.warning("session %s: video stop error: %s", session_id, exc)

        record.end_time_ns = time.time_ns()
        record.state = CaptureState.PROCESSING

        # Enqueue post-processing in the background.
        effective_sidecar_url = "" if record.use_python_ingest else sidecar_url
        asyncio.create_task(
            self._run_post_processing(session_id, effective_sidecar_url),
            name=f"capture_postprocess_{session_id[:8]}",
        )

        logger.info("session %s stopped, queued for post-processing", session_id)
        return record

    def get(self, session_id: str) -> Optional[CaptureSessionRecord]:
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[CaptureSessionRecord]:
        return list(self._sessions.values())

    # ── private ────────────────────────────────────────────────────────────────

    async def _heartbeat_loop(self, session_id: str, sidecar_url: str) -> None:
        record = self._sessions[session_id]
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)
            if record.range_lease_id is None:
                break
            try:
                await self._client().post(
                    f"{sidecar_url}/api/v1/capture/range-lease"
                    f"/{record.range_lease_id}/heartbeat"
                )
            except Exception as exc:
                logger.warning(
                    "session %s: heartbeat failed: %s", session_id, exc
                )

    async def _run_post_processing(self, session_id: str, sidecar_url: str) -> None:
        record = self._sessions[session_id]
        try:
            if self._post_process_callback is not None:
                await self._post_process_callback(record)
                record.state = CaptureState.COMPLETED
            else:
                logger.warning(
                    "session %s: no post-process callback registered; "
                    "transitioning to COMPLETED without rendering",
                    session_id,
                )
                record.state = CaptureState.COMPLETED
        except Exception as exc:
            record.state = CaptureState.FAILED
            record.error = str(exc)
            logger.error("session %s post-processing failed: %s", session_id, exc, exc_info=True)
            await self._cleanup_on_failure(record)
        finally:
            if record.use_python_ingest:
                buf = self._python_buffers.pop(session_id, None)
                if buf is not None:
                    buf.unpin(session_id)
            else:
                await self._release_range_lease(sidecar_url, record)

    async def _cleanup_on_failure(self, record: CaptureSessionRecord) -> None:
        """Remove large intermediate files to prevent disk exhaustion."""
        import shutil

        patterns = ["bed_full.wav", "bed.wav", "object_*.wav", "ambix.wav"]
        for pattern in patterns:
            for f in record.work_dir.glob(pattern):
                try:
                    f.unlink(missing_ok=True)
                    logger.info("cleanup: removed %s", f)
                except OSError as exc:
                    logger.warning("cleanup failed for %s: %s", f, exc)

    async def _create_range_lease(
        self,
        sidecar_url: str,
        stream_key: str,
        start_ns: int,
        max_duration_s: float,
    ) -> dict:
        requested_end_ns = start_ns + int(max_duration_s * 1_000_000_000)
        resp = await self._client().post(
            f"{sidecar_url}/api/v1/capture/range-lease",
            json={
                "owner": "python-capture-session",
                "stream_key": stream_key,
                "start_ns": start_ns,
                "requested_end_ns": requested_end_ns,
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def _release_range_lease(
        self, sidecar_url: str, record: CaptureSessionRecord
    ) -> None:
        if record.range_lease_id is None:
            return
        try:
            await self._client().delete(
                f"{sidecar_url}/api/v1/capture/range-lease/{record.range_lease_id}"
            )
        except Exception as exc:
            logger.warning(
                "session %s: lease release failed (not critical): %s",
                record.session_id,
                exc,
            )
        finally:
            record.range_lease_id = None
