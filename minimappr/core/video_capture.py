"""Headless ffmpeg video capture subprocess manager.

Handles platform-specific hardware codec selection, process lifecycle, and
first-frame timestamp extraction so the post-processor can align audio exactly
to the first actual video frame (avoiding 50–500 ms driver init latency).

Platform codec priority:
    Linux / Pi   → h264_v4l2m2m (V4L2 M2M kernel encoder)
    macOS        → h264_videotoolbox (VideoToolbox)
    fallback     → libx264 -preset veryfast -crf 18
"""

from __future__ import annotations

import asyncio
import logging
import platform
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Platform codec detection ──────────────────────────────────────────────────

def _default_codec() -> list[str]:
    system = platform.system()
    if system == "Linux":
        return ["-c:v", "h264_v4l2m2m"]
    if system == "Darwin":
        return ["-c:v", "h264_videotoolbox"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]


def _libcamera_source_args() -> list[str]:
    """On Pi (Linux with libcamera), route through the libcamera-vid pipe."""
    if platform.system() == "Linux":
        return []  # populated by build_command when libcamera_mode=True
    return []


# ── Public API ────────────────────────────────────────────────────────────────

@dataclass
class VideoCaptureConfig:
    output_path: Path
    """Where to write the raw (no audio) MP4."""
    width: int = 1920
    height: int = 1080
    fps: int = 30
    codec_args: list[str] = field(default_factory=_default_codec)
    """Override codec.  Defaults to platform-appropriate hardware encoder."""
    video_source: str | None = None
    """ffmpeg input source string.  None = autodetect."""
    libcamera_mode: bool = False
    """Use libcamera-vid | ffmpeg pipe on Raspberry Pi Bookworm."""
    stats_period_s: float = 0.1
    """ffmpeg -stats_period for pts_time extraction."""


@dataclass
class VideoCaptureResult:
    """Returned when capture stops successfully."""
    output_path: Path
    first_frame_pts_ns: int
    """Nanosecond host-clock timestamp of the first actual video frame."""
    process_start_ns: int
    """Wall-clock ns when the ffmpeg process was spawned (for diagnostics)."""


class VideoCaptureError(RuntimeError):
    pass


class VideoCapture:
    """Manages a single headless video capture session.

    Usage::

        cap = VideoCapture(config)
        await cap.start()
        # ... recording ...
        result = await cap.stop()
    """

    def __init__(self, config: VideoCaptureConfig) -> None:
        self._config = config
        self._process: Optional[asyncio.subprocess.Process] = None
        self._first_frame_pts_ns: Optional[int] = None
        self._process_start_ns: int = 0
        self._pts_event = asyncio.Event()
        self._reader_task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        """Spawn ffmpeg and wait until the first frame timestamp is available."""
        if self._process is not None:
            raise VideoCaptureError("capture already running")

        cmd = self._build_command()
        logger.info("starting video capture: %s", " ".join(cmd))

        self._process_start_ns = time.time_ns()
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader_task = asyncio.create_task(
            self._parse_stderr(self._process),
            name="video_capture_stderr_reader",
        )

        # Wait up to 5 s for the first frame pts_time before returning.
        try:
            await asyncio.wait_for(self._pts_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(
                "video capture: first pts_time not received within 5 s; "
                "using process-spawn time as fallback"
            )
            self._first_frame_pts_ns = self._process_start_ns

    async def stop(self) -> VideoCaptureResult:
        """Send SIGTERM to cleanly finalise the MP4 moov atom and wait."""
        if self._process is None:
            raise VideoCaptureError("capture is not running")

        proc = self._process
        logger.info("stopping video capture (SIGTERM)")
        try:
            proc.terminate()
        except ProcessLookupError:
            pass

        try:
            await asyncio.wait_for(proc.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            logger.warning("ffmpeg did not exit after SIGTERM; sending SIGKILL")
            proc.kill()
            await proc.wait()

        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass

        self._process = None

        pts_ns = self._first_frame_pts_ns or self._process_start_ns
        return VideoCaptureResult(
            output_path=self._config.output_path,
            first_frame_pts_ns=pts_ns,
            process_start_ns=self._process_start_ns,
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _build_command(self) -> list[str]:
        cfg = self._config

        if cfg.libcamera_mode and platform.system() == "Linux":
            # libcamera-vid | ffmpeg pipe pattern for Pi Bookworm.
            return [
                "bash", "-c",
                f"libcamera-vid -o - --codec h264 --width {cfg.width} "
                f"--height {cfg.height} --framerate {cfg.fps} -t 0 | "
                f"ffmpeg -f h264 -i pipe:0 "
                + " ".join(cfg.codec_args)
                + f" -progress pipe:2 -stats_period {cfg.stats_period_s} "
                f"-y {cfg.output_path}",
            ]

        # Standard ffmpeg path.
        source = cfg.video_source or _autodetect_source()
        cmd = ["ffmpeg"]

        # Input.
        cmd += _source_input_args(source, cfg)

        # Video size / rate.
        cmd += ["-video_size", f"{cfg.width}x{cfg.height}", "-framerate", str(cfg.fps)]

        # Codec.
        cmd += cfg.codec_args

        # Progress reporting to stderr so we can parse pts_time.
        cmd += ["-progress", "pipe:2", "-stats_period", str(cfg.stats_period_s)]

        # Output (overwrite without prompt).
        cmd += ["-y", str(cfg.output_path)]

        return cmd

    async def _parse_stderr(self, proc: asyncio.subprocess.Process) -> None:
        """Read ffmpeg's stderr progress stream and extract the first pts_time."""
        assert proc.stderr is not None
        # Pattern: pts_time=1.234567
        pts_pattern = re.compile(rb"pts_time=(\d+\.?\d*)")

        while True:
            try:
                line = await proc.stderr.readline()
            except Exception:
                break
            if not line:
                break

            if not self._pts_event.is_set():
                match = pts_pattern.search(line)
                if match:
                    pts_s = float(match.group(1))
                    # Derive the wall-clock time of PTS=0 (first frame) by
                    # subtracting pts_s from the current wall clock at parse
                    # time.  This corrects for camera init latency (50–500 ms)
                    # that makes process_start_ns an inaccurate video origin.
                    # process_start_ns + pts_s would anchor to spawn time, but
                    # the camera may not have delivered its first frame until
                    # well after spawn, introducing a constant AV drift.
                    self._first_frame_pts_ns = time.time_ns() - int(
                        pts_s * 1_000_000_000
                    )
                    self._pts_event.set()
                    logger.debug(
                        "video first frame pts_time=%.6f s → %d ns",
                        pts_s,
                        self._first_frame_pts_ns,
                    )


def _autodetect_source() -> str:
    system = platform.system()
    if system == "Darwin":
        # Return the device index, not the format name.  _source_input_args
        # already supplies "-f avfoundation", so the source string must be
        # the camera index (e.g. "0") that ffmpeg expects after "-i".
        return "0"
    if system == "Linux":
        return "/dev/video0"
    return "0"


def _source_input_args(source: str, cfg: VideoCaptureConfig) -> list[str]:
    system = platform.system()
    if system == "Darwin":
        return ["-f", "avfoundation", "-i", f"{source}:none"]
    if system == "Linux":
        return ["-f", "v4l2", "-i", source]
    return ["-f", "dshow", "-i", f"video={source}"]
