"""Platform-aware camera/video device enumeration.

Discovers available video capture devices for the recording UI dropdown:
  - macOS: parses ffmpeg's AVFoundation device listing
  - Linux / Pi: scans /dev/video* and optionally queries v4l2-ctl / libcamera
"""

from __future__ import annotations

import asyncio
import logging
import platform
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CameraInfo:
    """A single discoverable video capture device."""
    id: str
    """Device identifier to pass to ffmpeg `-i` (e.g. "0" on macOS, "/dev/video0" on Linux)."""
    label: str
    """Human-readable name (e.g. "FaceTime HD Camera")."""
    platform: str
    """Platform backend: "avfoundation", "v4l2", or "libcamera"."""


async def discover_cameras() -> list[CameraInfo]:
    """Return available video capture devices for the current platform."""
    system = platform.system()
    if system == "Darwin":
        return await _discover_avfoundation()
    if system == "Linux":
        return await _discover_linux()
    # Windows / unknown — return empty rather than crash.
    return []


async def _discover_avfoundation() -> list[CameraInfo]:
    """Enumerate AVFoundation video devices via ffmpeg on macOS."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", "",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
    except FileNotFoundError:
        logger.warning("ffmpeg not found; cannot enumerate AVFoundation cameras")
        return []

    text = stderr.decode("utf-8", errors="replace")
    devices: list[CameraInfo] = []
    # Parse lines like: [AVFoundation input device @ 0x...] [0] FaceTime HD Camera
    video_section = True
    for line in text.splitlines():
        if "AVFoundation audio devices" in line:
            video_section = False
            continue
        if not video_section:
            continue
        match = re.search(r"\[(\d+)\]\s+(.+)", line)
        if match:
            idx = match.group(1)
            label = match.group(2).strip()
            devices.append(CameraInfo(id=idx, label=label, platform="avfoundation"))

    return devices


async def _discover_linux() -> list[CameraInfo]:
    """Enumerate V4L2 and libcamera devices on Linux / Raspberry Pi."""
    import os

    devices: list[CameraInfo] = []

    # Scan /dev/video* entries.
    dev_dir = "/dev"
    if os.path.isdir(dev_dir):
        for entry in sorted(os.listdir(dev_dir)):
            if entry.startswith("video"):
                dev_path = os.path.join(dev_dir, entry)
                devices.append(CameraInfo(
                    id=dev_path,
                    label=_v4l2_device_label(dev_path) or entry,
                    platform="v4l2",
                ))

    # Also check libcamera devices on Pi.
    libcamera_devices = await _discover_libcamera()
    devices.extend(libcamera_devices)

    return devices


def _v4l2_device_label(dev_path: str) -> str | None:
    """Try to get a friendly name from v4l2-ctl (best-effort, non-blocking)."""
    try:
        import subprocess
        result = subprocess.run(
            ["v4l2-ctl", "--device", dev_path, "--info"],
            capture_output=True, text=True, timeout=2,
        )
        for line in result.stdout.splitlines():
            if "Card type" in line:
                return line.split(":", 1)[-1].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


async def _discover_libcamera() -> list[CameraInfo]:
    """Check for libcamera devices (Raspberry Pi Bookworm+)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "libcamera-hello", "--list-cameras",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
    except (FileNotFoundError, asyncio.TimeoutError, OSError):
        return []

    devices: list[CameraInfo] = []
    text = stdout.decode("utf-8", errors="replace")
    # Parse lines like: 0 : imx477 [4056x3040] (/base/soc/i2c0mux/i2c@1/imx477@1a)
    for line in text.splitlines():
        match = re.match(r"\s*(\d+)\s*:\s*(.+?)(?:\s+\[.*?\])?\s+\((.+?)\)", line)
        if match:
            idx = match.group(1)
            label = match.group(2).strip()
            devices.append(CameraInfo(id=f"libcamera:{idx}", label=label, platform="libcamera"))

    return devices
