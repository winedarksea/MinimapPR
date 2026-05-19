"""Deterministic COP-like recording visual renderer.

The live COP uses Leaflet in the browser. Recording exports need repeatable
post-processing, so this module renders directly from finalized IAMF object
slot metadata and writes raw RGB frames to ffmpeg.
"""

from __future__ import annotations

import asyncio
import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from minimappr.core.recording_visual_font import FONT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CopVisualSpec:
    background: tuple[int, int, int] = (15, 23, 42)
    grid_major: tuple[int, int, int] = (51, 65, 85)
    grid_minor: tuple[int, int, int] = (30, 41, 59)
    track: tuple[int, int, int] = (95, 214, 196)
    track_trail: tuple[int, int, int] = (45, 212, 191)
    track_highlight: tuple[int, int, int] = (226, 232, 240)
    node: tuple[int, int, int] = (88, 166, 255)
    text: tuple[int, int, int] = (226, 232, 240)
    text_dim: tuple[int, int, int] = (148, 163, 184)
    surface: tuple[int, int, int] = (22, 27, 34)
    warn: tuple[int, int, int] = (210, 153, 34)


COP_VISUAL_SPEC = CopVisualSpec()


class VisualTrajectory(Protocol):
    track_id: str
    label: str
    waypoints: list[tuple[int, tuple[float, float, float]]]


class VisualObjectSlot(Protocol):
    unit_track_ids: list[str | None]
    active_ranges: list[tuple[int, int]]


@dataclass(frozen=True)
class RecordingVisualFrame:
    frame_index: int
    sample_offset: int
    time_seconds: float
    track_id: str | None
    label: str | None
    position_m: tuple[float, float, float] | None


def build_recording_visual_timeline(
    slot: VisualObjectSlot | None,
    trajectories: list[VisualTrajectory],
    *,
    n_samples: int,
    sample_rate_hz: int,
    samples_per_unit: int,
    frame_rate_hz: int = 30,
) -> list[RecordingVisualFrame]:
    """Convert selected object-slot ownership into fixed-rate visual frames."""
    if slot is None or n_samples <= 0 or sample_rate_hz <= 0:
        return []
    frame_count = max(1, math.ceil(n_samples * frame_rate_hz / sample_rate_hz))
    trajectories_by_id = {traj.track_id: traj for traj in trajectories}
    frames: list[RecordingVisualFrame] = []
    for frame_index in range(frame_count):
        sample_offset = min(
            n_samples - 1,
            int(round(frame_index * sample_rate_hz / frame_rate_hz)),
        )
        unit_index = min(
            len(slot.unit_track_ids) - 1,
            max(0, sample_offset // max(1, samples_per_unit)),
        )
        track_id = slot.unit_track_ids[unit_index] if slot.unit_track_ids else None
        trajectory = trajectories_by_id.get(track_id or "")
        position = (
            _interpolate_waypoints(trajectory.waypoints, sample_offset)
            if trajectory is not None
            else None
        )
        label = _visual_label_for_track(trajectory) if trajectory is not None else None
        frames.append(
            RecordingVisualFrame(
                frame_index=frame_index,
                sample_offset=sample_offset,
                time_seconds=sample_offset / sample_rate_hz,
                track_id=track_id,
                label=label,
                position_m=position,
            )
        )
    return frames


async def render_recording_visual_mp4(
    output_path: Path,
    slot: VisualObjectSlot | None,
    trajectories: list[VisualTrajectory],
    *,
    n_samples: int,
    sample_rate_hz: int,
    samples_per_unit: int,
    frame_rate_hz: int = 30,
    width: int = 1920,
    height: int = 1080,
) -> bool:
    """Render a COP-like visual MP4. Returns False when no visual is possible."""
    frames = build_recording_visual_timeline(
        slot,
        trajectories,
        n_samples=n_samples,
        sample_rate_hz=sample_rate_hz,
        samples_per_unit=samples_per_unit,
        frame_rate_hz=frame_rate_hz,
    )
    if not frames:
        return False
    if shutil.which("ffmpeg") is None:
        logger.warning("recording visual skipped: ffmpeg not found")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(frame_rate_hz),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdin is not None
    bounds = _compute_bounds(trajectories)
    base_image = _render_base(bounds, width, height)
    try:
        for frame in frames:
            image = _render_frame(frame, trajectories, bounds, base_image)
            proc.stdin.write(image.tobytes())
            await proc.stdin.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        proc.stdin.close()

    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        output_path.unlink(missing_ok=True)
        logger.warning(
            "recording visual ffmpeg failed (rc=%d): %s%s",
            proc.returncode,
            stdout.decode(errors="replace")[-200:],
            stderr.decode(errors="replace")[-500:],
        )
        return False
    return output_path.exists() and output_path.stat().st_size > 0


def _render_frame(
    frame: RecordingVisualFrame,
    trajectories: list[VisualTrajectory],
    bounds: tuple[float, float, float, float],
    base_image: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    spec = COP_VISUAL_SPEC
    image = base_image.copy()
    height, width, _ = image.shape

    for trajectory in trajectories:
        points = [
            _project_xy(pos[1], bounds, width, height)
            for pos in trajectory.waypoints
            if pos[0] <= frame.sample_offset
        ]
        _draw_polyline(image, points, spec.track_trail)

    if frame.position_m is not None:
        x, y = _project_xy(frame.position_m, bounds, width, height)
        _draw_circle(image, x, y, 34, spec.track_highlight, fill=False, thickness=3)
        _draw_diamond(image, x, y, 24, spec.track, spec.surface)
        title = frame.label or frame.track_id or "TRACK"
        _draw_label_panel(image, 48, 42, title.upper()[:28], spec)
        _draw_text(image, 50, 110, f"T+{frame.time_seconds:05.1f}S", spec.text_dim, scale=3)
    else:
        _draw_label_panel(image, 48, 42, "NO ACTIVE OBJECT", spec)
        _draw_text(image, 50, 110, f"T+{frame.time_seconds:05.1f}S", spec.text_dim, scale=3)
    return image


def _render_base(
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
) -> NDArray[np.uint8]:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = COP_VISUAL_SPEC.background
    _draw_grid(image, bounds)
    _draw_node_origin(image, bounds)
    return image


def _compute_bounds(
    trajectories: list[VisualTrajectory],
) -> tuple[float, float, float, float]:
    points = [
        (float(pos[0]), float(pos[1]))
        for trajectory in trajectories
        for _, pos in trajectory.waypoints
    ]
    points.append((0.0, 0.0))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y, 5.0)
    pad = span * 0.25
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    half = 0.5 * span + pad
    return cx - half, cx + half, cy - half, cy + half


def _project_xy(
    position_m: tuple[float, float, float],
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int]:
    min_x, max_x, min_y, max_y = bounds
    margin_x = 120
    margin_y = 130
    x_norm = (float(position_m[0]) - min_x) / max(max_x - min_x, 1e-6)
    y_norm = (float(position_m[1]) - min_y) / max(max_y - min_y, 1e-6)
    x = margin_x + x_norm * (width - 2 * margin_x)
    y = height - margin_y - y_norm * (height - 2 * margin_y)
    return int(round(x)), int(round(y))


def _draw_grid(image: NDArray[np.uint8], bounds: tuple[float, float, float, float]) -> None:
    height, width, _ = image.shape
    min_x, max_x, min_y, max_y = bounds
    for value in _grid_values(min_x, max_x):
        x, _ = _project_xy((value, min_y, 0.0), bounds, width, height)
        color = COP_VISUAL_SPEC.grid_major if abs(value) < 1e-6 else COP_VISUAL_SPEC.grid_minor
        _draw_line(image, x, 0, x, height - 1, color, 1)
    for value in _grid_values(min_y, max_y):
        _, y = _project_xy((min_x, value, 0.0), bounds, width, height)
        color = COP_VISUAL_SPEC.grid_major if abs(value) < 1e-6 else COP_VISUAL_SPEC.grid_minor
        _draw_line(image, 0, y, width - 1, y, color, 1)


def _grid_values(min_value: float, max_value: float) -> list[float]:
    span = max_value - min_value
    step = 1.0
    while span / step > 10:
        step *= 2.0
    start = math.floor(min_value / step) * step
    values = []
    value = start
    while value <= max_value:
        values.append(value)
        value += step
    return values


def _draw_node_origin(image: NDArray[np.uint8], bounds: tuple[float, float, float, float]) -> None:
    height, width, _ = image.shape
    x, y = _project_xy((0.0, 0.0, 0.0), bounds, width, height)
    _draw_circle(image, x, y, 16, COP_VISUAL_SPEC.node, fill=True)
    _draw_circle(image, x, y, 24, COP_VISUAL_SPEC.node, fill=False, thickness=2)


def _draw_label_panel(
    image: NDArray[np.uint8],
    x: int,
    y: int,
    text: str,
    spec: CopVisualSpec,
) -> None:
    height = 52
    width = min(image.shape[1] - x - 48, max(360, 26 * len(text)))
    image[y : y + height, x : x + width] = spec.surface
    _draw_rect_outline(image, x, y, width, height, spec.track, 2)
    _draw_text(image, x + 18, y + 14, text, spec.text, scale=4)


def _draw_polyline(image: NDArray[np.uint8], points: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    if len(points) < 2:
        return
    for start, end in zip(points, points[1:]):
        _draw_line(image, start[0], start[1], end[0], end[1], color, 3)


def _draw_diamond(
    image: NDArray[np.uint8],
    x: int,
    y: int,
    radius: int,
    stroke: tuple[int, int, int],
    fill: tuple[int, int, int],
) -> None:
    points = [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)]
    min_y = max(0, y - radius)
    max_y = min(image.shape[0] - 1, y + radius)
    for py in range(min_y, max_y + 1):
        span = radius - abs(py - y)
        _draw_line(image, x - span, py, x + span, py, fill, 1)
    for p0, p1 in zip(points, points[1:] + points[:1]):
        _draw_line(image, p0[0], p0[1], p1[0], p1[1], stroke, 3)


def _draw_circle(
    image: NDArray[np.uint8],
    cx: int,
    cy: int,
    radius: int,
    color: tuple[int, int, int],
    *,
    fill: bool,
    thickness: int = 1,
) -> None:
    h, w, _ = image.shape
    r2 = radius * radius
    inner = max(0, radius - thickness)
    inner2 = inner * inner
    for y in range(max(0, cy - radius), min(h, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(w, cx + radius + 1)):
            d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
            if d2 <= r2 and (fill or d2 >= inner2):
                image[y, x] = color


def _draw_line(
    image: NDArray[np.uint8],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        _plot_thick(image, x0, y0, color, thickness)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _plot_thick(
    image: NDArray[np.uint8],
    x: int,
    y: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    radius = max(0, thickness // 2)
    h, w, _ = image.shape
    for py in range(max(0, y - radius), min(h, y + radius + 1)):
        for px in range(max(0, x - radius), min(w, x + radius + 1)):
            image[py, px] = color


def _draw_rect_outline(
    image: NDArray[np.uint8],
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    _draw_line(image, x, y, x + width, y, color, thickness)
    _draw_line(image, x, y + height, x + width, y + height, color, thickness)
    _draw_line(image, x, y, x, y + height, color, thickness)
    _draw_line(image, x + width, y, x + width, y + height, color, thickness)


def _draw_text(
    image: NDArray[np.uint8],
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int],
    *,
    scale: int,
) -> None:
    cursor = x
    for char in text.upper():
        glyph = FONT.get(char, FONT["?"])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    x0 = cursor + gx * scale
                    y0 = y + gy * scale
                    image[y0 : y0 + scale, x0 : x0 + scale] = color
        cursor += 6 * scale


def _interpolate_waypoints(
    waypoints: list[tuple[int, tuple[float, float, float]]],
    sample: int,
) -> tuple[float, float, float]:
    if not waypoints:
        return (0.0, 0.0, 0.0)
    if sample <= waypoints[0][0]:
        return waypoints[0][1]
    if sample >= waypoints[-1][0]:
        return waypoints[-1][1]
    for (s0, p0), (s1, p1) in zip(waypoints, waypoints[1:]):
        if s0 <= sample <= s1:
            alpha = (sample - s0) / max(1, s1 - s0)
            return (
                p0[0] + (p1[0] - p0[0]) * alpha,
                p0[1] + (p1[1] - p0[1]) * alpha,
                p0[2] + (p1[2] - p0[2]) * alpha,
            )
    return waypoints[-1][1]


def _visual_label_for_track(trajectory: VisualTrajectory | None) -> str:
    if trajectory is None:
        return "TRACK"
    label = getattr(trajectory, "label", "") or ""
    if label.strip():
        return label.strip()
    return trajectory.track_id[:12]
