"""Per-frame packet timing carried by calibration bundles.

A bundle's ``audio/{node_id}.wav`` is one continuous sample timeline, which
hides everything the packets said about time: a timestamp step or clock-class
change inside one upload, sequence gaps, the exact packet timestamp of every
frame. ``frame_timing/{node_id}.json`` keeps that, one entry per captured
firmware frame, so replay (``minimappr/sim/replay_frames.py``) can rebuild the
uploads the node actually sent.

File layout (``schema_version`` 1)::

    {"schema_version": 1, "node_id": "...", "sample_rate_hz": 48000,
     "frames": [{"sample_offset": 0, "sample_count": 960,
                 "start_time_ns": 1764000000000000000,
                 "time_quality": "gps_locked", "sequence": 41}, ...]}

``sample_offset`` indexes the node's WAV (sample 0 = the manifest's
``audio_start_time_ns``); ``time_quality`` is a ``TimeQuality`` value or null
when the source did not report one; ``sequence`` may be null. Frames are
sorted by ``sample_offset`` and never overlap; gaps (lost packets) are allowed.

Bundles written before this table existed have no ``frame_timing_file`` on
their node entries; ``uniform_frame_timing_table`` describes such a node's
audio as evenly cut frames on the flat ``audio_start_time_ns`` timeline, which
is exactly what the WAV alone can say.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from minimappr.core.capture_audio_buffer import CapturedFrameTiming
from minimappr.models import TimeQuality

FRAME_TIMING_SCHEMA_VERSION = 1
FRAME_TIMING_BUNDLE_DIRECTORY = "frame_timing"
NANOSECONDS_PER_SECOND = 1_000_000_000
_KNOWN_TIME_QUALITY_VALUES = frozenset(quality.value for quality in TimeQuality)


def frame_timing_bundle_path(node_id: str) -> str:
    return f"{FRAME_TIMING_BUNDLE_DIRECTORY}/{node_id}.json"


class FrameTimingTableError(ValueError):
    """A frame timing table is malformed (unsorted, overlapping, unknown clock class)."""


@dataclass(frozen=True, slots=True)
class NodeFrameTimingTable:
    node_id: str
    sample_rate_hz: int
    frames: tuple[CapturedFrameTiming, ...]

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise FrameTimingTableError(f"{self.node_id}: sample_rate_hz must be positive")
        previous_end_offset: int | None = None
        for frame in self.frames:
            if frame.sample_count <= 0:
                raise FrameTimingTableError(f"{self.node_id}: frame at {frame.sample_offset} has no samples")
            if previous_end_offset is not None and frame.sample_offset < previous_end_offset:
                raise FrameTimingTableError(
                    f"{self.node_id}: frame at {frame.sample_offset} overlaps or precedes the previous frame"
                )
            if frame.time_quality is not None and frame.time_quality not in _KNOWN_TIME_QUALITY_VALUES:
                raise FrameTimingTableError(f"{self.node_id}: unknown time_quality {frame.time_quality!r}")
            previous_end_offset = frame.sample_offset + frame.sample_count

    def to_json_payload(self) -> dict:
        return {
            "schema_version": FRAME_TIMING_SCHEMA_VERSION,
            "node_id": self.node_id,
            "sample_rate_hz": self.sample_rate_hz,
            "frames": [frame._asdict() for frame in self.frames],
        }

    @classmethod
    def from_json_payload(cls, payload: dict) -> NodeFrameTimingTable:
        if int(payload.get("schema_version", 0)) != FRAME_TIMING_SCHEMA_VERSION:
            raise FrameTimingTableError(f"unsupported frame timing schema_version: {payload.get('schema_version')}")
        frames = tuple(
            CapturedFrameTiming(
                sample_offset=int(entry["sample_offset"]),
                sample_count=int(entry["sample_count"]),
                start_time_ns=int(entry["start_time_ns"]),
                time_quality=entry.get("time_quality"),
                sequence=None if entry.get("sequence") is None else int(entry["sequence"]),
            )
            for entry in payload["frames"]
        )
        return cls(node_id=str(payload["node_id"]), sample_rate_hz=int(payload["sample_rate_hz"]), frames=frames)


def frame_timing_table_from_capture(
    node_id: str,
    sample_rate_hz: int,
    captured_frames: Sequence[CapturedFrameTiming],
    *,
    total_samples: int,
) -> NodeFrameTimingTable:
    """Clip captured frames to the extracted WAV (``[0, total_samples)``).

    A frame straddling the range edge keeps its own packet timestamp but only
    the samples inside the WAV, so its start time is advanced by the cut-off
    samples at the nominal rate.
    """
    clipped: list[CapturedFrameTiming] = []
    for frame in captured_frames:
        first_sample = max(0, frame.sample_offset)
        end_sample = min(total_samples, frame.sample_offset + frame.sample_count)
        if end_sample <= first_sample:
            continue
        leading_cut_samples = first_sample - frame.sample_offset
        clipped.append(
            frame._replace(
                sample_offset=first_sample,
                sample_count=end_sample - first_sample,
                start_time_ns=frame.start_time_ns
                + round(leading_cut_samples * NANOSECONDS_PER_SECOND / sample_rate_hz),
            )
        )
    return NodeFrameTimingTable(node_id=node_id, sample_rate_hz=sample_rate_hz, frames=tuple(clipped))


def uniform_frame_timing_table(
    node_id: str,
    *,
    sample_rate_hz: int,
    total_samples: int,
    audio_start_time_ns: int,
    frame_samples: int,
) -> NodeFrameTimingTable:
    """Evenly cut frames on the flat WAV timeline (a bundle with no frame timing table)."""
    if frame_samples <= 0:
        raise ValueError("frame_samples must be positive")
    frames = tuple(
        CapturedFrameTiming(
            sample_offset=first_sample,
            sample_count=min(frame_samples, total_samples - first_sample),
            start_time_ns=audio_start_time_ns + round(first_sample * NANOSECONDS_PER_SECOND / sample_rate_hz),
            time_quality=None,
            sequence=None,
        )
        for first_sample in range(0, total_samples, frame_samples)
    )
    return NodeFrameTimingTable(node_id=node_id, sample_rate_hz=sample_rate_hz, frames=frames)
