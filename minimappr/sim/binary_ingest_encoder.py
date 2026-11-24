"""Encoder for the firmware binary ingest upload (``MMB2`` / ``MMB3``).

Inverse of ``minimappr/api/binary_ingest.py`` (and the sidecar's Rust
decoder). Sim and test code use it to drive the real ingest endpoints with
packets whose timestamps are chosen by the caller: packet time is never read
from the wall clock here, so a test can deliberately set it apart from
receipt time, and every frame carries its own ``time_quality`` so a clock-class
change or timestamp step inside one upload can be expressed.

Diagnostics sections (timing, transport health, aux sensors, holdover) are not
emitted: v2 frames say "no timing diagnostics", v3 frames carry at most the
environment section.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from minimappr.models import GeoPoint, NodeType, TimeQuality, Vec3

NANOSECONDS_PER_SECOND = 1_000_000_000
# Firmware frame-count bounds enforced by both decoders.
MAX_FRAMES_PER_UPLOAD = 2048
MAX_STRING_BYTES = 255

_MAGIC_BY_FORMAT_VERSION = {2: b"MMB2", 3: b"MMB3"}
# Wire codes mirror the decoder tuples in minimappr/api/binary_ingest.py.
_NODE_TYPE_WIRE_CODE = {
    NodeType.POINT: 0,
    NodeType.SIRITH_TETRA: 1,
    NodeType.ARRAY: 2,
    NodeType.GATEWAY: 3,
    NodeType.SIRITH_PLANAR: 4,
}
_TIME_QUALITY_WIRE_CODE = {
    TimeQuality.GPS_LOCKED: 0,
    TimeQuality.GPS_HOLDOVER: 1,
    TimeQuality.NTP_DISCIPLINED: 2,
    TimeQuality.FREE_RUNNING: 3,
}
_AUDIO_SOURCE_TYPE_WIRE_CODE = {"tdm": 0, "i2s_mono": 1, "pdm_direct": 2, "synthetic": 3, "silence": 4}
_ENVIRONMENT_HAS_TEMPERATURE = 0x01
_ENVIRONMENT_HAS_HUMIDITY = 0x02
_V3_ENVIRONMENT_SECTION_BIT = 0x0002
# PCM16 full scale; the positive clip keeps +1.0 from wrapping to -32768.
_PCM16_FULL_SCALE = 32768.0
_PCM16_MAX_FLOAT = 32767.0 / _PCM16_FULL_SCALE


@dataclass(frozen=True, slots=True)
class BinaryIngestNodeHeader:
    """Node descriptor sent once per upload, ahead of the frames."""

    node_id: str
    node_type: NodeType
    sensor_offsets_m: Sequence[Vec3]
    position_geo: GeoPoint | None
    capabilities: Sequence[str] = ("audio", "array_localization")
    hardware: str = "sim-hardware"
    firmware: str = "sim-firmware"
    # "fix_3d" + a "gps*" position source is what the server treats as a
    # trusted GPS position with a measured altitude.
    gps_signal: str = "fix_3d"
    position_source: str = "gps_nmea_uart"
    boot_count: int = 1


@dataclass(frozen=True, slots=True)
class BinaryIngestFrame:
    """One captured audio frame with caller-chosen packet timing."""

    channels_first_samples: np.ndarray
    start_time_ns: int
    sample_rate_hz: int
    start_sample_index: int
    sequence: int
    time_quality: TimeQuality = TimeQuality.GPS_LOCKED
    # None derives the firmware defaults: end = start + duration, toa = start,
    # tor = end.
    end_time_ns: int | None = None
    toa_ns: int | None = None
    tor_ns: int | None = None
    temperature_c: float | None = None
    humidity_fraction: float | None = None
    audio_source_type: str = "pdm_direct"

    @property
    def samples_per_channel(self) -> int:
        return int(self.channels_first_samples.shape[1])

    def resolved_end_time_ns(self) -> int:
        if self.end_time_ns is not None:
            return int(self.end_time_ns)
        duration_ns = round(self.samples_per_channel * NANOSECONDS_PER_SECOND / self.sample_rate_hz)
        return int(self.start_time_ns) + int(duration_ns)


@dataclass(frozen=True, slots=True)
class ContinuousCaptureFraming:
    """How a continuous capture is cut into firmware frames.

    Packet time of frame k is ``first_frame_start_time_ns + k * frame_duration``
    plus any per-frame timestamp offset (a clock step), so the sample-index
    timeline and the packet-time timeline can be made to disagree on purpose.
    """

    sample_rate_hz: int
    first_frame_start_time_ns: int
    frame_samples: int
    first_sequence: int = 1
    first_sample_index: int = 0
    default_time_quality: TimeQuality = TimeQuality.GPS_LOCKED
    time_quality_by_frame_index: Mapping[int, TimeQuality] = field(default_factory=dict)
    timestamp_offset_ns_by_frame_index: Mapping[int, int] = field(default_factory=dict)


def frames_from_continuous_capture(
    channels_first_samples: np.ndarray,
    framing: ContinuousCaptureFraming,
) -> list[BinaryIngestFrame]:
    """Cut a channels-first capture into contiguous frames (tail frame may be short)."""
    if channels_first_samples.ndim != 2:
        raise ValueError("expected a channels-first (channels, samples) array")
    if framing.frame_samples <= 0:
        raise ValueError("frame_samples must be positive")
    total_samples = int(channels_first_samples.shape[1])
    frames: list[BinaryIngestFrame] = []
    for frame_index, frame_first_sample in enumerate(range(0, total_samples, framing.frame_samples)):
        frame_samples = channels_first_samples[:, frame_first_sample : frame_first_sample + framing.frame_samples]
        nominal_start_ns = framing.first_frame_start_time_ns + round(
            frame_first_sample * NANOSECONDS_PER_SECOND / framing.sample_rate_hz
        )
        frames.append(
            BinaryIngestFrame(
                channels_first_samples=frame_samples,
                start_time_ns=int(nominal_start_ns + framing.timestamp_offset_ns_by_frame_index.get(frame_index, 0)),
                sample_rate_hz=framing.sample_rate_hz,
                start_sample_index=framing.first_sample_index + frame_first_sample,
                sequence=framing.first_sequence + frame_index,
                time_quality=framing.time_quality_by_frame_index.get(frame_index, framing.default_time_quality),
            )
        )
    return frames


def encode_binary_ingest_payload(
    node: BinaryIngestNodeHeader,
    frames: Sequence[BinaryIngestFrame],
    *,
    format_version: int = 2,
    sort_by_toa: bool = True,
) -> bytes:
    """Encode one upload exactly as the firmware sends it to ``/api/v1/ingest/binary``."""
    magic = _MAGIC_BY_FORMAT_VERSION.get(format_version)
    if magic is None:
        raise ValueError(f"unsupported binary ingest format version {format_version}")
    if not 1 <= len(frames) <= MAX_FRAMES_PER_UPLOAD:
        raise ValueError(f"an upload carries 1..{MAX_FRAMES_PER_UPLOAD} frames, got {len(frames)}")
    payload = bytearray(magic)
    payload += struct.pack("<BBH", format_version, int(sort_by_toa), len(frames))
    payload += _encode_node_header(node)
    for frame in frames:
        payload += _encode_frame(frame, format_version=format_version)
    return bytes(payload)


def _encode_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    if len(encoded) > MAX_STRING_BYTES:
        raise ValueError(f"binary ingest strings are at most {MAX_STRING_BYTES} bytes: {value!r}")
    return struct.pack("<B", len(encoded)) + encoded


def _encode_node_header(node: BinaryIngestNodeHeader) -> bytes:
    if not node.sensor_offsets_m:
        raise ValueError("a node must declare at least one sensor")
    header = bytearray(_encode_string(node.node_id))
    header += struct.pack("<B", _NODE_TYPE_WIRE_CODE[node.node_type])
    if node.position_geo is None:
        header += struct.pack("<B", 0)
    else:
        header += struct.pack("<B", 1)
        header += struct.pack("<fff", node.position_geo.lat, node.position_geo.lon, node.position_geo.alt_m)
    header += struct.pack("<B", len(node.sensor_offsets_m))
    for offset_m in node.sensor_offsets_m:
        header += struct.pack("<fff", *(float(axis) for axis in offset_m))
    header += struct.pack("<B", len(node.capabilities))
    for capability in node.capabilities:
        header += _encode_string(capability)
    for text in (node.hardware, node.firmware, node.gps_signal, node.position_source):
        header += _encode_string(text)
    header += struct.pack("<I", node.boot_count)
    return bytes(header)


def _encode_environment(frame: BinaryIngestFrame) -> bytes:
    flags = 0
    body = bytearray()
    if frame.temperature_c is not None:
        flags |= _ENVIRONMENT_HAS_TEMPERATURE
        body += struct.pack("<f", frame.temperature_c)
    if frame.humidity_fraction is not None:
        flags |= _ENVIRONMENT_HAS_HUMIDITY
        body += struct.pack("<f", frame.humidity_fraction)
    return struct.pack("<B", flags) + bytes(body)


def _encode_pcm16_interleaved(channels_first_samples: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(channels_first_samples, dtype=np.float64).T, -1.0, _PCM16_MAX_FLOAT)
    return np.round(clipped * _PCM16_FULL_SCALE).astype("<i2").tobytes()


def _encode_frame(frame: BinaryIngestFrame, *, format_version: int) -> bytes:
    channel_count, samples_per_channel = frame.channels_first_samples.shape
    if channel_count < 1 or samples_per_channel < 1:
        raise ValueError("a frame needs at least one channel and one sample")
    toa_ns = frame.start_time_ns if frame.toa_ns is None else frame.toa_ns
    end_time_ns = frame.resolved_end_time_ns()
    tor_ns = end_time_ns if frame.tor_ns is None else frame.tor_ns
    encoded = bytearray(
        struct.pack(
            "<QQQQIB",
            int(frame.start_time_ns),
            end_time_ns,
            int(frame.start_sample_index),
            int(frame.start_sample_index) + samples_per_channel,
            int(frame.sample_rate_hz),
            channel_count,
        )
    )
    if format_version >= 3:
        encoded += struct.pack("<B", _AUDIO_SOURCE_TYPE_WIRE_CODE[frame.audio_source_type])
    encoded += struct.pack("<QQQB", int(frame.sequence), int(toa_ns), int(tor_ns), _TIME_QUALITY_WIRE_CODE[frame.time_quality])
    if format_version >= 3:
        encoded += struct.pack("<I", samples_per_channel)
        if frame.temperature_c is None and frame.humidity_fraction is None:
            encoded += struct.pack("<H", 0)
        else:
            environment_section = _encode_environment(frame)
            encoded += struct.pack("<HH", _V3_ENVIRONMENT_SECTION_BIT, len(environment_section))
            encoded += environment_section
    else:
        encoded += struct.pack("<B", 0)  # no timing diagnostics
        encoded += _encode_environment(frame)
        encoded += struct.pack("<I", samples_per_channel)
    encoded += _encode_pcm16_interleaved(frame.channels_first_samples)
    return bytes(encoded)
