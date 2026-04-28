"""Binary ingest payload parser.

The firmware binary endpoint keeps the existing frame/timestamp semantics but
removes JSON/base64 overhead from the audio transport. Integers are little
endian; audio samples are interleaved signed 16-bit PCM.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from minimappr.models import (
    AudioFrameIn,
    EnvironmentSampleIn,
    GeoPoint,
    NodeSpec,
    NodeType,
    TimeQuality,
)


_MAGIC = b"MMB1"
_VERSION = 1


@dataclass(slots=True)
class BinaryBufferedFrame:
    frame: AudioFrameIn
    environment: EnvironmentSampleIn | None
    decoded_audio: np.ndarray


@dataclass(slots=True)
class BinaryIngestPayload:
    node: NodeSpec
    buffered_frames: list[BinaryBufferedFrame]
    sort_by_toa: bool


class _BinaryReader:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0

    @property
    def remaining(self) -> int:
        return len(self._payload) - self._offset

    def read(self, length: int) -> bytes:
        if length < 0 or self._offset + length > len(self._payload):
            raise ValueError("Binary ingest payload ended unexpectedly")
        start = self._offset
        self._offset += length
        return self._payload[start:self._offset]

    def unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        return struct.unpack_from(fmt, self.read(size))[0]

    def u8(self) -> int:
        return self.unpack("<B")

    def u16(self) -> int:
        return self.unpack("<H")

    def u32(self) -> int:
        return self.unpack("<I")

    def u64(self) -> int:
        return self.unpack("<Q")

    def i32(self) -> int:
        return self.unpack("<i")

    def i64(self) -> int:
        return self.unpack("<q")

    def f32(self) -> float:
        return self.unpack("<f")

    def f64(self) -> float:
        return self.unpack("<d")

    def string(self) -> str:
        length = self.u8()
        return self.read(length).decode("utf-8")


def parse_binary_ingest_payload(raw_payload: bytes) -> BinaryIngestPayload:
    reader = _BinaryReader(raw_payload)
    if reader.read(4) != _MAGIC:
        raise ValueError("Invalid binary ingest magic")
    version = reader.u8()
    if version != _VERSION:
        raise ValueError(f"Unsupported binary ingest version {version}")

    sort_by_toa = bool(reader.u8())
    frame_count = reader.u16()
    if frame_count < 1 or frame_count > 2048:
        raise ValueError("Binary ingest frame count must be between 1 and 2048")

    node = _read_node(reader)
    buffered_frames = [_read_frame(reader) for _ in range(frame_count)]
    if reader.remaining != 0:
        raise ValueError("Binary ingest payload has trailing bytes")
    return BinaryIngestPayload(node=node, buffered_frames=buffered_frames, sort_by_toa=sort_by_toa)


def _read_node(reader: _BinaryReader) -> NodeSpec:
    node_id = reader.string()
    node_type_code = reader.u8()
    try:
        node_type = (NodeType.POINT, NodeType.SIRITH_TETRA, NodeType.ARRAY, NodeType.GATEWAY)[node_type_code]
    except IndexError as exc:
        raise ValueError(f"Unsupported binary node type {node_type_code}") from exc

    position_m = (reader.f32(), reader.f32(), reader.f32())
    has_geo_position = bool(reader.u8())
    position_geo = None
    if has_geo_position:
        position_geo = GeoPoint(lat=reader.f32(), lon=reader.f32(), alt_m=reader.f32())

    sensor_count = reader.u8()
    if sensor_count < 1:
        raise ValueError("Binary node must declare at least one sensor")
    sensor_offsets_m = [(reader.f32(), reader.f32(), reader.f32()) for _ in range(sensor_count)]

    capability_count = reader.u8()
    capabilities = [reader.string() for _ in range(capability_count)]
    hardware = reader.string()
    firmware = reader.string()
    gps_signal = reader.string()
    position_source = reader.string()
    boot_count = reader.u32()

    metadata = {
        "hardware": hardware or "unknown",
        "firmware": firmware or "dev",
        "boot_count": boot_count,
    }
    gps_metadata = {}
    if gps_signal:
        gps_metadata["signal"] = gps_signal
    if position_source:
        gps_metadata["position_source"] = position_source
    if gps_metadata:
        metadata["gps"] = gps_metadata

    return NodeSpec(
        id=node_id,
        node_type=node_type,
        position_m=position_m,
        position_geo=position_geo,
        sensor_offsets_m=sensor_offsets_m,
        capabilities=capabilities,
        metadata=metadata,
        properties={},
    )


def _read_frame(reader: _BinaryReader) -> BinaryBufferedFrame:
    start_time_ns = reader.u64()
    end_time_ns = reader.u64()
    start_sample_index = reader.u64()
    end_sample_index = reader.u64()
    sample_rate_hz = reader.u32()
    channels = reader.u8()
    sequence = reader.u64()
    toa_ns = reader.u64()
    tor_ns = reader.u64()
    time_quality = _read_time_quality(reader.u8())
    timing_diagnostics = _read_timing_diagnostics(reader)
    environment = _read_environment(reader)
    samples_per_channel = reader.u32()

    if channels < 1:
        raise ValueError("Binary frame must have at least one channel")
    if samples_per_channel < 1:
        raise ValueError("Binary frame must include at least one sample")
    expected_end_sample_index = start_sample_index + samples_per_channel
    if end_sample_index != expected_end_sample_index:
        raise ValueError("Binary frame sample indices do not match samples_per_channel")

    raw_audio = reader.read(samples_per_channel * channels * 2)
    decoded_audio = (
        np.frombuffer(raw_audio, dtype="<i2")
        .astype(np.float32)
        .reshape((samples_per_channel, channels))
        .T
        / 32768.0
    )

    frame = AudioFrameIn(
        start_time_ns=start_time_ns,
        utc_start_ns=start_time_ns,
        utc_end_ns=end_time_ns,
        start_sample_index=start_sample_index,
        end_sample_index=end_sample_index,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        encoding="pcm16le",
        samples_per_channel=samples_per_channel,
        samples_b64="AA==",
        sequence=sequence,
        time_quality=time_quality,
        toa_ns=toa_ns,
        tor_ns=tor_ns,
        timing_diagnostics=timing_diagnostics,
        source_type="raw_sensor",
    )
    return BinaryBufferedFrame(frame=frame, environment=environment, decoded_audio=decoded_audio)


def _read_time_quality(value: int) -> TimeQuality:
    qualities = (
        TimeQuality.GPS_LOCKED,
        TimeQuality.GPS_HOLDOVER,
        TimeQuality.NTP_DISCIPLINED,
        TimeQuality.FREE_RUNNING,
    )
    try:
        return qualities[value]
    except IndexError as exc:
        raise ValueError(f"Unsupported binary time quality {value}") from exc


def _read_timing_diagnostics(reader: _BinaryReader) -> dict:
    if not bool(reader.u8()):
        return {}
    return {
        "gps_anchor": bool(reader.u8()),
        "pps_edge_count": reader.u32(),
        "dma_ring_slot_index": reader.u32(),
        "pps_phase_error_ns": reader.i64(),
        "estimated_ppm": reader.f64(),
        "runner_frames_captured": reader.u64(),
        "runner_frames_dropped": reader.u64(),
        "runner_continuity_violations": reader.u64(),
        "runner_publish_errors": reader.u64(),
        "runner_queue_depth": reader.u32(),
        "runner_queue_overflows": reader.u64(),
        "runner_last_publish_status": reader.i32(),
        "packet_age_us": reader.u64(),
        "runner_last_publish_failure_stage": _read_publish_failure_stage(reader.u8()),
        "runner_last_publish_lwip_error": reader.i32(),
        "runner_consecutive_publish_failures": reader.u32(),
        "runner_publish_timeout_failures": reader.u64(),
        "runner_publish_connect_or_reset_failures": reader.u64(),
        "runner_publish_dns_failures": reader.u64(),
        "runner_publish_wifi_down_failures": reader.u64(),
    }


def _read_publish_failure_stage(value: int) -> str:
    stages = (
        "none",
        "dns",
        "connect",
        "send",
        "recv",
        "response_parse",
        "timeout",
        "wifi_disconnected",
    )
    try:
        return stages[value]
    except IndexError as exc:
        raise ValueError(f"Unsupported publish failure stage {value}") from exc


def _read_environment(reader: _BinaryReader) -> EnvironmentSampleIn | None:
    flags = reader.u8()
    if flags == 0:
        return None

    temperature_c = reader.f32() if flags & 0x01 else None
    humidity_fraction = reader.f32() if flags & 0x02 else None
    source = reader.string() if flags & 0x04 else None
    return EnvironmentSampleIn(
        temperature_c=temperature_c,
        humidity_fraction=humidity_fraction,
        source=source,
    )
