"""Calibration bundle -> the firmware binary uploads each node sent.

Each entry of a node's frame timing table (``calibration/frame_timing.py``)
becomes one firmware frame carrying that frame's own packet timestamp,
``time_quality`` and sequence number; frames are grouped into uploads of
``frames_per_upload``. A timestamp step recorded inside one upload is therefore
replayed inside one upload, which a flat one-timestamp-per-upload replay would
erase. Encoding is ``minimappr/sim/binary_ingest_encoder.py``.

Bundles that predate frame timing tables replay as evenly cut frames on the
WAV's flat timeline (``uniform_frame_timing_table``); the caller states the
frame size and the clock class to send, since such a bundle records neither.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from minimappr.calibration.bundle import CalibrationBundle
from minimappr.calibration.frame_timing import NodeFrameTimingTable, uniform_frame_timing_table
from minimappr.models import GeoPoint, NodeType, TimeQuality
from minimappr.sim.binary_ingest_encoder import (
    BinaryIngestFrame,
    BinaryIngestNodeHeader,
    encode_binary_ingest_payload,
)


class BundleReplayError(ValueError):
    """The bundle lacks something a faithful replay needs (node type, clock class, matching rate)."""


@dataclass(frozen=True, slots=True)
class ReplayUploadFraming:
    frames_per_upload: int
    # Added to every packet timestamp, e.g. to move a recording made in the
    # past next to the replaying server's clock. Relative timing is untouched.
    packet_time_shift_ns: int = 0
    # Only for a bundle without a frame timing table: firmware frame size to cut its WAV into.
    legacy_frame_samples: int | None = None
    # Sent for frames whose capture reported no clock class (the wire has no "unknown").
    unreported_time_quality: TimeQuality | None = None
    format_version: int = 2

    def __post_init__(self) -> None:
        if self.frames_per_upload <= 0:
            raise ValueError("frames_per_upload must be positive")


@dataclass(frozen=True, slots=True)
class ReplayUpload:
    node_id: str
    first_packet_time_ns: int
    frame_count: int
    payload: bytes


def bundle_node_frame_timing(bundle: CalibrationBundle, node_id: str, framing: ReplayUploadFraming) -> NodeFrameTimingTable:
    """The node's recorded frame timing, or the flat-timeline cut for an older bundle."""
    recorded = bundle.node_frame_timing(node_id)
    if recorded is not None:
        return recorded
    if framing.legacy_frame_samples is None:
        raise BundleReplayError(
            f"node {node_id} has no frame timing table; set ReplayUploadFraming.legacy_frame_samples to replay it"
        )
    channels_first, sample_rate_hz = bundle.node_audio(node_id)
    return uniform_frame_timing_table(
        node_id,
        sample_rate_hz=sample_rate_hz,
        total_samples=int(channels_first.shape[1]),
        audio_start_time_ns=int(bundle.node_manifest(node_id).get("audio_start_time_ns") or 0),
        frame_samples=framing.legacy_frame_samples,
    )


def bundle_node_header(bundle: CalibrationBundle, node_id: str, *, node_type: NodeType | None = None) -> BinaryIngestNodeHeader:
    """Node header from the bundle manifest.

    The manifest position is the best position the capture knew, so it is sent
    as a trusted GPS fix (the encoder's default ``fix_3d`` / ``gps*`` source),
    the only way a binary upload can place a node.
    """
    node = bundle.node_manifest(node_id)
    resolved_node_type = node_type
    if resolved_node_type is None and node.get("node_type"):
        resolved_node_type = NodeType(node["node_type"])
    if resolved_node_type is None:
        raise BundleReplayError(f"node {node_id} has no node_type in the bundle manifest; pass node_type")
    channel_count = len(node["channel_sensor_ids"])
    sensor_offsets_m = [tuple(float(axis) for axis in offset) for offset in node.get("sensor_offsets_m") or []]
    if len(sensor_offsets_m) != channel_count:
        raise BundleReplayError(
            f"node {node_id}: {len(sensor_offsets_m)} sensor offsets for {channel_count} audio channels"
        )
    position_geo = node.get("position_geo")
    return BinaryIngestNodeHeader(
        node_id=node_id,
        node_type=resolved_node_type,
        sensor_offsets_m=sensor_offsets_m,
        position_geo=GeoPoint(**position_geo) if position_geo else None,
    )


def _wire_time_quality(recorded: str | None, framing: ReplayUploadFraming, node_id: str) -> TimeQuality:
    if recorded is None:
        if framing.unreported_time_quality is None:
            raise BundleReplayError(
                f"node {node_id} has frames with no recorded time_quality; set ReplayUploadFraming.unreported_time_quality"
            )
        return framing.unreported_time_quality
    quality = TimeQuality(recorded)
    # Legacy spelling; the server reads it as ntp_disciplined and the wire has no code for it.
    return TimeQuality.NTP_DISCIPLINED if quality is TimeQuality.NTP_SYNC else quality


def bundle_node_frames(bundle: CalibrationBundle, node_id: str, framing: ReplayUploadFraming) -> list[BinaryIngestFrame]:
    """One firmware frame per timing-table entry, with that entry's own timing."""
    channels_first, sample_rate_hz = bundle.node_audio(node_id)
    table = bundle_node_frame_timing(bundle, node_id, framing)
    if table.sample_rate_hz != sample_rate_hz:
        raise BundleReplayError(
            f"node {node_id}: frame timing says {table.sample_rate_hz} Hz, audio is {sample_rate_hz} Hz"
        )
    total_samples = int(channels_first.shape[1])
    frames: list[BinaryIngestFrame] = []
    previous_sequence = 0
    for entry in table.frames:
        if entry.sample_offset < 0 or entry.sample_offset + entry.sample_count > total_samples:
            raise BundleReplayError(f"node {node_id}: frame at sample {entry.sample_offset} lies outside its audio")
        sequence = previous_sequence + 1 if entry.sequence is None else entry.sequence
        previous_sequence = sequence
        frames.append(
            BinaryIngestFrame(
                channels_first_samples=np.asarray(
                    channels_first[:, entry.sample_offset : entry.sample_offset + entry.sample_count]
                ),
                start_time_ns=entry.start_time_ns + framing.packet_time_shift_ns,
                sample_rate_hz=sample_rate_hz,
                start_sample_index=entry.sample_offset,
                sequence=sequence,
                time_quality=_wire_time_quality(entry.time_quality, framing, node_id),
            )
        )
    return frames


def bundle_node_uploads(
    bundle: CalibrationBundle,
    node_id: str,
    framing: ReplayUploadFraming,
    *,
    node_type: NodeType | None = None,
) -> list[ReplayUpload]:
    header = bundle_node_header(bundle, node_id, node_type=node_type)
    frames = bundle_node_frames(bundle, node_id, framing)
    uploads: list[ReplayUpload] = []
    for first_frame_index in range(0, len(frames), framing.frames_per_upload):
        upload_frames = frames[first_frame_index : first_frame_index + framing.frames_per_upload]
        uploads.append(
            ReplayUpload(
                node_id=node_id,
                first_packet_time_ns=upload_frames[0].start_time_ns,
                frame_count=len(upload_frames),
                payload=encode_binary_ingest_payload(header, upload_frames, format_version=framing.format_version),
            )
        )
    return uploads


def bundle_uploads_by_node(
    bundle: CalibrationBundle,
    framing: ReplayUploadFraming,
    *,
    node_ids: Iterable[str] | None = None,
    node_type_by_node_id: Mapping[str, NodeType] | None = None,
) -> dict[str, list[ReplayUpload]]:
    """Uploads for every bundle node (or ``node_ids``), keyed by node id."""
    selected_node_ids = list(node_ids) if node_ids is not None else [node["node_id"] for node in bundle.manifest["nodes"]]
    node_types = node_type_by_node_id or {}
    return {
        node_id: bundle_node_uploads(bundle, node_id, framing, node_type=node_types.get(node_id))
        for node_id in selected_node_ids
    }


def earliest_packet_time_ns(bundle: CalibrationBundle, framing: ReplayUploadFraming) -> int:
    """First recorded packet timestamp over all nodes (before ``packet_time_shift_ns``)."""
    first_frame_times_ns = [
        table.frames[0].start_time_ns
        for table in (bundle_node_frame_timing(bundle, node["node_id"], framing) for node in bundle.manifest["nodes"])
        if table.frames
    ]
    if not first_frame_times_ns:
        raise BundleReplayError("bundle has no frames to replay")
    return min(first_frame_times_ns)
