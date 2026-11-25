"""Per-frame packet timing in calibration bundles and their frame-by-frame
replay (plan item E4): capture -> bundle -> uploads, and legacy bundles."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from minimappr.api.binary_ingest import parse_binary_ingest_payload
from minimappr.calibration.bundle import build_ground_truth_payload, load_bundle, write_bundle_zip
from minimappr.calibration.frame_timing import (
    FrameTimingTableError,
    NodeFrameTimingTable,
    frame_timing_table_from_capture,
)
from minimappr.calibration.pipeline import CalibrationPipeline
from minimappr.core.capture_audio_buffer import CapturedFrameTiming, ChunkedCaptureSessionBuffer
from minimappr.core.capture_session import CaptureSessionRecord, CaptureState
from minimappr.core.environment import StaticEnvironmentProvider
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.models import GeoPoint, NodeType, TimeQuality
from minimappr.sim.replay_frames import (
    BundleReplayError,
    ReplayUpload,
    ReplayUploadFraming,
    bundle_node_uploads,
    earliest_packet_time_ns,
)
from minimappr.sim.replay_sidecar import drive_sidecar_with_uploads, interleave_uploads_by_packet_time
from tests.replay_realism_bundles import (
    BUNDLE_CAPTURE_START_NS,
    FRAMES_PER_UPLOAD,
    PLANAR_NODE_ID,
    SAMPLE_RATE_44_1_KHZ,
    SAMPLE_RATE_48_KHZ,
    TETRA_NODE_ID,
    PacketTimestampStep,
    build_replay_scene,
    frame_samples_for,
    planar_plan,
    render_captures_at_node_rates,
    tetra_plan,
    write_replay_bundle,
)

SAMPLE_RATE_HZ = 16_000
FRAME_SAMPLES = 160
CAPTURE_START_NS = 1_700_000_000_000_000_000
FRAME_DURATION_NS = 10_000_000
STEP_NS = 21_000_000
SENSOR_IDS = ["node-a:ch0", "node-a:ch1"]
ORIGIN = GeoPoint(lat=45.0, lon=-93.0, alt_m=250.0)


def _append_stepped_frames(capture: ChunkedCaptureSessionBuffer, *, frame_count: int, stepped_from: int) -> None:
    """Contiguous sample indices; packet time steps by STEP_NS from ``stepped_from``."""
    for frame_index in range(frame_count):
        start_ns = CAPTURE_START_NS + frame_index * FRAME_DURATION_NS + (STEP_NS if frame_index >= stepped_from else 0)
        for sensor_id in SENSOR_IDS:
            capture.append(
                sensor_id=sensor_id,
                sample_rate_hz=SAMPLE_RATE_HZ,
                start_time_ns=start_ns,
                samples=np.full(FRAME_SAMPLES, 0.1 * (frame_index + 1), dtype=np.float32),
                start_sample_index=frame_index * FRAME_SAMPLES,
                time_quality=TimeQuality.NTP_DISCIPLINED.value if frame_index == 3 else TimeQuality.GPS_LOCKED.value,
                sequence=100 + frame_index,
            )


def test_capture_buffer_keeps_each_frames_packet_timing_relative_to_the_extracted_range() -> None:
    capture = ChunkedCaptureSessionBuffer(SENSOR_IDS, max_duration_seconds=2.0)
    _append_stepped_frames(capture, frame_count=6, stepped_from=2)

    range_start_ns = CAPTURE_START_NS + FRAME_DURATION_NS  # from frame 1 on
    timings = capture.frame_timings_in_range(SENSOR_IDS[0], range_start_ns, CAPTURE_START_NS + 6 * FRAME_DURATION_NS)

    assert [timing.sample_offset for timing in timings] == [0, 160, 320, 480, 640]
    assert [timing.start_time_ns - CAPTURE_START_NS for timing in timings] == [
        FRAME_DURATION_NS,
        2 * FRAME_DURATION_NS + STEP_NS,
        3 * FRAME_DURATION_NS + STEP_NS,
        4 * FRAME_DURATION_NS + STEP_NS,
        5 * FRAME_DURATION_NS + STEP_NS,
    ]
    assert [timing.sequence for timing in timings] == [101, 102, 103, 104, 105]
    assert timings[2].time_quality == TimeQuality.NTP_DISCIPLINED.value
    assert {timing.time_quality for timing in timings[3:]} == {TimeQuality.GPS_LOCKED.value}
    assert capture.frame_timings_in_range("node-z:ch0", range_start_ns, range_start_ns + 1) == []


def test_frame_timing_table_validates_order_and_clock_class() -> None:
    frame = CapturedFrameTiming(0, 160, CAPTURE_START_NS, TimeQuality.GPS_LOCKED.value, 1)
    with pytest.raises(FrameTimingTableError, match="overlaps"):
        NodeFrameTimingTable("n", SAMPLE_RATE_HZ, (frame, frame._replace(sample_offset=100)))
    with pytest.raises(FrameTimingTableError, match="unknown time_quality"):
        NodeFrameTimingTable("n", SAMPLE_RATE_HZ, (frame._replace(time_quality="atomic"),))
    with pytest.raises(FrameTimingTableError, match="schema_version"):
        NodeFrameTimingTable.from_json_payload({"schema_version": 99, "node_id": "n", "sample_rate_hz": 1, "frames": []})
    table = NodeFrameTimingTable("n", SAMPLE_RATE_HZ, (frame, frame._replace(sample_offset=400, sequence=None)))
    assert NodeFrameTimingTable.from_json_payload(json.loads(json.dumps(table.to_json_payload()))) == table


def test_capture_frames_straddling_the_range_are_clipped_with_advanced_timestamps() -> None:
    frames = [
        CapturedFrameTiming(-40, 160, CAPTURE_START_NS, None, 7),
        CapturedFrameTiming(120, 160, CAPTURE_START_NS + FRAME_DURATION_NS, None, 8),
        CapturedFrameTiming(300, 160, CAPTURE_START_NS + 2 * FRAME_DURATION_NS, None, 9),
    ]
    table = frame_timing_table_from_capture("n", SAMPLE_RATE_HZ, frames, total_samples=200)
    assert table.frames == (
        CapturedFrameTiming(0, 120, CAPTURE_START_NS + 2_500_000, None, 7),
        CapturedFrameTiming(120, 80, CAPTURE_START_NS + FRAME_DURATION_NS, None, 8),
    )


class _NodeRowStorage:
    def __init__(self, frame: LocalCoordinateFrame) -> None:
        self._frame = frame

    async def get_node_by_id(self, node_id: str) -> dict | None:
        geo = self._frame.local_to_geo((3.0, 4.0, 1.0))
        return {
            "id": node_id,
            "node_type": NodeType.ARRAY.value,
            "position_geo": {"lat": geo.lat, "lon": geo.lon, "alt_m": geo.alt_m},
            "sensor_offsets_m": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            "orientation": {},
        }

    async def list_detections(self, limit: int = 100, **_kw):
        return []

    async def insert_large_artifact(self, **kwargs) -> str:
        return "lar-frame-timing"


async def test_calibration_capture_writes_frame_timing_that_replays_frame_by_frame(tmp_path: Path) -> None:
    frame = LocalCoordinateFrame(origin=ORIGIN, mode="flat")
    capture = ChunkedCaptureSessionBuffer(SENSOR_IDS, max_duration_seconds=2.0)
    _append_stepped_frames(capture, frame_count=6, stepped_from=2)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    end_ns = CAPTURE_START_NS + 6 * FRAME_DURATION_NS
    record = CaptureSessionRecord(
        session_id="frame-timing",
        state=CaptureState.PROCESSING,
        stream_key="calibration",
        range_lease_id=None,
        start_time_ns=CAPTURE_START_NS,
        end_time_ns=end_ns,
        first_frame_pts_ns=None,
        work_dir=work_dir,
        video_path=None,
        ambix_path=None,
        iamf_path=None,
        youtube_path=None,
        error=None,
        use_python_ingest=True,
        capture_audio_buffer=capture,
        capture_kind="calibration",
        node_channel_map={"node-a": SENSOR_IDS},
    )
    await CalibrationPipeline(
        storage=_NodeRowStorage(frame),
        coordinate_frame=frame,
        environment_provider=StaticEnvironmentProvider(temperature_c=20.0, humidity_fraction=0.5),
        artifact_dir=tmp_path / "artifacts",
    ).run(record)
    bundle_path = write_bundle_zip(record.work_dir, build_ground_truth_payload([]), tmp_path / "bundle.zip")
    with zipfile.ZipFile(bundle_path) as archive:
        assert "frame_timing/node-a.json" in archive.namelist()

    bundle = load_bundle(bundle_path)
    node_entry = bundle.node_manifest("node-a")
    assert node_entry["node_type"] == NodeType.ARRAY.value
    assert node_entry["frame_timing_file"] == "frame_timing/node-a.json"
    table = bundle.node_frame_timing("node-a")
    assert table is not None and len(table.frames) == 6
    assert table.frames[2].start_time_ns - table.frames[1].start_time_ns == FRAME_DURATION_NS + STEP_NS

    (upload,) = bundle_node_uploads(bundle, "node-a", ReplayUploadFraming(frames_per_upload=6))
    decoded = parse_binary_ingest_payload(upload.payload)
    assert decoded.node.node_type is NodeType.ARRAY
    assert [frame.frame.start_time_ns for frame in decoded.buffered_frames] == [t.start_time_ns for t in table.frames]
    assert [frame.frame.start_sample_index for frame in decoded.buffered_frames] == [0, 160, 320, 480, 640, 800]
    assert [frame.frame.sequence for frame in decoded.buffered_frames] == list(range(100, 106))
    assert decoded.buffered_frames[3].frame.time_quality is TimeQuality.NTP_DISCIPLINED


def _stepped_replay_bundle(tmp_path: Path) -> Path:
    plans = [planar_plan(SAMPLE_RATE_48_KHZ), tetra_plan(SAMPLE_RATE_44_1_KHZ)]
    scene = build_replay_scene(plans)
    return write_replay_bundle(
        tmp_path,
        scene=scene,
        node_plans=plans,
        captures=render_captures_at_node_rates(scene, plans),
        step=PacketTimestampStep(node_id=TETRA_NODE_ID, first_stepped_frame_index=22, step_ns=STEP_NS),
    )


def test_a_step_inside_one_upload_replays_inside_that_upload_without_a_gap(tmp_path: Path) -> None:
    bundle = load_bundle(_stepped_replay_bundle(tmp_path))
    shift_ns = 5_000_000_000
    framing = ReplayUploadFraming(frames_per_upload=FRAMES_PER_UPLOAD, packet_time_shift_ns=shift_ns)
    uploads = bundle_node_uploads(bundle, TETRA_NODE_ID, framing)
    frame_samples = frame_samples_for(SAMPLE_RATE_44_1_KHZ)
    frame_duration_ns = round(frame_samples * 1e9 / SAMPLE_RATE_44_1_KHZ)

    stepped_upload = parse_binary_ingest_payload(uploads[4].payload).buffered_frames
    packet_offsets_ns = [
        frame.frame.start_time_ns - (BUNDLE_CAPTURE_START_NS + shift_ns) - (20 + index) * frame_duration_ns
        for index, frame in enumerate(stepped_upload)
    ]
    assert packet_offsets_ns == [0, 0, STEP_NS, STEP_NS, STEP_NS]
    all_frames = [frame.frame for upload in uploads for frame in parse_binary_ingest_payload(upload.payload).buffered_frames]
    assert [frame.start_sample_index for frame in all_frames] == [index * frame_samples for index in range(len(all_frames))]
    assert [frame.sequence for frame in all_frames] == list(range(1, len(all_frames) + 1))
    assert {frame.sample_rate_hz for frame in all_frames} == {SAMPLE_RATE_44_1_KHZ}
    assert {frame.time_quality for frame in all_frames} == {TimeQuality.GPS_LOCKED}
    assert earliest_packet_time_ns(bundle, framing) == BUNDLE_CAPTURE_START_NS


def _strip_frame_timing(bundle_path: Path, out_path: Path, *, drop_node_type: bool = False) -> Path:
    """Rewrite a bundle as one written before frame timing tables existed."""
    with zipfile.ZipFile(bundle_path) as source, zipfile.ZipFile(out_path, "w") as target:
        for name in source.namelist():
            if name.startswith("frame_timing/"):
                continue
            payload = source.read(name)
            if name == "manifest.json":
                manifest = json.loads(payload)
                for node in manifest["nodes"]:
                    node.pop("frame_timing_file")
                    if drop_node_type:
                        node.pop("node_type")
                payload = json.dumps(manifest).encode()
            target.writestr(name, payload)
    return out_path


def test_a_bundle_without_frame_timing_replays_on_its_flat_timeline_when_told_how(tmp_path: Path) -> None:
    legacy = load_bundle(_strip_frame_timing(_stepped_replay_bundle(tmp_path), tmp_path / "legacy.zip"))
    assert legacy.node_frame_timing(PLANAR_NODE_ID) is None
    with pytest.raises(BundleReplayError, match="legacy_frame_samples"):
        bundle_node_uploads(legacy, PLANAR_NODE_ID, ReplayUploadFraming(frames_per_upload=FRAMES_PER_UPLOAD))
    with pytest.raises(BundleReplayError, match="unreported_time_quality"):
        bundle_node_uploads(
            legacy, PLANAR_NODE_ID, ReplayUploadFraming(frames_per_upload=FRAMES_PER_UPLOAD, legacy_frame_samples=480)
        )
    uploads = bundle_node_uploads(
        legacy,
        PLANAR_NODE_ID,
        ReplayUploadFraming(
            frames_per_upload=FRAMES_PER_UPLOAD,
            legacy_frame_samples=480,
            unreported_time_quality=TimeQuality.NTP_DISCIPLINED,
        ),
    )
    frames = [frame.frame for upload in uploads for frame in parse_binary_ingest_payload(upload.payload).buffered_frames]
    assert len(frames) == 200  # 2 s at 48 kHz in 10 ms frames
    assert [frame.start_time_ns - BUNDLE_CAPTURE_START_NS for frame in frames[:3]] == [0, 10_000_000, 20_000_000]
    assert {frame.time_quality for frame in frames} == {TimeQuality.NTP_DISCIPLINED}


def test_a_bundle_without_node_types_needs_them_stated(tmp_path: Path) -> None:
    legacy = load_bundle(_strip_frame_timing(_stepped_replay_bundle(tmp_path), tmp_path / "legacy.zip", drop_node_type=True))
    framing = ReplayUploadFraming(
        frames_per_upload=FRAMES_PER_UPLOAD, legacy_frame_samples=480, unreported_time_quality=TimeQuality.GPS_LOCKED
    )
    with pytest.raises(BundleReplayError, match="node_type"):
        bundle_node_uploads(legacy, PLANAR_NODE_ID, framing)
    (first, *_rest) = bundle_node_uploads(legacy, PLANAR_NODE_ID, framing, node_type=NodeType.SIRITH_PLANAR)
    assert parse_binary_ingest_payload(first.payload).node.node_type is NodeType.SIRITH_PLANAR


def _upload(node_id: str, first_packet_time_ns: int) -> ReplayUpload:
    return ReplayUpload(node_id=node_id, first_packet_time_ns=first_packet_time_ns, frame_count=5, payload=node_id.encode())


def test_uploads_post_in_live_arrival_order() -> None:
    uploads_by_node = {
        "b": [_upload("b", 100), _upload("b", 200)],
        "a": [_upload("a", 100), _upload("a", 221)],
    }
    schedule = interleave_uploads_by_packet_time(uploads_by_node, node_order=["a", "b"])
    assert [(upload.node_id, upload.first_packet_time_ns) for upload in schedule] == [
        ("a", 100),
        ("b", 100),
        ("b", 200),
        ("a", 221),
    ]
    with pytest.raises(ValueError, match="missing"):
        interleave_uploads_by_packet_time(uploads_by_node, node_order=["a"])

    posted: list[tuple[int, bytes]] = []
    summary = drive_sidecar_with_uploads(
        8123, schedule, post_upload=lambda port, payload: posted.append((port, payload)) or {"accepted": True}
    )
    assert posted == [(8123, b"a"), (8123, b"b"), (8123, b"b"), (8123, b"a")]
    assert summary.uploads_posted == 4
    assert summary.frames_by_node == {"a": 10, "b": 10}
    assert (summary.first_packet_time_ns, summary.last_upload_first_packet_time_ns) == (100, 221)
