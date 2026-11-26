"""Drive a realistic scene through normal ingest: real sidecar + full app.

Each node streams firmware-shaped binary uploads whose packet time is the
capture instant plus that node's own clock error, tagged with its clock class
(``time_quality``). A GPS node carries its fix in the packet; for a
hand-measured node the operator enters its position and stated accuracy as
an override before it streams (the app pushes that placement to the
sidecar; its packets carry the firmware's untrusted fallback fix). Shared by
the range matrix (E5) and the faint funnel (E7); E7's admission sweep also
streams the same uploads through Python direct ingest.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from minimappr.main import app
from minimappr.models import GeoPoint, TimeQuality
from minimappr.sim.binary_ingest_encoder import (
    BinaryIngestNodeHeader,
    ContinuousCaptureFraming,
    encode_binary_ingest_payload,
    frames_from_continuous_capture,
)
from tests.acoustic_scene_synthesis import AcousticScene, SceneNode
from tests.sidecar_manifest_collector import LiveSidecarManifestCollector
from tests.rust_sidecar_e2e_helpers import (
    POLL_INTERVAL_SECONDS,
    configure_full_app_with_rust_sidecar,
    fake_sidecar_classifier_command_json,
    http_json,
    pick_free_tcp_port,
    post_binary_ingest,
    fetch_live_dsp_results,
    replay_dsp_event_manifests,
    write_fake_sidecar_classifier_helper,
)

INGEST_BACKEND_RUST_SIDECAR = "rust_sidecar"
INGEST_BACKEND_PYTHON_DIRECT = "python_direct"
FRAME_SAMPLES = 960
# 100 ms uploads, as the firmware sends: the live trigger inspects the window
# ending at each upload, so one large upload would hide the onset.
FRAMES_PER_UPLOAD = 5
# Packet clock deliberately behind receipt, so a receipt-time fallback is visible.
PACKET_TIME_BEHIND_RECEIPT_NS = 4_321_987_654
# Half the 100 ms an upload spans: near real time without doubling the run.
UPLOAD_ROUND_PACING_SECONDS = 0.05
DETECTION_SETTLE_SECONDS = 3.0
# Live-collected runs wait (up to the timeout) until the sidecar has solved a
# window within this margin of the capture end (its periodic solves report progress).
SIDECAR_CAUGHT_UP_MARGIN_SECONDS = 1.0
SIDECAR_DRAIN_TIMEOUT_SECONDS = 180.0
# The replay stream's first byte can lag while the sidecar finishes renders.
SSE_REPLAY_IDLE_SECONDS = 3.0
DETECTIONS_QUERY_LIMIT = 1000
GEOMETRY_PUSH_WAIT_SECONDS = 10.0
# A node without a GPS fix still sends a position: its compiled-in fallback
# (``NmeaGpsSource::updateDescriptor``), tagged ``fallback_static``, which the
# server must not trust. The same fallback for every node, as provisioned.
NO_GPS_SIGNAL = "no_fix"
NO_GPS_POSITION_SOURCE = "fallback_static"
FIRMWARE_FALLBACK_GEO = GeoPoint(lat=44.9869884, lon=-93.2579198, alt_m=285.75)


@dataclass(frozen=True)
class StreamingNode:
    scene_node: SceneNode
    time_quality: TimeQuality
    packet_clock_error_ns: int
    # None: the node streams its packet GPS fix. Otherwise the operator's entry.
    entered_position_enu_m: tuple[float, float, float] | None = None
    entered_position_std_m: float | None = None


@dataclass(frozen=True, slots=True)
class UploadReceipt:
    """What ingest answered for one upload. The sidecar only acknowledges the
    enqueue; direct ingest also counts frames and carries per-frame
    ``triggered`` flags in ``frame_results``."""

    node_id: str
    upload_index: int
    accepted: bool
    total_frames: int | None = None
    accepted_frames: int | None = None
    frame_results: tuple[dict, ...] = ()

    @classmethod
    def from_response(cls, node_id: str, upload_index: int, response: dict) -> "UploadReceipt":
        return cls(
            node_id=node_id,
            upload_index=upload_index,
            accepted=bool(response.get("accepted")),
            total_frames=response.get("total_frames"),
            accepted_frames=response.get("accepted_frames"),
            frame_results=tuple(response.get("results") or ()),
        )

    @property
    def fully_received(self) -> bool:
        return self.accepted and (self.total_frames is None or self.accepted_frames == self.total_frames)


@dataclass(slots=True)
class FullAppRunResult:
    capture_start_packet_time_ns: int
    ingest_backend: str = INGEST_BACKEND_RUST_SIDECAR
    detections: list[dict] = field(default_factory=list)
    manifests_by_id: dict[str, dict] = field(default_factory=dict)
    upload_receipts: list[UploadReceipt] = field(default_factory=list)
    manifest_replay_gap_count: int = 0
    sidecar_drain_wait_seconds: float | None = None
    sidecar_caught_up: bool | None = None


def _node_header(node: StreamingNode) -> BinaryIngestNodeHeader:
    scene_node = node.scene_node
    if node.entered_position_enu_m is None:
        return BinaryIngestNodeHeader(
            node_id=scene_node.node_id,
            node_type=scene_node.node_type,
            sensor_offsets_m=scene_node.sensor_offsets_body_m,
            position_geo=scene_node.position_geo,
        )
    return BinaryIngestNodeHeader(
        node_id=scene_node.node_id,
        node_type=scene_node.node_type,
        sensor_offsets_m=scene_node.sensor_offsets_body_m,
        position_geo=FIRMWARE_FALLBACK_GEO,
        gps_signal=NO_GPS_SIGNAL,
        position_source=NO_GPS_POSITION_SOURCE,
    )


def build_uploads_by_node(
    nodes: Sequence[StreamingNode],
    captures_by_node: Mapping[str, np.ndarray],
    *,
    sample_rate_hz: int,
    capture_start_packet_time_ns: int,
) -> dict[str, list[bytes]]:
    uploads: dict[str, list[bytes]] = {}
    for node in nodes:
        node_id = node.scene_node.node_id
        frames = frames_from_continuous_capture(
            captures_by_node[node_id],
            ContinuousCaptureFraming(
                sample_rate_hz=sample_rate_hz,
                first_frame_start_time_ns=capture_start_packet_time_ns + node.packet_clock_error_ns,
                frame_samples=FRAME_SAMPLES,
                default_time_quality=node.time_quality,
            ),
        )
        header = _node_header(node)
        uploads[node_id] = [
            encode_binary_ingest_payload(header, frames[first : first + FRAMES_PER_UPLOAD])
            for first in range(0, len(frames), FRAMES_PER_UPLOAD)
        ]
    return uploads


def enter_hand_measured_positions(client: TestClient, nodes: Sequence[StreamingNode]) -> int:
    entered_count = 0
    for node in nodes:
        if node.entered_position_enu_m is None:
            continue
        scene_node = node.scene_node
        registration = client.post(
            "/api/v1/nodes",
            json={
                "id": scene_node.node_id,
                "node_type": scene_node.node_type.value,
                "position_m": list(node.entered_position_enu_m),
                "sensor_offsets_m": [list(offset) for offset in scene_node.sensor_offsets_body_m],
                "capabilities": ["audio", "array_localization"],
            },
        )
        assert registration.status_code == 201, registration.text
        patch = client.patch(
            f"/api/v1/nodes/{scene_node.node_id}",
            json={
                "overrides": {
                    "position_m": list(node.entered_position_enu_m),
                    "position_std_m": node.entered_position_std_m,
                }
            },
        )
        assert patch.status_code == 200, patch.text
        entered_count += 1
    return entered_count


def _wait_for_sidecar_placements(sidecar_port: int, expected_count: int) -> None:
    deadline = time.monotonic() + GEOMETRY_PUSH_WAIT_SECONDS
    applied: dict | list = {}
    while time.monotonic() < deadline:
        applied = http_json(sidecar_port, "/api/v1/dsp/geometry")
        if isinstance(applied, dict) and int(applied.get("node_placement_count") or 0) >= expected_count:
            return
        time.sleep(POLL_INTERVAL_SECONDS)
    raise AssertionError(f"sidecar never applied {expected_count} hand-measured placements: {applied}")


def _rust_handoff_detections(client: TestClient, node_ids: set[str]) -> list[dict]:
    response = client.get("/api/v1/detections", params={"limit": DETECTIONS_QUERY_LIMIT})
    assert response.status_code == 200
    return [
        detection
        for detection in response.json()
        if detection.get("source_node_id") in node_ids
        and (detection.get("feature_summary") or {}).get("rust_manifest_id")
    ]


def _wait_for_sidecar_to_reach(collector: LiveSidecarManifestCollector, capture_end_ns: int, result: FullAppRunResult) -> None:
    """A loaded machine can leave the sidecar tens of seconds behind the
    uploads; scoring before it reaches the end would drop the last calls."""
    started = time.monotonic()
    target_ns = capture_end_ns - int(SIDECAR_CAUGHT_UP_MARGIN_SECONDS * 1e9)
    while time.monotonic() - started < SIDECAR_DRAIN_TIMEOUT_SECONDS:
        if collector.latest_window_center_time_ns >= target_ns:
            break
        time.sleep(POLL_INTERVAL_SECONDS)
    result.sidecar_drain_wait_seconds = time.monotonic() - started
    result.sidecar_caught_up = collector.latest_window_center_time_ns >= target_ns


def stream_uploads(
    nodes: Sequence[StreamingNode],
    uploads: Mapping[str, list[bytes]],
    post_upload: Callable[[bytes], dict],
    result: FullAppRunResult,
) -> None:
    upload_count = max(len(node_uploads) for node_uploads in uploads.values())
    for upload_index in range(upload_count):
        for node in nodes:
            node_uploads = uploads[node.scene_node.node_id]
            if upload_index < len(node_uploads):
                response = post_upload(node_uploads[upload_index])
                result.upload_receipts.append(UploadReceipt.from_response(node.scene_node.node_id, upload_index, response))
        # Nodes stream concurrently in the field: every node's upload k is in
        # before anyone's k+1, so solves and renders see the cluster covered.
        time.sleep(UPLOAD_ROUND_PACING_SECONDS)


def wait_for_detections(
    fetch_detections: Callable[[], list[dict]],
    result: FullAppRunResult,
    detection_wait_seconds: float,
    detections_complete: Callable[[FullAppRunResult], bool],
) -> None:
    deadline = time.monotonic() + detection_wait_seconds
    while time.monotonic() < deadline:
        result.detections = fetch_detections()
        if detections_complete(result):
            break
        time.sleep(POLL_INTERVAL_SECONDS)
    time.sleep(DETECTION_SETTLE_SECONDS)
    result.detections = fetch_detections()


def run_streaming_scene_through_full_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    scene: AcousticScene,
    nodes: Sequence[StreamingNode],
    captures_by_node: Mapping[str, np.ndarray],
    app_environment: Mapping[str, str],
    detection_wait_seconds: float,
    detections_complete: Callable[[FullAppRunResult], bool],
    replay_manifests_over_sse: bool = False,
    ingest_backend: str = INGEST_BACKEND_RUST_SIDECAR,
    collect_manifests_live: bool = False,
) -> FullAppRunResult:
    """Stream every node's capture (uploads interleaved in node order) and collect
    the stored detections, the per-upload ingest receipts and (sidecar) manifests.

    ``ingest_backend`` picks the real Rust sidecar or the app's Python direct
    ingest (``tests/realistic_scene_direct_ingest.py``) for identical uploads.
    ``detections_complete(result)`` ends the wait early; a settle period
    always follows so later renders and merges still land. Manifests come
    from the short results ring unless ``replay_manifests_over_sse`` (the 512
    event SSE replay, which a busy sidecar can stall) or
    ``collect_manifests_live`` (follows the stream during the run; for runs
    long enough to overflow the replay ring).
    """
    if ingest_backend == INGEST_BACKEND_PYTHON_DIRECT:
        from tests.realistic_scene_direct_ingest import run_streaming_scene_through_direct_ingest

        return run_streaming_scene_through_direct_ingest(
            monkeypatch,
            tmp_path,
            scene=scene,
            nodes=nodes,
            captures_by_node=captures_by_node,
            app_environment=app_environment,
            detection_wait_seconds=detection_wait_seconds,
            detections_complete=detections_complete,
        )
    if ingest_backend != INGEST_BACKEND_RUST_SIDECAR:
        raise ValueError(f"unknown ingest backend {ingest_backend!r}")
    sidecar_port = pick_free_tcp_port()
    helper_path = write_fake_sidecar_classifier_helper(tmp_path)
    configure_full_app_with_rust_sidecar(
        monkeypatch,
        tmp_path,
        sidecar_port=sidecar_port,
        extra_env={
            **app_environment,
            "MINIMAPPR_SIDECAR_CLASSIFIER_COMMAND_JSON": fake_sidecar_classifier_command_json(helper_path, sys.executable),
        },
    )
    node_ids = {node.scene_node.node_id for node in nodes}
    with TestClient(app) as client:
        entered_count = enter_hand_measured_positions(client, nodes)
        if entered_count:
            _wait_for_sidecar_placements(sidecar_port, entered_count)
        result = FullAppRunResult(
            capture_start_packet_time_ns=time.time_ns() - PACKET_TIME_BEHIND_RECEIPT_NS,
            ingest_backend=INGEST_BACKEND_RUST_SIDECAR,
        )
        uploads = build_uploads_by_node(
            nodes,
            captures_by_node,
            sample_rate_hz=scene.sample_rate_hz,
            capture_start_packet_time_ns=result.capture_start_packet_time_ns,
        )
        collector = LiveSidecarManifestCollector(sidecar_port) if collect_manifests_live else None
        if collector is not None:
            collector.start()
        stream_uploads(nodes, uploads, lambda payload: post_binary_ingest(sidecar_port, payload), result)
        if collector is not None:
            capture_samples = max(capture.shape[1] for capture in captures_by_node.values())
            capture_end_ns = result.capture_start_packet_time_ns + int(capture_samples / scene.sample_rate_hz * 1e9)
            _wait_for_sidecar_to_reach(collector, capture_end_ns, result)
        wait_for_detections(
            lambda: _rust_handoff_detections(client, node_ids), result, detection_wait_seconds, detections_complete
        )
        if collector is not None:
            manifests = collector.stop()
            result.manifest_replay_gap_count = collector.replay_gap_count
        elif replay_manifests_over_sse:
            manifests = replay_dsp_event_manifests(sidecar_port, idle_seconds=SSE_REPLAY_IDLE_SECONDS)
        else:
            manifests = fetch_live_dsp_results(sidecar_port)
        result.manifests_by_id = {
            str(manifest.get("manifest_id")): manifest
            for manifest in manifests
            if manifest.get("manifest_type") in {"localization_result", "classifier_render"}
        }
    return result
