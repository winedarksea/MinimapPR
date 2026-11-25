"""Drive a realistic scene through normal ingest: real sidecar + full app.

Each node streams firmware-shaped binary uploads whose packet time is the
capture instant plus that node's own clock error, tagged with its clock class
(``time_quality``). A GPS node carries its fix in the packet; for a
hand-measured node the operator enters its position and stated accuracy as
an override before it streams (the app pushes that placement to the
sidecar; its packets carry the firmware's untrusted fallback fix). Shared by
the range matrix (E5) and the faint funnel (E7).
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

FRAME_SAMPLES = 960
# 100 ms uploads, as the firmware sends: the live trigger inspects the window
# ending at each upload, so one large upload would hide the onset.
FRAMES_PER_UPLOAD = 5
# Packet clock deliberately behind receipt, so a receipt-time fallback is visible.
PACKET_TIME_BEHIND_RECEIPT_NS = 4_321_987_654
# Half the 100 ms an upload spans: near real time without doubling the run.
UPLOAD_ROUND_PACING_SECONDS = 0.05
DETECTION_SETTLE_SECONDS = 3.0
# The replay stream's first byte can lag while the sidecar finishes renders.
SSE_REPLAY_IDLE_SECONDS = 3.0
DETECTIONS_QUERY_LIMIT = 200
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


@dataclass(slots=True)
class FullAppRunResult:
    capture_start_packet_time_ns: int
    detections: list[dict] = field(default_factory=list)
    manifests_by_id: dict[str, dict] = field(default_factory=dict)


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


def _enter_hand_measured_positions(client: TestClient, nodes: Sequence[StreamingNode]) -> int:
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
) -> FullAppRunResult:
    """Stream every node's capture (uploads interleaved in node order) and collect
    the stored Rust-handoff detections and sidecar manifests.

    ``detections_complete(result)`` ends the wait early; a settle period
    always follows so later renders and merges still land. Manifests come
    from the short results ring unless ``replay_manifests_over_sse``: the SSE
    replay keeps every event (512) but a busy sidecar can stall it, so only
    callers that score manifests pay for it.
    """
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
        entered_count = _enter_hand_measured_positions(client, nodes)
        if entered_count:
            _wait_for_sidecar_placements(sidecar_port, entered_count)
        result = FullAppRunResult(capture_start_packet_time_ns=time.time_ns() - PACKET_TIME_BEHIND_RECEIPT_NS)
        uploads = build_uploads_by_node(
            nodes,
            captures_by_node,
            sample_rate_hz=scene.sample_rate_hz,
            capture_start_packet_time_ns=result.capture_start_packet_time_ns,
        )
        upload_count = max(len(node_uploads) for node_uploads in uploads.values())
        for upload_index in range(upload_count):
            for node in nodes:
                node_uploads = uploads[node.scene_node.node_id]
                if upload_index < len(node_uploads):
                    post_binary_ingest(sidecar_port, node_uploads[upload_index])
            # Nodes stream concurrently in the field: every node's upload k is in
            # before anyone's k+1, so solves and renders see the cluster covered.
            time.sleep(UPLOAD_ROUND_PACING_SECONDS)
        deadline = time.monotonic() + detection_wait_seconds
        while time.monotonic() < deadline:
            result.detections = _rust_handoff_detections(client, node_ids)
            if detections_complete(result):
                break
            time.sleep(POLL_INTERVAL_SECONDS)
        time.sleep(DETECTION_SETTLE_SECONDS)
        result.detections = _rust_handoff_detections(client, node_ids)
        manifests = (
            replay_dsp_event_manifests(sidecar_port, idle_seconds=SSE_REPLAY_IDLE_SECONDS)
            if replay_manifests_over_sse
            else fetch_live_dsp_results(sidecar_port)
        )
        result.manifests_by_id = {
            str(manifest.get("manifest_id")): manifest
            for manifest in manifests
            if manifest.get("manifest_type") in {"localization_result", "classifier_render"}
        }
    return result
