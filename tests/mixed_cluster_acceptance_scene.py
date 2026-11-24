"""Scene and full-app harness for the mixed planar + point acceptance e2e
(``tests/test_mixed_cluster_acceptance_e2e.py``): geometry, capture
synthesis, firmware-shaped uploads, app wiring and the collected results.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization import speed_of_sound_mps
from minimappr.main import app
from minimappr.models import GeoPoint, NodeOrientation, NodeType
from minimappr.sim.binary_ingest_encoder import (
    BinaryIngestNodeHeader,
    ContinuousCaptureFraming,
    encode_binary_ingest_payload,
    frames_from_continuous_capture,
)
from tests.acoustic_scene_synthesis import AcousticScene, band_limited_noise_burst, scene_node_near_enu
from tests.rust_sidecar_e2e_helpers import (
    POLL_INTERVAL_SECONDS,
    configure_full_app_with_rust_sidecar,
    fake_sidecar_classifier_command_json,
    http_json,
    pick_free_tcp_port,
    post_binary_ingest,
    replay_dsp_event_manifests,
    write_fake_sidecar_classifier_helper,
)

# --- Scene ---
SITE_ORIGIN = GeoPoint(lat=44.98698840878797, lon=-93.2579197515542, alt_m=285.75)
COORDINATE_MODE = "flat"
SAMPLE_RATE_HZ = 48_000
# Sound speed the sidecar and Python derive from 20 C / 50 % RH.
SCENE_TEMPERATURE_C = 20.0
SCENE_HUMIDITY_FRACTION = 0.5
PLANAR_NODE_ID = "planar-acceptance"
POINT_NODE_ID = "point-acceptance"
PLANAR_CORNER_RADIUS_M = 0.025
PLANAR_SENSOR_OFFSETS_BODY_M = tuple(
    (PLANAR_CORNER_RADIUS_M * x_sign * np.sqrt(0.5), PLANAR_CORNER_RADIUS_M * y_sign * np.sqrt(0.5), 0.0)
    for x_sign, y_sign in ((1, 1), (-1, 1), (-1, -1), (1, -1))
) + ((0.0, 0.0, 0.0),)
PLANAR_ORIENTATION = NodeOrientation(yaw_deg=35.0)
PLANAR_APPROXIMATE_ENU_M = (37.0, -22.0, 1.5)
# Point on a 4 m pole: with the source below it and above the planar plane,
# the planar's mirrored bearing cannot meet the point's range difference, so
# the six-sensor fix is unique without a half-space prior.
POINT_OFFSET_FROM_PLANAR_M = (1.0, 2.0, 4.0)
SOURCE_OFFSET_FROM_PLANAR_M = (1.0, 2.5, 2.0)

SOURCE_DURATION_SECONDS = 0.5
SOURCE_BAND_HZ = (400.0, 3200.0)
SOURCE_PEAK_AMPLITUDE = 0.3
SOURCE_EDGE_RAMP_SECONDS = 0.005
SOURCE_SEED = 20261124
SENSOR_NOISE_RMS = 0.002
SENSOR_NOISE_SEED = 1124
# Quiet pre-roll so every node is streaming before the event onset.
PRE_ROLL_SECONDS = 1.0
CAPTURE_SECONDS = 2.0
FRAME_SAMPLES = 960
# Firmware-sized uploads (100 ms). The live trigger only inspects the
# localization window ending at each upload, so one large upload would hide
# the onset.
FRAMES_PER_UPLOAD = 5
# Packet clock deliberately behind receipt: a receipt-time fallback lands
# seconds outside the event's packet-time span.
PACKET_TIME_BEHIND_RECEIPT_NS = 4_321_987_654

# --- Run timing ---
LOCALIZATION_WINDOW_SECONDS = 0.08
DETECTION_WAIT_SECONDS = 15.0
# After the first detection, later renders/merges still have time to land.
DETECTION_SETTLE_SECONDS = 3.0
DETECTIONS_QUERY_LIMIT = 50
# The orientation override reaches the sidecar on the geometry publisher's own
# task; streaming starts only once the sidecar has applied it.
GEOMETRY_PUSH_WAIT_SECONDS = 10.0

APP_ENVIRONMENT = {
    "MINIMAPPR_SITE_ORIGIN_SOURCE": "manual",
    "MINIMAPPR_SITE_ORIGIN_LAT": repr(SITE_ORIGIN.lat),
    "MINIMAPPR_SITE_ORIGIN_LON": repr(SITE_ORIGIN.lon),
    "MINIMAPPR_SITE_ORIGIN_ALT_M": repr(SITE_ORIGIN.alt_m),
    "MINIMAPPR_COORDINATE_MODE": COORDINATE_MODE,
    "MINIMAPPR_DEFAULT_TEMPERATURE_C": repr(SCENE_TEMPERATURE_C),
    "MINIMAPPR_DEFAULT_HUMIDITY": repr(SCENE_HUMIDITY_FRACTION),
    # Above the noise floor, below the burst: one onset trigger per stream.
    "MINIMAPPR_TRIGGER_RMS": "0.02",
    "MINIMAPPR_TRIGGER_COOLDOWN_SECONDS": "5.0",
    # The sidecar's render interval is derived as classification window minus
    # BirdNET chunk overlap (2 s) = 1 s, so a render lands while the event is
    # buffered. The sidecar still clamps its own classification window to 15 s,
    # so a render here holds at most the audio buffered so far.
    "MINIMAPPR_CLASSIFICATION_WINDOW_SECONDS": "3.0",
    # Below the render spacing, so the pre-roll render's omni detection cannot
    # absorb the event detection in reporting dedupe.
    "MINIMAPPR_REPORTING_WINDOW_SECONDS": "0.25",
    "MINIMAPPR_LOCALIZATION_WINDOW_SECONDS": repr(LOCALIZATION_WINDOW_SECONDS),
    "MINIMAPPR_LOCALIZATION_SRP_GRID_RESOLUTION_M": "0.1",
    "MINIMAPPR_LOCALIZATION_SEARCH_PADDING_M": "1.0",
}


def _offset(base: tuple[float, float, float], delta: tuple[float, float, float]) -> tuple[float, float, float]:
    return (base[0] + delta[0], base[1] + delta[1], base[2] + delta[2])


def build_scene(*, include_point_node: bool, planar_orientation: NodeOrientation = PLANAR_ORIENTATION) -> AcousticScene:
    frame = LocalCoordinateFrame(origin=SITE_ORIGIN, mode=COORDINATE_MODE)
    planar = scene_node_near_enu(
        frame,
        node_id=PLANAR_NODE_ID,
        node_type=NodeType.SIRITH_PLANAR,
        approximate_position_enu_m=PLANAR_APPROXIMATE_ENU_M,
        sensor_offsets_body_m=PLANAR_SENSOR_OFFSETS_BODY_M,
        orientation=planar_orientation,
    )
    nodes = [planar]
    if include_point_node:
        nodes.append(
            scene_node_near_enu(
                frame,
                node_id=POINT_NODE_ID,
                node_type=NodeType.POINT,
                approximate_position_enu_m=_offset(PLANAR_APPROXIMATE_ENU_M, POINT_OFFSET_FROM_PLANAR_M),
                sensor_offsets_body_m=((0.0, 0.0, 0.0),),
            )
        )
    # Source placed relative to the planar node's float32 GPS fix, not the
    # approximate request, so the stated offset is exact.
    planar_position_m = frame.geo_to_local(planar.position_geo)
    return AcousticScene(
        site_origin=SITE_ORIGIN,
        coordinate_mode=COORDINATE_MODE,
        nodes=tuple(nodes),
        source_position_enu_m=_offset(planar_position_m, SOURCE_OFFSET_FROM_PLANAR_M),
        sample_rate_hz=SAMPLE_RATE_HZ,
        sound_speed_mps=speed_of_sound_mps(SCENE_TEMPERATURE_C, SCENE_HUMIDITY_FRACTION),
    )


@dataclass(slots=True)
class SceneRunResult:
    scene: AcousticScene
    detections: list[dict]
    manifests_by_id: dict[str, dict]
    emission_packet_time_ns: int
    first_upload_receipt_time_ns: int

    def event_arrival_span_ns(self) -> tuple[int, int]:
        """Packet-time span from the first sensor arrival to the last arrival's end."""
        first_arrival_s, last_arrival_end_s = self.scene.emission_to_arrival_span_seconds(SOURCE_DURATION_SECONDS)
        return (
            self.emission_packet_time_ns + int(first_arrival_s * 1e9),
            self.emission_packet_time_ns + int(last_arrival_end_s * 1e9),
        )

    def event_detections(self) -> list[dict]:
        """Detections whose classifier render overlaps the event; renders of
        pre-roll noise alone are not detections of this event."""
        event_start_ns, event_end_ns = self.event_arrival_span_ns()
        return [
            detection
            for detection in self.detections
            if int(detection["feature_summary"].get("rust_render_start_ns") or 0) <= event_end_ns
            and int(detection["feature_summary"].get("rust_render_end_ns") or 0) >= event_start_ns
        ]


def _register_planar_node_orientation(client: TestClient, scene: AcousticScene) -> None:
    """Operator registers the planar node (same GPS fix it streams) and pins its
    mount orientation as an override; position stays automatic GPS."""
    planar = scene.node(PLANAR_NODE_ID)
    registration = client.post(
        "/api/v1/nodes",
        json={
            "id": planar.node_id,
            "node_type": planar.node_type.value,
            "position_geo": planar.position_geo.model_dump(mode="json"),
            "sensor_offsets_m": [list(offset) for offset in planar.sensor_offsets_body_m],
            "capabilities": ["audio", "array_localization"],
        },
    )
    assert registration.status_code == 201, registration.text
    patch = client.patch(
        f"/api/v1/nodes/{planar.node_id}",
        json={"overrides": {"orientation": planar.orientation.model_dump(mode="json")}},
    )
    assert patch.status_code == 200, patch.text


def _wait_for_sidecar_orientation_placement(sidecar_port: int) -> None:
    """The operator configures the node before it streams: wait until the
    sidecar has applied a push carrying the planar node's placement."""
    deadline = time.monotonic() + GEOMETRY_PUSH_WAIT_SECONDS
    applied: dict | list = {}
    while time.monotonic() < deadline:
        applied = http_json(sidecar_port, "/api/v1/dsp/geometry")
        if isinstance(applied, dict) and int(applied.get("node_placement_count") or 0) >= 1:
            return
        time.sleep(POLL_INTERVAL_SECONDS)
    raise AssertionError(f"sidecar never applied the planar orientation placement: {applied}")


def _uploads_by_node(scene: AcousticScene, capture_start_packet_time_ns: int) -> dict[str, list[bytes]]:
    source = band_limited_noise_burst(
        sample_rate_hz=SAMPLE_RATE_HZ,
        duration_seconds=SOURCE_DURATION_SECONDS,
        band_hz=SOURCE_BAND_HZ,
        peak_amplitude=SOURCE_PEAK_AMPLITUDE,
        edge_ramp_seconds=SOURCE_EDGE_RAMP_SECONDS,
        seed=SOURCE_SEED,
    )
    captures = scene.render_node_captures(
        source,
        emission_offset_seconds=PRE_ROLL_SECONDS,
        capture_seconds=CAPTURE_SECONDS,
        sensor_noise_rms=SENSOR_NOISE_RMS,
        noise_seed=SENSOR_NOISE_SEED,
    )
    uploads: dict[str, list[bytes]] = {}
    for node in scene.nodes:
        header = BinaryIngestNodeHeader(
            node_id=node.node_id,
            node_type=node.node_type,
            sensor_offsets_m=node.sensor_offsets_body_m,
            position_geo=node.position_geo,
        )
        frames = frames_from_continuous_capture(
            captures[node.node_id],
            ContinuousCaptureFraming(
                sample_rate_hz=SAMPLE_RATE_HZ,
                first_frame_start_time_ns=capture_start_packet_time_ns,
                frame_samples=FRAME_SAMPLES,
            ),
        )
        uploads[node.node_id] = [
            encode_binary_ingest_payload(header, frames[first : first + FRAMES_PER_UPLOAD])
            for first in range(0, len(frames), FRAMES_PER_UPLOAD)
        ]
    return uploads


def _rust_handoff_detections(client: TestClient, node_ids: set[str]) -> list[dict]:
    response = client.get("/api/v1/detections", params={"limit": DETECTIONS_QUERY_LIMIT})
    assert response.status_code == 200
    return [
        detection
        for detection in response.json()
        if detection.get("source_node_id") in node_ids
        and (detection.get("feature_summary") or {}).get("rust_manifest_id")
    ]


def _manifests_by_id(sidecar_port: int) -> dict[str, dict]:
    return {
        str(manifest.get("manifest_id")): manifest
        for manifest in replay_dsp_event_manifests(sidecar_port)
        if manifest.get("manifest_type") in {"localization_result", "classifier_render"}
    }


def run_scene_through_full_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    scene: AcousticScene,
    arrival_order: tuple[str, ...],
    localization_solver: str,
) -> SceneRunResult:
    sidecar_port = pick_free_tcp_port()
    helper_path = write_fake_sidecar_classifier_helper(tmp_path)
    configure_full_app_with_rust_sidecar(
        monkeypatch,
        tmp_path,
        sidecar_port=sidecar_port,
        extra_env={
            **APP_ENVIRONMENT,
            "MINIMAPPR_LOCALIZATION_SINGLE_NODE_SOLVER": localization_solver,
            "MINIMAPPR_SIDECAR_CLASSIFIER_COMMAND_JSON": fake_sidecar_classifier_command_json(
                helper_path, sys.executable
            ),
        },
    )
    node_ids = {node.node_id for node in scene.nodes}
    result = SceneRunResult(
        scene=scene, detections=[], manifests_by_id={}, emission_packet_time_ns=0, first_upload_receipt_time_ns=0
    )
    with TestClient(app) as client:
        _register_planar_node_orientation(client, scene)
        _wait_for_sidecar_orientation_placement(sidecar_port)
        result.first_upload_receipt_time_ns = time.time_ns()
        capture_start_packet_time_ns = result.first_upload_receipt_time_ns - PACKET_TIME_BEHIND_RECEIPT_NS
        result.emission_packet_time_ns = capture_start_packet_time_ns + int(round(PRE_ROLL_SECONDS * 1e9))
        uploads = _uploads_by_node(scene, capture_start_packet_time_ns)
        for upload_index in range(len(uploads[arrival_order[0]])):
            for node_id in arrival_order:
                post_binary_ingest(sidecar_port, uploads[node_id][upload_index])

        deadline = time.monotonic() + DETECTION_WAIT_SECONDS
        while time.monotonic() < deadline and not result.event_detections():
            result.detections = _rust_handoff_detections(client, node_ids)
            time.sleep(POLL_INTERVAL_SECONDS)
        time.sleep(DETECTION_SETTLE_SECONDS)
        result.detections = _rust_handoff_detections(client, node_ids)
        result.manifests_by_id = _manifests_by_id(sidecar_port)
    return result


def manifest_digest(manifests_by_id: dict[str, dict]) -> list[str]:
    """One line per distinct manifest shape with its count, for failure messages."""
    shapes: Counter[str] = Counter()
    for manifest in manifests_by_id.values():
        localization = manifest.get("localization") or {}
        stream_key = str((manifest.get("source_handles") or [{}])[0].get("stream_key") or "")
        shapes[
            f"{manifest.get('manifest_type')} reporter={stream_key.split('__')[0]}"
            f" cluster={manifest.get('cluster_id')} cluster_sensors={len(manifest.get('cluster_sensor_positions') or [])}"
            f" active={(manifest.get('coverage_stats') or {}).get('active_channels')}"
            f" pairs={len(localization.get('pair_tdoas') or [])} algo={localization.get('resolved_algorithm')}"
            f" render_fallback={(manifest.get('classifier_render') or {}).get('fallback_reason')}"
        ] += 1
    return [f"{count}x {shape}" for shape, count in sorted(shapes.items())]


def measured_rust_lags_by_sensor_pair(manifest: dict, fallback_node_id: str) -> dict[tuple[str, str], float]:
    """Map a localization manifest's index pairs to sensor ids (Rust wire sign)."""
    cluster_sensor_positions = manifest.get("cluster_sensor_positions") or []
    sensor_ids = [str(entry[0]) for entry in cluster_sensor_positions]
    lags: dict[tuple[str, str], float] = {}
    for pair in manifest["localization"]["pair_tdoas"]:
        channel_a, channel_b = int(pair["ch_a"]), int(pair["ch_b"])
        sensor_a = sensor_ids[channel_a] if sensor_ids else f"{fallback_node_id}:ch{channel_a}"
        sensor_b = sensor_ids[channel_b] if sensor_ids else f"{fallback_node_id}:ch{channel_b}"
        lags[(sensor_a, sensor_b)] = float(pair["lag_seconds"])
    return lags


def event_localization_manifests(result: SceneRunResult) -> list[dict]:
    """Sidecar SRP solves whose source upload starts within the event. Includes
    render solves, whose window is the whole trailing buffer (mostly pre-roll
    noise), and a trigger solve merged with a render due on the same upload."""
    upload_samples = FRAMES_PER_UPLOAD * FRAME_SAMPLES
    _first_arrival_s, last_arrival_end_s = result.scene.emission_to_arrival_span_seconds(SOURCE_DURATION_SECONDS)
    first_event_upload_sample = int(PRE_ROLL_SECONDS * SAMPLE_RATE_HZ) - upload_samples
    last_event_sample = int((PRE_ROLL_SECONDS + last_arrival_end_s) * SAMPLE_RATE_HZ)
    return [
        manifest
        for manifest in result.manifests_by_id.values()
        if manifest.get("manifest_type") == "localization_result"
        and (manifest.get("localization") or {}).get("resolved_algorithm") == "srp_phat"
        and first_event_upload_sample
        <= int((manifest.get("source_handles") or [{}])[0].get("sample_index_start") or -1)
        <= last_event_sample
    ]
