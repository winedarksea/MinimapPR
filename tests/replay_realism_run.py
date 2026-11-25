"""Replay a calibration bundle through the real sidecar and the full app (E4).

The app launches the worktree's sidecar; the bundle's per-frame uploads are
posted by ``minimappr/sim/replay_sidecar.py`` with packet time shifted to just
behind receipt time (deliberately apart from it, as in the E2 acceptance run).
Collects the stored Rust-handoff detections and every localization /
classifier-render manifest still in the sidecar's SSE replay ring.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from minimappr.calibration.bundle import load_bundle
from minimappr.main import app
from minimappr.sim.replay_frames import ReplayUploadFraming, earliest_packet_time_ns
from minimappr.sim.replay_sidecar import SidecarReplaySummary, replay_bundle_through_sidecar
from tests.acoustic_scene_synthesis import AcousticScene
from tests.mixed_cluster_acceptance_scene import (
    APP_ENVIRONMENT,
    PACKET_TIME_BEHIND_RECEIPT_NS,
    SOURCE_DURATION_SECONDS,
    manifest_digest,
)
from tests.replay_realism_bundles import (
    FRAMES_PER_UPLOAD,
    PRE_ROLL_SECONDS,
    PacketClockDrift,
    PacketTimestampStep,
    ReplayNodePlan,
    build_replay_scene,
    render_captures_at_node_rates,
    write_replay_bundle,
)
from tests.rust_sidecar_e2e_helpers import (
    POLL_INTERVAL_SECONDS,
    configure_full_app_with_rust_sidecar,
    fake_sidecar_classifier_command_json,
    http_json,
    pick_free_tcp_port,
    replay_dsp_event_manifests,
    write_fake_sidecar_classifier_helper,
)

DETECTION_WAIT_SECONDS = 15.0
DETECTION_SETTLE_SECONDS = 3.0
DETECTIONS_QUERY_LIMIT = 100
SITE_FRAME_WAIT_SECONDS = 10.0
SOLVE_MANIFEST_TYPES = frozenset({"localization_result", "classifier_render"})
EVENT_TRIGGER_ATTEMPT_KINDS = frozenset({"event_trigger_absolute_rms", "event_trigger_noise_floor_relative"})


@dataclass(slots=True)
class ReplayRunResult:
    scene: AcousticScene
    replay_summary: SidecarReplaySummary
    packet_time_shift_ns: int
    emission_packet_time_ns: int
    detections: list[dict]
    manifests: list[dict]

    def digest(self) -> str:
        manifest_lines = manifest_digest({str(index): manifest for index, manifest in enumerate(self.manifests)})
        detection_lines = [
            f"detection {detection['source_node_id']} {detection.get('reporting_modality')} "
            f"t={(int(detection['timestamp_ns']) - self.emission_packet_time_ns) / 1e9:+.3f}s from emission "
            f"render={detection['feature_summary'].get('rust_render_start_ns')}..{detection['feature_summary'].get('rust_render_end_ns')} "
            f"fallback={detection['feature_summary'].get('rust_fallback_reason')}"
            for detection in self.detections
        ]
        return "\n".join(manifest_lines + detection_lines)

    def event_arrival_span_ns(self, source_duration_seconds: float) -> tuple[int, int]:
        first_arrival_s, last_arrival_end_s = self.scene.emission_to_arrival_span_seconds(source_duration_seconds)
        return (
            self.emission_packet_time_ns + int(first_arrival_s * 1e9),
            self.emission_packet_time_ns + int(last_arrival_end_s * 1e9),
        )


def _wait_for_sidecar_site_frame(sidecar_port: int) -> None:
    """Python pushes the site frame at launch; streaming before it lands would
    give solves no frame id, which the handoff rightly refuses."""
    deadline = time.monotonic() + SITE_FRAME_WAIT_SECONDS
    applied: dict | list = {}
    while time.monotonic() < deadline:
        applied = http_json(sidecar_port, "/api/v1/dsp/geometry")
        if isinstance(applied, dict) and (applied.get("coordinate_frame") or {}).get("frame_id"):
            return
        time.sleep(POLL_INTERVAL_SECONDS)
    raise AssertionError(f"sidecar never applied the site frame push: {applied}")


def _rust_handoff_detections(client: TestClient, node_ids: set[str]) -> list[dict]:
    response = client.get("/api/v1/detections", params={"limit": DETECTIONS_QUERY_LIMIT})
    assert response.status_code == 200
    return [
        detection
        for detection in response.json()
        if detection.get("source_node_id") in node_ids
        and (detection.get("feature_summary") or {}).get("rust_manifest_id")
    ]


def run_bundle_through_full_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    bundle_path: Path,
    scene: AcousticScene,
    node_order: Sequence[str],
    awaited_detection_node_ids: Sequence[str],
    localization_solver: str = "rust",
    extra_env: dict[str, str] | None = None,
) -> ReplayRunResult:
    """Wait (bounded) until each of ``awaited_detection_node_ids`` has a stored
    detection whose render covers the event, then let later renders settle."""
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
            **(extra_env or {}),
        },
    )
    bundle = load_bundle(bundle_path)
    node_ids = set(node_order)
    with TestClient(app) as client:
        _wait_for_sidecar_site_frame(sidecar_port)
        unshifted_framing = ReplayUploadFraming(frames_per_upload=FRAMES_PER_UPLOAD)
        first_packet_time_ns = time.time_ns() - PACKET_TIME_BEHIND_RECEIPT_NS
        packet_time_shift_ns = first_packet_time_ns - earliest_packet_time_ns(bundle, unshifted_framing)
        summary = replay_bundle_through_sidecar(
            bundle,
            sidecar_port,
            ReplayUploadFraming(frames_per_upload=FRAMES_PER_UPLOAD, packet_time_shift_ns=packet_time_shift_ns),
            node_order=node_order,
        )
        result = ReplayRunResult(
            scene=scene,
            replay_summary=summary,
            packet_time_shift_ns=packet_time_shift_ns,
            emission_packet_time_ns=first_packet_time_ns + int(round(PRE_ROLL_SECONDS * 1e9)),
            detections=[],
            manifests=[],
        )
        deadline = time.monotonic() + DETECTION_WAIT_SECONDS
        while time.monotonic() < deadline:
            result.detections = _rust_handoff_detections(client, node_ids)
            reporting_nodes = {detection["source_node_id"] for detection in event_detections(result)}
            if reporting_nodes >= set(awaited_detection_node_ids):
                break
            time.sleep(POLL_INTERVAL_SECONDS)
        time.sleep(DETECTION_SETTLE_SECONDS)
        result.detections = _rust_handoff_detections(client, node_ids)
        result.manifests = [
            manifest
            for manifest in replay_dsp_event_manifests(sidecar_port)
            if manifest.get("manifest_type") in SOLVE_MANIFEST_TYPES
        ]
    return result


def event_detections(result: ReplayRunResult) -> list[dict]:
    """Stored detections whose classifier render overlaps the event."""
    event_start_ns, event_end_ns = result.event_arrival_span_ns(SOURCE_DURATION_SECONDS)
    return [
        detection
        for detection in result.detections
        if int(detection["feature_summary"].get("rust_render_start_ns") or 0) <= event_end_ns
        and int(detection["feature_summary"].get("rust_render_end_ns") or 0) >= event_start_ns
    ]


def node_of_sensor(sensor_id: str) -> str:
    return sensor_id.rsplit(":ch", 1)[0]


def run_replay_scenario(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    node_plans: Sequence[ReplayNodePlan],
    *,
    arrival_order: Sequence[str],
    step: PacketTimestampStep | PacketClockDrift | None = None,
    extra_env: dict[str, str] | None = None,
) -> ReplayRunResult:
    scene = build_replay_scene(node_plans)
    bundle_path = write_replay_bundle(
        tmp_path,
        scene=scene,
        node_plans=node_plans,
        captures=render_captures_at_node_rates(scene, node_plans),
        step=step,
    )
    return run_bundle_through_full_app(
        monkeypatch,
        tmp_path,
        bundle_path=bundle_path,
        scene=scene,
        node_order=arrival_order,
        awaited_detection_node_ids=arrival_order,
        extra_env=extra_env,
    )


def event_srp_solves(result: ReplayRunResult) -> list[dict]:
    """``localization_result`` manifests of event-triggered SRP solves."""
    solves = []
    for manifest in result.manifests:
        localization = manifest.get("localization") or {}
        provenance = localization.get("solve_provenance") or {}
        if (
            manifest.get("manifest_type") == "localization_result"
            and localization.get("resolved_algorithm") == "srp_phat"
            and provenance.get("attempt_kind") in EVENT_TRIGGER_ATTEMPT_KINDS
        ):
            solves.append(manifest)
    return solves


def pair_sensor_ids(manifest: dict) -> list[tuple[str, str]]:
    sensors = manifest["localization"]["solve_provenance"]["sensors"]
    return [
        (sensors[int(pair["ch_a"])]["sensor_id"], sensors[int(pair["ch_b"])]["sensor_id"])
        for pair in manifest["localization"]["pair_tdoas"]
    ]


def lag_errors_seconds(manifest: dict, result: ReplayRunResult) -> dict[tuple[str, str], float]:
    """Rust ``lag_seconds`` is arrival(a) - arrival(b); truth lag is arrival(b) - arrival(a)."""
    return {
        (sensor_a, sensor_b): float(pair["lag_seconds"]) + result.scene.arrival_lag_seconds(sensor_a, sensor_b)
        for (sensor_a, sensor_b), pair in zip(pair_sensor_ids(manifest), manifest["localization"]["pair_tdoas"])
    }


def solve_digest(manifest: dict) -> str:
    provenance = manifest["localization"]["solve_provenance"]
    return (
        f"reporter={provenance['reporting_node_id']} rate={provenance['solve_sample_rate_hz']} "
        f"active={[provenance['sensors'][index]['sensor_id'] for index in provenance['active_sensor_indices']]} "
        f"excluded={[(entry['sensor_id'], entry['reason']) for entry in provenance['excluded_sensors']]} "
        f"pairs={len(manifest['localization']['pair_tdoas'])}"
    )
