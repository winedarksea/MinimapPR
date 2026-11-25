"""Replay realism through the real sidecar and the full app (plan item E4).

Every scenario is a synthetic calibration bundle (``tests/replay_realism_bundles.py``)
replayed frame by frame (``minimappr/sim/replay_frames.py`` +
``minimappr/sim/replay_sidecar.py``), so per-frame packet timing reaches the
sidecar exactly as written in the bundle:

1. control: planar + tetra, both 48 kHz, GPS-locked: the cross-node solve is
   admitted and stored (harness proof for scenario 2);
2. a 21 ms packet-timestamp step inside one binary upload on the tetra, no
   sequence or sample gap (milestone 2: R5 detects it, R7 gates the node's
   cross-node pairs);
3. mixed rate, planar 48 kHz + tetra 44.1 kHz: no cross-node pairs, each array
   localizes alone;
4. mixed rate, planar 48 kHz + point 44.1 kHz: the planar localizes, the point
   yields an omni classification only (SRP needs 4 channels).

Arrival order matters for the mixed-rate cases: the node posting first in each
100 ms slot is called the leader, the other the follower.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.replay_realism_bundles import (
    BUNDLE_CAPTURE_START_NS,
    FRAME_DURATION_SECONDS,
    FRAMES_PER_UPLOAD,
    PLANAR_NODE_ID,
    SAMPLE_RATE_48_KHZ,
    TETRA_NODE_ID,
    PacketTimestampStep,
    planar_plan,
    tetra_plan,
)
from tests.replay_realism_run import (
    ReplayRunResult,
    event_detections,
    event_srp_solves,
    lag_errors_seconds,
    node_of_sensor,
    pair_sensor_ids,
    run_replay_scenario,
    solve_digest,
)
from tests.rust_sidecar_e2e_helpers import FAKE_CLASSIFIER_LABEL

# --- Scenario 2: step inside one upload ---
TIMESTAMP_STEP_NS = 21_000_000
# Frame 22 is the third frame of the fifth upload (frames 20..24), so the step
# lands inside one upload, ~0.56 s before the event onset.
STEPPED_FRAME_INDEX = 22
CROSS_NODE_MAX_TIMELINE_RESIDUAL_NS = 10_000_000  # Settings default, 0.010 s
# The step is exact in packet time; anything within a millisecond is the same step.
STEP_SIZE_TOLERANCE_NS = 1_000_000

# --- Pass bounds ---
CROSS_NODE_POSITION_ERROR_BOUND_M = 0.5
PLANAR_SENSOR_COUNT = 5
TETRA_SENSOR_COUNT = 4
PLANAR_OWN_PAIR_COUNT = 10
TETRA_OWN_PAIR_COUNT = 6
CROSS_NODE_PAIR_COUNT = PLANAR_SENSOR_COUNT * TETRA_SENSOR_COUNT
LOCALIZED_MODALITY = "localized"


class TimestampStepContractViolation(AssertionError):
    """Raised only by the timestamp-step contract check, so a contract failure
    reads apart from a harness failure (rejected upload, app start-up)."""


# --- Scenario 1: control ---


def test_gps_locked_planar_and_tetra_store_an_admitted_cross_node_fix(monkeypatch, tmp_path: Path) -> None:
    result = run_replay_scenario(
        monkeypatch,
        tmp_path,
        [planar_plan(SAMPLE_RATE_48_KHZ), tetra_plan(SAMPLE_RATE_48_KHZ)],
        arrival_order=(PLANAR_NODE_ID, TETRA_NODE_ID),
    )
    all_sensor_ids = sorted(sensor_id for node in result.scene.nodes for sensor_id in node.sensor_ids)
    full_cluster_solves = [
        manifest
        for manifest in event_srp_solves(result)
        if sum(node_of_sensor(a) != node_of_sensor(b) for a, b in pair_sensor_ids(manifest)) == CROSS_NODE_PAIR_COUNT
        and len(manifest["localization"]["pair_tdoas"])
        == PLANAR_OWN_PAIR_COUNT + TETRA_OWN_PAIR_COUNT + CROSS_NODE_PAIR_COUNT
    ]
    assert full_cluster_solves, result.digest()
    lag_tolerance_s = 0.5 / SAMPLE_RATE_48_KHZ
    assert any(
        all(abs(error_s) <= lag_tolerance_s for error_s in lag_errors_seconds(manifest, result).values())
        for manifest in full_cluster_solves
    ), [solve_digest(manifest) for manifest in full_cluster_solves]

    localized = [
        detection for detection in event_detections(result) if detection.get("reporting_modality") == LOCALIZED_MODALITY
    ]
    assert localized, result.detections
    for detection in localized:
        assert detection["label"] == FAKE_CLASSIFIER_LABEL
        solve = detection["feature_summary"]["localization_solve"]
        assert sorted(solve["contributing_sensor_ids"]) == all_sensor_ids
        assert solve["cross_node_pair_count"] == CROSS_NODE_PAIR_COUNT
        error_m = float(np.linalg.norm(np.asarray(detection["position_m"]) - np.asarray(result.scene.source_position_enu_m)))
        assert error_m < CROSS_NODE_POSITION_ERROR_BOUND_M, (error_m, detection["position_m"])


# --- Scenario 2: timestamp step inside one upload ---


def timestamp_step_contract_violations(result: ReplayRunResult, step: PacketTimestampStep) -> list[str]:
    """Every way the post-step solves and stored detections miss the
    milestone-2 clock-evidence contract (empty = pass)."""
    stepped_node_id = step.node_id
    step_packet_time_ns = (
        BUNDLE_CAPTURE_START_NS
        + result.packet_time_shift_ns
        + round(step.first_stepped_frame_index * FRAME_DURATION_SECONDS * 1e9)
    )
    post_step_cluster_solves = [
        manifest
        for manifest in event_srp_solves(result)
        if int(manifest["localization"]["solve_provenance"]["window_center_time_ns"]) > step_packet_time_ns
        and {sensor["node_id"] for sensor in manifest["localization"]["solve_provenance"]["sensors"]}
        == {PLANAR_NODE_ID, TETRA_NODE_ID}
    ]
    if not post_step_cluster_solves:
        return [f"no event-triggered cluster solve after the step; manifests:\n{result.digest()}"]

    violations: list[str] = []
    for manifest in post_step_cluster_solves:
        provenance = manifest["localization"]["solve_provenance"]
        prefix = f"solve {provenance['solve_id']} ({solve_digest(manifest)})"
        timing_by_node = {entry["node_id"]: entry for entry in provenance.get("node_timing") or []}
        stepped_timing = timing_by_node.get(stepped_node_id)
        if stepped_timing is None:
            violations.append(f"{prefix}: node_timing has no entry for {stepped_node_id} (node_timing={provenance.get('node_timing')})")
        else:
            if int(stepped_timing.get("timestamp_step_count") or 0) < 1:
                violations.append(f"{prefix}: {stepped_node_id} timestamp_step_count {stepped_timing.get('timestamp_step_count')} < 1")
            last_step_ns = stepped_timing.get("last_timestamp_step_ns")
            if last_step_ns is None or abs(int(last_step_ns) - step.step_ns) > STEP_SIZE_TOLERANCE_NS:
                violations.append(f"{prefix}: {stepped_node_id} last_timestamp_step_ns {last_step_ns} != {step.step_ns}")
            residual_ns = stepped_timing.get("timeline_residual_ns")
            if residual_ns is None or abs(int(residual_ns)) <= CROSS_NODE_MAX_TIMELINE_RESIDUAL_NS:
                violations.append(f"{prefix}: {stepped_node_id} timeline_residual_ns {residual_ns} not over the 10 ms limit")
        steady_timing = timing_by_node.get(PLANAR_NODE_ID)
        if steady_timing is None or int(steady_timing.get("timestamp_step_count") or 0) != 0:
            violations.append(f"{prefix}: {PLANAR_NODE_ID} node_timing {steady_timing} should show no step")

        sensors = provenance["sensors"]
        active_ids = {sensors[index]["sensor_id"] for index in provenance["active_sensor_indices"]}
        expected_excluded = {
            frozenset((a, b))
            for a in active_ids
            for b in active_ids
            if node_of_sensor(a) == stepped_node_id and node_of_sensor(b) != stepped_node_id
        }
        excluded_for_residual = {
            frozenset((sensors[entry["sensor_index_a"]]["sensor_id"], sensors[entry["sensor_index_b"]]["sensor_id"]))
            for entry in provenance.get("excluded_pairs") or []
            if entry.get("reason") == "timeline_residual_over_limit"
        }
        if excluded_for_residual != expected_excluded:
            violations.append(
                f"{prefix}: {len(excluded_for_residual)} pairs excluded for timeline_residual_over_limit, "
                f"expected all {len(expected_excluded)} cross-node pairs of {stepped_node_id}"
            )
        admitted = pair_sensor_ids(manifest)
        cross_node_admitted = [pair for pair in admitted if node_of_sensor(pair[0]) != node_of_sensor(pair[1])]
        if cross_node_admitted:
            violations.append(f"{prefix}: cross-node pairs still admitted: {cross_node_admitted[:4]}...")
        stepped_own_pairs = [pair for pair in admitted if node_of_sensor(pair[0]) == node_of_sensor(pair[1]) == stepped_node_id]
        if len(stepped_own_pairs) != TETRA_OWN_PAIR_COUNT:
            violations.append(f"{prefix}: {len(stepped_own_pairs)} own pairs of {stepped_node_id} admitted, expected {TETRA_OWN_PAIR_COUNT}")

    stored = event_detections(result)
    if not stored:
        violations.append(f"no stored event detection; manifests:\n{result.digest()}")
    for detection in stored:
        solve = (detection.get("feature_summary") or {}).get("localization_solve") or {}
        if int(solve.get("cross_node_pair_count") or 0) != 0:
            violations.append(
                f"detection {detection['id']} stored a cross-node fix ({solve.get('cross_node_pair_count')} cross-node pairs, "
                f"position {detection.get('position_m')}) from a node whose timeline is known wrong"
            )
    return violations


def test_timestamp_step_inside_one_upload_gates_the_stepped_nodes_cross_node_pairs(monkeypatch, tmp_path: Path) -> None:
    step = PacketTimestampStep(node_id=TETRA_NODE_ID, first_stepped_frame_index=STEPPED_FRAME_INDEX, step_ns=TIMESTAMP_STEP_NS)
    assert STEPPED_FRAME_INDEX % FRAMES_PER_UPLOAD != 0, "the step must fall inside an upload, not on its first frame"
    result = run_replay_scenario(
        monkeypatch,
        tmp_path,
        [planar_plan(SAMPLE_RATE_48_KHZ), tetra_plan(SAMPLE_RATE_48_KHZ)],
        arrival_order=(PLANAR_NODE_ID, TETRA_NODE_ID),
        step=step,
    )
    violations = timestamp_step_contract_violations(result, step)
    if violations:
        raise TimestampStepContractViolation("\n".join(violations))
