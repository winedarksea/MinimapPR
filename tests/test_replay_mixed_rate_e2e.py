"""Mixed-sample-rate clusters through the real sidecar and the full app (plan item E4).

No resampling this round (R3): a sensor whose rate differs from the reporting
node's is excluded from that node's solves with ``sample_rate_mismatch``, so a
mixed-rate cluster never forms a cross-node pair and each capable array must
localize on its own channels:

- planar 48 kHz + tetra 44.1 kHz: both arrays localize alone;
- planar 48 kHz + point 44.1 kHz: the planar localizes, the point node yields
  an omni classification only (SRP needs 4 channels).

Both arrival orders are run: the node whose upload reaches the sidecar first in
each 100 ms slot leads, the other follows. Bundles are replayed frame by frame
(``minimappr/sim/replay_frames.py``), so each node's frames carry its own rate.
"""

from __future__ import annotations

from math import comb
from pathlib import Path

import pytest

from tests.replay_realism_bundles import (
    PLANAR_NODE_ID,
    POINT_NODE_ID,
    SAMPLE_RATE_44_1_KHZ,
    SAMPLE_RATE_48_KHZ,
    TETRA_NODE_ID,
    planar_plan,
    point_plan,
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

SAMPLE_RATE_MISMATCH_REASON = "sample_rate_mismatch"
LOCALIZED_MODALITY = "localized"
OMNI_MODALITY = "omni"
LAG_TOLERANCE_SAMPLES = 0.5


def _cross_node_admissions(result: ReplayRunResult) -> list[str]:
    return [
        f"{solve_digest(manifest)} admits {pair}"
        for manifest in event_srp_solves(result)
        for pair in pair_sensor_ids(manifest)
        if node_of_sensor(pair[0]) != node_of_sensor(pair[1])
    ]


def array_solves_alone_violations(
    result: ReplayRunResult, *, node_id: str, sample_rate_hz: int, other_node_id: str
) -> list[str]:
    """``node_id`` reports an event solve on exactly its own channels at its own
    rate, the other node's sensors excluded for rate, the lags those of the
    synthesis; its stored event detections are labelled and never cross-node."""
    node = result.scene.node(node_id)
    own_sensor_ids = sorted(node.sensor_ids)
    other_sensor_ids = sorted(result.scene.node(other_node_id).sensor_ids)
    lag_tolerance_s = LAG_TOLERANCE_SAMPLES / sample_rate_hz
    reported = [
        manifest
        for manifest in event_srp_solves(result)
        if manifest["localization"]["solve_provenance"]["reporting_node_id"] == node_id
    ]
    violations: list[str] = []
    if not reported:
        violations.append(f"{node_id} reported no event-triggered SRP solve; manifests:\n{result.digest()}")
    matching = []
    for manifest in reported:
        provenance = manifest["localization"]["solve_provenance"]
        sensors = provenance["sensors"]
        active = sorted(sensors[index]["sensor_id"] for index in provenance["active_sensor_indices"])
        rate_excluded = sorted(
            entry["sensor_id"] for entry in provenance["excluded_sensors"] if entry["reason"] == SAMPLE_RATE_MISMATCH_REASON
        )
        listed_other = sorted(sensor["sensor_id"] for sensor in sensors if sensor["node_id"] == other_node_id)
        pairs = pair_sensor_ids(manifest)
        problems = []
        if provenance["solve_sample_rate_hz"] != sample_rate_hz:
            problems.append(f"solve rate {provenance['solve_sample_rate_hz']} != {sample_rate_hz}")
        if active != own_sensor_ids:
            problems.append(f"active {active} != own {own_sensor_ids}")
        if listed_other and rate_excluded != other_sensor_ids:
            problems.append(f"rate-excluded {rate_excluded} != {other_sensor_ids}")
        for entry in provenance["excluded_pairs"]:
            pair_ids = {sensors[entry["sensor_index_a"]]["sensor_id"], sensors[entry["sensor_index_b"]]["sensor_id"]}
            if entry["reason"] != SAMPLE_RATE_MISMATCH_REASON or not pair_ids & set(other_sensor_ids):
                problems.append(f"unexpected excluded pair {entry}")
        if len(pairs) != comb(len(own_sensor_ids), 2) or any(node_of_sensor(a) != node_id or node_of_sensor(b) != node_id for a, b in pairs):
            problems.append(f"pairs {pairs} are not the {comb(len(own_sensor_ids), 2)} own pairs")
        lag_errors = lag_errors_seconds(manifest, result)
        if any(abs(error_s) > lag_tolerance_s for error_s in lag_errors.values()):
            problems.append(f"lag errors (us) { {f'{a}|{b}': round(e * 1e6, 1) for (a, b), e in lag_errors.items()} }")
        if problems:
            violations.append(f"{node_id} solve {provenance['solve_id']}: " + "; ".join(problems))
        else:
            matching.append(manifest)
    if reported and not matching:
        violations.append(f"{node_id}: no event solve matches the solve-alone contract")
    violations.extend(_cross_node_admissions(result))

    stored = [detection for detection in event_detections(result) if detection["source_node_id"] == node_id]
    if not stored:
        violations.append(f"{node_id} stored no event detection ({len(result.detections)} detections in total)")
    for detection in stored:
        solve = detection["feature_summary"].get("localization_solve") or {}
        if detection["label"] != FAKE_CLASSIFIER_LABEL:
            violations.append(f"detection {detection['id']}: label {detection['label']!r}")
        if solve and (solve.get("cross_node_pair_count") != 0 or solve.get("contributing_node_ids") != [node_id]):
            violations.append(
                f"detection {detection['id']}: contributing nodes {solve.get('contributing_node_ids')}, "
                f"{solve.get('cross_node_pair_count')} cross-node pairs"
            )
    if not any(detection.get("reporting_modality") == LOCALIZED_MODALITY for detection in stored):
        violations.append(f"{node_id}: no stored event detection is localized ({[d.get('reporting_modality') for d in stored]})")
    return violations


def point_node_omni_only_violations(result: ReplayRunResult, *, point_node_id: str) -> list[str]:
    """The point node solves nothing (SRP needs 4 channels) and stores omni, labelled detections only."""
    violations = [
        f"point node reported an SRP solve: {solve_digest(manifest)}"
        for manifest in event_srp_solves(result)
        if manifest["localization"]["solve_provenance"]["reporting_node_id"] == point_node_id
        and manifest["localization"]["pair_tdoas"]
    ]
    stored = [detection for detection in result.detections if detection["source_node_id"] == point_node_id]
    if not [detection for detection in stored if detection in event_detections(result)]:
        violations.append(f"{point_node_id} stored no event detection; manifests:\n{result.digest()}")
    for detection in stored:
        if detection["label"] != FAKE_CLASSIFIER_LABEL or detection.get("reporting_modality") != OMNI_MODALITY:
            violations.append(
                f"point detection {detection['id']}: label {detection['label']!r}, modality {detection.get('reporting_modality')!r}"
            )
    return violations


ARRAY_ARRIVAL_ORDERS = {
    "planar_leads": (PLANAR_NODE_ID, TETRA_NODE_ID),
    "tetra_leads": (TETRA_NODE_ID, PLANAR_NODE_ID),
}
POINT_ARRIVAL_ORDERS = {
    "planar_leads": (PLANAR_NODE_ID, POINT_NODE_ID),
    "point_leads": (POINT_NODE_ID, PLANAR_NODE_ID),
}


@pytest.mark.parametrize("arrival_order_name", sorted(ARRAY_ARRIVAL_ORDERS))
def test_mixed_rate_planar_and_tetra_each_localize_alone(monkeypatch, tmp_path: Path, arrival_order_name: str) -> None:
    result = run_replay_scenario(
        monkeypatch,
        tmp_path,
        [planar_plan(SAMPLE_RATE_48_KHZ), tetra_plan(SAMPLE_RATE_44_1_KHZ)],
        arrival_order=ARRAY_ARRIVAL_ORDERS[arrival_order_name],
    )
    violations = array_solves_alone_violations(
        result, node_id=PLANAR_NODE_ID, sample_rate_hz=SAMPLE_RATE_48_KHZ, other_node_id=TETRA_NODE_ID
    ) + array_solves_alone_violations(
        result, node_id=TETRA_NODE_ID, sample_rate_hz=SAMPLE_RATE_44_1_KHZ, other_node_id=PLANAR_NODE_ID
    )
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("arrival_order_name", sorted(POINT_ARRIVAL_ORDERS))
def test_mixed_rate_planar_localizes_and_point_classifies_omni(monkeypatch, tmp_path: Path, arrival_order_name: str) -> None:
    result = run_replay_scenario(
        monkeypatch,
        tmp_path,
        [planar_plan(SAMPLE_RATE_48_KHZ), point_plan(SAMPLE_RATE_44_1_KHZ)],
        arrival_order=POINT_ARRIVAL_ORDERS[arrival_order_name],
    )
    violations = array_solves_alone_violations(
        result, node_id=PLANAR_NODE_ID, sample_rate_hz=SAMPLE_RATE_48_KHZ, other_node_id=POINT_NODE_ID
    ) + point_node_omni_only_violations(result, point_node_id=POINT_NODE_ID)
    assert not violations, "\n".join(violations)
