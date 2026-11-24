"""Deterministic acceptance: a mixed planar + point cluster through the real
sidecar and the full app, down to the stored detection (plan item E2).

Scene: a 5-channel planar array yawed 35 degrees, placed ~40 m from the site
origin, and a 1-channel point node on a 4 m pole beside it, both on the
automatic GPS geometry path (positions only from packet GPS; the planar
orientation is an operator override). One broadband source, a fake sidecar
classifier for a deterministic label, and packet timestamps set seconds apart
from receipt time.

Geometry note: five coplanar sensors plus one point leave a mirror solution
about the planar plane unless the point's range difference cannot be met on
the mirrored ray. The placement in ``tests/mixed_cluster_acceptance_scene.py``
is chosen so it cannot, so the six-sensor solve is unique without any
half-space prior.

Parametrised over both node arrival orders and both
``localization_single_node_solver`` values, as a strict xfail until
localization overhaul milestone 1 (see the xfail reason). The non-xfail tests
prove the harness: the truth is solvable inside the bound, and a lone planar
node produces stored detections and sidecar lags that match the synthesis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from minimappr.config import Settings
from minimappr.core.rust_tdoa_solve import solve_localization_from_rust_tdoas
from tests.acoustic_scene_synthesis import AcousticScene
from tests.mixed_cluster_acceptance_scene import (
    LOCALIZATION_WINDOW_SECONDS,
    PACKET_TIME_BEHIND_RECEIPT_NS,
    PLANAR_NODE_ID,
    POINT_NODE_ID,
    SAMPLE_RATE_HZ,
    SceneRunResult,
    build_scene,
    event_localization_manifests,
    manifest_digest,
    measured_rust_lags_by_sensor_pair,
    run_scene_through_full_app,
)
from tests.rust_sidecar_e2e_helpers import FAKE_CLASSIFIER_LABEL

# --- Stored-detection localization contract (written by milestone-1 item P2) ---
STORED_LOCALIZATION_SOLVE_FEATURE_KEY = "localization_solve"
"""``detection["feature_summary"]["localization_solve"]`` is a dict describing the
one solve the stored position came from. Field names below are the contract:

- ``solve_id``: ``localization.solve_provenance.solve_id`` of the solve used.
- ``contributing_sensor_ids``: list[str], the sensors whose audio entered the
  admitted pairs, in solve (buffer-channel) order.
- ``contributing_node_ids``: list[str], sorted unique nodes of those sensors.
- ``cluster_id`` / ``reporting_node_id``: kept separate (reporter is the node
  whose manifest delivered the render; cluster is the shared buffer scope).
- ``admitted_pair_count`` / ``cross_node_pair_count``: counts of admitted pairs.
- ``pair_arrival_lag_seconds_by_sensor_pair``: {"<a>|<b>": seconds} for every
  admitted pair with ``a < b`` lexicographically; value is
  ``arrival_time(b) - arrival_time(a)`` (Rust ``lag_seconds`` is
  ``arrival(ch_a) - arrival(ch_b)``, so re-key and negate as needed).
- ``coordinate_frame_id``: ``solve_provenance.coordinate_frame.frame_id``.
- ``window_center_time_ns``: packet-derived window centre of that solve; the
  detection's top-level ``timestamp_ns`` must equal it exactly. The solve is
  the event-triggered solve the classifier render covers (a render repeats its
  ``solve_id``), so the centre lies within the event's packet-time arrival
  span widened by half a localization window.

Only detections whose classifier render (``rust_render_start_ns`` ..
``rust_render_end_ns``) overlaps the event are judged; renders of pre-roll
noise are not detections of this event.
"""
STORED_SOLVE_ID_FIELD = "solve_id"
STORED_CONTRIBUTING_SENSOR_IDS_FIELD = "contributing_sensor_ids"
STORED_CONTRIBUTING_NODE_IDS_FIELD = "contributing_node_ids"
STORED_CLUSTER_ID_FIELD = "cluster_id"
STORED_REPORTING_NODE_ID_FIELD = "reporting_node_id"
STORED_ADMITTED_PAIR_COUNT_FIELD = "admitted_pair_count"
STORED_CROSS_NODE_PAIR_COUNT_FIELD = "cross_node_pair_count"
STORED_PAIR_LAGS_FIELD = "pair_arrival_lag_seconds_by_sensor_pair"
STORED_COORDINATE_FRAME_ID_FIELD = "coordinate_frame_id"
STORED_WINDOW_CENTER_TIME_NS_FIELD = "window_center_time_ns"
STORED_PAIR_KEY_SEPARATOR = "|"
MANIFEST_SOLVE_PROVENANCE_FIELD = "solve_provenance"

# --- Pass bounds ---
POSITION_ERROR_BOUND_M = 0.5
LAG_TOLERANCE_SECONDS = 0.5 / SAMPLE_RATE_HZ
EXPECTED_ADMITTED_PAIR_COUNT = 15
EXPECTED_CROSS_NODE_PAIR_COUNT = 5
# Any confident peak; the oracle solve only needs equal pair weights.
TRUTH_PAIR_CONFIDENCE = 0.9


def _stored_pair_key(sensor_a: str, sensor_b: str) -> str:
    return f"{sensor_a}{STORED_PAIR_KEY_SEPARATOR}{sensor_b}"


def _solve_provenance_by_solve_id(manifests_by_id: dict[str, dict]) -> dict[str, dict]:
    provenance_by_solve_id: dict[str, dict] = {}
    for manifest in manifests_by_id.values():
        provenance = (manifest.get("localization") or {}).get(MANIFEST_SOLVE_PROVENANCE_FIELD)
        if isinstance(provenance, dict) and provenance.get("solve_id"):
            provenance_by_solve_id[str(provenance["solve_id"])] = provenance
    return provenance_by_solve_id


def stored_detection_contract_violations(result: SceneRunResult) -> list[str]:
    """Every way the stored detections miss the acceptance contract (empty = pass)."""
    scene = result.scene
    expected_sensor_ids = sorted(sensor_id for node in scene.nodes for sensor_id in node.sensor_ids)
    expected_node_ids = sorted(node.node_id for node in scene.nodes)
    expected_lags = scene.expected_lags_by_sorted_sensor_pair(expected_sensor_ids)
    half_window_ns = int(LOCALIZATION_WINDOW_SECONDS * 1e9 / 2)
    event_start_ns, event_end_ns = result.event_arrival_span_ns()
    event_span_ns = (event_start_ns - half_window_ns, event_end_ns + half_window_ns)
    provenance_by_solve_id = _solve_provenance_by_solve_id(result.manifests_by_id)
    event_detections = result.event_detections()
    if not event_detections:
        return [
            f"no stored detection covers the event ({len(result.detections)} Rust-handoff detections in total); "
            f"manifests seen: {manifest_digest(result.manifests_by_id)}"
        ]

    violations: list[str] = []
    for detection in event_detections:
        prefix = f"detection {detection['id']} (reporter {detection.get('source_node_id')})"
        feature_summary = detection.get("feature_summary") or {}
        if detection.get("label") != FAKE_CLASSIFIER_LABEL:
            violations.append(f"{prefix}: label {detection.get('label')!r} != {FAKE_CLASSIFIER_LABEL!r}")
        if detection.get("reporting_modality") != "localized":
            violations.append(
                f"{prefix}: modality {detection.get('reporting_modality')!r}, method "
                f"{feature_summary.get('localization_method')!r}, rejected "
                f"{feature_summary.get('localization_rejected_reason')!r}, rust fallback "
                f"{feature_summary.get('rust_fallback_reason')!r}"
            )
        position_error_m = float(np.linalg.norm(np.asarray(detection["position_m"]) - np.asarray(scene.source_position_enu_m)))
        if position_error_m > POSITION_ERROR_BOUND_M:
            violations.append(
                f"{prefix}: position error {position_error_m:.2f} m > {POSITION_ERROR_BOUND_M} m "
                f"(stored {np.round(detection['position_m'], 2).tolist()}, truth {np.round(scene.source_position_enu_m, 2).tolist()}, "
                f"method {feature_summary.get('localization_method')!r})"
            )
        stored_time_ns = int(detection["timestamp_ns"])
        if not event_span_ns[0] <= stored_time_ns <= event_span_ns[1]:
            violations.append(
                f"{prefix}: stored timestamp_ns is {(stored_time_ns - event_span_ns[0]) / 1e9:+.3f} s from the event's "
                f"packet-time onset (outside the arrival span; receipt time is "
                f"{PACKET_TIME_BEHIND_RECEIPT_NS / 1e9:.3f} s later than packet time)"
            )
        solve = feature_summary.get(STORED_LOCALIZATION_SOLVE_FEATURE_KEY)
        if not isinstance(solve, dict):
            violations.append(
                f"{prefix}: feature_summary[{STORED_LOCALIZATION_SOLVE_FEATURE_KEY!r}] missing; source_sensors="
                f"{detection.get('source_sensors')}, tdoa_s keys={sorted(detection.get('tdoa_s') or {})}"
            )
            continue
        violations.extend(
            f"{prefix}: {message}"
            for message in _solve_block_violations(
                solve,
                stored_time_ns=stored_time_ns,
                expected_sensor_ids=expected_sensor_ids,
                expected_node_ids=expected_node_ids,
                expected_lags=expected_lags,
                provenance_by_solve_id=provenance_by_solve_id,
            )
        )
    return violations


def _solve_block_violations(
    solve: dict,
    *,
    stored_time_ns: int,
    expected_sensor_ids: list[str],
    expected_node_ids: list[str],
    expected_lags: dict[tuple[str, str], float],
    provenance_by_solve_id: dict[str, dict],
) -> list[str]:
    violations: list[str] = []
    if sorted(solve.get(STORED_CONTRIBUTING_SENSOR_IDS_FIELD) or []) != expected_sensor_ids:
        violations.append(f"contributing sensors {solve.get(STORED_CONTRIBUTING_SENSOR_IDS_FIELD)} != {expected_sensor_ids}")
    if solve.get(STORED_CONTRIBUTING_NODE_IDS_FIELD) != expected_node_ids:
        violations.append(f"contributing nodes {solve.get(STORED_CONTRIBUTING_NODE_IDS_FIELD)} != {expected_node_ids}")
    if not solve.get(STORED_CLUSTER_ID_FIELD) or not solve.get(STORED_REPORTING_NODE_ID_FIELD):
        violations.append("cluster_id and reporting_node_id must both be recorded")
    if solve.get(STORED_ADMITTED_PAIR_COUNT_FIELD) != EXPECTED_ADMITTED_PAIR_COUNT:
        violations.append(f"admitted pairs {solve.get(STORED_ADMITTED_PAIR_COUNT_FIELD)} != {EXPECTED_ADMITTED_PAIR_COUNT}")
    if solve.get(STORED_CROSS_NODE_PAIR_COUNT_FIELD) != EXPECTED_CROSS_NODE_PAIR_COUNT:
        violations.append(f"cross-node pairs {solve.get(STORED_CROSS_NODE_PAIR_COUNT_FIELD)} != {EXPECTED_CROSS_NODE_PAIR_COUNT}")
    stored_lags = solve.get(STORED_PAIR_LAGS_FIELD) or {}
    expected_keys = {_stored_pair_key(a, b) for a, b in expected_lags}
    if set(stored_lags) != expected_keys:
        violations.append(f"lag pairs {sorted(stored_lags)} != {sorted(expected_keys)}")
    for (sensor_a, sensor_b), expected_lag_s in expected_lags.items():
        stored_lag_s = stored_lags.get(_stored_pair_key(sensor_a, sensor_b))
        if stored_lag_s is not None and abs(float(stored_lag_s) - expected_lag_s) > LAG_TOLERANCE_SECONDS:
            violations.append(
                f"lag {sensor_a}->{sensor_b} {float(stored_lag_s) * 1e6:+.1f} us != truth {expected_lag_s * 1e6:+.1f} us"
            )
    if solve.get(STORED_WINDOW_CENTER_TIME_NS_FIELD) != stored_time_ns:
        violations.append(f"timestamp_ns {stored_time_ns} != stored window centre {solve.get(STORED_WINDOW_CENTER_TIME_NS_FIELD)}")
    provenance = provenance_by_solve_id.get(str(solve.get(STORED_SOLVE_ID_FIELD)))
    if provenance is None:
        violations.append(f"no sidecar manifest carries solve_provenance.solve_id {solve.get(STORED_SOLVE_ID_FIELD)!r}")
    else:
        if provenance.get("window_center_time_ns") != stored_time_ns:
            violations.append(f"timestamp_ns {stored_time_ns} != manifest window centre {provenance.get('window_center_time_ns')}")
        manifest_frame_id = (provenance.get("coordinate_frame") or {}).get("frame_id")
        if not manifest_frame_id or solve.get(STORED_COORDINATE_FRAME_ID_FIELD) != manifest_frame_id:
            violations.append(f"frame id {solve.get(STORED_COORDINATE_FRAME_ID_FIELD)!r} != manifest {manifest_frame_id!r}")
    return violations


def _rust_pair_tdoas_from_truth(scene: AcousticScene, ordered_sensor_ids: list[str], lag_quantum_s: float) -> list[dict]:
    """Pair TDOAs in the Rust wire convention (lag = arrival(ch_a) - arrival(ch_b)), quantised."""
    return [
        {
            "ch_a": index_a,
            "ch_b": index_b,
            "lag_seconds": lag_quantum_s * round(-scene.arrival_lag_seconds(sensor_a, sensor_b) / lag_quantum_s),
            "confidence": TRUTH_PAIR_CONFIDENCE,
        }
        for index_a, sensor_a in enumerate(ordered_sensor_ids)
        for index_b, sensor_b in enumerate(ordered_sensor_ids)
        if index_b > index_a
    ]


def test_mixed_scene_truth_is_solvable_within_the_position_bound() -> None:
    """Guards the bound: exact-geometry lags at the solver's lag resolution must
    solve well inside POSITION_ERROR_BOUND_M, so a failure of the e2e is the
    pipeline's, not an unreachable bound."""
    defaults = Settings()
    scene = build_scene(include_point_node=True)
    ordered_sensor_ids = [sensor_id for node in scene.nodes for sensor_id in node.sensor_ids]
    lag_quantum_s = 1.0 / (SAMPLE_RATE_HZ * defaults.gcc_phat_interp_factor)
    # No steering prior: the six-sensor geometry alone must exclude the mirror.
    solved = solve_localization_from_rust_tdoas(
        sensor_positions=scene.sensor_positions_enu_m(),
        ordered_sensor_ids=ordered_sensor_ids,
        pair_tdoas=_rust_pair_tdoas_from_truth(scene, ordered_sensor_ids, lag_quantum_s),
        steering_direction=None,
        sound_speed_mps=scene.sound_speed_mps,
        sample_rate_hz=SAMPLE_RATE_HZ,
        interpolation_factor=defaults.gcc_phat_interp_factor,
        far_field_default_range_m=defaults.localization_far_field_default_range_m,
    )
    assert solved is not None
    error_m = float(np.linalg.norm(np.asarray(solved.position_m) - np.asarray(scene.source_position_enu_m)))
    assert error_m < 0.5 * POSITION_ERROR_BOUND_M, (error_m, solved.position_m, scene.source_position_enu_m)


def test_lone_rotated_planar_node_localizes_and_stores_a_detection(monkeypatch, tmp_path: Path) -> None:
    """Harness proof for the xfail below: the same encoder, scene synthesis,
    packet timing and app wiring give a stored, labelled event detection for
    the rotated planar node alone, and the sidecar's trigger solve measures the
    lags of the synthesized rotated geometry. (Its stored modality is not
    asserted: which scheduled render covers the event, and so how much
    pre-roll noise its solve holds, varies run to run today.)"""
    scene = build_scene(include_point_node=False)
    result = run_scene_through_full_app(
        monkeypatch,
        tmp_path,
        scene=scene,
        arrival_order=(PLANAR_NODE_ID,),
        localization_solver="python_cartesian",
    )

    event_detections = result.event_detections()
    assert event_detections, f"no stored event detection; manifests: {manifest_digest(result.manifests_by_id)}"
    assert {detection["label"] for detection in event_detections} == {FAKE_CLASSIFIER_LABEL}
    assert {detection["source_node_id"] for detection in event_detections} == {PLANAR_NODE_ID}
    event_localizations = event_localization_manifests(result)
    assert event_localizations, "\n".join(manifest_digest(result.manifests_by_id))
    # Rust lag_seconds = arrival(a) - arrival(b). At least one event solve (the
    # trigger window) must match the synthesis; noisier render windows may not.
    lag_errors_by_solve = [
        {
            pair: measured_lag_s + scene.arrival_lag_seconds(*pair)
            for pair, measured_lag_s in measured_rust_lags_by_sensor_pair(manifest, PLANAR_NODE_ID).items()
        }
        for manifest in event_localizations
    ]
    assert any(
        len(lag_errors) == 10 and all(abs(error_s) <= LAG_TOLERANCE_SECONDS for error_s in lag_errors.values())
        for lag_errors in lag_errors_by_solve
    ), [{f"{a}|{b}": round(error_s * 1e6, 1) for (a, b), error_s in errors.items()} for errors in lag_errors_by_solve]


ARRIVAL_ORDERS = {
    "planar_first": (PLANAR_NODE_ID, POINT_NODE_ID),
    "point_first": (POINT_NODE_ID, PLANAR_NODE_ID),
}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "localization overhaul milestone 1: Rust drops the planar node from the mixed auto-GPS cluster, "
        "Python maps cluster pair indices onto the reporter's channels, stored event time is not the "
        "packet-derived window centre, geometry ignores the planar orientation and site origin, and the "
        "stored-detection localization_solve fields do not exist yet"
    ),
)
@pytest.mark.parametrize("localization_solver", ["rust", "python_cartesian"])
@pytest.mark.parametrize("arrival_order_name", sorted(ARRIVAL_ORDERS))
def test_mixed_planar_and_point_cluster_stores_an_exact_detection(
    monkeypatch, tmp_path: Path, arrival_order_name: str, localization_solver: str
) -> None:
    result = run_scene_through_full_app(
        monkeypatch,
        tmp_path,
        scene=build_scene(include_point_node=True),
        arrival_order=ARRIVAL_ORDERS[arrival_order_name],
        localization_solver=localization_solver,
    )
    violations = stored_detection_contract_violations(result)
    assert not violations, "\n".join(violations)
