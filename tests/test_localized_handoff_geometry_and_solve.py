"""Handoff geometry, solve and provenance record from a solve's own provenance (plan item P2).

Built against manifests constructed per ``docs/localization_solve_contract.md``
(the sidecar emits ``solve_provenance`` from milestone 1 / R6).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from minimappr.api.rust_dsp_manifests import LocalizedClassifierRenderRequest
from minimappr.api.rust_localization_solve_contract import parse_rust_localization_solve
from minimappr.config import Settings
from minimappr.core.fusion_node import FusionMetrics
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization_provenance import (
    site_frame_id_for,
    site_frame_id_for_coordinate_frame,
    stored_localization_solve_record,
)
from minimappr.core.localized_handoff_geometry import (
    SOLVE_REJECTED_COORDINATE_FRAME_MISMATCH,
    SOLVE_REJECTED_NODE_BODY_SENSORS_OFF_REPORTING_NODE,
    SOLVE_REJECTED_SIDECAR_AUTOMATIC_FRAME,
    ManifestSolveGeometryResolver,
)
from minimappr.core.localized_handoff_solution import SnapshotGeometrySolutionResolver
from minimappr.core.localized_render_handoff_hooks import HandoffSolveGeometry
from minimappr.core.rust_tdoa_solve import solve_localization_from_rust_tdoas
from minimappr.models import GeoPoint, NodeSpec
from tests.localization_solve_manifest_fixtures import (
    MIXED_SAMPLE_RATE_HZ,
    MIXED_WINDOW_CENTER_TIME_NS,
    PLANAR_BODY_OFFSETS_M,
    PLANAR_NODE_ID,
    POINT_NODE_ID,
    mixed_cluster_sensor_positions_m,
    mixed_cluster_solve_localization_payload,
    planar_node_payload,
    point_node_payload,
)

SITE_FRAME_ID = "site-under-test"
SOURCE_POSITION_M = (44.0, 37.0, 6.0)
ARRIVAL_ORDERS = {
    "planar_first": (PLANAR_NODE_ID, POINT_NODE_ID),
    "point_first": (POINT_NODE_ID, PLANAR_NODE_ID),
}


# --- site frame id ----------------------------------------------------------------


def test_site_frame_id_is_stable_and_changes_with_origin_or_projection() -> None:
    origin = GeoPoint(lat=45.0, lon=-93.0, alt_m=250.0)
    frame_id = site_frame_id_for(origin, "flat_wgs84_series")

    assert frame_id == site_frame_id_for(GeoPoint(lat=45.0 + 1e-12, lon=-93.0, alt_m=250.0), "flat_wgs84_series")
    assert frame_id.startswith("site-")
    assert frame_id != site_frame_id_for(origin, "geodetic_ecef_enu")
    assert frame_id != site_frame_id_for(GeoPoint(lat=45.0001, lon=-93.0, alt_m=250.0), "flat_wgs84_series")
    assert frame_id != site_frame_id_for(GeoPoint(lat=45.0, lon=-93.0, alt_m=251.0), "flat_wgs84_series")
    assert site_frame_id_for_coordinate_frame(LocalCoordinateFrame(origin=origin, mode="flat")) == frame_id
    assert site_frame_id_for_coordinate_frame(LocalCoordinateFrame(origin=origin, mode="geodetic")) == site_frame_id_for(
        origin, "geodetic_ecef_enu"
    )
    with pytest.raises(ValueError):
        site_frame_id_for(origin, "sphere_6371km")


# --- helpers ------------------------------------------------------------------------


def _request(localization: dict, *, reporter_payload: dict) -> LocalizedClassifierRenderRequest:
    solve = parse_rust_localization_solve(localization)
    return LocalizedClassifierRenderRequest(
        manifest_id="m-1",
        node=NodeSpec.model_validate(reporter_payload),
        event_time_ns=MIXED_WINDOW_CENTER_TIME_NS,
        sample_rate_hz=MIXED_SAMPLE_RATE_HZ,
        decoded_audio=np.zeros(160, dtype=np.float32),
        localization_position_m=tuple(localization["position_m"]),
        localization_confidence=localization["confidence"],
        localization_position_covariance_m2=localization.get("position_covariance_m2"),
        localization_range_projection_mode=localization.get("range_projection_mode"),
        localization_pair_tdoas=localization.get("pair_tdoas"),
        localization_steering_direction=localization.get("steering_direction"),
        localization_sound_speed_mps=localization.get("sound_speed_mps"),
        localization_method="srp_phat",
        localization_solve=solve,
    )


def _registry_geometry_for(node: NodeSpec) -> HandoffSolveGeometry:
    """What the registry would say *now*: deliberately shifted from the solve snapshot."""
    sensor_ids = [f"{node.id}:ch{index}" for index in range(len(node.sensor_offsets_m))]
    return HandoffSolveGeometry(
        selected_sensor_ids=sensor_ids,
        selected_positions={
            sensor_id: np.asarray(node.position_m) + np.asarray(offset) + np.asarray([100.0, 0.0, 0.0])
            for sensor_id, offset in zip(sensor_ids, node.sensor_offsets_m)
        },
        reference_sensor=sensor_ids[0],
    )


def _resolve(request: LocalizedClassifierRenderRequest, frame_id: str = SITE_FRAME_ID) -> HandoffSolveGeometry:
    resolver = ManifestSolveGeometryResolver(site_frame_id_provider=lambda: frame_id)
    return resolver.resolve_handoff_geometry(
        payload=request, node=request.node, registry_geometry=_registry_geometry_for(request.node)
    )


def _solution_host(solver: str) -> SimpleNamespace:
    settings = Settings(localization_single_node_solver=solver)
    return SimpleNamespace(settings=settings, localization_config=settings.localization_config(), _metrics=FusionMetrics())


# --- geometry -------------------------------------------------------------------------


@pytest.mark.parametrize("arrival_order_name", sorted(ARRIVAL_ORDERS))
@pytest.mark.parametrize("reporter_payload", [planar_node_payload(), point_node_payload()], ids=["planar", "point"])
def test_geometry_is_the_solve_snapshot_in_solve_order(arrival_order_name: str, reporter_payload: dict) -> None:
    arrival_order = ARRIVAL_ORDERS[arrival_order_name]
    localization = mixed_cluster_solve_localization_payload(
        arrival_order=arrival_order,
        source_position_m=SOURCE_POSITION_M,
        frame_id=SITE_FRAME_ID,
        reporting_node_id=reporter_payload["id"],
    )
    request = _request(localization, reporter_payload=reporter_payload)
    geometry = _resolve(request)

    solve_order = [sensor["sensor_id"] for sensor in localization["solve_provenance"]["sensors"]]
    assert geometry.rejected_reason is None
    assert geometry.ordered_pair_sensor_ids() == solve_order
    assert geometry.selected_sensor_ids == solve_order  # all six contribute
    assert [sensor_id.split(":ch")[0] for sensor_id in solve_order][0] == arrival_order[0]
    snapshot = mixed_cluster_sensor_positions_m()
    for sensor_id in solve_order:
        np.testing.assert_allclose(geometry.selected_positions[sensor_id], snapshot[sensor_id])
    assert geometry.reference_sensor.startswith(f"{reporter_payload['id']}:")
    assert geometry.solve_frame_to_site is None


def test_geometry_selection_follows_contributing_sensors_only() -> None:
    localization = mixed_cluster_solve_localization_payload(
        arrival_order=ARRIVAL_ORDERS["planar_first"],
        source_position_m=SOURCE_POSITION_M,
        frame_id=SITE_FRAME_ID,
        reporting_node_id=POINT_NODE_ID,
    )
    # The point node's cross-node pairs were excluded (index 5 is the point).
    localization["pair_tdoas"] = [pair for pair in localization["pair_tdoas"] if 5 not in (pair["ch_a"], pair["ch_b"])]
    localization["solve_provenance"]["excluded_pairs"] = [
        {"sensor_index_a": index, "sensor_index_b": 5, "reason": "timeline_residual_over_limit"} for index in range(5)
    ]
    geometry = _resolve(_request(localization, reporter_payload=point_node_payload()))

    assert geometry.selected_sensor_ids == [f"{PLANAR_NODE_ID}:ch{index}" for index in range(5)]
    assert len(geometry.ordered_pair_sensor_ids()) == 6
    # No reporter sensor contributed, so the reference is the first contributor.
    assert geometry.reference_sensor == f"{PLANAR_NODE_ID}:ch0"


@pytest.mark.parametrize(
    ("frame_id", "frame_kind", "expected_reason"),
    [
        ("site-from-another-origin", "site_enu", SOLVE_REJECTED_COORDINATE_FRAME_MISMATCH),
        (None, "site_enu", SOLVE_REJECTED_COORDINATE_FRAME_MISMATCH),
        (None, "sidecar_automatic_enu", SOLVE_REJECTED_SIDECAR_AUTOMATIC_FRAME),
    ],
)
def test_geometry_outside_the_site_frame_is_rejected_with_a_reason(frame_id, frame_kind, expected_reason) -> None:
    localization = mixed_cluster_solve_localization_payload(
        arrival_order=ARRIVAL_ORDERS["planar_first"],
        source_position_m=SOURCE_POSITION_M,
        frame_id=frame_id,
        frame_kind=frame_kind,
        reporting_node_id=PLANAR_NODE_ID,
    )
    request = _request(localization, reporter_payload=planar_node_payload())
    geometry = _resolve(request)

    assert geometry.rejected_reason == expected_reason
    # The reporter's registry geometry is kept for the omni report.
    assert geometry.selected_sensor_ids == _registry_geometry_for(request.node).selected_sensor_ids


def _node_body_localization(*, reporter_id: str = PLANAR_NODE_ID, body_position_m=(0.0, 10.0, 0.0)) -> dict:
    localization = mixed_cluster_solve_localization_payload(
        arrival_order=(PLANAR_NODE_ID,),
        source_position_m=SOURCE_POSITION_M,
        frame_id=None,
        frame_kind="node_body",
        reporting_node_id=reporter_id,
        sidecar_position_m=body_position_m,
    )
    provenance = localization["solve_provenance"]
    provenance.update(buffer_scope="node_local", cluster_id=None)
    provenance["coordinate_frame"].update(origin=None, projection=None)
    for sensor, offset in zip(provenance["sensors"], PLANAR_BODY_OFFSETS_M):
        sensor.update(position_m=list(offset), position_source="node_body_offset")
    localization["position_covariance_m2"] = [[4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    localization["steering_direction"] = [0.0, 1.0, 0.0]
    return localization


def test_node_body_solve_is_placed_with_the_overridden_node_position_and_rotation() -> None:
    node_position_m = (30.0, 25.0, 1.5)
    request = _request(_node_body_localization(), reporter_payload=planar_node_payload(node_position_m, yaw_deg=90.0))
    geometry = _resolve(request)

    assert geometry.rejected_reason is None
    # orientation_rotation_matrix: yaw rotates about +Z, so body +x maps to site +y.
    np.testing.assert_allclose(geometry.selected_positions[f"{PLANAR_NODE_ID}:ch0"], (30.0, 25.05, 1.5), atol=1e-9)
    estimate = SnapshotGeometrySolutionResolver(_solution_host("rust")).resolve_handoff_solution(
        payload=request, geometry=geometry, reporting_modality="localized"
    )
    np.testing.assert_allclose(estimate.position_m, (20.0, 25.0, 1.5), atol=1e-9)  # body +y -> site -x
    np.testing.assert_allclose(np.diag(estimate.position_covariance_m2), (1.0, 4.0, 1.0), atol=1e-9)


def test_node_body_solve_naming_another_node_is_rejected() -> None:
    request = _request(_node_body_localization(reporter_id=PLANAR_NODE_ID), reporter_payload=point_node_payload())
    assert _resolve(request).rejected_reason == SOLVE_REJECTED_NODE_BODY_SENSORS_OFF_REPORTING_NODE


# --- solution -------------------------------------------------------------------------


@pytest.mark.parametrize("arrival_order_name", sorted(ARRIVAL_ORDERS))
@pytest.mark.parametrize("reporter_payload", [planar_node_payload(), point_node_payload()], ids=["planar", "point"])
def test_python_cartesian_resolve_uses_the_full_solve_order(arrival_order_name: str, reporter_payload: dict) -> None:
    """The old mapping read cluster pair indices against the reporter's own
    channels (one channel for the point node); the snapshot mapping must give
    the same, correct answer whichever node arrived first or reported."""
    localization = mixed_cluster_solve_localization_payload(
        arrival_order=ARRIVAL_ORDERS[arrival_order_name],
        source_position_m=SOURCE_POSITION_M,
        frame_id=SITE_FRAME_ID,
        reporting_node_id=reporter_payload["id"],
        sidecar_position_m=(0.0, 0.0, 0.0),
    )
    request = _request(localization, reporter_payload=reporter_payload)
    host = _solution_host("python_cartesian")
    estimate = SnapshotGeometrySolutionResolver(host).resolve_handoff_solution(
        payload=request, geometry=_resolve(request), reporting_modality="localized"
    )

    assert estimate.method == "python_cartesian_rust_tdoa"
    assert host._metrics.localization_single_node_python_solved_count == 1
    assert math.dist(estimate.position_m, SOURCE_POSITION_M) < 0.5


def test_rust_solver_adopts_the_sidecar_estimate_in_a_site_frame() -> None:
    localization = mixed_cluster_solve_localization_payload(
        arrival_order=ARRIVAL_ORDERS["point_first"],
        source_position_m=SOURCE_POSITION_M,
        frame_id=SITE_FRAME_ID,
        reporting_node_id=POINT_NODE_ID,
        sidecar_position_m=(44.2, 36.9, 6.1),
    )
    request = _request(localization, reporter_payload=point_node_payload())
    estimate = SnapshotGeometrySolutionResolver(_solution_host("rust")).resolve_handoff_solution(
        payload=request, geometry=_resolve(request), reporting_modality="localized"
    )
    assert tuple(estimate.position_m) == (44.2, 36.9, 6.1)


def test_solution_resolver_leaves_legacy_and_rejected_renders_to_the_default_path() -> None:
    localization = mixed_cluster_solve_localization_payload(
        arrival_order=ARRIVAL_ORDERS["planar_first"],
        source_position_m=SOURCE_POSITION_M,
        frame_id="other-site",
        reporting_node_id=PLANAR_NODE_ID,
    )
    request = _request(localization, reporter_payload=planar_node_payload())
    resolver = SnapshotGeometrySolutionResolver(_solution_host("python_cartesian"))
    assert resolver.resolve_handoff_solution(payload=request, geometry=_resolve(request), reporting_modality="omni") is None


# --- rust_tdoa_solve id map -----------------------------------------------------------


def _tetra_positions() -> dict[str, np.ndarray]:
    return {f"s{index}": np.asarray(offset) for index, offset in enumerate(PLANAR_BODY_OFFSETS_M[:4])}


def test_rust_tdoa_solve_refuses_an_out_of_range_pair_index() -> None:
    with pytest.raises(ValueError, match="outside the 4 ordered solve sensors"):
        solve_localization_from_rust_tdoas(
            sensor_positions=_tetra_positions(),
            ordered_sensor_ids=list(_tetra_positions()),
            pair_tdoas=[{"ch_a": 0, "ch_b": 5, "lag_seconds": 1e-4, "confidence": 0.9}],
            steering_direction=None,
            sound_speed_mps=343.0,
            sample_rate_hz=MIXED_SAMPLE_RATE_HZ,
            interpolation_factor=4,
            far_field_default_range_m=50.0,
        )


def test_rust_tdoa_solve_refuses_a_sensor_without_a_position() -> None:
    positions = _tetra_positions()
    del positions["s3"]
    with pytest.raises(ValueError, match="no position"):
        solve_localization_from_rust_tdoas(
            sensor_positions=positions,
            ordered_sensor_ids=["s0", "s1", "s2", "s3"],
            pair_tdoas=[{"ch_a": 0, "ch_b": 1, "lag_seconds": 1e-4, "confidence": 0.9}],
            steering_direction=None,
            sound_speed_mps=343.0,
            sample_rate_hz=MIXED_SAMPLE_RATE_HZ,
            interpolation_factor=4,
            far_field_default_range_m=50.0,
        )


# --- stored record --------------------------------------------------------------------


@pytest.mark.parametrize("arrival_order_name", sorted(ARRIVAL_ORDERS))
def test_stored_record_matches_the_e2e_contract(arrival_order_name: str) -> None:
    localization = mixed_cluster_solve_localization_payload(
        arrival_order=ARRIVAL_ORDERS[arrival_order_name],
        source_position_m=SOURCE_POSITION_M,
        frame_id=SITE_FRAME_ID,
        reporting_node_id=POINT_NODE_ID,
    )
    record = stored_localization_solve_record(parse_rust_localization_solve(localization), stored_position_source="solve")

    positions = mixed_cluster_sensor_positions_m()
    assert sorted(record["contributing_sensor_ids"]) == sorted(positions)
    assert record["contributing_node_ids"] == sorted([PLANAR_NODE_ID, POINT_NODE_ID])
    assert record["cluster_id"] == "auto-gps"
    assert record["reporting_node_id"] == POINT_NODE_ID
    assert record["admitted_pair_count"] == 15
    assert record["cross_node_pair_count"] == 5
    assert record["coordinate_frame_id"] == SITE_FRAME_ID
    assert record["window_center_time_ns"] == MIXED_WINDOW_CENTER_TIME_NS
    lags = record["pair_arrival_lag_seconds_by_sensor_pair"]
    assert len(lags) == 15
    source = np.asarray(SOURCE_POSITION_M)
    for key, lag_s in lags.items():
        sensor_a, sensor_b = key.split("|")
        assert sensor_a < sensor_b
        expected_s = (
            np.linalg.norm(source - np.asarray(positions[sensor_b])) - np.linalg.norm(source - np.asarray(positions[sensor_a]))
        ) / 343.0
        assert lag_s == pytest.approx(expected_s, abs=1e-12)
