"""Site-geometry snapshot (P3): overrides + stationary estimates + orientation."""

from __future__ import annotations

import math

import pytest

from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.localization_provenance import site_frame_id_for
from minimappr.models import GeoPoint, NodeOrientation
from minimappr.sidecar_site_geometry import (
    DEFAULT_OPERATOR_PLACEMENT_POSITION_STD_M,
    PLACEMENT_SOURCE_OPERATOR_OVERRIDE,
    PLACEMENT_SOURCE_STATIONARY_ESTIMATE,
    build_site_geometry_snapshot,
)
from tests.site_geometry_fixtures import (
    SITE_ORIGIN,
    estimator_checkpoint_row,
    node_row,
)

FLAT_FRAME = LocalCoordinateFrame(origin=SITE_ORIGIN, mode="flat")


def _snapshot(node_rows, estimator_states=None, frame=FLAT_FRAME):
    return build_site_geometry_snapshot(
        node_rows=node_rows,
        estimator_states_by_node_id=estimator_states or {},
        coordinate_frame=frame,
    )


def test_operator_override_uses_entered_accuracy_and_orientation_quaternion() -> None:
    snapshot = _snapshot(
        [
            node_row(
                "planar-a",
                overrides={"position_m": [1.0, 2.0, 1.5], "position_std_m": 0.05, "orientation": {"yaw_deg": 90.0}},
                orientation={"yaw_deg": 90.0},
            )
        ]
    )
    body = snapshot.to_push_body(geometry_push_revision=3)

    placement = body["node_placements"]["planar-a"]
    assert placement["position_m"] == [1.0, 2.0, 1.5]
    assert placement["position_std_m"] == 0.05
    assert placement["placement_source"] == PLACEMENT_SOURCE_OPERATOR_OVERRIDE
    assert placement["body_to_world_quaternion_wxyz"] == pytest.approx(
        [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
    )


def test_override_without_stated_accuracy_uses_the_documented_default() -> None:
    snapshot = _snapshot([node_row("n", overrides={"position_m": [0.0, 0.0, 0.0]})])
    assert snapshot.node_placements["n"].position_std_m == DEFAULT_OPERATOR_PLACEMENT_POSITION_STD_M


def test_geo_only_override_is_projected_into_the_site_frame() -> None:
    entered = GeoPoint(lat=SITE_ORIGIN.lat + 0.0001, lon=SITE_ORIGIN.lon + 0.0002, alt_m=SITE_ORIGIN.alt_m + 3.0)
    snapshot = _snapshot([node_row("geo-pinned", overrides={"position_geo": entered.model_dump(mode="json")})])

    assert snapshot.node_placements["geo-pinned"].position_m == pytest.approx(FLAT_FRAME.geo_to_local(entered))


def test_stationary_estimate_comes_from_the_checkpoint_not_the_reported_gps() -> None:
    estimate_geo = GeoPoint(lat=SITE_ORIGIN.lat + 0.0003, lon=SITE_ORIGIN.lon - 0.0001, alt_m=SITE_ORIGIN.alt_m + 1.0)
    snapshot = _snapshot(
        [node_row("gps-node", reported_position_m=[999.0, 999.0, 0.0])],
        {"gps-node": estimator_checkpoint_row(estimate_geo, horizontal_std_m=0.7)},
    )

    placement = snapshot.node_placements["gps-node"]
    assert placement.placement_source == PLACEMENT_SOURCE_STATIONARY_ESTIMATE
    assert placement.position_std_m == 0.7
    assert placement.position_m == pytest.approx(FLAT_FRAME.geo_to_local(estimate_geo), abs=1e-6)
    assert placement.body_to_world_quaternion_wxyz is None


def test_operator_override_wins_over_a_stationary_estimate() -> None:
    snapshot = _snapshot(
        [node_row("both", overrides={"position_m": [5.0, 6.0, 7.0]})],
        {"both": estimator_checkpoint_row(SITE_ORIGIN, horizontal_std_m=0.3)},
    )
    assert snapshot.node_placements["both"].placement_source == PLACEMENT_SOURCE_OPERATOR_OVERRIDE
    assert snapshot.node_placements["both"].position_m == (5.0, 6.0, 7.0)


@pytest.mark.parametrize(
    "row_overrides",
    [
        {"mobility": "mobile"},  # stale KDE checkpoint of a node now moving
        {"position_filter": "kalman"},
    ],
)
def test_estimates_of_non_kde_or_mobile_nodes_are_not_solve_geometry(row_overrides) -> None:
    snapshot = _snapshot(
        [node_row("moving", **row_overrides)],
        {"moving": estimator_checkpoint_row(SITE_ORIGIN, horizontal_std_m=0.3)},
    )
    assert "moving" not in snapshot.node_placements


def test_warming_or_missing_estimates_leave_the_node_on_raw_gps() -> None:
    snapshot = _snapshot(
        [node_row("warming"), node_row("never-fixed")],
        {"warming": estimator_checkpoint_row(None, horizontal_std_m=None)},
    )
    assert dict(snapshot.node_placements) == {}


def test_identity_orientation_omits_the_quaternion_key() -> None:
    snapshot = _snapshot(
        [node_row("level", overrides={"position_m": [0.0, 0.0, 0.0]}, orientation={"yaw_deg": 0.0})]
    )
    assert "body_to_world_quaternion_wxyz" not in snapshot.to_push_body(1)["node_placements"]["level"]


def test_one_malformed_row_does_not_withhold_other_nodes() -> None:
    snapshot = _snapshot(
        [
            node_row("bad", overrides={"position_m": [0.0, 0.0, 0.0]}, orientation={"yaw_deg": 9999.0}),
            node_row("good", overrides={"position_m": [1.0, 0.0, 0.0]}),
        ]
    )
    assert set(snapshot.node_placements) == {"good"}


@pytest.mark.parametrize(("mode", "projection"), [("flat", "flat_wgs84_series"), ("geodetic", "geodetic_ecef_enu")])
def test_frame_id_and_projection_follow_the_configured_local_frame(mode: str, projection: str) -> None:
    frame = LocalCoordinateFrame(origin=SITE_ORIGIN, mode=mode)
    body = _snapshot([], frame=frame).to_push_body(1)

    assert body["projection"] == projection
    assert body["site_frame_id"] == site_frame_id_for(SITE_ORIGIN, projection)
    assert body["site_origin"] == {"lat": SITE_ORIGIN.lat, "lon": SITE_ORIGIN.lon, "alt_m": SITE_ORIGIN.alt_m}
    assert body["node_placements"] == {}


def test_frame_id_changes_with_origin_and_projection_but_not_float_noise() -> None:
    nudged = GeoPoint(lat=SITE_ORIGIN.lat + 1e-12, lon=SITE_ORIGIN.lon, alt_m=SITE_ORIGIN.alt_m)
    moved = GeoPoint(lat=SITE_ORIGIN.lat + 1e-5, lon=SITE_ORIGIN.lon, alt_m=SITE_ORIGIN.alt_m)
    base = site_frame_id_for(SITE_ORIGIN, "flat_wgs84_series")

    assert site_frame_id_for(nudged, "flat_wgs84_series") == base
    assert site_frame_id_for(moved, "flat_wgs84_series") != base
    assert site_frame_id_for(SITE_ORIGIN, "geodetic_ecef_enu") != base


def test_body_matches_the_contract_field_set() -> None:
    body = _snapshot(
        [node_row("n", overrides={"position_m": [1.0, 2.0, 3.0], "orientation": {"pitch_deg": 10.0}},
                  orientation={"pitch_deg": 10.0})]
    ).to_push_body(7)

    assert set(body) == {"site_frame_id", "geometry_push_revision", "site_origin", "projection", "node_placements"}
    assert body["geometry_push_revision"] == 7
    assert set(body["node_placements"]["n"]) == {
        "position_m",
        "position_std_m",
        "placement_source",
        "body_to_world_quaternion_wxyz",
    }


class TestEstimateDrift:
    def _estimate_snapshot(self, east_offset_m: float, *, orientation: dict | None = None):
        estimate_geo = FLAT_FRAME.local_to_geo((10.0 + east_offset_m, 20.0, 0.0))
        return _snapshot(
            [node_row("gps-node", orientation=orientation or {})],
            {"gps-node": estimator_checkpoint_row(estimate_geo, horizontal_std_m=0.4)},
        )

    def test_estimate_drift_within_threshold_is_not_a_change(self) -> None:
        assert not self._estimate_snapshot(0.3).differs_beyond_estimate_drift(
            self._estimate_snapshot(0.0), estimate_move_threshold_m=0.5
        )

    def test_estimate_move_beyond_threshold_is_a_change(self) -> None:
        assert self._estimate_snapshot(0.8).differs_beyond_estimate_drift(
            self._estimate_snapshot(0.0), estimate_move_threshold_m=0.5
        )

    def test_orientation_change_is_always_a_change(self) -> None:
        assert self._estimate_snapshot(0.0, orientation={"yaw_deg": 5.0}).differs_beyond_estimate_drift(
            self._estimate_snapshot(0.0), estimate_move_threshold_m=0.5
        )

    def test_any_operator_placement_change_is_a_change(self) -> None:
        before = _snapshot([node_row("n", overrides={"position_m": [0.0, 0.0, 0.0]})])
        after = _snapshot([node_row("n", overrides={"position_m": [0.01, 0.0, 0.0]})])
        assert after.differs_beyond_estimate_drift(before, estimate_move_threshold_m=0.5)

    def test_node_set_change_is_a_change(self) -> None:
        before = _snapshot([node_row("n", overrides={"position_m": [0.0, 0.0, 0.0]})])
        assert _snapshot([]).differs_beyond_estimate_drift(before, estimate_move_threshold_m=0.5)


def test_orientation_validation_matches_node_orientation_model() -> None:
    # Guard the assumption the malformed-row test relies on.
    with pytest.raises(ValueError):
        NodeOrientation.model_validate({"yaw_deg": 9999.0})
