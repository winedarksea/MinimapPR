"""Python parse of ``localization.solve_provenance`` + ``pair_tdoas`` (plan item P1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from minimappr.api.rust_dsp_manifests import LocalizedClassifierRenderRequest, load_localized_render_manifest_bundle
from minimappr.api.rust_localization_solve_contract import (
    CoordinateFrameKind,
    CoordinateProjection,
    NodeTimeQuality,
    SensorPositionSource,
    SolveAttemptKind,
    SolveBufferScope,
    SolveContractError,
    SolveExclusionReason,
    parse_rust_localization_solve,
    parse_solve_provenance,
)
from minimappr.models import NodeSpec
from tests.localization_solve_manifest_fixtures import (
    classifier_render_manifest,
    contract_doc_example_solve_provenance,
    planar_node_payload,
)

PACKET_TOA_NS = 1_764_000_000_000_000_000


def test_contract_doc_example_parses_field_for_field() -> None:
    provenance = parse_solve_provenance(contract_doc_example_solve_provenance())

    assert provenance.schema_version == 1
    assert provenance.solve_id == "solve-3f2a"
    assert provenance.attempt_kind is SolveAttemptKind.EVENT_TRIGGER_ABSOLUTE_RMS
    assert provenance.buffer_scope is SolveBufferScope.AUTOMATIC_SITE_CLUSTER
    assert provenance.cluster_id == "auto-gps"
    assert provenance.reporting_node_id == "planar-a"
    frame = provenance.coordinate_frame
    assert frame.frame_id == "site-7"
    assert frame.kind is CoordinateFrameKind.SITE_ENU
    assert (frame.origin.lat, frame.origin.lon, frame.origin.alt_m) == (45.0, -93.0, 250.0)
    assert frame.projection is CoordinateProjection.FLAT_WGS84_SERIES
    assert provenance.layout_revision == 12
    assert provenance.geometry_push_revision == 3
    assert provenance.ordered_sensor_ids == ["planar-a:ch0", "planar-a:ch4", "point-b:ch0"]
    planar_ch4 = provenance.sensors[1]
    assert planar_ch4.node_id == "planar-a"
    assert planar_ch4.node_channel_index == 4
    assert planar_ch4.position_m == (1.0, 2.0, 1.5)
    assert planar_ch4.sample_rate_hz == 16000
    assert planar_ch4.position_source is SensorPositionSource.OPERATOR_OVERRIDE
    assert planar_ch4.position_std_m == 0.05
    assert planar_ch4.clock_source is NodeTimeQuality.GPS_LOCKED
    assert planar_ch4.clock_domain == "utc"
    assert planar_ch4.clock_offset_std_s == 1e-6
    assert provenance.sensors[2].position_std_m is None
    assert provenance.active_sensor_indices == (0, 1, 2)
    assert provenance.excluded_sensors == ()
    assert [(pair.sensor_index_a, pair.sensor_index_b, pair.reason) for pair in provenance.excluded_pairs] == [
        (0, 2, SolveExclusionReason.TIMELINE_RESIDUAL_OVER_LIMIT),
        (1, 2, SolveExclusionReason.TIMELINE_RESIDUAL_OVER_LIMIT),
    ]
    point_timing = provenance.node_timing[1]
    assert point_timing.time_quality is NodeTimeQuality.NTP_DISCIPLINED
    assert point_timing.timeline_residual_ns == 15_000_000
    assert point_timing.last_timestamp_step_ns == -21_000_000
    assert point_timing.timestamp_step_count == 1
    assert provenance.node_timing[0].last_timestamp_step_ns is None
    assert provenance.window_center_time_ns == 1764000000016000000
    assert provenance.window_seconds == 0.032
    assert provenance.solve_sample_rate_hz == 16000


def test_pair_tdoas_parse_with_and_without_contract_additions() -> None:
    solve = parse_rust_localization_solve(
        {
            "solve_provenance": contract_doc_example_solve_provenance(),
            "pair_tdoas": [
                {"ch_a": 0, "ch_b": 1, "delay_samples": 0.5, "lag_seconds": 3.1e-5, "confidence": 0.8,
                 "is_cross_node": False, "timing_uncertainty_seconds": 1.4e-6},
                {"ch_a": 1, "ch_b": 0, "delay_samples": -0.5, "lag_seconds": -3.1e-5, "confidence": 0.7},
            ],
        }
    )
    assert solve is not None
    first, second = solve.admitted_pair_tdoas
    assert (first.ch_a, first.ch_b, first.lag_seconds, first.confidence) == (0, 1, 3.1e-5, 0.8)
    assert first.delay_samples == 0.5
    assert first.is_cross_node is False
    assert first.timing_uncertainty_seconds == 1.4e-6
    assert second.is_cross_node is None
    assert second.timing_uncertainty_seconds is None
    assert solve.contributing_sensor_indices() == [0, 1]


def test_legacy_localization_without_provenance_parses_to_none() -> None:
    assert parse_rust_localization_solve({"position_m": [1.0, 2.0, 3.0], "pair_tdoas": []}) is None
    assert parse_rust_localization_solve(None) is None


def test_solve_without_pairs_reports_its_active_sensors_as_contributors() -> None:
    provenance = contract_doc_example_solve_provenance()
    provenance["active_sensor_indices"] = [2, 0]
    solve = parse_rust_localization_solve({"solve_provenance": provenance})
    assert solve is not None and solve.admitted_pair_tdoas == ()
    assert solve.contributing_sensor_indices() == [0, 2]


def _break(mutator):
    provenance = contract_doc_example_solve_provenance()
    mutator(provenance)
    return provenance


@pytest.mark.parametrize(
    "broken_provenance",
    [
        _break(lambda p: p.pop("solve_id")),
        _break(lambda p: p.update(schema_version=2)),
        _break(lambda p: p.update(attempt_kind="cadence")),
        _break(lambda p: p["coordinate_frame"].update(kind="wgs84")),
        _break(lambda p: p["coordinate_frame"].update(projection="sphere_6371km")),
        _break(lambda p: p["coordinate_frame"].pop("frame_id")),
        _break(lambda p: p["sensors"][0].update(position_m=[1.0, 2.0])),
        _break(lambda p: p["sensors"][0].update(clock_source="ntp_sync")),
        _break(lambda p: p["sensors"][0].pop("position_std_m")),
        _break(lambda p: p["sensors"][2].update(sensor_id="planar-a:ch0")),
        _break(lambda p: p.update(active_sensor_indices=[0, 3])),
        _break(lambda p: p["excluded_pairs"][0].update(sensor_index_b=7)),
        _break(lambda p: p["excluded_pairs"][0].update(reason="weak_clock")),
        _break(lambda p: p.update(window_center_time_ns=1.5e18)),
        _break(lambda p: p.update(layout_revision=True)),
    ],
)
def test_contract_violations_raise(broken_provenance) -> None:
    with pytest.raises(SolveContractError):
        parse_solve_provenance(broken_provenance)


@pytest.mark.parametrize(
    "pair",
    [
        {"ch_a": 0, "ch_b": 3, "lag_seconds": 1e-4, "confidence": 0.5},  # index past sensors
        {"ch_a": -1, "ch_b": 1, "lag_seconds": 1e-4, "confidence": 0.5},
        {"ch_a": 1, "ch_b": 1, "lag_seconds": 1e-4, "confidence": 0.5},
        {"ch_a": 0, "ch_b": 1, "lag_seconds": float("nan"), "confidence": 0.5},
        {"ch_a": 0, "ch_b": 1, "lag_seconds": 1e-4, "confidence": 0.5, "is_cross_node": "yes"},
    ],
)
def test_pair_tdoas_that_do_not_index_this_solve_raise(pair) -> None:
    with pytest.raises(SolveContractError):
        parse_rust_localization_solve({"solve_provenance": contract_doc_example_solve_provenance(), "pair_tdoas": [pair]})


# --- manifest loader --------------------------------------------------------------


def _classifier_render_manifest(localization: dict | None, *, toa_ns: int | None = PACKET_TOA_NS) -> dict:
    return classifier_render_manifest(node_payload=planar_node_payload(), localization=localization, toa_ns=toa_ns)


def test_loader_stores_window_center_as_event_time_and_attaches_the_solve() -> None:
    localization = {
        "position_m": [5.0, 6.0, 1.0],
        "confidence": 0.8,
        "pair_tdoas": [{"ch_a": 0, "ch_b": 2, "lag_seconds": -2e-3, "confidence": 0.9, "is_cross_node": True,
                        "timing_uncertainty_seconds": 2e-3}],
        "solve_provenance": contract_doc_example_solve_provenance(),
    }
    request = load_localized_render_manifest_bundle(manifest_payload=_classifier_render_manifest(localization)).request

    assert request.localization_solve is not None
    assert request.localization_solve.provenance.solve_id == "solve-3f2a"
    assert request.event_time_ns == 1764000000016000000 != PACKET_TOA_NS
    assert request.localization_pair_tdoas[0]["is_cross_node"] is True
    assert request.localization_pair_tdoas[0]["timing_uncertainty_seconds"] == 2e-3


def test_loader_parses_a_legacy_manifest_without_provenance() -> None:
    request = load_localized_render_manifest_bundle(
        manifest_payload=_classifier_render_manifest({"position_m": [5.0, 6.0, 1.0], "confidence": 0.8})
    ).request
    assert request.localization_solve is None
    assert request.event_time_ns == PACKET_TOA_NS


def test_loader_never_substitutes_receipt_time_for_a_missing_packet_time() -> None:
    with pytest.raises(ValueError, match="packet-derived time"):
        load_localized_render_manifest_bundle(manifest_payload=_classifier_render_manifest(None, toa_ns=None))


def test_loader_rejects_a_manifest_whose_provenance_breaks_the_contract() -> None:
    provenance = contract_doc_example_solve_provenance()
    provenance["sensors"][0].pop("sensor_id")
    with pytest.raises(SolveContractError):
        load_localized_render_manifest_bundle(
            manifest_payload=_classifier_render_manifest({"position_m": [5.0, 6.0, 1.0], "solve_provenance": provenance})
        )


def test_request_refuses_an_event_time_other_than_the_solve_window_center(tmp_path: Path) -> None:
    solve = parse_rust_localization_solve({"solve_provenance": contract_doc_example_solve_provenance()})
    with pytest.raises(ValueError, match="window_center_time_ns"):
        LocalizedClassifierRenderRequest(
            manifest_id="m",
            node=NodeSpec.model_validate(planar_node_payload()),
            event_time_ns=PACKET_TOA_NS,
            sample_rate_hz=16000,
            decoded_audio=np.zeros(16, dtype=np.float32),
            localization_position_m=(0.0, 0.0, 0.0),
            localization_confidence=0.5,
            localization_solve=solve,
        )
