"""P9 clock policy: node/pair timing uncertainty and the inverse-variance pair weight."""

from __future__ import annotations

import math

import pytest

from minimappr.config import Settings
from minimappr.core.node_registry import NodeRegistry
from minimappr.core.pair_timing_uncertainty import (
    MINIMUM_PAIR_MEASUREMENT_QUALITY,
    ClockUncertaintyPolicy,
    NodeClockEvidence,
    holdover_predicted_error_seconds,
    inverse_variance_pair_weight,
    node_clock_evidence_from_frame,
    node_timing_uncertainty_seconds,
    optimizer_residual_scale,
    pair_information_score,
    pair_timing_uncertainty_seconds,
    solver_reference_time_std_seconds,
)
from minimappr.models import NodeSpec, NodeType, TimeQuality

DEFAULT_POLICY = ClockUncertaintyPolicy.from_settings(Settings())


@pytest.mark.parametrize(
    ("time_quality", "expected_seconds"),
    [
        (TimeQuality.GPS_LOCKED.value, 1.0e-6),
        (TimeQuality.GPS_HOLDOVER.value, 1.0e-4),
        (TimeQuality.NTP_DISCIPLINED.value, 2.0e-3),
        (TimeQuality.FREE_RUNNING.value, 5.0e-2),
        # Legacy database value: same class as ntp_disciplined, as in Rust.
        (TimeQuality.NTP_SYNC.value, 2.0e-3),
        ("sundial", 5.0e-3),
        (None, 5.0e-3),
    ],
)
def test_node_uncertainty_uses_class_default_or_unknown(time_quality, expected_seconds) -> None:
    evidence = NodeClockEvidence(time_quality=time_quality)
    assert node_timing_uncertainty_seconds(evidence, DEFAULT_POLICY) == pytest.approx(expected_seconds)


def test_node_without_any_clock_report_takes_unknown_default() -> None:
    assert node_timing_uncertainty_seconds(None, DEFAULT_POLICY) == pytest.approx(
        DEFAULT_POLICY.unknown_uncertainty_seconds
    )


def test_reported_holdover_error_wins_when_larger_than_class_default() -> None:
    drifting = NodeClockEvidence(TimeQuality.GPS_HOLDOVER.value, holdover_predicted_error_seconds=3.0e-4)
    fresh = NodeClockEvidence(TimeQuality.GPS_HOLDOVER.value, holdover_predicted_error_seconds=2.0e-6)
    assert node_timing_uncertainty_seconds(drifting, DEFAULT_POLICY) == pytest.approx(3.0e-4)
    assert node_timing_uncertainty_seconds(fresh, DEFAULT_POLICY) == pytest.approx(1.0e-4)


def test_configured_class_map_replaces_defaults_rather_than_merging() -> None:
    policy = ClockUncertaintyPolicy(class_uncertainty_seconds={"gps_locked": 2.0e-6}, unknown_uncertainty_seconds=0.01)
    assert node_timing_uncertainty_seconds(NodeClockEvidence("gps_locked"), policy) == pytest.approx(2.0e-6)
    assert node_timing_uncertainty_seconds(NodeClockEvidence("ntp_disciplined"), policy) == pytest.approx(0.01)


def test_pair_uncertainty_is_zero_on_one_clock_and_quadrature_across_two() -> None:
    node_ids = {"a:ch0": "a", "a:ch1": "a", "b:ch0": "b", "orphan": None}
    uncertainty = {"a:ch0": 3.0e-3, "a:ch1": 3.0e-3, "b:ch0": 4.0e-3, "orphan": 1.0}
    same_node = pair_timing_uncertainty_seconds(
        "a:ch0", "a:ch1", sensor_node_ids=node_ids, sensor_timing_uncertainty_seconds=uncertainty
    )
    cross_node = pair_timing_uncertainty_seconds(
        "a:ch0", "b:ch0", sensor_node_ids=node_ids, sensor_timing_uncertainty_seconds=uncertainty
    )
    unknown_membership = pair_timing_uncertainty_seconds(
        "a:ch0", "orphan", sensor_node_ids=node_ids, sensor_timing_uncertainty_seconds=uncertainty
    )
    assert same_node == 0.0
    assert cross_node == pytest.approx(5.0e-3)
    assert unknown_membership == 0.0
    assert pair_timing_uncertainty_seconds(
        "a:ch0", "b:ch0", sensor_node_ids=None, sensor_timing_uncertainty_seconds=uncertainty
    ) == 0.0


def test_inverse_variance_weight_reduces_to_quality_without_timing_uncertainty() -> None:
    reference_std_s = solver_reference_time_std_seconds(48_000, 4)
    assert reference_std_s == pytest.approx(1.0 / 192_000)
    for quality in (0.05, 0.4, 1.0):
        assert inverse_variance_pair_weight(
            measurement_quality=quality, timing_uncertainty_seconds=0.0, reference_time_std_seconds=reference_std_s
        ) == pytest.approx(quality)


def test_inverse_variance_weight_combines_measurement_and_timing_variance() -> None:
    reference_std_s = 1.0e-5
    quality = 0.5
    timing_s = 2.0e-5
    # Measurement variance sigma_ref**2 / q, plus timing variance; weight is
    # sigma_ref**2 over their sum.
    expected = reference_std_s**2 / (reference_std_s**2 / quality + timing_s**2)
    weight = inverse_variance_pair_weight(
        measurement_quality=quality, timing_uncertainty_seconds=timing_s, reference_time_std_seconds=reference_std_s
    )
    assert weight == pytest.approx(expected)
    weaker = inverse_variance_pair_weight(
        measurement_quality=quality, timing_uncertainty_seconds=10 * timing_s, reference_time_std_seconds=reference_std_s
    )
    assert 0.0 < weaker < weight


def test_pair_information_score_ranks_by_baseline_without_timing_uncertainty() -> None:
    reference_std_s = 1.0e-5
    assert pair_information_score(
        baseline_m=3.0, timing_uncertainty_seconds=0.0, reference_time_std_seconds=reference_std_s
    ) == pytest.approx(3.0)
    assert pair_information_score(
        baseline_m=3.0, timing_uncertainty_seconds=reference_std_s, reference_time_std_seconds=reference_std_s
    ) == pytest.approx(3.0 / math.sqrt(2.0))


def test_holdover_error_is_read_from_firmware_timing_diagnostics() -> None:
    assert holdover_predicted_error_seconds({"clock_holdover": {"predicted_error_ns": 250_000}}) == pytest.approx(2.5e-4)
    assert holdover_predicted_error_seconds({"clock_holdover": {"predicted_error_ns": True}}) is None
    assert holdover_predicted_error_seconds({"transport_health": {}}) is None
    assert holdover_predicted_error_seconds(None) is None
    evidence = node_clock_evidence_from_frame(
        TimeQuality.GPS_HOLDOVER, {"clock_holdover": {"holdover_active": True, "predicted_error_ns": 400_000}}
    )
    assert evidence == NodeClockEvidence("gps_holdover", 4.0e-4)


@pytest.mark.asyncio
async def test_node_registry_serves_per_sensor_uncertainty_and_forgets_deleted_nodes() -> None:
    registry = NodeRegistry()
    for node_id, x in (("a", 0.0), ("b", 10.0)):
        await registry.upsert(
            NodeSpec(id=node_id, node_type=NodeType.POINT, position_m=(x, 0.0, 1.0), capabilities=["audio"]),
            last_seen_ns=1,
        )
    await registry.record_node_clock_evidence("a", NodeClockEvidence(TimeQuality.NTP_DISCIPLINED.value))
    uncertainty = await registry.sensor_timing_uncertainty_seconds(DEFAULT_POLICY)
    assert uncertainty == {"a:ch0": pytest.approx(2.0e-3), "b:ch0": pytest.approx(5.0e-3)}

    await registry.delete_node("a")
    await registry.upsert(
        NodeSpec(id="a", node_type=NodeType.POINT, position_m=(0.0, 0.0, 1.0), capabilities=["audio"]),
        last_seen_ns=2,
    )
    assert (await registry.sensor_timing_uncertainty_seconds(DEFAULT_POLICY))["a:ch0"] == pytest.approx(5.0e-3)


def test_optimizer_scale_only_lifts_clock_dominated_pair_sets() -> None:
    assert optimizer_residual_scale([0.9, 0.05]) == 1.0
    assert optimizer_residual_scale([MINIMUM_PAIR_MEASUREMENT_QUALITY]) == 1.0
    assert optimizer_residual_scale([]) == 1.0
    assert optimizer_residual_scale([1.0e-6, 4.0e-6]) == pytest.approx(500.0)
