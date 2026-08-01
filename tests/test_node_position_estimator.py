from __future__ import annotations

import numpy as np
import pytest

from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.node_position_estimator import StationaryKdeState, compute_stationary_kde
from minimappr.core.node_registry import NodeRegistry
from minimappr.models import GeoPoint, NodeOverrides


def test_stationary_kde_selects_primary_cluster_and_robust_altitude() -> None:
    rng = np.random.default_rng(9)
    primary = [(float(rng.normal(4.0, 1.2)), float(rng.normal(-2.0, 1.2)), float(rng.normal(10.0, 3.0))) for _ in range(300)]
    secondary = [(float(rng.normal(30.0, 1.2)), float(rng.normal(30.0, 1.2)), 200.0) for _ in range(20)]
    estimate, uncertainty = compute_stationary_kde(primary + secondary, 2.5)
    assert abs(estimate[0] - 4.0) < 0.5
    assert abs(estimate[1] + 2.0) < 0.5
    assert abs(estimate[2] - 10.0) < 0.7
    assert uncertainty < 2.5


def test_kde_state_round_trip_and_reservoir_is_bounded() -> None:
    frame = LocalCoordinateFrame(
        origin=GeoPoint(lat=44.39053726, lon=-92.08418655, alt_m=237.85), mode="flat"
    )
    state = StationaryKdeState()
    for index in range(100):
        state.add((float(index), 0.0, 1.0), capacity=8)
    restored = StationaryKdeState.from_snapshot(state.snapshot(frame), frame)
    assert len(restored.samples) == 8
    assert restored.seen_count == 100
    # Samples survive the geodetic round trip through their own frame.
    assert np.allclose(np.array(sorted(restored.samples)), np.array(sorted(state.samples)), atol=1e-3)


def test_position_policy_defaults_and_override() -> None:
    registry = NodeRegistry()
    assert registry.position_policy_for("fixed", "stationary") == ("stationary", "kde")
    assert registry.position_policy_for("mobile", "mobile") == ("mobile", "kalman")
    registry._overrides["node"] = NodeOverrides(mobility="mobile", position_filter="raw")
    assert registry.position_policy_for("node", "stationary") == ("mobile", "raw")
