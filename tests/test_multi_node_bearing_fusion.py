from __future__ import annotations

import numpy as np
import pytest

from minimappr.core.multi_node_bearing_fusion import (
    BearingFusionResult,
    BearingFusionStore,
    BearingObservation,
    fuse_bearings,
)


def _obs(node_id, origin, source, *, lateral_std_m=4.0, confidence=0.8, event_ns=1_000, ttl_ns=10**15):
    origin = np.asarray(origin, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    direction = (source - origin) / np.linalg.norm(source - origin)
    return BearingObservation(
        node_id=node_id,
        origin_m=origin,
        direction=direction,
        lateral_std_m=lateral_std_m,
        confidence=confidence,
        event_time_ns=event_ns,
        range_prior_m=float(np.linalg.norm(source - origin)),
        expiry_ns=ttl_ns,
    )


@pytest.mark.parametrize("distance_m", [100.0, 300.0])
def test_two_nodes_30m_apart_recover_range(distance_m: float) -> None:
    """Nodes 30 m apart triangulate a 100/300 m source to <15% range error."""
    oa = np.array([0.0, 0.0, 0.0])
    ob = np.array([30.0, 0.0, 0.0])
    source = np.array([15.0, distance_m, 0.0])  # roughly broadside
    result = fuse_bearings(
        [_obs("a", oa, source), _obs("b", ob, source)],
        max_range_m=1200.0,
    )
    assert isinstance(result, BearingFusionResult)
    err = float(np.linalg.norm(result.position_m - source))
    assert err / distance_m < 0.15, f"range error {err:.1f} m at {distance_m} m"
    assert set(result.contributor_node_ids) == {"a", "b"}


def test_near_parallel_bearings_are_degenerate() -> None:
    """3 m spacing at 1 km: parallax too small — must return a degeneracy reason."""
    oa = np.array([0.0, 0.0, 0.0])
    ob = np.array([3.0, 0.0, 0.0])
    source = np.array([0.0, 1000.0, 0.0])
    result = fuse_bearings([_obs("a", oa, source), _obs("b", ob, source)], max_range_m=1200.0)
    assert result in {"near_parallel", "degenerate"}


def test_single_node_insufficient() -> None:
    oa = np.array([0.0, 0.0, 0.0])
    source = np.array([10.0, 100.0, 0.0])
    assert fuse_bearings([_obs("a", oa, source)]) == "insufficient_nodes"
    # Two observations from the same node also cannot resolve range.
    assert fuse_bearings([_obs("a", oa, source), _obs("a", oa, source)]) == "insufficient_nodes"


def test_time_skew_beyond_window_is_stale() -> None:
    oa = np.array([0.0, 0.0, 0.0])
    ob = np.array([30.0, 0.0, 0.0])
    source = np.array([15.0, 200.0, 0.0])
    a = _obs("a", oa, source, event_ns=0)
    b = _obs("b", ob, source, event_ns=5_000_000_000)  # 5 s later
    assert fuse_bearings([a, b], window_seconds=1.5) == "stale"


def test_out_of_range_rejected() -> None:
    oa = np.array([0.0, 0.0, 0.0])
    ob = np.array([30.0, 0.0, 0.0])
    source = np.array([15.0, 300.0, 0.0])
    result = fuse_bearings([_obs("a", oa, source), _obs("b", ob, source)], max_range_m=50.0)
    assert result == "out_of_range"


def test_fused_covariance_is_finite_and_positive_definite() -> None:
    oa = np.array([0.0, 0.0, 0.0])
    ob = np.array([40.0, 10.0, 0.0])
    source = np.array([20.0, 150.0, 5.0])
    result = fuse_bearings([_obs("a", oa, source), _obs("b", ob, source)], max_range_m=1200.0)
    assert isinstance(result, BearingFusionResult)
    eigs = np.linalg.eigvalsh(result.covariance_m2)
    assert np.all(eigs > 0.0) and np.all(np.isfinite(eigs))
    assert 0.0 < result.confidence <= 0.95


@pytest.mark.asyncio
async def test_store_prunes_expired_and_bounds_size() -> None:
    store = BearingFusionStore(max_entries=4, ttl_seconds=1.0)
    now = 1_000_000_000_000
    src = np.array([10.0, 100.0, 0.0])
    # Register 10 observations; store must stay bounded at 4.
    for i in range(10):
        await store.register(
            _obs(f"n{i}", [float(i), 0.0, 0.0], src, ttl_ns=now + 10**12),
            now_ns=now,
        )
    assert await store.size() <= 4

    # An expired observation is pruned on the next access.
    store2 = BearingFusionStore(max_entries=16, ttl_seconds=1.0)
    await store2.register(_obs("old", [0.0, 0.0, 0.0], src, ttl_ns=now + 500_000_000), now_ns=now)
    assert await store2.size() == 1
    later = now + 2_000_000_000  # 2 s later, past the 1 s TTL
    corr = await store2.corroborators(
        _obs("new", [30.0, 0.0, 0.0], src, event_ns=1_000),
        now_ns=later,
        window_seconds=1.5,
    )
    assert corr == []
    assert await store2.size() == 0


@pytest.mark.asyncio
async def test_store_corroborators_exclude_same_node_and_respect_window() -> None:
    store = BearingFusionStore(max_entries=16, ttl_seconds=10.0)
    now = 1_000_000_000_000
    src = np.array([10.0, 100.0, 0.0])
    await store.register(_obs("a", [0.0, 0.0, 0.0], src, event_ns=1_000_000_000, ttl_ns=now + 10**12), now_ns=now)
    await store.register(_obs("b", [30.0, 0.0, 0.0], src, event_ns=1_100_000_000, ttl_ns=now + 10**12), now_ns=now)

    probe = _obs("a", [0.0, 0.0, 0.0], src, event_ns=1_050_000_000)
    corr = await store.corroborators(probe, now_ns=now, window_seconds=1.5)
    # Only node b (different node, within window) corroborates.
    assert [o.node_id for o in corr] == ["b"]
