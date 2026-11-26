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


def _ray(node_id, origin, direction, *, lateral_std_m=1.0, range_prior_m=50.0, confidence=0.8):
    """Cone given by an explicit bearing rather than a true source position."""
    unit_direction = np.asarray(direction, dtype=np.float64)
    return BearingObservation(
        node_id=node_id,
        origin_m=np.asarray(origin, dtype=np.float64),
        direction=unit_direction / np.linalg.norm(unit_direction),
        lateral_std_m=lateral_std_m,
        confidence=confidence,
        event_time_ns=0,
        range_prior_m=range_prior_m,
        expiry_ns=10**15,
    )


PRODUCTION_MAX_RANGE_M = 6000.0


def test_review_reproduction_intersection_behind_both_arrays_is_rejected() -> None:
    """Lines cross at (10, 0, 0), 10 m behind each array: formerly fused at 0.95."""
    result = fuse_bearings(
        [_ray("a", [0, 0, 0], [-1, 0, 0]), _ray("b", [10, 10, 0], [0, 1, 0])],
        max_range_m=PRODUCTION_MAX_RANGE_M,
    )
    assert result == "behind_array"


def test_review_reproduction_vertically_separated_rays_disagree() -> None:
    """Perpendicular forward rays 30 m apart vertically: the midpoint misses each by 15 m."""
    result = fuse_bearings(
        [_ray("a", [0, 0, 0], [1, 0, 0]), _ray("b", [20, -20, 30], [0, 1, 0])],
        max_range_m=PRODUCTION_MAX_RANGE_M,
    )
    assert result == "rays_disagree"


def test_anti_parallel_rays_are_rejected_whether_facing_or_diverging() -> None:
    facing = [_ray("a", [0, 0, 0], [1, 0, 0]), _ray("b", [100, 10, 0], [-1, 0, 0])]
    diverging = [_ray("a", [0, 0, 0], [-1, 0, 0]), _ray("b", [100, 10, 0], [1, 0, 0])]
    assert fuse_bearings(facing, max_range_m=PRODUCTION_MAX_RANGE_M) == "near_parallel"
    assert fuse_bearings(diverging, max_range_m=PRODUCTION_MAX_RANGE_M) == "near_parallel"


def test_nearly_parallel_rays_resolve_only_when_they_converge_forward() -> None:
    """30 m baseline, 250 m source (~7 deg parallax): true rays fuse; a bearing error
    that makes them diverge puts the line crossing behind the arrays."""
    oa, ob = np.array([0.0, 0.0, 0.0]), np.array([30.0, 0.0, 0.0])
    source = np.array([15.0, 250.0, 0.0])
    converging = fuse_bearings(
        [_obs("a", oa, source, lateral_std_m=2.0), _obs("b", ob, source, lateral_std_m=2.0)],
        max_range_m=PRODUCTION_MAX_RANGE_M,
    )
    assert isinstance(converging, BearingFusionResult)
    assert float(np.linalg.norm(converging.position_m - source)) / 250.0 < 0.05

    diverging_b_direction = np.array([np.sin(np.radians(12.0)), np.cos(np.radians(12.0)), 0.0])
    diverging = fuse_bearings(
        [_obs("a", oa, source, lateral_std_m=2.0), _ray("b", ob, diverging_b_direction, lateral_std_m=2.0)],
        max_range_m=PRODUCTION_MAX_RANGE_M,
    )
    assert diverging == "behind_array"


def test_lateral_std_is_scaled_from_solve_range_to_fused_range() -> None:
    """Cones stated at a 50 m solve range keep their angular width at ~300 m; the
    same lateral std stated at 1 km is a much narrower cone and the pair disagrees."""
    oa, ob = np.array([0.0, 0.0, 1.5]), np.array([60.0, 0.0, 1.5])
    toward_a, toward_b = np.array([150.0, 200.0, 10.0]), np.array([160.0, 190.0, 0.0])

    def pair(range_prior_m: float):
        return [
            _ray("a", oa, toward_a - oa, lateral_std_m=2.0, range_prior_m=range_prior_m),
            _ray("b", ob, toward_b - ob, lateral_std_m=2.0, range_prior_m=range_prior_m),
        ]

    assert isinstance(fuse_bearings(pair(50.0), max_range_m=PRODUCTION_MAX_RANGE_M), BearingFusionResult)
    assert fuse_bearings(pair(1000.0), max_range_m=PRODUCTION_MAX_RANGE_M) == "rays_disagree"


def test_fused_confidence_reflects_ray_agreement() -> None:
    oa, ob = np.array([0.0, 0.0, 0.0]), np.array([40.0, 0.0, 0.0])
    exact = fuse_bearings([_obs("a", oa, [20, 30, 5]), _obs("b", ob, [20, 30, 5])])
    skewed = fuse_bearings(
        [_obs("a", oa, [20, 30, 5], lateral_std_m=1.0), _obs("b", ob, [20, 30, 7], lateral_std_m=1.0)]
    )
    assert isinstance(exact, BearingFusionResult) and isinstance(skewed, BearingFusionResult)
    assert exact.confidence == pytest.approx(0.95)  # noisy-OR of 0.8, 0.8 capped
    assert 0.0 < skewed.confidence < exact.confidence


def _two_bird_scores(trial_count: int, seed: int) -> tuple[int, int]:
    """(recovered true-pair ranges, false promotions of cross-bird pairs).

    Two birds call at once; node A hears bird 1 while node B hears either bird 1
    (true pair, recovered when range error < 25%) or bird 2 (cross pair, any fused
    result is a false range promotion). 2 deg bearing noise; cones stated at the
    production 50 m bearing-projection range.
    """

    rng = np.random.default_rng(seed)
    angular_std_rad = np.radians(2.0)
    range_prior_m = 50.0

    def noisy_cone(node_id, origin, source):
        true_direction = (source - origin) / np.linalg.norm(source - origin)
        direction = true_direction + rng.normal(0.0, angular_std_rad, 3)
        return _ray(node_id, origin, direction, lateral_std_m=range_prior_m * np.tan(angular_std_rad))

    def random_bird(center):
        distance_m, azimuth = rng.uniform(20.0, 300.0), rng.uniform(0.0, 2.0 * np.pi)
        return np.array([center[0] + distance_m * np.cos(azimuth), center[1] + distance_m * np.sin(azimuth),
                         rng.uniform(0.0, 25.0)])

    recovered = false_promotions = 0
    for _ in range(trial_count):
        node_a, node_b = np.array([0.0, 0.0, 1.5]), np.array([rng.uniform(30.0, 100.0), 0.0, 1.5])
        center = (node_a + node_b) / 2.0
        bird_one, bird_two = random_bird(center), random_bird(center)
        true_pair = fuse_bearings(
            [noisy_cone("a", node_a, bird_one), noisy_cone("b", node_b, bird_one)], max_range_m=PRODUCTION_MAX_RANGE_M
        )
        if isinstance(true_pair, BearingFusionResult):
            range_error = np.linalg.norm(true_pair.position_m - bird_one) / np.linalg.norm(bird_one - center)
            recovered += int(range_error < 0.25)
        cross_pair = fuse_bearings(
            [noisy_cone("a", node_a, bird_one), noisy_cone("b", node_b, bird_two)], max_range_m=PRODUCTION_MAX_RANGE_M
        )
        false_promotions += int(isinstance(cross_pair, BearingFusionResult))
    return recovered, false_promotions


def test_two_simultaneous_birds_false_promotions_vs_recovered_ranges() -> None:
    """Seed 7, 1000 trials: before the agreement gates 983 cross-bird pairs fused
    (701 true ranges recovered); now 204 fuse and 699 are recovered. The remainder
    cross in front of both arrays within the cones and are geometrically
    indistinguishable from a real fix without a label or solve ID."""
    trial_count = 1000
    recovered, false_promotions = _two_bird_scores(trial_count, seed=7)
    assert recovered >= 0.65 * trial_count
    assert false_promotions <= 0.25 * trial_count
