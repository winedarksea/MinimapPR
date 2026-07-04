"""Tests for pluggable TrackAssociator / TrackFilter Protocol wiring.

Verifies:
  - NearestNeighborAssociator and LinearTrackFilter/KalmanTrackFilter
    satisfy their respective runtime-checkable Protocols.
  - TrackManager correctly delegates to injected custom strategies.
  - A stub associator and stub filter can replace the defaults,
    demonstrating the pluggability required for Phase 3 MHT/JPDA/IMM.
"""

from __future__ import annotations

import pytest
import numpy as np

from minimappr.config import Settings
from minimappr.core.track_associators import NearestNeighborAssociator
from minimappr.core.track_filters import KalmanTrackFilter, LinearTrackFilter
from minimappr.core.tracking import TrackManager
from minimappr.interfaces import TrackAssociator, TrackFilter
from minimappr.models import TrackState, TrackStatus


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def test_nearest_neighbor_satisfies_associator_protocol() -> None:
    assoc = NearestNeighborAssociator(association_distance_m=10.0)
    assert isinstance(assoc, TrackAssociator)


def test_linear_filter_satisfies_filter_protocol() -> None:
    filt = LinearTrackFilter(
        position_alpha=0.6,
        velocity_alpha=0.5,
        default_covariance_diagonal=5.0,
    )
    assert isinstance(filt, TrackFilter)


def test_kalman_filter_satisfies_filter_protocol() -> None:
    filt = KalmanTrackFilter(
        process_noise=1.0,
        measurement_noise=0.8,
        initial_position_variance=10.0,
        initial_velocity_variance=1.0,
    )
    assert isinstance(filt, TrackFilter)


# ---------------------------------------------------------------------------
# NearestNeighborAssociator unit tests
# ---------------------------------------------------------------------------

def test_associator_returns_none_for_empty_tracks() -> None:
    assoc = NearestNeighborAssociator(association_distance_m=10.0)
    result = assoc.associate(
        timestamp_ns=100,
        position_m=(5.0, 0.0, 0.0),
        existing_tracks=[],
    )
    assert result is None


def test_associator_matches_closest_track() -> None:
    assoc = NearestNeighborAssociator(association_distance_m=20.0)
    t0 = 1_000_000_000_000_000_000
    tracks = [
        TrackState(id="trk-a", first_seen_ns=t0, last_seen_ns=t0, position_m=(10.0, 0.0, 0.0)),
        TrackState(id="trk-b", first_seen_ns=t0, last_seen_ns=t0, position_m=(50.0, 0.0, 0.0)),
    ]
    result = assoc.associate(
        timestamp_ns=t0 + 1_000_000_000,
        position_m=(12.0, 0.0, 0.0),
        existing_tracks=tracks,
    )
    assert result == "trk-a"


def test_associator_skips_dropped_tracks() -> None:
    assoc = NearestNeighborAssociator(association_distance_m=20.0)
    t0 = 1_000_000_000_000_000_000
    tracks = [
        TrackState(
            id="trk-dropped",
            first_seen_ns=t0,
            last_seen_ns=t0,
            position_m=(1.0, 0.0, 0.0),
            status=TrackStatus.DROPPED.value,
        ),
    ]
    result = assoc.associate(
        timestamp_ns=t0 + 1_000_000_000,
        position_m=(1.0, 0.0, 0.0),
        existing_tracks=tracks,
    )
    assert result is None


def test_associator_rejects_out_of_gate() -> None:
    assoc = NearestNeighborAssociator(association_distance_m=5.0)
    t0 = 1_000_000_000_000_000_000
    tracks = [
        TrackState(id="trk-far", first_seen_ns=t0, last_seen_ns=t0, position_m=(100.0, 0.0, 0.0)),
    ]
    result = assoc.associate(
        timestamp_ns=t0 + 1_000_000_000,
        position_m=(0.0, 0.0, 0.0),
        existing_tracks=tracks,
    )
    assert result is None


def test_associator_uses_bounded_mahalanobis_gate() -> None:
    assoc = NearestNeighborAssociator(association_distance_m=5.0)
    t0 = 1_000_000_000_000_000_000
    tracks = [
        TrackState(
            id="trk-near",
            first_seen_ns=t0,
            last_seen_ns=t0,
            position_m=(0.0, 0.0, 0.0),
            position_covariance_m2=[
                [4.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        ),
        TrackState(
            id="trk-far",
            first_seen_ns=t0,
            last_seen_ns=t0,
            position_m=(60.0, 0.0, 0.0),
            position_covariance_m2=[
                [10_000.0, 0.0, 0.0],
                [0.0, 10_000.0, 0.0],
                [0.0, 0.0, 10_000.0],
            ],
        ),
    ]

    matched = assoc.associate(
        timestamp_ns=t0 + 1_000_000_000,
        position_m=(5.5, 0.0, 0.0),
        existing_tracks=tracks,
        measurement_covariance_m2=[
            [4.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    assert matched == "trk-near"

    rejected = assoc.associate(
        timestamp_ns=t0 + 1_000_000_000,
        position_m=(35.0, 0.0, 0.0),
        existing_tracks=tracks,
        measurement_covariance_m2=[
            [10_000.0, 0.0, 0.0],
            [0.0, 10_000.0, 0.0],
            [0.0, 0.0, 10_000.0],
        ],
    )
    assert rejected is None


# ---------------------------------------------------------------------------
# LinearTrackFilter unit tests
# ---------------------------------------------------------------------------

def test_linear_filter_predict_returns_state_unchanged() -> None:
    filt = LinearTrackFilter(position_alpha=0.6, velocity_alpha=0.5, default_covariance_diagonal=5.0)
    t0 = 1_000_000_000_000_000_000
    state = TrackState(
        id="trk-1", first_seen_ns=t0, last_seen_ns=t0,
        position_m=(10.0, 20.0, 0.0), velocity_mps=(1.0, 0.0, 0.0),
    )
    predicted = filt.predict(state, dt_s=1.0)
    # Linear predict returns state unchanged (no extrapolation).
    assert predicted.position_m == state.position_m


def test_linear_filter_update_smooths() -> None:
    filt = LinearTrackFilter(position_alpha=0.6, velocity_alpha=0.5, default_covariance_diagonal=5.0)
    t0 = 1_000_000_000_000_000_000
    state = TrackState(
        id="trk-1", first_seen_ns=t0, last_seen_ns=t0,
        position_m=(0.0, 0.0, 0.0), velocity_mps=(0.0, 0.0, 0.0),
    )
    updated = filt.update(state, measurement_m=(10.0, 0.0, 0.0), dt_s=1.0)
    # With alpha=0.6: position = 0.6*0 + 0.4*10 = 4.0
    assert updated.position_m == pytest.approx((4.0, 0.0, 0.0), abs=1e-6)


def test_linear_filter_reports_measurement_covariance_without_freezing_position() -> None:
    filt = LinearTrackFilter(position_alpha=0.6, velocity_alpha=0.5, default_covariance_diagonal=5.0)
    t0 = 1_000_000_000_000_000_000
    state = TrackState(
        id="trk-1",
        first_seen_ns=t0,
        last_seen_ns=t0,
        position_m=(0.0, 0.0, 0.0),
        velocity_mps=(0.0, 0.0, 0.0),
        position_covariance_m2=[
            [1.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

    updated = filt.update(
        state,
        measurement_m=(10.0, 10.0, 10.0),
        dt_s=1.0,
        measurement_covariance_m2=[
            [100.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 100.0],
        ],
    )

    assert updated.position_m == pytest.approx((4.0, 4.0, 4.0), abs=1e-6)
    assert updated.position_covariance_m2 == [
        [100.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 100.0],
    ]


def test_linear_filter_remains_responsive_during_sustained_motion() -> None:
    filt = LinearTrackFilter(
        position_alpha=0.6,
        velocity_alpha=0.5,
        default_covariance_diagonal=5.0,
    )
    state = TrackState(
        id="trk-moving",
        first_seen_ns=0,
        last_seen_ns=0,
        position_m=(0.0, 0.0, 0.0),
        velocity_mps=(0.0, 0.0, 0.0),
    )
    measurement_covariance = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    for position_x_m in range(1, 51):
        state = filt.update(
            state,
            measurement_m=(float(position_x_m), 0.0, 0.0),
            dt_s=1.0,
            measurement_covariance_m2=measurement_covariance,
        )

    assert state.position_m[0] > 48.0
    assert state.position_m[0] < 50.0
    assert state.position_covariance_m2 == measurement_covariance


# ---------------------------------------------------------------------------
# KalmanTrackFilter unit tests
# ---------------------------------------------------------------------------

def test_kalman_filter_predict_extrapolates() -> None:
    filt = KalmanTrackFilter(
        process_noise=1.0, measurement_noise=0.8,
        initial_position_variance=10.0, initial_velocity_variance=1.0,
    )
    t0 = 1_000_000_000_000_000_000
    state = TrackState(
        id="trk-1", first_seen_ns=t0, last_seen_ns=t0,
        position_m=(0.0, 0.0, 0.0),
    )
    # Initialize internal state, then run an update to establish velocity.
    filt.initialize_track("trk-1", (0.0, 0.0, 0.0))
    updated = filt.update(state, measurement_m=(2.0, 0.0, 0.0), dt_s=1.0)
    # Now predict forward — position should extrapolate using Kalman velocity.
    predicted = filt.predict(updated, dt_s=1.0)
    assert predicted.position_m[0] > updated.position_m[0]  # Should be ahead


def test_kalman_filter_predict_grows_position_covariance() -> None:
    filt = KalmanTrackFilter(
        process_noise=1.0,
        measurement_noise=0.8,
        initial_position_variance=10.0,
        initial_velocity_variance=1.0,
    )
    t0 = 1_000_000_000_000_000_000
    state = TrackState(
        id="trk-2",
        first_seen_ns=t0,
        last_seen_ns=t0,
        position_m=(0.0, 0.0, 0.0),
    )

    filt.initialize_track("trk-2", (0.0, 0.0, 0.0))
    updated = filt.update(state, measurement_m=(2.0, 0.0, 0.0), dt_s=1.0)
    predicted = filt.predict(updated, dt_s=2.0)

    assert predicted.position_covariance_m2 is not None
    assert updated.position_covariance_m2 is not None
    assert predicted.position_covariance_m2[0][0] > updated.position_covariance_m2[0][0]


def test_kalman_filter_inflates_non_positive_definite_measurement_covariance() -> None:
    filt = KalmanTrackFilter(
        process_noise=1.0,
        measurement_noise=0.8,
        initial_position_variance=10.0,
        initial_velocity_variance=1.0,
    )
    t0 = 1_000_000_000_000_000_000
    state = TrackState(
        id="trk-pd",
        first_seen_ns=t0,
        last_seen_ns=t0,
        position_m=(0.0, 0.0, 0.0),
    )
    filt.initialize_track("trk-pd", (0.0, 0.0, 0.0))

    updated = filt.update(
        state,
        measurement_m=(1.0, 0.0, 0.0),
        dt_s=1.0,
        measurement_covariance_m2=[
            [-100.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    covariance = np.asarray(updated.position_covariance_m2, dtype=np.float64)
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)


def test_kalman_filter_remove_track_cleanup() -> None:
    filt = KalmanTrackFilter(
        process_noise=1.0, measurement_noise=0.8,
        initial_position_variance=10.0, initial_velocity_variance=1.0,
    )
    filt.initialize_track("trk-x", (1.0, 2.0, 3.0))
    assert "trk-x" in filt._states
    filt.remove_track("trk-x")
    assert "trk-x" not in filt._states


# ---------------------------------------------------------------------------
# TrackManager custom strategy injection
# ---------------------------------------------------------------------------

class _AlwaysNewAssociator:
    """Stub associator that never matches — every detection creates a new track."""

    def associate(
        self,
        timestamp_ns: int,
        position_m: tuple[float, float, float],
        existing_tracks: list[TrackState],
    ) -> str | None:
        return None


class _IdentityFilter:
    """Stub filter that returns the measurement directly as the track state."""

    def predict(self, state: TrackState, dt_s: float) -> TrackState:
        return state

    def update(
        self,
        state: TrackState,
        measurement_m: tuple[float, float, float],
        dt_s: float,
    ) -> TrackState:
        return state.model_copy(
            update={
                "position_m": measurement_m,
                "velocity_mps": (0.0, 0.0, 0.0),
                "position_covariance_m2": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            },
        )

    def initialize_track(self, track_id: str, position_m: tuple[float, float, float]) -> None:
        pass

    def remove_track(self, track_id: str) -> None:
        pass


@pytest.mark.asyncio
async def test_trackmanager_with_custom_associator() -> None:
    """Verify that a custom associator that always creates new tracks is honoured."""
    settings = Settings(association_distance_m=20.0)
    manager = TrackManager(
        settings,
        associator=_AlwaysNewAssociator(),
    )
    t0 = 1_700_000_000_000_000_000
    trk1 = await manager.update(t0, (5.0, 0.0, 0.0), "bird", 0.8)
    trk2 = await manager.update(t0 + 1_000_000_000, (5.1, 0.0, 0.0), "bird", 0.8)
    # Both should be new tracks despite being 0.1m apart.
    assert trk1.id != trk2.id


@pytest.mark.asyncio
async def test_trackmanager_with_custom_filter() -> None:
    """Verify that a custom filter is used for state updates."""
    settings = Settings(association_distance_m=20.0)
    manager = TrackManager(
        settings,
        track_filter=_IdentityFilter(),
    )
    t0 = 1_700_000_000_000_000_000
    # First update creates a new track.
    trk = await manager.update(t0, (0.0, 0.0, 0.0), "bird", 0.5)
    # Second update should use the identity filter — position becomes measurement directly.
    trk2 = await manager.update(t0 + 1_000_000_000, (10.0, 0.0, 0.0), "bird", 0.6)
    assert trk2.id == trk.id
    assert trk2.position_m == pytest.approx((10.0, 0.0, 0.0), abs=1e-6)


@pytest.mark.asyncio
async def test_trackmanager_with_both_custom_strategies() -> None:
    """Verify both custom associator and filter compose correctly."""
    settings = Settings(association_distance_m=20.0)
    manager = TrackManager(
        settings,
        associator=NearestNeighborAssociator(association_distance_m=50.0),
        track_filter=_IdentityFilter(),
    )
    t0 = 1_700_000_000_000_000_000
    trk1 = await manager.update(t0, (0.0, 0.0, 0.0), "vehicle", 0.7)
    # 30m away — default gate (20m) would reject, but custom gate (50m) accepts.
    trk2 = await manager.update(t0 + 1_000_000_000, (30.0, 0.0, 0.0), "vehicle", 0.8)
    assert trk2.id == trk1.id
    assert trk2.position_m == pytest.approx((30.0, 0.0, 0.0), abs=1e-6)


@pytest.mark.asyncio
async def test_trackmanager_filter_lifecycle_initialize_and_remove() -> None:
    """Verify filter.initialize_track and remove_track are called at the right times."""

    class _SpyFilter(_IdentityFilter):
        def __init__(self) -> None:
            self.initialized: list[str] = []
            self.removed: list[str] = []

        def initialize_track(self, track_id: str, position_m: tuple[float, float, float]) -> None:
            self.initialized.append(track_id)

        def remove_track(self, track_id: str) -> None:
            self.removed.append(track_id)

    spy = _SpyFilter()
    settings = Settings(
        association_distance_m=20.0,
        track_stale_seconds=1.0,
        track_drop_multiplier=2.0,
        track_reap_multiplier=5.0,
    )
    manager = TrackManager(settings, track_filter=spy)
    t0 = 1_700_000_000_000_000_000

    trk = await manager.update(t0, (0.0, 0.0, 0.0), "bird", 0.5)
    assert trk.id in spy.initialized

    # Advance far enough to trigger drop + reap.
    far_future = t0 + int(20 * 1_000_000_000)
    await manager.snapshot(now_ns=far_future)
    assert trk.id in spy.removed


# ---------------------------------------------------------------------------
# Phase 3: cross-node cone fusion (configurable gate + contributor tracking)
# ---------------------------------------------------------------------------

def _cone_covariance(radial_axis, *, lateral_std_m: float, radial_std_m: float):
    """Bearing-projected 'cone': tight laterally, huge radially along the bearing."""
    r = np.asarray(radial_axis, dtype=np.float64)
    r = r / (np.linalg.norm(r) + 1e-12)
    lat = lateral_std_m**2 * (np.eye(3) - np.outer(r, r))
    rad = radial_std_m**2 * np.outer(r, r)
    return (lat + rad).tolist()


def test_default_gate_blocks_cross_node_cone_fusion_regression() -> None:
    """Regression: with the default gate (4x association_distance_m = 32 m), two
    cones whose apexes are 60 m apart do NOT associate — behaviour is unchanged."""
    assoc = NearestNeighborAssociator(association_distance_m=8.0)  # max gate 32 m
    t0 = 1_000_000_000_000_000_000
    # Track cone from node A pointing +x; a new detection 60 m away along +x.
    tracks = [
        TrackState(
            id="trk-a",
            first_seen_ns=t0,
            last_seen_ns=t0,
            position_m=(50.0, 0.0, 0.0),
            position_covariance_m2=_cone_covariance([1, 0, 0], lateral_std_m=3.0, radial_std_m=400.0),
        )
    ]
    result = assoc.associate(
        timestamp_ns=t0 + 1_000_000_000,
        position_m=(110.0, 0.0, 0.0),
        existing_tracks=tracks,
        measurement_covariance_m2=_cone_covariance([1, 0, 0], lateral_std_m=3.0, radial_std_m=400.0),
    )
    assert result is None


def test_wide_gate_enables_cross_node_cone_fusion() -> None:
    """With a widened gate, two crossing cones for the same distant source associate
    via the Mahalanobis test even though their apexes are >32 m apart."""
    assoc = NearestNeighborAssociator(association_distance_m=8.0, max_gate_m=500.0)
    t0 = 1_000_000_000_000_000_000
    # Node A at origin sees the source at ~200 m along +x → cone along +x.
    tracks = [
        TrackState(
            id="trk-a",
            first_seen_ns=t0,
            last_seen_ns=t0,
            position_m=(200.0, 0.0, 0.0),
            position_covariance_m2=_cone_covariance([1, 0, 0], lateral_std_m=4.0, radial_std_m=500.0),
        )
    ]
    # Node B 50 m off in +y sees the same source; its cone points slightly different,
    # but the true source (200, 0, 0) lies inside track A's elongated cone.
    matched = assoc.associate(
        timestamp_ns=t0 + 1_000_000_000,
        position_m=(205.0, 2.0, 0.0),
        existing_tracks=tracks,
        measurement_covariance_m2=_cone_covariance([0.97, -0.24, 0.0], lateral_std_m=4.0, radial_std_m=500.0),
    )
    assert matched == "trk-a"


@pytest.mark.asyncio
async def test_trackmanager_records_contributors_and_multi_node_metric() -> None:
    cfg = Settings(
        association_distance_m=8.0,
        association_max_gate_m=500.0,
        tracking_filter="kalman",
    ).tracking_config()
    manager = TrackManager(cfg)
    t0 = 1_000_000_000_000_000_000
    cone = _cone_covariance([1, 0, 0], lateral_std_m=4.0, radial_std_m=500.0)

    # Node A detects the source cone.
    first = await manager.update(
        timestamp_ns=t0,
        position_m=(200.0, 0.0, 0.0),
        label="drone",
        confidence=0.7,
        measurement_covariance_m2=cone,
        capability_tier="full_3d",
        source_node_id="node-a",
    )
    assert first.contributor_node_ids == ["node-a"]
    assert manager.multi_node_association_count() == 0

    # Node B detects the same source a moment later; it associates to the same track.
    second = await manager.update(
        timestamp_ns=t0 + 300_000_000,
        position_m=(203.0, 1.0, 0.0),
        label="drone",
        confidence=0.7,
        measurement_covariance_m2=_cone_covariance([0.98, 0.2, 0.0], lateral_std_m=4.0, radial_std_m=500.0),
        capability_tier="full_3d",
        source_node_id="node-b",
    )
    assert second.id == first.id
    assert set(second.contributor_node_ids) == {"node-a", "node-b"}
    assert manager.multi_node_association_count() == 1
    assert manager.multi_node_active_count() == 1

    # The Kalman posterior after fusing two crossing cones is tighter than a single
    # cone's radial variance (no longer 500 m along the shared bearing).
    posterior = np.asarray(second.position_covariance_m2, dtype=np.float64)
    assert float(np.max(np.linalg.eigvalsh(posterior))) < 500.0**2


@pytest.mark.asyncio
async def test_multi_node_tqi_boosted_over_single_node() -> None:
    cfg = Settings(association_distance_m=8.0, association_max_gate_m=500.0).tracking_config()
    manager = TrackManager(cfg)
    t0 = 1_000_000_000_000_000_000
    cone = _cone_covariance([1, 0, 0], lateral_std_m=4.0, radial_std_m=500.0)
    a = await manager.update(
        timestamp_ns=t0, position_m=(200.0, 0.0, 0.0), label="x", confidence=0.5,
        measurement_covariance_m2=cone, capability_tier="full_3d", source_node_id="node-a",
    )
    single_node_tqi = a.tqi
    b = await manager.update(
        timestamp_ns=t0 + 200_000_000, position_m=(202.0, 1.0, 0.0), label="x", confidence=0.5,
        measurement_covariance_m2=_cone_covariance([0.98, 0.2, 0.0], lateral_std_m=4.0, radial_std_m=500.0),
        capability_tier="full_3d", source_node_id="node-b",
    )
    assert b.tqi > single_node_tqi
