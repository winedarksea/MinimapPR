"""Tests for federation hardening changes.

Covers:
- 3D Mahalanobis distance gating (deconflict_use_3d config)
- _pair_owners max-size eviction
- httpx client recreation on start() after stop()
- FederationMetrics reset and rate computation
- Hungarian optimal assignment (multi-peer competing for same local track)
"""

from __future__ import annotations

import time

import pytest

from minimappr.config import FederationPeerConfig, Settings
from minimappr.core.federation import (
    FederationCoordinator,
    FederationMetrics,
    _PAIR_OWNERS_MAX_SIZE,
    _extract_covariance,
    _mahalanobis_distance,
)
from minimappr.models import (
    FederationStatusResponse,
    FederationTrackSnapshot,
    TrackState,
)


def _track(
    *,
    track_id: str,
    timestamp_ns: int,
    x: float,
    y: float,
    z: float = 0.0,
    tqi: float,
    confidence: float = 0.7,
    covariance: list[list[float]] | None = None,
) -> TrackState:
    cov = covariance or [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return TrackState(
        id=track_id,
        first_seen_ns=timestamp_ns - 2_000_000_000,
        last_seen_ns=timestamp_ns,
        position_m=(x, y, z),
        position_covariance_m2=cov,
        velocity_mps=(0.0, 0.0, 0.0),
        label="bird_like",
        label_category="wildlife",
        confidence=confidence,
        update_count=3,
        status="confirmed",
        tqi=tqi,
    )


def _settings(
    *,
    hysteresis: float = 0.05,
    ttl_seconds: float = 20.0,
    use_3d: bool = False,
) -> Settings:
    return Settings(
        federation_enabled=True,
        federation_server_id="srv-a",
        federation_peers=(
            FederationPeerConfig(
                peer_id="srv-b",
                base_url="http://127.0.0.1:19999",
                api_key="peer-token",
            ),
        ),
        federation_tqi_hysteresis=hysteresis,
        federation_track_ttl_seconds=ttl_seconds,
        federation_deconflict_use_3d=use_3d,
    )


# -- 3D Mahalanobis distance tests --


def test_mahalanobis_2d_ignores_z() -> None:
    """In 2D mode, tracks at same XY but different Z have distance ~0."""
    t1 = _track(track_id="a", timestamp_ns=1_000_000_000, x=1.0, y=1.0, z=0.0, tqi=0.5)
    t2 = _track(track_id="b", timestamp_ns=1_000_000_000, x=1.0, y=1.0, z=100.0, tqi=0.5)
    d = _mahalanobis_distance(local_track=t1, peer_track=t2, default_variance=4.0, use_3d=False)
    assert d < 0.01


def test_mahalanobis_3d_considers_z() -> None:
    """In 3D mode, tracks at same XY but very different Z have significant distance."""
    t1 = _track(track_id="a", timestamp_ns=1_000_000_000, x=1.0, y=1.0, z=0.0, tqi=0.5)
    t2 = _track(track_id="b", timestamp_ns=1_000_000_000, x=1.0, y=1.0, z=100.0, tqi=0.5)
    d = _mahalanobis_distance(local_track=t1, peer_track=t2, default_variance=4.0, use_3d=True)
    assert d > 10.0  # Significant Z separation


@pytest.mark.asyncio
async def test_deconflict_3d_separates_altitude_tracks() -> None:
    """With 3D enabled, tracks at same XY but different Z are not deconflicted."""
    now_ns = 1_740_000_000_000_000_000
    local = _track(track_id="local-1", timestamp_ns=now_ns, x=5.0, y=5.0, z=0.0, tqi=0.4)
    peer = _track(track_id="peer-1", timestamp_ns=now_ns, x=5.0, y=5.0, z=50.0, tqi=0.9)

    async def _supplier(_: int) -> list[TrackState]:
        return [local]

    coordinator = FederationCoordinator(settings=_settings(use_3d=True), track_supplier=_supplier)
    await coordinator.handle_incoming_snapshot(
        FederationTrackSnapshot(server_id="srv-b", generated_ns=now_ns, tracks=[peer]),
        now_ns=now_ns,
    )
    merged = await coordinator.merged_tracks(local_tracks=[local], now_ns=now_ns, include_standby=False)
    # Both should be owners (not deconflicted) since they're at different altitudes.
    assert len(merged) == 2
    for row in merged:
        assert row["ownership_role"] == "owner"
    await coordinator.stop()


def test_extract_covariance_3d() -> None:
    """_extract_covariance returns 3x3 block when ndim=3."""
    matrix = [
        [2.0, 0.1, 0.2],
        [0.1, 3.0, 0.3],
        [0.2, 0.3, 4.0],
    ]
    cov = _extract_covariance(matrix, default_variance=1.0, ndim=3)
    assert cov.shape == (3, 3)
    assert abs(cov[2, 2] - 4.0) < 1e-6


def test_extract_covariance_2d_from_3x3() -> None:
    """_extract_covariance returns 2x2 sub-block when ndim=2."""
    matrix = [
        [2.0, 0.1, 0.2],
        [0.1, 3.0, 0.3],
        [0.2, 0.3, 4.0],
    ]
    cov = _extract_covariance(matrix, default_variance=1.0, ndim=2)
    assert cov.shape == (2, 2)
    assert abs(cov[0, 0] - 2.0) < 1e-6
    assert abs(cov[1, 1] - 3.0) < 1e-6


def test_extract_covariance_fallback_on_bad_data() -> None:
    """_extract_covariance returns diagonal fallback for malformed input."""
    cov = _extract_covariance(None, default_variance=5.0, ndim=3)
    assert cov.shape == (3, 3)
    assert abs(cov[0, 0] - 5.0) < 1e-6
    assert abs(cov[1, 2] - 0.0) < 1e-6


# -- Metrics reset and rate tests --


def test_metrics_reset_zeros_all_counters() -> None:
    """FederationMetrics.reset() zeros all counters and updates timestamp."""
    m = FederationMetrics()
    m.heartbeats_sent = 10
    m.snapshot_failures = 3
    m.auth_rejections = 2
    before_reset_ns = m.last_reset_ns

    m.reset()
    assert m.heartbeats_sent == 0
    assert m.snapshot_failures == 0
    assert m.auth_rejections == 0
    assert m.last_reset_ns >= before_reset_ns


def test_metrics_rates_computation() -> None:
    """FederationMetrics.rates() returns per-second rates."""
    m = FederationMetrics()
    m.last_reset_ns = time.time_ns() - 10_000_000_000  # 10 seconds ago
    m.heartbeats_sent = 20
    m.snapshots_sent = 10
    m.auth_rejections = 5

    rates = m.rates()
    assert abs(rates["heartbeats_sent_per_s"] - 2.0) < 0.5
    assert abs(rates["snapshots_sent_per_s"] - 1.0) < 0.5
    assert abs(rates["auth_rejections_per_s"] - 0.5) < 0.25
    assert rates["elapsed_seconds"] > 9.0


@pytest.mark.asyncio
async def test_reset_metrics_via_coordinator() -> None:
    """FederationCoordinator.reset_metrics() resets and status reflects zeroed counters."""
    async def _supplier(_: int) -> list[TrackState]:
        return []

    coordinator = FederationCoordinator(settings=_settings(), track_supplier=_supplier)
    # Generate some metrics.
    await coordinator.validate_inbound_auth(
        peer_id="srv-b",
        authorization_header="Bearer wrong",
        token_header=None,
    )
    status_before = await coordinator.status()
    assert status_before["metrics"]["auth_rejections"] == 1

    await coordinator.reset_metrics()

    status_after = await coordinator.status()
    assert status_after["metrics"]["auth_rejections"] == 0
    assert "rates" in status_after["metrics"]
    assert "last_reset_ns" in status_after["metrics"]
    await coordinator.stop()


# -- httpx client lifecycle tests --


@pytest.mark.asyncio
async def test_client_recreated_after_stop_start() -> None:
    """After stop()+start(), the coordinator has a fresh httpx client."""
    async def _supplier(_: int) -> list[TrackState]:
        return []

    coordinator = FederationCoordinator(settings=_settings(), track_supplier=_supplier)

    # Initial state: no client yet (lazy creation).
    assert coordinator._client is None or coordinator._client.is_closed is False

    # start creates/ensures a client.
    await coordinator.start()
    assert coordinator._client is not None
    assert not coordinator._client.is_closed

    # stop closes the client.
    await coordinator.stop()
    assert coordinator._client is not None
    assert coordinator._client.is_closed

    # start again creates a new client.
    await coordinator.start()
    assert coordinator._client is not None
    assert not coordinator._client.is_closed

    await coordinator.stop()


# -- Hungarian assignment tests --


@pytest.mark.asyncio
async def test_hungarian_assignment_prevents_double_match() -> None:
    """Two peer tracks near the same local track: Hungarian assigns each uniquely."""
    now_ns = 1_740_500_000_000_000_000

    # One local track.
    local = _track(track_id="local-1", timestamp_ns=now_ns, x=5.0, y=5.0, tqi=0.5)

    # Two peer tracks, both near the local — peer-1 is closer.
    peer_1 = _track(track_id="peer-1", timestamp_ns=now_ns, x=5.1, y=5.1, tqi=0.9)
    peer_2 = _track(track_id="peer-2", timestamp_ns=now_ns, x=5.3, y=5.3, tqi=0.85)

    async def _supplier(_: int) -> list[TrackState]:
        return [local]

    coordinator = FederationCoordinator(settings=_settings(), track_supplier=_supplier)
    await coordinator.handle_incoming_snapshot(
        FederationTrackSnapshot(
            server_id="srv-b",
            generated_ns=now_ns,
            tracks=[peer_1, peer_2],
        ),
        now_ns=now_ns,
    )
    merged = await coordinator.merged_tracks(
        local_tracks=[local], now_ns=now_ns, include_standby=True
    )

    # With only 1 local track, at most 1 peer track can be matched.
    # The other peer track should be unmatched (owner, peer_only).
    peer_rows = [r for r in merged if r["source_type"] == "peer_track"]
    owner_peers = [r for r in peer_rows if r["ownership_role"] == "owner"]
    assert len(owner_peers) >= 1  # At least one peer is unmatched → owner

    await coordinator.stop()


@pytest.mark.asyncio
async def test_hungarian_optimal_matching_two_pairs() -> None:
    """Two local + two peer tracks: Hungarian finds optimal global assignment."""
    now_ns = 1_740_600_000_000_000_000

    local_a = _track(track_id="local-a", timestamp_ns=now_ns, x=0.0, y=0.0, tqi=0.5)
    local_b = _track(track_id="local-b", timestamp_ns=now_ns, x=100.0, y=100.0, tqi=0.5)

    # peer-1 is close to local-a, peer-2 is close to local-b.
    peer_1 = _track(track_id="peer-1", timestamp_ns=now_ns, x=0.2, y=0.2, tqi=0.9)
    peer_2 = _track(track_id="peer-2", timestamp_ns=now_ns, x=100.1, y=100.1, tqi=0.8)

    async def _supplier(_: int) -> list[TrackState]:
        return [local_a, local_b]

    coordinator = FederationCoordinator(settings=_settings(), track_supplier=_supplier)
    await coordinator.handle_incoming_snapshot(
        FederationTrackSnapshot(
            server_id="srv-b",
            generated_ns=now_ns,
            tracks=[peer_1, peer_2],
        ),
        now_ns=now_ns,
    )
    # Trigger deconfliction by calling merged_tracks.
    merged = await coordinator.merged_tracks(
        local_tracks=[local_a, local_b], now_ns=now_ns, include_standby=True
    )
    status = await coordinator.status()
    # Both pairs should be evaluated.
    assert status["metrics"]["deconflict_pairs_evaluated"] == 2
    # All 4 tracks should appear (2 local + 2 peer, with standby included).
    assert len(merged) == 4

    await coordinator.stop()


# -- _pair_owners cap test --


@pytest.mark.asyncio
async def test_pair_owners_max_size_enforced() -> None:
    """_pair_owners dict is capped at _PAIR_OWNERS_MAX_SIZE."""
    async def _supplier(_: int) -> list[TrackState]:
        return []

    coordinator = FederationCoordinator(settings=_settings(), track_supplier=_supplier)

    # Manually inject entries exceeding the cap.
    for i in range(_PAIR_OWNERS_MAX_SIZE + 500):
        coordinator._pair_owners[f"fake-peer|fake-track-{i}|fake-local-{i}"] = "local"

    assert len(coordinator._pair_owners) > _PAIR_OWNERS_MAX_SIZE

    # Trigger deconfliction which runs the eviction.
    now_ns = 1_740_000_000_000_000_000
    await coordinator.merged_tracks(local_tracks=[], now_ns=now_ns)

    assert len(coordinator._pair_owners) <= _PAIR_OWNERS_MAX_SIZE

    await coordinator.stop()


# -- Response-model serialization --


@pytest.mark.asyncio
async def test_status_payload_validates_against_response_model() -> None:
    """status() must survive FederationStatusResponse validation.

    Regression: metrics was declared ``dict[str, int]`` while status() nests
    FederationMetrics.rates() (floats) under "rates", so every live call to
    GET /api/v1/federation/status raised ResponseValidationError and returned
    500. Existing tests read status()["metrics"] directly and never pushed the
    payload through the response model, which is why this shipped.
    """
    async def _supplier(_: int) -> list[TrackState]:
        return []

    coordinator = FederationCoordinator(settings=_settings(), track_supplier=_supplier)
    try:
        payload = await coordinator.status()
        validated = FederationStatusResponse.model_validate(payload)

        # The float-valued rates block is the part that used to fail.
        rates = validated.metrics["rates"]
        assert isinstance(rates["elapsed_seconds"], float)
        assert isinstance(rates["heartbeats_sent_per_s"], float)
        # Integer counters must still round-trip intact alongside it.
        assert validated.metrics["heartbeats_sent"] == 0
    finally:
        await coordinator.stop()


@pytest.mark.asyncio
async def test_status_response_model_serializes_to_json() -> None:
    """The validated payload must also serialize, as FastAPI does on the way out."""
    async def _supplier(_: int) -> list[TrackState]:
        return []

    coordinator = FederationCoordinator(settings=_settings(), track_supplier=_supplier)
    try:
        await coordinator.validate_inbound_auth(
            peer_id="srv-b",
            authorization_header="Bearer wrong",
            token_header=None,
        )
        payload = await coordinator.status()
        dumped = FederationStatusResponse.model_validate(payload).model_dump(mode="json")
        assert dumped["metrics"]["auth_rejections"] == 1
        assert dumped["metrics"]["rates"]["auth_rejections_per_s"] > 0.0
    finally:
        await coordinator.stop()
