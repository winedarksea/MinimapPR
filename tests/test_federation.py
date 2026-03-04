from __future__ import annotations

import pytest

from minimappr.config import FederationPeerConfig, Settings
from minimappr.core.federation import FederationCoordinator
from minimappr.models import FederationHeartbeat, FederationTrackSnapshot, TrackState


def _track(
    *,
    track_id: str,
    timestamp_ns: int,
    x: float,
    y: float,
    tqi: float,
    confidence: float = 0.7,
) -> TrackState:
    return TrackState(
        id=track_id,
        first_seen_ns=timestamp_ns - 2_000_000_000,
        last_seen_ns=timestamp_ns,
        position_m=(x, y, 0.0),
        position_covariance_m2=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        velocity_mps=(0.0, 0.0, 0.0),
        label="bird_like",
        label_category="wildlife",
        confidence=confidence,
        update_count=3,
        status="confirmed",
        tqi=tqi,
    )


def _settings(*, hysteresis: float = 0.05, ttl_seconds: float = 20.0) -> Settings:
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
    )


@pytest.mark.asyncio
async def test_peer_track_with_higher_tqi_becomes_owner() -> None:
    now_ns = 1_740_000_000_000_000_000
    local_track = _track(track_id="trk-local-1", timestamp_ns=now_ns, x=10.0, y=5.0, tqi=0.42)
    peer_track = _track(track_id="trk-peer-1", timestamp_ns=now_ns, x=10.2, y=5.1, tqi=0.88)

    async def _supplier(_: int) -> list[TrackState]:
        return [local_track]

    coordinator = FederationCoordinator(settings=_settings(), track_supplier=_supplier)
    accepted = await coordinator.handle_incoming_snapshot(
        FederationTrackSnapshot(
            server_id="srv-b",
            generated_ns=now_ns,
            tracks=[peer_track],
        ),
        now_ns=now_ns,
    )
    assert accepted is True

    owner_view = await coordinator.merged_tracks(
        local_tracks=[local_track],
        now_ns=now_ns,
        include_standby=False,
    )
    assert len(owner_view) == 1
    assert owner_view[0]["source_type"] == "peer_track"
    assert owner_view[0]["id"] == "srv-b:trk-peer-1"
    assert owner_view[0]["ownership_role"] == "owner"

    full_view = await coordinator.merged_tracks(
        local_tracks=[local_track],
        now_ns=now_ns,
        include_standby=True,
    )
    by_source = {row["source_type"]: row for row in full_view}
    assert by_source["local_track"]["ownership_role"] == "standby"
    assert by_source["peer_track"]["ownership_role"] == "owner"

    await coordinator.stop()


@pytest.mark.asyncio
async def test_tqi_hysteresis_holds_previous_owner() -> None:
    now_ns = 1_740_100_000_000_000_000
    local_track = _track(track_id="trk-local-1", timestamp_ns=now_ns, x=2.0, y=2.0, tqi=0.6)
    peer_track = _track(track_id="trk-peer-1", timestamp_ns=now_ns, x=2.1, y=2.1, tqi=0.62)

    async def _supplier(_: int) -> list[TrackState]:
        return [local_track]

    coordinator = FederationCoordinator(settings=_settings(hysteresis=0.05), track_supplier=_supplier)

    await coordinator.handle_incoming_snapshot(
        FederationTrackSnapshot(
            server_id="srv-b",
            generated_ns=now_ns,
            tracks=[peer_track],
        ),
        now_ns=now_ns,
    )
    first_view = await coordinator.merged_tracks(local_tracks=[local_track], now_ns=now_ns, include_standby=False)
    assert first_view[0]["source_type"] == "local_track"

    stronger_peer = _track(track_id="trk-peer-1", timestamp_ns=now_ns + 1_000_000_000, x=2.1, y=2.1, tqi=0.8)
    await coordinator.handle_incoming_snapshot(
        FederationTrackSnapshot(
            server_id="srv-b",
            generated_ns=now_ns + 1_000_000_000,
            tracks=[stronger_peer],
        ),
        now_ns=now_ns + 1_000_000_000,
    )
    second_view = await coordinator.merged_tracks(
        local_tracks=[local_track],
        now_ns=now_ns + 1_000_000_000,
        include_standby=False,
    )
    assert second_view[0]["source_type"] == "peer_track"

    near_equal_peer = _track(track_id="trk-peer-1", timestamp_ns=now_ns + 2_000_000_000, x=2.1, y=2.1, tqi=0.62)
    await coordinator.handle_incoming_snapshot(
        FederationTrackSnapshot(
            server_id="srv-b",
            generated_ns=now_ns + 2_000_000_000,
            tracks=[near_equal_peer],
        ),
        now_ns=now_ns + 2_000_000_000,
    )
    third_view = await coordinator.merged_tracks(
        local_tracks=[local_track],
        now_ns=now_ns + 2_000_000_000,
        include_standby=False,
    )
    assert third_view[0]["source_type"] == "peer_track"
    assert third_view[0]["ownership_reason"] == "hysteresis_hold"

    await coordinator.stop()


@pytest.mark.asyncio
async def test_snapshot_tracks_expire_after_ttl() -> None:
    now_ns = 1_740_200_000_000_000_000
    local_track = _track(track_id="trk-local-1", timestamp_ns=now_ns, x=0.0, y=0.0, tqi=0.6)
    peer_track = _track(track_id="trk-peer-1", timestamp_ns=now_ns, x=30.0, y=30.0, tqi=0.7)

    async def _supplier(_: int) -> list[TrackState]:
        return [local_track]

    coordinator = FederationCoordinator(settings=_settings(ttl_seconds=0.01), track_supplier=_supplier)
    await coordinator.handle_incoming_snapshot(
        FederationTrackSnapshot(
            server_id="srv-b",
            generated_ns=now_ns,
            tracks=[peer_track],
        ),
        now_ns=now_ns,
    )
    fresh = await coordinator.merged_tracks(local_tracks=[local_track], now_ns=now_ns, include_standby=False)
    assert any(track["source_type"] == "peer_track" for track in fresh)

    expired = await coordinator.merged_tracks(
        local_tracks=[local_track],
        now_ns=now_ns + 30_000_000,
        include_standby=False,
    )
    assert all(track["source_type"] != "peer_track" for track in expired)

    await coordinator.stop()


@pytest.mark.asyncio
async def test_inbound_auth_uses_peer_api_key() -> None:
    async def _supplier(_: int) -> list[TrackState]:
        return []

    coordinator = FederationCoordinator(settings=_settings(), track_supplier=_supplier)
    assert await coordinator.validate_inbound_auth(
        peer_id="srv-b",
        authorization_header="Bearer peer-token",
        token_header=None,
    )
    assert not await coordinator.validate_inbound_auth(
        peer_id="srv-b",
        authorization_header="Bearer wrong",
        token_header=None,
    )

    await coordinator.stop()


@pytest.mark.asyncio
async def test_auth_rejection_increments_metric() -> None:
    """Wrong bearer token increments the auth_rejections metric."""
    async def _supplier(_: int) -> list[TrackState]:
        return []

    coordinator = FederationCoordinator(settings=_settings(), track_supplier=_supplier)
    result = await coordinator.validate_inbound_auth(
        peer_id="srv-b",
        authorization_header="Bearer totally-wrong",
        token_header=None,
    )
    assert result is False
    status = await coordinator.status()
    assert status["metrics"]["auth_rejections"] == 1
    await coordinator.stop()


@pytest.mark.asyncio
async def test_auth_bypass_when_no_token_configured() -> None:
    """When neither peer api_key nor global auth_token is set, all requests pass."""
    settings = Settings(
        federation_enabled=True,
        federation_server_id="srv-a",
        federation_peers=(
            FederationPeerConfig(peer_id="srv-b", base_url="http://127.0.0.1:19999"),
        ),
        # federation_auth_token defaults to "" → resolves to None
    )

    async def _supplier(_: int) -> list[TrackState]:
        return []

    coordinator = FederationCoordinator(settings=settings, track_supplier=_supplier)
    assert await coordinator.validate_inbound_auth(
        peer_id="srv-b", authorization_header=None, token_header=None
    )
    assert await coordinator.validate_inbound_auth(
        peer_id="srv-b", authorization_header="Bearer anything", token_header=None
    )
    status = await coordinator.status()
    assert status["metrics"]["auth_rejections"] == 0
    await coordinator.stop()


@pytest.mark.asyncio
async def test_unknown_peer_heartbeat_not_accepted() -> None:
    """Heartbeat from a server_id not in the peer list is rejected and counted."""
    async def _supplier(_: int) -> list[TrackState]:
        return []

    coordinator = FederationCoordinator(settings=_settings(), track_supplier=_supplier)
    now_ns = 1_740_700_000_000_000_000
    accepted = await coordinator.handle_incoming_heartbeat(
        FederationHeartbeat(server_id="unknown-ghost", sent_ns=now_ns),
        now_ns=now_ns,
    )
    assert accepted is False
    status = await coordinator.status()
    assert status["metrics"]["unknown_peer_messages"] == 1
    await coordinator.stop()


@pytest.mark.asyncio
async def test_unknown_peer_snapshot_not_accepted() -> None:
    """Snapshot from a server_id not in the peer list is rejected and counted."""
    async def _supplier(_: int) -> list[TrackState]:
        return []

    coordinator = FederationCoordinator(settings=_settings(), track_supplier=_supplier)
    now_ns = 1_740_800_000_000_000_000
    ghost_track = _track(track_id="ghost-1", timestamp_ns=now_ns, x=0.0, y=0.0, tqi=0.5)
    accepted = await coordinator.handle_incoming_snapshot(
        FederationTrackSnapshot(
            server_id="unknown-ghost",
            generated_ns=now_ns,
            tracks=[ghost_track],
        ),
        now_ns=now_ns,
    )
    assert accepted is False
    status = await coordinator.status()
    assert status["metrics"]["unknown_peer_messages"] == 1
    await coordinator.stop()


@pytest.mark.asyncio
async def test_non_overlapping_peer_tracks_appear_as_owners() -> None:
    """Peer tracks far from all local tracks are independent owners — no deconfliction."""
    now_ns = 1_740_900_000_000_000_000
    local = _track(track_id="local-1", timestamp_ns=now_ns, x=0.0, y=0.0, tqi=0.7)
    peer_far = _track(track_id="peer-far-1", timestamp_ns=now_ns, x=1000.0, y=1000.0, tqi=0.5)

    async def _supplier(_: int) -> list[TrackState]:
        return [local]

    coordinator = FederationCoordinator(settings=_settings(), track_supplier=_supplier)
    await coordinator.handle_incoming_snapshot(
        FederationTrackSnapshot(server_id="srv-b", generated_ns=now_ns, tracks=[peer_far]),
        now_ns=now_ns,
    )
    merged = await coordinator.merged_tracks(local_tracks=[local], now_ns=now_ns)
    source_types = {row["source_type"] for row in merged}
    assert "local_track" in source_types
    assert "peer_track" in source_types
    for row in merged:
        assert row["ownership_role"] == "owner"
    await coordinator.stop()


@pytest.mark.asyncio
async def test_stale_peer_track_drops_increment_metric() -> None:
    """peer_tracks_stale_dropped counter increments once TTL elapses."""
    now_ns = 1_741_000_000_000_000_000
    peer_track = _track(track_id="trk-stale", timestamp_ns=now_ns, x=5.0, y=5.0, tqi=0.6)

    async def _supplier(_: int) -> list[TrackState]:
        return []

    coordinator = FederationCoordinator(
        settings=_settings(ttl_seconds=0.001), track_supplier=_supplier
    )
    await coordinator.handle_incoming_snapshot(
        FederationTrackSnapshot(server_id="srv-b", generated_ns=now_ns, tracks=[peer_track]),
        now_ns=now_ns,
    )
    # Advance past TTL (0.001 s = 1_000_000 ns)
    future_ns = now_ns + 10_000_000_000
    _ = await coordinator.merged_tracks(local_tracks=[], now_ns=future_ns)
    status = await coordinator.status()
    assert status["metrics"]["peer_tracks_stale_dropped"] >= 1
    await coordinator.stop()


@pytest.mark.asyncio
async def test_merged_tracks_respects_limit() -> None:
    """The limit parameter truncates the merged track list."""
    now_ns = 1_741_100_000_000_000_000
    local_tracks = [
        _track(track_id=f"local-{i}", timestamp_ns=now_ns, x=float(i * 500), y=0.0, tqi=0.5)
        for i in range(5)
    ]

    async def _supplier(_: int) -> list[TrackState]:
        return local_tracks

    coordinator = FederationCoordinator(settings=_settings(), track_supplier=_supplier)
    full = await coordinator.merged_tracks(local_tracks=local_tracks, now_ns=now_ns, limit=0)
    assert len(full) == 5
    limited = await coordinator.merged_tracks(local_tracks=local_tracks, now_ns=now_ns, limit=2)
    assert len(limited) == 2
    await coordinator.stop()


@pytest.mark.asyncio
async def test_deconflict_metrics_increment() -> None:
    """Deconfliction metrics track evaluated pairs and standby assignments."""
    now_ns = 1_741_200_000_000_000_000
    local = _track(track_id="local-1", timestamp_ns=now_ns, x=1.0, y=1.0, tqi=0.5)
    peer = _track(track_id="peer-1", timestamp_ns=now_ns, x=1.1, y=1.1, tqi=0.9)

    async def _supplier(_: int) -> list[TrackState]:
        return [local]

    coordinator = FederationCoordinator(settings=_settings(), track_supplier=_supplier)
    await coordinator.handle_incoming_snapshot(
        FederationTrackSnapshot(server_id="srv-b", generated_ns=now_ns, tracks=[peer]),
        now_ns=now_ns,
    )
    await coordinator.merged_tracks(local_tracks=[local], now_ns=now_ns)
    status = await coordinator.status()
    assert status["metrics"]["deconflict_pairs_evaluated"] >= 1
    assert (
        status["metrics"]["deconflict_standby_local"] >= 1
        or status["metrics"]["deconflict_standby_peer"] >= 1
    )
    await coordinator.stop()
