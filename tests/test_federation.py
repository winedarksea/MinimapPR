from __future__ import annotations

import pytest

from minimappr.config import FederationPeerConfig, Settings
from minimappr.core.federation import FederationCoordinator
from minimappr.models import FederationTrackSnapshot, TrackState


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
