from __future__ import annotations

import asyncio

import pytest

from minimappr.core.live_ingest_state import FrameIdentity, LiveIngestState
from minimappr.models import NodeSpec, NodeType


def _node(
    *,
    position_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    capabilities: list[str] | None = None,
) -> NodeSpec:
    return NodeSpec(
        id="node-1",
        node_type=NodeType.POINT,
        position_m=position_m,
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=capabilities or ["audio"],
    )


def _identity() -> FrameIdentity:
    return FrameIdentity.from_frame(
        node_id="node-1",
        boot_session="boot-1",
        source_type="raw_sensor",
        start_sample_index=0,
        end_sample_index=512,
        start_time_ns=1_000_000_000,
        frame_sequence=1,
    )


@pytest.mark.asyncio
async def test_live_frame_claim_is_atomic_and_process_local() -> None:
    state = LiveIngestState()

    claims = await asyncio.gather(
        state.claim_processed_frame(_identity()),
        state.claim_processed_frame(_identity()),
    )

    assert sorted(claims) == [False, True]
    assert await LiveIngestState().claim_processed_frame(_identity()) is True


@pytest.mark.asyncio
async def test_reserved_frame_is_retryable_until_buffer_insertion_commits() -> None:
    state = LiveIngestState()
    identity = _identity()

    assert await state.reserve_frame(identity)
    assert not await state.reserve_frame(identity)

    await state.release_reserved_frame(identity)
    assert await state.reserve_frame(identity)
    await state.commit_reserved_frame(identity)
    assert not await state.reserve_frame(identity)


@pytest.mark.asyncio
async def test_environment_persistence_is_limited_to_one_sample_per_minute() -> None:
    state = LiveIngestState(environment_persistence_interval_seconds=60.0)

    assert await state.should_persist_environment_sample(node_id="node-1", timestamp_ns=1_000_000_000)
    assert not await state.should_persist_environment_sample(node_id="node-1", timestamp_ns=59_999_999_999)
    assert await state.should_persist_environment_sample(node_id="node-1", timestamp_ns=60_000_000_000)


@pytest.mark.asyncio
async def test_unchanged_registration_still_refreshes_the_persisted_heartbeat() -> None:
    """A frozen last_seen_ns reads as "offline" wherever the live registry is absent."""
    state = LiveIngestState(node_liveness_persistence_interval_seconds=5.0)
    node = _node()

    assert await state.should_persist_node_row(node, now_ns=1_000_000_000)
    # Same registration inside the interval: no write, the row is still fresh.
    assert not await state.should_persist_node_row(node, now_ns=3_000_000_000)
    assert not await state.should_persist_node_row(node, now_ns=5_999_999_999)
    # Past the interval the heartbeat is written even though nothing changed.
    assert await state.should_persist_node_row(node, now_ns=6_000_000_000)
    assert not await state.should_persist_node_row(node, now_ns=7_000_000_000)


@pytest.mark.asyncio
async def test_position_churn_alone_does_not_trigger_a_write() -> None:
    """Per-frame position is excluded from the fingerprint and must stay excluded."""
    state = LiveIngestState(node_liveness_persistence_interval_seconds=5.0)

    assert await state.should_persist_node_row(_node(position_m=(0.0, 0.0, 0.0)), now_ns=1_000_000_000)
    assert not await state.should_persist_node_row(
        _node(position_m=(1.0, 2.0, 3.0)), now_ns=1_500_000_000
    )


@pytest.mark.asyncio
async def test_reconfiguration_persists_immediately_and_restarts_the_interval() -> None:
    state = LiveIngestState(node_liveness_persistence_interval_seconds=5.0)
    node = _node()
    reconfigured = _node(capabilities=["audio", "temperature"])

    assert await state.should_persist_node_row(node, now_ns=1_000_000_000)
    assert await state.should_persist_node_row(reconfigured, now_ns=1_100_000_000)
    assert not await state.should_persist_node_row(reconfigured, now_ns=2_000_000_000)
