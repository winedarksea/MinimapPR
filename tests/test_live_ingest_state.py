from __future__ import annotations

import asyncio

import pytest

from minimappr.core.live_ingest_state import FrameIdentity, LiveIngestState


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
async def test_environment_persistence_is_limited_to_one_sample_per_minute() -> None:
    state = LiveIngestState(environment_persistence_interval_seconds=60.0)

    assert await state.should_persist_environment_sample(node_id="node-1", timestamp_ns=1_000_000_000)
    assert not await state.should_persist_environment_sample(node_id="node-1", timestamp_ns=59_999_999_999)
    assert await state.should_persist_environment_sample(node_id="node-1", timestamp_ns=60_000_000_000)
