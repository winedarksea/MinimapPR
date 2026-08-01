"""Tests for batch-transaction integrity on the shared SQLite connection.

sqlite3 leaves its implicit DML transaction open when the commit that should
have closed it raises. Both regressions here start there: an explicit ``BEGIN``
on top of that open transaction raises "cannot start a transaction within a
transaction" for every batch that follows, and a cancelled batch used to pin
``_batch_depth`` above zero so nothing ever committed again.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from minimappr.models import NodeSpec, NodeType
from minimappr.storage.db import Storage


def _node(node_id: str = "batch-node") -> NodeSpec:
    return NodeSpec(
        id=node_id,
        node_type=NodeType.POINT,
        position_m=(0.0, 0.0, 0.0),
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
    )


async def _node_ids(db_path: Path) -> set[str]:
    """Read through a fresh connection so only committed rows are visible."""
    reopened = Storage(db_path)
    await reopened.initialize()
    try:
        return {str(row["id"]) for row in await reopened.list_nodes()}
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_batch_recovers_after_a_commit_fails(tmp_path: Path) -> None:
    """One transient `database is locked` must not wedge the connection."""
    storage = Storage(tmp_path / "commit-failure.db")
    await storage.initialize()
    db = storage._require_db()
    real_commit = db.commit
    armed = True

    async def flaky_commit() -> None:
        nonlocal armed
        if armed:
            armed = False
            raise sqlite3.OperationalError("database is locked")
        await real_commit()

    db.commit = flaky_commit  # type: ignore[method-assign]

    with pytest.raises(sqlite3.OperationalError):
        async with storage.begin_batch():
            await storage.upsert_node(_node("wedged"), last_seen_ns=1)

    # The failed commit left the transaction open; the next batch must clear it
    # rather than raising "cannot start a transaction within a transaction".
    async with storage.begin_batch():
        await storage.upsert_node(_node("after-recovery"), last_seen_ns=2)

    assert storage._batch_depth == 0
    await storage.close()
    assert await _node_ids(storage.db_path) == {"after-recovery"}


@pytest.mark.asyncio
async def test_cancelled_batch_does_not_pin_the_batch_depth(tmp_path: Path) -> None:
    """CancelledError is a BaseException, so it skipped the old cleanup branch."""
    storage = Storage(tmp_path / "cancelled.db")
    await storage.initialize()

    async def worker() -> None:
        async with storage.begin_batch():
            await storage.upsert_node(_node("abandoned"), last_seen_ns=1)
            await asyncio.sleep(10)

    task = asyncio.create_task(worker())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert storage._batch_depth == 0

    # A pinned depth made _commit_if_needed a no-op for the life of the process.
    await storage.upsert_node(_node("after-cancel"), last_seen_ns=2)
    await storage.close()
    assert await _node_ids(storage.db_path) == {"after-cancel"}


@pytest.mark.asyncio
async def test_nested_batches_commit_once_at_the_outermost_exit(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "nested.db")
    await storage.initialize()
    db = storage._require_db()
    real_commit = db.commit
    commits = 0

    async def counting_commit() -> None:
        nonlocal commits
        commits += 1
        await real_commit()

    db.commit = counting_commit  # type: ignore[method-assign]

    async with storage.begin_batch():
        await storage.upsert_node(_node("outer"), last_seen_ns=1)
        async with storage.begin_batch():
            await storage.upsert_node(_node("inner"), last_seen_ns=2)
        # The inner exit must leave the transaction open for the outer frame.
        assert commits == 0

    assert commits == 1
    assert storage._batch_depth == 0
    await storage.close()
    assert await _node_ids(storage.db_path) == {"outer", "inner"}


@pytest.mark.asyncio
async def test_failed_batch_rolls_back_its_writes(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "rollback.db")
    await storage.initialize()

    with pytest.raises(RuntimeError):
        async with storage.begin_batch():
            await storage.upsert_node(_node("discarded"), last_seen_ns=1)
            raise RuntimeError("stage failed")

    assert storage._batch_depth == 0
    await storage.close()
    assert await _node_ids(storage.db_path) == set()
