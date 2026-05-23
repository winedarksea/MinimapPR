"""Tests for SQLite stale-state recovery on startup.

Covers the WAL/SHM hygiene path that prevents "database is locked" failures on
the first write-side PRAGMA after an unclean shutdown left an oversized but
empty WAL or a stale SHM file behind.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite
import pytest

from minimappr.storage.db import Storage


async def _create_initialized_db(tmp_path: Path, name: str = "store.db") -> Path:
    db_path = tmp_path / name
    storage = Storage(db_path)
    await storage.initialize()
    await storage._require_db().close()  # type: ignore[union-attr]
    storage._db = None
    return db_path


@pytest.mark.asyncio
async def test_proactive_checkpoint_drains_oversized_empty_wal(tmp_path: Path) -> None:
    """Simulates the production failure mode: a previously-initialized DB whose
    -wal file is huge but contains zero committed frames (unclean shutdown after
    WAL pre-extension). _configure_connection must drain it without erroring."""
    db_path = await _create_initialized_db(tmp_path)
    wal_path = Path(f"{db_path}-wal")

    # Pre-extend the WAL with zeros to simulate the leftover pre-allocated space.
    with open(wal_path, "ab") as f:
        f.write(b"\x00" * (4 * 1024 * 1024))
    assert wal_path.stat().st_size > 0

    storage = Storage(db_path)
    await storage.initialize()
    try:
        assert wal_path.stat().st_size == 0
        assert storage.stale_recoveries_count == 0  # checkpoint succeeded, no recovery needed
    finally:
        await storage._require_db().close()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_initialize_succeeds_with_stale_shm(tmp_path: Path) -> None:
    """A stale -shm file must not block startup. SQLite rebuilds it from -wal."""
    db_path = await _create_initialized_db(tmp_path)
    shm_path = Path(f"{db_path}-shm")

    # Stomp on the SHM with junk; SQLite will detect mismatch and rebuild.
    shm_path.write_bytes(b"\xff" * 32768)

    storage = Storage(db_path)
    await storage.initialize()
    try:
        # Either the proactive checkpoint handled it or graduated recovery did.
        # In both cases, startup completed and we can query.
        cursor = await storage._require_db().execute("SELECT COUNT(*) FROM nodes;")  # type: ignore[union-attr]
        row = await cursor.fetchone()
        await cursor.close()
        assert row is not None
    finally:
        await storage._require_db().close()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_recovery_preserves_committed_wal_frames(tmp_path: Path) -> None:
    """If the WAL has real committed frames, recovery must NOT delete it —
    the checkpoint path drains them into the main DB instead. Verifies no
    data is lost across the recovery."""
    db_path = await _create_initialized_db(tmp_path)

    # Write a row via raw sqlite + WAL mode, but don't checkpoint so the data
    # is still in the WAL when we test recovery.
    conn = await aiosqlite.connect(db_path)
    try:
        await conn.execute("PRAGMA journal_mode=WAL;")
        await conn.execute("PRAGMA wal_autocheckpoint=0;")
        await conn.execute(
            """
            INSERT INTO nodes (
                id, node_type, x, y, z, sensor_offsets_json, capabilities_json,
                metadata_json, last_seen_ns
            ) VALUES ('test-node', 'point', 0, 0, 0, '[]', '[]', '{}', 1)
            """
        )
        await conn.commit()
    finally:
        await conn.close()

    # Force-trigger recovery to ensure the safety path is exercised even when
    # the WAL holds committed data.
    storage = Storage(db_path)
    trigger = sqlite3.OperationalError("database is locked")
    recovered = await storage._recover_stale_sqlite_state(trigger)

    # Recovery may succeed (via checkpoint draining the frames) or be a no-op,
    # but it must NEVER lose the row.
    assert recovered in (True, False)
    storage = Storage(db_path)
    await storage.initialize()
    try:
        cursor = await storage._require_db().execute(  # type: ignore[union-attr]
            "SELECT id FROM nodes WHERE id = 'test-node';"
        )
        row = await cursor.fetchone()
        await cursor.close()
        assert row is not None, "recovery dropped committed data"
        assert row[0] == "test-node"
    finally:
        await storage._require_db().close()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_recovery_deletes_provably_empty_wal_after_shm_removal_fails(
    tmp_path: Path,
) -> None:
    """If the SHM is gone and the WAL is provably empty of committed frames,
    step C deletes the WAL and reports success."""
    db_path = await _create_initialized_db(tmp_path)
    wal_path = Path(f"{db_path}-wal")
    shm_path = Path(f"{db_path}-shm")

    # Remove SHM and pre-extend WAL with zeros (no committed frames).
    if shm_path.exists():
        shm_path.unlink()
    with open(wal_path, "wb") as f:
        f.write(b"\x00" * (1024 * 1024))

    storage = Storage(db_path)
    trigger = sqlite3.OperationalError("database is locked")
    ok = await storage._recover_stale_sqlite_state(trigger)
    assert ok is True
    assert storage.stale_recoveries_count == 1
    assert wal_path.stat().st_size == 0 or not wal_path.exists()


@pytest.mark.asyncio
async def test_recovery_returns_false_when_db_missing(tmp_path: Path) -> None:
    """Recovery is a no-op (returns False) when there is no DB file to recover."""
    storage = Storage(tmp_path / "missing.db")
    trigger = sqlite3.OperationalError("database is locked")
    assert await storage._recover_stale_sqlite_state(trigger) is False
    assert storage.stale_recoveries_count == 0


@pytest.mark.asyncio
async def test_initialize_retries_after_configure_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """initialize() must invoke recovery and retry when _configure_connection
    raises 'database is locked' — the actual failure site from the production bug."""
    storage = Storage(tmp_path / "lock_retry.db")
    call_counts = {"open": 0, "configure": 0, "schema": 0, "recover": 0}

    async def _open_connection_stub() -> None:
        call_counts["open"] += 1

    async def _configure_connection_stub() -> None:
        call_counts["configure"] += 1
        if call_counts["configure"] == 1:
            raise sqlite3.OperationalError("database is locked")

    async def _initialize_schema_stub() -> None:
        call_counts["schema"] += 1

    async def _recover_stub(_exc: sqlite3.OperationalError) -> bool:
        call_counts["recover"] += 1
        return True

    async def _ensure_av_stub() -> None:
        pass

    monkeypatch.setattr(storage, "_open_connection", _open_connection_stub)
    monkeypatch.setattr(storage, "_configure_connection", _configure_connection_stub)
    monkeypatch.setattr(storage, "_initialize_schema_and_migrations", _initialize_schema_stub)
    monkeypatch.setattr(storage, "_recover_stale_sqlite_state", _recover_stub)
    monkeypatch.setattr(storage, "_ensure_incremental_auto_vacuum", _ensure_av_stub)

    await storage.initialize()

    assert call_counts == {"open": 2, "configure": 2, "schema": 1, "recover": 1}


@pytest.mark.asyncio
async def test_initialize_retries_schema_after_disk_io_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserved behavior from before: schema migration disk I/O errors trigger
    recovery + retry."""
    storage = Storage(tmp_path / "schema_retry.db")
    call_counts = {"open": 0, "configure": 0, "schema": 0, "recover": 0}

    async def _open_connection_stub() -> None:
        call_counts["open"] += 1

    async def _configure_connection_stub() -> None:
        call_counts["configure"] += 1

    async def _initialize_schema_stub() -> None:
        call_counts["schema"] += 1
        if call_counts["schema"] == 1:
            raise sqlite3.OperationalError("disk I/O error")

    async def _recover_stub(_exc: sqlite3.OperationalError) -> bool:
        call_counts["recover"] += 1
        return True

    async def _ensure_av_stub() -> None:
        pass

    monkeypatch.setattr(storage, "_open_connection", _open_connection_stub)
    monkeypatch.setattr(storage, "_configure_connection", _configure_connection_stub)
    monkeypatch.setattr(storage, "_initialize_schema_and_migrations", _initialize_schema_stub)
    monkeypatch.setattr(storage, "_recover_stale_sqlite_state", _recover_stub)
    monkeypatch.setattr(storage, "_ensure_incremental_auto_vacuum", _ensure_av_stub)

    await storage.initialize()

    assert call_counts == {"open": 2, "configure": 2, "schema": 2, "recover": 1}


@pytest.mark.asyncio
async def test_initialize_propagates_non_recoverable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Errors other than 'disk I/O error' or 'database is locked' must not be
    silently swallowed — they propagate to the caller."""
    storage = Storage(tmp_path / "fatal.db")

    async def _open_connection_stub() -> None:
        pass

    async def _configure_connection_stub() -> None:
        raise sqlite3.OperationalError("no such table: corruption")

    monkeypatch.setattr(storage, "_open_connection", _open_connection_stub)
    monkeypatch.setattr(storage, "_configure_connection", _configure_connection_stub)

    with pytest.raises(sqlite3.OperationalError, match="corruption"):
        await storage.initialize()


@pytest.mark.asyncio
async def test_clean_initialize_does_not_trigger_recovery(tmp_path: Path) -> None:
    """First-time startup on an empty directory must not invoke recovery."""
    storage = Storage(tmp_path / "fresh.db")
    await storage.initialize()
    try:
        assert storage.stale_recoveries_count == 0
    finally:
        await storage._require_db().close()  # type: ignore[union-attr]
