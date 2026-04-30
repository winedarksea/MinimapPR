from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class JournalConsumerCursor:
    journal_epoch: int
    last_fully_processed_journal_sequence: int


class JournalConsumerStateStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def ensure_initialized(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS consumer_cursors (
                    consumer_name TEXT NOT NULL,
                    stream_key TEXT NOT NULL,
                    journal_epoch INTEGER NOT NULL,
                    last_fully_processed_journal_sequence INTEGER NOT NULL,
                    updated_ns INTEGER NOT NULL,
                    PRIMARY KEY (consumer_name, stream_key)
                );

                CREATE TABLE IF NOT EXISTS consumer_exceptions (
                    consumer_name TEXT NOT NULL,
                    journal_id TEXT NOT NULL,
                    stream_key TEXT NOT NULL,
                    journal_epoch INTEGER NOT NULL,
                    journal_sequence INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    detail TEXT,
                    handled_ns INTEGER NOT NULL,
                    PRIMARY KEY (consumer_name, journal_id)
                );
                """
            )

    def load_cursors(self, consumer_name: str) -> dict[str, JournalConsumerCursor]:
        self.ensure_initialized()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT stream_key, journal_epoch, last_fully_processed_journal_sequence
                FROM consumer_cursors
                WHERE consumer_name = ?
                """,
                (consumer_name,),
            ).fetchall()
        return {
            str(row["stream_key"]): JournalConsumerCursor(
                journal_epoch=int(row["journal_epoch"]),
                last_fully_processed_journal_sequence=int(row["last_fully_processed_journal_sequence"]),
            )
            for row in rows
        }

    def get_cursor(self, consumer_name: str, stream_key: str) -> JournalConsumerCursor | None:
        self.ensure_initialized()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT journal_epoch, last_fully_processed_journal_sequence
                FROM consumer_cursors
                WHERE consumer_name = ? AND stream_key = ?
                """,
                (consumer_name, stream_key),
            ).fetchone()
        if row is None:
            return None
        return JournalConsumerCursor(
            journal_epoch=int(row["journal_epoch"]),
            last_fully_processed_journal_sequence=int(row["last_fully_processed_journal_sequence"]),
        )

    def load_exception_ids(self, consumer_name: str) -> set[str]:
        self.ensure_initialized()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT journal_id
                FROM consumer_exceptions
                WHERE consumer_name = ?
                """,
                (consumer_name,),
            ).fetchall()
        return {str(row["journal_id"]) for row in rows}

    def mark_handled(
        self,
        *,
        consumer_name: str,
        stream_key: str,
        journal_epoch: int,
        journal_sequence: int,
        journal_id: str,
        status: str,
        handled_ns: int,
        detail: str | None = None,
    ) -> None:
        self.ensure_initialized()
        with self._connect() as connection:
            existing_row = connection.execute(
                """
                SELECT journal_epoch, last_fully_processed_journal_sequence
                FROM consumer_cursors
                WHERE consumer_name = ? AND stream_key = ?
                """,
                (consumer_name, stream_key),
            ).fetchone()

            if existing_row is None or _entry_is_newer(
                int(existing_row["journal_epoch"]),
                int(existing_row["last_fully_processed_journal_sequence"]),
                journal_epoch,
                journal_sequence,
            ):
                connection.execute(
                    """
                    INSERT INTO consumer_cursors (
                        consumer_name,
                        stream_key,
                        journal_epoch,
                        last_fully_processed_journal_sequence,
                        updated_ns
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(consumer_name, stream_key) DO UPDATE SET
                        journal_epoch = excluded.journal_epoch,
                        last_fully_processed_journal_sequence = excluded.last_fully_processed_journal_sequence,
                        updated_ns = excluded.updated_ns
                    """,
                    (consumer_name, stream_key, journal_epoch, journal_sequence, handled_ns),
                )

            if status != "processed":
                connection.execute(
                    """
                    INSERT OR REPLACE INTO consumer_exceptions (
                        consumer_name,
                        journal_id,
                        stream_key,
                        journal_epoch,
                        journal_sequence,
                        status,
                        detail,
                        handled_ns
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        consumer_name,
                        journal_id,
                        stream_key,
                        journal_epoch,
                        journal_sequence,
                        status,
                        detail,
                        handled_ns,
                    ),
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection


def _entry_is_newer(
    existing_epoch: int,
    existing_sequence: int,
    candidate_epoch: int,
    candidate_sequence: int,
) -> bool:
    if candidate_epoch != existing_epoch:
        return candidate_epoch > existing_epoch
    return candidate_sequence > existing_sequence