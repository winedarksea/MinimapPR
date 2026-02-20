"""SQLite persistence layer."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import aiosqlite

from minimappr.models import DetectionEvent, NodeSpec, TrackState


class Storage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.execute("PRAGMA synchronous=NORMAL;")

        await self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                sensor_offsets_json TEXT NOT NULL,
                capabilities_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                last_seen_ns INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS detections (
                id TEXT PRIMARY KEY,
                timestamp_ns INTEGER NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                confidence REAL NOT NULL,
                gdop REAL NOT NULL,
                label TEXT NOT NULL,
                label_confidence REAL NOT NULL,
                track_id TEXT,
                reference_sensor TEXT NOT NULL,
                source_sensors_json TEXT NOT NULL,
                tdoa_json TEXT NOT NULL,
                classifier_scores_json TEXT NOT NULL,
                feature_summary_json TEXT NOT NULL,
                snippet_path TEXT,
                snippet_expires_ns INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_detections_ts ON detections(timestamp_ns DESC);

            CREATE TABLE IF NOT EXISTS tracks (
                id TEXT PRIMARY KEY,
                first_seen_ns INTEGER NOT NULL,
                last_seen_ns INTEGER NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                vx REAL NOT NULL,
                vy REAL NOT NULL,
                vz REAL NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                update_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                tqi REAL NOT NULL DEFAULT 0.0
            );

            CREATE INDEX IF NOT EXISTS idx_tracks_last_seen ON tracks(last_seen_ns DESC);
            """
        )
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def upsert_node(self, spec: NodeSpec, last_seen_ns: int) -> None:
        db = self._require_db()
        async with self._lock:
            await db.execute(
                """
                INSERT INTO nodes (id, node_type, x, y, z, sensor_offsets_json, capabilities_json, metadata_json, last_seen_ns)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    node_type=excluded.node_type,
                    x=excluded.x,
                    y=excluded.y,
                    z=excluded.z,
                    sensor_offsets_json=excluded.sensor_offsets_json,
                    capabilities_json=excluded.capabilities_json,
                    metadata_json=excluded.metadata_json,
                    last_seen_ns=excluded.last_seen_ns
                """,
                (
                    spec.id,
                    spec.node_type.value,
                    spec.position_m[0],
                    spec.position_m[1],
                    spec.position_m[2],
                    json.dumps(spec.sensor_offsets_m),
                    json.dumps(spec.capabilities),
                    json.dumps(spec.metadata),
                    last_seen_ns,
                ),
            )
            await db.commit()

    async def insert_detection(
        self,
        detection: DetectionEvent,
        snippet_path: str | None,
        snippet_expires_ns: int | None,
    ) -> None:
        db = self._require_db()
        async with self._lock:
            await db.execute(
                """
                INSERT INTO detections (
                    id, timestamp_ns, x, y, z, confidence, gdop, label, label_confidence,
                    track_id, reference_sensor, source_sensors_json, tdoa_json,
                    classifier_scores_json, feature_summary_json, snippet_path, snippet_expires_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    detection.id,
                    detection.timestamp_ns,
                    detection.position_m[0],
                    detection.position_m[1],
                    detection.position_m[2],
                    detection.confidence,
                    detection.gdop,
                    detection.label,
                    detection.label_confidence,
                    detection.track_id,
                    detection.reference_sensor,
                    json.dumps(detection.source_sensors),
                    json.dumps(detection.tdoa_s),
                    json.dumps(detection.classifier_scores),
                    json.dumps(detection.feature_summary),
                    snippet_path,
                    snippet_expires_ns,
                ),
            )
            await db.commit()

    async def upsert_track(self, track: TrackState) -> None:
        db = self._require_db()
        async with self._lock:
            await db.execute(
                """
                INSERT INTO tracks (
                    id, first_seen_ns, last_seen_ns,
                    x, y, z, vx, vy, vz,
                    label, confidence, update_count, status, tqi
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    last_seen_ns=excluded.last_seen_ns,
                    x=excluded.x,
                    y=excluded.y,
                    z=excluded.z,
                    vx=excluded.vx,
                    vy=excluded.vy,
                    vz=excluded.vz,
                    label=excluded.label,
                    confidence=excluded.confidence,
                    update_count=excluded.update_count,
                    status=excluded.status,
                    tqi=excluded.tqi
                """,
                (
                    track.id,
                    track.first_seen_ns,
                    track.last_seen_ns,
                    track.position_m[0],
                    track.position_m[1],
                    track.position_m[2],
                    track.velocity_mps[0],
                    track.velocity_mps[1],
                    track.velocity_mps[2],
                    track.label,
                    track.confidence,
                    track.update_count,
                    track.status,
                    track.tqi,
                ),
            )
            await db.commit()

    async def list_nodes(self) -> list[dict]:
        db = self._require_db()
        async with self._lock:
            rows = await (await db.execute("SELECT * FROM nodes ORDER BY id ASC")).fetchall()

        result = []
        for row in rows:
            result.append(
                {
                    "id": row["id"],
                    "node_type": row["node_type"],
                    "position_m": [row["x"], row["y"], row["z"]],
                    "sensor_offsets_m": json.loads(row["sensor_offsets_json"]),
                    "capabilities": json.loads(row["capabilities_json"]),
                    "metadata": json.loads(row["metadata_json"]),
                    "last_seen_ns": row["last_seen_ns"],
                }
            )
        return result

    async def list_detections(self, limit: int = 100) -> list[dict]:
        db = self._require_db()
        async with self._lock:
            rows = await (
                await db.execute(
                    "SELECT * FROM detections ORDER BY timestamp_ns DESC LIMIT ?",
                    (limit,),
                )
            ).fetchall()

        result = []
        for row in rows:
            result.append(
                {
                    "id": row["id"],
                    "timestamp_ns": row["timestamp_ns"],
                    "position_m": [row["x"], row["y"], row["z"]],
                    "confidence": row["confidence"],
                    "gdop": row["gdop"],
                    "label": row["label"],
                    "label_confidence": row["label_confidence"],
                    "track_id": row["track_id"],
                    "reference_sensor": row["reference_sensor"],
                    "source_sensors": json.loads(row["source_sensors_json"]),
                    "tdoa_s": json.loads(row["tdoa_json"]),
                    "classifier_scores": json.loads(row["classifier_scores_json"]),
                    "feature_summary": json.loads(row["feature_summary_json"]),
                    "snippet_path": row["snippet_path"],
                }
            )
        return result

    async def list_tracks(self, limit: int = 200) -> list[dict]:
        db = self._require_db()
        async with self._lock:
            rows = await (
                await db.execute("SELECT * FROM tracks ORDER BY last_seen_ns DESC LIMIT ?", (limit,))
            ).fetchall()

        result = []
        for row in rows:
            result.append(
                {
                    "id": row["id"],
                    "first_seen_ns": row["first_seen_ns"],
                    "last_seen_ns": row["last_seen_ns"],
                    "position_m": [row["x"], row["y"], row["z"]],
                    "velocity_mps": [row["vx"], row["vy"], row["vz"]],
                    "label": row["label"],
                    "confidence": row["confidence"],
                    "update_count": row["update_count"],
                    "status": row["status"],
                    "tqi": row["tqi"],
                }
            )
        return result

    async def cleanup_expired_snippets(self, now_ns: int) -> int:
        db = self._require_db()
        async with self._lock:
            rows = await (
                await db.execute(
                    """
                    SELECT id, snippet_path
                    FROM detections
                    WHERE snippet_path IS NOT NULL
                    AND snippet_expires_ns IS NOT NULL
                    AND snippet_expires_ns <= ?
                    """,
                    (now_ns,),
                )
            ).fetchall()

            for row in rows:
                snippet_path = row["snippet_path"]
                if snippet_path:
                    path = Path(snippet_path)
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
                await db.execute(
                    "UPDATE detections SET snippet_path = NULL, snippet_expires_ns = NULL WHERE id = ?",
                    (row["id"],),
                )

            await db.commit()
        return len(rows)

    def _require_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Storage is not initialized")
        return self._db
