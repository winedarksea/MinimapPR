"""SQLite persistence layer."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

from minimappr.cleanup_policy import CleanupPolicy
from minimappr.models import DetectionEvent, GeoPoint, LabelId, NodeSpec, TrackState


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _json_loads(raw: str | None, fallback: Any) -> Any:
    if raw is None:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.strip().lower())
    cleaned = cleaned.strip("-")
    return cleaned or "unknown"


def _ingested_frame_key(
    *,
    node_id: str,
    frame_sequence: int | None,
    start_time_ns: int,
    source_type: str,
) -> str:
    sequence_token = str(frame_sequence) if frame_sequence is not None else "none"
    return f"{node_id}:{source_type}:{start_time_ns}:{sequence_token}"


class Storage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()
        self._write_owner: asyncio.Task[Any] | None = None
        self._batch_depth = 0

    async def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA foreign_keys=ON;")
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
                lat REAL,
                lon REAL,
                alt REAL,
                sensor_offsets_json TEXT NOT NULL,
                capabilities_json TEXT NOT NULL,
                mobility TEXT NOT NULL DEFAULT 'stationary',
                metadata_json TEXT NOT NULL,
                properties_json TEXT NOT NULL DEFAULT '{}',
                last_seen_ns INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                sensor_id TEXT NOT NULL,
                sensor_type TEXT NOT NULL,
                source_type TEXT NOT NULL,
                toa_ns INTEGER NOT NULL,
                tor_ns INTEGER NOT NULL,
                time_quality TEXT NOT NULL,
                sample_rate_hz INTEGER,
                channel_index INTEGER,
                frame_sequence INTEGER,
                retention_tier TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_ns INTEGER NOT NULL,
                FOREIGN KEY(node_id) REFERENCES nodes(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_observations_toa ON observations(toa_ns DESC);
            CREATE INDEX IF NOT EXISTS idx_observations_sensor ON observations(sensor_id, toa_ns DESC);
            CREATE INDEX IF NOT EXISTS idx_observations_event_id ON observations(event_id);
            CREATE INDEX IF NOT EXISTS idx_observations_node_id ON observations(node_id);

            CREATE TABLE IF NOT EXISTS ingested_frames (
                frame_key TEXT PRIMARY KEY,
                node_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                start_time_ns INTEGER NOT NULL,
                frame_sequence INTEGER,
                toa_ns INTEGER NOT NULL,
                tor_ns INTEGER NOT NULL,
                created_ns INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ingested_frames_node_start ON ingested_frames(node_id, start_time_ns DESC);
            CREATE INDEX IF NOT EXISTS idx_ingested_frames_node_seq ON ingested_frames(node_id, frame_sequence, start_time_ns DESC);

            CREATE TABLE IF NOT EXISTS labels (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                category TEXT NOT NULL,
                parent_label_id TEXT,
                source TEXT NOT NULL,
                external_taxonomy TEXT,
                external_label_id TEXT,
                created_ns INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_labels_category ON labels(category);

            CREATE TABLE IF NOT EXISTS detections (
                id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_node_id TEXT REFERENCES nodes(id) ON DELETE SET NULL,
                timestamp_ns INTEGER NOT NULL,
                toa_ns INTEGER NOT NULL,
                tor_ns INTEGER NOT NULL,
                time_quality TEXT NOT NULL,
                stale_ns INTEGER,
                report_window_start_ns INTEGER,
                report_window_end_ns INTEGER,
                reporting_modality TEXT NOT NULL DEFAULT 'localized',
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                lat REAL,
                lon REAL,
                alt REAL,
                covariance_json TEXT,
                confidence REAL NOT NULL,
                gdop REAL NOT NULL,
                label_id TEXT REFERENCES labels(id) ON DELETE SET NULL,
                label TEXT NOT NULL,
                label_category TEXT NOT NULL DEFAULT 'unknown',
                iff_category TEXT NOT NULL DEFAULT 'unknown',
                label_confidence REAL NOT NULL,
                spl_db REAL,
                track_id TEXT REFERENCES tracks(id) ON DELETE SET NULL,
                reference_sensor TEXT NOT NULL,
                source_sensors_json TEXT NOT NULL,
                source_observation_ids_json TEXT NOT NULL DEFAULT '[]',
                zone_ids_json TEXT NOT NULL DEFAULT '[]',
                tdoa_json TEXT NOT NULL,
                classifier_scores_json TEXT NOT NULL,
                feature_summary_json TEXT NOT NULL,
                retention_tier TEXT NOT NULL DEFAULT 'short',
                snippet_path TEXT,
                snippet_expires_ns INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_detections_ts ON detections(timestamp_ns DESC);
            CREATE INDEX IF NOT EXISTS idx_detections_track ON detections(track_id, timestamp_ns DESC);
            CREATE INDEX IF NOT EXISTS idx_detections_label ON detections(label_category, timestamp_ns DESC);
            CREATE INDEX IF NOT EXISTS idx_detections_reporting_window
                ON detections(source_node_id, label, report_window_start_ns DESC);

            CREATE TABLE IF NOT EXISTS tracks (
                id TEXT PRIMARY KEY,
                first_seen_ns INTEGER NOT NULL,
                last_seen_ns INTEGER NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                lat REAL,
                lon REAL,
                alt REAL,
                covariance_json TEXT,
                vx REAL NOT NULL,
                vy REAL NOT NULL,
                vz REAL NOT NULL,
                label_id TEXT,
                label TEXT NOT NULL,
                label_category TEXT NOT NULL DEFAULT 'unknown',
                iff_category TEXT NOT NULL DEFAULT 'unknown',
                confidence REAL NOT NULL,
                update_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                capability_tier TEXT NOT NULL DEFAULT 'full_3d',
                tqi REAL NOT NULL DEFAULT 0.0
            );
            CREATE INDEX IF NOT EXISTS idx_tracks_last_seen ON tracks(last_seen_ns DESC);

            CREATE TABLE IF NOT EXISTS track_updates (
                id TEXT PRIMARY KEY,
                track_id TEXT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
                timestamp_ns INTEGER NOT NULL,
                event_id TEXT,
                update_type TEXT NOT NULL,
                status TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                lat REAL,
                lon REAL,
                alt REAL,
                vx REAL NOT NULL,
                vy REAL NOT NULL,
                vz REAL NOT NULL,
                covariance_json TEXT,
                tqi REAL NOT NULL,
                confidence REAL NOT NULL,
                label TEXT NOT NULL,
                label_category TEXT NOT NULL,
                detection_id TEXT REFERENCES detections(id) ON DELETE SET NULL,
                observation_ids_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_track_updates_track_ts ON track_updates(track_id, timestamp_ns DESC);

            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                timestamp_ns INTEGER NOT NULL,
                rule_id TEXT,
                detection_id TEXT REFERENCES detections(id) ON DELETE SET NULL,
                track_id TEXT REFERENCES tracks(id) ON DELETE SET NULL,
                destination TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_ns INTEGER NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp_ns DESC);

            CREATE TABLE IF NOT EXISTS pings (
                id TEXT PRIMARY KEY,
                timestamp_ns INTEGER NOT NULL,
                lat REAL,
                lon REAL,
                alt REAL,
                x REAL,
                y REAL,
                z REAL,
                ping_type TEXT NOT NULL,
                label_id TEXT REFERENCES labels(id) ON DELETE SET NULL,
                label TEXT,
                spl_db REAL,
                source_detection_id TEXT REFERENCES detections(id) ON DELETE SET NULL,
                source_observation_id TEXT REFERENCES observations(id) ON DELETE SET NULL,
                source_track_id TEXT REFERENCES tracks(id) ON DELETE SET NULL,
                retention_tier TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_pings_ts ON pings(timestamp_ns DESC);
            CREATE INDEX IF NOT EXISTS idx_pings_type ON pings(ping_type, timestamp_ns DESC);

            CREATE TABLE IF NOT EXISTS large_artifacts (
                id TEXT PRIMARY KEY,
                artifact_type TEXT NOT NULL,
                path TEXT NOT NULL,
                retention_tier TEXT NOT NULL,
                source_detection_id TEXT REFERENCES detections(id) ON DELETE SET NULL,
                source_track_id TEXT REFERENCES tracks(id) ON DELETE SET NULL,
                metadata_json TEXT NOT NULL,
                created_ns INTEGER NOT NULL,
                expires_ns INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_large_artifacts_expires ON large_artifacts(expires_ns);

            CREATE TABLE IF NOT EXISTS zones (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                zone_type TEXT NOT NULL,
                polygon_geo_json TEXT NOT NULL,
                properties_json TEXT NOT NULL,
                created_ns INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS environment (
                id TEXT PRIMARY KEY,
                node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                timestamp_ns INTEGER NOT NULL,
                temperature_k REAL,
                pressure_pa REAL,
                humidity REAL,
                wind_speed_mps REAL,
                wind_dir_deg REAL,
                solar_lux REAL,
                metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_environment_ts ON environment(timestamp_ns DESC);
            CREATE INDEX IF NOT EXISTS idx_environment_node_ts ON environment(node_id, timestamp_ns DESC);

            CREATE TABLE IF NOT EXISTS bit_reports (
                id TEXT PRIMARY KEY,
                node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                report_type TEXT NOT NULL,
                overall_status TEXT NOT NULL,
                timestamp_ns INTEGER NOT NULL,
                received_ns INTEGER NOT NULL,
                results_json TEXT NOT NULL,
                failure_codes_json TEXT NOT NULL DEFAULT '[]',
                firmware_version TEXT,
                uptime_seconds REAL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_bit_reports_node_ts ON bit_reports(node_id, timestamp_ns DESC);
            CREATE INDEX IF NOT EXISTS idx_bit_reports_type ON bit_reports(report_type, timestamp_ns DESC);
            """
        )

        # Migration-safe additions for older DBs.
        await self._ensure_columns(
            "nodes",
            {
                "lat": "REAL",
                "lon": "REAL",
                "alt": "REAL",
                "mobility": "TEXT NOT NULL DEFAULT 'stationary'",
                "properties_json": "TEXT NOT NULL DEFAULT '{}'",
            },
        )
        await self._ensure_columns(
            "detections",
            {
                "event_id": "TEXT",
                "source_type": "TEXT NOT NULL DEFAULT 'raw_sensor'",
                "source_node_id": "TEXT",
                "toa_ns": "INTEGER",
                "tor_ns": "INTEGER",
                "time_quality": "TEXT NOT NULL DEFAULT 'freerunning'",
                "stale_ns": "INTEGER",
                "report_window_start_ns": "INTEGER",
                "report_window_end_ns": "INTEGER",
                "reporting_modality": "TEXT NOT NULL DEFAULT 'localized'",
                "lat": "REAL",
                "lon": "REAL",
                "alt": "REAL",
                "covariance_json": "TEXT",
                "label_id": "TEXT",
                "label_category": "TEXT NOT NULL DEFAULT 'unknown'",
                "iff_category": "TEXT NOT NULL DEFAULT 'unknown'",
                "spl_db": "REAL",
                "source_observation_ids_json": "TEXT NOT NULL DEFAULT '[]'",
                "zone_ids_json": "TEXT NOT NULL DEFAULT '[]'",
                "retention_tier": "TEXT NOT NULL DEFAULT 'short'",
            },
        )
        await self._ensure_columns(
            "alerts",
            {
                "updated_ns": "INTEGER NOT NULL DEFAULT 0",
            },
        )
        await self._ensure_columns(
            "tracks",
            {
                "lat": "REAL",
                "lon": "REAL",
                "alt": "REAL",
                "covariance_json": "TEXT",
                "label_id": "TEXT",
                "label_category": "TEXT NOT NULL DEFAULT 'unknown'",
                "iff_category": "TEXT NOT NULL DEFAULT 'unknown'",
                "capability_tier": "TEXT NOT NULL DEFAULT 'full_3d'",
            },
        )
        await self._ensure_columns(
            "environment",
            {
                "temperature_k": "REAL",
                "pressure_pa": "REAL",
                "humidity": "REAL",
                "wind_speed_mps": "REAL",
                "wind_dir_deg": "REAL",
                "solar_lux": "REAL",
                "metadata_json": "TEXT NOT NULL DEFAULT '{}'",
            },
        )
        await self._deduplicate_reporting_window_canonicals()
        await self._ensure_reporting_window_uniqueness_index()
        await self._db.commit()

    async def _deduplicate_reporting_window_canonicals(self) -> None:
        """Keep one canonical row per reporting key before enforcing uniqueness."""
        db = self._require_db()
        await db.execute(
            """
            WITH ranked AS (
                SELECT
                    rowid,
                    ROW_NUMBER() OVER (
                        PARTITION BY
                            COALESCE(source_node_id, '__none__'),
                            lower(label),
                            report_window_start_ns,
                            report_window_end_ns
                        ORDER BY
                            CASE WHEN reporting_modality = 'localized' THEN 1 ELSE 0 END DESC,
                            timestamp_ns DESC,
                            rowid DESC
                    ) AS duplicate_rank
                FROM detections
                WHERE report_window_start_ns IS NOT NULL
                  AND report_window_end_ns IS NOT NULL
            )
            DELETE FROM detections
            WHERE rowid IN (
                SELECT rowid FROM ranked WHERE duplicate_rank > 1
            )
            """
        )

    async def _ensure_reporting_window_uniqueness_index(self) -> None:
        """Enforce one canonical detection per node/label/reporting window."""
        db = self._require_db()
        await db.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_detections_reporting_window_canonical
                ON detections(
                    COALESCE(source_node_id, '__none__'),
                    lower(label),
                    report_window_start_ns,
                    report_window_end_ns
                )
            WHERE report_window_start_ns IS NOT NULL
              AND report_window_end_ns IS NOT NULL
            """
        )

    async def _table_columns(self, table: str) -> set[str]:
        db = self._require_db()
        rows = await (await db.execute(f"PRAGMA table_info({table})")).fetchall()
        return {row["name"] for row in rows}

    async def _ensure_columns(self, table: str, expected: dict[str, str]) -> None:
        db = self._require_db()
        current = await self._table_columns(table)
        for name, definition in expected.items():
            if name in current:
                continue
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    @asynccontextmanager
    async def _write_guard(self):
        current = asyncio.current_task()
        if current is not None and self._write_owner is current:
            yield
            return
        async with self._write_lock:
            self._write_owner = asyncio.current_task()
            try:
                yield
            finally:
                self._write_owner = None

    async def _commit_if_needed(self, db: aiosqlite.Connection) -> None:
        if self._batch_depth == 0:
            await db.commit()

    @asynccontextmanager
    async def begin_batch(self):
        db = self._require_db()
        async with self._write_guard():
            if self._batch_depth == 0:
                await db.execute("BEGIN")
            self._batch_depth += 1
            try:
                yield
            except Exception:
                self._batch_depth = max(0, self._batch_depth - 1)
                if self._batch_depth == 0:
                    await db.rollback()
                raise
            else:
                self._batch_depth = max(0, self._batch_depth - 1)
                if self._batch_depth == 0:
                    await db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def upsert_node(
        self,
        spec: NodeSpec,
        last_seen_ns: int,
        position_geo: GeoPoint | None = None,
    ) -> None:
        db = self._require_db()
        if spec.position_m is None:
            raise ValueError("NodeSpec.position_m is required for persistence")
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO nodes (
                    id, node_type, x, y, z, lat, lon, alt,
                    sensor_offsets_json, capabilities_json, mobility,
                    metadata_json, properties_json, last_seen_ns
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    node_type=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.node_type
                        ELSE nodes.node_type
                    END,
                    x=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.x
                        ELSE nodes.x
                    END,
                    y=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.y
                        ELSE nodes.y
                    END,
                    z=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.z
                        ELSE nodes.z
                    END,
                    lat=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.lat
                        ELSE nodes.lat
                    END,
                    lon=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.lon
                        ELSE nodes.lon
                    END,
                    alt=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.alt
                        ELSE nodes.alt
                    END,
                    sensor_offsets_json=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.sensor_offsets_json
                        ELSE nodes.sensor_offsets_json
                    END,
                    capabilities_json=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.capabilities_json
                        ELSE nodes.capabilities_json
                    END,
                    mobility=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.mobility
                        ELSE nodes.mobility
                    END,
                    metadata_json=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.metadata_json
                        ELSE nodes.metadata_json
                    END,
                    properties_json=CASE
                        WHEN excluded.last_seen_ns >= nodes.last_seen_ns THEN excluded.properties_json
                        ELSE nodes.properties_json
                    END,
                    last_seen_ns=CASE
                        WHEN excluded.last_seen_ns > nodes.last_seen_ns THEN excluded.last_seen_ns
                        ELSE nodes.last_seen_ns
                    END
                """,
                (
                    spec.id,
                    spec.node_type.value,
                    spec.position_m[0],
                    spec.position_m[1],
                    spec.position_m[2],
                    position_geo.lat if position_geo is not None else None,
                    position_geo.lon if position_geo is not None else None,
                    position_geo.alt_m if position_geo is not None else None,
                    _json_dumps(spec.sensor_offsets_m),
                    _json_dumps(spec.capabilities),
                    spec.mobility,
                    _json_dumps(spec.metadata),
                    _json_dumps(spec.properties),
                    last_seen_ns,
                ),
            )
            await self._commit_if_needed(db)

    # ------------------------------------------------------------------
    # BIT Reports
    # ------------------------------------------------------------------

    async def insert_bit_report(
        self,
        *,
        report_id: str,
        node_id: str,
        report_type: str,
        overall_status: str,
        timestamp_ns: int,
        received_ns: int,
        results_json: str,
        failure_codes_json: str,
        firmware_version: str | None,
        uptime_seconds: float | None,
        metadata_json: str,
    ) -> None:
        db = self._require_db()
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO bit_reports (
                    id, node_id, report_type, overall_status,
                    timestamp_ns, received_ns, results_json,
                    failure_codes_json, firmware_version,
                    uptime_seconds, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report_id,
                    node_id,
                    report_type,
                    overall_status,
                    timestamp_ns,
                    received_ns,
                    results_json,
                    failure_codes_json,
                    firmware_version,
                    uptime_seconds,
                    metadata_json,
                ),
            )
            await self._commit_if_needed(db)

    async def list_bit_reports(
        self,
        node_id: str | None = None,
        report_type: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        db = self._require_db()
        clauses: list[str] = []
        params: list[object] = []
        if node_id is not None:
            clauses.append("node_id = ?")
            params.append(node_id)
        if report_type is not None:
            clauses.append("report_type = ?")
            params.append(report_type)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = await (
            await db.execute(
                f"SELECT * FROM bit_reports{where} ORDER BY timestamp_ns DESC LIMIT ?",
                (*params, limit),
            )
        ).fetchall()
        result: list[dict] = []
        for row in rows:
            result.append(
                {
                    "id": row["id"],
                    "node_id": row["node_id"],
                    "report_type": row["report_type"],
                    "overall_status": row["overall_status"],
                    "timestamp_ns": row["timestamp_ns"],
                    "received_ns": row["received_ns"],
                    "results": _json_loads(row["results_json"], []),
                    "failure_codes": _json_loads(row["failure_codes_json"], []),
                    "firmware_version": row["firmware_version"],
                    "uptime_seconds": row["uptime_seconds"],
                    "metadata": _json_loads(row["metadata_json"], {}),
                }
            )
        return result

    async def latest_bit_report_per_type(self, node_id: str) -> list[dict]:
        """Return the single newest BIT report per report_type for a node."""
        db = self._require_db()
        rows = await (
            await db.execute(
                """
                SELECT b.* FROM bit_reports b
                INNER JOIN (
                    SELECT report_type, MAX(timestamp_ns) AS max_ts
                    FROM bit_reports
                    WHERE node_id = ?
                    GROUP BY report_type
                ) latest ON b.report_type = latest.report_type
                    AND b.timestamp_ns = latest.max_ts
                    AND b.node_id = ?
                ORDER BY b.report_type
                """,
                (node_id, node_id),
            )
        ).fetchall()
        return [
            {
                "id": row["id"],
                "node_id": row["node_id"],
                "report_type": row["report_type"],
                "overall_status": row["overall_status"],
                "timestamp_ns": row["timestamp_ns"],
                "received_ns": row["received_ns"],
                "results": _json_loads(row["results_json"], []),
                "failure_codes": _json_loads(row["failure_codes_json"], []),
                "firmware_version": row["firmware_version"],
                "uptime_seconds": row["uptime_seconds"],
                "metadata": _json_loads(row["metadata_json"], {}),
            }
            for row in rows
        ]

    async def insert_observation(
        self,
        *,
        node_id: str,
        sensor_id: str,
        sensor_type: str,
        source_type: str,
        toa_ns: int,
        tor_ns: int,
        time_quality: str,
        sample_rate_hz: int | None,
        channel_index: int | None,
        frame_sequence: int | None,
        retention_tier: str = "short",
        metadata: dict[str, Any] | None = None,
        event_id: str | None = None,
    ) -> str:
        db = self._require_db()
        observation_id = event_id or f"obs-{uuid.uuid4().hex[:16]}"
        payload = metadata or {}
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO observations (
                    id, event_id, node_id, sensor_id, sensor_type, source_type,
                    toa_ns, tor_ns, time_quality, sample_rate_hz, channel_index,
                    frame_sequence, retention_tier, metadata_json, created_ns
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation_id,
                    observation_id,
                    node_id,
                    sensor_id,
                    sensor_type,
                    source_type,
                    toa_ns,
                    tor_ns,
                    time_quality,
                    sample_rate_hz,
                    channel_index,
                    frame_sequence,
                    retention_tier,
                    _json_dumps(payload),
                    tor_ns,
                ),
            )
            await self._commit_if_needed(db)
        return observation_id

    async def has_ingested_frame(
        self,
        *,
        node_id: str,
        frame_sequence: int | None,
        start_time_ns: int,
        source_type: str,
    ) -> bool:
        db = self._require_db()
        frame_key = _ingested_frame_key(
            node_id=node_id,
            frame_sequence=frame_sequence,
            start_time_ns=start_time_ns,
            source_type=source_type,
        )
        row = await (
            await db.execute(
                """
                SELECT 1
                FROM ingested_frames
                WHERE frame_key = ?
                LIMIT 1
                """,
                (frame_key,),
            )
        ).fetchone()
        return row is not None

    async def register_ingested_frame(
        self,
        *,
        node_id: str,
        frame_sequence: int | None,
        start_time_ns: int,
        toa_ns: int,
        tor_ns: int,
        source_type: str,
    ) -> bool:
        db = self._require_db()
        frame_key = _ingested_frame_key(
            node_id=node_id,
            frame_sequence=frame_sequence,
            start_time_ns=start_time_ns,
            source_type=source_type,
        )
        async with self._write_guard():
            cursor = await db.execute(
                """
                INSERT INTO ingested_frames (
                    frame_key, node_id, source_type, start_time_ns,
                    frame_sequence, toa_ns, tor_ns, created_ns
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(frame_key) DO NOTHING
                """,
                (
                    frame_key,
                    node_id,
                    source_type,
                    start_time_ns,
                    frame_sequence,
                    toa_ns,
                    tor_ns,
                    tor_ns,
                ),
            )
            await self._commit_if_needed(db)
            return cursor.rowcount > 0

    async def upsert_label(
        self,
        *,
        name: str,
        category: str,
        source: str = "runtime",
        parent_label_id: LabelId | None = None,
        external_taxonomy: str | None = None,
        external_label_id: str | None = None,
        created_ns: int,
    ) -> LabelId:
        db = self._require_db()
        label_id = f"lbl-{_slugify(name)[:48]}"
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO labels (
                    id, name, category, parent_label_id, source,
                    external_taxonomy, external_label_id, created_ns
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    category=excluded.category,
                    parent_label_id=COALESCE(excluded.parent_label_id, labels.parent_label_id),
                    source=excluded.source,
                    external_taxonomy=COALESCE(excluded.external_taxonomy, labels.external_taxonomy),
                    external_label_id=COALESCE(excluded.external_label_id, labels.external_label_id)
                """,
                (
                    label_id,
                    name,
                    category,
                    parent_label_id,
                    source,
                    external_taxonomy,
                    external_label_id,
                    created_ns,
                ),
            )
            await self._commit_if_needed(db)
        return label_id

    async def insert_detection(
        self,
        detection: DetectionEvent,
        snippet_path: str | None,
        snippet_expires_ns: int | None,
        retention_tier: str = "short",
    ) -> None:
        db = self._require_db()
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO detections (
                    id, event_id, source_type, source_node_id,
                    timestamp_ns, toa_ns, tor_ns, time_quality, stale_ns,
                    report_window_start_ns, report_window_end_ns, reporting_modality,
                    x, y, z, lat, lon, alt, covariance_json,
                    confidence, gdop, label_id, label, label_category, iff_category,
                    label_confidence, spl_db, track_id,
                    reference_sensor, source_sensors_json, source_observation_ids_json,
                    zone_ids_json, tdoa_json, classifier_scores_json, feature_summary_json,
                    retention_tier, snippet_path, snippet_expires_ns
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    detection.id,
                    detection.event_id or detection.id,
                    detection.source_type,
                    detection.source_node_id,
                    detection.timestamp_ns,
                    detection.toa_ns,
                    detection.tor_ns,
                    detection.time_quality.value,
                    detection.stale_ns,
                    detection.report_window_start_ns,
                    detection.report_window_end_ns,
                    detection.reporting_modality,
                    detection.position_m[0],
                    detection.position_m[1],
                    detection.position_m[2],
                    detection.position_geo.lat if detection.position_geo else None,
                    detection.position_geo.lon if detection.position_geo else None,
                    detection.position_geo.alt_m if detection.position_geo else None,
                    _json_dumps(detection.position_covariance_m2)
                    if detection.position_covariance_m2 is not None
                    else None,
                    detection.confidence,
                    detection.gdop,
                    detection.label_id,
                    detection.label,
                    detection.label_category,
                    detection.iff_category,
                    detection.label_confidence,
                    detection.spl_db,
                    detection.track_id,
                    detection.reference_sensor,
                    _json_dumps(detection.source_sensors),
                    _json_dumps(detection.source_observation_ids),
                    _json_dumps(detection.zone_ids),
                    _json_dumps(detection.tdoa_s),
                    _json_dumps(detection.classifier_scores),
                    _json_dumps(detection.feature_summary),
                    retention_tier,
                    snippet_path,
                    snippet_expires_ns,
                ),
            )
            await self._commit_if_needed(db)

    async def update_detection(
        self,
        detection: DetectionEvent,
        snippet_path: str | None,
        snippet_expires_ns: int | None,
        retention_tier: str = "short",
    ) -> bool:
        db = self._require_db()
        async with self._write_guard():
            cursor = await db.execute(
                """
                UPDATE detections
                SET event_id = ?, source_type = ?, source_node_id = ?,
                    timestamp_ns = ?, toa_ns = ?, tor_ns = ?, time_quality = ?, stale_ns = ?,
                    report_window_start_ns = ?, report_window_end_ns = ?, reporting_modality = ?,
                    x = ?, y = ?, z = ?, lat = ?, lon = ?, alt = ?, covariance_json = ?,
                    confidence = ?, gdop = ?, label_id = ?, label = ?, label_category = ?, iff_category = ?,
                    label_confidence = ?, spl_db = ?, track_id = ?,
                    reference_sensor = ?, source_sensors_json = ?, source_observation_ids_json = ?,
                    zone_ids_json = ?, tdoa_json = ?, classifier_scores_json = ?, feature_summary_json = ?,
                    retention_tier = ?, snippet_path = COALESCE(?, snippet_path),
                    snippet_expires_ns = COALESCE(?, snippet_expires_ns)
                WHERE id = ?
                """,
                (
                    detection.event_id or detection.id,
                    detection.source_type,
                    detection.source_node_id,
                    detection.timestamp_ns,
                    detection.toa_ns,
                    detection.tor_ns,
                    detection.time_quality.value,
                    detection.stale_ns,
                    detection.report_window_start_ns,
                    detection.report_window_end_ns,
                    detection.reporting_modality,
                    detection.position_m[0],
                    detection.position_m[1],
                    detection.position_m[2],
                    detection.position_geo.lat if detection.position_geo else None,
                    detection.position_geo.lon if detection.position_geo else None,
                    detection.position_geo.alt_m if detection.position_geo else None,
                    _json_dumps(detection.position_covariance_m2)
                    if detection.position_covariance_m2 is not None
                    else None,
                    detection.confidence,
                    detection.gdop,
                    detection.label_id,
                    detection.label,
                    detection.label_category,
                    detection.iff_category,
                    detection.label_confidence,
                    detection.spl_db,
                    detection.track_id,
                    detection.reference_sensor,
                    _json_dumps(detection.source_sensors),
                    _json_dumps(detection.source_observation_ids),
                    _json_dumps(detection.zone_ids),
                    _json_dumps(detection.tdoa_s),
                    _json_dumps(detection.classifier_scores),
                    _json_dumps(detection.feature_summary),
                    retention_tier,
                    snippet_path,
                    snippet_expires_ns,
                    detection.id,
                ),
            )
            await self._commit_if_needed(db)
        return bool(cursor.rowcount)

    async def find_detection_for_reporting_window(
        self,
        *,
        source_node_id: str | None,
        label: str,
        report_window_start_ns: int,
        report_window_end_ns: int,
    ) -> dict[str, Any] | None:
        db = self._require_db()
        if source_node_id is None:
            row = await (
                await db.execute(
                    """
                    SELECT * FROM detections
                    WHERE source_node_id IS NULL
                      AND label = ?
                      AND report_window_start_ns = ?
                      AND report_window_end_ns = ?
                    ORDER BY timestamp_ns DESC
                    LIMIT 1
                    """,
                    (label, report_window_start_ns, report_window_end_ns),
                )
            ).fetchone()
        else:
            row = await (
                await db.execute(
                    """
                    SELECT * FROM detections
                    WHERE source_node_id = ?
                      AND label = ?
                      AND report_window_start_ns = ?
                      AND report_window_end_ns = ?
                    ORDER BY timestamp_ns DESC
                    LIMIT 1
                    """,
                    (source_node_id, label, report_window_start_ns, report_window_end_ns),
                )
            ).fetchone()
        if row is None:
            return None
        return self._row_to_detection(row).model_dump(mode="json")

    async def insert_track_update(
        self,
        *,
        track: TrackState,
        timestamp_ns: int,
        event_id: str | None,
        update_type: str,
        detection_id: str | None,
        observation_ids: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        db = self._require_db()
        update_id = f"tup-{uuid.uuid4().hex[:16]}"
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO track_updates (
                    id, track_id, timestamp_ns, event_id, update_type, status,
                    x, y, z, lat, lon, alt, vx, vy, vz, covariance_json,
                    tqi, confidence, label, label_category,
                    detection_id, observation_ids_json, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    update_id,
                    track.id,
                    timestamp_ns,
                    event_id,
                    update_type,
                    track.status,
                    track.position_m[0],
                    track.position_m[1],
                    track.position_m[2],
                    track.position_geo.lat if track.position_geo else None,
                    track.position_geo.lon if track.position_geo else None,
                    track.position_geo.alt_m if track.position_geo else None,
                    track.velocity_mps[0],
                    track.velocity_mps[1],
                    track.velocity_mps[2],
                    _json_dumps(track.position_covariance_m2)
                    if track.position_covariance_m2 is not None
                    else None,
                    track.tqi,
                    track.confidence,
                    track.label,
                    track.label_category,
                    detection_id,
                    _json_dumps(observation_ids),
                    _json_dumps(metadata or {}),
                ),
            )
            await self._commit_if_needed(db)
        return update_id

    async def upsert_track(self, track: TrackState) -> None:
        db = self._require_db()
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO tracks (
                    id, first_seen_ns, last_seen_ns,
                    x, y, z, lat, lon, alt, covariance_json,
                    vx, vy, vz,
                    label_id, label, label_category, iff_category,
                    confidence, update_count, status, capability_tier, tqi
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    last_seen_ns=excluded.last_seen_ns,
                    x=excluded.x,
                    y=excluded.y,
                    z=excluded.z,
                    lat=excluded.lat,
                    lon=excluded.lon,
                    alt=excluded.alt,
                    covariance_json=excluded.covariance_json,
                    vx=excluded.vx,
                    vy=excluded.vy,
                    vz=excluded.vz,
                    label_id=excluded.label_id,
                    label=excluded.label,
                    label_category=excluded.label_category,
                    iff_category=excluded.iff_category,
                    confidence=excluded.confidence,
                    update_count=excluded.update_count,
                    status=excluded.status,
                    capability_tier=excluded.capability_tier,
                    tqi=excluded.tqi
                """,
                (
                    track.id,
                    track.first_seen_ns,
                    track.last_seen_ns,
                    track.position_m[0],
                    track.position_m[1],
                    track.position_m[2],
                    track.position_geo.lat if track.position_geo else None,
                    track.position_geo.lon if track.position_geo else None,
                    track.position_geo.alt_m if track.position_geo else None,
                    _json_dumps(track.position_covariance_m2)
                    if track.position_covariance_m2 is not None
                    else None,
                    track.velocity_mps[0],
                    track.velocity_mps[1],
                    track.velocity_mps[2],
                    track.label_id,
                    track.label,
                    track.label_category,
                    track.iff_category,
                    track.confidence,
                    track.update_count,
                    track.status,
                    track.capability_tier,
                    track.tqi,
                ),
            )
            await self._commit_if_needed(db)

    async def list_nodes(self, limit: int | None = None) -> list[dict]:
        db = self._require_db()
        if limit is not None and limit > 0:
            rows = await (await db.execute("SELECT * FROM nodes ORDER BY id ASC LIMIT ?", (limit,))).fetchall()
        else:
            rows = await (await db.execute("SELECT * FROM nodes ORDER BY id ASC")).fetchall()

        result = []
        for row in rows:
            position_geo = None
            if row["lat"] is not None and row["lon"] is not None:
                position_geo = {"lat": row["lat"], "lon": row["lon"], "alt_m": row["alt"] or 0.0}
            result.append(
                {
                    "id": row["id"],
                    "node_type": row["node_type"],
                    "position_m": [row["x"], row["y"], row["z"]],
                    "position_geo": position_geo,
                    "sensor_offsets_m": _json_loads(row["sensor_offsets_json"], []),
                    "capabilities": _json_loads(row["capabilities_json"], []),
                    "mobility": row["mobility"] or "stationary",
                    "metadata": _json_loads(row["metadata_json"], {}),
                    "properties": _json_loads(row["properties_json"], {}),
                    "last_seen_ns": row["last_seen_ns"],
                }
            )
        return result

    async def get_node_by_id(self, node_id: str) -> dict | None:
        db = self._require_db()
        row = await (await db.execute("SELECT * FROM nodes WHERE id = ? LIMIT 1", (node_id,))).fetchone()
        if row is None:
            return None
        position_geo = None
        if row["lat"] is not None and row["lon"] is not None:
            position_geo = {"lat": row["lat"], "lon": row["lon"], "alt_m": row["alt"] or 0.0}
        return {
            "id": row["id"],
            "node_type": row["node_type"],
            "position_m": [row["x"], row["y"], row["z"]],
            "position_geo": position_geo,
            "sensor_offsets_m": _json_loads(row["sensor_offsets_json"], []),
            "capabilities": _json_loads(row["capabilities_json"], []),
            "mobility": row["mobility"] or "stationary",
            "metadata": _json_loads(row["metadata_json"], {}),
            "properties": _json_loads(row["properties_json"], {}),
            "last_seen_ns": row["last_seen_ns"],
        }

    async def get_detection_by_event_id(self, event_id: str) -> dict | None:
        db = self._require_db()
        row = await (
            await db.execute(
                """
                SELECT * FROM detections
                WHERE event_id = ? OR id = ?
                ORDER BY timestamp_ns DESC
                LIMIT 1
                """,
                (event_id, event_id),
            )
        ).fetchone()
        if row is None:
            return None
        return self._row_to_detection(row).model_dump(mode="json")

    async def list_detections(
        self,
        limit: int = 100,
        *,
        min_label_confidence: float | None = None,
    ) -> list[dict]:
        db = self._require_db()
        if min_label_confidence is None:
            rows = await (
                await db.execute(
                    "SELECT * FROM detections ORDER BY timestamp_ns DESC LIMIT ?",
                    (limit,),
                )
            ).fetchall()
        else:
            rows = await (
                await db.execute(
                    "SELECT * FROM detections WHERE label_confidence >= ? ORDER BY timestamp_ns DESC LIMIT ?",
                    (float(min_label_confidence), limit),
                )
            ).fetchall()
        return [self._row_to_detection(row).model_dump(mode="json") for row in rows]

    async def list_tracks(self, limit: int = 200) -> list[dict]:
        db = self._require_db()
        rows = await (
            await db.execute("SELECT * FROM tracks ORDER BY last_seen_ns DESC LIMIT ?", (limit,))
        ).fetchall()
        return [self._row_to_track(row).model_dump(mode="json") for row in rows]

    async def list_labels(self) -> list[dict]:
        db = self._require_db()
        rows = await (await db.execute("SELECT * FROM labels ORDER BY name ASC")).fetchall()
        return [dict(row) for row in rows]

    async def list_alerts(
        self,
        limit: int = 100,
        *,
        detection_id: str | None = None,
        track_id: str | None = None,
    ) -> list[dict]:
        db = self._require_db()
        clauses: list[str] = []
        params: list[object] = []
        if detection_id is not None:
            clauses.append("detection_id = ?")
            params.append(detection_id)
        if track_id is not None:
            clauses.append("track_id = ?")
            params.append(track_id)
        where = f" WHERE {' OR '.join(clauses)}" if clauses else ""
        rows = await (
            await db.execute(
                f"SELECT * FROM alerts{where} ORDER BY timestamp_ns DESC LIMIT ?",
                (*params, limit),
            )
        ).fetchall()
        result: list[dict] = []
        for row in rows:
            item = dict(row)
            item["payload"] = _json_loads(item.pop("payload_json"), {})
            if "updated_ns" not in item or item["updated_ns"] is None:
                item["updated_ns"] = item["timestamp_ns"]
            result.append(item)
        return result

    async def list_observations_by_ids(self, observation_ids: list[str]) -> list[dict]:
        db = self._require_db()
        if not observation_ids:
            return []
        placeholders = ",".join("?" for _ in observation_ids)
        rows = await (
            await db.execute(
                f"SELECT * FROM observations WHERE id IN ({placeholders})",
                tuple(observation_ids),
            )
        ).fetchall()
        by_id = {
            row["id"]: {
                "id": row["id"],
                "event_id": row["event_id"],
                "node_id": row["node_id"],
                "sensor_id": row["sensor_id"],
                "sensor_type": row["sensor_type"],
                "source_type": row["source_type"],
                "toa_ns": row["toa_ns"],
                "tor_ns": row["tor_ns"],
                "time_quality": row["time_quality"],
                "sample_rate_hz": row["sample_rate_hz"],
                "channel_index": row["channel_index"],
                "frame_sequence": row["frame_sequence"],
                "retention_tier": row["retention_tier"],
                "metadata": _json_loads(row["metadata_json"], {}),
                "created_ns": row["created_ns"],
            }
            for row in rows
        }
        return [by_id[observation_id] for observation_id in observation_ids if observation_id in by_id]

    async def list_track_updates(
        self,
        *,
        track_id: str | None = None,
        detection_id: str | None = None,
        event_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        db = self._require_db()
        clauses: list[str] = []
        params: list[object] = []
        if track_id is not None:
            clauses.append("track_id = ?")
            params.append(track_id)
        if detection_id is not None:
            clauses.append("detection_id = ?")
            params.append(detection_id)
        if event_id is not None:
            clauses.append("event_id = ?")
            params.append(event_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = await (
            await db.execute(
                f"SELECT * FROM track_updates{where} ORDER BY timestamp_ns DESC LIMIT ?",
                (*params, limit),
            )
        ).fetchall()
        return [
            {
                "id": row["id"],
                "track_id": row["track_id"],
                "timestamp_ns": row["timestamp_ns"],
                "event_id": row["event_id"],
                "update_type": row["update_type"],
                "status": row["status"],
                "position_m": [row["x"], row["y"], row["z"]],
                "position_geo": self._row_to_geo(row).model_dump(mode="json") if self._row_to_geo(row) else None,
                "velocity_mps": [row["vx"], row["vy"], row["vz"]],
                "covariance": _json_loads(row["covariance_json"], None),
                "tqi": row["tqi"],
                "confidence": row["confidence"],
                "label": row["label"],
                "label_category": row["label_category"],
                "detection_id": row["detection_id"],
                "observation_ids": _json_loads(row["observation_ids_json"], []),
                "metadata": _json_loads(row["metadata_json"], {}),
            }
            for row in rows
        ]

    async def latest_detection_audio_for_track(self, track_id: str) -> tuple[str, str] | None:
        db = self._require_db()
        row = await (
            await db.execute(
                """
                SELECT id, snippet_path
                FROM detections
                WHERE track_id = ? AND snippet_path IS NOT NULL
                ORDER BY timestamp_ns DESC
                LIMIT 1
                """,
                (track_id,),
            )
        ).fetchone()
        if row is None:
            return None
        detection_id = str(row["id"])
        snippet_path = row["snippet_path"]
        if not snippet_path:
            return None
        return detection_id, str(snippet_path)

    async def insert_alert(
        self,
        *,
        timestamp_ns: int,
        rule_id: str | None,
        detection_id: str | None,
        track_id: str | None,
        destination: str,
        priority: str,
        status: str,
        payload: dict[str, Any] | None = None,
        alert_id: str | None = None,
    ) -> str:
        db = self._require_db()
        final_id = alert_id or f"alr-{uuid.uuid4().hex[:16]}"
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO alerts (
                    id, timestamp_ns, rule_id, detection_id, track_id,
                    destination, priority, status, updated_ns, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    final_id,
                    timestamp_ns,
                    rule_id,
                    detection_id,
                    track_id,
                    destination,
                    priority,
                    status,
                    timestamp_ns,
                    _json_dumps(payload or {}),
                ),
            )
            await self._commit_if_needed(db)
        return final_id

    async def update_alert_status(
        self,
        *,
        alert_id: str,
        status: str,
        updated_ns: int,
        payload_patch: dict[str, Any] | None = None,
    ) -> bool:
        db = self._require_db()
        async with self._write_guard():
            row = await (
                await db.execute("SELECT payload_json FROM alerts WHERE id = ? LIMIT 1", (alert_id,))
            ).fetchone()
            if row is None:
                return False
            payload = _json_loads(row["payload_json"], {})
            if payload_patch:
                payload.update(payload_patch)
            await db.execute(
                """
                UPDATE alerts
                SET status = ?, updated_ns = ?, payload_json = ?
                WHERE id = ?
                """,
                (status, updated_ns, _json_dumps(payload), alert_id),
            )
            await self._commit_if_needed(db)
        return True

    async def list_pings(self, limit: int = 500) -> list[dict]:
        db = self._require_db()
        rows = await (
            await db.execute("SELECT * FROM pings ORDER BY timestamp_ns DESC LIMIT ?", (limit,))
        ).fetchall()
        result: list[dict] = []
        for row in rows:
            item = dict(row)
            item["metadata"] = _json_loads(item.pop("metadata_json"), {})
            result.append(item)
        return result

    async def insert_ping(
        self,
        *,
        timestamp_ns: int,
        ping_type: str,
        label: str | None,
        label_id: LabelId | None,
        spl_db: float | None,
        position_m: tuple[float, float, float] | None,
        position_geo: GeoPoint | None,
        source_detection_id: str | None,
        source_observation_id: str | None,
        source_track_id: str | None,
        retention_tier: str,
        metadata: dict[str, Any] | None = None,
        ping_id: str | None = None,
    ) -> str:
        db = self._require_db()
        final_id = ping_id or f"png-{uuid.uuid4().hex[:16]}"
        x = y = z = None
        if position_m is not None:
            x, y, z = position_m
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO pings (
                    id, timestamp_ns, lat, lon, alt, x, y, z, ping_type,
                    label_id, label, spl_db, source_detection_id, source_observation_id,
                    source_track_id, retention_tier, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    final_id,
                    timestamp_ns,
                    position_geo.lat if position_geo is not None else None,
                    position_geo.lon if position_geo is not None else None,
                    position_geo.alt_m if position_geo is not None else None,
                    x,
                    y,
                    z,
                    ping_type,
                    label_id,
                    label,
                    spl_db,
                    source_detection_id,
                    source_observation_id,
                    source_track_id,
                    retention_tier,
                    _json_dumps(metadata or {}),
                ),
            )
            await self._commit_if_needed(db)
        return final_id

    async def insert_environment(
        self,
        *,
        node_id: str,
        timestamp_ns: int,
        temperature_c: float | None,
        pressure_pa: float | None,
        humidity_fraction: float | None,
        wind_speed_mps: float | None,
        wind_dir_deg: float | None,
        solar_lux: float | None,
        metadata: dict[str, Any] | None = None,
        environment_id: str | None = None,
    ) -> str:
        db = self._require_db()
        final_id = environment_id or f"env-{uuid.uuid4().hex[:16]}"
        temperature_k = (float(temperature_c) + 273.15) if temperature_c is not None else None
        humidity_value = None if humidity_fraction is None else max(0.0, min(1.0, float(humidity_fraction)))
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO environment (
                    id, node_id, timestamp_ns, temperature_k, pressure_pa, humidity,
                    wind_speed_mps, wind_dir_deg, solar_lux, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    final_id,
                    node_id,
                    timestamp_ns,
                    temperature_k,
                    pressure_pa,
                    humidity_value,
                    wind_speed_mps,
                    wind_dir_deg,
                    solar_lux,
                    _json_dumps(metadata or {}),
                ),
            )
            await self._commit_if_needed(db)
        return final_id

    async def list_environment(self, limit: int = 500, node_id: str | None = None) -> list[dict]:
        db = self._require_db()
        if node_id:
            rows = await (
                await db.execute(
                    """
                    SELECT e.*, n.x AS x, n.y AS y, n.z AS z
                    FROM environment e
                    LEFT JOIN nodes n ON n.id = e.node_id
                    WHERE e.node_id = ?
                    ORDER BY e.timestamp_ns DESC
                    LIMIT ?
                    """,
                    (node_id, limit),
                )
            ).fetchall()
        else:
            rows = await (
                await db.execute(
                    """
                    SELECT e.*, n.x AS x, n.y AS y, n.z AS z
                    FROM environment e
                    LEFT JOIN nodes n ON n.id = e.node_id
                    ORDER BY e.timestamp_ns DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            ).fetchall()
        return [self._row_to_environment(row) for row in rows]

    async def list_latest_environment_per_node(self, limit: int = 500) -> list[dict[str, Any]]:
        db = self._require_db()
        rows = await (
            await db.execute(
                """
                SELECT e.*, n.x AS x, n.y AS y, n.z AS z
                FROM environment e
                INNER JOIN (
                    SELECT node_id, MAX(timestamp_ns) AS latest_ts
                    FROM environment
                    GROUP BY node_id
                ) latest
                ON latest.node_id = e.node_id AND latest.latest_ts = e.timestamp_ns
                LEFT JOIN nodes n ON n.id = e.node_id
                ORDER BY e.timestamp_ns DESC
                LIMIT ?
                """,
                (limit,),
            )
        ).fetchall()
        return [self._row_to_environment(row) for row in rows]

    async def insert_large_artifact(
        self,
        *,
        artifact_type: str,
        path: str,
        retention_tier: str,
        source_detection_id: str | None,
        source_track_id: str | None,
        created_ns: int,
        expires_ns: int | None,
        metadata: dict[str, Any] | None = None,
        artifact_id: str | None = None,
    ) -> str:
        db = self._require_db()
        final_id = artifact_id or f"lar-{uuid.uuid4().hex[:16]}"
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO large_artifacts (
                    id, artifact_type, path, retention_tier, source_detection_id,
                    source_track_id, metadata_json, created_ns, expires_ns
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    final_id,
                    artifact_type,
                    path,
                    retention_tier,
                    source_detection_id,
                    source_track_id,
                    _json_dumps(metadata or {}),
                    created_ns,
                    expires_ns,
                ),
            )
            await self._commit_if_needed(db)
        return final_id

    async def list_zones(self) -> list[dict]:
        db = self._require_db()
        rows = await (await db.execute("SELECT * FROM zones ORDER BY id ASC")).fetchall()
        result: list[dict] = []
        for row in rows:
            result.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "zone_type": row["zone_type"],
                    "polygon_geo": _json_loads(row["polygon_geo_json"], []),
                    "properties": _json_loads(row["properties_json"], {}),
                    "created_ns": row["created_ns"],
                }
            )
        return result

    async def upsert_zone(
        self,
        *,
        zone_id: str,
        name: str,
        zone_type: str,
        polygon_geo: list[list[float]],
        properties: dict[str, Any] | None,
        created_ns: int,
    ) -> None:
        db = self._require_db()
        async with self._write_guard():
            await db.execute(
                """
                INSERT INTO zones (id, name, zone_type, polygon_geo_json, properties_json, created_ns)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    zone_type=excluded.zone_type,
                    polygon_geo_json=excluded.polygon_geo_json,
                    properties_json=excluded.properties_json
                """,
                (zone_id, name, zone_type, _json_dumps(polygon_geo), _json_dumps(properties or {}), created_ns),
            )
            await self._commit_if_needed(db)

    async def delete_zone(self, zone_id: str) -> bool:
        db = self._require_db()
        async with self._write_guard():
            cursor = await db.execute("DELETE FROM zones WHERE id = ?", (zone_id,))
            await self._commit_if_needed(db)
            return cursor.rowcount > 0

    async def recent_alert_count(self, since_ns: int) -> int:
        db = self._require_db()
        row = await (
            await db.execute("SELECT COUNT(1) AS c FROM alerts WHERE timestamp_ns >= ?", (since_ns,))
        ).fetchone()
        return int(row["c"] if row is not None else 0)

    async def count_nodes_by_status(
        self,
        *,
        now_ns: int,
        degraded_after_seconds: float,
        offline_after_seconds: float,
    ) -> dict[str, int]:
        db = self._require_db()
        degraded_ns = int(degraded_after_seconds * 1_000_000_000)
        offline_ns = int(offline_after_seconds * 1_000_000_000)
        row = await (
            await db.execute(
                """
                SELECT
                    SUM(CASE WHEN (? - last_seen_ns) < ? THEN 1 ELSE 0 END) AS online_nodes,
                    SUM(CASE WHEN (? - last_seen_ns) >= ? AND (? - last_seen_ns) < ? THEN 1 ELSE 0 END) AS degraded_nodes,
                    SUM(CASE WHEN (? - last_seen_ns) >= ? THEN 1 ELSE 0 END) AS offline_nodes
                FROM nodes
                """,
                (
                    now_ns,
                    degraded_ns,
                    now_ns,
                    degraded_ns,
                    now_ns,
                    offline_ns,
                    now_ns,
                    offline_ns,
                ),
            )
        ).fetchone()
        if row is None:
            return {"online_nodes": 0, "degraded_nodes": 0, "offline_nodes": 0}
        return {
            "online_nodes": int(row["online_nodes"] or 0),
            "degraded_nodes": int(row["degraded_nodes"] or 0),
            "offline_nodes": int(row["offline_nodes"] or 0),
        }

    async def count_active_tracks(self) -> int:
        db = self._require_db()
        row = await (
            await db.execute(
                """
                SELECT COUNT(1) AS c
                FROM tracks
                WHERE status IN ('tentative', 'confirmed', 'coasting')
                """,
            )
        ).fetchone()
        return int(row["c"] if row is not None else 0)

    async def get_detection(self, detection_id: str) -> dict | None:
        db = self._require_db()
        row = await (
            await db.execute("SELECT * FROM detections WHERE id = ? LIMIT 1", (detection_id,))
        ).fetchone()
        if row is None:
            return None
        return self._row_to_detection(row).model_dump(mode="json")

    async def snippet_path_for_detection(self, detection_id: str) -> str | None:
        db = self._require_db()
        row = await (
            await db.execute(
                "SELECT snippet_path FROM detections WHERE id = ? LIMIT 1",
                (detection_id,),
            )
        ).fetchone()
        if row is None:
            return None
        return row["snippet_path"]

    async def cleanup_expired_snippets(self, now_ns: int) -> int:
        db = self._require_db()
        async with self._write_guard():
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

            await self._commit_if_needed(db)
        return len(rows)

    async def cleanup_policy_managed_files(
        self,
        *,
        now_ns: int,
        policy: CleanupPolicy,
        dry_run: bool = False,
    ) -> dict[str, int]:
        db = self._require_db()
        summary = {
            "snippet_files_deleted": 0,
            "snippet_records_cleared": 0,
            "artifact_files_deleted": 0,
            "artifact_rows_deleted": 0,
        }
        protected_tiers = {"config", "permanent"}

        async with self._write_guard():
            snippet_rows = await (
                await db.execute(
                    """
                    SELECT id, snippet_path, timestamp_ns, label, retention_tier
                    FROM detections
                    WHERE snippet_path IS NOT NULL
                    """
                )
            ).fetchall()
            snippet_ids_to_clear: list[str] = []
            for row in snippet_rows:
                if row["retention_tier"] in protected_tiers:
                    continue
                max_age_seconds = policy.snippet_max_age_seconds_for_label(row["label"])
                if max_age_seconds is None:
                    continue
                threshold_ns = now_ns - int(max_age_seconds * 1_000_000_000)
                if int(row["timestamp_ns"]) > threshold_ns:
                    continue
                snippet_ids_to_clear.append(str(row["id"]))
                snippet_path = row["snippet_path"]
                if snippet_path and Path(snippet_path).exists():
                    summary["snippet_files_deleted"] += 1
                    if not dry_run:
                        try:
                            Path(snippet_path).unlink(missing_ok=True)
                        except OSError:
                            pass
            summary["snippet_records_cleared"] = len(snippet_ids_to_clear)
            if snippet_ids_to_clear and not dry_run:
                await db.executemany(
                    "UPDATE detections SET snippet_path = NULL, snippet_expires_ns = NULL WHERE id = ?",
                    ((detection_id,) for detection_id in snippet_ids_to_clear),
                )

            artifact_rows = await (
                await db.execute(
                    """
                    SELECT la.id, la.path, la.created_ns, la.retention_tier,
                           COALESCE(d.label, t.label) AS label
                    FROM large_artifacts la
                    LEFT JOIN detections d ON d.id = la.source_detection_id
                    LEFT JOIN tracks t ON t.id = la.source_track_id
                    """
                )
            ).fetchall()
            artifact_ids_to_delete: list[str] = []
            for row in artifact_rows:
                if row["retention_tier"] in protected_tiers:
                    continue
                max_age_seconds = policy.artifact_max_age_seconds_for_label(row["label"])
                if max_age_seconds is None:
                    continue
                threshold_ns = now_ns - int(max_age_seconds * 1_000_000_000)
                if int(row["created_ns"]) > threshold_ns:
                    continue
                artifact_ids_to_delete.append(str(row["id"]))
                artifact_path = row["path"]
                if artifact_path and Path(artifact_path).exists():
                    summary["artifact_files_deleted"] += 1
                    if not dry_run:
                        try:
                            Path(artifact_path).unlink(missing_ok=True)
                        except OSError:
                            pass
            summary["artifact_rows_deleted"] = len(artifact_ids_to_delete)
            if artifact_ids_to_delete and not dry_run:
                await db.executemany(
                    "DELETE FROM large_artifacts WHERE id = ?",
                    ((artifact_id,) for artifact_id in artifact_ids_to_delete),
                )

            if not dry_run:
                await self._commit_if_needed(db)
        return summary

    async def cleanup_retention(
        self,
        *,
        now_ns: int,
        tier_ttls_seconds: dict[str, int],
        operational_ttls_seconds: dict[str, int] | None = None,
    ) -> dict[str, int]:
        db = self._require_db()
        summary = {
            "observations": 0,
            "detections": 0,
            "pings": 0,
            "large_artifacts": 0,
            "ingested_frames": 0,
            "track_updates": 0,
            "alerts": 0,
            "environment": 0,
            "dropped_tracks": 0,
        }
        protected_tiers = {"config", "permanent"}

        async with self._write_guard():
            for tier, ttl_s in tier_ttls_seconds.items():
                if tier in protected_tiers or ttl_s < 0:
                    continue
                threshold_ns = now_ns - int(ttl_s * 1_000_000_000)

                obs_cursor = await db.execute(
                    """
                    DELETE FROM observations
                    WHERE retention_tier = ?
                    AND created_ns <= ?
                    """,
                    (tier, threshold_ns),
                )
                summary["observations"] += max(0, obs_cursor.rowcount)

                det_rows = await (
                    await db.execute(
                        """
                        SELECT id, snippet_path
                        FROM detections
                        WHERE retention_tier = ?
                        AND timestamp_ns <= ?
                        """,
                        (tier, threshold_ns),
                    )
                ).fetchall()
                for row in det_rows:
                    snippet_path = row["snippet_path"]
                    if snippet_path:
                        try:
                            Path(snippet_path).unlink(missing_ok=True)
                        except OSError:
                            pass
                det_cursor = await db.execute(
                    """
                    DELETE FROM detections
                    WHERE retention_tier = ?
                    AND timestamp_ns <= ?
                    """,
                    (tier, threshold_ns),
                )
                summary["detections"] += max(0, det_cursor.rowcount)

                ping_cursor = await db.execute(
                    """
                    DELETE FROM pings
                    WHERE retention_tier = ?
                    AND timestamp_ns <= ?
                    """,
                    (tier, threshold_ns),
                )
                summary["pings"] += max(0, ping_cursor.rowcount)

                artifact_rows = await (
                    await db.execute(
                        """
                        SELECT id, path
                        FROM large_artifacts
                        WHERE retention_tier = ?
                        AND ((expires_ns IS NOT NULL AND expires_ns <= ?) OR created_ns <= ?)
                        """,
                        (tier, now_ns, threshold_ns),
                    )
                ).fetchall()
                for row in artifact_rows:
                    try:
                        Path(row["path"]).unlink(missing_ok=True)
                    except OSError:
                        pass
                artifact_cursor = await db.execute(
                    """
                    DELETE FROM large_artifacts
                    WHERE retention_tier = ?
                    AND ((expires_ns IS NOT NULL AND expires_ns <= ?) OR created_ns <= ?)
                    """,
                    (tier, now_ns, threshold_ns),
                )
                summary["large_artifacts"] += max(0, artifact_cursor.rowcount)

            # Frame receipt dedupe state tracks recent ingest only; align its TTL to short retention.
            ingest_receipt_ttl_seconds = max(0, int(tier_ttls_seconds.get("short", 0)))
            if ingest_receipt_ttl_seconds > 0:
                ingest_threshold_ns = now_ns - int(ingest_receipt_ttl_seconds * 1_000_000_000)
                ingested_frames_cursor = await db.execute(
                    """
                    DELETE FROM ingested_frames
                    WHERE created_ns <= ?
                    """,
                    (ingest_threshold_ns,),
                )
                summary["ingested_frames"] += max(0, ingested_frames_cursor.rowcount)

            operational_ttls = operational_ttls_seconds or {}
            for key, sql in (
                (
                    "track_updates",
                    """
                    DELETE FROM track_updates
                    WHERE timestamp_ns <= ?
                    """,
                ),
                (
                    "alerts",
                    """
                    DELETE FROM alerts
                    WHERE timestamp_ns <= ?
                    """,
                ),
                (
                    "environment",
                    """
                    DELETE FROM environment
                    WHERE timestamp_ns <= ?
                    """,
                ),
                (
                    "dropped_tracks",
                    """
                    DELETE FROM tracks
                    WHERE status = 'dropped'
                    AND last_seen_ns <= ?
                    """,
                ),
            ):
                ttl_s = operational_ttls.get(key)
                if ttl_s is None or ttl_s < 0:
                    continue
                threshold_ns = now_ns - int(ttl_s * 1_000_000_000)
                cursor = await db.execute(sql, (threshold_ns,))
                summary[key] += max(0, cursor.rowcount)

            await self._commit_if_needed(db)

        return summary

    @staticmethod
    def _row_to_geo(row: aiosqlite.Row) -> GeoPoint | None:
        if row["lat"] is None or row["lon"] is None:
            return None
        return GeoPoint(lat=float(row["lat"]), lon=float(row["lon"]), alt_m=float(row["alt"] or 0.0))

    @staticmethod
    def _row_to_environment(row: aiosqlite.Row) -> dict[str, Any]:
        position_m = None
        if row["x"] is not None and row["y"] is not None and row["z"] is not None:
            position_m = [float(row["x"]), float(row["y"]), float(row["z"])]
        temperature_c = None
        if row["temperature_k"] is not None:
            temperature_c = float(row["temperature_k"]) - 273.15
        return {
            "id": row["id"],
            "node_id": row["node_id"],
            "timestamp_ns": int(row["timestamp_ns"]),
            "temperature_c": temperature_c,
            "pressure_pa": float(row["pressure_pa"]) if row["pressure_pa"] is not None else None,
            "humidity_fraction": float(row["humidity"]) if row["humidity"] is not None else None,
            "wind_speed_mps": float(row["wind_speed_mps"]) if row["wind_speed_mps"] is not None else None,
            "wind_dir_deg": float(row["wind_dir_deg"]) if row["wind_dir_deg"] is not None else None,
            "solar_lux": float(row["solar_lux"]) if row["solar_lux"] is not None else None,
            "position_m": position_m,
            "metadata": _json_loads(row["metadata_json"], {}),
        }

    def _row_to_detection(self, row: aiosqlite.Row) -> DetectionEvent:
        return DetectionEvent(
            id=row["id"],
            event_id=row["event_id"] or row["id"],
            source_type=row["source_type"] or "raw_sensor",
            source_node_id=row["source_node_id"],
            timestamp_ns=int(row["timestamp_ns"]),
            toa_ns=int(row["toa_ns"] or row["timestamp_ns"]),
            tor_ns=int(row["tor_ns"] or row["timestamp_ns"]),
            time_quality=row["time_quality"] or "freerunning",
            stale_ns=row["stale_ns"],
            report_window_start_ns=row["report_window_start_ns"],
            report_window_end_ns=row["report_window_end_ns"],
            reporting_modality=row["reporting_modality"] or "localized",
            position_m=(float(row["x"]), float(row["y"]), float(row["z"])),
            position_geo=self._row_to_geo(row),
            position_covariance_m2=_json_loads(row["covariance_json"], None),
            confidence=float(row["confidence"]),
            gdop=float(row["gdop"]),
            label_id=row["label_id"],
            label=row["label"],
            label_category=row["label_category"] or "unknown",
            iff_category=row["iff_category"] or "unknown",
            label_confidence=float(row["label_confidence"]),
            spl_db=row["spl_db"],
            track_id=row["track_id"],
            reference_sensor=row["reference_sensor"],
            source_sensors=_json_loads(row["source_sensors_json"], []),
            source_observation_ids=_json_loads(row["source_observation_ids_json"], []),
            zone_ids=_json_loads(row["zone_ids_json"], []),
            tdoa_s=_json_loads(row["tdoa_json"], {}),
            classifier_scores=_json_loads(row["classifier_scores_json"], {}),
            feature_summary=_json_loads(row["feature_summary_json"], {}),
            retention_tier=row["retention_tier"] or "short",
            snippet_path=row["snippet_path"],
        )

    def _row_to_track(self, row: aiosqlite.Row) -> TrackState:
        return TrackState(
            id=row["id"],
            first_seen_ns=int(row["first_seen_ns"]),
            last_seen_ns=int(row["last_seen_ns"]),
            position_m=(float(row["x"]), float(row["y"]), float(row["z"])),
            position_geo=self._row_to_geo(row),
            position_covariance_m2=_json_loads(row["covariance_json"], None),
            velocity_mps=(float(row["vx"]), float(row["vy"]), float(row["vz"])),
            label_id=row["label_id"],
            label=row["label"],
            label_category=row["label_category"] or "unknown",
            iff_category=row["iff_category"] or "unknown",
            confidence=float(row["confidence"]),
            update_count=int(row["update_count"]),
            status=row["status"],
            capability_tier=row["capability_tier"] or "full_3d",
            tqi=float(row["tqi"]),
        )

    def _require_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Storage is not initialized")
        return self._db
