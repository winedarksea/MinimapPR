from __future__ import annotations

import hashlib
import json

import pytest

from minimappr.api.journal_reader import JournalPayloadHandle
from minimappr.core.journal_promotion import JournalPromotionRequest, promote_journal_handle
from minimappr.models import DetectionEvent, NodeSpec, NodeType, TrackState
from minimappr.storage.db import Storage


async def _seed_promotion_source_rows(
    storage: Storage,
    *,
    source_detection_id: str | None,
    source_track_id: str | None,
) -> None:
    if source_track_id is not None:
        await storage.upsert_track(
            TrackState(
                id=source_track_id,
                first_seen_ns=4_000,
                last_seen_ns=5_000,
                position_m=(1.0, 2.0, 3.0),
                label="unknown",
            )
        )
    if source_detection_id is not None:
        await storage.insert_detection(
            DetectionEvent(
                id=source_detection_id,
                event_id=source_detection_id,
                source_type="raw_sensor",
                source_node_id="node-2",
                timestamp_ns=4_500,
                toa_ns=4_500,
                tor_ns=4_500,
                position_m=(1.0, 2.0, 3.0),
                confidence=0.9,
                gdop=1.1,
                label="bird",
                label_category="avian",
                label_confidence=0.9,
                track_id=source_track_id,
                reference_sensor="audio_aux",
            ),
            snippet_path=None,
            snippet_expires_ns=None,
        )


@pytest.mark.asyncio
async def test_promote_journal_handle_materializes_artifact_and_observation(tmp_path):
    segment_path = tmp_path / "seg-1.bin"
    payload = b"classifier-ready-audio"
    segment_path.write_bytes(payload)
    storage = Storage(tmp_path / "promotion.db")
    await storage.initialize()
    try:
        await storage.upsert_node(
            NodeSpec(
                id="node-1",
                node_type=NodeType.POINT,
                position_m=(0.0, 0.0, 0.0),
                sensor_offsets_m=[(0.0, 0.0, 0.0)],
                capabilities=["audio"],
            ),
            last_seen_ns=1_000,
        )
        handle = JournalPayloadHandle(
            journal_epoch=7,
            segment_id="seg-1",
            stream_key="node-1__audio_main__test",
            payload_offset_bytes=0,
            payload_length_bytes=len(payload),
            sample_index_start=123,
            sample_count=456,
            integrity_hash=hashlib.sha256(payload).hexdigest(),
            segment_path=segment_path,
        )
        result = await promote_journal_handle(
            storage=storage,
            request=JournalPromotionRequest(
                handle=handle,
                journal_entry={
                    "observation_id": "obs-promoted-1",
                    "node_id": "node-1",
                    "stream_id": "audio_main",
                    "stream_key": handle.stream_key,
                    "sensor_type": "audio",
                    "source_type": "raw_sensor",
                    "toa_ns": 1_000,
                    "tor_ns": 2_000,
                    "time_quality": "gps_locked",
                    "sample_rate_hz": 16_000,
                    "journal_sequence": 9,
                },
                artifact_type="classifier_input_audio",
                retention_tier="long",
                large_artifact_dir=tmp_path / "large_artifacts",
            ),
            now_ns=3_000,
        )

        assert result.observation_id == "obs-promoted-1"
        assert result.artifact_path.read_bytes() == payload
        observations = await storage.list_observations_by_ids([result.observation_id])
        assert observations[0]["metadata"]["promotion_source"] == "journal_handle"
        assert observations[0]["metadata"]["artifact_type"] == "classifier_input_audio"
        artifact_rows = await (
            await storage._require_db().execute(
                "SELECT id, path, metadata_json FROM large_artifacts"
            )
        ).fetchall()
        assert artifact_rows[0]["id"] == result.artifact_id
        assert json.loads(artifact_rows[0]["metadata_json"])["artifact_type"] == "classifier_input_audio"
    finally:
        await storage.close()


@pytest.mark.parametrize(
    ("artifact_type", "source_detection_id", "source_track_id", "expires_ns"),
    [
        ("localized_review_audio", "det-1", "trk-1", 9_000),
        ("fault_capture_bundle", None, "trk-fault", None),
        ("raw_capture_bundle", None, None, 12_000),
    ],
)
@pytest.mark.asyncio
async def test_promote_journal_handle_records_review_fault_and_export_metadata(
    tmp_path,
    artifact_type,
    source_detection_id,
    source_track_id,
    expires_ns,
):
    segment_path = tmp_path / f"{artifact_type}.bin"
    payload = f"payload-{artifact_type}".encode("utf-8")
    segment_path.write_bytes(payload)
    storage = Storage(tmp_path / f"{artifact_type}.db")
    await storage.initialize()
    try:
        await storage.upsert_node(
            NodeSpec(
                id="node-2",
                node_type=NodeType.POINT,
                position_m=(0.0, 0.0, 0.0),
                sensor_offsets_m=[(0.0, 0.0, 0.0)],
                capabilities=["audio"],
            ),
            last_seen_ns=2_000,
        )
        handle = JournalPayloadHandle(
            journal_epoch=8,
            segment_id=f"seg-{artifact_type}",
            stream_key="node-2__audio_aux__test",
            payload_offset_bytes=0,
            payload_length_bytes=len(payload),
            sample_index_start=321,
            sample_count=654,
            integrity_hash=hashlib.sha256(payload).hexdigest(),
            segment_path=segment_path,
        )
        await _seed_promotion_source_rows(
            storage,
            source_detection_id=source_detection_id,
            source_track_id=source_track_id,
        )

        result = await promote_journal_handle(
            storage=storage,
            request=JournalPromotionRequest(
                handle=handle,
                journal_entry={
                    "node_id": "node-2",
                    "stream_key": handle.stream_key,
                    "sensor_type": "audio",
                    "source_type": "raw_sensor",
                    "ingest_received_ns": 5_000,
                    "journal_sequence": 14,
                },
                artifact_type=artifact_type,
                retention_tier="short",
                large_artifact_dir=tmp_path / "large_artifacts",
                source_detection_id=source_detection_id,
                source_track_id=source_track_id,
                expires_ns=expires_ns,
            ),
            now_ns=6_000,
        )

        observations = await storage.list_observations_by_ids([result.observation_id])
        observation = observations[0]
        assert observation["sensor_id"] == handle.stream_key
        assert observation["toa_ns"] == 5_000
        assert observation["tor_ns"] == 5_000
        assert observation["time_quality"] == "freerunning"
        assert observation["frame_sequence"] == 14
        assert observation["metadata"]["artifact_type"] == artifact_type
        assert observation["metadata"]["source_detection_id"] == source_detection_id
        assert observation["metadata"]["source_track_id"] == source_track_id
        assert observation["metadata"]["artifact_expires_ns"] == expires_ns

        artifact_rows = await (
            await storage._require_db().execute(
                "SELECT artifact_type, source_detection_id, source_track_id, expires_ns, metadata_json FROM large_artifacts"
            )
        ).fetchall()
        assert len(artifact_rows) == 1
        artifact_row = artifact_rows[0]
        assert artifact_row["artifact_type"] == artifact_type
        assert artifact_row["source_detection_id"] == source_detection_id
        assert artifact_row["source_track_id"] == source_track_id
        assert artifact_row["expires_ns"] == expires_ns
        artifact_metadata = json.loads(artifact_row["metadata_json"])
        assert artifact_metadata["source_observation_id"] == result.observation_id
        assert artifact_metadata["artifact_type"] == artifact_type
        assert artifact_metadata["source_detection_id"] == source_detection_id
        assert artifact_metadata["source_track_id"] == source_track_id
        assert artifact_metadata["expires_ns"] == expires_ns
    finally:
        await storage.close()
