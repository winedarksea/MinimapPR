from __future__ import annotations

import asyncio
import base64
import json

import pytest

from minimappr.api.stream_consumer import IngestStreamConsumer, StreamConsumerConfig


class _RecordingIngestTransport:
    def __init__(self) -> None:
        self.localized_render_deliveries = []
        self.node_heartbeats = []
        self.environment_samples = []

    async def deliver_localized_render(self, payload) -> None:
        self.localized_render_deliveries.append(payload)

    async def deliver_node_heartbeat(
        self,
        node,
        *,
        last_sample_time_ns=None,
        sample_rate_hz=None,
        active_sensor_count=None,
        rms=None,
    ) -> None:
        self.node_heartbeats.append(
            {
                "node": node,
                "last_sample_time_ns": last_sample_time_ns,
                "sample_rate_hz": sample_rate_hz,
                "active_sensor_count": active_sensor_count,
                "rms": rms,
            }
        )

    async def deliver_environment_sample(self, *, node_id, sample) -> None:
        self.environment_samples.append((node_id, sample))


class _HeartbeatFirstIngestTransport(_RecordingIngestTransport):
    def __init__(self) -> None:
        super().__init__()
        self._heartbeat_seen = False

    async def deliver_node_heartbeat(
        self,
        node,
        *,
        last_sample_time_ns=None,
        sample_rate_hz=None,
        active_sensor_count=None,
        rms=None,
    ) -> None:
        self._heartbeat_seen = True
        await super().deliver_node_heartbeat(
            node,
            last_sample_time_ns=last_sample_time_ns,
            sample_rate_hz=sample_rate_hz,
            active_sensor_count=active_sensor_count,
            rms=rms,
        )

    async def deliver_environment_sample(self, *, node_id, sample) -> None:
        if not self._heartbeat_seen:
            raise AssertionError("environment sample arrived before node heartbeat")
        await super().deliver_environment_sample(node_id=node_id, sample=sample)


@pytest.mark.asyncio
async def test_stream_consumer_tracks_last_event_id_after_message_dispatch() -> None:
    transport = _RecordingIngestTransport()
    consumer = IngestStreamConsumer(
        config=StreamConsumerConfig(sidecar_base_url="http://127.0.0.1:8081"),
        ingest_transport=transport,
    )

    await consumer._dispatch_sse_event(
        event_type="message",
        event_id="42",
        data_lines=[
            json.dumps(
                {
                    "manifest_type": "env_sample_append",
                    "env_samples": {
                        "samples": [
                            {
                                "node_id": "sirith-1",
                                "sample": {
                                    "timestamp_ns": 123,
                                    "temperature_c": 19.5,
                                },
                            }
                        ]
                    },
                }
            )
        ],
    )

    assert consumer._stream_request_headers()["Last-Event-ID"] == "42"
    assert len(transport.environment_samples) == 1
    node_id, sample = transport.environment_samples[0]
    assert node_id == "sirith-1"
    assert sample.temperature_c == pytest.approx(19.5)


@pytest.mark.asyncio
async def test_stream_consumer_replay_gap_does_not_advance_cursor() -> None:
    transport = _RecordingIngestTransport()
    consumer = IngestStreamConsumer(
        config=StreamConsumerConfig(sidecar_base_url="http://127.0.0.1:8081"),
        ingest_transport=transport,
    )
    consumer._last_event_id = "42"

    await consumer._dispatch_sse_event(
        event_type="replay_gap",
        event_id=None,
        data_lines=[json.dumps({"requested_after": 42, "oldest_available": 100})],
    )

    assert consumer._stream_request_headers()["Last-Event-ID"] == "42"
    assert transport.environment_samples == []
    assert transport.localized_render_deliveries == []
    assert transport.node_heartbeats == []


@pytest.mark.asyncio
async def test_stream_consumer_uses_localization_toa_for_node_audio_freshness() -> None:
    transport = _RecordingIngestTransport()
    consumer = IngestStreamConsumer(
        config=StreamConsumerConfig(sidecar_base_url="http://127.0.0.1:8081"),
        ingest_transport=transport,
    )

    await consumer._dispatch_sse_event(
        event_type="message",
        event_id="99",
        data_lines=[
            json.dumps(
                {
                    "manifest_type": "localization_result",
                    "created_ns": 555,
                    "node_context": {
                        "toa_ns": 123456789,
                        "node": {
                            "id": "sirith-1",
                            "node_type": "sirith_tetra",
                            "position_m": [0.0, 0.0, 0.0],
                            "sensor_offsets_m": [[0.0, 0.0, 0.0]],
                            "capabilities": ["audio"],
                            "metadata": {},
                        },
                    },
                }
            )
        ],
    )

    assert len(transport.node_heartbeats) == 1
    heartbeat = transport.node_heartbeats[0]
    node = heartbeat["node"]
    assert node.id == "sirith-1"
    assert heartbeat["last_sample_time_ns"] == 123456789


@pytest.mark.asyncio
async def test_stream_consumer_forwards_environment_and_audio_debug_from_localization_context() -> None:
    transport = _RecordingIngestTransport()
    consumer = IngestStreamConsumer(
        config=StreamConsumerConfig(sidecar_base_url="http://127.0.0.1:8081"),
        ingest_transport=transport,
    )

    await consumer._dispatch_sse_event(
        event_type="message",
        event_id="100",
        data_lines=[
            json.dumps(
                {
                    "manifest_type": "localization_result",
                    "created_ns": 777,
                    "node_context": {
                        "toa_ns": 123456790,
                        "time_quality": "gps_locked",
                        "environment": {
                            "temperature_c": 21.5,
                            "humidity_fraction": 0.44,
                            "source": "sht45",
                        },
                        "audio_debug": {
                            "sample_rate_hz": 16000,
                            "active_sensor_count": 1,
                            "rms": 0.03125,
                        },
                        "node": {
                            "id": "sirith-point-1",
                            "node_type": "point",
                            "position_m": [0.0, 0.0, 0.0],
                            "sensor_offsets_m": [[0.0, 0.0, 0.0]],
                            "capabilities": ["audio"],
                            "metadata": {"gps": {"signal": "fix_3d", "position_source": "gps_nmea_uart"}},
                        },
                    },
                }
            )
        ],
    )

    assert len(transport.environment_samples) == 1
    node_id, sample = transport.environment_samples[0]
    assert node_id == "sirith-point-1"
    assert sample.temperature_c == pytest.approx(21.5)
    assert sample.humidity_fraction == pytest.approx(0.44)
    assert sample.source == "sht45"
    assert sample.timestamp_ns == 123456790

    assert len(transport.node_heartbeats) == 1
    heartbeat = transport.node_heartbeats[0]
    assert heartbeat["sample_rate_hz"] == 16000
    assert heartbeat["active_sensor_count"] == 1
    assert heartbeat["rms"] == pytest.approx(0.03125)
    assert heartbeat["node"].metadata["time_quality"] == "gps_locked"


@pytest.mark.asyncio
async def test_stream_consumer_persists_node_before_environment_from_localization_context() -> None:
    transport = _HeartbeatFirstIngestTransport()
    consumer = IngestStreamConsumer(
        config=StreamConsumerConfig(sidecar_base_url="http://127.0.0.1:8081"),
        ingest_transport=transport,
    )

    await consumer._dispatch_sse_event(
        event_type="message",
        event_id="101",
        data_lines=[
            json.dumps(
                {
                    "manifest_type": "localization_result",
                    "created_ns": 888,
                    "node_context": {
                        "toa_ns": 123456791,
                        "environment": {
                            "temperature_c": 22.0,
                            "humidity_fraction": 0.4,
                            "source": "sht45",
                        },
                        "audio_debug": {
                            "sample_rate_hz": 16000,
                            "active_sensor_count": 1,
                            "rms": 0.02,
                        },
                        "node": {
                            "id": "sirith-point-2",
                            "node_type": "point",
                            "position_m": [0.0, 0.0, 0.0],
                            "sensor_offsets_m": [[0.0, 0.0, 0.0]],
                            "capabilities": ["audio"],
                            "metadata": {},
                        },
                    },
                }
            )
        ],
    )

    assert len(transport.node_heartbeats) == 1
    assert len(transport.environment_samples) == 1


@pytest.mark.asyncio
async def test_stream_consumer_decodes_single_point_classifier_render_as_omni() -> None:
    transport = _RecordingIngestTransport()
    consumer = IngestStreamConsumer(
        config=StreamConsumerConfig(sidecar_base_url="http://127.0.0.1:8081"),
        ingest_transport=transport,
    )

    render_pcm16 = (1000).to_bytes(2, "little", signed=True) * 16

    await consumer._dispatch_sse_event(
        event_type="message",
        event_id="102",
        data_lines=[
            json.dumps(
                {
                    "manifest_type": "classifier_render",
                    "manifest_id": "manifest-point-render-1",
                    "created_ns": 999,
                    "source_handles": [
                        {
                            "journal_epoch": 1,
                            "segment_id": "seg-mem-00000000000000000001-00000000000000000002",
                            "stream_key": "sirith-point-1__audio_main__abcd",
                            "payload_offset_bytes": 0,
                            "payload_length_bytes": 0,
                            "toa_ns": 123456792,
                            "tor_ns": 123456793,
                            "sample_index_start": 0,
                            "sample_count": 16,
                            "integrity_hash": "",
                            "segment_path": "/tmp/unused-memory-path.bin",
                        }
                    ],
                    "derived_handle": {
                        "journal_epoch": 1,
                        "segment_id": "derived-mem-1",
                        "stream_key": "sirith-point-1__audio_main__abcd",
                        "payload_offset_bytes": 0,
                        "payload_length_bytes": len(render_pcm16),
                        "sample_index_start": 0,
                        "sample_count": 16,
                        "integrity_hash": "",
                        "segment_path": "/tmp/unused-derived.bin",
                    },
                    "classifier_render": {
                        "sample_rate_hz": 16000,
                        "sample_count": 16,
                        "channels": 1,
                        "sample_format": "pcm16le",
                        "render_kind": "birdnet_omni_fallback",
                        "fallback_reason": "single_point_node",
                    },
                    "node_context": {
                        "toa_ns": 123456792,
                        "time_quality": "gps_locked",
                        "node": {
                            "id": "sirith-point-1",
                            "node_type": "point",
                            "position_m": [3.0, 1.0, 2.0],
                            "sensor_offsets_m": [[0.0, 0.0, 0.0]],
                            "capabilities": ["audio"],
                            "metadata": {},
                        },
                    },
                    "raw_render_bytes": base64.b64encode(render_pcm16).decode("ascii"),
                }
            )
        ],
    )

    assert len(transport.localized_render_deliveries) == 1
    payload = transport.localized_render_deliveries[0]
    assert payload.node.id == "sirith-point-1"
    assert payload.reporting_modality == "omni"
    assert payload.localization_method == "rust_classifier_render_fallback"
    assert payload.fallback_reason == "single_point_node"
    assert tuple(payload.localization_position_m) == pytest.approx((3.0, 1.0, 2.0))


def test_stream_consumer_builds_valid_httpx_timeout() -> None:
    consumer = IngestStreamConsumer(
        config=StreamConsumerConfig(
            sidecar_base_url="http://127.0.0.1:8081",
            read_timeout_seconds=30.0,
        ),
        ingest_transport=_RecordingIngestTransport(),
    )

    timeout = consumer._client_timeout()

    assert timeout.connect == pytest.approx(10.0)
    assert timeout.read == pytest.approx(30.0)
    assert timeout.write == pytest.approx(10.0)
    assert timeout.pool == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_stream_consumer_start_restarts_when_previous_task_is_done() -> None:
    consumer = IngestStreamConsumer(
        config=StreamConsumerConfig(sidecar_base_url="http://127.0.0.1:8081"),
        ingest_transport=_RecordingIngestTransport(),
    )
    started = asyncio.Event()

    async def fake_run_loop() -> None:
        started.set()
        while consumer._running:
            await asyncio.sleep(0.01)

    consumer._run_loop = fake_run_loop  # type: ignore[method-assign]
    completed_task = asyncio.create_task(asyncio.sleep(0))
    await completed_task
    consumer._running = True
    consumer._task = completed_task

    consumer.start()

    await asyncio.wait_for(started.wait(), timeout=1.0)
    assert consumer.is_running
    assert consumer._task is not completed_task

    await consumer.stop()