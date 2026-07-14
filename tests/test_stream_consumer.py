from __future__ import annotations

import asyncio
import base64
import json

import numpy as np
import pytest

from minimappr.api.stream_consumer import IngestStreamConsumer, StreamConsumerConfig
from minimappr.core.audio_buffer import MultiSensorBuffer
from minimappr.utils.audio import decode_pcm16le_b64, encode_pcm16le_b64


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

    assert transport.node_heartbeats == []
    snapshot = consumer.snapshot_nodes()["sirith-1"]
    assert snapshot.node_payload["id"] == "sirith-1"
    assert snapshot.last_sample_time_ns == 123456789


@pytest.mark.asyncio
async def test_stream_consumer_mirrors_raw_audio_frame_into_audio_buffer() -> None:
    transport = _RecordingIngestTransport()
    audio_buffer = MultiSensorBuffer(max_duration_seconds=2.0)
    consumer = IngestStreamConsumer(
        config=StreamConsumerConfig(sidecar_base_url="http://127.0.0.1:8081"),
        ingest_transport=transport,
        audio_buffer=audio_buffer,
    )
    channels_first = np.array(
        [
            [0.10, 0.20, 0.30, 0.40],
            [-0.10, -0.20, -0.30, -0.40],
        ],
        dtype=np.float32,
    )

    await consumer._dispatch_sse_event(
        event_type="message",
        event_id="100",
        data_lines=[
            json.dumps(
                {
                    "manifest_type": "raw_audio_frame",
                    "created_ns": 10_000,
                    "node_context": {
                        "toa_ns": 1_000_000_000,
                        "time_quality": "gps_locked",
                        "node": {
                            "id": "sirith-raw-1",
                            "node_type": "sirith_tetra",
                            "position_m": [0.0, 0.0, 0.0],
                            "sensor_offsets_m": [
                                [0.0, 0.0, 0.0],
                                [0.1, 0.0, 0.0],
                            ],
                            "capabilities": ["audio"],
                            "metadata": {},
                        },
                    },
                    "raw_audio_frame": {
                        "stream_key": "sirith-raw-1",
                        "sample_rate_hz": 4,
                        "channel_count": 2,
                        "sample_count": 4,
                        "sample_format": "pcm16le",
                        "start_time_ns": 1_000_000_000,
                        "end_time_ns": 2_000_000_000,
                        "start_sample_index": 400,
                        "end_sample_index": 404,
                    },
                    "raw_audio_bytes": encode_pcm16le_b64(channels_first),
                }
            )
        ],
    )

    recent = await audio_buffer.get_recent_window_for_sensors(
        ["sirith-raw-1:ch0", "sirith-raw-1:ch1"],
        window_seconds=1.0,
    )

    assert recent is not None
    channel_windows, sample_rate_hz, latest_end_ns = recent
    assert sample_rate_hz == 4
    assert latest_end_ns == 2_000_000_000
    assert channel_windows["sirith-raw-1:ch0"] == pytest.approx(channels_first[0], abs=4e-5)
    assert channel_windows["sirith-raw-1:ch1"] == pytest.approx(channels_first[1], abs=4e-5)
    snapshot = consumer.snapshot_nodes()["sirith-raw-1"]
    assert snapshot.sample_rate_hz == 4
    assert snapshot.active_sensor_count == 2
    assert snapshot.last_sample_time_ns == 2_000_000_000


@pytest.mark.asyncio
async def test_stream_consumer_drops_bad_raw_coverage_and_advances_event_cursor(
    caplog: pytest.LogCaptureFixture,
) -> None:
    audio_buffer = MultiSensorBuffer(max_duration_seconds=2.0)
    consumer = IngestStreamConsumer(
        config=StreamConsumerConfig(sidecar_base_url="http://127.0.0.1:8081"),
        ingest_transport=_RecordingIngestTransport(),
        audio_buffer=audio_buffer,
    )
    samples = np.array([[0.10, 0.20, 0.30, 0.40]], dtype=np.float32)

    def raw_event(*, end_sample_index: int) -> str:
        return json.dumps(
            {
                "manifest_type": "raw_audio_frame",
                "created_ns": 10_000,
                "node_context": {
                    "toa_ns": 1_000_000_000,
                    "time_quality": "gps_locked",
                    "node": {
                        "id": "sirith-coverage-guard",
                        "node_type": "sirith_tetra",
                        "position_m": [0.0, 0.0, 0.0],
                        "sensor_offsets_m": [[0.0, 0.0, 0.0]],
                        "capabilities": ["audio"],
                        "metadata": {},
                    },
                },
                "raw_audio_frame": {
                    "stream_key": "sirith-coverage-guard",
                    "sample_rate_hz": 4,
                    "channel_count": 1,
                    "sample_count": 4,
                    "sample_format": "pcm16le",
                    "start_time_ns": 1_000_000_000,
                    "end_time_ns": 2_000_000_000,
                    "start_sample_index": 400,
                    "end_sample_index": end_sample_index,
                    "source_manifest_id": "source-coverage-guard",
                },
                "raw_audio_bytes": encode_pcm16le_b64(samples),
            }
        )

    with caplog.at_level("WARNING", logger="minimappr.api.stream_consumer"):
        await consumer._dispatch_sse_event(
            event_type="message",
            event_id="200",
            data_lines=[raw_event(end_sample_index=408)],
        )

    assert consumer._stream_request_headers()["Last-Event-ID"] == "200"
    assert consumer.snapshot_nodes() == {}
    assert await audio_buffer.get_recent_window_for_sensors(
        ["sirith-coverage-guard:ch0"], window_seconds=1.0
    ) is None
    assert "raw_audio_frame coverage validation failed" in caplog.text

    await consumer._dispatch_sse_event(
        event_type="message",
        event_id="201",
        data_lines=[raw_event(end_sample_index=404)],
    )

    assert consumer._stream_request_headers()["Last-Event-ID"] == "201"
    assert "sirith-coverage-guard" in consumer.snapshot_nodes()
    recent = await audio_buffer.get_recent_window_for_sensors(
        ["sirith-coverage-guard:ch0"], window_seconds=1.0
    )
    assert recent is not None
    assert recent[0]["sirith-coverage-guard:ch0"] == pytest.approx(samples[0], abs=4e-5)


@pytest.mark.asyncio
async def test_stream_consumer_contains_raw_buffer_value_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _RejectingAudioBuffer:
        async def append(self, **_kwargs) -> None:
            raise ValueError("synthetic append rejection")

    consumer = IngestStreamConsumer(
        config=StreamConsumerConfig(sidecar_base_url="http://127.0.0.1:8081"),
        ingest_transport=_RecordingIngestTransport(),
        audio_buffer=_RejectingAudioBuffer(),  # type: ignore[arg-type]
    )
    samples = np.array([[0.10, 0.20]], dtype=np.float32)
    payload = json.dumps(
        {
            "manifest_type": "raw_audio_frame",
            "node_context": {
                "node": {
                    "id": "sirith-value-error",
                    "node_type": "sirith_tetra",
                    "position_m": [0.0, 0.0, 0.0],
                    "sensor_offsets_m": [[0.0, 0.0, 0.0]],
                    "capabilities": ["audio"],
                    "metadata": {},
                }
            },
            "raw_audio_frame": {
                "sample_rate_hz": 2,
                "channel_count": 1,
                "sample_count": 2,
                "start_time_ns": 1_000_000_000,
                "end_time_ns": 2_000_000_000,
                "start_sample_index": 20,
                "end_sample_index": 22,
            },
            "raw_audio_bytes": encode_pcm16le_b64(samples),
        }
    )

    with caplog.at_level("WARNING", logger="minimappr.api.stream_consumer"):
        await consumer._dispatch_sse_event(
            event_type="message", event_id="300", data_lines=[payload]
        )

    assert consumer._stream_request_headers()["Last-Event-ID"] == "300"
    assert consumer.snapshot_nodes() == {}
    assert "raw_audio_frame buffer append failed; dropping event" in caplog.text


@pytest.mark.asyncio
async def test_stream_consumer_raw_audio_frame_matches_direct_buffer_append_for_late_gap_fill() -> None:
    transport = _RecordingIngestTransport()
    mirrored_buffer = MultiSensorBuffer(max_duration_seconds=4.0)
    reference_buffer = MultiSensorBuffer(max_duration_seconds=4.0)
    consumer = IngestStreamConsumer(
        config=StreamConsumerConfig(sidecar_base_url="http://127.0.0.1:8081"),
        ingest_transport=transport,
        audio_buffer=mirrored_buffer,
    )
    sensor_ids = ["sirith-raw-2:ch0", "sirith-raw-2:ch1"]
    frame_specs = [
        {
            "event_id": "101",
            "start_time_ns": 1_000_000_000,
            "end_time_ns": 2_000_000_000,
            "start_sample_index": 400,
            "end_sample_index": 404,
            "samples": np.array(
                [
                    [0.10, 0.20, 0.30, 0.40],
                    [-0.10, -0.20, -0.30, -0.40],
                ],
                dtype=np.float32,
            ),
        },
        {
            "event_id": "102",
            "start_time_ns": 3_000_000_000,
            "end_time_ns": 4_000_000_000,
            "start_sample_index": 408,
            "end_sample_index": 412,
            "samples": np.array(
                [
                    [0.90, 0.80, 0.70, 0.60],
                    [-0.90, -0.80, -0.70, -0.60],
                ],
                dtype=np.float32,
            ),
        },
        {
            "event_id": "103",
            "start_time_ns": 2_000_000_000,
            "end_time_ns": 3_000_000_000,
            "start_sample_index": 404,
            "end_sample_index": 408,
            "samples": np.array(
                [
                    [0.50, 0.60, 0.70, 0.80],
                    [-0.50, -0.60, -0.70, -0.80],
                ],
                dtype=np.float32,
            ),
        },
    ]

    for frame in frame_specs:
        quantized_channels = decode_pcm16le_b64(
            encode_pcm16le_b64(frame["samples"]),
            2,
        )
        for channel_index, sensor_id in enumerate(sensor_ids):
            await reference_buffer.append(
                sensor_id=sensor_id,
                sample_rate_hz=4,
                start_time_ns=frame["start_time_ns"],
                samples=quantized_channels[channel_index],
                start_sample_index=frame["start_sample_index"],
                end_sample_index=frame["end_sample_index"],
                end_time_ns=frame["end_time_ns"],
            )

        await consumer._dispatch_sse_event(
            event_type="message",
            event_id=frame["event_id"],
            data_lines=[
                json.dumps(
                    {
                        "manifest_type": "raw_audio_frame",
                        "created_ns": 10_000 + int(frame["event_id"]),
                        "node_context": {
                            "toa_ns": frame["start_time_ns"],
                            "time_quality": "gps_locked",
                            "node": {
                                "id": "sirith-raw-2",
                                "node_type": "sirith_tetra",
                                "position_m": [0.0, 0.0, 0.0],
                                "sensor_offsets_m": [
                                    [0.0, 0.0, 0.0],
                                    [0.1, 0.0, 0.0],
                                ],
                                "capabilities": ["audio"],
                                "metadata": {},
                            },
                        },
                        "raw_audio_frame": {
                            "stream_key": "sirith-raw-2",
                            "sample_rate_hz": 4,
                            "channel_count": 2,
                            "sample_count": 4,
                            "sample_format": "pcm16le",
                            "start_time_ns": frame["start_time_ns"],
                            "end_time_ns": frame["end_time_ns"],
                            "start_sample_index": frame["start_sample_index"],
                            "end_sample_index": frame["end_sample_index"],
                        },
                        "raw_audio_bytes": encode_pcm16le_b64(frame["samples"]),
                    }
                )
            ],
        )

    mirrored_recent = await mirrored_buffer.get_recent_window_for_sensors(
        sensor_ids,
        window_seconds=3.0,
    )
    reference_recent = await reference_buffer.get_recent_window_for_sensors(
        sensor_ids,
        window_seconds=3.0,
    )

    assert mirrored_recent is not None
    assert reference_recent is not None
    mirrored_windows, mirrored_sample_rate_hz, mirrored_end_ns = mirrored_recent
    reference_windows, reference_sample_rate_hz, reference_end_ns = reference_recent
    assert mirrored_sample_rate_hz == reference_sample_rate_hz == 4
    assert mirrored_end_ns == reference_end_ns == 4_000_000_000
    for sensor_id in sensor_ids:
        assert mirrored_windows[sensor_id] == pytest.approx(reference_windows[sensor_id], abs=4e-5)

    mirrored_coverage = await mirrored_buffer.get_synchronized_window_ending_at_coverage_stats(
        sensor_ids=sensor_ids,
        end_time_ns=4_000_000_000,
        window_seconds=3.0,
        sample_rate_hz=4,
    )
    reference_coverage = await reference_buffer.get_synchronized_window_ending_at_coverage_stats(
        sensor_ids=sensor_ids,
        end_time_ns=4_000_000_000,
        window_seconds=3.0,
        sample_rate_hz=4,
    )

    assert {
        sensor_id: stats.to_json() for sensor_id, stats in mirrored_coverage.items()
    } == {
        sensor_id: stats.to_json() for sensor_id, stats in reference_coverage.items()
    }
    assert all(stats.coverage_ratio == 1.0 for stats in mirrored_coverage.values())


@pytest.mark.asyncio
async def test_stream_consumer_records_environment_and_audio_debug_from_localization_context() -> None:
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

    assert transport.node_heartbeats == []

    snapshot = consumer.snapshot_nodes()["sirith-point-1"]
    assert snapshot.sample_rate_hz == 16000
    assert snapshot.active_sensor_count == 1
    assert snapshot.rms == pytest.approx(0.03125)
    assert snapshot.node_payload["metadata"]["time_quality"] == "gps_locked"
    assert snapshot.latest_environment["temperature_c"] == pytest.approx(21.5)
    assert snapshot.latest_environment["humidity_fraction"] == pytest.approx(0.44)
    assert snapshot.latest_environment["source"] == "sht45"
    assert snapshot.latest_environment["timestamp_ns"] == 123456790

    # Environment from node_context must be forwarded to the provider, not only
    # stored in the in-memory snapshot.
    assert len(transport.environment_samples) == 1
    env_node_id, env_sample = transport.environment_samples[0]
    assert env_node_id == "sirith-point-1"
    assert env_sample.temperature_c == pytest.approx(21.5)
    assert env_sample.humidity_fraction == pytest.approx(0.44)


@pytest.mark.asyncio
async def test_stream_consumer_keeps_localization_context_in_memory_only() -> None:
    transport = _RecordingIngestTransport()
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

    assert transport.node_heartbeats == []
    snapshot = consumer.snapshot_nodes()["sirith-point-2"]
    assert snapshot.latest_environment["temperature_c"] == pytest.approx(22.0)
    assert snapshot.latest_environment["humidity_fraction"] == pytest.approx(0.4)

    # Environment must be forwarded to the provider, not only kept in the snapshot.
    assert len(transport.environment_samples) == 1
    env_node_id, env_sample = transport.environment_samples[0]
    assert env_node_id == "sirith-point-2"
    assert env_sample.temperature_c == pytest.approx(22.0)
    assert env_sample.humidity_fraction == pytest.approx(0.4)


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
                        "render_start_ns": 123455792,
                        "render_end_ns": 123456792,
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
    assert payload.render_start_ns == 123455792
    assert payload.render_end_ns == 123456792
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
