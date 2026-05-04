from __future__ import annotations

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

    async def deliver_node_heartbeat(self, node) -> None:
        self.node_heartbeats.append(node)

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