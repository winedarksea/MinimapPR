"""Round-trip the sim binary ingest encoder through the server's own parser."""

from __future__ import annotations

import numpy as np
import pytest

from minimappr.api.binary_ingest import parse_binary_ingest_payload
from minimappr.models import GeoPoint, NodeType, TimeQuality
from minimappr.sim.binary_ingest_encoder import (
    BinaryIngestFrame,
    BinaryIngestNodeHeader,
    ContinuousCaptureFraming,
    encode_binary_ingest_payload,
    frames_from_continuous_capture,
)

SAMPLE_RATE_HZ = 48_000
FRAME_SAMPLES = 480
PACKET_START_TIME_NS = 1_739_970_000_123_456_789
TIMESTAMP_STEP_NS = 21_000_000


def _planar_node() -> BinaryIngestNodeHeader:
    return BinaryIngestNodeHeader(
        node_id="planar-roundtrip",
        node_type=NodeType.SIRITH_PLANAR,
        sensor_offsets_m=[(0.01, 0.01, 0.0), (-0.01, 0.01, 0.0), (-0.01, -0.01, 0.0), (0.01, -0.01, 0.0), (0.0, 0.0, 0.0)],
        position_geo=GeoPoint(lat=44.9875, lon=-93.2575, alt_m=281.5),
    )


@pytest.mark.parametrize("format_version", [2, 3])
def test_encoded_upload_round_trips_timing_quality_and_samples(format_version: int) -> None:
    capture = np.random.default_rng(7).uniform(-0.5, 0.5, size=(5, 3 * FRAME_SAMPLES)).astype(np.float32)
    frames = frames_from_continuous_capture(
        capture,
        ContinuousCaptureFraming(
            sample_rate_hz=SAMPLE_RATE_HZ,
            first_frame_start_time_ns=PACKET_START_TIME_NS,
            frame_samples=FRAME_SAMPLES,
            first_sequence=40,
            time_quality_by_frame_index={2: TimeQuality.NTP_DISCIPLINED},
            timestamp_offset_ns_by_frame_index={2: TIMESTAMP_STEP_NS},
        ),
    )
    frames[0] = BinaryIngestFrame(
        channels_first_samples=frames[0].channels_first_samples,
        start_time_ns=frames[0].start_time_ns,
        sample_rate_hz=SAMPLE_RATE_HZ,
        start_sample_index=0,
        sequence=40,
        temperature_c=12.5,
        humidity_fraction=0.25,
    )

    parsed = parse_binary_ingest_payload(
        encode_binary_ingest_payload(_planar_node(), frames, format_version=format_version)
    )

    assert parsed.node.id == "planar-roundtrip"
    assert parsed.node.node_type == NodeType.SIRITH_PLANAR
    assert parsed.node.position_geo is not None
    assert parsed.node.position_geo.lat == pytest.approx(44.9875, abs=1e-5)
    assert len(parsed.node.sensor_offsets_m) == 5
    assert parsed.node.metadata["gps"] == {"signal": "fix_3d", "position_source": "gps_nmea_uart"}
    frame_duration_ns = FRAME_SAMPLES * 1_000_000_000 // SAMPLE_RATE_HZ
    expected_start_times_ns = [
        PACKET_START_TIME_NS,
        PACKET_START_TIME_NS + frame_duration_ns,
        PACKET_START_TIME_NS + 2 * frame_duration_ns + TIMESTAMP_STEP_NS,
    ]
    assert [buffered.frame.start_time_ns for buffered in parsed.buffered_frames] == expected_start_times_ns
    assert [buffered.frame.start_sample_index for buffered in parsed.buffered_frames] == [0, FRAME_SAMPLES, 2 * FRAME_SAMPLES]
    assert [buffered.frame.sequence for buffered in parsed.buffered_frames] == [40, 41, 42]
    assert [buffered.frame.time_quality for buffered in parsed.buffered_frames] == [
        TimeQuality.GPS_LOCKED,
        TimeQuality.GPS_LOCKED,
        TimeQuality.NTP_DISCIPLINED,
    ]
    assert parsed.buffered_frames[0].environment is not None
    assert parsed.buffered_frames[0].environment.temperature_c == pytest.approx(12.5)
    decoded = np.concatenate([buffered.decoded_audio for buffered in parsed.buffered_frames], axis=1)
    np.testing.assert_allclose(decoded, capture, atol=1.0 / 32768.0)
