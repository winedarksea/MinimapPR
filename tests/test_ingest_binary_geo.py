from __future__ import annotations

import struct
from types import SimpleNamespace

import pytest

from pydantic import ValidationError

from minimappr.api.binary_ingest import parse_binary_ingest_payload
from minimappr.config import Settings
from minimappr.core.geo import LocalCoordinateFrame
from minimappr.core.ingest import IngestProcessor
from minimappr.models import GeoPoint, NodeSpec, NodeType


def _binary_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    assert len(encoded) <= 255
    return struct.pack("<B", len(encoded)) + encoded


def _binary_node(*, position_source: str, geo: GeoPoint) -> bytes:
    payload = bytearray()
    payload += _binary_string("binary-node-1")
    payload += struct.pack("<B", 0)
    payload += struct.pack("<B", 1)
    payload += struct.pack("<fff", geo.lat, geo.lon, geo.alt_m)
    payload += struct.pack("<B", 1)
    payload += struct.pack("<fff", 0.0, 0.0, 0.0)
    payload += struct.pack("<B", 2)
    payload += _binary_string("audio")
    payload += _binary_string("gps_optional")
    payload += _binary_string("test-hardware")
    payload += _binary_string("test-firmware")
    payload += _binary_string("fix_3d")
    payload += _binary_string(position_source)
    payload += struct.pack("<I", 3)
    return bytes(payload)


def _binary_node_no_geo(*, position_source: str) -> bytes:
    """Legacy firmware: reports no geo position (has_geo flag == 0)."""
    payload = bytearray()
    payload += _binary_string("legacy-node-1")
    payload += struct.pack("<B", 0)  # node type
    payload += struct.pack("<B", 0)  # has_geo_position = False
    payload += struct.pack("<B", 1)  # sensor count
    payload += struct.pack("<fff", 0.0, 0.0, 0.0)
    payload += struct.pack("<B", 1)  # capability count
    payload += _binary_string("audio")
    payload += _binary_string("test-hardware")
    payload += _binary_string("test-firmware")
    payload += _binary_string("missing")
    payload += _binary_string(position_source)
    payload += struct.pack("<I", 3)  # boot count
    return bytes(payload)


def _binary_frame(*, sample_rate_hz: int = 16_000, channels: int = 1, samples_per_channel: int = 2) -> bytes:
    samples = b"".join(
        struct.pack("<h", 32767 if sample_index % 2 else 0)
        for sample_index in range(samples_per_channel * channels)
    )
    payload = bytearray()
    payload += struct.pack(
        "<QQQQIBQQQB",
        1_000,
        2_000,
        0,
        samples_per_channel,
        sample_rate_hz,
        channels,
        1,
        1_000,
        1_250,
        0,
    )
    payload += struct.pack("<B", 0)
    payload += struct.pack("<B", 0)
    payload += struct.pack("<I", samples_per_channel)
    payload += samples
    return bytes(payload)


def _clock_holdover_section() -> bytes:
    section = bytearray()
    # flags: holdover_active | lt_valid | temp_model_valid | temp_comp_applied
    section += struct.pack("<B", 0x01 | 0x02 | 0x04 | 0x08)
    section += struct.pack("<I", 34_000)  # holdover_age_ms
    section += struct.pack("<I", 12_300)  # predicted_error_ns
    section += struct.pack("<f", -3.5)    # lt_ppm
    section += struct.pack("<f", 0.25)    # lt_ppm_sigma
    section += struct.pack("<f", -0.8)    # temp_slope_ppm_per_c
    section += struct.pack("<f", 0.05)    # temp_resid_rms_ppm
    return bytes(section)


def _binary_frame_mmb3(
    *,
    sample_rate_hz: int = 16_000,
    channels: int = 1,
    samples_per_channel: int = 2,
    include_aux_sensors: bool = False,
    include_clock_holdover: bool = False,
    include_unknown_section: bool = False,
) -> bytes:
    samples = b"".join(
        struct.pack("<h", 32767 if sample_index % 2 else 0)
        for sample_index in range(samples_per_channel * channels)
    )
    payload = bytearray()
    section_flags = 0x0004 | (0x0008 if include_aux_sensors else 0)
    if include_clock_holdover:
        section_flags |= 0x0010
    if include_unknown_section:
        section_flags |= 0x0020
    payload += struct.pack(
        "<QQQQIBBQQQBIH",
        1_000,
        2_000,
        0,
        samples_per_channel,
        sample_rate_hz,
        channels,
        3,  # synthetic audio source
        1,
        1_000,
        1_250,
        0,
        samples_per_channel,
        section_flags,
    )
    transport_health = bytearray()
    transport_health += struct.pack("<HHHHHHH", 1, 16, 2, 40, 10, 11, 12)
    transport_health += struct.pack("<bII", -55, 0, 0x12345678)
    payload += struct.pack("<H", len(transport_health))
    payload += transport_health
    if include_aux_sensors:
        aux_sensors = bytearray()
        aux_sensors += struct.pack("<B", 1)  # stream count
        aux_sensors += struct.pack("<BBH", 0, 3, 2)
        aux_sensors += struct.pack("<QI", 123_456_789, 4_000)
        aux_sensors += struct.pack("<ffffff", 1.0, 2.0, 3.0, 1.5, 2.5, 3.5)
        payload += struct.pack("<H", len(aux_sensors))
        payload += aux_sensors
    # Sections are emitted in ascending bit order to match the firmware loop.
    if include_clock_holdover:
        holdover = _clock_holdover_section()
        payload += struct.pack("<H", len(holdover))
        payload += holdover
    if include_unknown_section:
        unknown = b"\xde\xad\xbe\xef"  # opaque future payload
        payload += struct.pack("<H", len(unknown))
        payload += unknown
    payload += samples
    return bytes(payload)


def _binary_ingest_payload(
    *,
    position_source: str,
    geo: GeoPoint,
    sample_rate_hz: int = 16_000,
    channels: int = 1,
    samples_per_channel: int = 2,
) -> bytes:
    payload = bytearray()
    payload += b"MMB2"
    payload += struct.pack("<BBH", 2, 0, 1)
    payload += _binary_node(position_source=position_source, geo=geo)
    payload += _binary_frame(
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        samples_per_channel=samples_per_channel,
    )
    return bytes(payload)


def _binary_ingest_payload_no_geo(*, position_source: str = "fallback_static") -> bytes:
    payload = bytearray()
    payload += b"MMB2"
    payload += struct.pack("<BBH", 2, 0, 1)
    payload += _binary_node_no_geo(position_source=position_source)
    payload += _binary_frame()
    return bytes(payload)


def _binary_ingest_payload_mmb3(
    *,
    position_source: str,
    geo: GeoPoint,
    sample_rate_hz: int = 16_000,
    channels: int = 1,
    samples_per_channel: int = 2,
    include_aux_sensors: bool = False,
    include_clock_holdover: bool = False,
    include_unknown_section: bool = False,
) -> bytes:
    payload = bytearray()
    payload += b"MMB3"
    payload += struct.pack("<BBH", 3, 0, 1)
    payload += _binary_node(position_source=position_source, geo=geo)
    payload += _binary_frame_mmb3(
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        samples_per_channel=samples_per_channel,
        include_aux_sensors=include_aux_sensors,
        include_clock_holdover=include_clock_holdover,
        include_unknown_section=include_unknown_section,
    )
    return bytes(payload)


def _make_processor(coordinate_frame: LocalCoordinateFrame) -> IngestProcessor:
    settings = Settings()
    return IngestProcessor(
        localization_config=settings.localization_config(),
        fusion_config=settings.fusion_config(),
        registry=SimpleNamespace(),
        buffer=SimpleNamespace(),
        storage=SimpleNamespace(),
        coordinate_frame=coordinate_frame,
        preprocessor_factory=SimpleNamespace(),
        node_position_kalman_q=0.25,
        node_position_kalman_r=25.0,
        node_position_kalman_init_p=100.0,
    )


def _node_spec(*, node_id: str, geo: GeoPoint, position_source: str) -> NodeSpec:
    # Mobile so the position Kalman uses its reactive process noise and no jump gate;
    # stationary nodes now strongly smooth and reject jumps (see test_localization_health_fixes).
    return NodeSpec(
        id=node_id,
        node_type=NodeType.POINT,
        position_geo=geo,
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
        mobility="mobile",
        metadata={"gps": {"signal": "fix_3d", "position_source": position_source}},
    )


def test_parse_binary_ingest_payload_mmb2_uses_geo_only_node_position() -> None:
    geo = GeoPoint(lat=44.987, lon=-93.258, alt_m=281.5)

    payload = parse_binary_ingest_payload(
        _binary_ingest_payload(position_source="gps_nmea_uart", geo=geo)
    )

    assert payload.node.position_m is None
    assert payload.node.position_geo is not None
    assert payload.node.position_geo.lat == pytest.approx(geo.lat)
    assert payload.node.position_geo.lon == pytest.approx(geo.lon)
    assert payload.node.metadata["gps"]["position_source"] == "gps_nmea_uart"


def test_parse_binary_ingest_payload_mmb3_parses_transport_health_section() -> None:
    geo = GeoPoint(lat=44.987, lon=-93.258, alt_m=281.5)

    payload = parse_binary_ingest_payload(
        _binary_ingest_payload_mmb3(position_source="gps_nmea_uart", geo=geo)
    )

    frame = payload.buffered_frames[0].frame
    assert frame.sample_rate_hz == 16_000
    assert frame.timing_diagnostics["audio_source_type"] == "synthetic"
    assert frame.timing_diagnostics["transport_health"] == {
        "ring_frames_high_water": 1,
        "ring_frames_capacity": 16,
        "queue_slots_high_water": 2,
        "queue_slots_capacity": 40,
        "publish_latency_last_ms": 10,
        "publish_latency_ewma_ms": 11,
        "publish_latency_max_ms": 12,
        "wifi_rssi_dbm": -55,
        "heap_free_bytes": 0,
        "boot_id": 0x12345678,
    }


def test_parse_binary_ingest_payload_mmb3_parses_aux_sensor_section() -> None:
    geo = GeoPoint(lat=44.987, lon=-93.258, alt_m=281.5)

    payload = parse_binary_ingest_payload(
        _binary_ingest_payload_mmb3(
            position_source="gps_nmea_uart",
            geo=geo,
            include_aux_sensors=True,
        )
    )

    aux_sensors = payload.buffered_frames[0].frame.timing_diagnostics["aux_sensors"]
    assert aux_sensors == [
        {
            "sensor_type": 0,
            "values_per_sample": 3,
            "sample_count": 2,
            "first_sample_utc_ns": 123_456_789,
            "sample_interval_us": 4_000,
            "values": pytest.approx([1.0, 2.0, 3.0, 1.5, 2.5, 3.5]),
        }
    ]


def test_parse_binary_ingest_payload_mmb3_parses_clock_holdover_section() -> None:
    geo = GeoPoint(lat=44.987, lon=-93.258, alt_m=281.5)

    payload = parse_binary_ingest_payload(
        _binary_ingest_payload_mmb3(
            position_source="gps_nmea_uart",
            geo=geo,
            include_clock_holdover=True,
        )
    )

    clock_holdover = payload.buffered_frames[0].frame.timing_diagnostics["clock_holdover"]
    assert clock_holdover["holdover_active"] is True
    assert clock_holdover["lt_valid"] is True
    assert clock_holdover["temp_model_valid"] is True
    assert clock_holdover["temp_comp_applied"] is True
    assert clock_holdover["holdover_age_ms"] == 34_000
    assert clock_holdover["predicted_error_ns"] == 12_300
    assert clock_holdover["lt_ppm"] == pytest.approx(-3.5)
    assert clock_holdover["lt_ppm_sigma"] == pytest.approx(0.25)
    assert clock_holdover["temp_slope_ppm_per_c"] == pytest.approx(-0.8)
    assert clock_holdover["temp_resid_rms_ppm"] == pytest.approx(0.05)


def test_parse_binary_ingest_payload_mmb3_accepts_holdover_gate_without_section() -> None:
    geo = GeoPoint(lat=44.987, lon=-93.258, alt_m=281.5)

    payload = parse_binary_ingest_payload(
        _binary_ingest_payload_mmb3(
            position_source="gps_nmea_uart",
            geo=geo,
            include_clock_holdover=False,
        )
    )

    timing = payload.buffered_frames[0].frame.timing_diagnostics
    assert "clock_holdover" not in timing
    assert timing["transport_health"]["queue_slots_capacity"] == 40


def test_parse_binary_ingest_payload_mmb3_skips_unknown_section_without_raising() -> None:
    geo = GeoPoint(lat=44.987, lon=-93.258, alt_m=281.5)

    # An unknown future section (bit 0x0020) must be consumed and ignored, not
    # raise — this is what lets old servers keep ingesting from newer firmware.
    payload = parse_binary_ingest_payload(
        _binary_ingest_payload_mmb3(
            position_source="gps_nmea_uart",
            geo=geo,
            include_clock_holdover=True,
            include_unknown_section=True,
        )
    )

    frame = payload.buffered_frames[0].frame
    # Known sections around the unknown one still decode correctly.
    assert "clock_holdover" in frame.timing_diagnostics
    assert "transport_health" in frame.timing_diagnostics
    assert frame.samples_per_channel == 2


@pytest.mark.parametrize("version", [2, 3])
@pytest.mark.parametrize(
    ("sample_rate_hz", "channels", "samples_per_channel"),
    [
        (32_000, 4, 512),
        (48_000, 4, 512),
        (76_800, 1, 768),
        (96_000, 1, 960),
    ],
)
def test_parse_binary_ingest_payload_accepts_high_rate_matrix(
    version: int,
    sample_rate_hz: int,
    channels: int,
    samples_per_channel: int,
) -> None:
    geo = GeoPoint(lat=44.987, lon=-93.258, alt_m=281.5)
    payload_bytes = (
        _binary_ingest_payload_mmb3(
            position_source="gps_nmea_uart",
            geo=geo,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            samples_per_channel=samples_per_channel,
        )
        if version == 3
        else _binary_ingest_payload(
            position_source="gps_nmea_uart",
            geo=geo,
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            samples_per_channel=samples_per_channel,
        )
    )

    payload = parse_binary_ingest_payload(payload_bytes)

    frame = payload.buffered_frames[0].frame
    decoded_audio = payload.buffered_frames[0].decoded_audio
    assert frame.sample_rate_hz == sample_rate_hz
    assert frame.channels == channels
    assert frame.samples_per_channel == samples_per_channel
    assert frame.end_sample_index == samples_per_channel
    assert decoded_audio.shape == (channels, samples_per_channel)


def test_normalize_node_spec_keeps_geo_consistent_with_smoothed_local_position() -> None:
    coordinate_frame = LocalCoordinateFrame(
        origin=GeoPoint(lat=44.980, lon=-93.260, alt_m=250.0),
        mode="flat",
    )
    processor = _make_processor(coordinate_frame)

    live_initial = _node_spec(
        node_id="gps-node-1",
        geo=coordinate_frame.local_to_geo((0.0, 0.0, 0.0)),
        position_source="gps_nmea_uart",
    )
    normalized_initial, normalized_initial_geo = processor._normalize_node_spec(live_initial)

    assert normalized_initial.position_m == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    assert normalized_initial.position_geo is not None
    assert normalized_initial_geo.lat == pytest.approx(normalized_initial.position_geo.lat)

    live_update = _node_spec(
        node_id="gps-node-1",
        geo=coordinate_frame.local_to_geo((20.0, 0.0, 0.0)),
        position_source="gps_nmea_uart",
    )
    normalized_live, normalized_live_geo = processor._normalize_node_spec(live_update)

    assert 0.0 < normalized_live.position_m[0] < 20.0
    expected_live_geo = coordinate_frame.local_to_geo(normalized_live.position_m)
    assert normalized_live.position_geo is not None
    assert normalized_live.position_geo.lat == pytest.approx(expected_live_geo.lat, abs=1e-9)
    assert normalized_live.position_geo.lon == pytest.approx(expected_live_geo.lon, abs=1e-9)
    assert normalized_live_geo.lat == pytest.approx(expected_live_geo.lat, abs=1e-9)
    assert normalized_live_geo.lon == pytest.approx(expected_live_geo.lon, abs=1e-9)

    fallback_raw_geo = coordinate_frame.local_to_geo((100.0, 0.0, 0.0))
    fallback_update = _node_spec(
        node_id="gps-node-1",
        geo=fallback_raw_geo,
        position_source="gps_fallback",
    )
    normalized_fallback, normalized_fallback_geo = processor._normalize_node_spec(fallback_update)

    assert normalized_fallback.position_m == pytest.approx(normalized_live.position_m, abs=1e-6)
    expected_fallback_geo = coordinate_frame.local_to_geo(normalized_fallback.position_m)
    assert normalized_fallback.position_geo is not None
    assert normalized_fallback.position_geo.lat == pytest.approx(expected_fallback_geo.lat, abs=1e-9)
    assert normalized_fallback.position_geo.lon == pytest.approx(expected_fallback_geo.lon, abs=1e-9)
    assert normalized_fallback_geo.lat == pytest.approx(expected_fallback_geo.lat, abs=1e-9)
    assert normalized_fallback_geo.lon == pytest.approx(expected_fallback_geo.lon, abs=1e-9)
    assert normalized_fallback_geo.lon != pytest.approx(fallback_raw_geo.lon, abs=1e-9)


def test_normalize_node_spec_derives_geo_from_local_only_position() -> None:
    coordinate_frame = LocalCoordinateFrame(
        origin=GeoPoint(lat=44.980, lon=-93.260, alt_m=250.0),
        mode="flat",
    )
    processor = _make_processor(coordinate_frame)

    local_position = (12.5, -3.0, 4.0)
    local_only_node = NodeSpec(
        id="local-only-node-1",
        node_type=NodeType.POINT,
        position_m=local_position,
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
        metadata={},
    )

    normalized_node, normalized_geo = processor._normalize_node_spec(local_only_node)

    expected_geo = coordinate_frame.local_to_geo(local_position)
    assert normalized_node.position_m == pytest.approx(local_position, abs=1e-9)
    assert normalized_node.position_geo is not None
    assert normalized_node.position_geo.lat == pytest.approx(expected_geo.lat, abs=1e-9)
    assert normalized_node.position_geo.lon == pytest.approx(expected_geo.lon, abs=1e-9)
    assert normalized_node.position_geo.alt_m == pytest.approx(expected_geo.alt_m, abs=1e-9)
    assert normalized_geo.lat == pytest.approx(expected_geo.lat, abs=1e-9)
    assert normalized_geo.lon == pytest.approx(expected_geo.lon, abs=1e-9)
    assert normalized_geo.alt_m == pytest.approx(expected_geo.alt_m, abs=1e-9)


def test_legacy_node_without_position_rejected_when_fallback_disabled() -> None:
    """Strict behavior is preserved: no position + no fallback -> rejected."""
    with pytest.raises(ValidationError):
        parse_binary_ingest_payload(
            _binary_ingest_payload_no_geo(), fallback_position_m=None
        )


def test_legacy_node_without_position_accepted_with_fallback() -> None:
    fallback = (1.0, 2.0, 3.0)
    payload = parse_binary_ingest_payload(
        _binary_ingest_payload_no_geo(), fallback_position_m=fallback
    )

    assert payload.node.position_geo is None
    assert payload.node.position_m == pytest.approx(fallback)
    gps_meta = payload.node.metadata["gps"]
    assert gps_meta["server_position_fallback"] is True
    # Firmware-reported source is preserved rather than clobbered.
    assert gps_meta["position_source"] == "fallback_static"


def test_default_settings_enable_legacy_fallback_position() -> None:
    settings = Settings()
    assert settings.legacy_ingest_fallback_position_m == (0.0, 0.0, 0.0)

    payload = parse_binary_ingest_payload(
        _binary_ingest_payload_no_geo(),
        fallback_position_m=settings.legacy_ingest_fallback_position_m,
    )
    assert payload.node.position_m == pytest.approx((0.0, 0.0, 0.0))


def test_node_with_geo_ignores_fallback_position() -> None:
    """New firmware (sends geo) is unaffected by the legacy fallback."""
    geo = GeoPoint(lat=44.987, lon=-93.258, alt_m=281.5)
    payload = parse_binary_ingest_payload(
        _binary_ingest_payload(position_source="gps_nmea_uart", geo=geo),
        fallback_position_m=(1.0, 2.0, 3.0),
    )

    assert payload.node.position_m is None
    assert payload.node.position_geo is not None
    assert "server_position_fallback" not in payload.node.metadata.get("gps", {})
