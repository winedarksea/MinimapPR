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


def _binary_frame() -> bytes:
    samples = struct.pack("<hh", 0, 32767)
    payload = bytearray()
    payload += struct.pack(
        "<QQQQIBQQQB",
        1_000,
        2_000,
        0,
        2,
        16_000,
        1,
        1,
        1_000,
        1_250,
        0,
    )
    payload += struct.pack("<B", 0)
    payload += struct.pack("<B", 0)
    payload += struct.pack("<I", 2)
    payload += samples
    return bytes(payload)


def _binary_ingest_payload(*, position_source: str, geo: GeoPoint) -> bytes:
    payload = bytearray()
    payload += b"MMB2"
    payload += struct.pack("<BBH", 2, 0, 1)
    payload += _binary_node(position_source=position_source, geo=geo)
    payload += _binary_frame()
    return bytes(payload)


def _binary_ingest_payload_no_geo(*, position_source: str = "fallback_static") -> bytes:
    payload = bytearray()
    payload += b"MMB2"
    payload += struct.pack("<BBH", 2, 0, 1)
    payload += _binary_node_no_geo(position_source=position_source)
    payload += _binary_frame()
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
    return NodeSpec(
        id=node_id,
        node_type=NodeType.POINT,
        position_geo=geo,
        sensor_offsets_m=[(0.0, 0.0, 0.0)],
        capabilities=["audio"],
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