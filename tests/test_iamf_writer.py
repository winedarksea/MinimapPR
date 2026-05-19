from __future__ import annotations

import numpy as np

from minimappr.core.iamf_pipeline import LoudnessMeasurement
from minimappr.core.iamf_writer import (
    _codec_config,
    _ia_sequence_header,
    write_iamf,
)


def _read_uleb128(raw: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = raw[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte & 0x80 == 0:
            return value, offset
        shift += 7


def _obu_payload(raw: bytes) -> bytes:
    size, payload_offset = _read_uleb128(raw, 1)
    return raw[payload_offset : payload_offset + size]


def _iter_obus(raw: bytes):
    offset = 0
    while offset < len(raw):
        obu_type = raw[offset] >> 3
        size, payload_offset = _read_uleb128(raw, offset + 1)
        end = payload_offset + size
        yield obu_type, raw[payload_offset:end]
        offset = end


def test_ia_sequence_header_payload_starts_with_iamf_magic() -> None:
    payload = _obu_payload(_ia_sequence_header())

    assert payload[:4] == b"iamf"
    assert payload[4:] == b"\x00\x00"


def test_codec_config_places_codec_id_before_frame_size() -> None:
    payload = _obu_payload(_codec_config(0, 48_000, 512))

    assert payload[0] == 0
    assert payload[1:5] == b"ipcm"


def test_write_iamf_file_begins_with_parseable_ia_sequence_header() -> None:
    bed = np.zeros((4, 512), dtype=np.float32)
    loudness = LoudnessMeasurement(integrated_lufs=-20.0, true_peak_dbfs=-3.0)

    encoded = write_iamf(bed, [], [], loudness, [])
    payload = _obu_payload(encoded)

    assert payload[:4] == b"iamf"


def test_write_iamf_object_track_uses_object_element_and_position_blocks() -> None:
    bed = np.zeros((4, 1024), dtype=np.float32)
    obj = np.zeros(1024, dtype=np.float32)
    loudness = LoudnessMeasurement(integrated_lufs=-20.0, true_peak_dbfs=-3.0)
    positions = [
        {
            0: {
                "azimuth_deg": 10.0,
                "elevation_deg": 5.0,
                "distance_norm": 0.25,
                "end_azimuth_deg": 20.0,
                "end_elevation_deg": 10.0,
                "end_distance_norm": 0.35,
            }
        },
        {0: {"azimuth_deg": 20.0, "elevation_deg": 10.0, "distance_norm": 0.35}},
    ]

    encoded = write_iamf(bed, [obj], positions, loudness, [loudness])
    obus = list(_iter_obus(encoded))
    audio_elements = [payload for obu_type, payload in obus if obu_type == 1]
    parameter_blocks = [payload for obu_type, payload in obus if obu_type == 4]

    object_payload = audio_elements[1]
    audio_element_id, offset = _read_uleb128(object_payload, 0)
    assert audio_element_id == 1
    assert object_payload[offset] >> 5 == 2  # OBJECT_BASED

    # Skip element type, codec_config_id, substream count/id, and num_parameters.
    offset += 1
    _, offset = _read_uleb128(object_payload, offset)
    _, offset = _read_uleb128(object_payload, offset)
    _, offset = _read_uleb128(object_payload, offset)
    _, offset = _read_uleb128(object_payload, offset)
    objects_config_size, offset = _read_uleb128(object_payload, offset)
    assert objects_config_size == 1
    assert object_payload[offset] == 1

    assert parameter_blocks
    parameter_id, offset = _read_uleb128(parameter_blocks[0], 0)
    assert parameter_id == 1
    position_data_size, offset = _read_uleb128(parameter_blocks[0], offset)
    assert position_data_size > 0
    animation_type, _ = _read_uleb128(parameter_blocks[0], offset)
    assert animation_type == 1  # LINEAR
