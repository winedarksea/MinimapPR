from __future__ import annotations

import numpy as np

from minimappr.core.iamf_pipeline import LoudnessMeasurement
from minimappr.core.iamf_writer import (
    _OBJECT_POSITION_PARAMETER_ID,
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


def _read_signed_bits(raw: bytes, bit_offset: int, n_bits: int) -> tuple[int, int]:
    value = 0
    for _ in range(n_bits):
        byte = raw[bit_offset // 8]
        bit = (byte >> (7 - (bit_offset % 8))) & 1
        value = (value << 1) | bit
        bit_offset += 1
    sign_bit = 1 << (n_bits - 1)
    if value & sign_bit:
        value -= 1 << n_bits
    return value, bit_offset


def _read_unsigned_bits(raw: bytes, bit_offset: int, n_bits: int) -> tuple[int, int]:
    value = 0
    for _ in range(n_bits):
        byte = raw[bit_offset // 8]
        bit = (byte >> (7 - (bit_offset % 8))) & 1
        value = (value << 1) | bit
        bit_offset += 1
    return value, bit_offset


def _decode_single_position_payload(raw: bytes, offset: int) -> tuple[int, tuple[int, ...]]:
    data_size, offset = _read_uleb128(raw, offset)
    data_end = offset + data_size
    animation_type, offset = _read_uleb128(raw, offset)
    bit_base = offset * 8
    if animation_type == 0:
        azimuth, bit_base = _read_signed_bits(raw, bit_base, 9)
        elevation, bit_base = _read_signed_bits(raw, bit_base, 8)
        distance, bit_base = _read_unsigned_bits(raw, bit_base, 3)
        assert bit_base <= data_end * 8
        return animation_type, (azimuth, elevation, distance)
    if animation_type == 1:
        start_azimuth, bit_base = _read_signed_bits(raw, bit_base, 9)
        end_azimuth, bit_base = _read_signed_bits(raw, bit_base, 9)
        start_elevation, bit_base = _read_signed_bits(raw, bit_base, 8)
        end_elevation, bit_base = _read_signed_bits(raw, bit_base, 8)
        start_distance, bit_base = _read_unsigned_bits(raw, bit_base, 3)
        end_distance, bit_base = _read_unsigned_bits(raw, bit_base, 3)
        assert bit_base <= data_end * 8
        return animation_type, (
            start_azimuth,
            end_azimuth,
            start_elevation,
            end_elevation,
            start_distance,
            end_distance,
        )
    raise AssertionError(f"unexpected animation type {animation_type}")


def test_ia_sequence_header_payload_starts_with_iamf_magic() -> None:
    payload = _obu_payload(_ia_sequence_header())

    assert payload[:4] == b"iamf"
    assert payload[4:] == b"\x01\x01"


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
    mix_presentations = [payload for obu_type, payload in obus if obu_type == 2]
    parameter_blocks = [payload for obu_type, payload in obus if obu_type == 3]

    object_payload = audio_elements[1]
    audio_element_id, offset = _read_uleb128(object_payload, 0)
    assert audio_element_id == 2
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
    assert parameter_id == _OBJECT_POSITION_PARAMETER_ID
    position_data_offset = offset
    position_data_size, offset = _read_uleb128(parameter_blocks[0], offset)
    assert position_data_size > 0
    animation_type, linear_values = _decode_single_position_payload(parameter_blocks[0], position_data_offset)
    assert animation_type == 1  # LINEAR
    assert linear_values == (10, 20, 5, 10, 2, 2)

    mix_payload = mix_presentations[0]
    # Skip mix_presentation_id, count_label, num_sub_mixes, num_audio_elements.
    _, offset = _read_uleb128(mix_payload, 0)
    _, offset = _read_uleb128(mix_payload, offset)
    _, offset = _read_uleb128(mix_payload, offset)
    _, offset = _read_uleb128(mix_payload, offset)
    # Skip bed submix element: audio_element_id, rendering config, mix gain definition.
    _, offset = _read_uleb128(mix_payload, offset)
    offset += 1
    rendering_size, offset = _read_uleb128(mix_payload, offset)
    offset += rendering_size
    for _ in range(5):
        _, offset = _read_uleb128(mix_payload, offset)
    offset += 2
    # Object submix element should declare the single-position parameter.
    object_mix_audio_element_id, offset = _read_uleb128(mix_payload, offset)
    assert object_mix_audio_element_id == 2
    offset += 1
    rendering_size, offset = _read_uleb128(mix_payload, offset)
    rendering_end = offset + rendering_size
    num_parameters, offset = _read_uleb128(mix_payload, offset)
    assert num_parameters == 1
    param_type, offset = _read_uleb128(mix_payload, offset)
    assert param_type == 3
    position_param_id, offset = _read_uleb128(mix_payload, offset)
    assert position_param_id == _OBJECT_POSITION_PARAMETER_ID
    _, offset = _read_uleb128(mix_payload, offset)  # parameter_rate
    offset += 1  # param_definition_mode/reserved
    _, offset = _read_uleb128(mix_payload, offset)  # duration
    _, offset = _read_uleb128(mix_payload, offset)  # constant_subblock_duration
    default_triplet = mix_payload[offset:rendering_end]
    azimuth, bit_offset = _read_signed_bits(default_triplet, 0, 9)
    elevation, bit_offset = _read_signed_bits(default_triplet, bit_offset, 8)
    distance, _ = _read_unsigned_bits(default_triplet, bit_offset, 3)
    assert (azimuth, elevation, distance) == (10, 5, 2)

    second_parameter_id, offset = _read_uleb128(parameter_blocks[1], 0)
    assert second_parameter_id == _OBJECT_POSITION_PARAMETER_ID
    animation_type, step_values = _decode_single_position_payload(parameter_blocks[1], offset)
    assert animation_type == 0  # STEP
    assert step_values == (20, 10, 2)
