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
