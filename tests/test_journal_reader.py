from pathlib import Path

import pytest

from minimappr.api.journal_reader import JournalHandleReadError, JournalPayloadHandle


def test_zero_length_payload_skips_mmap_and_missing_file() -> None:
    handle = JournalPayloadHandle(
        journal_epoch=1,
        segment_id="seg-missing",
        stream_key="node__audio_main__abc",
        payload_offset_bytes=0,
        payload_length_bytes=0,
        sample_index_start=None,
        sample_count=None,
        integrity_hash="",
        segment_path=Path("/tmp/this-file-does-not-need-to-exist.bin"),
    )

    assert handle.read_bytes() == b""


def test_non_zero_length_missing_file_raises() -> None:
    handle = JournalPayloadHandle(
        journal_epoch=1,
        segment_id="seg-missing",
        stream_key="node__audio_main__abc",
        payload_offset_bytes=0,
        payload_length_bytes=16,
        sample_index_start=None,
        sample_count=None,
        integrity_hash="",
        segment_path=Path("/tmp/definitely-missing-segment.bin"),
    )

    with pytest.raises(JournalHandleReadError):
        handle.read_bytes()
