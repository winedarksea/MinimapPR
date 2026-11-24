"""Verifier logic of scripts/bench/mmb3_capacity_sink.py (Pico capacity bench).

Each injected fault must flip the verdict to FAIL; a clean stream must PASS.
Payloads follow the firmware MMB3 layout (see tests/test_ingest_binary_geo.py),
generalised to multi-frame batches with the synthetic sample pattern.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pytest

_SINK_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bench" / "mmb3_capacity_sink.py"
_spec = importlib.util.spec_from_file_location("mmb3_capacity_sink", _SINK_PATH)
sink_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sink_module  # dataclasses resolve annotations via sys.modules
_spec.loader.exec_module(sink_module)

SAMPLE_RATE_HZ = 48_000
CHANNELS = 5
FRAME_SAMPLES = 512
BOOT_ID = 0xB007


def _string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<B", len(encoded)) + encoded


def _node(channels: int) -> bytes:
    payload = bytearray(_string("sirith-tetra-test"))
    payload += struct.pack("<BB", 4, 1)  # node type planar, has_geo
    payload += struct.pack("<fff", 44.98, -93.25, 0.0)
    payload += struct.pack("<B", channels)
    for channel in range(channels):
        payload += struct.pack("<fff", 0.03 * channel, 0.0, 0.0)
    payload += struct.pack("<B", 1) + _string("audio")
    payload += _string("sirith_tetra_pico2w_synthetic") + _string("dev")
    payload += _string("missing") + _string("fallback_static")
    payload += struct.pack("<I", 7)  # boot count
    return bytes(payload)


def _frame(sequence: int, start_sample_index: int, *, samples: np.ndarray | None = None) -> bytes:
    if samples is None:
        samples = sink_module.expected_synthetic_block(start_sample_index, FRAME_SAMPLES, CHANNELS)
    start_ns = 1_700_000_000_000_000_000 + start_sample_index * 1_000_000_000 // SAMPLE_RATE_HZ
    end_ns = start_ns + FRAME_SAMPLES * 1_000_000_000 // SAMPLE_RATE_HZ
    payload = bytearray(
        struct.pack(
            "<QQQQIBBQQQBIH",
            start_ns,
            end_ns,
            start_sample_index,
            start_sample_index + FRAME_SAMPLES,
            SAMPLE_RATE_HZ,
            CHANNELS,
            3,  # synthetic source
            sequence,
            start_ns,
            start_ns,
            0,
            FRAME_SAMPLES,
            0x0004,  # transport health section only
        )
    )
    health = struct.pack("<HHHHHHH", 1, 16, 2, 40, 10, 11, 12) + struct.pack("<bII", -55, 0, BOOT_ID)
    payload += struct.pack("<H", len(health)) + health
    payload += samples.astype("<i2").tobytes()
    return bytes(payload)


def _request(frames: list[bytes]) -> bytes:
    return b"MMB3" + struct.pack("<BBH", 3, 0, len(frames)) + _node(CHANNELS) + b"".join(frames)


def _stream(sink, frame_specs: list[tuple[int, int]], batch: int = 4, corrupt_sequence: int | None = None) -> None:
    frames = []
    for sequence, start in frame_specs:
        samples = None
        if sequence == corrupt_sequence:
            samples = sink_module.expected_synthetic_block(start, FRAME_SAMPLES, CHANNELS).copy()
            samples[100, 2] ^= 0x0001
        frames.append(_frame(sequence, start, samples=samples))
    for offset in range(0, len(frames), batch):
        sink.verify_payload(time.monotonic(), _request(frames[offset : offset + batch]))


def _clean_specs(count: int) -> list[tuple[int, int]]:
    return [(sequence, sequence * FRAME_SAMPLES) for sequence in range(count)]


def test_clean_stream_passes():
    sink = sink_module.CapacitySink()
    _stream(sink, _clean_specs(40))
    summary = sink.summary()
    assert summary["verdict"] == "PASS", summary["failures"]
    assert summary["frames"] == 40
    assert summary["channels"] == [CHANNELS]


def test_sequence_gap_fails():
    sink = sink_module.CapacitySink()
    specs = [spec for spec in _clean_specs(40) if spec[0] != 17]  # frame 17 lost on the node
    _stream(sink, specs)
    summary = sink.summary()
    assert summary["verdict"] == "FAIL"
    assert summary["sequence_gaps"] == 1 and summary["frames_missing"] == 1
    assert summary["sample_index_gaps"] == 1


def test_sample_index_gap_without_sequence_gap_fails():
    sink = sink_module.CapacitySink()
    specs = _clean_specs(40)
    # Capture-side drop: sequence stays contiguous but samples jump one block.
    specs = [(seq, start + (FRAME_SAMPLES if seq >= 20 else 0)) for seq, start in specs]
    _stream(sink, specs)
    summary = sink.summary()
    assert summary["verdict"] == "FAIL"
    assert summary["sequence_gaps"] == 0
    assert summary["sample_index_gaps"] == 1 and summary["samples_missing"] == FRAME_SAMPLES


def test_corrupted_sample_fails():
    sink = sink_module.CapacitySink()
    _stream(sink, _clean_specs(40), corrupt_sequence=9)
    summary = sink.summary()
    assert summary["verdict"] == "FAIL"
    assert summary["bit_errors"] == 1 and summary["frames_with_bit_errors"] == 1


def test_resent_batch_counts_as_duplicate_not_gap():
    sink = sink_module.CapacitySink()
    specs = _clean_specs(12)
    _stream(sink, specs[:8])
    _stream(sink, specs[4:])  # node re-sends a batch the sink already accepted
    summary = sink.summary()
    assert summary["verdict"] == "PASS", summary["failures"]
    assert summary["duplicate_frames"] == 4


@pytest.mark.parametrize("start_sample_index", [0, 2**32 - 100])
def test_expected_pattern_matches_firmware_formula(start_sample_index):
    block = sink_module.expected_synthetic_block(start_sample_index, 300, 3)
    for sample in (0, 99, 299):
        for channel in range(3):
            index = (start_sample_index + sample) & 0xFFFFFFFF
            word = ((index * 257) & 0xFFFFFFFF) ^ (channel * 0x1F1F) ^ 0x5A5A
            assert block[sample, channel] == np.int16(np.uint16(word & 0xFFFF).view(np.int16))
