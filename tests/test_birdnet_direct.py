"""BirdNET in-process (direct) TFLite inference.

The birdnet package routes every prediction through a multiprocess pipeline
whose drain barrier polls on a 1.0 s timeout, so each `run_arrays()` call costs
~1.0 s regardless of payload. `_DirectBirdNET` drives the same TFLite graph in
process instead — measured 48x on a 3 s clip and 6.2x on 30 s, with scores
verified identical to the session path to 2.8e-07 across all 6522 species.

These tests use a fake interpreter so the segmentation, top-k, threshold and
species-filtering logic is verified without the ~125 MB model.
"""

from __future__ import annotations

import numpy as np
import pytest

from minimappr.classifiers.birdnet import _DirectBirdNET


SEGMENT_SAMPLES = 144_000  # 3 s @ 48 kHz, the model's fixed input width
SPECIES = [
    "Genus a_Alpha Bird",
    "Genus b_Beta Bird",
    "Genus c_Gamma Bird",
    "Genus d_Delta Bird",
]


class _FakeInterpreter:
    """Returns fixed logits and records every segment it was handed."""

    def __init__(self, logits: np.ndarray) -> None:
        self._logits = np.asarray(logits, dtype=np.float32)
        self.segments: list[np.ndarray] = []
        self._tensor: np.ndarray | None = None

    # --- the tf.lite.Interpreter / LiteRT surface _DirectBirdNET uses ---
    def allocate_tensors(self) -> None:
        pass

    def get_input_details(self):
        return [{"index": 0, "shape": [1, SEGMENT_SAMPLES], "dtype": np.float32}]

    def get_output_details(self):
        return [{"index": 1, "shape": [1, len(SPECIES)], "dtype": np.float32}]

    def set_tensor(self, index: int, value: np.ndarray) -> None:
        # Copy: _DirectBirdNET reuses one buffer across segments.
        self._tensor = np.array(value, copy=True)

    def invoke(self) -> None:
        assert self._tensor is not None
        self.segments.append(self._tensor[0])

    def get_tensor(self, index: int) -> np.ndarray:
        return self._logits[None, :]


def _build(
    monkeypatch: pytest.MonkeyPatch,
    *,
    logits: np.ndarray,
    min_confidence: float = 0.0,
    top_k: int = 5,
    overlap_duration_s: float = 0.0,
    allowed_species: list[str] | None = None,
) -> tuple[_DirectBirdNET, _FakeInterpreter]:
    fake = _FakeInterpreter(logits)
    monkeypatch.setattr(
        "minimappr.classifiers.birdnet._resolve_interpreter_cls",
        lambda: (lambda model_path, num_threads: fake),
    )
    engine = _DirectBirdNET(
        model_path="unused.tflite",
        species=list(SPECIES),
        segment_samples=SEGMENT_SAMPLES,
        overlap_duration_s=overlap_duration_s,
        min_confidence=min_confidence,
        top_k=top_k,
        allowed_species=allowed_species,
        pool_size=1,
        num_threads=1,
    )
    return engine, fake


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def test_single_segment_yields_rows_in_confidence_order(monkeypatch) -> None:
    engine, _ = _build(monkeypatch, logits=np.array([0.0, 2.0, -3.0, 1.0]))
    rows = engine.predict_rows(np.zeros(SEGMENT_SAMPLES, dtype=np.float32))

    assert [row["species_name"] for row in rows] == [
        "Genus b_Beta Bird",
        "Genus d_Delta Bird",
        "Genus a_Alpha Bird",
        "Genus c_Gamma Bird",
    ]
    assert rows[0]["confidence"] == pytest.approx(_sigmoid(2.0))
    assert rows[0]["start_time"] == 0.0
    assert rows[0]["end_time"] == pytest.approx(3.0)


def test_min_confidence_drops_rows_below_the_floor(monkeypatch) -> None:
    # sigmoid(2.0)=0.881, sigmoid(1.0)=0.731, sigmoid(0.0)=0.5, sigmoid(-3)=0.047
    engine, _ = _build(
        monkeypatch, logits=np.array([0.0, 2.0, -3.0, 1.0]), min_confidence=0.6
    )
    rows = engine.predict_rows(np.zeros(SEGMENT_SAMPLES, dtype=np.float32))
    assert [row["species_name"] for row in rows] == [
        "Genus b_Beta Bird",
        "Genus d_Delta Bird",
    ]


def test_top_k_caps_rows_per_segment(monkeypatch) -> None:
    engine, _ = _build(monkeypatch, logits=np.array([0.0, 2.0, -3.0, 1.0]), top_k=2)
    rows = engine.predict_rows(np.zeros(SEGMENT_SAMPLES, dtype=np.float32))
    assert [row["species_name"] for row in rows] == [
        "Genus b_Beta Bird",
        "Genus d_Delta Bird",
    ]


def test_allowed_species_restricts_scoring(monkeypatch) -> None:
    """Filtering happens before top-k, so excluded species cannot crowd it out."""
    engine, _ = _build(
        monkeypatch,
        logits=np.array([0.0, 2.0, -3.0, 1.0]),
        top_k=2,
        allowed_species=["Genus a_Alpha Bird", "Genus c_Gamma Bird"],
    )
    rows = engine.predict_rows(np.zeros(SEGMENT_SAMPLES, dtype=np.float32))
    assert [row["species_name"] for row in rows] == [
        "Genus a_Alpha Bird",
        "Genus c_Gamma Bird",
    ]


def test_long_clip_is_split_into_non_overlapping_segments(monkeypatch) -> None:
    engine, fake = _build(monkeypatch, logits=np.array([0.0, 2.0, -3.0, 1.0]), top_k=1)
    audio = np.ones(SEGMENT_SAMPLES * 3, dtype=np.float32)
    rows = engine.predict_rows(audio)

    assert len(fake.segments) == 3
    assert [row["start_time"] for row in rows] == pytest.approx([0.0, 3.0, 6.0])
    assert [row["end_time"] for row in rows] == pytest.approx([3.0, 6.0, 9.0])


def test_overlap_advances_by_the_shortened_stride(monkeypatch) -> None:
    engine, fake = _build(
        monkeypatch,
        logits=np.array([0.0, 2.0, -3.0, 1.0]),
        top_k=1,
        overlap_duration_s=2.0,  # 3 s segment, 1 s stride
    )
    audio = np.ones(SEGMENT_SAMPLES + 48_000, dtype=np.float32)  # 4 s
    rows = engine.predict_rows(audio)

    assert [row["start_time"] for row in rows] == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert len(fake.segments) == 4


def test_short_tail_segment_is_zero_padded(monkeypatch) -> None:
    """The graph has a fixed 3 s input; a 1 s clip must be padded, not rejected."""
    engine, fake = _build(monkeypatch, logits=np.array([0.0, 2.0, -3.0, 1.0]), top_k=1)
    audio = np.ones(48_000, dtype=np.float32)  # 1 s
    rows = engine.predict_rows(audio)

    assert len(rows) == 1
    assert len(fake.segments) == 1
    segment = fake.segments[0]
    assert segment.shape == (SEGMENT_SAMPLES,)
    assert np.all(segment[:48_000] == 1.0)
    assert np.all(segment[48_000:] == 0.0)


def test_reused_buffer_does_not_leak_between_segments(monkeypatch) -> None:
    """A long segment followed by a short tail must not leave stale samples."""
    engine, fake = _build(monkeypatch, logits=np.array([0.0, 2.0, -3.0, 1.0]), top_k=1)
    audio = np.ones(SEGMENT_SAMPLES + 48_000, dtype=np.float32)  # 4 s -> 3 s + 1 s tail
    engine.predict_rows(audio)

    assert len(fake.segments) == 2
    tail = fake.segments[1]
    assert np.all(tail[:48_000] == 1.0)
    assert np.all(tail[48_000:] == 0.0), "stale samples from the previous segment"


def test_empty_clip_yields_no_rows(monkeypatch) -> None:
    engine, fake = _build(monkeypatch, logits=np.array([0.0, 2.0, -3.0, 1.0]))
    assert engine.predict_rows(np.zeros(0, dtype=np.float32)) == []
    assert fake.segments == []
