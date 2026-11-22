"""BirdNET+ V3.0 classifier: silence guard, windowing, labels, version selection.

Everything above the "model-backed" section runs without model files (the
ONNX session is a fake). The model-backed tests skip loudly unless
``scripts/fetch_birdnet_v3.py`` has been run.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from minimappr.classifiers import factory
from minimappr.classifiers.base import AudioClassifier
from minimappr.classifiers.birdnet_v3 import BirdNETV3Classifier
from minimappr.classifiers.birdnet_v3_audio import (
    BIRDNET_V3_SAMPLE_RATE_HZ,
    SILENCE_GUARD_REASON_NONFINITE,
    SILENCE_GUARD_REASON_ZERO_VARIANCE,
    BirdNETV3WindowPolicy,
    apply_silence_guard,
    plan_window_spans,
    prepare_birdnet_v3_windows,
)
from minimappr.classifiers.routing import default_routing
from minimappr.config import Settings
from minimappr.models import ClassificationResult

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
V3_MODEL_DIRECTORY = REPOSITORY_ROOT / "data/models/birdnet_v3"
V3_MODEL_PATH = V3_MODEL_DIRECTORY / "BirdNET+_V3.0_Global_12K_FP16_pruned.onnx"
V3_LABELS_PATH = V3_MODEL_DIRECTORY / "BirdNET+_V3.0_Global_12K_Labels.csv"
HOUSE_FINCH_WAV = REPOSITORY_ROOT / "tests/house_finch.wav"

FAKE_LABEL_ROWS = (
    ("BN00001", "Haemorhous mexicanus", "House Finch", "Aves"),
    ("BN00002", "Turdus migratorius", "American Robin", "Aves"),
    ("BN00003", "Canis latrans", "Coyote", "Mammalia"),
    ("BN90067", "Gun", "Gun", "Anthropogenic"),
)
POLICY = BirdNETV3WindowPolicy(min_window_seconds=2.0, max_window_seconds=6.0, window_overlap_seconds=3.0)


def _noise(seconds: float, amplitude: float, seed: int = 1, sample_rate_hz: int = BIRDNET_V3_SAMPLE_RATE_HZ) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(int(seconds * sample_rate_hz)) * amplitude).astype(np.float32)


def _write_fake_labels(directory: Path) -> Path:
    path = directory / "labels.csv"
    lines = ["ix,birdnet_id,scientific_name,common_name,taxon_group,common_name_en"]
    lines += [f"{i},{bid},{sci},{com},{grp},{com}" for i, (bid, sci, com, grp) in enumerate(FAKE_LABEL_ROWS)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class _FakeInput:
    name = "input"


class _FakeSession:
    """Scores class ``target_index`` 0.9 and records every batch it sees."""

    def __init__(self, target_index: int = 0, emit_nan: bool = False) -> None:
        self.batches: list[np.ndarray] = []
        self.target_index = target_index
        self.emit_nan = emit_nan

    def get_inputs(self):
        return [_FakeInput()]

    def run(self, output_names, feeds):
        batch = feeds["input"]
        assert batch.dtype == np.float32 and batch.ndim == 2
        assert np.all(np.isfinite(batch)), "model must never see NaN/inf"
        self.batches.append(batch)
        probabilities = np.full((batch.shape[0], len(FAKE_LABEL_ROWS)), 0.01, dtype=np.float32)
        probabilities[:, self.target_index] = 0.9
        probabilities[:, 1] = 0.3
        if self.emit_nan:
            probabilities[:] = np.nan
        return [probabilities]


def _classifier(tmp_path: Path, session: _FakeSession, **kwargs) -> BirdNETV3Classifier:
    model_path = tmp_path / "fake_fp16_pruned.onnx"
    model_path.write_bytes(b"not a model")
    return BirdNETV3Classifier(
        model_path=model_path,
        labels_path=_write_fake_labels(tmp_path),
        min_confidence=0.1,
        window_policy=POLICY,
        session_factory=lambda path, threads: session,
        **kwargs,
    )


# ── silence guard ────────────────────────────────────────────────────────────


def test_silence_guard_rejects_digital_silence_and_dc_offset():
    assert apply_silence_guard(np.zeros(64_000, np.float32), POLICY).reason == SILENCE_GUARD_REASON_ZERO_VARIANCE
    assert apply_silence_guard(np.full(64_000, 0.25, np.float32), POLICY).reason == SILENCE_GUARD_REASON_ZERO_VARIANCE


def test_silence_guard_rejects_mostly_nan_but_repairs_a_few_nan_samples():
    corrupt = _noise(3.0, 0.01)
    corrupt[::10] = np.nan
    assert apply_silence_guard(corrupt, POLICY).reason == SILENCE_GUARD_REASON_NONFINITE

    mostly_clean = _noise(3.0, 0.01)
    mostly_clean[[5, 500, 5000]] = [np.nan, np.inf, -np.inf]
    verdict = apply_silence_guard(mostly_clean, POLICY)
    assert verdict.accepted and verdict.nonfinite_sample_count == 3
    assert np.all(np.isfinite(verdict.audio))
    # Repaired samples are noise at the floor, not hard zeros.
    assert np.all(verdict.audio[[5, 500, 5000]] != 0.0)


def test_silence_guard_passes_faint_but_valid_signal():
    """A distant bird ~120 dB below full scale must not be gated out (no RMS threshold)."""
    time_axis = np.arange(3 * BIRDNET_V3_SAMPLE_RATE_HZ) / BIRDNET_V3_SAMPLE_RATE_HZ
    faint_call = (1e-6 * np.sin(2 * np.pi * 3_000 * time_axis)).astype(np.float32) + _noise(3.0, 1e-7)
    verdict = apply_silence_guard(faint_call, POLICY)
    assert verdict.accepted
    windows, _ = prepare_birdnet_v3_windows(faint_call, BIRDNET_V3_SAMPLE_RATE_HZ, POLICY)
    assert len(windows) == 1


def test_all_zero_window_inside_long_clip_is_skipped_not_fed():
    clip = _noise(20.0, 0.01)
    clip[: 7 * BIRDNET_V3_SAMPLE_RATE_HZ] = 0.0  # zero-filled packet gap
    windows, verdict = prepare_birdnet_v3_windows(clip, BIRDNET_V3_SAMPLE_RATE_HZ, POLICY)
    assert verdict.accepted
    assert all(start >= 1.0 for start, _, _ in windows)
    assert all(np.std(audio) > 0 for _, _, audio in windows)


# ── length handling ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("seconds", "expected_window_count", "expected_window_seconds"),
    [(1.0, 1, 2.0), (2.0, 1, 2.0), (7.0, 2, 6.0), (12.0, 3, 6.0), (20.0, 6, 6.0)],
)
def test_length_handling(seconds, expected_window_count, expected_window_seconds):
    windows, verdict = prepare_birdnet_v3_windows(_noise(seconds, 0.01, sample_rate_hz=48_000), 48_000, POLICY)
    assert verdict.accepted
    assert len(windows) == expected_window_count
    for start, end, audio in windows:
        assert audio.size == int(expected_window_seconds * BIRDNET_V3_SAMPLE_RATE_HZ)
        assert 0.0 <= start < end <= seconds + 1e-6
    # Windows cover the whole clip, the last one flush to its end.
    assert windows[0][0] == 0.0
    assert windows[-1][1] == pytest.approx(min(seconds, max(seconds, POLICY.min_window_seconds)), abs=1e-3)


def test_short_clip_padding_is_noise_at_the_clip_floor_not_zeros():
    windows, _ = prepare_birdnet_v3_windows(_noise(1.0, 0.01), BIRDNET_V3_SAMPLE_RATE_HZ, POLICY)
    padded = windows[0][2]
    head = padded[: BIRDNET_V3_SAMPLE_RATE_HZ // 4]
    assert np.count_nonzero(head == 0.0) == 0
    assert 0.002 < float(np.std(head)) < 0.05


def test_long_clip_spans_respect_overlap():
    spans = plan_window_spans(20 * BIRDNET_V3_SAMPLE_RATE_HZ, POLICY)
    hop = (POLICY.max_window_seconds - POLICY.window_overlap_seconds) * BIRDNET_V3_SAMPLE_RATE_HZ
    starts = [start for start, _ in spans]
    assert all(b - a <= hop for a, b in zip(starts, starts[1:]))
    assert spans[-1][1] == 20 * BIRDNET_V3_SAMPLE_RATE_HZ


def test_window_policy_rejects_inverted_windows():
    with pytest.raises(ValueError):
        BirdNETV3WindowPolicy(min_window_seconds=8.0, max_window_seconds=6.0)


# ── classifier with a fake session ───────────────────────────────────────────


def test_classifier_batches_windows_and_returns_v24_shaped_result(tmp_path):
    session = _FakeSession(target_index=0)
    result = _classifier(tmp_path, session).classify(_noise(20.0, 0.01, sample_rate_hz=48_000), 48_000)
    assert len(session.batches) == 1 and session.batches[0].shape == (6, 6 * BIRDNET_V3_SAMPLE_RATE_HZ)
    assert result.label == "house finch"
    assert result.confidence == pytest.approx(0.9, abs=1e-6)
    assert result.scores["american robin"] == pytest.approx(0.3, abs=1e-6)
    assert result.features["model"] == "birdnet_v3"
    assert result.features["raw_species"] == "Haemorhous mexicanus_House Finch"
    assert result.features["taxon_group"] == "Aves"
    assert result.features["model_precision"] == "fp16_pruned"


def test_classifier_never_calls_model_on_silence(tmp_path):
    session = _FakeSession()
    result = _classifier(tmp_path, session).classify(np.zeros(96_000, np.float32), BIRDNET_V3_SAMPLE_RATE_HZ)
    assert session.batches == []
    assert result.label == "unknown" and result.features["silence_guard"] == SILENCE_GUARD_REASON_ZERO_VARIANCE


def test_classifier_drops_nan_model_output(tmp_path):
    result = _classifier(tmp_path, _FakeSession(emit_nan=True)).classify(_noise(3.0, 0.01), BIRDNET_V3_SAMPLE_RATE_HZ)
    assert result.label == "unknown" and result.features["silence_guard"] == "nonfinite_model_output"


def test_regional_species_list_restricts_labels(tmp_path):
    species_list = tmp_path / "region.txt"
    species_list.write_text("# backyard\nTurdus migratorius\ncoyote\n", encoding="utf-8")
    classifier = _classifier(tmp_path, _FakeSession(target_index=0), species_list_path=species_list)
    result = classifier.classify(_noise(3.0, 0.01), BIRDNET_V3_SAMPLE_RATE_HZ)
    assert result.label == "american robin"  # house finch (0.9) is out of region
    assert "house finch" not in result.scores


def test_regional_species_list_matching_nothing_is_an_error(tmp_path):
    species_list = tmp_path / "region.txt"
    species_list.write_text("Not a species\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _classifier(tmp_path, _FakeSession(), species_list_path=species_list)


def test_v3_taxonomy_fallback_only_marks_animals_as_wildlife():
    from minimappr.core.classification import _resolve_label_category

    class _UnknownTaxonomy:
        def category_for_label(self, label):
            return "unknown"

    def category(taxon_group: str) -> str:
        result = ClassificationResult(
            label="x", confidence=0.9, features={"model": "birdnet_v3", "taxon_group": taxon_group}
        )
        return _resolve_label_category(result, _UnknownTaxonomy())

    assert category("Aves") == "wildlife"
    assert category("Anthropogenic") == "unknown"


# ── version selection ────────────────────────────────────────────────────────


def _routing_with_birdnet_only():
    routing = default_routing()
    routing.contexts["detection_trigger"].run = ("birdnet",)
    return routing


def test_default_version_builds_v24_classifier(monkeypatch):
    built: list[str] = []

    class _StubV24(AudioClassifier):
        def __init__(self, min_confidence: float = 0.1, **_kwargs) -> None:
            built.append("2.4")

        def classify(self, samples, sample_rate_hz):
            return ClassificationResult(label="unknown", confidence=0.0)

    stub_module = types.ModuleType("minimappr.classifiers.birdnet")
    stub_module.BirdNETClassifier = _StubV24
    monkeypatch.setitem(sys.modules, "minimappr.classifiers.birdnet", stub_module)
    monkeypatch.setattr(factory, "_build_birdnet_v3", lambda spec, settings: pytest.fail("V3 built by default"))

    settings = Settings()
    assert settings.birdnet_model_version == "2.4"
    factory.create_context_classifier(settings, "detection_trigger", routing=_routing_with_birdnet_only())
    assert built == ["2.4"]


def test_version_3_selects_v3_classifier(monkeypatch, tmp_path):
    captured: dict = {}

    class _StubV3(AudioClassifier):
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def classify(self, samples, sample_rate_hz):
            return ClassificationResult(label="house finch", confidence=0.9)

    monkeypatch.setattr("minimappr.classifiers.birdnet_v3.BirdNETV3Classifier", _StubV3)
    settings = Settings(
        birdnet_model_version="3.0",
        birdnet_v3_max_window_seconds=8.0,
        birdnet_v3_species_list_path=str(tmp_path / "region.txt"),
    )
    classifier = factory.create_context_classifier(settings, "detection_trigger", routing=_routing_with_birdnet_only())
    assert isinstance(classifier, _StubV3)
    assert captured["window_policy"].max_window_seconds == 8.0
    assert captured["species_list_path"] == tmp_path / "region.txt"
    assert captured["model_path"] == settings.birdnet_v3_model_path


def test_version_3_with_missing_model_drops_member(tmp_path):
    settings = Settings(birdnet_model_version="3.0", birdnet_v3_model_path=tmp_path / "absent.onnx")
    classifier = factory.create_context_classifier(settings, "detection_trigger", routing=_routing_with_birdnet_only())
    assert not isinstance(classifier, BirdNETV3Classifier)  # fell back to heuristic


def test_unknown_version_is_rejected():
    with pytest.raises(ValueError):
        Settings(birdnet_model_version="2.5")


# ── model-backed (skip loudly when not fetched) ──────────────────────────────

_MODEL_MISSING_REASON = (
    f"BirdNET V3 model not fetched ({V3_MODEL_PATH}); run: python scripts/fetch_birdnet_v3.py"
)
requires_v3_model = pytest.mark.skipif(
    not (V3_MODEL_PATH.is_file() and V3_LABELS_PATH.is_file()), reason=_MODEL_MISSING_REASON
)


@pytest.fixture(scope="module")
def real_v3_classifier():
    if not (V3_MODEL_PATH.is_file() and V3_LABELS_PATH.is_file()):
        pytest.skip(_MODEL_MISSING_REASON)
    classifier = BirdNETV3Classifier(model_path=V3_MODEL_PATH, labels_path=V3_LABELS_PATH, min_confidence=0.1, num_threads=2)
    yield classifier
    classifier.close()


@requires_v3_model
def test_real_model_labels_match_output_width(real_v3_classifier):
    assert len(real_v3_classifier.labels) == 11_725
    probabilities = real_v3_classifier.predict_window_probabilities([_noise(3.0, 0.01)])
    assert probabilities.shape == (1, 11_725)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


@requires_v3_model
def test_real_model_identifies_house_finch_including_faint_copy(real_v3_classifier):
    soundfile = pytest.importorskip("soundfile")
    audio, sample_rate_hz = soundfile.read(HOUSE_FINCH_WAV, dtype="float32")
    loud = real_v3_classifier.classify(audio, sample_rate_hz)
    faint = real_v3_classifier.classify(audio * 1e-3, sample_rate_hz)
    assert loud.label == "house finch" and loud.confidence > 0.5
    assert faint.label == "house finch", "faint (-60 dB) copy must still classify"


@requires_v3_model
@pytest.mark.parametrize("seconds", [1.0, 2.0, 7.0, 12.0, 20.0])
def test_real_model_accepts_all_lengths(real_v3_classifier, seconds):
    result = real_v3_classifier.classify(_noise(seconds, 0.005), BIRDNET_V3_SAMPLE_RATE_HZ)
    assert 0.0 <= result.confidence <= 1.0


# ── review follow-ups ────────────────────────────────────────────────────────


def test_window_policy_defaults_mirror_settings_defaults():
    settings_policy = Settings().birdnet_v3_config()
    default_policy = BirdNETV3WindowPolicy()
    assert default_policy.min_window_seconds == settings_policy.min_window_seconds
    assert default_policy.max_window_seconds == settings_policy.max_window_seconds
    assert default_policy.window_overlap_seconds == settings_policy.window_overlap_seconds


def test_predict_after_close_raises_runtime_error_not_attribute_error(tmp_path):
    classifier = _classifier(tmp_path, _FakeSession())
    classifier.close()
    with pytest.raises(RuntimeError, match="closed"):
        classifier.predict_window_probabilities([_noise(3.0, 0.01)])
