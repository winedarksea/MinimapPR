"""Unit tests for the WAV-only drone-head training pipeline.

No TensorFlow required — the YAMNet embedder is replaced with a fake, and the
per-class group split / dataset discovery / augmentation are pure numpy.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from minimappr.training.augment import (
    AMBIENT_NOISE_PROFILE_NAMES,
    augment_waveform,
    synthesize_ambient_windows,
)
from minimappr.training.embedding_cache import EmbeddingCache
from minimappr.training.wav_dataset import (
    discover_wavs,
    group_key_for,
    load_promoted,
)

SR = 16_000


def _write_wav(path: Path, seconds: float, freq: float = 440.0, rate: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * rate)) / rate
    data = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    pcm = (data * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


# --------------------------------------------------------------------------- #
# discover_wavs
# --------------------------------------------------------------------------- #
def test_discover_wavs_maps_labels_and_skips_unknown(tmp_path, caplog):
    _write_wav(tmp_path / "ambient" / "a.wav", 2.0)
    _write_wav(tmp_path / "drone" / "d.wav", 2.0)
    _write_wav(tmp_path / "junk" / "x.wav", 2.0)  # unknown label dir
    examples = discover_wavs(tmp_path, ["ambient", "drone", "coyote"])
    labels = {e.label for e in examples}
    assert labels == {"ambient", "drone"}
    assert all(e.origin == "wav" for e in examples)


def test_discover_wavs_segments_long_files(tmp_path):
    _write_wav(tmp_path / "coyote" / "long.wav", 20.0)
    examples = discover_wavs(tmp_path, ["coyote"], segment_seconds=4.0, segment_overlap=0.5)
    # 20s file with 4s windows @ 50% overlap -> many segments, all one group.
    assert len(examples) > 5
    assert len({e.group for e in examples}) == 1
    # Offsets are strictly increasing and within the file.
    offs = sorted(e.offset_s for e in examples)
    assert offs[0] == 0.0
    assert all(o < 20.0 for o in offs)


def test_discover_wavs_short_file_single_segment(tmp_path):
    _write_wav(tmp_path / "drone" / "s.wav", 2.0)
    examples = discover_wavs(tmp_path, ["drone"], segment_seconds=4.0)
    assert len(examples) == 1
    assert examples[0].offset_s == 0.0


# --------------------------------------------------------------------------- #
# group_key_for
# --------------------------------------------------------------------------- #
def test_group_key_timestamp_minute_stability():
    a = group_key_for(Path("2026-07-12_10-30-01_0.wav"))
    b = group_key_for(Path("2026-07-12_10-30-59_9.wav"))
    assert a == b  # same minute
    c = group_key_for(Path("2026-07-12_10-31-00_0.wav"))
    assert a != c


def test_group_key_hash_fallback_is_stable():
    p = Path("hf-abc123.wav")
    assert group_key_for(p) == group_key_for(p)
    assert group_key_for(Path("hf-def456.wav")) != group_key_for(p)


# --------------------------------------------------------------------------- #
# load_promoted
# --------------------------------------------------------------------------- #
def _write_promoted(dirpath: Path, det_id: str, label: str, kind: str, *, wav=True, npy=True):
    dirpath.mkdir(parents=True, exist_ok=True)
    manifest = {"training": {"label": label, "example_kind": kind}}
    (dirpath / f"{det_id}.json").write_text(json.dumps(manifest))
    if wav:
        _write_wav(dirpath / f"{det_id}.wav", 1.0)
    if npy:
        np.save(dirpath / f"{det_id}.npy", np.ones(1024, dtype=np.float32))


def test_load_promoted_negative_maps_to_ambient(tmp_path):
    _write_promoted(tmp_path, "det1", "drone", "negative")
    examples = load_promoted(tmp_path, ["ambient", "drone", "coyote"])
    assert len(examples) == 1
    assert examples[0].label == "ambient"


def test_load_promoted_prefers_wav_over_npy(tmp_path):
    _write_promoted(tmp_path, "det2", "drone", "positive", wav=True, npy=True)
    examples = load_promoted(tmp_path, ["ambient", "drone"])
    assert examples[0].origin == "promoted"


def test_load_promoted_npy_fallback(tmp_path):
    _write_promoted(tmp_path, "det3", "coyote", "positive", wav=False, npy=True)
    examples = load_promoted(tmp_path, ["ambient", "drone", "coyote"])
    assert examples[0].origin == "promoted_npy"
    assert examples[0].label == "coyote"


def test_load_promoted_alias_no_drone(tmp_path):
    _write_promoted(tmp_path, "det4", "no_drone", "positive")
    examples = load_promoted(tmp_path, ["ambient", "drone"])
    assert examples[0].label == "ambient"


def test_load_promoted_ignores_tmp(tmp_path):
    _write_promoted(tmp_path, "det5", "drone", "positive")
    (tmp_path / "det6.json.tmp").write_text("{}")
    examples = load_promoted(tmp_path, ["ambient", "drone"])
    assert len(examples) == 1


# --------------------------------------------------------------------------- #
# augment
# --------------------------------------------------------------------------- #
def test_augment_deterministic():
    wave_arr = np.sin(np.linspace(0, 10, SR)).astype(np.float32)
    a = augment_waveform(wave_arr, seed=7)
    b = augment_waveform(wave_arr, seed=7)
    assert np.array_equal(a, b)
    c = augment_waveform(wave_arr, seed=8)
    assert not np.array_equal(a, c)


def test_augment_bounds_and_shape():
    wave_arr = np.sin(np.linspace(0, 10, SR)).astype(np.float32)
    out = augment_waveform(wave_arr, seed=3)
    assert out.shape == wave_arr.shape
    assert np.max(np.abs(out)) <= 1.0


def test_synthesize_ambient_all_profiles_and_determinism():
    a = list(synthesize_ambient_windows(8, 4.0, seed=1))
    b = list(synthesize_ambient_windows(8, 4.0, seed=1))
    assert len(a) == 8
    for (ka, wa), (kb, wb) in zip(a, b):
        assert ka == kb
        assert np.array_equal(wa, wb)
    profiles = {key.split("-")[1] if "-" in key else key for key, _ in a}
    for name in AMBIENT_NOISE_PROFILE_NAMES:
        assert any(name in key for key, _ in a), name
    for _, w in a:
        assert w.shape[0] == int(4.0 * SR)
        assert np.max(np.abs(w)) <= 1.0


# --------------------------------------------------------------------------- #
# EmbeddingCache
# --------------------------------------------------------------------------- #
class _FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed(self, waveform, sample_rate=SR):
        self.calls += 1
        return np.full((2, 1024), float(len(waveform)), dtype=np.float32)


def test_embedding_cache_hit_and_key(tmp_path):
    cache = EmbeddingCache(tmp_path, prep_version="v1")
    fake = _FakeEmbedder()
    key = cache.make_key("hash", offset_ms=0, dur_ms=4000, variant="orig")
    assert key == "hash-v1-0-4000-orig"
    a = cache.get_or_compute(key, lambda: fake.embed(np.zeros(100)))
    b = cache.get_or_compute(key, lambda: fake.embed(np.zeros(100)))
    assert fake.calls == 1  # second call served from disk
    assert np.array_equal(a, b)
    assert cache.hits == 1 and cache.misses == 1


def test_embedding_cache_atomic_no_tmp_left(tmp_path):
    cache = EmbeddingCache(tmp_path)
    cache.get_or_compute("k", lambda: np.ones((1, 1024), np.float32))
    assert not list(tmp_path.glob("*.tmp"))
    assert (tmp_path / "k.npy").exists()


# --------------------------------------------------------------------------- #
# per-class group split
# --------------------------------------------------------------------------- #
def test_static_quantization_reader_yields_every_embedding_frame_in_batches():
    from scripts.train_drone_head import FullDatasetCalibrationDataReader

    frames = np.arange(5 * 1024, dtype=np.float64).reshape(5, 1024)
    reader = FullDatasetCalibrationDataReader("embedding", frames, batch_size=2)

    batches = []
    while (item := reader.get_next()) is not None:
        assert set(item) == {"embedding"}
        batches.append(item["embedding"])

    assert [len(batch) for batch in batches] == [2, 2, 1]
    received = np.concatenate(batches)
    assert received.dtype == np.float32
    assert np.array_equal(received, frames.astype(np.float32))

    reader.rewind()
    assert np.array_equal(reader.get_next()["embedding"], frames[:2].astype(np.float32))


def test_quantized_model_quality_metrics_prioritize_decisions_over_single_prob_delta():
    from scripts.train_drone_head import quantized_model_quality_metrics

    labels = np.array([0, 1, 1])
    float_probs = np.array([[0.95, 0.05], [0.05, 0.95], [0.01, 0.99]], dtype=np.float32)
    # One softmax score moves by 0.45, but neither prediction nor accuracy changes.
    int8_probs = np.array([[0.51, 0.49], [0.49, 0.51], [0.46, 0.54]], dtype=np.float32)

    metrics = quantized_model_quality_metrics(float_probs, int8_probs, labels)

    assert metrics["max_probability_difference"] == pytest.approx(0.45)
    assert metrics["prediction_agreement"] == 1.0
    assert metrics["float_accuracy"] == 1.0
    assert metrics["int8_accuracy"] == 1.0
    assert metrics["accuracy_drop"] == 0.0


def test_empty_quantized_model_quality_metrics_allows_datasets_without_test_frames():
    from scripts.train_drone_head import empty_quantized_model_quality_metrics

    metrics = empty_quantized_model_quality_metrics()

    assert metrics["prediction_agreement"] == 1.0
    assert metrics["accuracy_drop"] == 0.0


def test_per_class_group_split_invariants():
    from scripts.train_drone_head import per_class_group_split

    labels = ["ambient", "drone", "coyote"]
    # Build frames: many groups per class; coyote has 4 groups.
    y = []
    groups = []
    train_only = []
    for li, name in enumerate(labels):
        n_groups = 4 if name == "coyote" else 20
        for g in range(n_groups):
            for _ in range(5):  # 5 frames/group
                y.append(li)
                groups.append(f"{name}-g{g}")
                train_only.append(False)
    # Add augmentation frames sharing a real coyote group + synthetic ambient.
    for _ in range(5):
        y.append(2)
        groups.append("coyote-g0")  # aug shares a real source group
        train_only.append(True)
    for _ in range(5):
        y.append(0)
        groups.append("synth:x")
        train_only.append(True)

    y = np.asarray(y)
    groups = np.asarray(groups, dtype=object)
    train_only = np.asarray(train_only, dtype=bool)

    is_tr, is_va, is_te = per_class_group_split(y, groups, train_only, labels, seed=1)

    # Folds are disjoint; every real frame is kept in exactly one fold.
    assert not (is_tr & is_va).any() and not (is_tr & is_te).any() and not (is_va & is_te).any()
    real = ~train_only
    assert (is_tr | is_va | is_te)[real].all()

    # No group's real frames straddle folds.
    for g in np.unique(groups):
        mask = (groups == g) & real
        if not mask.any():
            continue
        folds = (is_tr[mask].any(), is_va[mask].any(), is_te[mask].any())
        assert sum(folds) == 1, f"group {g} straddles folds"

    # Train-only frames never appear in val/test (kept only in train, or dropped).
    assert not is_va[train_only].any()
    assert not is_te[train_only].any()
    # Synthetic ambient group (pure train-only) stays in train.
    synth = groups == "synth:x"
    assert is_tr[synth].all()


def test_retained_train_frame_counts_include_only_frames_that_contribute_loss():
    from scripts.train_drone_head import retained_train_frame_counts

    labels = ["ambient", "drone", "coyote"]
    # Two real ambient frames and one synthetic ambient frame are retained. The
    # final coyote augmentation was dropped with its held-out source group.
    y = np.asarray([0, 0, 0, 1, 2, 2], dtype=np.int64)
    is_train = np.asarray([True, True, True, True, True, False], dtype=bool)

    assert retained_train_frame_counts(y, is_train, labels) == {
        "ambient": 3,
        "drone": 1,
        "coyote": 1,
    }


# --------------------------------------------------------------------------- #
# auto-balance
# --------------------------------------------------------------------------- #
def test_estimate_real_frame_weights_sums_durations():
    from minimappr.training.wav_dataset import WavExample
    from scripts.train_drone_head import estimate_real_frame_weights

    examples = [
        WavExample(Path("a.wav"), "drone", "g1", 0.0, 4.0, "wav"),
        WavExample(Path("b.wav"), "drone", "g2", 0.0, 4.0, "wav"),
        WavExample(Path("c.wav"), "coyote", "g3", 0.0, 2.0, "wav"),
        WavExample(Path("promoted.wav"), "ambient", "promoted:x", 0.0, 0.0, "promoted"),  # whole-file
    ]
    weights = estimate_real_frame_weights(examples, ["ambient", "drone", "coyote"], segment_seconds=4.0)
    assert weights["drone"] == pytest.approx(8.0)
    assert weights["coyote"] == pytest.approx(2.0)
    # duration_s == 0 (whole-file) falls back to segment_seconds.
    assert weights["ambient"] == pytest.approx(4.0)


def test_auto_balance_boosts_only_classes_below_ratio():
    from scripts.train_drone_head import compute_auto_balance_aug_copies

    weights = {"ambient": 6000.0, "drone": 10000.0, "coyote": 200.0}
    manual = {"coyote": 8}
    result = compute_auto_balance_aug_copies(weights, target_ratio=3.0, manual_per_label=manual, base_aug_copies=1)

    # drone is the reference class and ambient is already within ratio -> not
    # added to the result dict, so callers fall back to base_aug_copies (1).
    assert result.get("drone", 1) == 1
    assert result.get("ambient", 1) == 1
    # coyote: manual floor of 8 copies (200*9=1800) still short of target (10000/3=3333.3)
    # -> bumped up so 200*(1+copies) >= 3333.3  =>  copies >= 15.67 -> 16.
    assert result["coyote"] == 16


def test_auto_balance_never_lowers_manual_value():
    from scripts.train_drone_head import compute_auto_balance_aug_copies

    # Manual copies already exceed what the ratio requires -> left untouched.
    weights = {"ambient": 1000.0, "coyote": 900.0}
    result = compute_auto_balance_aug_copies(
        weights, target_ratio=3.0, manual_per_label={"coyote": 50}, base_aug_copies=1
    )
    assert result["coyote"] == 50


def test_auto_balance_disabled_at_zero_ratio():
    from scripts.train_drone_head import compute_auto_balance_aug_copies

    weights = {"ambient": 6000.0, "coyote": 10.0}
    manual = {"coyote": 8}
    result = compute_auto_balance_aug_copies(weights, target_ratio=0.0, manual_per_label=manual, base_aug_copies=1)
    assert result == manual  # untouched, pass-through


def test_auto_balance_skips_zero_weight_labels():
    from scripts.train_drone_head import compute_auto_balance_aug_copies

    weights = {"ambient": 6000.0, "coyote": 0.0}  # no coyote examples discovered at all
    result = compute_auto_balance_aug_copies(weights, target_ratio=3.0, manual_per_label={}, base_aug_copies=1)
    assert "coyote" not in result or result.get("coyote", 1) == 1  # nothing to multiply


def test_per_class_split_gives_coyote_val_and_test():
    from scripts.train_drone_head import per_class_group_split

    labels = ["ambient", "drone", "coyote"]
    y, groups, train_only = [], [], []
    for li, name in enumerate(labels):
        for g in range(6):
            y.append(li)
            groups.append(f"{name}-{g}")
            train_only.append(False)
    y = np.asarray(y)
    groups = np.asarray(groups, dtype=object)
    train_only = np.asarray(train_only, dtype=bool)
    is_tr, is_va, is_te = per_class_group_split(y, groups, train_only, labels, seed=2)
    coy = y == 2
    # With 6 groups the coyote class should land >=1 group in both val and test.
    assert is_va[coy].any()
    assert is_te[coy].any()
