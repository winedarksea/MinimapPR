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
