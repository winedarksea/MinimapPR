#!/usr/bin/env python3
"""Train the drone-head classifier from WAV audio and export int8 ONNX.

The drone head is a small dense classifier over YAMNet embedding frames
(``[1024]`` float32) that emits an N-class softmax over
``["ambient", "drone", "coyote"]`` (index 0 = negative class). It is chained
after YAMNet at runtime (see ``classifiers/drone_head.py`` and the ``chains``
block of ``data/classifier_routing.json``).

Dataset layout is **WAV-only** — all training audio is embedded via YAMNet at
training time with the *identical* runtime conditioning
(:func:`minimappr.classifiers.yamnet.prepare_waveform_for_yamnet`), so train and
serve see the same input.

    drone_dataset/wav/<label>/**/*.wav      # drop-dir; FPs go to wav/ambient/
    data/training/{id}.json + .wav / .npy   # server-promoted examples

Long recordings are chopped into overlapping segments; scarce classes are
multiplied with deterministic waveform augmentation; the negative class is
hardened with synthetic ambience. Augmentation and synthetic windows are
train-only (val/test see real audio only).

Outputs (``data/models/`` by default):

    drone_head.onnx            statically-quantized int8 QDQ model shipped + loaded
    drone_head.metadata.json   schema v2: labels, negative_label, preprocessing,
                               per-class thresholds/metrics, dataset counts
    drone_head.float.onnx      float32 reference (gitignored)
    drone_head.tflite          int8 TFLite (gitignored)
    eval_report.json           per-class metrics + threshold sweeps (gitignored)

Usage::

    python scripts/train_drone_head.py --wav-dir drone_dataset/wav \\
        --promoted-dir data/training --out-dir data/models \\
        --cache-dir drone_dataset/.embed_cache --labels ambient,drone,coyote \\
        --epochs 40 --aug-copies 1 --aug-copies-per-label coyote=8 \\
        --auto-balance-ratio 3.0 --synth-ambient-count 1000

``--auto-balance-ratio`` (default 3.0) automatically raises aug-copy counts,
on top of any manual ``--aug-copies-per-label`` floor, so no class's estimated
real-audio size trails the largest class by more than that ratio. Set to 0 to
disable and rely purely on manual per-label counts.

Requires the ``train`` extra: ``pip install -e '.[train]'`` (tf2onnx, onnx) plus
tensorflow / tensorflow-hub / onnxruntime (core dependencies).
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import wave
from pathlib import Path
from typing import Iterable

import numpy as np

from minimappr.classifiers.yamnet import (
    YAMNET_MAX_INPUT_GAIN,
    YAMNET_PREPROCESS_VERSION,
    YAMNET_TARGET_RMS,
)
from minimappr.training.augment import augment_waveform, synthesize_ambient_windows
from minimappr.training.embedding_cache import (
    EMBEDDING_DIM,
    EmbeddingCache,
    YamnetEmbedder,
    file_content_hash,
)
from minimappr.training.wav_dataset import (
    WavExample,
    discover_wavs,
    load_promoted,
)

DEFAULT_LABELS = ["ambient", "drone", "coyote"]
NEGATIVE_LABEL = "ambient"
OPSET = 13
SAMPLE_RATE = 16_000

# Reliability floor for weak positive classes (marks "reliable": false).
_RELIABLE_MIN_GROUPS = 5
_RELIABLE_MIN_FRAMES = 50

logger = logging.getLogger("train_drone_head")


# --------------------------------------------------------------------------- #
# WAV reading
# --------------------------------------------------------------------------- #
def read_wav_waveform(
    path: Path, offset_s: float = 0.0, duration_s: float = 0.0
) -> tuple[np.ndarray, int]:
    """Read a mono float32 waveform (segment) from a WAV file at its native rate.

    ``duration_s <= 0`` reads to end. Returns ``(waveform, sample_rate)``.
    """
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        n_channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        total = wav.getnframes()
        start = int(round(offset_s * rate)) if offset_s > 0 else 0
        start = max(0, min(start, total))
        if duration_s and duration_s > 0:
            count = int(round(duration_s * rate))
        else:
            count = total - start
        count = max(0, min(count, total - start))
        wav.setpos(start)
        raw = wav.readframes(count)

    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"{path}: unsupported sample width {sampwidth}")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data.astype(np.float32), rate


# --------------------------------------------------------------------------- #
# Variant planning + embedding
# --------------------------------------------------------------------------- #
class PlannedVariant:
    """One embedding job: a WavExample (or synthetic window) + variant tag."""

    __slots__ = ("example", "label", "group", "variant", "seed", "synth_wave", "train_only")

    def __init__(self, *, example, label, group, variant, seed=0, synth_wave=None, train_only=False):
        self.example = example
        self.label = label
        self.group = group
        self.variant = variant
        self.seed = seed
        self.synth_wave = synth_wave
        self.train_only = train_only


def _aug_copies_for(label: str, base: int, per_label: dict[str, int]) -> int:
    return per_label.get(label, base)


def estimate_real_frame_weights(
    examples: list[WavExample], labels: list[str], segment_seconds: float
) -> dict[str, float]:
    """Cheap per-label size estimate, in seconds of real (pre-augmentation) audio.

    Avoids running YAMNet just to count frames: seconds scale linearly with frame
    count for a fixed hop, so ratios computed from duration match ratios computed
    from frames. ``promoted`` WAV examples carry ``duration_s == 0`` (whole file,
    unread here); ``promoted_npy`` examples are a single mean-pooled vector. Both
    are approximated as one ``segment_seconds`` window — a deliberate estimate,
    not an exact count.
    """
    weights = {name: 0.0 for name in labels}
    for ex in examples:
        dur = ex.duration_s if ex.duration_s > 0 else segment_seconds
        weights[ex.label] = weights.get(ex.label, 0.0) + dur
    return weights


def compute_auto_balance_aug_copies(
    weights: dict[str, float],
    target_ratio: float,
    manual_per_label: dict[str, int],
    base_aug_copies: int,
) -> dict[str, int]:
    """Bump aug-copy counts so no label's estimated size trails the largest by
    more than ``target_ratio`` (e.g. ``3.0`` => no class smaller than 1/3 of the
    biggest after augmentation).

    Manual ``--aug-copies-per-label`` values are a floor, never lowered — this
    only adds copies on top when a class would otherwise fall short of the ratio.
    Labels with zero real examples are left alone (nothing to multiply).
    """
    result = dict(manual_per_label)
    if target_ratio <= 0 or not weights:
        return result
    max_weight = max(weights.values(), default=0.0)
    if max_weight <= 0:
        return result
    target_weight = max_weight / target_ratio

    for label, weight in weights.items():
        if weight <= 0:
            continue
        manual_copies = manual_per_label.get(label, base_aug_copies)
        effective_weight = weight * (1 + manual_copies)
        if effective_weight >= target_weight:
            continue
        needed_multiplier = target_weight / weight  # total copies factor incl. original
        needed_copies = max(manual_copies, int(np.ceil(needed_multiplier)) - 1)
        result[label] = needed_copies
    return result


def build_variant_plan(
    examples: list[WavExample],
    labels: list[str],
    *,
    aug_copies: int,
    aug_copies_per_label: dict[str, int],
    synth_ambient_count: int,
    segment_seconds: float,
    seed: int,
) -> list[PlannedVariant]:
    """Return the full list of embedding jobs (orig + aug copies + synth ambience)."""
    plan: list[PlannedVariant] = []
    for ex in examples:
        plan.append(PlannedVariant(example=ex, label=ex.label, group=ex.group, variant="orig"))
        # promoted_npy has no waveform to augment.
        if ex.origin == "promoted_npy":
            continue
        n_aug = _aug_copies_for(ex.label, aug_copies, aug_copies_per_label)
        for k in range(n_aug):
            plan.append(
                PlannedVariant(
                    example=ex,
                    label=ex.label,
                    group=ex.group,  # aug inherits source group (train-only drop later)
                    variant=f"aug{k}-s{seed}",
                    seed=seed + k + 1,
                    train_only=True,
                )
            )
    # Synthetic ambience: forced train-only, its own groups.
    for variant_key, wave_arr in synthesize_ambient_windows(
        synth_ambient_count, segment_seconds, seed
    ):
        plan.append(
            PlannedVariant(
                example=None,
                label=NEGATIVE_LABEL if NEGATIVE_LABEL in labels else labels[0],
                group=f"synth:{variant_key}",
                variant=variant_key,
                synth_wave=wave_arr,
                train_only=True,
            )
        )
    return plan


def embed_plan(
    plan: list[PlannedVariant],
    embedder: YamnetEmbedder,
    cache: EmbeddingCache,
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Embed every planned variant, returning frame-level arrays + per-label counts.

    Returns ``(frames, y, groups, train_only_mask, counts)``.
    """
    label_to_index = {name: i for i, name in enumerate(labels)}
    content_hashes: dict[Path, str] = {}
    frames: list[np.ndarray] = []
    y: list[int] = []
    groups: list[str] = []
    train_only: list[bool] = []
    counts: dict[str, dict[str, int]] = {
        name: {"files": 0, "real": 0, "aug": 0, "synth": 0, "promoted": 0} for name in labels
    }
    seen_files: dict[str, set] = {name: set() for name in labels}

    total = len(plan)
    for idx, pv in enumerate(plan):
        if idx % 200 == 0:
            logger.info("Embedding %d/%d (cache hit rate %.2f)", idx, total, cache.hit_rate)
        emb = _embed_variant(pv, embedder, cache, content_hashes)
        if emb.shape[0] == 0:
            continue
        li = label_to_index[pv.label]
        for frame in emb:
            frames.append(frame.astype(np.float32))
            y.append(li)
            groups.append(pv.group)
            train_only.append(pv.train_only)

        # Bookkeeping.
        n = emb.shape[0]
        if pv.synth_wave is not None:
            counts[pv.label]["synth"] += n
        elif pv.variant == "orig":
            if pv.example is not None and pv.example.origin in {"promoted", "promoted_npy"}:
                counts[pv.label]["promoted"] += n
            else:
                counts[pv.label]["real"] += n
            if pv.example is not None:
                seen_files[pv.label].add(str(pv.example.path))
        else:
            counts[pv.label]["aug"] += n

    for name in labels:
        counts[name]["files"] = len(seen_files[name])

    logger.info(
        "Embedded %d frames (cache hits=%d misses=%d hit_rate=%.2f)",
        len(frames), cache.hits, cache.misses, cache.hit_rate,
    )
    if not frames:
        raise SystemExit("No embeddings produced — check --wav-dir / --promoted-dir")
    return (
        np.stack(frames).astype(np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(groups, dtype=object),
        np.asarray(train_only, dtype=bool),
        counts,
    )


def _embed_variant(pv, embedder, cache, content_hashes) -> np.ndarray:
    # Synthetic ambience: waveform is in-memory, cache by its self-describing key.
    if pv.synth_wave is not None:
        key = cache.make_key("synth", variant=pv.variant)
        return cache.get_or_compute(key, lambda: embedder.embed(pv.synth_wave, SAMPLE_RATE))

    ex: WavExample = pv.example
    # Promoted npy: a precomputed mean-pooled 1024-d vector (no waveform).
    if ex.origin == "promoted_npy":
        def _load_npy():
            vec = np.load(ex.path).astype(np.float32).reshape(-1)
            if vec.shape[0] != EMBEDDING_DIM:
                logger.warning("%s: bad npy dim %d; skipping", ex.path, vec.shape[0])
                return np.zeros((0, EMBEDDING_DIM), np.float32)
            return vec[None, :]
        return _load_npy()

    path = ex.path
    if path not in content_hashes:
        content_hashes[path] = file_content_hash(path)
    chash = content_hashes[path]
    offset_ms = int(round(ex.offset_s * 1000))
    dur_ms = int(round(ex.duration_s * 1000))
    key = cache.make_key(chash, offset_ms=offset_ms, dur_ms=dur_ms, variant=pv.variant)

    def _compute():
        wave_arr, rate = read_wav_waveform(path, ex.offset_s, ex.duration_s)
        if pv.variant != "orig":
            wave_arr = augment_waveform(wave_arr, pv.seed)
        return embedder.embed(wave_arr, rate)

    return cache.get_or_compute(key, _compute)


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def per_class_group_split(
    y: np.ndarray,
    groups: np.ndarray,
    train_only: np.ndarray,
    labels: list[str],
    seed: int,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group-aware split done independently per class.

    Groups are split on their **real** (non-``train_only``) frames, so splitting
    each label's groups independently guarantees scarce classes (e.g. coyote with
    ~4 source groups) land at least one group in val AND test. Then:

    * real frames follow their group's fold;
    * augmentation frames whose source group landed in val/test are **dropped**
      (excluded from every fold) so an aug copy never sits beside its source;
    * augmentation frames in a train group, and synthetic frames (whose groups
      have no real frames), go to train.

    Dropped frames are in none of the returned masks — the three masks partition
    only the kept frames.
    """
    n = len(y)
    is_train = np.zeros(n, dtype=bool)
    is_val = np.zeros(n, dtype=bool)
    is_test = np.zeros(n, dtype=bool)
    rng = np.random.default_rng(seed)

    real = ~train_only
    for li in range(len(labels)):
        cls_mask = y == li
        # Groups eligible for splitting: those with at least one real frame.
        real_groups = np.array(
            sorted(set(np.unique(groups[cls_mask & real]))), dtype=object
        )
        rng.shuffle(real_groups)
        m = len(real_groups)
        if m <= 1:
            val_groups: set = set()
            test_groups: set = set()
        elif m == 2:
            test_groups, val_groups = {real_groups[0]}, set()
        else:
            n_test = max(1, int(round(m * test_frac)))
            n_val = max(1, int(round(m * val_frac)))
            test_groups = set(real_groups[:n_test])
            val_groups = set(real_groups[n_test : n_test + n_val])

        for i in np.flatnonzero(cls_mask):
            g = groups[i]
            in_val = g in val_groups
            in_test = g in test_groups
            if real[i]:
                if in_test:
                    is_test[i] = True
                elif in_val:
                    is_val[i] = True
                else:
                    is_train[i] = True
            else:
                # Augmentation / synthetic: keep only in train, drop from val/test.
                if in_val or in_test:
                    continue  # dropped
                is_train[i] = True
    return is_train, is_val, is_test


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(n_labels: int):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(EMBEDDING_DIM,), name="embedding"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(n_labels, activation="softmax", name="probs"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def class_weights(labels_int: np.ndarray, n_labels: int, cap: float) -> dict[int, float]:
    counts = np.bincount(labels_int, minlength=n_labels).astype(np.float64)
    counts[counts == 0] = 1.0
    total = counts.sum()
    weights = {i: float(total / (n_labels * counts[i])) for i in range(n_labels)}
    if cap and cap > 0:
        weights = {i: min(w, cap) for i, w in weights.items()}
    return weights


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _roc_auc(scores: np.ndarray, y: np.ndarray) -> float:
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    ranks = avg[inv]
    r_pos = ranks[y == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _binary_metrics(scores: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(np.int64)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": round(threshold, 3),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def sweep_thresholds(scores: np.ndarray, y: np.ndarray, min_precision: float = 0.95):
    sweep = [_binary_metrics(scores, y, t) for t in np.linspace(0.05, 0.95, 19)]
    passing = [m for m in sweep if m["precision"] >= min_precision]
    alert = min(passing, key=lambda m: m["threshold"]) if passing else max(sweep, key=lambda m: m["precision"])
    detect = max(sweep, key=lambda m: m["f1"])
    return alert, detect, sweep


def clip_max_by_group(frame_scores: np.ndarray, groups: np.ndarray, y_pos: np.ndarray):
    """Aggregate per-frame scores to clip level via group max; label = any-positive."""
    clip_scores: list[float] = []
    clip_labels: list[int] = []
    for g in np.unique(groups):
        mask = groups == g
        clip_scores.append(float(frame_scores[mask].max()))
        clip_labels.append(int(y_pos[mask].max()))
    return np.asarray(clip_scores), np.asarray(clip_labels, dtype=np.int64)


def quantized_model_quality_metrics(
    float_probs: np.ndarray, int8_probs: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Measure whether quantization changed classifier decisions on labelled data.

    The largest probability difference is intentionally diagnostic-only: one
    low-margin frame can have a large delta without changing a prediction or
    reducing test accuracy.  Agreement and accuracy regression instead measure
    the model behavior users receive from the quantized artifact.
    """
    if float_probs.shape != int8_probs.shape:
        raise ValueError("Float and int8 predictions must have matching shapes")
    if len(float_probs) != len(labels):
        raise ValueError("Prediction and label counts must match")
    if len(labels) == 0:
        return empty_quantized_model_quality_metrics()

    float_labels = np.argmax(float_probs, axis=1)
    int8_labels = np.argmax(int8_probs, axis=1)
    float_accuracy = float(np.mean(float_labels == labels))
    int8_accuracy = float(np.mean(int8_labels == labels))
    return {
        "max_probability_difference": float(np.max(np.abs(float_probs - int8_probs))),
        "mean_probability_difference": float(np.mean(np.abs(float_probs - int8_probs))),
        "prediction_agreement": float(np.mean(float_labels == int8_labels)),
        "float_accuracy": float_accuracy,
        "int8_accuracy": int8_accuracy,
        "accuracy_drop": float_accuracy - int8_accuracy,
    }


def empty_quantized_model_quality_metrics() -> dict[str, float]:
    """Return a non-failing quality result when no held-out test frame exists."""
    return {
        "max_probability_difference": 0.0,
        "mean_probability_difference": 0.0,
        "prediction_agreement": 1.0,
        "float_accuracy": 0.0,
        "int8_accuracy": 0.0,
        "accuracy_drop": 0.0,
    }


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export_onnx(model, float_path: Path) -> None:
    import onnx
    import tensorflow as tf
    import tf2onnx

    @tf.function(input_signature=[tf.TensorSpec((None, EMBEDDING_DIM), tf.float32, name="embedding")])
    def _forward(embedding):
        return tf.identity(model(embedding, training=False), name="probs")

    onnx_model, _ = tf2onnx.convert.from_function(
        _forward,
        input_signature=[tf.TensorSpec((None, EMBEDDING_DIM), tf.float32, name="embedding")],
        opset=OPSET,
    )
    onnx.save(onnx_model, str(float_path))


class FullDatasetCalibrationDataReader:
    """Yield every embedding frame to ONNX Runtime static quantization.

    Static quantization needs representative activation values.  Calibration is
    intentionally performed over the complete embedded dataset, rather than a
    random subset, because rare real-world acoustic conditions are precisely
    the cases where clipping activation ranges harms detector quality.
    """

    def __init__(self, input_name: str, frames: np.ndarray, batch_size: int):
        if frames.ndim != 2 or frames.shape[0] == 0:
            raise ValueError("Static quantization requires at least one 2-D embedding frame")
        if batch_size <= 0:
            raise ValueError("calibration batch size must be positive")
        self.input_name = input_name
        self.frames = frames.astype(np.float32, copy=False)
        self.batch_size = batch_size
        self._offset = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._offset >= len(self.frames):
            return None
        next_offset = min(self._offset + self.batch_size, len(self.frames))
        batch = self.frames[self._offset:next_offset]
        self._offset = next_offset
        return {self.input_name: batch}

    def rewind(self) -> None:
        self._offset = 0


def quantize_int8(
    float_path: Path,
    int8_path: Path,
    calibration_frames: np.ndarray,
    calibration_batch_size: int,
) -> int:
    """Create a per-channel, full-dataset static QDQ int8 ONNX model.

    Returns the number of frames used for calibration so it can be recorded in
    the artifact metadata.
    """
    import onnx
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    # Shape inference and graph optimization must run before quantization.  It
    # gives the calibrator tensor shapes/ranges and keeps optimization separate
    # from quantization, so quantization accuracy regressions remain debuggable.
    with tempfile.TemporaryDirectory(prefix="drone_head_onnx_preprocess_") as temp_dir:
        preprocessed_path = Path(temp_dir) / "drone_head.preprocessed.onnx"
        quant_pre_process(
            str(float_path),
            str(preprocessed_path),
            skip_symbolic_shape=False,
            skip_optimization=False,
            skip_onnx_shape=False,
        )
        float_model = onnx.load(str(preprocessed_path))
        if not float_model.graph.input:
            raise ValueError(f"{preprocessed_path} has no model inputs for calibration")
        reader = FullDatasetCalibrationDataReader(
            float_model.graph.input[0].name, calibration_frames, calibration_batch_size
        )
        quantize_static(
            str(preprocessed_path),
            str(int8_path),
            reader,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
        )
    return len(calibration_frames)


def export_tflite(model, tflite_path: Path, calib: np.ndarray) -> None:
    import tensorflow as tf

    @tf.function(input_signature=[tf.TensorSpec((None, EMBEDDING_DIM), tf.float32, name="embedding")])
    def _serve(embedding):
        return model(embedding, training=False)

    converter = tf.lite.TFLiteConverter.from_concrete_functions([_serve.get_concrete_function()], model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def _rep():
        for row in calib:
            yield [row[None, :].astype(np.float32)]

    converter.representative_dataset = _rep
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    _validate_fully_integer_tflite(tf, tflite_model)
    tflite_path.write_bytes(tflite_model)


def _validate_fully_integer_tflite(tf, model_bytes: bytes) -> None:
    """Fail export if the requested int8 TFLite artifact has float tensors."""
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if any(detail["dtype"] != np.int8 for detail in input_details + output_details):
        raise ValueError("TFLite export did not produce int8 input/output tensors")
    float_tensors = [
        detail["name"]
        for detail in interpreter.get_tensor_details()
        if np.issubdtype(detail["dtype"], np.floating)
    ]
    if float_tensors:
        raise ValueError(
            "TFLite export contains float tensors despite integer-only configuration: "
            + ", ".join(float_tensors[:5])
        )


def onnx_predict(model_path: Path, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return sess.run(None, {name: x.astype(np.float32)})[0]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _parse_per_label(raw: str | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for pair in (raw or "").split(","):
        pair = pair.strip()
        if not pair:
            continue
        key, _, value = pair.partition("=")
        try:
            out[key.strip().lower()] = int(value)
        except ValueError:
            continue
    return out


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wav-dir", type=Path, default=Path("drone_dataset/wav"))
    parser.add_argument("--promoted-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("data/models"))
    parser.add_argument("--cache-dir", type=Path, default=Path("drone_dataset/.embed_cache"))
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--segment-overlap", type=float, default=0.5)
    parser.add_argument("--aug-copies", type=int, default=1)
    parser.add_argument("--aug-copies-per-label", default="coyote=8")
    parser.add_argument(
        "--auto-balance-ratio", type=float, default=3.0,
        help="Bump aug copies so no label's estimated real-audio size trails the "
             "largest by more than this ratio (e.g. 3.0 = no class below 1/3 of "
             "the biggest). Never lowers --aug-copies-per-label values. 0 disables.",
    )
    parser.add_argument("--synth-ambient-count", type=int, default=1000)
    parser.add_argument("--max-files-per-label", type=int, default=0)
    parser.add_argument(
        "--calibration-batch-size", type=int, default=256,
        help="Embedding frames per static-quantization calibration inference.",
    )
    parser.add_argument(
        "--min-int8-prediction-agreement", type=float, default=0.97,
        help="Minimum float-vs-int8 top-class agreement on labelled test frames.",
    )
    parser.add_argument(
        "--max-int8-accuracy-drop", type=float, default=0.02,
        help="Largest permitted float-to-int8 test-accuracy regression (fraction).",
    )
    parser.add_argument("--class-weight-cap", type=float, default=10.0)
    parser.add_argument("--no-tflite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    labels = [x.strip().lower() for x in args.labels.split(",") if x.strip()]
    if NEGATIVE_LABEL not in labels:
        logger.warning("Negative label %r not in labels %s", NEGATIVE_LABEL, labels)
    n_labels = len(labels)

    # ---- Discover ----
    examples = discover_wavs(
        args.wav_dir, labels,
        segment_seconds=args.segment_seconds, segment_overlap=args.segment_overlap,
    )
    if args.promoted_dir is not None:
        examples += load_promoted(args.promoted_dir, labels)
    if args.max_files_per_label > 0:
        examples = _cap_per_label(examples, labels, args.max_files_per_label)
    logger.info("Discovered %d WAV/promoted examples", len(examples))

    aug_per_label = _parse_per_label(args.aug_copies_per_label)
    if args.auto_balance_ratio > 0:
        real_weights = estimate_real_frame_weights(examples, labels, args.segment_seconds)
        balanced = compute_auto_balance_aug_copies(
            real_weights, args.auto_balance_ratio, aug_per_label, args.aug_copies
        )
        if balanced != aug_per_label:
            logger.info(
                "Auto-balance (ratio=%.1f) adjusted aug copies: %s -> %s (weights_s=%s)",
                args.auto_balance_ratio, aug_per_label, balanced,
                {k: round(v, 1) for k, v in real_weights.items()},
            )
        aug_per_label = balanced
    plan = build_variant_plan(
        examples, labels,
        aug_copies=args.aug_copies, aug_copies_per_label=aug_per_label,
        synth_ambient_count=args.synth_ambient_count,
        segment_seconds=args.segment_seconds, seed=args.seed,
    )
    logger.info("Variant plan: %d embedding jobs", len(plan))

    # ---- Embed ----
    embedder = YamnetEmbedder()
    cache = EmbeddingCache(args.cache_dir)
    frames, y, groups, train_only, counts = embed_plan(plan, embedder, cache, labels)

    logger.info("Frames per label: %s", {labels[i]: int((y == i).sum()) for i in range(n_labels)})

    # ---- Split (per-class, group-aware) ----
    is_train, is_val, is_test = per_class_group_split(y, groups, train_only, labels, args.seed)
    logger.info("Split frames: train=%d val=%d test=%d", is_train.sum(), is_val.sum(), is_test.sum())

    import tensorflow as tf

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # Class weights from REAL frames only (exclude aug/synth train-only).
    real_train = is_train & ~train_only
    weights = class_weights(y[real_train] if real_train.any() else y[is_train], n_labels, args.class_weight_cap)
    logger.info("Class weights: %s", {labels[i]: round(w, 3) for i, w in weights.items()})

    model = build_model(n_labels)
    val_data = (frames[is_val], y[is_val]) if is_val.any() else None
    callbacks = []
    if val_data is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=6, restore_best_weights=True))
    model.fit(
        frames[is_train], y[is_train],
        validation_data=val_data,
        epochs=args.epochs, batch_size=args.batch_size,
        class_weight=weights, callbacks=callbacks, verbose=2,
    )

    # ---- Evaluate on real-only test frames ----
    real_test = is_test  # is_test already excludes train-only groups by construction
    test_frames = frames[real_test]
    test_y = y[real_test]
    test_groups = groups[real_test]
    probs = model.predict(test_frames, verbose=0) if len(test_frames) else np.zeros((0, n_labels))

    per_class: dict[str, dict] = {}
    thresholds: dict[str, dict] = {}
    weak_classes: list[str] = []
    for li, name in enumerate(labels):
        if name == NEGATIVE_LABEL:
            continue
        if len(probs) == 0:
            per_class[name] = {"frame_auc": None, "clip_auc": None, "reliable": False, "note": "no test frames"}
            weak_classes.append(name)
            continue
        pos = (test_y == li).astype(np.int64)
        frame_scores = probs[:, li]
        frame_auc = _roc_auc(frame_scores, pos)
        clip_scores, clip_labels = clip_max_by_group(frame_scores, test_groups, pos)
        clip_auc = _roc_auc(clip_scores, clip_labels)
        alert, detect, sweep = sweep_thresholds(clip_scores, clip_labels)
        n_pos_groups = int(len(np.unique(test_groups[pos == 1]))) if pos.any() else 0
        n_pos_frames = int(pos.sum())
        reliable = n_pos_groups >= _RELIABLE_MIN_GROUPS and n_pos_frames >= _RELIABLE_MIN_FRAMES
        if not reliable:
            weak_classes.append(name)
        per_class[name] = {
            "frame_auc": round(frame_auc, 4) if frame_auc == frame_auc else None,
            "clip_auc": round(clip_auc, 4) if clip_auc == clip_auc else None,
            "test_groups": n_pos_groups,
            "test_frames": n_pos_frames,
            "reliable": reliable,
            "sweep": sweep,
        }
        thresholds[name] = {"alert": alert["threshold"], "detect": detect["threshold"]}
        logger.info("%s: frame AUC=%s clip AUC=%s reliable=%s alert=%s detect=%s",
                    name, per_class[name]["frame_auc"], per_class[name]["clip_auc"],
                    reliable, alert["threshold"], detect["threshold"])

    # ---- Export ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    float_path = args.out_dir / "drone_head.float.onnx"
    int8_path = args.out_dir / "drone_head.onnx"
    tflite_path = args.out_dir / "drone_head.tflite"

    export_onnx(model, float_path)
    # Use every embedded frame for both ONNX static calibration and TFLite's
    # representative data.  This includes held-out real frames and the
    # train-only variants/synthetic ambience that can occur at runtime.
    calibration_frames = frames
    calibration_frame_count = quantize_int8(
        float_path, int8_path, calibration_frames, args.calibration_batch_size
    )
    if not args.no_tflite:
        try:
            export_tflite(model, tflite_path, calibration_frames)
        except Exception as exc:  # noqa: BLE001
            logger.warning("TFLite export failed (non-fatal): %s", exc)

    # ---- Quantized-model quality gate ----
    # A max probability delta is logged for diagnosis, but is not a quality
    # gate: static quantization can move a low-margin softmax value sharply
    # while preserving the classifier's decision and its measured accuracy.
    quantization_quality = empty_quantized_model_quality_metrics()
    if len(test_frames):
        float_pred = onnx_predict(float_path, test_frames)
        int8_pred = onnx_predict(int8_path, test_frames)
        quantization_quality = quantized_model_quality_metrics(float_pred, int8_pred, test_y)
    logger.info(
        "int8 quality: agreement=%.4f float_acc=%.4f int8_acc=%.4f "
        "accuracy_drop=%.4f max_prob_diff=%.4f mean_prob_diff=%.4f",
        quantization_quality["prediction_agreement"],
        quantization_quality["float_accuracy"],
        quantization_quality["int8_accuracy"],
        quantization_quality["accuracy_drop"],
        quantization_quality["max_probability_difference"],
        quantization_quality["mean_probability_difference"],
    )
    if (
        quantization_quality["prediction_agreement"] < args.min_int8_prediction_agreement
        or quantization_quality["accuracy_drop"] > args.max_int8_accuracy_drop
    ):
        logger.error(
            "Quantized-model quality gate FAILED: agreement %.4f < %.4f or "
            "accuracy drop %.4f > %.4f",
            quantization_quality["prediction_agreement"], args.min_int8_prediction_agreement,
            quantization_quality["accuracy_drop"], args.max_int8_accuracy_drop,
        )
        return 1

    # ---- Metadata v2 + report ----
    metadata = {
        "schema_version": 2,
        "labels": labels,
        "negative_label": NEGATIVE_LABEL,
        "input_shape": [1, EMBEDDING_DIM],
        "embedding_model": "yamnet",
        "opset": OPSET,
        "preprocessing": {
            "version": YAMNET_PREPROCESS_VERSION,
            "target_rms": YAMNET_TARGET_RMS,
            "max_input_gain": YAMNET_MAX_INPUT_GAIN,
        },
        "thresholds": thresholds,
        "metrics": {name: {"frame_auc": v.get("frame_auc"), "clip_auc": v.get("clip_auc")}
                    for name, v in per_class.items()},
        "dataset_counts": counts,
        "weak_classes": sorted(set(weak_classes)),
        "augmentation": {
            "aug_copies": args.aug_copies,
            "aug_copies_per_label": aug_per_label,
            "auto_balance_ratio": args.auto_balance_ratio,
            "synth_ambient_count": args.synth_ambient_count,
            "segment_seconds": args.segment_seconds,
            "segment_overlap": args.segment_overlap,
        },
        "quantized_model_quality": {
            key: round(value, 4) for key, value in quantization_quality.items()
        },
        "quantization": {
            "method": "static_qdq",
            "preprocessed": True,
            "calibration_frames": calibration_frame_count,
            "calibration_dataset": "all_embedded_frames",
            "calibration_batch_size": args.calibration_batch_size,
        },
    }
    (args.out_dir / "drone_head.metadata.json").write_text(json.dumps(metadata, indent=2))
    (args.out_dir / "eval_report.json").write_text(
        json.dumps({"per_class": per_class, "thresholds": thresholds, "metadata": metadata}, indent=2)
    )

    logger.info("Wrote %s + metadata v2 (weak_classes=%s)", int8_path, metadata["weak_classes"])
    return 0


def _cap_per_label(examples: list[WavExample], labels: list[str], cap: int) -> list[WavExample]:
    """Keep at most ``cap`` distinct source files per label (deterministic by path)."""
    by_label: dict[str, list[WavExample]] = {name: [] for name in labels}
    for ex in examples:
        by_label.setdefault(ex.label, []).append(ex)
    kept: list[WavExample] = []
    for name, exs in by_label.items():
        # Rank files by path; keep all segments of the first `cap` files.
        files = sorted({ex.path for ex in exs}, key=str)[:cap]
        keep_files = set(files)
        kept += [ex for ex in exs if ex.path in keep_files]
    return kept


if __name__ == "__main__":
    raise SystemExit(main())
