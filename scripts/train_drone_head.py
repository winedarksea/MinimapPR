#!/usr/bin/env python3
"""Train the drone-head classifier on YAMNet embeddings and export int8 ONNX.

The drone head is a small dense classifier that consumes YAMNet embedding
frames (``[1024]`` float32) and emits a binary ``no_drone``/``drone`` softmax.
It is chained after YAMNet at runtime (see ``classifiers/drone_head.py`` and the
``chains`` block of ``data/classifier_routing.json``).

Dataset layout (``drone_dataset/``)::

    drone/{matrice,mavic3,mavicmini}/<ts>_<idx>.tfdata   # tf.io.serialize_tensor'd [2,1024] f32
    no_drone/<ts>_<idx>.tfdata

Each ``.tfdata`` holds two YAMNet embedding frames for a ~1s clip. Optional
promoted examples (``data/training/{id}.npy``, 1024-d mean-pooled) can be folded
in with ``--promoted-dir``.

Outputs (``data/models/`` by default):

    drone_head.onnx            int8 QDQ model shipped + loaded at runtime
    drone_head.metadata.json   labels, input shape, thresholds, metrics, counts
    drone_head.float.onnx      float32 reference (gitignored)
    drone_head.tflite          int8 TFLite (gitignored)
    eval_report.json           frame/clip metrics + threshold sweep (gitignored)

Group-aware split keys on the filename timestamp truncated to the minute so
consecutive near-duplicate clips never straddle train/val/test.

Usage::

    python scripts/train_drone_head.py \
        --dataset-dir drone_dataset --out-dir data/models --epochs 40

Requires the ``train`` extra: ``pip install -e '.[train]'`` (tf2onnx, onnx) plus
onnxruntime (already a core dependency).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np

LABELS = ["no_drone", "drone"]  # index order = softmax output order
LABEL_TO_INDEX = {name: i for i, name in enumerate(LABELS)}
EMBEDDING_DIM = 1024
OPSET = 13

logger = logging.getLogger("train_drone_head")

# Group key = date + hour:minute from a "YYYY-MM-DD_HH-MM-SS_idx" style stem.
_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[_T](\d{2})[-:](\d{2})")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _group_key(path: Path) -> str:
    """Minute-resolution group so near-duplicate consecutive clips stay together."""
    m = _TS_RE.search(path.stem)
    if m:
        return f"{m.group(1)}_{m.group(2)}-{m.group(3)}"
    # Fall back to a stable per-file group (no leakage risk, just no grouping).
    return hashlib.sha1(path.stem.encode()).hexdigest()[:12]


def _parse_tfdata(path: Path) -> np.ndarray:
    """Return a ``[n_frames, 1024]`` float32 array from a serialized tensor file."""
    import tensorflow as tf  # local import: heavy, optional at module import time

    raw = tf.io.read_file(str(path))
    tensor = tf.io.parse_tensor(raw, out_type=tf.float32).numpy()
    tensor = np.asarray(tensor, dtype=np.float32)
    if tensor.ndim == 1:
        tensor = tensor[None, :]
    if tensor.shape[-1] != EMBEDDING_DIM:
        raise ValueError(f"{path}: expected last dim {EMBEDDING_DIM}, got {tensor.shape}")
    return tensor.reshape(-1, EMBEDDING_DIM)


def load_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load every ``.tfdata`` under ``drone/`` and ``no_drone/``.

    Returns ``(frames[N,1024], labels[N], groups[N], drone_types[N])`` where each
    embedding frame is a separate training example. ``drone_types`` records the
    drone-model subdir (or ``"none"``) for future multi-class work.
    """
    frames: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    drone_types: list[str] = []

    for label_name in LABELS:
        base = dataset_dir / label_name
        if not base.exists():
            logger.warning("Missing label dir %s; skipping", base)
            continue
        files = sorted(base.rglob("*.tfdata"))
        logger.info("%s: %d files", label_name, len(files))
        for path in files:
            try:
                clip = _parse_tfdata(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping %s: %s", path, exc)
                continue
            subdir = path.parent.name if path.parent.name != label_name else "none"
            group = _group_key(path)
            for frame in clip:
                frames.append(frame)
                labels.append(LABEL_TO_INDEX[label_name])
                groups.append(group)
                drone_types.append(subdir if label_name == "drone" else "none")

    if not frames:
        raise SystemExit(f"No .tfdata examples found under {dataset_dir}")
    return (
        np.stack(frames).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups, dtype=object),
        np.asarray(drone_types, dtype=object),
    )


def load_promoted(promoted_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optionally load promoted mean-pooled examples from ``data/training``.

    Expects ``{id}.npy`` (1024-d) plus a sibling ``manifest.json`` mapping id -> label.
    Each promoted example is its own group so it can land anywhere in the split.
    """
    manifest_path = promoted_dir / "manifest.json"
    if not manifest_path.exists():
        logger.info("No promoted manifest at %s; skipping promoted examples", manifest_path)
        return np.empty((0, EMBEDDING_DIM), np.float32), np.empty((0,), np.int64), np.empty((0,), object)
    manifest = json.loads(manifest_path.read_text())
    frames: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    for example_id, label_name in manifest.items():
        if label_name not in LABEL_TO_INDEX:
            continue
        npy = promoted_dir / f"{example_id}.npy"
        if not npy.exists():
            continue
        vec = np.load(npy).astype(np.float32).reshape(-1)
        if vec.shape[0] != EMBEDDING_DIM:
            continue
        frames.append(vec)
        labels.append(LABEL_TO_INDEX[label_name])
        groups.append(f"promoted:{example_id}")
    if not frames:
        return np.empty((0, EMBEDDING_DIM), np.float32), np.empty((0,), np.int64), np.empty((0,), object)
    logger.info("Loaded %d promoted examples", len(frames))
    return np.stack(frames).astype(np.float32), np.asarray(labels, np.int64), np.asarray(groups, object)


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def group_split(
    groups: np.ndarray, seed: int, val_frac: float = 0.1, test_frac: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group-aware split returning boolean masks (train, val, test).

    Whole groups are assigned to a single fold so near-duplicate frames never
    leak across folds. Pure numpy, no sklearn dependency.
    """
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    test_groups = set(unique[:n_test])
    val_groups = set(unique[n_test : n_test + n_val])

    is_test = np.array([g in test_groups for g in groups])
    is_val = np.array([g in val_groups for g in groups])
    is_train = ~(is_test | is_val)
    return is_train, is_val, is_test


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model():
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(EMBEDDING_DIM,), name="embedding"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(LABELS), activation="softmax", name="probs"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],  # AUC is computed manually at eval (2-class softmax)
    )
    return model


def class_weights(labels: np.ndarray) -> dict[int, float]:
    counts = np.bincount(labels, minlength=len(LABELS)).astype(np.float64)
    counts[counts == 0] = 1.0
    total = counts.sum()
    return {i: float(total / (len(LABELS) * counts[i])) for i in range(len(LABELS))}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _binary_metrics(probs_drone: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    pred = (probs_drone >= threshold).astype(np.int64)
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


def _roc_auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC (Mann–Whitney U), no sklearn."""
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    ranks = avg[inv]
    r_pos = ranks[y == 1].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def clip_probs_from_frames(
    frame_probs: np.ndarray, groups: np.ndarray, y_frame: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate frame drone-probabilities to clip level via max over the group."""
    clip_scores: list[float] = []
    clip_labels: list[int] = []
    for g in np.unique(groups):
        mask = groups == g
        clip_scores.append(float(frame_probs[mask].max()))
        clip_labels.append(int(y_frame[mask].max()))  # any drone frame => drone clip
    return np.asarray(clip_scores), np.asarray(clip_labels, dtype=np.int64)


def sweep_thresholds(
    scores: np.ndarray, y: np.ndarray, min_precision: float = 0.95
) -> tuple[dict, dict, list[dict]]:
    """Return (alert_metrics, detect_metrics, full_sweep).

    ``alert`` = lowest threshold whose precision ≥ ``min_precision`` (or the
    highest-precision point if none clears). ``detect`` = best-F1 threshold.
    """
    sweep = [_binary_metrics(scores, y, t) for t in np.linspace(0.05, 0.95, 19)]
    passing = [m for m in sweep if m["precision"] >= min_precision]
    alert = min(passing, key=lambda m: m["threshold"]) if passing else max(sweep, key=lambda m: m["precision"])
    detect = max(sweep, key=lambda m: m["f1"])
    return alert, detect, sweep


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export_onnx(model, float_path: Path) -> None:
    import onnx
    import tensorflow as tf
    import tf2onnx

    # from_function is Keras-3 safe (from_keras trips on keras_tensor output names).
    @tf.function(input_signature=[tf.TensorSpec((None, EMBEDDING_DIM), tf.float32, name="embedding")])
    def _forward(embedding):
        return tf.identity(model(embedding, training=False), name="probs")

    onnx_model, _ = tf2onnx.convert.from_function(
        _forward,
        input_signature=[tf.TensorSpec((None, EMBEDDING_DIM), tf.float32, name="embedding")],
        opset=OPSET,
    )
    onnx.save(onnx_model, str(float_path))


def quantize_int8(float_path: Path, int8_path: Path, calib: np.ndarray) -> None:
    from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static
    from onnxruntime.quantization.quant_utils import QuantFormat

    class _Reader(CalibrationDataReader):
        def __init__(self, data: np.ndarray) -> None:
            self._rows = iter([{"embedding": row[None, :].astype(np.float32)} for row in data])

        def get_next(self):
            return next(self._rows, None)

    quantize_static(
        str(float_path),
        str(int8_path),
        _Reader(calib),
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )


def export_tflite(model, tflite_path: Path, calib: np.ndarray) -> None:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def _rep():
        for row in calib:
            yield [row[None, :].astype(np.float32)]

    converter.representative_dataset = _rep
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_path.write_bytes(converter.convert())


def onnx_predict(model_path: Path, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return sess.run(None, {name: x.astype(np.float32)})[0]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("drone_dataset"))
    parser.add_argument("--promoted-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("data/models"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--calib-size", type=int, default=512)
    parser.add_argument("--parity-tol", type=float, default=0.05)
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    frames, labels, groups, drone_types = load_dataset(args.dataset_dir)
    if args.promoted_dir is not None:
        pf, pl, pg = load_promoted(args.promoted_dir)
        if len(pf):
            frames = np.concatenate([frames, pf])
            labels = np.concatenate([labels, pl])
            groups = np.concatenate([groups, pg])
            drone_types = np.concatenate([drone_types, np.array(["promoted"] * len(pf), object)])

    logger.info("Total frames: %d (%d drone, %d no_drone), %d groups",
                len(frames), int((labels == 1).sum()), int((labels == 0).sum()), len(np.unique(groups)))

    is_train, is_val, is_test = group_split(groups, args.seed)
    logger.info("Split frames: train=%d val=%d test=%d", is_train.sum(), is_val.sum(), is_test.sum())

    import tensorflow as tf

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    model = build_model()
    model.fit(
        frames[is_train], labels[is_train],
        validation_data=(frames[is_val], labels[is_val]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights(labels[is_train]),
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=6, restore_best_weights=True)],
        verbose=2,
    )

    # ---- Evaluate (Keras float) ----
    test_probs = model.predict(frames[is_test], verbose=0)[:, LABEL_TO_INDEX["drone"]]
    frame_auc = _roc_auc(test_probs, labels[is_test])
    clip_scores, clip_labels = clip_probs_from_frames(test_probs, groups[is_test], labels[is_test])
    clip_auc = _roc_auc(clip_scores, clip_labels)
    alert, detect, sweep = sweep_thresholds(clip_scores, clip_labels)
    logger.info("frame AUC=%.4f  clip AUC=%.4f  alert=%s  detect=%s",
                frame_auc, clip_auc, alert, detect)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    float_path = args.out_dir / "drone_head.float.onnx"
    int8_path = args.out_dir / "drone_head.onnx"
    tflite_path = args.out_dir / "drone_head.tflite"

    # ---- Export ----
    export_onnx(model, float_path)
    rng = np.random.default_rng(args.seed)
    calib_idx = rng.choice(np.flatnonzero(is_train), size=min(args.calib_size, int(is_train.sum())), replace=False)
    calib = frames[calib_idx]
    quantize_int8(float_path, int8_path, calib)
    try:
        export_tflite(model, tflite_path, calib)
    except Exception as exc:  # noqa: BLE001 — TFLite is a nice-to-have artifact
        logger.warning("TFLite export failed (non-fatal): %s", exc)

    # ---- Parity gate ----
    float_pred = onnx_predict(float_path, frames[is_test])[:, LABEL_TO_INDEX["drone"]]
    int8_pred = onnx_predict(int8_path, frames[is_test])[:, LABEL_TO_INDEX["drone"]]
    max_diff = float(np.max(np.abs(float_pred - int8_pred)))
    logger.info("int8 vs float32 max prob diff: %.4f (tol %.3f)", max_diff, args.parity_tol)
    if max_diff > args.parity_tol:
        logger.error("Parity gate FAILED: %.4f > %.3f", max_diff, args.parity_tol)
        return 1

    # ---- Metadata + report ----
    metadata = {
        "labels": LABELS,
        "input_shape": [1, EMBEDDING_DIM],
        "embedding_model": "yamnet",
        "opset": OPSET,
        "thresholds": {"alert": alert["threshold"], "detect": detect["threshold"]},
        "metrics": {"frame_auc": round(frame_auc, 4), "clip_auc": round(clip_auc, 4)},
        "dataset_counts": {
            "total_frames": int(len(frames)),
            "drone_frames": int((labels == 1).sum()),
            "no_drone_frames": int((labels == 0).sum()),
            "groups": int(len(np.unique(groups))),
        },
        "int8_parity_max_diff": round(max_diff, 4),
    }
    (args.out_dir / "drone_head.metadata.json").write_text(json.dumps(metadata, indent=2))
    (args.out_dir / "eval_report.json").write_text(json.dumps(
        {"alert": alert, "detect": detect, "sweep": sweep, "metadata": metadata}, indent=2))

    logger.info("Wrote %s + metadata (clip AUC %.4f)", int8_path, clip_auc)
    if clip_auc < 0.95:
        logger.warning("clip AUC %.4f below expected 0.95 — inspect eval_report.json", clip_auc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
