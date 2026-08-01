"""Drone-head runtime classifier tests with a tiny in-test ONNX head.

Builds a minimal 1024->2 Gemm+Softmax ONNX model on the fly so CI never depends
on the real trained artifact from ``scripts/train_drone_head.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from minimappr.classifiers.base import AudioClassifier, EmbeddingClassifier
from minimappr.classifiers.chaining import ChainStage, ChainedClassifier
from minimappr.classifiers.drone_head import (
    DroneHeadClassifier,
    PreprocessingProfileMismatchError,
)
from minimappr.models import ClassificationResult

EMBEDDING_DIM = 1024


def _write_tiny_head(path: Path, labels=("no_drone", "drone"), metadata=None) -> Path:
    """Softmax(Gemm(embedding, W, B)) with a per-class slice detector.

    Class ``c`` responds to the mean of its own contiguous slice of the embedding
    dims, so an input that is high in slice ``c`` (and low elsewhere) makes class
    ``c`` win. For the legacy 2-class case this reduces to drone=+mean/no_drone=−mean.
    """
    from onnx import TensorProto, helper, numpy_helper

    n = len(labels)
    w = np.zeros((EMBEDDING_DIM, n), dtype=np.float32)
    if n == 2:
        w[:, 0] = -1.0 / EMBEDDING_DIM
        w[:, 1] = 1.0 / EMBEDDING_DIM
    else:
        slice_size = EMBEDDING_DIM // n
        for c in range(n):
            lo = c * slice_size
            hi = EMBEDDING_DIM if c == n - 1 else (c + 1) * slice_size
            w[lo:hi, c] = 4.0 / max(1, hi - lo)
    b = np.zeros((n,), dtype=np.float32)

    nodes = [
        helper.make_node("Gemm", ["embedding", "W", "B"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["probs"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "drone_head",
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [None, EMBEDDING_DIM])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, [None, n])],
        [numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "B")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, str(path))

    meta = {"labels": list(labels), "input_shape": [1, EMBEDDING_DIM]}
    if metadata:
        meta.update(metadata)
    (path.parent / (path.stem + ".metadata.json")).write_text(json.dumps(meta))
    return path


def _slice_embedding(class_index: int, n_classes: int) -> np.ndarray:
    """Embedding that is high in class_index's slice, zero elsewhere."""
    emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    slice_size = EMBEDDING_DIM // n_classes
    lo = class_index * slice_size
    hi = EMBEDDING_DIM if class_index == n_classes - 1 else (class_index + 1) * slice_size
    emb[lo:hi] = 1.0
    return emb


def test_classify_embedding_positive_is_drone(tmp_path):
    model = _write_tiny_head(tmp_path / "drone_head.onnx")
    clf = DroneHeadClassifier(model, min_confidence=0.5)
    result = clf.classify_embedding(np.ones(EMBEDDING_DIM, dtype=np.float32))
    assert result.label == "drone"
    assert result.confidence > 0.5
    assert "drone" in result.scores and "no_drone" in result.scores


LABELS_3 = ("ambient", "drone", "coyote")


def test_three_class_coyote_dominant(tmp_path):
    model = _write_tiny_head(
        tmp_path / "drone_head.onnx", labels=LABELS_3, metadata={"negative_label": "ambient"}
    )
    clf = DroneHeadClassifier(model, min_confidence=0.4)
    result = clf.classify_embedding(_slice_embedding(2, 3))  # coyote slice
    assert result.label == "coyote"
    assert set(result.scores) == {"ambient", "drone", "coyote"}
    assert result.features["coyote_prob_max"] > result.features["drone_prob_max"]


def test_three_class_ambient_dominant_is_unknown(tmp_path):
    model = _write_tiny_head(
        tmp_path / "drone_head.onnx", labels=LABELS_3, metadata={"negative_label": "ambient"}
    )
    clf = DroneHeadClassifier(model, min_confidence=0.5)
    result = clf.classify_embedding(_slice_embedding(0, 3))  # ambient slice
    # Negative class wins => never emitted; falls back to unknown.
    assert result.label == "unknown"
    # Unknown must report zero confidence so it can never win a chain merge.
    assert result.confidence == 0.0


def test_three_class_per_class_max_across_frames(tmp_path):
    model = _write_tiny_head(
        tmp_path / "drone_head.onnx", labels=LABELS_3, metadata={"negative_label": "ambient"}
    )
    clf = DroneHeadClassifier(model, min_confidence=0.4)
    # One drone frame, one coyote frame: coyote frame should still be captured.
    frames = np.stack([_slice_embedding(1, 3), _slice_embedding(2, 3)]).astype(np.float32)
    result = clf.classify_embedding(frames)
    assert result.features["drone_prob_max"] > 0.4
    assert result.features["coyote_prob_max"] > 0.4


def test_missing_metadata_defaults_to_no_drone_negative(tmp_path):
    # No metadata file at all -> default binary labels, no_drone negative.
    from onnx import TensorProto, helper, numpy_helper

    w = np.zeros((EMBEDDING_DIM, 2), dtype=np.float32)
    w[:, 0] = -1.0 / EMBEDDING_DIM
    w[:, 1] = 1.0 / EMBEDDING_DIM
    graph = helper.make_graph(
        [
            helper.make_node("Gemm", ["embedding", "W", "B"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["probs"], axis=1),
        ],
        "drone_head",
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [None, EMBEDDING_DIM])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, [None, 2])],
        [numpy_helper.from_array(w, "W"), numpy_helper.from_array(np.zeros(2, np.float32), "B")],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    m.ir_version = 9
    path = tmp_path / "bare.onnx"
    onnx.save(m, str(path))
    clf = DroneHeadClassifier(path, min_confidence=0.5)
    assert clf.classify_embedding(np.ones(EMBEDDING_DIM, dtype=np.float32)).label == "drone"
    assert clf.classify_embedding(-np.ones(EMBEDDING_DIM, dtype=np.float32)).label == "unknown"


def test_classify_embedding_negative_is_unknown(tmp_path):
    model = _write_tiny_head(tmp_path / "drone_head.onnx")
    clf = DroneHeadClassifier(model, min_confidence=0.5)
    result = clf.classify_embedding(-np.ones(EMBEDDING_DIM, dtype=np.float32))
    assert result.label == "unknown"
    assert result.scores["drone"] < 0.5


def test_minority_frame_fraction_still_qualifies(tmp_path):
    model = _write_tiny_head(tmp_path / "drone_head.onnx")
    clf = DroneHeadClassifier(model, min_confidence=0.5)
    frames = np.stack([-np.ones(EMBEDDING_DIM), np.ones(EMBEDDING_DIM)]).astype(np.float32)
    result = clf.classify_embedding(frames)
    assert result.label == "drone"  # 1 of 2 frames >= threshold: fraction 0.5 >= 0.2
    # Reported confidence is the clip-mean probability, not the frame max.
    assert result.confidence == pytest.approx(0.5, abs=0.02)
    assert result.features["frame_count"] == 2.0


def test_single_noisy_frame_does_not_relabel_clip(tmp_path):
    """Production regression: one drone-ish frame in a long ambient clip must
    not flip the label (previously max-over-frames labeled the clip drone)."""
    model = _write_tiny_head(tmp_path / "drone_head.onnx")
    clf = DroneHeadClassifier(model, min_confidence=0.5)
    frames = np.vstack(
        [-np.ones((61, EMBEDDING_DIM)), np.ones((1, EMBEDDING_DIM))]
    ).astype(np.float32)
    result = clf.classify_embedding(frames)
    assert result.label == "unknown"
    assert result.confidence == 0.0
    # Observability retained: the noisy frame is still visible in scores/features.
    assert result.scores["drone"] > 0.5
    assert result.features["drone_prob_max"] > 0.5
    assert result.features["drone_frame_fraction"] < 0.2


def test_frame_fraction_boundary(tmp_path):
    model = _write_tiny_head(tmp_path / "drone_head.onnx")
    clf = DroneHeadClassifier(model, min_confidence=0.5, min_frame_fraction=0.2)
    hot = np.ones((1, EMBEDDING_DIM))
    # 1 of 5 frames = fraction 0.2 -> qualifies at the boundary.
    frames_5 = np.vstack([-np.ones((4, EMBEDDING_DIM)), hot]).astype(np.float32)
    assert clf.classify_embedding(frames_5).label == "drone"
    # 1 of 6 frames = fraction ~0.167 -> below the floor.
    frames_6 = np.vstack([-np.ones((5, EMBEDDING_DIM)), hot]).astype(np.float32)
    assert clf.classify_embedding(frames_6).label == "unknown"


def test_confidence_is_clip_mean(tmp_path):
    model = _write_tiny_head(tmp_path / "drone_head.onnx")
    clf = DroneHeadClassifier(model, min_confidence=0.5)
    frames = np.vstack(
        [-np.ones((2, EMBEDDING_DIM)), np.ones((2, EMBEDDING_DIM))]
    ).astype(np.float32)
    result = clf.classify_embedding(frames)
    assert result.label == "drone"
    assert result.confidence == pytest.approx(result.features["drone_prob_mean"])
    assert result.confidence < result.features["drone_prob_max"]


def test_zero_fraction_still_requires_one_hit(tmp_path):
    model = _write_tiny_head(tmp_path / "drone_head.onnx")
    clf = DroneHeadClassifier(model, min_confidence=0.5, min_frame_fraction=0.0)
    frames = -np.ones((4, EMBEDDING_DIM), dtype=np.float32)
    result = clf.classify_embedding(frames)
    assert result.label == "unknown"
    assert result.confidence == 0.0


def test_missing_model_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        DroneHeadClassifier(tmp_path / "nope.onnx")


def test_preprocessing_fingerprint_mismatch_fails_model_startup(tmp_path):
    model = _write_tiny_head(
        tmp_path / "drone_head.onnx",
        metadata={"preprocessing": {"profile": "yamnet", "fingerprint": "trained-profile"}},
    )
    with pytest.raises(PreprocessingProfileMismatchError, match="preprocessing mismatch"):
        DroneHeadClassifier(model, expected_preprocessing_fingerprint="runtime-profile")


class _StubYamnet(AudioClassifier):
    """Emits per-frame embedding frames like YAMNet with keep_embeddings=True."""

    def __init__(self, frames: np.ndarray) -> None:
        self._frames = frames.astype(np.float32)

    def classify(self, samples, sample_rate_hz):
        return ClassificationResult(
            label="Aircraft",
            confidence=0.6,
            scores={"Aircraft": 0.6},
            features={
                "embedding_frames": self._frames,
                "embedding": self._frames.mean(axis=0),
                "embedding_model": "yamnet",
                "embedding_dim": float(EMBEDDING_DIM),
            },
        )

    def close(self): ...
    def cancel_pending(self): ...


def test_chain_stage_feeds_embedding_frames_and_strips_ndarrays(tmp_path):
    model = _write_tiny_head(tmp_path / "drone_head.onnx")
    drone = DroneHeadClassifier(model, min_confidence=0.5)
    assert isinstance(drone, EmbeddingClassifier)

    frames = np.stack([np.ones(EMBEDDING_DIM), np.ones(EMBEDDING_DIM)]).astype(np.float32)
    chained = ChainedClassifier(
        base_classifier=_StubYamnet(frames),
        stages=[ChainStage(stage_id="drone_head", classifier=drone, input_kind="embedding")],
    )
    result = chained.classify(np.zeros(16000, dtype=np.float32), 16000)

    assert result.label == "drone"
    assert result.features["winner_member"] == "drone_head"
    assert result.scores["drone_head:drone"] > 0.5
    # ndarrays must never survive into the feature summary (JSON hygiene).
    assert "embedding" not in result.features
    assert "embedding_frames" not in result.features
    json.dumps(result.features)  # must not raise


def test_ambient_dominant_frames_with_subthreshold_drone_do_not_relabel(tmp_path):
    """2026-08-01 live regression shape: every frame carries a drone probability
    above the per-class threshold, but ambient WINS every frame. The old vote
    counted any frame with P(drone) >= threshold — never asking whether ambient
    won it — so the clip was labeled drone. The argmax vote must say unknown."""
    model = _write_tiny_head(
        tmp_path / "drone_head.onnx", labels=LABELS_3, metadata={"negative_label": "ambient"}
    )
    clf = DroneHeadClassifier(model, min_confidence=0.3, min_frame_fraction=0.2)
    # Blend: mostly ambient slice with a real drone-slice component on EVERY
    # frame -> P(ambient) ≈ 0.57 > P(drone) ≈ 0.38 > 0.3 per frame.
    blended = (0.6 * _slice_embedding(0, 3) + 0.5 * _slice_embedding(1, 3)).astype(np.float32)
    frames = np.stack([blended] * 5)
    probs_check = clf.classify_embedding(frames)
    assert probs_check.features["drone_prob_max"] >= 0.3  # would have qualified before
    assert probs_check.label == "unknown"
    assert probs_check.confidence == 0.0


def test_ambient_margin_gate_requires_mean_lead(tmp_path):
    """With ambient_margin set, a legitimate minority burst must also beat the
    ambient clip mean by the margin; without the margin it labels as before."""
    model = _write_tiny_head(tmp_path / "drone_head.onnx")
    frames = np.vstack([-np.ones((4, EMBEDDING_DIM)), np.ones((1, EMBEDDING_DIM))]).astype(
        np.float32
    )
    permissive = DroneHeadClassifier(model, min_confidence=0.5, min_frame_fraction=0.2)
    assert permissive.classify_embedding(frames).label == "drone"
    strict = DroneHeadClassifier(
        model, min_confidence=0.5, min_frame_fraction=0.2, ambient_margin=0.1
    )
    result = strict.classify_embedding(frames)
    assert result.label == "unknown"
    assert result.confidence == 0.0


def test_min_mean_confidence_floor(tmp_path):
    """min_mean_confidence floors the reported clip mean: a winner whose mean
    sits below it collapses to unknown/0.0 instead of labeling at e.g. 0.4."""
    model = _write_tiny_head(tmp_path / "drone_head.onnx")
    frames = np.vstack([-np.ones((1, EMBEDDING_DIM)), np.ones((1, EMBEDDING_DIM))]).astype(
        np.float32
    )
    floored = DroneHeadClassifier(model, min_confidence=0.5, min_mean_confidence=0.6)
    result = floored.classify_embedding(frames)
    assert result.label == "unknown"
    assert result.confidence == 0.0
    # Observability retained even when floored.
    assert result.scores["drone"] > 0.5
