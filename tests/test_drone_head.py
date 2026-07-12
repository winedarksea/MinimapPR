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
from minimappr.classifiers.drone_head import DroneHeadClassifier
from minimappr.models import ClassificationResult

EMBEDDING_DIM = 1024


def _write_tiny_head(path: Path, labels=("no_drone", "drone")) -> Path:
    """Softmax(Gemm(embedding, W, B)) where drone logit = +mean, no_drone = -mean."""
    from onnx import TensorProto, helper, numpy_helper

    w = np.zeros((EMBEDDING_DIM, 2), dtype=np.float32)
    w[:, 0] = -1.0 / EMBEDDING_DIM  # no_drone
    w[:, 1] = 1.0 / EMBEDDING_DIM   # drone
    b = np.zeros((2,), dtype=np.float32)

    nodes = [
        helper.make_node("Gemm", ["embedding", "W", "B"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["probs"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "drone_head",
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [None, EMBEDDING_DIM])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, [None, 2])],
        [numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "B")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, str(path))

    (path.parent / (path.stem + ".metadata.json")).write_text(
        json.dumps({"labels": list(labels), "input_shape": [1, EMBEDDING_DIM]})
    )
    return path


def test_classify_embedding_positive_is_drone(tmp_path):
    model = _write_tiny_head(tmp_path / "drone_head.onnx")
    clf = DroneHeadClassifier(model, min_confidence=0.5)
    result = clf.classify_embedding(np.ones(EMBEDDING_DIM, dtype=np.float32))
    assert result.label == "drone"
    assert result.confidence > 0.5
    assert abs(result.scores["drone"] + result.scores["no_drone"] - 1.0) < 1e-5


def test_classify_embedding_negative_is_unknown(tmp_path):
    model = _write_tiny_head(tmp_path / "drone_head.onnx")
    clf = DroneHeadClassifier(model, min_confidence=0.5)
    result = clf.classify_embedding(-np.ones(EMBEDDING_DIM, dtype=np.float32))
    assert result.label == "unknown"
    assert result.scores["drone"] < 0.5


def test_clip_prob_is_max_over_frames(tmp_path):
    model = _write_tiny_head(tmp_path / "drone_head.onnx")
    clf = DroneHeadClassifier(model, min_confidence=0.5)
    frames = np.stack([-np.ones(EMBEDDING_DIM), np.ones(EMBEDDING_DIM)]).astype(np.float32)
    result = clf.classify_embedding(frames)
    assert result.label == "drone"  # any drone-audible frame counts
    assert result.features["frame_count"] == 2.0


def test_missing_model_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        DroneHeadClassifier(tmp_path / "nope.onnx")


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

    assert result.scores["drone_head:drone"] > 0.5
    # ndarrays must never survive into the feature summary (JSON hygiene).
    assert "embedding" not in result.features
    assert "embedding_frames" not in result.features
    json.dumps(result.features)  # must not raise
