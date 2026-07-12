"""Drone detection head on YAMNet embeddings (int8 ONNX, onnxruntime CPU).

Trained by ``scripts/train_drone_head.py``; binary ``no_drone``/``drone``
softmax over a single 1024-d YAMNet embedding frame. Embedding-only: it runs
as a routing chain stage with ``input: "embedding"``, never on raw audio.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from minimappr.classifiers.base import EmbeddingClassifier
from minimappr.models import ClassificationResult

logger = logging.getLogger(__name__)

_DEFAULT_LABELS = ("no_drone", "drone")


class DroneHeadClassifier(EmbeddingClassifier):
    def __init__(
        self,
        model_path: str | Path = "data/models/drone_head.onnx",
        *,
        min_confidence: float = 0.5,
    ) -> None:
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Drone head requires onnxruntime") from exc

        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(f"Drone head model not found: {self._model_path}")

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(self._model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._min_confidence = float(min_confidence)
        self._labels = _DEFAULT_LABELS
        self._metadata: dict = {}

        metadata_path = self._model_path.parent / (self._model_path.stem + ".metadata.json")
        if metadata_path.exists():
            try:
                self._metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                labels = self._metadata.get("labels")
                if isinstance(labels, list) and len(labels) >= 2:
                    self._labels = tuple(str(label) for label in labels)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Unable to read drone head metadata %s: %s", metadata_path, exc)

    def classify_embedding(self, frames: np.ndarray) -> ClassificationResult:
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim == 1:
            frames = frames[None, :]
        if frames.ndim != 2:
            raise ValueError(f"Drone head expects [1024] or [N,1024] frames, got {frames.shape}")

        drone_index = self._labels.index("drone") if "drone" in self._labels else 1
        frame_probs: list[float] = []
        for frame in frames:
            outputs = self._session.run(None, {self._input_name: frame[None, :]})
            probs = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
            frame_probs.append(float(probs[drone_index]))

        # Clip-level probability: max over frames (a drone audible in any
        # single ~1s frame counts).
        drone_prob = max(frame_probs)
        label = "drone" if drone_prob >= self._min_confidence else "unknown"
        return ClassificationResult(
            label=label,
            confidence=drone_prob,
            scores={"drone": drone_prob, "no_drone": 1.0 - drone_prob},
            features={
                "model": "drone_head",
                "frame_count": float(len(frame_probs)),
                "drone_prob_max": drone_prob,
                "drone_prob_mean": float(np.mean(frame_probs)),
            },
        )


__all__ = ["DroneHeadClassifier"]
