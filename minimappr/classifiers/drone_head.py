"""Drone / wildlife detection head on YAMNet embeddings (int8 ONNX, onnxruntime CPU).

Trained by ``scripts/train_drone_head.py``; an N-class softmax over 1024-d YAMNet
embedding frames. The first positive class is ``drone`` (index 1 by convention),
with a dedicated *negative* class (``ambient``/``no_drone``) that maps to the
runtime ``"unknown"`` label. Embedding-only: it runs as a routing chain stage
with ``input: "embedding"``, never on raw audio.

Legacy binary ``no_drone``/``drone`` models remain supported: labels default to
``_DEFAULT_LABELS`` and the negative class is inferred when metadata is absent.
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
# Labels treated as the negative / non-detection class when metadata does not
# name one explicitly. Lowercased comparison.
_NEGATIVE_LABEL_CANDIDATES = ("ambient", "no_drone")


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
        self._negative_label: str | None = None

        metadata_path = self._model_path.parent / (self._model_path.stem + ".metadata.json")
        if metadata_path.exists():
            try:
                self._metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                labels = self._metadata.get("labels")
                if isinstance(labels, list) and len(labels) >= 2:
                    self._labels = tuple(str(label) for label in labels)
                negative_label = self._metadata.get("negative_label")
                if isinstance(negative_label, str) and negative_label.strip():
                    self._negative_label = negative_label.strip().lower()
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Unable to read drone head metadata %s: %s", metadata_path, exc)

        # Resolve the negative class: explicit metadata wins; otherwise pick the
        # first known negative candidate present in labels, else labels[0].
        lowered = [str(label).strip().lower() for label in self._labels]
        if self._negative_label is None or self._negative_label not in lowered:
            self._negative_label = next(
                (cand for cand in _NEGATIVE_LABEL_CANDIDATES if cand in lowered),
                lowered[0],
            )
        # Positive classes = everything that is not the negative class, in order.
        self._positive_indices = [
            i for i, name in enumerate(lowered) if name != self._negative_label
        ]

    def classify_embedding(self, frames: np.ndarray) -> ClassificationResult:
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim == 1:
            frames = frames[None, :]
        if frames.ndim != 2:
            raise ValueError(f"Drone head expects [1024] or [N,1024] frames, got {frames.shape}")

        # Per-frame full probability vectors -> [n_frames, n_labels].
        frame_prob_rows: list[np.ndarray] = []
        for frame in frames:
            outputs = self._session.run(None, {self._input_name: frame[None, :]})
            probs = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
            frame_prob_rows.append(probs)
        frame_matrix = np.stack(frame_prob_rows)  # [n_frames, n_labels]

        # Clip-level score per class: max over frames (a class audible in any
        # single ~1s frame counts).
        clip_max = frame_matrix.max(axis=0)
        clip_mean = frame_matrix.mean(axis=0)

        # Winner = argmax over the positive classes only.
        best_pos_index = max(self._positive_indices, key=lambda i: float(clip_max[i]))
        best_score = float(clip_max[best_pos_index])
        best_label = str(self._labels[best_pos_index])
        label = best_label if best_score >= self._min_confidence else "unknown"

        # scores = per-class clip maxes for every label (drop the 1-p construction).
        scores = {str(self._labels[i]): float(clip_max[i]) for i in range(len(self._labels))}

        features: dict = {
            "model": "drone_head",
            "frame_count": float(len(frame_prob_rows)),
        }
        # Back-compat drone_prob_* keys (fall back to 0.0 if no drone class).
        if "drone" in [name.lower() for name in self._labels]:
            drone_idx = [name.lower() for name in self._labels].index("drone")
            features["drone_prob_max"] = float(clip_max[drone_idx])
            features["drone_prob_mean"] = float(clip_mean[drone_idx])
        else:
            features["drone_prob_max"] = 0.0
            features["drone_prob_mean"] = 0.0
        # Per positive class max/mean feature keys.
        for i in self._positive_indices:
            name = str(self._labels[i])
            features[f"{name}_prob_max"] = float(clip_max[i])
            features[f"{name}_prob_mean"] = float(clip_mean[i])

        return ClassificationResult(
            label=label,
            confidence=best_score,
            scores=scores,
            features=features,
        )


__all__ = ["DroneHeadClassifier"]
