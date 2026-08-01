"""Drone / wildlife detection head on YAMNet embeddings (int8 ONNX, onnxruntime CPU).

Trained by ``scripts/train_drone_head.py``; an N-class softmax over 1024-d YAMNet
embedding frames. The first positive class is ``drone`` (index 1 by convention),
with a dedicated *negative* class (``ambient``/``no_drone``) that maps to the
runtime ``"unknown"`` label. Embedding-only: it runs as a routing chain stage
with ``input: "embedding"``, never on raw audio.

Clip-level decision: a positive class qualifies only when at least one frame
reaches ``min_confidence`` AND the fraction of such frames meets
``min_frame_fraction``; the winner among qualified classes is the one with the
highest clip-mean probability, reported with that mean as the confidence. A
single noisy frame in a long clip therefore cannot flip the label, and on
single-frame clips the fraction degenerates to 0/1 so one confident frame
still qualifies. Per-class clip maxes stay in ``scores``/``features`` for
observability.

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


class PreprocessingProfileMismatchError(RuntimeError):
    """The embedding head was trained with a different waveform profile."""


class DroneHeadClassifier(EmbeddingClassifier):
    def __init__(
        self,
        model_path: str | Path = "data/models/drone_head.onnx",
        *,
        min_confidence: float = 0.5,
        min_frame_fraction: float = 0.2,
        ambient_margin: float = 0.0,
        min_mean_confidence: float = 0.0,
        expected_preprocessing_fingerprint: str | None = None,
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
        self._min_frame_fraction = float(np.clip(min_frame_fraction, 0.0, 1.0))
        # Extra qualification gates on top of the frame vote (both default off):
        # a positive class must beat the negative class's clip mean by this
        # margin, and the reported clip mean must clear the floor to label.
        self._ambient_margin = float(np.clip(ambient_margin, 0.0, 1.0))
        self._min_mean_confidence = float(np.clip(min_mean_confidence, 0.0, 1.0))
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

        trained_fingerprint = (self._metadata.get("preprocessing") or {}).get("fingerprint")
        if (
            expected_preprocessing_fingerprint
            and trained_fingerprint
            and trained_fingerprint != expected_preprocessing_fingerprint
        ):
            raise PreprocessingProfileMismatchError(
                "Drone head preprocessing mismatch: model expects "
                f"{trained_fingerprint}, runtime resolved {expected_preprocessing_fingerprint}"
            )

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
        self._negative_index = lowered.index(self._negative_label)

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

        clip_max = frame_matrix.max(axis=0)
        clip_mean = frame_matrix.mean(axis=0)
        n_frames = frame_matrix.shape[0]

        # A positive class qualifies only when the fraction of frames it WINS
        # (argmax across all classes, negative included) while also reaching the
        # per-frame threshold meets min_frame_fraction; winner = qualified class
        # with highest clip mean. The argmax condition is what keeps an
        # ambient-dominant clip from relabeling: a 2026-08-01 live clip with
        # ambient 0.969 AND drone 0.835 (maxes over different frames) was
        # labeled "drone" because the old vote counted any frame with
        # P(drone) >= threshold, never asking whether ambient won that frame.
        frame_argmax = frame_matrix.argmax(axis=1)
        frame_fractions = {
            i: float(
                ((frame_argmax == i) & (frame_matrix[:, i] >= self._min_confidence)).sum()
            )
            / n_frames
            for i in self._positive_indices
        }
        # Optional ambient-margin gate: only engages when a margin is set (0.0 =
        # off). A qualified minority burst — e.g. 1 hot frame of 5 — is a
        # legitimate label even though the clip MEAN is ambient-dominated, so
        # mean-vs-mean comparison must stay opt-in.
        negative_mean = float(clip_mean[self._negative_index])
        qualified = [
            i
            for i in self._positive_indices
            if frame_fractions[i] > 0.0
            and frame_fractions[i] >= self._min_frame_fraction
            and (
                self._ambient_margin <= 0.0
                or float(clip_mean[i]) >= negative_mean + self._ambient_margin
            )
        ]
        if qualified:
            winner = max(qualified, key=lambda i: (float(clip_mean[i]), -i))
            label = str(self._labels[winner])
            best_score = float(clip_mean[winner])
            # Optional floor: today best_score (a clip MEAN) can sit below the
            # per-frame min_confidence with no recourse; 0.0 keeps legacy shape.
            if best_score < self._min_mean_confidence:
                label = "unknown"
                best_score = 0.0
        else:
            label = "unknown"
            best_score = 0.0

        # scores = per-class clip maxes for every label (drop the 1-p construction).
        scores = {str(self._labels[i]): float(clip_max[i]) for i in range(len(self._labels))}

        features: dict = {
            "model": "drone_head",
            "frame_count": float(len(frame_prob_rows)),
            "min_frame_fraction": self._min_frame_fraction,
        }
        # Back-compat drone_prob_* keys (fall back to 0.0 if no drone class).
        if "drone" in [name.lower() for name in self._labels]:
            drone_idx = [name.lower() for name in self._labels].index("drone")
            features["drone_prob_max"] = float(clip_max[drone_idx])
            features["drone_prob_mean"] = float(clip_mean[drone_idx])
        else:
            features["drone_prob_max"] = 0.0
            features["drone_prob_mean"] = 0.0
        # Per positive class max/mean/fraction feature keys.
        for i in self._positive_indices:
            name = str(self._labels[i])
            features[f"{name}_prob_max"] = float(clip_max[i])
            features[f"{name}_prob_mean"] = float(clip_mean[i])
            features[f"{name}_frame_fraction"] = frame_fractions[i]

        return ClassificationResult(
            label=label,
            confidence=best_score,
            scores=scores,
            features=features,
        )


__all__ = ["DroneHeadClassifier", "PreprocessingProfileMismatchError"]
