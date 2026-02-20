"""Optional YAMNet classifier wrapper."""

from __future__ import annotations

import csv
import io
import urllib.request

import numpy as np
from scipy.signal import resample_poly

from minimappr.classifiers.base import AudioClassifier
from minimappr.models import ClassificationResult


class YAMNetClassifier(AudioClassifier):
    CLASS_MAP_URL = (
        "https://raw.githubusercontent.com/tensorflow/models/master/"
        "research/audioset/yamnet/yamnet_class_map.csv"
    )

    def __init__(self, min_confidence: float = 0.25) -> None:
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("YAMNet backend requires tensorflow and tensorflow-hub") from exc

        self._tf = tf
        self._hub = hub
        self._model = hub.load("https://tfhub.dev/google/yamnet/1")
        self._class_names = self._load_class_names()
        self._min_confidence = min_confidence

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        waveform = samples.astype(np.float32)
        if sample_rate_hz != 16000:
            waveform = resample_poly(waveform, up=16000, down=sample_rate_hz).astype(np.float32)

        scores, _, _ = self._model(waveform)
        mean_scores = np.mean(scores.numpy(), axis=0)
        top_idx = int(np.argmax(mean_scores))
        top_conf = float(mean_scores[top_idx])

        label = self._class_name(top_idx)
        if top_conf < self._min_confidence:
            label = "unknown"

        top_k = np.argsort(mean_scores)[-5:][::-1]
        scores_map = {self._class_name(int(index)): float(mean_scores[int(index)]) for index in top_k}

        return ClassificationResult(
            label=label,
            confidence=max(0.0, min(1.0, top_conf)),
            scores=scores_map,
            features={"model": "yamnet"},
        )

    def _class_name(self, index: int) -> str:
        if index < len(self._class_names):
            return self._class_names[index]
        return f"class_{index}"

    def _load_class_names(self) -> list[str]:
        try:
            with urllib.request.urlopen(self.CLASS_MAP_URL, timeout=10) as response:  # nosec B310
                text = response.read().decode("utf-8")
        except Exception:
            return []

        rows = csv.DictReader(io.StringIO(text))
        return [row["display_name"] for row in rows if "display_name" in row]
