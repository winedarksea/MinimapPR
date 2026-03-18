"""Optional YAMNet classifier wrapper."""

from __future__ import annotations

import csv
import io
import logging
import urllib.request
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from minimappr.classifiers.base import AudioClassifier
from minimappr.models import ClassificationResult


logger = logging.getLogger(__name__)

# Local fallback path for the class map so the server starts offline after the
# first successful fetch.  Relative to CWD (the project root when run normally).
_CLASS_MAP_CACHE_PATH = Path("data/yamnet_class_map.csv")


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
        # 1. Try the local disk cache written by a previous successful fetch.
        if _CLASS_MAP_CACHE_PATH.exists():
            try:
                text = _CLASS_MAP_CACHE_PATH.read_text(encoding="utf-8")
                names = _parse_class_map_csv(text)
                if names:
                    return names
            except Exception:
                pass  # Fall through to network fetch.

        # 2. Fetch from upstream and persist to disk for future offline starts.
        try:
            with urllib.request.urlopen(self.CLASS_MAP_URL, timeout=10) as response:  # nosec B310
                text = response.read().decode("utf-8")
            names = _parse_class_map_csv(text)
            if names:
                try:
                    _CLASS_MAP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                    _CLASS_MAP_CACHE_PATH.write_text(text, encoding="utf-8")
                except Exception:
                    pass  # Cache write failure is non-fatal.
            return names
        except Exception as exc:
            logger.warning(
                "YAMNet class map unavailable (network: %s; cache: %s). "
                "Class names will fall back to 'class_N' indices.",
                exc,
                _CLASS_MAP_CACHE_PATH,
            )
            return []


def _parse_class_map_csv(text: str) -> list[str]:
    rows = csv.DictReader(io.StringIO(text))
    return [row["display_name"] for row in rows if "display_name" in row]

