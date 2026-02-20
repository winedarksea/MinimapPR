"""Fast baseline heuristic classifier for MVP."""

from __future__ import annotations

import numpy as np

from minimappr.classifiers.base import AudioClassifier
from minimappr.models import ClassificationResult
from minimappr.utils.audio import spectral_features


class HeuristicClassifier(AudioClassifier):
    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        features = spectral_features(samples=samples, sample_rate_hz=sample_rate_hz)

        rms_value = features["rms"]
        centroid = features["centroid_hz"]
        crest = features["crest"]
        zcr = features["zcr"]
        flatness = features["flatness"]
        bandwidth = features["bandwidth_hz"]

        scores: dict[str, float] = {
            "ambient": 0.0,
            "bird_like": 0.0,
            "speech_like": 0.0,
            "impulse": 0.0,
            "machine_hum": 0.0,
            "unknown": 0.0,
        }

        if rms_value < 0.01:
            scores["ambient"] = 0.95
        else:
            if crest > 10.0 and bandwidth > 1200.0:
                scores["impulse"] = min(0.95, 0.35 + (crest / 25.0))

            if centroid > 2200.0 and zcr > 0.12:
                scores["bird_like"] = min(0.95, 0.25 + (centroid / 8000.0))

            if 200.0 < centroid < 2200.0 and 0.04 < zcr < 0.2 and flatness < 0.75:
                scores["speech_like"] = min(0.9, 0.25 + (1.0 - abs(centroid - 1000.0) / 1200.0))

            if centroid < 450.0 and flatness < 0.55:
                scores["machine_hum"] = min(0.9, 0.3 + (450.0 - centroid) / 700.0)

            if all(v < 0.2 for v in scores.values()):
                scores["unknown"] = 0.6

        label, confidence = max(scores.items(), key=lambda item: item[1])

        return ClassificationResult(
            label=label,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            scores=scores,
            features=features,
        )
