"""Fast baseline heuristic classifier for MVP."""

from __future__ import annotations

import numpy as np

from minimappr.classifiers.base import AudioClassifier
from minimappr.models import ClassificationResult
from minimappr.utils.audio import spectral_features


class HeuristicClassifier(AudioClassifier):
    def __init__(
        self,
        *,
        ambient_rms_threshold: float = 0.01,
        impulse_crest_threshold: float = 10.0,
        impulse_bandwidth_threshold_hz: float = 1200.0,
        bird_centroid_min_hz: float = 2200.0,
        bird_zcr_min: float = 0.12,
        speech_centroid_min_hz: float = 200.0,
        speech_centroid_max_hz: float = 2200.0,
        speech_zcr_min: float = 0.04,
        speech_zcr_max: float = 0.2,
        speech_flatness_max: float = 0.75,
        machine_centroid_max_hz: float = 450.0,
        machine_flatness_max: float = 0.55,
        unknown_min_score: float = 0.2,
        unknown_score: float = 0.6,
    ) -> None:
        self.ambient_rms_threshold = float(ambient_rms_threshold)
        self.impulse_crest_threshold = float(impulse_crest_threshold)
        self.impulse_bandwidth_threshold_hz = float(impulse_bandwidth_threshold_hz)
        self.bird_centroid_min_hz = float(bird_centroid_min_hz)
        self.bird_zcr_min = float(bird_zcr_min)
        self.speech_centroid_min_hz = float(speech_centroid_min_hz)
        self.speech_centroid_max_hz = float(speech_centroid_max_hz)
        self.speech_zcr_min = float(speech_zcr_min)
        self.speech_zcr_max = float(speech_zcr_max)
        self.speech_flatness_max = float(speech_flatness_max)
        self.machine_centroid_max_hz = float(machine_centroid_max_hz)
        self.machine_flatness_max = float(machine_flatness_max)
        self.unknown_min_score = float(unknown_min_score)
        self.unknown_score = float(unknown_score)

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

        if rms_value < self.ambient_rms_threshold:
            scores["ambient"] = 0.95
        else:
            if crest > self.impulse_crest_threshold and bandwidth > self.impulse_bandwidth_threshold_hz:
                scores["impulse"] = min(0.95, 0.35 + (crest / 25.0))

            if centroid > self.bird_centroid_min_hz and zcr > self.bird_zcr_min:
                scores["bird_like"] = min(0.95, 0.25 + (centroid / 8000.0))

            if (
                self.speech_centroid_min_hz < centroid < self.speech_centroid_max_hz
                and self.speech_zcr_min < zcr < self.speech_zcr_max
                and flatness < self.speech_flatness_max
            ):
                scores["speech_like"] = min(0.9, 0.25 + (1.0 - abs(centroid - 1000.0) / 1200.0))

            if centroid < self.machine_centroid_max_hz and flatness < self.machine_flatness_max:
                scores["machine_hum"] = min(0.9, 0.3 + (self.machine_centroid_max_hz - centroid) / 700.0)

            if all(v < self.unknown_min_score for v in scores.values()):
                scores["unknown"] = self.unknown_score

        label, confidence = max(scores.items(), key=lambda item: item[1])

        return ClassificationResult(
            label=label,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            scores=scores,
            features=features,
        )
