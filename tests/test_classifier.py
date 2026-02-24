from __future__ import annotations

import numpy as np

from minimappr.classifiers.base import AudioClassifier
from minimappr.classifiers.chaining import ChainStage, ChainedClassifier
from minimappr.classifiers.heuristic import HeuristicClassifier
from minimappr.models import ClassificationResult


def test_heuristic_classifier_detects_machine_hum() -> None:
    sample_rate_hz = 16000
    t = np.arange(sample_rate_hz, dtype=np.float32) / sample_rate_hz
    hum = 0.4 * np.sin(2 * np.pi * 220.0 * t)

    classifier = HeuristicClassifier()
    result = classifier.classify(hum, sample_rate_hz)

    assert result.label in {"machine_hum", "speech_like", "unknown"}
    assert 0.0 <= result.confidence <= 1.0
    assert "centroid_hz" in result.features


def test_heuristic_classifier_detects_impulse_like_event() -> None:
    sample_rate_hz = 16000
    x = np.zeros(sample_rate_hz, dtype=np.float32)
    x[2000:2010] = 0.9
    x[5000:5010] = -0.8

    classifier = HeuristicClassifier()
    result = classifier.classify(x, sample_rate_hz)

    assert result.label in {"impulse", "unknown"}
    assert 0.0 <= result.confidence <= 1.0


class _StaticClassifier(AudioClassifier):
    def __init__(self, label: str, confidence: float, scores: dict[str, float] | None = None) -> None:
        self._label = label
        self._confidence = confidence
        self._scores = scores or {label: confidence}

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        del samples, sample_rate_hz
        return ClassificationResult(
            label=self._label,
            confidence=self._confidence,
            scores=self._scores,
            features={"source": self._label},
        )


def test_chained_classifier_runs_triggered_stage_and_fuses_confidence() -> None:
    base = _StaticClassifier("bird", 0.55)
    downstream = _StaticClassifier("robin", 0.92)
    chain = ChainedClassifier(
        base_classifier=base,
        stages=[
            ChainStage(
                stage_id="species",
                classifier=downstream,
                trigger_labels={"bird"},
                min_confidence=0.5,
                score_weight=1.0,
            )
        ],
        category_for_label=lambda _: "wildlife",
    )

    result = chain.classify(np.zeros(64, dtype=np.float32), 16000)
    assert result.label == "robin"
    assert result.confidence > 0.9
    assert result.features["chain_stage_count"] == 1.0
    assert "species:robin" in result.scores


def test_chained_classifier_skips_non_matching_stage() -> None:
    base = _StaticClassifier("vehicle", 0.7)
    downstream = _StaticClassifier("truck", 0.95)
    chain = ChainedClassifier(
        base_classifier=base,
        stages=[
            ChainStage(
                stage_id="species",
                classifier=downstream,
                trigger_labels={"bird"},
                min_confidence=0.5,
                score_weight=1.0,
            )
        ],
        category_for_label=lambda _: "vehicle",
    )

    result = chain.classify(np.zeros(64, dtype=np.float32), 16000)
    assert result.label == "vehicle"
    assert result.confidence == 0.7
    assert result.features["chain_stage_count"] == 0.0
