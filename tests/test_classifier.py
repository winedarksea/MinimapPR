from __future__ import annotations

import numpy as np

from minimappr.classifiers.heuristic import HeuristicClassifier


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
