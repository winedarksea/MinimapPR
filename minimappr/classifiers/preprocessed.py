"""Per-member audio preprocessing wrapper for classifier routing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minimappr.classifiers.base import AudioClassifier
from minimappr.interfaces import AudioPreprocessor
from minimappr.models import ClassificationResult


@dataclass(slots=True)
class PreprocessedClassifier(AudioClassifier):
    classifier: AudioClassifier
    preprocessor: AudioPreprocessor
    profile_name: str

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        conditioned = self.preprocessor.process(samples, sample_rate_hz)
        return self.classifier.classify(conditioned, sample_rate_hz)

    def close(self) -> None:
        self.classifier.close()

    def cancel_pending(self) -> None:
        self.classifier.cancel_pending()
