"""Classifier interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from minimappr.models import ClassificationResult


class AudioClassifier(ABC):
    @abstractmethod
    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        raise NotImplementedError
