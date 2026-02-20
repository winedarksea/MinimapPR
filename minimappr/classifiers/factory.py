"""Classifier backend selection."""

from __future__ import annotations

import logging

from minimappr.classifiers.base import AudioClassifier
from minimappr.classifiers.heuristic import HeuristicClassifier
from minimappr.classifiers.yamnet import YAMNetClassifier
from minimappr.config import Settings


logger = logging.getLogger(__name__)


def create_classifier(settings: Settings) -> AudioClassifier:
    backend = settings.classifier_backend.strip().lower()
    if backend == "yamnet":
        try:
            return YAMNetClassifier(min_confidence=settings.yamnet_min_confidence)
        except Exception as exc:  # pragma: no cover - optional runtime backend
            logger.warning("YAMNet unavailable (%s). Falling back to heuristic classifier.", exc)
            return HeuristicClassifier()

    return HeuristicClassifier()
