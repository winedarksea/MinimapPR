"""Generic model chaining for classifier pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from minimappr.classifiers.base import AudioClassifier
from minimappr.models import ClassificationResult


@dataclass(slots=True)
class ChainStage:
    stage_id: str
    classifier: AudioClassifier
    trigger_labels: set[str] = field(default_factory=set)
    trigger_categories: set[str] = field(default_factory=set)
    min_confidence: float = 0.0
    score_weight: float = 1.0


class ChainedClassifier(AudioClassifier):
    """Runs a base classifier and optional downstream stages."""

    def __init__(
        self,
        base_classifier: AudioClassifier,
        stages: list[ChainStage] | None = None,
        *,
        category_for_label=None,
    ) -> None:
        self._base = base_classifier
        self._stages = stages or []
        self._category_for_label = category_for_label or (lambda _: "unknown")

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        base = self._base.classify(samples, sample_rate_hz)
        combined_scores = dict(base.scores)
        feature_summary = dict(base.features)
        chain_meta: list[dict[str, object]] = []
        winner_label = base.label
        winner_conf = float(base.confidence)

        base_category = str(self._category_for_label(base.label)).strip().lower()
        for stage in self._stages:
            if stage.trigger_labels and base.label.strip().lower() not in stage.trigger_labels:
                continue
            if stage.trigger_categories and base_category not in stage.trigger_categories:
                continue
            if base.confidence < stage.min_confidence:
                continue

            result = stage.classifier.classify(samples, sample_rate_hz)
            chain_meta.append(
                {
                    "stage_id": stage.stage_id,
                    "label": result.label,
                    "confidence": result.confidence,
                }
            )
            for key, value in result.scores.items():
                combined_scores[f"{stage.stage_id}:{key}"] = float(value) * stage.score_weight

            weighted_conf = float(result.confidence) * stage.score_weight
            if weighted_conf > winner_conf:
                winner_conf = weighted_conf
                winner_label = result.label
            for key, value in result.features.items():
                feature_summary[f"{stage.stage_id}:{key}"] = float(value) if isinstance(value, (int, float)) else value

        feature_summary["chain_stage_count"] = float(len(chain_meta))
        if chain_meta:
            feature_summary["chain"] = chain_meta
        return ClassificationResult(
            label=winner_label,
            confidence=float(np.clip(winner_conf, 0.0, 1.0)),
            scores=combined_scores,
            features=feature_summary,
        )

