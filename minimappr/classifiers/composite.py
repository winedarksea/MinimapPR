"""Composite classifier: run several independent members on the same audio.

This is how "YAMNet always runs and BirdNET always runs, not chained" is
expressed — each context's ``run`` list becomes one CompositeClassifier whose
members execute sequentially inside the single worker-thread ``classify()``
call the orchestrator already makes. Scores are namespaced
``{member_id}:{key}`` (the primary/first member stays unprefixed for backward
compatibility); the winner is the member result with the highest weighted
confidence and its id is recorded as ``features["winner_member"]``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from minimappr.classifiers.base import AudioClassifier
from minimappr.models import ClassificationResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CompositeMember:
    member_id: str
    classifier: AudioClassifier
    score_weight: float = 1.0


class CompositeClassifier(AudioClassifier):
    def __init__(self, members: list[CompositeMember]) -> None:
        if not members:
            raise ValueError("CompositeClassifier requires at least one member")
        self._members = members

    @property
    def members(self) -> list[CompositeMember]:
        return self._members

    def close(self) -> None:
        for member in self._members:
            member.classifier.close()

    def cancel_pending(self) -> None:
        for member in self._members:
            member.classifier.cancel_pending()

    def classify(self, samples: np.ndarray, sample_rate_hz: int) -> ClassificationResult:
        combined_scores: dict[str, float] = {}
        feature_summary: dict[str, object] = {}
        ensemble: list[dict[str, object]] = []
        primary_id = self._members[0].member_id
        first_result: ClassificationResult | None = None
        first_member_id = primary_id
        winner_label: str | None = None
        winner_conf = 0.0
        winner_member = primary_id

        for member in self._members:
            try:
                result = member.classifier.classify(samples, sample_rate_hz)
            except Exception:  # noqa: BLE001 - one broken member must not sink the rest
                logger.exception("Composite member %s failed; skipping", member.member_id)
                continue
            prefix = "" if member.member_id == primary_id else f"{member.member_id}:"
            for key, value in result.scores.items():
                combined_scores[f"{prefix}{key}"] = float(value)
            for key, value in result.features.items():
                feature_summary[f"{prefix}{key}"] = value
            ensemble.append(
                {
                    "member_id": member.member_id,
                    "label": result.label,
                    "confidence": float(result.confidence),
                }
            )
            if first_result is None:
                first_result = result
                first_member_id = member.member_id
            weighted_conf = float(result.confidence) * member.score_weight
            if result.label not in ("", "unknown") and (
                winner_label is None or weighted_conf > winner_conf
            ):
                winner_conf = weighted_conf
                winner_label = result.label
                winner_member = member.member_id

        if winner_label is None:
            # All-unknown ensemble: report the first successful member's result.
            winner_label = first_result.label if first_result is not None else "unknown"
            winner_conf = float(first_result.confidence) if first_result is not None else 0.0
            winner_member = first_member_id

        # ndarrays (embeddings) must never reach feature_summary_json.
        for key in [k for k in feature_summary if k.endswith(("embedding", "embedding_frames"))]:
            feature_summary.pop(key, None)

        feature_summary["ensemble"] = ensemble
        feature_summary["winner_member"] = winner_member
        return ClassificationResult(
            label=winner_label,
            confidence=float(np.clip(winner_conf, 0.0, 1.0)),
            scores=combined_scores,
            features=feature_summary,
        )


__all__ = ["CompositeClassifier", "CompositeMember"]
