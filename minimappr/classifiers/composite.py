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


# Labels that mean "nothing recognised" rather than naming a class.
_ABSTENTION_LABELS = frozenset({"", "unknown"})

# Models whose abstention still carries a usable ranking. YAMNet scores the
# whole AudioSet ontology and reports "unknown" only because its top class fell
# under the configured threshold, so that top class remains the best available
# description of the sound. BirdNET's vocabulary is species-only, so its
# sub-threshold ranking is noise about which bird it *isn't*, not a weaker
# answer — surfacing it would invent wildlife out of ambient audio.
_ABSTENTION_RANKING_MODELS = frozenset({"yamnet"})


def _recover_abstained_label(
    abstained: list[tuple[str, ClassificationResult]],
) -> tuple[float, str, str, ClassificationResult] | None:
    """Best sub-threshold label from a member that ranks the whole sound space.

    Returns ``(confidence, label, member_id, result)`` matching the winner
    tuple, or ``None`` when no abstaining member qualifies. The recovered
    confidence is the member's own score for that class, so it stays honestly
    low and downstream confidence gates still apply.
    """
    best: tuple[float, str, str, ClassificationResult] | None = None
    for member_id, result in abstained:
        model = result.features.get("model")
        if not isinstance(model, str):
            continue
        if model.strip().lower() not in _ABSTENTION_RANKING_MODELS:
            continue
        for label, score in result.scores.items():
            if label.strip().lower() in _ABSTENTION_LABELS:
                continue
            # ChainedClassifier merges its stages' scores under "{stage_id}:",
            # so a chained member's dict also carries e.g. "drone_head:ambient"
            # -- an internal negative class, not a description of the sound.
            # Only the base model's own (unprefixed) vocabulary is a label.
            if ":" in label:
                continue
            if best is None or float(score) > best[0]:
                best = (float(score), label, member_id, result)
    return best


@dataclass(slots=True)
class CompositeMember:
    member_id: str
    classifier: AudioClassifier
    score_weight: float = 1.0


class CompositeClassifier(AudioClassifier):
    def __init__(
        self,
        members: list[CompositeMember],
        *,
        priority_labels: frozenset[str] = frozenset(),
    ) -> None:
        if not members:
            raise ValueError("CompositeClassifier requires at least one member")
        self._members = members
        # Labels that must surface even when another member scores higher.
        # A plain highest-confidence winner would let a loud, confident bird
        # call in the same omni window mask a safety-critical alarm cadence,
        # silently dropping the detection the alerting rules key off. Any
        # member emitting one of these labels is promoted to winner (highest
        # confidence among them if several); empty set = pure winner-take-all.
        self._priority_labels = priority_labels

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
        winner_result: ClassificationResult | None = None
        # (conf, label, member_id, result)
        priority_best: tuple[float, str, str, ClassificationResult] | None = None
        abstained: list[tuple[str, ClassificationResult]] = []

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
            if result.label in _ABSTENTION_LABELS:
                abstained.append((member.member_id, result))
            weighted_conf = float(result.confidence) * member.score_weight
            if result.label not in _ABSTENTION_LABELS and (
                winner_label is None or weighted_conf > winner_conf
            ):
                winner_conf = weighted_conf
                winner_label = result.label
                winner_member = member.member_id
                winner_result = result
            if result.label in self._priority_labels and (
                priority_best is None or float(result.confidence) > priority_best[0]
            ):
                priority_best = (float(result.confidence), result.label, member.member_id, result)

        if priority_best is not None:
            # A safety-critical label was detected: surface it regardless of a
            # higher-scoring non-priority member (see _priority_labels).
            winner_conf, winner_label, winner_member, winner_result = priority_best
        elif winner_label is None:
            # All-unknown ensemble. Prefer a sub-threshold label from a member
            # that ranks the whole sound space (see _ABSTENTION_RANKING_MODELS)
            # over the bare "unknown" sentinel, so a detection still says what
            # the sound most resembled. Falls back to the first successful
            # member's result when no such member abstained.
            recovered = _recover_abstained_label(abstained)
            if recovered is not None:
                winner_conf, winner_label, winner_member, winner_result = recovered
                # Provenance: consumers must be able to tell a recovered guess
                # from a member that actually cleared its threshold.
                feature_summary["abstention_recovered_from"] = winner_member
            else:
                winner_label = first_result.label if first_result is not None else "unknown"
                winner_conf = float(first_result.confidence) if first_result is not None else 0.0
                winner_member = first_member_id
                winner_result = first_result

        # ndarrays (embeddings) must never reach feature_summary_json.
        for key in [k for k in feature_summary if k.endswith(("embedding", "embedding_frames"))]:
            feature_summary.pop(key, None)

        feature_summary["ensemble"] = ensemble
        # A chained member that won via one of its stages carries the stage id
        # in its own winner_member feature; that attribution must survive so
        # classifier_source (and e.g. drone-head audio retention) is correct.
        nested_winner = winner_result.features.get("winner_member") if winner_result else None
        feature_summary["winner_member"] = str(nested_winner or winner_member)
        return ClassificationResult(
            label=winner_label,
            confidence=float(np.clip(winner_conf, 0.0, 1.0)),
            scores=combined_scores,
            features=feature_summary,
        )


__all__ = ["CompositeClassifier", "CompositeMember"]
