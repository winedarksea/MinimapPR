"""Per-track species belief as capped log-odds (Field Mode W2-tracker, S3).

A track used to carry the label of whichever detection hit it last. For an
intermittent songbird that means one weak or wrong bout overwrites three good
ones. Here each track keeps an independent log-odds per label, fused across
detections, so three bouts scoring 0.30, 0.45 and 0.40 from one perch add up to
a confident identification, and one bad bout cannot flip it.

Update rule (independent-evidence log-odds, one Bernoulli per label):

    L(label) += clip(logit(p) - logit(prior), -per_update_cap, +per_update_cap)
    L(label)  = clip(L(label), logit(prior) - total_cap, logit(prior) + total_cap)

with ``L`` initialised to ``logit(prior)``. A single detection scoring ``p``
(within the per-update cap) therefore gives belief ``p`` exactly, so a display
threshold on belief means the same thing as a threshold on one detection.
Labels a detection does not score are left untouched (absence of a score is not
evidence of absence: classifiers only report their top-k).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Clamp keeps logit finite for scores of exactly 0.0 or 1.0.
_PROBABILITY_EPSILON = 1e-6
# Labels that carry no species information and never enter a belief.
_NON_EVIDENCE_LABELS = frozenset({"", "unknown", "background", "silence"})


def probability_to_log_odds(probability: float) -> float:
    clamped = min(max(float(probability), _PROBABILITY_EPSILON), 1.0 - _PROBABILITY_EPSILON)
    return math.log(clamped / (1.0 - clamped))


def log_odds_to_probability(log_odds: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(log_odds)))


@dataclass(frozen=True, slots=True)
class SpeciesBeliefConfig:
    """Tuning for ``SpeciesLogOddsBelief`` (all from ``TrackingConfig``)."""

    prior_probability: float = 0.1
    per_update_log_odds_cap: float = 3.0
    total_log_odds_cap: float = 6.0
    evidence_top_k: int = 3
    display_min_belief: float = 0.5
    exposed_top_n: int = 5

    def __post_init__(self) -> None:
        if not 0.0 < self.prior_probability < 1.0:
            raise ValueError("species belief prior_probability must be in (0, 1)")
        if self.per_update_log_odds_cap <= 0.0 or self.total_log_odds_cap <= 0.0:
            raise ValueError("species belief log-odds caps must be > 0")
        if self.evidence_top_k < 1 or self.exposed_top_n < 1:
            raise ValueError("species belief top-k / top-n must be >= 1")
        if not 0.0 <= self.display_min_belief <= 1.0:
            raise ValueError("species belief display_min_belief must be in [0, 1]")


class SpeciesLogOddsBelief:
    """Running per-label log-odds for one track."""

    __slots__ = ("_config", "_prior_log_odds", "_log_odds_by_label")

    def __init__(self, config: SpeciesBeliefConfig) -> None:
        self._config = config
        self._prior_log_odds = probability_to_log_odds(config.prior_probability)
        self._log_odds_by_label: dict[str, float] = {}

    def update(self, evidence_probability_by_label: dict[str, float]) -> None:
        cap = self._config.per_update_log_odds_cap
        floor = self._prior_log_odds - self._config.total_log_odds_cap
        ceiling = self._prior_log_odds + self._config.total_log_odds_cap
        for label, probability in evidence_probability_by_label.items():
            # isnan (not isfinite) for parity with the Rust port: a NaN score
            # would otherwise poison this label's log-odds permanently.
            if label.strip().lower() in _NON_EVIDENCE_LABELS or math.isnan(probability):
                continue
            delta = probability_to_log_odds(probability) - self._prior_log_odds
            delta = min(max(delta, -cap), cap)
            current = self._log_odds_by_label.get(label, self._prior_log_odds)
            self._log_odds_by_label[label] = min(max(current + delta, floor), ceiling)

    def probability_for(self, label: str) -> float:
        return log_odds_to_probability(self._log_odds_by_label.get(label, self._prior_log_odds))

    def top_label(self) -> tuple[str | None, float]:
        """(label, probability) of the most-believed label; ``(None, 0.0)`` if empty."""
        if not self._log_odds_by_label:
            return None, 0.0
        label, log_odds = max(self._log_odds_by_label.items(), key=lambda item: item[1])
        return label, log_odds_to_probability(log_odds)

    def top_labels(self, count: int) -> list[str]:
        ranked = sorted(self._log_odds_by_label.items(), key=lambda item: item[1], reverse=True)
        return [label for label, _ in ranked[: max(0, count)]]

    def exposed_probabilities(self) -> dict[str, float]:
        """Top-N label -> probability, for API/WS exposure."""
        ranked = sorted(self._log_odds_by_label.items(), key=lambda item: item[1], reverse=True)
        return {
            label: round(log_odds_to_probability(log_odds), 4)
            for label, log_odds in ranked[: self._config.exposed_top_n]
        }

    def passes_display_threshold(self) -> bool:
        _, probability = self.top_label()
        return probability >= self._config.display_min_belief

    def copy(self) -> "SpeciesLogOddsBelief":
        duplicate = SpeciesLogOddsBelief(self._config)
        duplicate._log_odds_by_label = dict(self._log_odds_by_label)
        return duplicate


def species_evidence_from_detection(
    *,
    label: str,
    confidence: float,
    classifier_scores: dict[str, float] | None,
    winner_member: str | None,
    top_k: int,
) -> dict[str, float]:
    """Label -> probability evidence for one detection.

    Only the winning classifier member's scores are used: composite score maps
    mix vocabularies (YAMNet "Animal" next to BirdNET species), and a belief over
    a mixed vocabulary would let a generic class outrank the species. Composite
    keys carry a ``member:`` prefix for every member except the primary one, so
    the winner's keys are the prefixed ones (or the unprefixed ones when the
    winner is the primary member). The detection's own label always counts.
    """
    evidence: dict[str, float] = {}
    if classifier_scores:
        prefix = f"{winner_member}:" if winner_member else ""
        prefixed = {
            key[len(prefix):]: float(value)
            for key, value in classifier_scores.items()
            if prefix and key.startswith(prefix)
        }
        member_scores = prefixed or {
            key: float(value) for key, value in classifier_scores.items() if ":" not in key
        }
        ranked = sorted(member_scores.items(), key=lambda item: item[1], reverse=True)
        evidence.update(dict(ranked[: max(1, top_k)]))
    normalised_label = (label or "").strip()
    if normalised_label and normalised_label.lower() not in _NON_EVIDENCE_LABELS:
        evidence[normalised_label] = max(float(confidence), evidence.get(normalised_label, 0.0))
    return evidence
