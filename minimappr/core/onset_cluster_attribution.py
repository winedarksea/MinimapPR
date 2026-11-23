"""Which classifier labels belong to which onset cluster (Field Mode W3-attribution, S5).

A cluster's classification window (on its nearest/classifying node) can also
hold other birds. The plan's fast-to-slow cascade, server flavour:

1. **SNR gradient (no inference).** Per-node noise-floor-relative band SNR of
   every cluster in the window. If the target out-reads every in-window
   interferer on the classifying node by ``dominance_margin_db`` *and* that
   dominance agrees with the clusters' positions (spreading-loss prediction
   within ``spreading_tolerance_db``), the classifier mostly heard the target:
   its labels are the target's.
2. **LOCO.** Otherwise mask the interferers' onset regions with noise at the
   floor and rerun the classifier. The rerun's top label belongs to the target
   when it is confident and clear of the runner-up; a label that the masking
   made collapse belonged to an interferer.
3. **Unsettled -> unlabelled.** No guess: the position stands, no species.

Pure decision functions; the async glue lives in onset_cluster_attribution_stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from minimappr.core.onset_cluster_attribution_evidence import (
    DECIBELS_PER_AMPLITUDE_DECADE,
    MINIMUM_SPREADING_DISTANCE_M,
    ClusterAttributionEvidence,
)
from minimappr.core.onset_cluster_echo_checks import SAME_BAND_MIN_OVERLAP_FRACTION
from minimappr.core.taxonomy import ABSTENTION_LABELS

ATTRIBUTION_TIER_SINGLE_CLUSTER = "single_cluster"
ATTRIBUTION_TIER_SNR_GRADIENT = "snr_gradient"
ATTRIBUTION_TIER_LOCO = "loco"
ATTRIBUTION_TIER_UNSETTLED = "unsettled"
# Labels that say "no species here"; they can never settle a cluster's species.
NON_SPECIES_LABELS = ABSTENTION_LABELS | frozenset({"background", "silence", "ambient", "no_drone"})
# onset_cluster_echo_checks status reason for a single-array cluster with too few independent onsets.
PERSISTENCE_HOLD_REASON = "not_persistent"


@dataclass(frozen=True, slots=True)
class AttributionThresholds:
    audible_snr_db: float
    dominance_margin_db: float
    spreading_tolerance_db: float
    loco_min_confidence: float
    loco_label_margin: float
    loco_collapse_ratio: float


@dataclass(frozen=True, slots=True)
class ClusterAttributionContext:
    """Everything attribution needs about one cluster's classification window.

    Carried on the LocalizedCandidate from the onset stage to the classifier
    worker. Sample indices are on the analysis-span timeline.
    """

    window_id: str  # analysis window (trigger candidate) shared by sibling clusters
    target_index: int
    cluster_evidence: dict[int, ClusterAttributionEvidence]  # non-echo clusters of the window
    classifying_node_id: str
    classification_start_sample: int
    classification_end_sample: int
    sample_rate_hz: int
    # Onsets this close to a target onset (same band) are the target's own note or echo.
    shared_onset_max_lag_samples: int | None = None
    # Products of this window carrying a context (window label reconciliation waits for them).
    sibling_product_count: int = 1


@dataclass(frozen=True, slots=True)
class WindowInterferer:
    cluster_index: int
    evidence: ClusterAttributionEvidence
    onset_seconds_in_window: tuple[tuple[float, float, float], ...]  # (onset_s, band_min, band_max)


@dataclass(frozen=True, slots=True)
class GradientDecision:
    settled: bool
    reason: str
    dominance_db: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LocoDecision:
    settled: bool
    reason: str
    label: str | None = None
    confidence: float = 0.0
    collapsed_labels: tuple[str, ...] = ()


def interferers_in_window(
    *,
    target_index: int,
    cluster_evidence: dict[int, ClusterAttributionEvidence],
    classifying_node_id: str,
    window_start_sample: int,
    window_end_sample: int,
    sample_rate_hz: int,
    audible_snr_db: float,
    lookback_samples: int = 0,
    shared_onset_max_lag_samples: int | None = None,
    same_band_min_overlap_fraction: float = 1.0,
) -> list[WindowInterferer]:
    """Other clusters audible on the classifying node with an onset arriving inside the
    window, or up to ``lookback_samples`` before it (a note still sounding at the start).

    An interferer onset within ``shared_onset_max_lag_samples`` of one of the target's
    own onsets, in the same band, is the target's note (or its reflection) seen again:
    masking it would erase the target, so it is not masked and does not count.
    """
    target_arrivals = cluster_evidence[target_index].node_onset_arrivals.get(classifying_node_id, ())

    def shared_with_target(arrival) -> bool:
        if shared_onset_max_lag_samples is None:
            return False
        return any(
            abs(arrival.span_sample - own.span_sample) <= shared_onset_max_lag_samples
            and _band_overlap_fraction((arrival.band_min_hz, arrival.band_max_hz), (own.band_min_hz, own.band_max_hz))
            >= same_band_min_overlap_fraction
            for own in target_arrivals
        )

    interferers = []
    for index, evidence in sorted(cluster_evidence.items()):
        if index == target_index:
            continue
        if evidence.node_band_snr_db.get(classifying_node_id, float("-inf")) < audible_snr_db:
            continue
        onsets = tuple(
            ((arrival.span_sample - window_start_sample) / sample_rate_hz, arrival.band_min_hz, arrival.band_max_hz)
            for arrival in evidence.node_onset_arrivals.get(classifying_node_id, ())
            if window_start_sample - lookback_samples <= arrival.span_sample < window_end_sample
            and not shared_with_target(arrival)
        )
        if onsets:
            interferers.append(WindowInterferer(index, evidence, onsets))
    return interferers


def snr_gradient_decision(
    *,
    target: ClusterAttributionEvidence,
    interferers: list[WindowInterferer],
    classifying_node_id: str,
    thresholds: AttributionThresholds,
) -> GradientDecision:
    target_snr = target.node_band_snr_db.get(classifying_node_id)
    if target_snr is None:
        return GradientDecision(False, "target_level_missing")
    dominance: dict[int, float] = {}
    target_band = _arrival_band_hz(target.node_onset_arrivals.get(classifying_node_id, ()))
    for interferer in interferers:
        interferer_snr = interferer.evidence.node_band_snr_db[classifying_node_id]
        measured = target_snr - interferer_snr
        dominance[interferer.cluster_index] = measured
        if measured < thresholds.dominance_margin_db:
            return GradientDecision(False, "not_dominant", dominance)
        # Single-array windows have no cross-node gradient to check against.
        target_predicted = target.predicted_snr_db(classifying_node_id)
        interferer_predicted = interferer.evidence.predicted_snr_db(classifying_node_id)
        if (
            target.spreading_residual_db is None
            or interferer.evidence.spreading_residual_db is None
            or target_predicted is None
            or interferer_predicted is None
        ):
            return GradientDecision(False, "no_cross_node_gradient", dominance)
        if max(target.spreading_residual_db, interferer.evidence.spreading_residual_db) > (
            thresholds.spreading_tolerance_db
        ):
            return GradientDecision(False, "spreading_inconsistent", dominance)
        if abs(measured - (target_predicted - interferer_predicted)) > thresholds.spreading_tolerance_db:
            return GradientDecision(False, "dominance_not_predicted", dominance)
        # Field Mode S-fixes: band SNR is measured over each cluster's band, so when the
        # target's band covers the interferer's onsets the target "level" (and its fitted
        # source level) includes the interferer (collinear trail: a thrush cluster whose band
        # reached the vireo's trill read 13 dB "dominant" and took the vireo's label). Then
        # only geometry can justify dominance: the interferer must be far enough away.
        if _bands_overlap(target_band, interferer) and _geometric_dominance_db(
            target, interferer.evidence, classifying_node_id
        ) < thresholds.dominance_margin_db - thresholds.spreading_tolerance_db:
            return GradientDecision(False, "bands_overlap_not_geometric", dominance)
    return GradientDecision(True, "dominant_and_consistent", dominance)


def _bands_overlap(target_band: tuple[float, float] | None, interferer: WindowInterferer) -> bool:
    return target_band is not None and any(
        _band_overlap_fraction(target_band, (band_min, band_max)) >= SAME_BAND_MIN_OVERLAP_FRACTION
        for _, band_min, band_max in interferer.onset_seconds_in_window
    )


def _geometric_dominance_db(
    target: ClusterAttributionEvidence, interferer: ClusterAttributionEvidence, node_id: str
) -> float:
    """Spreading-only level advantage of the target at a node (equal source levels)."""
    target_distance = max(target.node_distances_m.get(node_id, 0.0), MINIMUM_SPREADING_DISTANCE_M)
    interferer_distance = max(interferer.node_distances_m.get(node_id, 0.0), MINIMUM_SPREADING_DISTANCE_M)
    return DECIBELS_PER_AMPLITUDE_DECADE * math.log10(interferer_distance / target_distance)


def _arrival_band_hz(arrivals) -> tuple[float, float] | None:
    if not arrivals:
        return None
    return min(arrival.band_min_hz for arrival in arrivals), max(arrival.band_max_hz for arrival in arrivals)


def _band_overlap_fraction(band_a: tuple[float, float], band_b: tuple[float, float]) -> float:
    narrower = min(band_a[1] - band_a[0], band_b[1] - band_b[0])
    if narrower <= 0.0:
        return 0.0
    return max(0.0, min(band_a[1], band_b[1]) - max(band_a[0], band_b[0])) / narrower


def is_band_distinct_persistence_hold(
    *,
    status_reasons: tuple[str, ...],
    band_hz: tuple[float, float],
    accepted_bands_hz: list[tuple[float, float]],
    same_band_min_overlap_fraction: float,
) -> bool:
    """A single-array cluster held back *only* by persistence whose band does not
    overlap any accepted cluster's band cannot be that cluster's echo or sidelobe
    (both repeat the source's spectrum): it is a second source and gets its own
    product, so attribution can label it instead of it vanishing."""
    if tuple(status_reasons) != (PERSISTENCE_HOLD_REASON,):
        return False
    return all(
        _band_overlap_fraction(band_hz, accepted) < same_band_min_overlap_fraction for accepted in accepted_bands_hz
    )


def _species_ranking(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(
        ((label, float(score)) for label, score in scores.items() if label.strip().lower() not in NON_SPECIES_LABELS),
        key=lambda item: -item[1],
    )


def loco_decision(
    *,
    original_label: str,
    original_scores: dict[str, float],
    masked_label: str,
    masked_confidence: float,
    masked_scores: dict[str, float],
    thresholds: AttributionThresholds,
) -> LocoDecision:
    """Settle the target's label from a leave-interferers-out rerun, or refuse."""
    masked_scores = dict(masked_scores) or {masked_label: masked_confidence}
    collapsed = tuple(
        label
        for label, score in original_scores.items()
        if score > 0.0 and masked_scores.get(label, 0.0) <= thresholds.loco_collapse_ratio * score
    )
    if masked_label.strip().lower() in NON_SPECIES_LABELS:
        return LocoDecision(False, "masked_rerun_abstained", collapsed_labels=collapsed)
    if masked_confidence < thresholds.loco_min_confidence:
        return LocoDecision(False, "masked_rerun_not_confident", collapsed_labels=collapsed)
    ranking = [item for item in _species_ranking(masked_scores) if item[0] != masked_label]
    runner_up = ranking[0][1] if ranking else 0.0
    if masked_confidence - runner_up < thresholds.loco_label_margin:
        return LocoDecision(False, "masked_rerun_ambiguous", collapsed_labels=collapsed)
    if masked_label != original_label and original_label.strip().lower() not in NON_SPECIES_LABELS:
        # The window's label changed: only trust it if the old winner collapsed
        # (it was the interferer's), not if both survive (masking was not decisive).
        if original_label not in collapsed:
            return LocoDecision(False, "original_label_survived", collapsed_labels=collapsed)
    return LocoDecision(True, "target_label_retained", masked_label, float(masked_confidence), collapsed)
