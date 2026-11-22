"""Classifier-worker seam for onset-cluster attribution (flag `onset_attribution_enabled`).

Runs after a cluster's window has been classified (`FusionNode._classify_and_assemble`),
only for products carrying a `ClusterAttributionContext` from the onset stage:

* no other cluster audible in the window -> unchanged (tier `single_cluster`);
* SNR gradient settles it -> unchanged, annotated (tier `snr_gradient`);
* else LOCO: mask the interferers' regions in the in-memory classification
  signal with noise at its floor (worker thread) and rerun the classifier
  (worker thread, existing timeout); capped per analysis window;
* still unsettled (or cap hit / LOCO off) -> label `unknown`, confidence 0: the
  position is kept, no species is guessed.

Masked audio lives only in local numpy arrays and is never assigned to the
result's `classification_signal`, so it can never reach a snippet on disk.
"""

from __future__ import annotations

import asyncio
import time
import zlib
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from minimappr.classifiers.base import ClassificationResult
from minimappr.config import OnsetClusterAttributionConfig
from minimappr.core.onset_cluster_attribution import (
    ATTRIBUTION_TIER_LOCO,
    ATTRIBUTION_TIER_SINGLE_CLUSTER,
    ATTRIBUTION_TIER_SNR_GRADIENT,
    ATTRIBUTION_TIER_UNSETTLED,
    AttributionThresholds,
    ClusterAttributionContext,
    interferers_in_window,
    loco_decision,
    snr_gradient_decision,
)
from minimappr.core.onset_cluster_masking import (
    MaskedSignal,
    MaskingParameters,
    MaskRegionSeed,
    mask_regions_with_noise_floor,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from minimappr.core.classification import ClassificationOrchestrator, ClassifiedResult

UNLABELLED_LABEL = "unknown"
ATTRIBUTION_FEATURE_KEY = "onset_attribution"
# Rerun budgets are remembered for this many recent analysis windows (memory bound only).
MAX_TRACKED_RERUN_WINDOWS = 512
MILLISECONDS_PER_NANOSECOND = 1e-6
FAILED_CLASSIFICATION_REASONS = frozenset({"classification_error", "classification_timeout"})


class LocoRerunBudget:
    """Per-analysis-window rerun counter shared by that window's sibling clusters."""

    def __init__(self, max_reruns_per_window: int) -> None:
        self._max_reruns = max(0, int(max_reruns_per_window))
        self._used: OrderedDict[str, int] = OrderedDict()

    def try_consume(self, window_id: str) -> bool:
        used = self._used.get(window_id, 0)
        if used >= self._max_reruns:
            return False
        self._used[window_id] = used + 1
        self._used.move_to_end(window_id)
        while len(self._used) > MAX_TRACKED_RERUN_WINDOWS:
            self._used.popitem(last=False)
        return True


class OnsetClusterAttributor:
    def __init__(
        self,
        config: OnsetClusterAttributionConfig,
        orchestrator: ClassificationOrchestrator,
        metrics_provider: Callable[[], Any],
    ) -> None:
        self.config = config
        self._orchestrator = orchestrator
        # FusionMetrics is created after the orchestrator, so it is looked up per call.
        self._metrics_provider = metrics_provider
        self._budget = LocoRerunBudget(config.loco_max_reruns_per_window)
        self._thresholds = AttributionThresholds(
            audible_snr_db=config.audible_snr_db,
            dominance_margin_db=config.dominance_margin_db,
            spreading_tolerance_db=config.spreading_tolerance_db,
            loco_min_confidence=config.loco_min_confidence,
            loco_label_margin=config.loco_label_margin,
            loco_collapse_ratio=config.loco_collapse_ratio,
        )
        self._masking = MaskingParameters(
            stft_frame_seconds=config.loco_stft_frame_seconds,
            noise_floor_percentile=config.loco_noise_floor_percentile,
            active_snr_db=config.loco_mask_active_snr_db,
            release_seconds=config.loco_mask_release_seconds,
            max_region_seconds=config.loco_mask_max_seconds,
            band_padding_hz=config.loco_mask_band_padding_hz,
            pre_roll_seconds=config.mask_pre_roll_seconds,
        )

    async def attribute(
        self, classified: ClassifiedResult, context: ClusterAttributionContext, *, event_time_ns: int
    ) -> ClassifiedResult:
        original = classified.classification
        if original.features.get("reason") in FAILED_CLASSIFICATION_REASONS:
            return classified
        metrics = self._metrics_provider()
        metrics.onset_attribution_window_count += 1
        target = context.cluster_evidence.get(context.target_index)
        if target is None:
            return classified
        interferers = interferers_in_window(
            target_index=context.target_index,
            cluster_evidence=context.cluster_evidence,
            classifying_node_id=context.classifying_node_id,
            window_start_sample=context.classification_start_sample,
            window_end_sample=context.classification_end_sample,
            sample_rate_hz=context.sample_rate_hz,
            audible_snr_db=self._thresholds.audible_snr_db,
            lookback_samples=int(round(self.config.loco_mask_max_seconds * context.sample_rate_hz)),
        )
        annotation: dict[str, Any] = {"interferers": len(interferers), "original_label": original.label}
        if not interferers:
            metrics.onset_attribution_single_cluster_count += 1
            original.features[ATTRIBUTION_FEATURE_KEY] = {**annotation, "tier": ATTRIBUTION_TIER_SINGLE_CLUSTER}
            return classified
        gradient = snr_gradient_decision(
            target=target,
            interferers=interferers,
            classifying_node_id=context.classifying_node_id,
            thresholds=self._thresholds,
        )
        annotation["gradient_reason"] = gradient.reason
        annotation["dominance_db"] = {str(k): round(v, 2) for k, v in gradient.dominance_db.items()}
        if gradient.settled:
            metrics.onset_attribution_snr_gradient_settled_count += 1
            original.features[ATTRIBUTION_FEATURE_KEY] = {**annotation, "tier": ATTRIBUTION_TIER_SNR_GRADIENT}
            return classified
        if not self.config.loco_enabled:
            return await self._unlabelled(classified, {**annotation, "reason": "loco_disabled"}, event_time_ns)
        started_ns = time.perf_counter_ns()
        masked_signal = await self._mask_interferers(classified, context, interferers)
        annotation["loco_masked_cell_fraction"] = round(masked_signal.masked_cell_fraction, 4)
        if masked_signal.masked_cell_fraction <= 0.0:
            # Interferer notes ended before this window: nothing of theirs was classified.
            metrics.onset_attribution_single_cluster_count += 1
            original.features[ATTRIBUTION_FEATURE_KEY] = {
                **annotation, "tier": ATTRIBUTION_TIER_SINGLE_CLUSTER, "reason": "interferers_silent_in_window"
            }
            return classified
        if not self._budget.try_consume(context.window_id):
            metrics.onset_attribution_loco_rerun_cap_hit_count += 1
            return await self._unlabelled(classified, {**annotation, "reason": "loco_rerun_cap"}, event_time_ns)
        masked = await self._orchestrator.rerun_classifier_on_prepared_signal(
            masked_signal.samples, context.sample_rate_hz
        )
        self._record_rerun_cost(started_ns)
        decision = loco_decision(
            original_label=original.label,
            original_scores=dict(original.scores),
            masked_label=masked.label,
            masked_confidence=float(masked.confidence),
            masked_scores=dict(masked.scores),
            thresholds=self._thresholds,
        )
        annotation.update(
            loco_reason=decision.reason,
            loco_masked_label=masked.label,
            loco_masked_confidence=round(float(masked.confidence), 4),
            loco_collapsed_labels=list(decision.collapsed_labels),
        )
        if not decision.settled:
            return await self._unlabelled(classified, annotation, event_time_ns)
        metrics.onset_attribution_loco_settled_count += 1
        features = {**original.features, **masked.features}
        features[ATTRIBUTION_FEATURE_KEY] = {**annotation, "tier": ATTRIBUTION_TIER_LOCO}
        attributed = ClassificationResult(
            label=decision.label, confidence=decision.confidence, scores=dict(masked.scores), features=features
        )
        return await self._orchestrator.relabel_classified_result(
            classified, attributed, event_time_ns=event_time_ns
        )

    async def _mask_interferers(self, classified, context, interferers) -> MaskedSignal:
        seeds = [
            MaskRegionSeed(onset_seconds=onset_s, band_min_hz=band_min, band_max_hz=band_max)
            for interferer in interferers
            for onset_s, band_min, band_max in interferer.onset_seconds_in_window
        ]
        # Deterministic noise per (window, cluster): replays score identically.
        rng = np.random.default_rng(zlib.crc32(f"{context.window_id}:{context.target_index}".encode()))
        return await asyncio.to_thread(
            mask_regions_with_noise_floor,
            np.asarray(classified.classification_signal),
            sample_rate_hz=context.sample_rate_hz,
            seeds=seeds,
            parameters=self._masking,
            rng=rng,
        )

    def _record_rerun_cost(self, started_ns: int) -> None:
        """Mask + rerun wall time (both off the event loop)."""
        elapsed_ms = (time.perf_counter_ns() - started_ns) * MILLISECONDS_PER_NANOSECOND
        metrics = self._metrics_provider()
        metrics.onset_attribution_loco_rerun_count += 1
        metrics.onset_attribution_loco_rerun_total_ms += elapsed_ms
        metrics.onset_attribution_loco_rerun_max_ms = max(
            metrics.onset_attribution_loco_rerun_max_ms, elapsed_ms
        )

    async def _unlabelled(self, classified, annotation: dict[str, Any], event_time_ns: int) -> ClassifiedResult:
        self._metrics_provider().onset_attribution_unlabelled_count += 1
        original = classified.classification
        features = dict(original.features)
        features[ATTRIBUTION_FEATURE_KEY] = {
            **annotation,
            "tier": ATTRIBUTION_TIER_UNSETTLED,
            "unattributed_scores": {label: round(float(score), 4) for label, score in original.scores.items()},
        }
        unlabelled = ClassificationResult(label=UNLABELLED_LABEL, confidence=0.0, scores={}, features=features)
        return await self._orchestrator.relabel_classified_result(
            classified, unlabelled, event_time_ns=event_time_ns
        )
