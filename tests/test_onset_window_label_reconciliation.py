"""Window label reconciliation (closeout B2): the rule, the classified-result
rewrite, and the non-blocking sibling buffer (timeout, withdrawal, bounds)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np

from minimappr.core.classification import ClassifiedResult
from minimappr.core.onset_cluster_attribution import ClusterAttributionContext
from minimappr.core.onset_srp_localization_stage import OnsetSrpLocalizationStage
from minimappr.core.onset_window_label_reconciliation import (
    RECONCILED_DROPPED_LABELS_KEY,
    OnsetWindowLabelReconciler,
    WindowSibling,
    labels_to_drop_by_cluster,
    reconcile_window_classifications,
)
from minimappr.models import ClassificationResult

ROBIN, TANAGER, WREN = "american robin", "scarlet tanager", "house wren"


def _classified(label: str, scores: dict[str, float], tier: str | None) -> ClassifiedResult:
    features = {"onset_attribution": {"tier": tier}} if tier else {}
    classification = ClassificationResult(
        label=label, confidence=max(scores.values(), default=0.0), scores=scores, features=features
    )
    return ClassifiedResult(
        classification=classification,
        omni_classification=classification,
        beamformed_classification=classification,
        classification_path="beamformed",
        classification_signal=np.zeros(4),
        beamforming_error=None,
        label_category="bird",
        iff_category="neutral",
        label_id=None,
    )


def test_gradient_cluster_drops_the_loco_siblings_top_label():
    assert labels_to_drop_by_cluster([("snr_gradient", ROBIN), ("loco", TANAGER)]) == [[TANAGER], []]


def test_own_top_label_survives_a_same_species_sibling():
    assert labels_to_drop_by_cluster([("snr_gradient", ROBIN), ("snr_gradient", ROBIN)]) == [[], []]


def test_unsettled_and_single_cluster_siblings_claim_nothing():
    dropped = labels_to_drop_by_cluster(
        [("snr_gradient", ROBIN), ("unsettled", None), ("single_cluster", WREN), ("snr_gradient", TANAGER)]
    )
    assert dropped == [[TANAGER], [], [], [ROBIN]]


def test_reconcile_rewrites_scores_and_keeps_aliases_in_step():
    robin = _classified(ROBIN, {ROBIN: 0.85, TANAGER: 0.83, WREN: 0.2}, "snr_gradient")
    tanager = _classified(TANAGER, {TANAGER: 0.85}, "loco")
    (robin_after, tanager_after), dropped = reconcile_window_classifications([robin, tanager])
    assert dropped == 1
    assert robin_after.classification.scores == {ROBIN: 0.85, WREN: 0.2}
    assert robin_after.classification.label == ROBIN
    assert robin_after.beamformed_classification is robin_after.classification
    assert robin_after.omni_classification is robin_after.classification
    assert robin_after.classification.features["onset_attribution"][RECONCILED_DROPPED_LABELS_KEY] == [TANAGER]
    assert tanager_after is tanager
    assert robin.classification.scores[TANAGER] == 0.83  # the input is not mutated


def _sibling(label: str, scores: dict[str, float], tier: str) -> WindowSibling:
    return WindowSibling(product=SimpleNamespace(label=label), classified=_classified(label, scores, tier))


class _Emitted:
    def __init__(self) -> None:
        self.windows: list[list[WindowSibling]] = []

    async def __call__(self, siblings: list[WindowSibling]) -> None:
        self.windows.append(siblings)


def _reconciler(emitted: _Emitted, timeout_seconds: float = 60.0, **kwargs) -> OnsetWindowLabelReconciler:
    return OnsetWindowLabelReconciler(timeout_seconds=timeout_seconds, emit_released_window=emitted, **kwargs)


def test_single_product_window_passes_straight_through():
    async def scenario():
        reconciler = _reconciler(_Emitted())
        sibling = _sibling(ROBIN, {ROBIN: 0.9}, "single_cluster")
        assert reconciler.submit(window_id="w", sibling_count=1, candidate_id="a", sibling=sibling) == [sibling]
        assert reconciler.pending_window_count() == 0

    asyncio.run(scenario())


def test_last_sibling_releases_the_reconciled_window():
    async def scenario():
        emitted = _Emitted()
        reconciler = _reconciler(emitted)
        robin = _sibling(ROBIN, {ROBIN: 0.85, TANAGER: 0.83}, "snr_gradient")
        tanager = _sibling(TANAGER, {TANAGER: 0.85}, "loco")
        assert reconciler.submit(window_id="w", sibling_count=2, candidate_id="a", sibling=robin) == []
        ready = reconciler.submit(window_id="w", sibling_count=2, candidate_id="b", sibling=tanager)
        assert [s.product.label for s in ready] == [ROBIN, TANAGER]
        assert ready[0].classified.classification.scores == {ROBIN: 0.85}
        assert reconciler.pending_window_count() == 0 and not emitted.windows
        assert reconciler.counters.labels_dropped == 1

    asyncio.run(scenario())


def test_timed_out_window_is_emitted_and_a_late_sibling_passes_through():
    async def scenario():
        emitted = _Emitted()
        reconciler = _reconciler(emitted, timeout_seconds=0.01)
        robin = _sibling(ROBIN, {ROBIN: 0.85, TANAGER: 0.83}, "snr_gradient")
        assert reconciler.submit(window_id="w", sibling_count=2, candidate_id="a", sibling=robin) == []
        await asyncio.sleep(0.05)
        await reconciler.release_all()
        assert len(emitted.windows) == 1 and emitted.windows[0][0].product.label == ROBIN
        assert reconciler.pending_window_count() == 0
        assert reconciler.counters.windows_timed_out == 1
        late = _sibling(TANAGER, {TANAGER: 0.85}, "loco")
        assert reconciler.submit(window_id="w", sibling_count=2, candidate_id="b", sibling=late) == [late]
        assert reconciler.counters.late_siblings == 1

    asyncio.run(scenario())


def test_withdrawn_sibling_completes_the_window_without_waiting():
    async def scenario():
        emitted = _Emitted()
        reconciler = _reconciler(emitted)
        robin = _sibling(ROBIN, {ROBIN: 0.85}, "snr_gradient")
        assert reconciler.submit(window_id="w", sibling_count=2, candidate_id="a", sibling=robin) == []
        reconciler.withdraw(window_id="w", sibling_count=2, candidate_id="b")
        reconciler.withdraw(window_id="w", sibling_count=2, candidate_id="b")  # idempotent
        await reconciler.release_all()
        assert len(emitted.windows) == 1 and reconciler.counters.windows_timed_out == 0

    asyncio.run(scenario())


def test_open_windows_are_bounded_oldest_released_first():
    async def scenario():
        emitted = _Emitted()
        reconciler = _reconciler(emitted, max_pending_windows=2)
        for index in range(5):
            sibling = _sibling(ROBIN, {ROBIN: 0.85}, "snr_gradient")
            reconciler.submit(window_id=f"w{index}", sibling_count=3, candidate_id=f"c{index}", sibling=sibling)
            assert reconciler.pending_window_count() <= 2
        await asyncio.sleep(0)
        await asyncio.gather(*list(reconciler._emit_tasks))
        assert len(emitted.windows) == 3  # w0..w2 evicted, w3/w4 still open
        await reconciler.release_all()
        assert len(emitted.windows) == 5 and reconciler.pending_window_count() == 0

    asyncio.run(scenario())


def test_onset_stage_attaches_the_sibling_count_to_attributed_products_only():
    context = ClusterAttributionContext(
        window_id="cand", target_index=0, cluster_evidence={}, classifying_node_id="tetra-a",
        classification_start_sample=0, classification_end_sample=10, sample_rate_hz=24_000,
    )
    products = [SimpleNamespace(onset_attribution_context=context) for _ in range(3)]
    products.append(SimpleNamespace(onset_attribution_context=None))
    OnsetSrpLocalizationStage._attach_sibling_count(products)
    assert [p.onset_attribution_context.sibling_product_count for p in products[:3]] == [3, 3, 3]
    assert products[3].onset_attribution_context is None
