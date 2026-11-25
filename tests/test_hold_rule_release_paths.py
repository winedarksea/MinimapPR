"""Hold-rule release paths found by the localization-core overhaul review.

A held fix (gating only a labelled stationary_hold track) must end exactly one
way: applied with its label, dropped ("classified as nothing", an abstention,
an untracked label), or applied geometry-only (no classification came, or none
could be made). These cases each used to take the wrong one.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from minimappr.config import Settings
from minimappr.core.classification import ClassifiedResult
from minimappr.core.classification_empty_outcome import (
    CLASSIFICATION_UNAVAILABLE_DROP_REASON,
    EMPTY_CLASSIFICATION_DROP_REASON,
)
from minimappr.core.fusion_node import FusionMetrics
from minimappr.core.localized_render_handoff_hooks import SOLVE_ORIGIN_PYTHON_INGEST
from minimappr.core.onset_cluster_attribution import ATTRIBUTION_TIER_SNR_GRADIENT
from minimappr.core.onset_window_label_reconciliation import ATTRIBUTION_FEATURE_KEY, OnsetWindowLabelReconciler
from minimappr.core.position_observation_pipeline import SolveLedgerState
from minimappr.core.tracking import TrackManager
from minimappr.core.tracking_mode_config import TrackingMode, tracking_mode_by_category_with
from minimappr.models import ClassificationResult
from tests.test_localized_render_handoff_seams import _started_fusion
from tests.test_position_observation_pipeline import T0_NS, _classified_kwargs, _localized_product, _pipeline
from tests.test_position_observation_review_fixes import ROBIN, WILDLIFE, _bird_kwargs, _observe_python, _perched_robin

PERCHED_ROBIN_POSITION_M = (0.0, 0.0, 3.0)
NEIGHBOUR_FIX_POSITION_M = (0.3, 0.0, 3.0)
FAR_FIX_POSITION_M = (60.0, 0.0, 3.0)
# Long enough that no test waits for a window to time out.
RECONCILIATION_TIMEOUT_SECONDS = 60.0


async def test_untracked_label_on_a_held_fix_drops_it_for_good() -> None:
    """Reviewer's probe: a held fix labelled with an untracked category resolved to
    no track but stayed held, so the worker's release then applied it geometry-only
    onto the perched robin."""
    settings = Settings()
    settings.tracking_mode_by_category = tracking_mode_by_category_with("vehicle", TrackingMode.NONE)
    tracker, metrics = TrackManager(settings), FusionMetrics()
    pipeline = _pipeline(tracker, metrics)
    await _perched_robin(tracker, PERCHED_ROBIN_POSITION_M)
    await _observe_python(pipeline, "evt-car", NEIGHBOUR_FIX_POSITION_M)
    assert pipeline._ledger["evt-car"].state is SolveLedgerState.HELD_FOR_LABEL

    car_kwargs = _classified_kwargs(NEIGHBOUR_FIX_POSITION_M, label="Car", label_category="vehicle", timestamp_ns=T0_NS)
    car_kwargs["confidence"] = 0.9
    assert await pipeline.track_for_classified_solve(solve_id="evt-car", classified_update_kwargs=car_kwargs) is None
    entry = pipeline._ledger["evt-car"]
    assert entry.state is SolveLedgerState.HELD_DROPPED and entry.held_deadline is None

    # The classification worker's ``finally`` release, twice: still a no-op.
    await pipeline.release_held_position_without_classification("evt-car")
    await pipeline.release_held_position_without_classification("evt-car")
    [robin] = await tracker.snapshot()
    assert robin.update_count == 1 and robin.label == ROBIN
    assert metrics.position_observations_held_applied_geometry_only == 0
    assert metrics.position_observations_held_applied_with_label == 0
    assert metrics.position_observations_held_dropped == 1


async def test_untracked_label_never_moves_the_perched_track_with_the_category_gate_off() -> None:
    settings = Settings(association_category_gate_enabled=False)
    settings.tracking_mode_by_category = tracking_mode_by_category_with("vehicle", TrackingMode.NONE)
    tracker, metrics = TrackManager(settings), FusionMetrics()
    pipeline = _pipeline(tracker, metrics, association_category_gate_enabled=False)
    robin = await _perched_robin(tracker, PERCHED_ROBIN_POSITION_M)
    position_before = robin.position_m
    await _observe_python(pipeline, "evt-car", NEIGHBOUR_FIX_POSITION_M)
    car_kwargs = _classified_kwargs(NEIGHBOUR_FIX_POSITION_M, label="Car", label_category="vehicle", timestamp_ns=T0_NS)
    assert await pipeline.track_for_classified_solve(solve_id="evt-car", classified_update_kwargs=car_kwargs) is None
    await pipeline.release_held_position_without_classification("evt-car")
    [track] = await tracker.snapshot()
    assert (track.update_count, track.position_m, track.label) == (1, position_before, ROBIN)


# -- "could not classify" is not "classified as nothing" ----------------------------


class _RaisingBeamformer:
    def beamform(self, *args, **kwargs):
        raise RuntimeError("beamformer exploded")


async def _fusion_with_held_python_fix(tmp_path: Path, product):
    fusion, storage = await _started_fusion(tmp_path)
    robin = await fusion.tracker.update_with_classification(
        **_bird_kwargs(PERCHED_ROBIN_POSITION_M, ROBIN, T0_NS - 2_000_000_000)
    )
    pipeline = fusion._position_observations()
    await fusion._localization_seam_collaborators.position_observation_sink.observe_localized_solve(
        localized_product=product, reporting_modality="localized", solve_origin=SOLVE_ORIGIN_PYTHON_INGEST
    )
    await pipeline.drain()
    assert fusion._metrics.position_observations_held_for_label == 1
    return fusion, storage, robin, pipeline


async def _run_classification_worker(fusion, product) -> None:
    await fusion._classification_queue.put(product)
    await fusion._classification_queue.join()
    await fusion._position_observations().drain()


async def _assert_held_fix_applied_geometry_only(fusion, robin) -> None:
    [track] = await fusion.tracker.snapshot()
    assert track.id == robin.id and track.update_count == 2 and track.label == ROBIN
    metrics = fusion._metrics
    assert metrics.position_observations_held_applied_geometry_only == 1
    assert metrics.position_observations_held_dropped == 0
    drops = metrics.classification_drops_by_reason
    assert drops.get(CLASSIFICATION_UNAVAILABLE_DROP_REASON) == 1
    assert EMPTY_CLASSIFICATION_DROP_REASON not in drops


async def test_beamform_exception_applies_the_held_fix_geometry_only(tmp_path: Path) -> None:
    product = _localized_product(candidate_id="evt-beamform-error", position_m=NEIGHBOUR_FIX_POSITION_M)
    fusion, storage, robin, _ = await _fusion_with_held_python_fix(tmp_path, product)
    try:
        fusion._classification_orchestrator._beamformer = _RaisingBeamformer()
        await _run_classification_worker(fusion, product)
        await _assert_held_fix_applied_geometry_only(fusion, robin)
    finally:
        await fusion.stop()
        await storage.close()


async def test_array_too_small_to_beamform_applies_the_held_fix_geometry_only(tmp_path: Path) -> None:
    product = _localized_product(candidate_id="evt-one-sensor", position_m=NEIGHBOUR_FIX_POSITION_M)
    product.selected_sensor_ids = product.selected_sensor_ids[:1]
    fusion, storage, robin, _ = await _fusion_with_held_python_fix(tmp_path, product)
    try:
        await _run_classification_worker(fusion, product)
        await _assert_held_fix_applied_geometry_only(fusion, robin)
    finally:
        await fusion.stop()
        await storage.close()


# -- parked window-reconciliation siblings that assemble empty ----------------------


def _gradient_settled_robin() -> ClassifiedResult:
    classification = ClassificationResult(
        label=ROBIN,
        confidence=0.8,
        scores={ROBIN: 0.8},
        features={ATTRIBUTION_FEATURE_KEY: {"tier": ATTRIBUTION_TIER_SNR_GRADIENT}},
    )
    return ClassifiedResult(
        classification=classification,
        omni_classification=classification,
        beamformed_classification=classification,
        classification_path="beamformed",
        classification_signal=np.zeros(4),
        beamforming_error=None,
        label_category=WILDLIFE,
        iff_category="neutral",
        label_id=None,
    )


def _window_sibling(candidate_id: str, position_m) -> object:
    product = _localized_product(candidate_id=candidate_id, position_m=position_m)
    product.onset_attribution_context = SimpleNamespace(window_id="window-1", sibling_product_count=2)
    return product


def _reconciling_fusion_assembling_nothing(fusion) -> None:
    """Every sibling classifies (gradient-settled, so it parks) and assembles empty
    (merged into another reporter, zone-suppressed): "classified as nothing"."""
    fusion._window_label_reconciler = OnsetWindowLabelReconciler(
        timeout_seconds=RECONCILIATION_TIMEOUT_SECONDS,
        emit_released_window=fusion._emit_released_reconciliation_window,
    )

    async def classified_robin(product):
        return _gradient_settled_robin()

    async def assembles_nothing(product, localized):
        return []

    fusion._classify_and_attribute_localized = classified_robin
    fusion._assemble_localized_classification = assembles_nothing


async def test_parked_sibling_that_assembles_empty_drops_its_held_fix_at_once(tmp_path: Path) -> None:
    parked = _window_sibling("sib-parked", NEIGHBOUR_FIX_POSITION_M)
    fusion, storage, robin, pipeline = await _fusion_with_held_python_fix(tmp_path, parked)
    try:
        _reconciling_fusion_assembling_nothing(fusion)
        await _run_classification_worker(fusion, parked)
        assert parked.held_for_window_label_reconciliation
        assert pipeline._ledger["sib-parked"].state is SolveLedgerState.HELD_FOR_LABEL

        await _run_classification_worker(fusion, _window_sibling("sib-last", FAR_FIX_POSITION_M))
        # Dropped now, not applied geometry-only when the 30 s deadline fires.
        assert pipeline._ledger["sib-parked"].state is SolveLedgerState.HELD_DROPPED
        assert pipeline._ledger["sib-parked"].held_deadline is None
        [track] = [track for track in await fusion.tracker.snapshot() if track.id == robin.id]
        assert track.update_count == 1
        assert fusion._metrics.classification_drops_by_reason.get(EMPTY_CLASSIFICATION_DROP_REASON) == 2
    finally:
        await fusion.stop()
        await storage.close()


async def test_released_window_sibling_that_assembles_empty_drops_its_held_fix(tmp_path: Path) -> None:
    parked = _window_sibling("sib-released", NEIGHBOUR_FIX_POSITION_M)
    fusion, storage, robin, pipeline = await _fusion_with_held_python_fix(tmp_path, parked)
    try:
        _reconciling_fusion_assembling_nothing(fusion)
        await _run_classification_worker(fusion, parked)
        # The drain / timeout path: the window is released off the worker.
        await fusion.release_pending_window_reconciliation()
        assert pipeline._ledger["sib-released"].state is SolveLedgerState.HELD_DROPPED
        [track] = await fusion.tracker.snapshot()
        assert track.id == robin.id and track.update_count == 1
    finally:
        await fusion.stop()
        await storage.close()


async def test_completing_sibling_that_assembles_empty_beside_a_reported_one_drops_its_held_fix(tmp_path: Path) -> None:
    completing = _window_sibling("sib-completing", NEIGHBOUR_FIX_POSITION_M)
    fusion, storage, robin, pipeline = await _fusion_with_held_python_fix(tmp_path, completing)
    try:
        _reconciling_fusion_assembling_nothing(fusion)
        reported = _window_sibling("sib-reported", FAR_FIX_POSITION_M)

        async def only_the_far_sibling_reports(product, localized):
            return [SimpleNamespace(detection=None)] if product is reported else []

        fusion._assemble_localized_classification = only_the_far_sibling_reports
        assert await fusion._classify_and_assemble(reported) == []
        detection_products = await fusion._classify_and_assemble(completing)
        assert len(detection_products) == 1
        assert pipeline._ledger["sib-completing"].state is SolveLedgerState.HELD_DROPPED
        [track] = await fusion.tracker.snapshot()
        assert track.id == robin.id and track.update_count == 1
    finally:
        await fusion.stop()
        await storage.close()
