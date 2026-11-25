"""P7 point-network renders on the classification worker keep the inline gates.

Moving the clip classification onto a classification worker must not change
which renders are reported: a render carrying the sidecar's label is never
chunk-suppressed (as inline), and no Rust render is under the localized
report-window cap.
"""

from __future__ import annotations

from pathlib import Path

from minimappr.classifiers.base import ClassificationResult
from tests.point_network_fusion_harness import started_point_network_fusion, stored_detections_by_solve
from tests.point_network_scene import append_captures, render_point_node_captures
from tests.test_point_network_review_fixes import (
    CALLER_NEAR_A,
    RENDER_SOLVE_ID,
    SIDECAR_CONFIDENCE,
    SIDECAR_LABEL,
    _render_near_a,
)


class _SuppressEveryDispatch:
    """A chunking policy whose every dispatch is suppressed."""

    def __init__(self) -> None:
        self.recorded_outcomes: list[dict] = []

    def should_dispatch(self, *, source_node_id, event_time_ns) -> bool:
        return False

    def record_dispatch_outcome(self, **kwargs) -> None:
        self.recorded_outcomes.append(kwargs)


def _with_sidecar_label(request):
    request.authoritative_classification = ClassificationResult(
        label=SIDECAR_LABEL, confidence=SIDECAR_CONFIDENCE, scores={SIDECAR_LABEL: SIDECAR_CONFIDENCE}, features={}
    )
    return request


async def _report_render(tmp_path: Path, *, prepare_fusion, prepare_request) -> tuple[dict, object]:
    fusion, storage, _ = await started_point_network_fusion(tmp_path)
    try:
        await append_captures(fusion.buffer, render_point_node_captures([CALLER_NEAR_A]))
        request = prepare_request(_render_near_a(fusion))
        prepare_fusion(fusion, request)
        await fusion.ingest_localized_render(request)
        return await stored_detections_by_solve(storage, fusion), fusion._metrics
    finally:
        await fusion.stop()
        await storage.close()


def _suppress_every_chunk(fusion, request) -> None:
    fusion._classification_chunking_policy = _SuppressEveryDispatch()


async def test_a_sidecar_labelled_point_network_render_is_never_chunk_suppressed(tmp_path: Path) -> None:
    detections, metrics = await _report_render(
        tmp_path, prepare_fusion=_suppress_every_chunk, prepare_request=_with_sidecar_label
    )
    assert RENDER_SOLVE_ID in detections
    assert metrics.birdnet_chunk_dispatches_suppressed == 0


async def test_an_unlabelled_point_network_render_is_still_chunk_suppressed(tmp_path: Path) -> None:
    detections, metrics = await _report_render(
        tmp_path, prepare_fusion=_suppress_every_chunk, prepare_request=lambda request: request
    )
    assert RENDER_SOLVE_ID not in detections
    assert metrics.birdnet_chunk_dispatches_suppressed == 1


def _saturate_the_render_report_window(fusion, request) -> None:
    window_start_ns, _ = fusion.reporting_policy.report_window_bounds(request.event_time_ns)
    for _ in range(fusion.fusion_config.report_window_localized_emission_cap):
        fusion._note_localized_window_emission(source_node_id=request.node.id, report_window_start_ns=window_start_ns)


async def test_a_point_network_render_is_not_under_the_report_window_cap(tmp_path: Path) -> None:
    detections, metrics = await _report_render(
        tmp_path, prepare_fusion=_saturate_the_render_report_window, prepare_request=lambda request: request
    )
    assert RENDER_SOLVE_ID in detections
    assert metrics.classification_pregate_skipped_count == 0
