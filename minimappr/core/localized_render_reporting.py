"""Classification and reporting of a placed Rust classifier render.

``LocalizedRenderHandoff`` places and gates a render's solve, then hands it
here. Most renders are classified inline: the sidecar's label (or one mono
render inference) is cheap enough for the SSE consumer that awaits ingest.

A point-network render is not (plan item P7): its label comes from a Python
inference on the clearest node's aligned clip, plus buffer reads and FFTs.
That work must not run on the single SSE consumer, which also mirrors every
node's ``raw_audio_frame`` into the buffer. So the render's product is queued
on the classification stage like a Python-ingest product, carrying the
remaining work as ``deferred_classification``; it gets that stage's chunking
suppression, report-window gate, backpressure drops and timeout, and a
classification worker releases a held fix (P5) when classification never
happens. The detection then reaches rules through the rules queue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from minimappr.api.rust_dsp_manifests import LocalizedClassifierRenderRequest
from minimappr.core.fusion_stage_carriers import DetectionProduct, LocalizedCandidate
from minimappr.core.localized_handoff_duplicate_merge import CrossReporterDuplicateMerger, LocalizedClusterEmission
from minimappr.core.localized_render_handoff_hooks import HandoffSolveGeometry
from minimappr.core.localized_render_handoff_solve_features import classify_render_audio
from minimappr.core.point_network_call_classification import PointNetworkCallRequest
from minimappr.core.point_network_call_requests import classify_rust_render_call, point_network_request_for_rust_render
from minimappr.models import NodeSpec

if TYPE_CHECKING:
    from minimappr.core.classification import ClassifiedResult
    from minimappr.core.fusion_node import FusionNode

CROSS_REPORTER_DUPLICATE_MERGED_REASON = "cross_reporter_duplicate_merged"


@dataclass(slots=True)
class RenderReport:
    """One render's placed solve, ready to classify and report."""

    payload: LocalizedClassifierRenderRequest
    node: NodeSpec
    geometry: HandoffSolveGeometry
    localized_product: LocalizedCandidate
    reporting_modality: str
    capability_tier: str
    normalized_audio: np.ndarray
    rust_extra_features: dict[str, Any]

    @property
    def reference_sensor(self) -> str:
        return self.geometry.reference_sensor


class LocalizedRenderReporter:
    def __init__(self, host: "FusionNode") -> None:
        self._host = host
        self._cross_reporter_duplicate_merger = CrossReporterDuplicateMerger()

    async def classify_and_report(self, report: RenderReport) -> bool:
        """Classify and report ``report`` inline, or queue a point-network render's
        clip classification. True when a classification worker now owns the
        product (and with it the held-fix release)."""
        host = self._host
        request = await self._point_network_request(report)
        if request is not None:
            return await self._queue_point_network_classification(report, request)
        if host._classification_chunking_policy is not None and report.payload.authoritative_classification is None:
            if host._should_suppress_chunked_classification(report.localized_product):
                host._metrics.birdnet_chunk_dispatches_suppressed += 1
                return False
        classified = await self._render_classification(report)
        detection_product = await self._assemble_detection(report, classified)
        if detection_product is not None:
            await host._process_rules_and_delivery(detection_product)
        return False

    async def _point_network_request(self, report: RenderReport) -> PointNetworkCallRequest | None:
        if not self._host._point_network_call_classifier.config.enabled:
            return None
        return await point_network_request_for_rust_render(
            self._host,
            payload=report.payload,
            geometry=report.geometry,
            localized_product=report.localized_product,
            reporting_modality=report.reporting_modality,
        )

    async def _queue_point_network_classification(self, report: RenderReport, request: PointNetworkCallRequest) -> bool:
        host = self._host
        product = report.localized_product

        async def classify_clip_and_assemble() -> list[DetectionProduct]:
            classified = await classify_rust_render_call(
                host,
                payload=report.payload,
                request=request,
                default_classification=lambda: self._render_classification(report),
            )
            detection_product = await self._assemble_detection(report, classified)
            return [] if detection_product is None else [detection_product]

        product.deferred_classification = classify_clip_and_assemble
        if host._classification_priority_enabled:
            product.classification_significance = await host._score_classification_significance(product)
        return await host._enqueue_stage(host._classification_queue, product)

    async def _render_classification(self, report: RenderReport) -> "ClassifiedResult":
        # The orchestrator is read per call: a replaced one applies immediately.
        return await classify_render_audio(self._host._classification_orchestrator, report.payload, report.normalized_audio)

    async def _assemble_detection(self, report: RenderReport, classified: "ClassifiedResult") -> DetectionProduct | None:
        """The render's detection, ready for rules; None when it merged into an
        earlier reporter's (its track still takes the label) or was not assembled."""
        host = self._host
        payload, node, product = report.payload, report.node, report.localized_product
        classified.classification.features.update(report.rust_extra_features)
        branch = product.localization_branch
        cluster_emission_key = self._localized_cluster_emission_key(
            payload, reporting_modality=report.reporting_modality, label=classified.classification.label
        )
        if cluster_emission_key is not None and self._merges_into_earlier_reporter(
            payload, cluster_emission_key=cluster_emission_key, position_m=branch.localization_position_m
        ):
            await host._label_track_of_unreported_classification(host._classification_tracker_feed(
                product=product, classified=classified, source_node_id=node.id, capability_tier=report.capability_tier,
                reference_sensor=report.reference_sensor, tdoa_s={}, localization_position_m=branch.localization_position_m,
                localization_position_covariance_m2=branch.localization_position_covariance_m2,
            ))
            return None
        await host._maybe_trigger_speech_capture(
            node_id=node.id,
            sensor_id=report.reference_sensor,
            event_time_ns=payload.event_time_ns,
            scores=classified.classification.scores,
        )
        detection_product = await host._assemble_reporting_branch(
            product=product,
            classified=classified,
            reporting_modality=report.reporting_modality,
            localization_position_m=branch.localization_position_m,
            localization_confidence=branch.localization_confidence,
            localization_gdop=branch.localization_gdop,
            localization_position_covariance_m2=branch.localization_position_covariance_m2,
            localization_range_observability=branch.localization_range_observability,
            localization_residual_rms_seconds=branch.localization_residual_rms_seconds,
            localization_range_projection_mode=branch.localization_range_projection_mode,
            reference_sensor=report.reference_sensor,
            reference_signal=report.normalized_audio,
            tdoa_s={},
            capability_tier=report.capability_tier,
            localization_method=branch.localization_method,
        )
        if detection_product is None:
            return None
        if cluster_emission_key is not None:
            cluster_id, label, report_window_start_ns = cluster_emission_key
            self._cross_reporter_duplicate_merger.note_emission(
                cluster_id=cluster_id,
                label=label,
                report_window_start_ns=report_window_start_ns,
                report_window_ns=host.reporting_policy.reporting_window_ns,
                emission=LocalizedClusterEmission(
                    reporting_node_id=node.id,
                    position_m=tuple(float(value) for value in branch.localization_position_m),
                    solve_id=payload.localization_solve.provenance.solve_id,
                ),
            )
        return detection_product

    def _localized_cluster_emission_key(
        self,
        payload: LocalizedClassifierRenderRequest,
        *,
        reporting_modality: str,
        label: str,
    ) -> tuple[str, str, int] | None:
        """(cluster, label, report window) for a localized cluster solve; else ``None``."""
        solve = payload.localization_solve
        if reporting_modality != "localized" or solve is None or solve.provenance.cluster_id is None:
            return None
        report_window_start_ns, _ = self._host.reporting_policy.report_window_bounds(payload.event_time_ns)
        return solve.provenance.cluster_id, label, report_window_start_ns

    def _merges_into_earlier_reporter(
        self,
        payload: LocalizedClassifierRenderRequest,
        *,
        cluster_emission_key: tuple[str, str, int],
        position_m,
    ) -> bool:
        cluster_id, label, report_window_start_ns = cluster_emission_key
        merge_target = self._cross_reporter_duplicate_merger.merge_target(
            cluster_id=cluster_id,
            label=label,
            report_window_start_ns=report_window_start_ns,
            reporting_node_id=payload.node.id,
            position_m=position_m,
            max_separation_m=self._host.settings.localized_handoff_duplicate_merge_max_separation_m,
        )
        if merge_target is None:
            return False
        self._host._record_silent_drop(
            stage="classification",
            reason=CROSS_REPORTER_DUPLICATE_MERGED_REASON,
            event_time_ns=payload.event_time_ns,
            source_node_id=payload.node.id,
            merged_into_reporting_node_id=merge_target.reporting_node_id,
            merged_into_solve_id=merge_target.solve_id,
        )
        return True


__all__ = ["CROSS_REPORTER_DUPLICATE_MERGED_REASON", "LocalizedRenderReporter", "RenderReport"]
