"""Ingest of Rust-sidecar classifier renders into the fusion pipeline.

``FusionNode.ingest_localized_render`` delegates here. A render arrives with
the sidecar's localization already attached; this collaborator registers the
node, resolves solve geometry and the position estimate, gates it exactly like
the Python solver path, then classifies and reports synchronously so callers
can observe the resulting detection as soon as ingest returns.

The handoff reads FusionNode state through ``host`` at call time rather than
copying it, so monkeypatched host helpers and late-bound collaborators (for
example a replaced classification orchestrator) are honoured.

The mono render is never written into Python's raw per-sensor buffers: those
are mirrored from the sidecar's real per-channel ``raw_audio_frame`` events at
packet time (``api/stream_consumer.py``), and a receipt-timed mono copy
(possibly at a different sample rate) would corrupt them for every reader
(omni scanner, speech capture, capture sessions / IAMF, audio debug).
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from minimappr.api.rust_dsp_manifests import LocalizedClassifierRenderRequest
from minimappr.core.fusion_stage_carriers import EventCandidate, LocalizationBranch, LocalizedCandidate
from minimappr.core.localized_handoff_duplicate_merge import (
    CrossReporterDuplicateMerger,
    LocalizedClusterEmission,
)
from minimappr.core.localized_handoff_geometry import (
    SOLVE_REJECTED_LEGACY_SENSOR_MAP_UNAVAILABLE,
    legacy_pair_indices_exceed_registry_sensors,
)
from minimappr.core.localized_render_handoff_hooks import (
    SOLVE_ORIGIN_RUST_CLASSIFIER_RENDER,
    HandoffSolveGeometry,
    LocalizationSeamCollaborators,
)
from minimappr.core.localized_render_localization import (
    LOCALIZATION_REJECTED_OUT_OF_RANGE_REASON,
    HandoffGatingOutcome,
    HandoffLocalizationEstimate,
    gate_handoff_localization,
    resolve_default_handoff_estimate,
)
from minimappr.models import NodeSpec

if TYPE_CHECKING:
    from minimappr.core.fusion_node import FusionNode

OMNI_RENDER_LOCALIZATION_METHOD = "rust_classifier_render_fallback"
RUST_CLASSIFIER_RENDER_AUDIO_QUALITY_SOURCE = "rust_classifier_render"
BIRDNET_OMNI_FALLBACK_RENDER_KIND = "birdnet_omni_fallback"
CROSS_REPORTER_DUPLICATE_MERGED_REASON = "cross_reporter_duplicate_merged"


class LocalizedRenderHandoff:
    def __init__(self, host: "FusionNode", collaborators: LocalizationSeamCollaborators) -> None:
        self._host = host
        self.collaborators = collaborators
        self._cross_reporter_duplicate_merger = CrossReporterDuplicateMerger()

    async def ingest(self, payload: LocalizedClassifierRenderRequest) -> None:
        host = self._host
        if host._stopping:
            raise ValueError("Fusion node is stopping")

        server_received_ns = time.time_ns()
        normalized_audio = _validated_mono_render_audio(payload)
        node = self._node_with_local_position(payload)

        # A render may retain the DSP localization payload for diagnostics even
        # after the sidecar deliberately selected an omni render (e.g. a
        # low-confidence SRP-PHAT). The render contract is authoritative so an
        # ill-conditioned solve cannot become a far-away localized track.
        reporting_modality = payload.reporting_modality
        if payload.render_kind == BIRDNET_OMNI_FALLBACK_RENDER_KIND or payload.fallback_reason is not None:
            reporting_modality = "omni"
        if reporting_modality == "omni" and node.position_m is None:
            raise ValueError("Omni classifier render requires node.position_m")

        registered_node, geometry = await self._register_node_and_resolve_geometry(
            payload,
            node=node,
            server_received_ns=server_received_ns,
        )
        if geometry.rejected_reason is not None and reporting_modality == "localized":
            host._record_silent_drop(stage="localization", reason=geometry.rejected_reason, source_node_id=node.id)
            reporting_modality = "omni"
        estimate = self.collaborators.solution_resolver.resolve_handoff_solution(
            payload=payload,
            geometry=geometry,
            reporting_modality=reporting_modality,
        )
        if estimate is None:
            estimate = resolve_default_handoff_estimate(host, payload, geometry, reporting_modality)
        gating = gate_handoff_localization(
            host,
            payload=payload,
            geometry=geometry,
            estimate=estimate,
            reporting_modality=reporting_modality,
            source_node_id=node.id,
        )
        reporting_modality = gating.reporting_modality

        rust_extra_features = _rust_render_features(payload, gating, geometry)
        if reporting_modality == "omni":
            _replace_estimate_with_node_position(host, estimate, registered_node)
        localization_branch = await self._localization_branch_with_bearing_fusion(
            payload,
            node=node,
            geometry=geometry,
            estimate=estimate,
            gating=gating,
            reporting_modality=reporting_modality,
            normalized_audio=normalized_audio,
        )
        localized_product = self._build_localized_product(
            payload,
            node=node,
            geometry=geometry,
            localization_branch=localization_branch,
            normalized_audio=normalized_audio,
            rust_extra_features=rust_extra_features,
        )
        host._record_degraded_audio_quality_metrics(
            candidate=localized_product.candidate,
            source_window_type=localized_product.classification_audio_quality_source,
            audio_quality=localized_product.classification_audio_quality,
        )
        self.collaborators.provenance_recorder.record_handoff_provenance(
            payload=payload,
            geometry=geometry,
            localized_product=localized_product,
            reporting_modality=reporting_modality,
        )
        await self.collaborators.position_observation_sink.observe_localized_solve(
            localized_product=localized_product,
            reporting_modality=reporting_modality,
            solve_origin=SOLVE_ORIGIN_RUST_CLASSIFIER_RENDER,
        )
        await self._classify_and_report(
            payload,
            node=node,
            localized_product=localized_product,
            reporting_modality=reporting_modality,
            reference_sensor=geometry.reference_sensor,
            capability_tier=gating.capability_tier,
            normalized_audio=normalized_audio,
            rust_extra_features=rust_extra_features,
        )

    def _node_with_local_position(self, payload: LocalizedClassifierRenderRequest) -> NodeSpec:
        metadata = dict(payload.node.metadata)
        metadata["time_quality"] = payload.time_quality.value
        node = payload.node.model_copy(update={"metadata": metadata})
        if node.position_m is None and node.position_geo is not None:
            node = node.model_copy(
                update={"position_m": self._host.coordinate_frame.geo_to_local(node.position_geo)}
            )
        return node

    async def _register_node_and_resolve_geometry(
        self,
        payload: LocalizedClassifierRenderRequest,
        *,
        node: NodeSpec,
        server_received_ns: int,
    ) -> tuple[NodeSpec, HandoffSolveGeometry]:
        """Register the reporter; return its registry spec (operator overrides
        applied) and the solve geometry."""
        host = self._host
        # Renders are derived products, not raw sensor frames: event-time
        # semantics stay for localization/classification, but heartbeat
        # freshness uses server receipt time so online/offline status reflects
        # connectivity rather than source clock skew.
        runtime = await host.registry.upsert(node, server_received_ns)
        position_geo = node.position_geo
        if position_geo is None and node.position_m is not None:
            position_geo = host.coordinate_frame.local_to_geo(node.position_m)
        await host._persist_node_registration_if_changed(
            node,
            last_seen_ns=server_received_ns,
            position_geo=position_geo,
        )
        sensor_descriptors = await host.registry.sensors_for_node(node.id)
        selected_positions = {
            descriptor.sensor_id: descriptor.position_m for descriptor in sensor_descriptors
        }
        selected_sensor_ids = list(runtime.sensor_ids)
        if not selected_sensor_ids:
            raise ValueError(f"Node {node.id!r} did not register any sensors")
        registry_geometry = HandoffSolveGeometry(
            selected_sensor_ids=selected_sensor_ids,
            selected_positions=selected_positions,
            reference_sensor=selected_sensor_ids[0],
        )
        # Checked here, not in a seam, so no collaborator can ever hand the
        # solver a partial index map.
        if legacy_pair_indices_exceed_registry_sensors(payload, registry_geometry):
            registry_geometry = replace(
                registry_geometry, rejected_reason=SOLVE_REJECTED_LEGACY_SENSOR_MAP_UNAVAILABLE
            )
        resolved_geometry = self.collaborators.geometry_resolver.resolve_handoff_geometry(
            payload=payload,
            node=runtime.spec,
            registry_geometry=registry_geometry,
        )
        return runtime.spec, (resolved_geometry if resolved_geometry is not None else registry_geometry)

    async def _localization_branch_with_bearing_fusion(
        self,
        payload: LocalizedClassifierRenderRequest,
        *,
        node: NodeSpec,
        geometry: HandoffSolveGeometry,
        estimate: HandoffLocalizationEstimate,
        gating: HandoffGatingOutcome,
        reporting_modality: str,
        normalized_audio: np.ndarray,
    ) -> LocalizationBranch:
        branch = LocalizationBranch(
            localization_position_m=estimate.position_m,
            localization_confidence=estimate.confidence,
            localization_gdop=estimate.gdop,
            reference_sensor=geometry.reference_sensor,
            reference_signal=normalized_audio,
            classification_reference_signal=normalized_audio,
            tdoa_s={},
            localization_method=estimate.method,
            capability_tier=gating.capability_tier,
            localization_position_covariance_m2=estimate.position_covariance_m2,
            localization_range_observability=estimate.range_observability,
            localization_residual_rms_seconds=estimate.residual_rms_seconds,
            localization_range_projection_mode=estimate.range_projection_mode,
        )
        if reporting_modality != "localized":
            return branch
        # Bearing fusion may upgrade the branch (fused position, refined range,
        # new covariance/confidence/method); downstream reads the branch, so the
        # product and the reporting branch carry the fused values.
        return await self._host._maybe_fuse_multi_node_bearing(
            branch,
            node_id=node.id,
            contributing_centroid_m=gating.contributing_centroid_m,
            event_time_ns=payload.event_time_ns,
            sound_speed_mps=payload.localization_sound_speed_mps,
        )

    def _build_localized_product(
        self,
        payload: LocalizedClassifierRenderRequest,
        *,
        node: NodeSpec,
        geometry: HandoffSolveGeometry,
        localization_branch: LocalizationBranch,
        normalized_audio: np.ndarray,
        rust_extra_features: dict[str, Any],
    ) -> LocalizedCandidate:
        reference_sensor = geometry.reference_sensor
        rust_audio_quality = (
            {reference_sensor: payload.audio_quality} if payload.audio_quality is not None else {}
        )
        candidate = EventCandidate(
            id=f"rust-{payload.manifest_id}",
            source_node_id=node.id,
            event_time_ns=payload.event_time_ns,
            sample_rate_hz=payload.sample_rate_hz,
            source_type=payload.source_type,
            time_quality=payload.time_quality,
            source_observation_ids=list(payload.source_observation_ids),
        )
        return LocalizedCandidate(
            candidate=candidate,
            localization_branch=localization_branch,
            selected_sensor_ids=geometry.selected_sensor_ids,
            selected_windows={},
            selected_positions=geometry.selected_positions,
            classification_selected_windows={},
            omni_reference_sensor=reference_sensor,
            omni_reference_signal=normalized_audio,
            omni_position_m=localization_branch.localization_position_m,
            omni_classification_reference_signal=normalized_audio,
            localization_audio_quality=rust_audio_quality,
            classification_audio_quality=rust_audio_quality,
            classification_audio_quality_source=RUST_CLASSIFIER_RENDER_AUDIO_QUALITY_SOURCE,
            environment=dict(payload.environment),
            extra_classification_features=rust_extra_features,
        )

    async def _classify_and_report(
        self,
        payload: LocalizedClassifierRenderRequest,
        *,
        node: NodeSpec,
        localized_product: LocalizedCandidate,
        reporting_modality: str,
        reference_sensor: str,
        capability_tier: str,
        normalized_audio: np.ndarray,
        rust_extra_features: dict[str, Any],
    ) -> None:
        host = self._host
        if host._classification_chunking_policy is not None and payload.authoritative_classification is None:
            if host._should_suppress_chunked_classification(localized_product):
                host._metrics.birdnet_chunk_dispatches_suppressed += 1
                return

        if payload.authoritative_classification is not None:
            classified = await host._classification_orchestrator.adopt_authoritative_classification(
                classification=payload.authoritative_classification,
                event_time_ns=payload.event_time_ns,
                classification_signal=normalized_audio,
                sample_rate_hz=payload.sample_rate_hz,
            )
            classified.classification.features["rust_classification_authoritative"] = True
        else:
            classified = await host._classification_orchestrator.classify_omni_only(
                reference_signal=normalized_audio,
                sample_rate_hz=payload.sample_rate_hz,
                event_time_ns=payload.event_time_ns,
            )
        classified.classification.features.update(rust_extra_features)
        branch = localized_product.localization_branch
        cluster_emission_key = self._localized_cluster_emission_key(
            payload, reporting_modality=reporting_modality, label=classified.classification.label
        )
        if cluster_emission_key is not None and self._merges_into_earlier_reporter(
            payload, cluster_emission_key=cluster_emission_key, position_m=branch.localization_position_m
        ):
            return
        await host._maybe_trigger_speech_capture(
            node_id=node.id,
            sensor_id=reference_sensor,
            event_time_ns=payload.event_time_ns,
            scores=classified.classification.scores,
        )
        detection_product = await host._assemble_reporting_branch(
            product=localized_product,
            classified=classified,
            reporting_modality=reporting_modality,
            localization_position_m=branch.localization_position_m,
            localization_confidence=branch.localization_confidence,
            localization_gdop=branch.localization_gdop,
            localization_position_covariance_m2=branch.localization_position_covariance_m2,
            localization_range_observability=branch.localization_range_observability,
            localization_residual_rms_seconds=branch.localization_residual_rms_seconds,
            localization_range_projection_mode=branch.localization_range_projection_mode,
            reference_sensor=reference_sensor,
            reference_signal=normalized_audio,
            tdoa_s={},
            capability_tier=capability_tier,
            localization_method=branch.localization_method,
        )
        if detection_product is None:
            return
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
        await host._process_rules_and_delivery(detection_product)

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


def _validated_mono_render_audio(payload: LocalizedClassifierRenderRequest) -> np.ndarray:
    normalized_audio = np.asarray(payload.decoded_audio, dtype=np.float32)
    if normalized_audio.ndim != 1 or normalized_audio.size == 0:
        raise ValueError("Localized render audio must be a non-empty mono signal")
    if payload.sample_rate_hz <= 0:
        raise ValueError("Localized render sample_rate_hz must be positive")
    return normalized_audio


def _rust_render_features(
    payload: LocalizedClassifierRenderRequest,
    gating: HandoffGatingOutcome,
    geometry: HandoffSolveGeometry,
) -> dict[str, Any]:
    rust_extra_features: dict[str, Any] = {"rust_manifest_id": payload.manifest_id}
    if payload.render_kind is not None:
        rust_extra_features["rust_render_kind"] = payload.render_kind
    if payload.render_start_ns is not None:
        rust_extra_features["rust_render_start_ns"] = int(payload.render_start_ns)
    if payload.render_end_ns is not None:
        rust_extra_features["rust_render_end_ns"] = int(payload.render_end_ns)
    if payload.fallback_reason is not None:
        rust_extra_features["rust_fallback_reason"] = payload.fallback_reason
    if geometry.rejected_reason is not None:
        rust_extra_features["localization_rejected_reason"] = geometry.rejected_reason
    elif gating.rejected_out_of_range:
        rust_extra_features["localization_rejected_reason"] = LOCALIZATION_REJECTED_OUT_OF_RANGE_REASON
    elif gating.rejected_low_quality_reason is not None:
        rust_extra_features["localization_rejected_reason"] = gating.rejected_low_quality_reason
    return rust_extra_features


def _replace_estimate_with_node_position(
    host: "FusionNode",
    estimate: HandoffLocalizationEstimate,
    node: NodeSpec,
) -> None:
    """Position an omni detection at its source node; it cannot create a track.

    ``node`` is the registry spec, so an operator position override wins over
    the packet's reported position. A render that arrived "localized" can be
    degraded to omni by the gates, so it needs the same node-position guard as
    a render that arrived omni. The raw sidecar localization survives only as
    provenance in the features.
    """
    if node.position_m is None:
        raise ValueError("Omni classifier render requires node.position_m")
    estimate.position_m = tuple(float(value) for value in node.position_m)
    estimate.confidence = host.fusion_config.fallback_localization_confidence
    estimate.gdop = float("inf")
    estimate.position_covariance_m2 = None
    estimate.range_observability = None
    estimate.residual_rms_seconds = None
    estimate.range_projection_mode = None
    estimate.method = OMNI_RENDER_LOCALIZATION_METHOD
