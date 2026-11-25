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

import asyncio
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from minimappr.api.rust_dsp_manifests import LocalizedClassifierRenderRequest
from minimappr.api.rust_solve_contract_degradation import SOLVE_CONTRACT_VIOLATION_FALLBACK_REASON
from minimappr.core.bounded_recent_ids import BoundedRecentIds
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
    SOLVE_ORIGIN_RUST_EVENT_LOCALIZATION,
    HandoffSolveGeometry,
    LocalizationSeamCollaborators,
)
from minimappr.core.localized_render_handoff_solve_features import (
    classify_render_audio,
    replace_estimate_with_node_position,
    rust_render_features,
    validated_mono_render_audio,
)
from minimappr.core.localized_render_localization import (
    HandoffGatingOutcome,
    HandoffLocalizationEstimate,
    gate_handoff_localization,
    python_cartesian_resolve_requested,
    resolve_handoff_estimate,
)
from minimappr.core.point_network_call_requests import classify_rust_render_call, localized_call_for_rust_solve
from minimappr.models import NodeSpec

if TYPE_CHECKING:
    from minimappr.core.fusion_node import FusionNode

RUST_CLASSIFIER_RENDER_AUDIO_QUALITY_SOURCE = "rust_classifier_render"
BIRDNET_OMNI_FALLBACK_RENDER_KIND = "birdnet_omni_fallback"
CROSS_REPORTER_DUPLICATE_MERGED_REASON = "cross_reporter_duplicate_merged"


@dataclass(slots=True)
class _HandoffSolve:
    """A render's (or event's) solve placed and gated, before any audio work."""

    node: NodeSpec
    registered_node: NodeSpec
    geometry: HandoffSolveGeometry
    estimate: HandoffLocalizationEstimate
    gating: HandoffGatingOutcome
    reporting_modality: str
    rust_extra_features: dict[str, Any]


class LocalizedRenderHandoff:
    def __init__(self, host: "FusionNode", collaborators: LocalizationSeamCollaborators) -> None:
        self._host = host
        self.collaborators = collaborators
        self._cross_reporter_duplicate_merger = CrossReporterDuplicateMerger()
        # A covering render repeats its event's solve: its rejections count once.
        self._processed_solve_ids = BoundedRecentIds(host.settings.position_observation_solve_id_memory_max_entries)
        # One off-loop solve at a time: the solver's metric counters are not thread-safe.
        self._off_loop_solve_lock = asyncio.Lock()

    async def ingest(self, payload: LocalizedClassifierRenderRequest) -> None:
        host = self._host
        if host._stopping:
            raise ValueError("Fusion node is stopping")

        normalized_audio = validated_mono_render_audio(payload)
        solve = await self._localize_solve(payload)
        if payload.fallback_reason == SOLVE_CONTRACT_VIOLATION_FALLBACK_REASON:
            # The loader already degraded the broken solve; count the lost localization.
            host._record_silent_drop(
                stage="localization", reason=SOLVE_CONTRACT_VIOLATION_FALLBACK_REASON, source_node_id=solve.node.id
            )
        if solve.reporting_modality == "omni":
            replace_estimate_with_node_position(host, solve.estimate, solve.registered_node)
        localized_product = await self._localized_product_for_solve(payload, solve, normalized_audio)
        host._record_degraded_audio_quality_metrics(
            candidate=localized_product.candidate,
            source_window_type=localized_product.classification_audio_quality_source,
            audio_quality=localized_product.classification_audio_quality,
        )
        await self.collaborators.position_observation_sink.observe_localized_solve(
            localized_product=localized_product,
            reporting_modality=solve.reporting_modality,
            solve_origin=SOLVE_ORIGIN_RUST_CLASSIFIER_RENDER,
        )
        self._register_localized_call(payload, solve, localized_product)
        try:
            await self._classify_and_report(
                payload,
                node=solve.node,
                geometry=solve.geometry,
                localized_product=localized_product,
                reporting_modality=solve.reporting_modality,
                reference_sensor=solve.geometry.reference_sensor,
                capability_tier=solve.gating.capability_tier,
                normalized_audio=normalized_audio,
                rust_extra_features=solve.rust_extra_features,
            )
        finally:
            # A held fix this render's classification never resolved (suppressed, failed).
            await host._release_held_position_without_classification(localized_product)

    async def observe_event_solve(self, payload: LocalizedClassifierRenderRequest) -> None:
        """An event-only solve (no render): position only, never classified or reported.

        Same geometry, solution and gates as a render, so the tracker sees the
        position the solve's later render would report. A solve the gates turn
        omni is not observed: its position would be the node's.
        """
        if self._host._stopping:
            return
        solve = await self._localize_solve(payload)
        if solve.reporting_modality != "localized":
            return
        localized_product = await self._localized_product_for_solve(payload, solve, payload.decoded_audio)
        await self.collaborators.position_observation_sink.observe_localized_solve(
            localized_product=localized_product,
            reporting_modality=solve.reporting_modality,
            solve_origin=SOLVE_ORIGIN_RUST_EVENT_LOCALIZATION,
        )
        # Usually ahead of the renders, so a point-network clip classified
        # first still sees a later caller its event solve already placed.
        self._register_localized_call(payload, solve, localized_product)

    def _register_localized_call(
        self, payload: LocalizedClassifierRenderRequest, solve: "_HandoffSolve", localized_product: LocalizedCandidate
    ) -> None:
        self._host._point_network_call_classifier.register_localized_call(
            localized_call_for_rust_solve(
                self._host,
                payload=payload,
                geometry=solve.geometry,
                localized_product=localized_product,
                reporting_modality=solve.reporting_modality,
            )
        )

    async def _localize_solve(self, payload: LocalizedClassifierRenderRequest) -> "_HandoffSolve":
        host = self._host
        server_received_ns = time.time_ns()
        node = self._node_with_local_position(payload)
        counts_solve_outcome = self._first_processing_of_solve(payload)

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
            if counts_solve_outcome:
                host._record_silent_drop(stage="localization", reason=geometry.rejected_reason, source_node_id=node.id)
            reporting_modality = "omni"
        resolve_args = (host, self.collaborators.solution_resolver, payload, geometry, reporting_modality)
        if python_cartesian_resolve_requested(host, payload, reporting_modality):
            # The Python re-solve takes ~10 ms per solve; keep it off the event loop.
            async with self._off_loop_solve_lock:
                estimate = await asyncio.to_thread(resolve_handoff_estimate, *resolve_args)
        else:
            estimate = resolve_handoff_estimate(*resolve_args)
        gating = gate_handoff_localization(
            host,
            payload=payload,
            geometry=geometry,
            estimate=estimate,
            reporting_modality=reporting_modality,
            source_node_id=node.id,
            records_rejections=counts_solve_outcome,
        )
        return _HandoffSolve(
            node=node,
            registered_node=registered_node,
            geometry=geometry,
            estimate=estimate,
            gating=gating,
            reporting_modality=gating.reporting_modality,
            rust_extra_features=rust_render_features(payload, gating, geometry),
        )

    def _first_processing_of_solve(self, payload: LocalizedClassifierRenderRequest) -> bool:
        solve = payload.localization_solve
        return solve is None or self._processed_solve_ids.add_if_new(solve.provenance.solve_id)

    async def _localized_product_for_solve(
        self, payload: LocalizedClassifierRenderRequest, solve: "_HandoffSolve", audio: np.ndarray
    ) -> LocalizedCandidate:
        localization_branch = await self._localization_branch_with_bearing_fusion(
            payload,
            node=solve.node,
            geometry=solve.geometry,
            estimate=solve.estimate,
            gating=solve.gating,
            reporting_modality=solve.reporting_modality,
            normalized_audio=audio,
        )
        localized_product = self._build_localized_product(
            payload,
            node=solve.node,
            geometry=solve.geometry,
            localization_branch=localization_branch,
            normalized_audio=audio,
            rust_extra_features=solve.rust_extra_features,
        )
        self.collaborators.provenance_recorder.record_handoff_provenance(
            payload=payload,
            geometry=solve.geometry,
            localized_product=localized_product,
            reporting_modality=solve.reporting_modality,
        )
        return localized_product

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
        geometry: HandoffSolveGeometry,
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

        # P7: a point-only solve is classified from its clearest node's aligned
        # clip, not the render (an unaligned mix of every node).
        classified = await classify_rust_render_call(
            host,
            payload=payload,
            geometry=geometry,
            localized_product=localized_product,
            reporting_modality=reporting_modality,
            default_classification=lambda: classify_render_audio(
                host._classification_orchestrator, payload, normalized_audio
            ),
        )
        classified.classification.features.update(rust_extra_features)
        branch = localized_product.localization_branch
        cluster_emission_key = self._localized_cluster_emission_key(
            payload, reporting_modality=reporting_modality, label=classified.classification.label
        )
        if cluster_emission_key is not None and self._merges_into_earlier_reporter(
            payload, cluster_emission_key=cluster_emission_key, position_m=branch.localization_position_m
        ):
            await host._label_track_of_unreported_classification(host._classification_tracker_feed(
                product=localized_product, classified=classified, source_node_id=node.id, capability_tier=capability_tier,
                reference_sensor=reference_sensor, tdoa_s={}, localization_position_m=branch.localization_position_m,
                localization_position_covariance_m2=branch.localization_position_covariance_m2,
            ))
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
