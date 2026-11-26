"""Ingest of Rust-sidecar classifier renders into the fusion pipeline.

``FusionNode.ingest_localized_render`` delegates here. A render arrives with
the sidecar's localization already attached; this collaborator registers the
node, resolves solve geometry and the position estimate, gates it exactly like
the Python solver path, then classifies and reports (``LocalizedRenderReporter``).
Most renders are reported synchronously, so callers observe the detection as
soon as ingest returns; a point-network render's clip classification (P7) is
queued on the classification stage instead, off the SSE consumer.

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
from minimappr.core.handoff_spatial_status import (
    CLASSIFICATION_AUDIO_MODALITY_FEATURE_KEY,
    LOCALIZATION_SOLVE_BEARING_FEATURE_KEY,
    handoff_spatial_modality,
    render_audio_modality,
    solve_time_bearing,
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
from minimappr.core.localized_render_reporting import (
    CROSS_REPORTER_DUPLICATE_MERGED_REASON,
    LocalizedRenderReporter,
    RenderReport,
)
from minimappr.core.point_network_call_requests import localized_call_for_rust_solve
from minimappr.models import NodeSpec

if TYPE_CHECKING:
    from minimappr.core.fusion_node import FusionNode

RUST_CLASSIFIER_RENDER_AUDIO_QUALITY_SOURCE = "rust_classifier_render"


@dataclass(slots=True)
class _HandoffSolve:
    """A render's (or event's) solve placed and gated, before any audio work."""

    node: NodeSpec
    registered_node: NodeSpec
    geometry: HandoffSolveGeometry
    estimate: HandoffLocalizationEstimate
    gating: HandoffGatingOutcome
    # Spatial status after the gates; the clip's own modality is ``render_audio_modality``.
    reporting_modality: str
    rust_extra_features: dict[str, Any]


class LocalizedRenderHandoff:
    def __init__(self, host: "FusionNode", collaborators: LocalizationSeamCollaborators) -> None:
        self._host = host
        self.collaborators = collaborators
        self._reporter = LocalizedRenderReporter(host)
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
        audio_modality = render_audio_modality(payload)
        solve.rust_extra_features[CLASSIFICATION_AUDIO_MODALITY_FEATURE_KEY] = audio_modality
        if solve.reporting_modality == "omni":
            replace_estimate_with_node_position(host, solve.estimate, solve.registered_node)
        elif audio_modality == "omni":
            host._metrics.omni_audio_render_bearing_carried_count += 1
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
        classification_worker_owns_product = False
        try:
            self._register_localized_call(payload, solve, localized_product)
            classification_worker_owns_product = await self._reporter.classify_and_report(
                RenderReport(
                    payload=payload,
                    node=solve.node,
                    geometry=solve.geometry,
                    localized_product=localized_product,
                    reporting_modality=solve.reporting_modality,
                    capability_tier=solve.gating.capability_tier,
                    normalized_audio=normalized_audio,
                    rust_extra_features=solve.rust_extra_features,
                )
            )
        finally:
            if not classification_worker_owns_product:
                # A held fix this render's classification never resolved (suppressed, failed, not queued).
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

        # The clip's audio modality is not the solve's spatial status: an omni
        # fallback render (e.g. a low-confidence SRP-PHAT) whose solve still
        # carries a steering direction is re-solved and gated exactly like a
        # localized render, so the gates, not the sidecar's render threshold,
        # decide whether it is placed.
        reporting_modality = handoff_spatial_modality(payload)
        if render_audio_modality(payload) == "omni" and node.position_m is None:
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
        self._record_solve_time_bearing(solve, localized_product)
        self.collaborators.provenance_recorder.record_handoff_provenance(
            payload=payload,
            geometry=solve.geometry,
            localized_product=localized_product,
            reporting_modality=solve.reporting_modality,
        )
        return localized_product

    def _record_solve_time_bearing(self, solve: "_HandoffSolve", localized_product: LocalizedCandidate) -> None:
        """The gated solve's ray, before bearing fusion, on a placed solve only
        (an omni outcome sits at its node and states no direction)."""
        centroid_m = solve.gating.contributing_centroid_m
        if solve.reporting_modality != "localized" or centroid_m is None:
            return
        bearing = solve_time_bearing(
            reporting_node_id=solve.node.id,
            contributing_centroid_m=centroid_m,
            position_m=solve.estimate.position_m,
            covariance_m2=solve.estimate.position_covariance_m2,
            range_projection_mode=solve.estimate.range_projection_mode,
            coordinate_frame=self._host.coordinate_frame,
            settings=self._host.settings,
        )
        if bearing is not None:
            localized_product.extra_classification_features[LOCALIZATION_SOLVE_BEARING_FEATURE_KEY] = (
                bearing.model_dump(mode="json")
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


__all__ = ["CROSS_REPORTER_DUPLICATE_MERGED_REASON", "LocalizedRenderHandoff"]
