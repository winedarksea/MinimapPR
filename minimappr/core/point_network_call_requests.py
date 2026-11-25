"""Build P7 localized calls and clip requests from each ingest path's solve.

Both paths anchor the solve window on the node whose trigger it is centred on:
the Rust event span is "still centred on the trigger"
(``localization_event_span.rs``) at the reporting node, and the Python
candidate's ``event_time_ns`` comes from the triggering node's own buffer
(``core/ingest.py``). That node is the alignment reference; a call's audio at
any other sensor is the same span shifted by the propagation difference.

Base clip (what the path classifies without P7, on the reference timeline):
- Rust render: ``render_start_ns..render_end_ns`` (else the render's length
  centred on the solve window);
- Python ingest: the trailing classification window ending at the event when
  it is longer than the localization window, else the centred localization
  window (``FusionNode._classification_windows_for_event``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from minimappr.core.point_network_call_classification import (
    POINT_NETWORK_CLIP_FEATURE_KEY,
    DefaultClassification,
    PointNetworkCallRequest,
    is_point_network,
)
from minimappr.core.point_network_localized_calls import NANOSECONDS_PER_SECOND, LocalizedCall, seconds_to_nanoseconds
from minimappr.core.position_observation import position_observation_solve_id

if TYPE_CHECKING:
    from minimappr.api.rust_dsp_manifests import LocalizedClassifierRenderRequest
    from minimappr.core.classification import ClassifiedResult
    from minimappr.core.fusion_node import FusionNode
    from minimappr.core.fusion_stage_carriers import LocalizedCandidate
    from minimappr.core.localized_render_handoff_hooks import HandoffSolveGeometry

logger = logging.getLogger(__name__)


def _site_position(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(3)


def _stated_covariance(branch) -> np.ndarray | None:
    covariance = branch.localization_position_covariance_m2
    return None if covariance is None else np.asarray(covariance, dtype=np.float64)


# -- Rust classifier render / event-only solve -----------------------------------


def _rust_alignment_reference(
    payload: "LocalizedClassifierRenderRequest", geometry: "HandoffSolveGeometry"
) -> tuple[str, np.ndarray]:
    """A reporting-node sensor of the solve (contributing or not), else the
    geometry's reference sensor."""
    positions = geometry.ordered_pair_sensor_positions()
    solve = payload.localization_solve
    if solve is not None:
        reporting_node_id = solve.provenance.reporting_node_id
        for sensor in solve.provenance.sensors:
            if sensor.node_id == reporting_node_id and sensor.sensor_id in positions:
                return sensor.sensor_id, _site_position(positions[sensor.sensor_id])
    return geometry.reference_sensor, _site_position(positions[geometry.reference_sensor])


def _rust_base_clip_span_ns(payload: "LocalizedClassifierRenderRequest") -> tuple[int, int]:
    if payload.render_start_ns is not None and payload.render_end_ns is not None:
        return payload.render_start_ns, payload.render_end_ns
    duration_ns = int(round(payload.decoded_audio.size * NANOSECONDS_PER_SECOND / payload.sample_rate_hz))
    start_ns = payload.event_time_ns - duration_ns // 2
    return start_ns, start_ns + duration_ns


def localized_call_for_rust_solve(
    host: "FusionNode",
    *,
    payload: "LocalizedClassifierRenderRequest",
    geometry: "HandoffSolveGeometry",
    localized_product: "LocalizedCandidate",
    reporting_modality: str,
) -> LocalizedCall | None:
    branch = localized_product.localization_branch
    if reporting_modality != "localized" or branch is None:
        return None
    source_position_m = _site_position(branch.localization_position_m)
    reference_sensor_id, reference_position_m = _rust_alignment_reference(payload, geometry)
    solve = payload.localization_solve
    if solve is not None:
        half_window_ns = seconds_to_nanoseconds(solve.provenance.window_seconds / 2.0)
        span_start_ns = solve.provenance.window_center_time_ns - half_window_ns
        span_end_ns = solve.provenance.window_center_time_ns + half_window_ns
    else:
        span_start_ns, span_end_ns = _rust_base_clip_span_ns(payload)
    sound_speed_mps = payload.localization_sound_speed_mps or host.environment_provider.get_speed_of_sound(
        tuple(float(axis) for axis in source_position_m)
    )
    return LocalizedCall(
        call_id=position_observation_solve_id(localized_product),
        source_position_m=source_position_m,
        alignment_reference_sensor_id=reference_sensor_id,
        alignment_reference_position_m=reference_position_m,
        event_span_start_ns=span_start_ns,
        event_span_end_ns=span_end_ns,
        sound_speed_mps=float(sound_speed_mps),
        position_covariance_m2=_stated_covariance(branch),
        cluster_id=None if solve is None else solve.provenance.cluster_id,
    )


async def _rust_render_sensor_node_ids(
    host: "FusionNode", payload: "LocalizedClassifierRenderRequest", geometry: "HandoffSolveGeometry"
) -> dict[str, str]:
    """Each selected sensor's node: the solve's provenance, else (no solve) the registry.

    The manifest geometry resolver selects only provenance sensors, so a
    selected sensor missing from them means a geometry seam broke that
    invariant. It is left out, counted and logged rather than raising (which
    lost the render); the caller then declines P7, as the Python path does for
    a sensor the registry does not know: it could be a second sensor of a
    listed node, so the solve is not provably a point network.
    """
    selected_sensor_ids = list(geometry.selected_sensor_ids)
    solve = payload.localization_solve
    if solve is None:
        resolved = await host.registry.sensor_node_ids(selected_sensor_ids)
    else:
        resolved = {sensor.sensor_id: sensor.node_id for sensor in solve.provenance.sensors}
    unresolved_sensor_ids = [sensor_id for sensor_id in selected_sensor_ids if sensor_id not in resolved]
    if unresolved_sensor_ids:
        host._metrics.point_network_request_unresolved_sensor_count += len(unresolved_sensor_ids)
        logger.warning(
            "Rust render selected a sensor with no known node; not classified as a point network",
            extra={"manifest_id": payload.manifest_id, "unresolved_sensor_ids": unresolved_sensor_ids},
        )
    return {sensor_id: resolved[sensor_id] for sensor_id in selected_sensor_ids if sensor_id in resolved}


async def point_network_request_for_rust_render(
    host: "FusionNode",
    *,
    payload: "LocalizedClassifierRenderRequest",
    geometry: "HandoffSolveGeometry",
    localized_product: "LocalizedCandidate",
    reporting_modality: str,
) -> PointNetworkCallRequest | None:
    call = localized_call_for_rust_solve(
        host,
        payload=payload,
        geometry=geometry,
        localized_product=localized_product,
        reporting_modality=reporting_modality,
    )
    if call is None:
        return None
    solve = payload.localization_solve
    sensor_node_ids = await _rust_render_sensor_node_ids(host, payload, geometry)
    if len(sensor_node_ids) != len(geometry.selected_sensor_ids) or not is_point_network(sensor_node_ids):
        return None
    base_start_ns, base_end_ns = _rust_base_clip_span_ns(payload)
    return PointNetworkCallRequest(
        call=call,
        sensor_node_ids=sensor_node_ids,
        sensor_positions_m={sensor_id: _site_position(geometry.selected_positions[sensor_id]) for sensor_id in sensor_node_ids},
        base_clip_start_ns=base_start_ns,
        clip_seconds=(base_end_ns - base_start_ns) / NANOSECONDS_PER_SECOND,
        # The sensors' own buffer rate (the solve's), which the mirrored raw
        # audio carries even if a render were produced at another rate.
        sample_rate_hz=solve.provenance.solve_sample_rate_hz if solve is not None else payload.sample_rate_hz,
        event_time_ns=payload.event_time_ns,
    )


async def classify_rust_render_call(
    host: "FusionNode",
    *,
    payload: "LocalizedClassifierRenderRequest",
    request: PointNetworkCallRequest,
    default_classification: DefaultClassification,
) -> "ClassifiedResult":
    """A point-network render's classification: the aligned point-node clip,
    else ``default_classification`` (the render itself: the sidecar's label
    when it sent one, which also stands if the clip classifier fails)."""
    classified = await host._point_network_call_classifier.classify(
        request,
        default_classification=default_classification,
        keep_default_on_classifier_failure=payload.authoritative_classification is not None,
    )
    clip_provenance = classified.classification.features.get(POINT_NETWORK_CLIP_FEATURE_KEY)
    authoritative = payload.authoritative_classification
    if authoritative is not None and isinstance(clip_provenance, dict) and "node_id" in clip_provenance:
        # The sidecar labelled the unaligned mix; keep its answer beside the clip's.
        clip_provenance["superseded_sidecar_label"] = authoritative.label
        clip_provenance["superseded_sidecar_confidence"] = float(authoritative.confidence)
    return classified


# -- Python ingest -----------------------------------------------------------------


async def _python_alignment_reference(
    host: "FusionNode", product: "LocalizedCandidate", sensor_node_ids: dict[str, str]
) -> tuple[str, np.ndarray] | None:
    """The triggering node's sensor (its selected one, else its first registered
    one), whether or not the solve selected it: ``event_time_ns`` is on that
    node's timeline, so any other node would shift every clip by the
    propagation difference. None when the registry has no sensor for it."""
    triggering_node_id = product.candidate.source_node_id
    for sensor_id in product.selected_sensor_ids:
        if sensor_node_ids.get(sensor_id) == triggering_node_id:
            return sensor_id, _site_position(product.selected_positions[sensor_id])
    descriptors = await host.registry.sensors_for_node(triggering_node_id)
    if not descriptors:
        return None
    return descriptors[0].sensor_id, _site_position(descriptors[0].position_m)


async def _python_localized_call_and_nodes(
    host: "FusionNode", product: "LocalizedCandidate"
) -> tuple[LocalizedCall, dict[str, str]] | None:
    branch = product.localization_branch
    if branch is None or not product.selected_sensor_ids:
        return None
    sensor_node_ids = await host.registry.sensor_node_ids(list(product.selected_sensor_ids))
    source_position_m = _site_position(branch.localization_position_m)
    reference = await _python_alignment_reference(host, product, sensor_node_ids)
    if reference is None:
        return None
    reference_sensor_id, reference_position_m = reference
    half_window_ns = seconds_to_nanoseconds(host.localization_config.localization_window_seconds / 2.0)
    event_time_ns = product.candidate.event_time_ns
    call = LocalizedCall(
        call_id=position_observation_solve_id(product),
        source_position_m=source_position_m,
        alignment_reference_sensor_id=reference_sensor_id,
        alignment_reference_position_m=reference_position_m,
        event_span_start_ns=event_time_ns - half_window_ns,
        event_span_end_ns=event_time_ns + half_window_ns,
        sound_speed_mps=float(host.environment_provider.get_speed_of_sound(branch.localization_position_m)),
        position_covariance_m2=_stated_covariance(branch),
    )
    return call, sensor_node_ids


async def localized_call_for_python_ingest_product(host: "FusionNode", product: "LocalizedCandidate") -> LocalizedCall | None:
    resolved = await _python_localized_call_and_nodes(host, product)
    return None if resolved is None else resolved[0]


async def point_network_request_for_python_ingest_product(
    host: "FusionNode", product: "LocalizedCandidate"
) -> PointNetworkCallRequest | None:
    # The onset-SRP stage attributes callers inside its own windows (LOCO);
    # its products keep that path.
    if product.onset_attribution_context is not None:
        return None
    resolved = await _python_localized_call_and_nodes(host, product)
    if resolved is None:
        return None
    call, sensor_node_ids = resolved
    if len(sensor_node_ids) != len(product.selected_sensor_ids) or not is_point_network(sensor_node_ids):
        return None
    localization_window_seconds = host.localization_config.localization_window_seconds
    classification_window_seconds = host.localization_config.classification_window_seconds
    event_time_ns = product.candidate.event_time_ns
    if classification_window_seconds > localization_window_seconds:
        clip_seconds = classification_window_seconds
        base_clip_start_ns = event_time_ns - seconds_to_nanoseconds(classification_window_seconds)
    else:
        clip_seconds = localization_window_seconds
        base_clip_start_ns = call.event_span_start_ns
    return PointNetworkCallRequest(
        call=call,
        sensor_node_ids=sensor_node_ids,
        sensor_positions_m={sensor_id: _site_position(product.selected_positions[sensor_id]) for sensor_id in sensor_node_ids},
        base_clip_start_ns=base_clip_start_ns,
        clip_seconds=clip_seconds,
        sample_rate_hz=product.candidate.sample_rate_hz,
        event_time_ns=event_time_ns,
    )


async def classify_python_ingest_call(
    host: "FusionNode", product: "LocalizedCandidate", *, default_classification: DefaultClassification
) -> "ClassifiedResult | None":
    """The product's classification: the aligned point-node clip for a point
    network, else ``default_classification`` (the beamformed render)."""
    request = await point_network_request_for_python_ingest_product(host, product)
    return await host._point_network_call_classifier.classify(request, default_classification=default_classification)


async def register_python_ingest_call(host: "FusionNode", product: "LocalizedCandidate") -> None:
    """Remember a localized Python-ingest solve before it is classified."""
    host._point_network_call_classifier.register_localized_call(
        await localized_call_for_python_ingest_product(host, product)
    )


__all__ = [
    "classify_python_ingest_call",
    "classify_rust_render_call",
    "register_python_ingest_call",
    "localized_call_for_python_ingest_product",
    "localized_call_for_rust_solve",
    "point_network_request_for_python_ingest_product",
    "point_network_request_for_rust_render",
]
