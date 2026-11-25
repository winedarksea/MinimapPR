"""Event-only ``localization_result`` manifests as position-only solve requests (P5).

The sidecar publishes a ``localization_result`` for every event solve, but a
``classifier_render`` only every render interval (tens of seconds). Before P5
an event-only result was a heartbeat, so a continuously audible source moved
its track once per render. Each one that carries a usable solve now reaches
the observation sink (one tracker update per solve; the render that covers the
solve later repeats its ``solve_id`` and attaches only the label).

The request reuses ``LocalizedClassifierRenderRequest`` so the render handoff's
geometry, solution and gating apply unchanged; it carries no audio.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from minimappr.api.rust_dsp_manifests import (
    LocalizedClassifierRenderRequest,
    load_source_context_from_channel_manifest,
    sidecar_localization_request_fields,
)
from minimappr.api.rust_localization_solve_contract import SolveAttemptKind
from minimappr.api.rust_solve_contract_degradation import parse_rust_localization_solve_or_degrade


def load_event_localization_request(manifest_payload: dict[str, Any]) -> LocalizedClassifierRenderRequest | None:
    """The solve of an event-only ``localization_result``, or ``None`` when it has none.

    ``None`` covers a legacy result without provenance (it cannot be matched to
    its later render), a ``not_attempted`` heartbeat, a result without a
    position, and a solve that breaks the contract (the render path degrades
    and counts that one).
    """
    localization_payload = manifest_payload.get("localization")
    node_context_payload = manifest_payload.get("node_context")
    if not isinstance(localization_payload, dict) or not isinstance(node_context_payload, dict):
        return None
    if not isinstance(localization_payload.get("position_m"), list):
        return None
    manifest_id = str(manifest_payload.get("manifest_id") or "")
    localization_solve, solve_contract_violation = parse_rust_localization_solve_or_degrade(
        localization_payload, manifest_id=manifest_id
    )
    if localization_solve is None or solve_contract_violation is not None:
        return None
    provenance = localization_solve.provenance
    if provenance.attempt_kind is SolveAttemptKind.NOT_ATTEMPTED:
        return None
    node, event_time_ns, source_type, time_quality, environment = load_source_context_from_channel_manifest(
        node_context_payload, solve_window_center_time_ns=provenance.window_center_time_ns
    )
    return LocalizedClassifierRenderRequest(
        manifest_id=manifest_id,
        node=node,
        event_time_ns=event_time_ns,
        sample_rate_hz=provenance.solve_sample_rate_hz,
        decoded_audio=np.zeros(0, dtype=np.float32),
        **sidecar_localization_request_fields(localization_payload),
        source_type=source_type,
        reporting_modality="localized",
        time_quality=time_quality,
        environment=environment,
        localization_solve=localization_solve,
    )
