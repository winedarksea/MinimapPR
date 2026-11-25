"""What a classification that delivered no detection means for its solve's fix.

Two different outcomes produce an empty result, and the hold rule
(``position_observation_pipeline``) must tell them apart:

- "Classified as nothing": the classifier ran and found nothing trackable
  (no label, a merged or zone-suppressed result). A held fix is dropped and a
  provisional track only this fix started is retracted.
- "Could not classify": no render or clip existed (an array too small to
  beamform, a beamform exception). That is no verdict about the source, so the
  fix is left to the classification worker's release, which applies a held fix
  geometry-only, exactly like a backend failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from minimappr.core.position_observation import position_observation_solve_id

if TYPE_CHECKING:
    from minimappr.core.fusion_node import FusionNode
    from minimappr.core.fusion_stage_carriers import LocalizedCandidate

EMPTY_CLASSIFICATION_DROP_REASON = "empty_classification"
CLASSIFICATION_UNAVAILABLE_DROP_REASON = "classification_unavailable"


async def resolve_classification_without_detections(host: "FusionNode", product: "LocalizedCandidate") -> None:
    """Record the empty outcome; on "classified as nothing" drop the held fix and its lone provisional track."""
    host._record_silent_drop(
        stage="classification",
        reason=(
            CLASSIFICATION_UNAVAILABLE_DROP_REASON
            if product.classification_unavailable
            else EMPTY_CLASSIFICATION_DROP_REASON
        ),
        candidate_id=product.candidate.id,
        event_time_ns=product.candidate.event_time_ns,
        source_node_id=product.candidate.source_node_id,
    )
    if product.classification_unavailable:
        return
    position_observations = host._position_observations()
    if position_observations is not None:
        await position_observations.classification_found_nothing_trackable(position_observation_solve_id(product))


__all__ = [
    "CLASSIFICATION_UNAVAILABLE_DROP_REASON",
    "EMPTY_CLASSIFICATION_DROP_REASON",
    "resolve_classification_without_detections",
]
