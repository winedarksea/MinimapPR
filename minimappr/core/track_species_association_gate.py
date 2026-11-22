"""Species-compatibility term for track association (Field Mode S-fixes tracking).

The association gate is geometric plus a category check (wildlife vs
wildlife), so two birds of different species 7 m apart share a track: the S
gate's real-audio scene lost its Scarlet Tanager this way (a grosbeak fix
associated to the tanager's perched track and the filter restarted there).

A detection may not associate to a track whose belief is *confident*
(top species at least ``confident_probability``) in another species: the
detection's top species claims at least ``detection_claim_probability``, the
track's belief gives that species less than ``confident_probability``, and the
detection gives the track's species less than ``detection_claim_probability``.

The detection-side floor is deliberately lower than the display floor. On the
S gate's real-audio scene, sub-display (two-threshold) tanager bouts joined
the grosbeak's track 7 m away, built a tanager belief inside it, and then
let the confident tanager bouts in too. Refused, they start (and fill) the
tanager's own, initially hidden, track. Labels with no species claim at all
(unknown, abstentions, scores under the claim floor) stay wildcards, and a
track that is not yet confident accepts anything the geometry picks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from minimappr.models import TrackState


@dataclass(frozen=True, slots=True)
class SpeciesAssociationGateConfig:
    enabled: bool
    # A track belief at or above this is a confident species (the display floor).
    confident_probability: float
    # A detection score at or above this is a species claim that can contradict
    # a confident track (default: the two-threshold localize-to-track floor).
    detection_claim_probability: float


def _confident_top(probability_by_label: dict[str, float], floor: float) -> str | None:
    if not probability_by_label:
        return None
    label, probability = max(probability_by_label.items(), key=lambda item: item[1])
    return label if probability >= floor else None


def species_claims_contradict(
    detection_evidence: dict[str, float],
    track_belief: dict[str, float],
    confident_probability: float,
    detection_claim_probability: float,
) -> bool:
    """True when a detection's species claim contradicts a confident track belief."""
    detection_top = _confident_top(detection_evidence, detection_claim_probability)
    track_top = _confident_top(track_belief, confident_probability)
    if detection_top is None or track_top is None or detection_top == track_top:
        return False
    track_supports_detection = track_belief.get(detection_top, 0.0) >= confident_probability
    detection_supports_track = detection_evidence.get(track_top, 0.0) >= detection_claim_probability
    return not (track_supports_detection or detection_supports_track)


def species_compatible_tracks(
    tracks: Iterable[TrackState],
    *,
    detection_evidence: dict[str, float],
    belief_for_track: Callable[[str], dict[str, float]],
    config: SpeciesAssociationGateConfig,
) -> list[TrackState]:
    """*tracks* minus those whose confident belief contradicts the detection."""
    candidates = list(tracks)
    if not config.enabled or not detection_evidence:
        return candidates
    return [
        track
        for track in candidates
        if not species_claims_contradict(
            detection_evidence,
            belief_for_track(track.id),
            config.confident_probability,
            config.detection_claim_probability,
        )
    ]
