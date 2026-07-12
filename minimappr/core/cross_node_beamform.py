"""Cross-node audio for classification (BEAMFORMED_RENDER_CONTRACT.md Phase 6).

Runs in the Python fusion node — the only place with node registry
positions/orientations, sync grades, and the cross-node window machinery.
For a localized event, eligible nodes each render a band-split DAS beam
toward the source in their own local frame (with their own alias cutoff),
each beam is classified independently, and per-label confidences are fused
with **max + contributing-node evidence**. Noisy-OR fusion was rejected: it
inflates correlated observations of the same source. Incoherent audio
summation was rejected (blends noise floors/reverb) and coherent summation
is rejected on physics (inter-node phase is not usable at these baselines).

Degradation ladder: multi-node late fusion → single-node band-split beam →
omni.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable

import numpy as np

from minimappr.core.beamforming import BandSplitDasRenderer, BandSplitRenderConfig
from minimappr.models import NodeOrientation, SyncGrade
from minimappr.spatial_audio.geometry import (
    alias_cutoff_from_positions,
    orientation_rotation_matrix,
)

# Sync grades trusted for multi-node use. FREE-running clocks are excluded
# from cross-node geometry only — they still contribute omni evidence.
_MULTI_NODE_SYNC_GRADES = {SyncGrade.GPS_PPS, SyncGrade.PTP, SyncGrade.NTP}

# Position provenance values that mark a pseudo-position (e.g. the static
# config fallback). Excluded from geometric use; omni evidence only.
_UNTRUSTED_POSITION_SOURCES = {"gps_fallback", "static", "fallback_static"}


def position_source_trusted(position_source: str | None) -> bool:
    if position_source is None:
        return False
    value = position_source.strip().lower()
    return bool(value) and value not in _UNTRUSTED_POSITION_SOURCES


@dataclass(slots=True)
class CrossNodeBeamConfig:
    enabled: bool = False
    # Generous: 1/r² ranking keeps far nodes from winning anyway.
    max_range_m: float = 75.0
    # Caps classifier-subprocess cost per event.
    max_nodes: int = 3
    sound_speed_mps: float = 343.2
    render_config: BandSplitRenderConfig = field(default_factory=BandSplitRenderConfig)
    # "beamformed" (default) runs multi-node band-split late fusion; "nearest_node_omni"
    # classifies only the nearest node's raw omni mix (no beam, no late fusion).
    classification_audio_source: str = "beamformed"


@dataclass(slots=True)
class NodeAudioCandidate:
    """One node's audio window covering the event interval."""

    node_id: str
    node_position_m: np.ndarray  # (3,) world/site frame
    orientation: NodeOrientation | None
    mic_positions_local_m: np.ndarray  # (M, 3) node-local frame
    channels: np.ndarray  # (M, N) float, aligned to the event interval
    sample_rate_hz: int
    sync_grade: SyncGrade = SyncGrade.GPS_PPS
    position_trusted: bool = True
    covers_event: bool = True


@dataclass(slots=True)
class NodeBeamRender:
    node_id: str
    samples: np.ndarray
    alias_cutoff_hz: float
    distance_m: float


@dataclass(slots=True)
class FusedLabelResult:
    label: str
    confidence: float
    best_node_id: str
    per_node_confidence: dict[str, float]
    contributing_node_ids: list[str]


def propagation_delay_s(
    node_position_m: np.ndarray,
    source_position_m: tuple[float, float, float],
    sound_speed_mps: float = 343.2,
) -> float:
    """Window alignment term: |p_src − p_node| / c (millisecond-class)."""
    distance = float(np.linalg.norm(np.asarray(source_position_m, dtype=np.float64) - np.asarray(node_position_m, dtype=np.float64)))
    return distance / max(sound_speed_mps, 1.0)


def world_to_node_local(
    world_position_m: tuple[float, float, float],
    node_position_m: np.ndarray,
    orientation: NodeOrientation | None,
) -> tuple[float, float, float]:
    """Transform a site-frame position into the node-local mic frame."""
    rotation = orientation_rotation_matrix(orientation)
    offset = np.asarray(world_position_m, dtype=np.float64) - np.asarray(node_position_m, dtype=np.float64)
    local = rotation.T @ offset
    return (float(local[0]), float(local[1]), float(local[2]))


def select_nodes(
    candidates: list[NodeAudioCandidate],
    source_position_m: tuple[float, float, float],
    config: CrossNodeBeamConfig,
) -> list[tuple[NodeAudioCandidate, float]]:
    """Rank eligible nodes by a 1/r² SNR proxy; return the top max_nodes.

    Eligibility: audio covers the event interval, trusted position (static
    pseudo-positions excluded from geometric use), sync grade usable for
    multi-node work, and distance within max_range_m.
    """
    source = np.asarray(source_position_m, dtype=np.float64)
    ranked: list[tuple[NodeAudioCandidate, float]] = []
    for candidate in candidates:
        if not candidate.covers_event or not candidate.position_trusted:
            continue
        if candidate.sync_grade not in _MULTI_NODE_SYNC_GRADES:
            continue
        distance = float(np.linalg.norm(source - np.asarray(candidate.node_position_m, dtype=np.float64)))
        if distance > config.max_range_m:
            continue
        ranked.append((candidate, distance))
    ranked.sort(key=lambda item: item[1])
    return ranked[: max(1, config.max_nodes)] if ranked else []


def omni_mix(candidate: NodeAudioCandidate) -> np.ndarray:
    """Mean of the candidate node's mic channels — a raw omni reference window."""
    channels = np.asarray(candidate.channels, dtype=np.float64)
    if channels.ndim == 1:
        return channels
    if channels.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return channels.mean(axis=0)


def render_node_beam(
    candidate: NodeAudioCandidate,
    source_position_m: tuple[float, float, float],
    config: CrossNodeBeamConfig,
) -> NodeBeamRender:
    """Band-split DAS beam toward the source in the node's own local frame."""
    steer_local = world_to_node_local(
        source_position_m,
        candidate.node_position_m,
        candidate.orientation,
    )
    renderer = BandSplitDasRenderer(config=replace(config.render_config))
    mic_positions = np.asarray(candidate.mic_positions_local_m, dtype=np.float64)
    sensor_ids = [f"{candidate.node_id}:{i}" for i in range(mic_positions.shape[0])]
    positions = {sid: mic_positions[i] for i, sid in enumerate(sensor_ids)}
    windows = {sid: np.asarray(candidate.channels[i], dtype=np.float64) for i, sid in enumerate(sensor_ids)}
    samples = renderer.beamform(
        sensor_positions=positions,
        sensor_windows=windows,
        sample_rate_hz=candidate.sample_rate_hz,
        steer_position_m=steer_local,
        sound_speed_mps=config.sound_speed_mps,
    )
    return NodeBeamRender(
        node_id=candidate.node_id,
        samples=samples,
        alias_cutoff_hz=float(alias_cutoff_from_positions(mic_positions, config.sound_speed_mps)),
        distance_m=float(
            np.linalg.norm(
                np.asarray(source_position_m, dtype=np.float64)
                - np.asarray(candidate.node_position_m, dtype=np.float64)
            )
        ),
    )


def fuse_label_scores(
    per_node_scores: dict[str, dict[str, float]],
) -> FusedLabelResult | None:
    """Late fusion: per-label max over nodes, with contributing-node evidence.

    The winning label's best node is the canonical audio-evidence node.
    """
    if not per_node_scores:
        return None
    label_best: dict[str, tuple[float, str]] = {}
    label_nodes: dict[str, dict[str, float]] = {}
    for node_id, scores in per_node_scores.items():
        for label, confidence in scores.items():
            conf = float(confidence)
            label_nodes.setdefault(label, {})[node_id] = conf
            best = label_best.get(label)
            if best is None or conf > best[0]:
                label_best[label] = (conf, node_id)
    if not label_best:
        return None
    winning_label = max(label_best, key=lambda lbl: label_best[lbl][0])
    confidence, best_node_id = label_best[winning_label]
    per_node = label_nodes[winning_label]
    return FusedLabelResult(
        label=winning_label,
        confidence=confidence,
        best_node_id=best_node_id,
        per_node_confidence=dict(sorted(per_node.items())),
        contributing_node_ids=sorted(per_node),
    )


class CrossNodeBeamformer:
    """Select nodes, render per-node beams, classify each, and fuse labels."""

    def __init__(self, config: CrossNodeBeamConfig | None = None) -> None:
        self.config = config or CrossNodeBeamConfig()

    async def classify_across_nodes(
        self,
        *,
        candidates: list[NodeAudioCandidate],
        source_position_m: tuple[float, float, float],
        classify_fn: Callable[[np.ndarray, int], Awaitable[dict[str, float]]],
    ) -> dict[str, Any] | None:
        """Full pipeline; returns None when no node is eligible (caller falls
        back to the single-node beam / omni rungs of the degradation ladder)."""
        if not self.config.enabled:
            return None
        selected = select_nodes(candidates, source_position_m, self.config)
        if not selected:
            return None
        if self.config.classification_audio_source == "nearest_node_omni":
            # Distance ranking already applied by select_nodes; take the nearest.
            nearest_candidate = selected[0][0]
            mix = omni_mix(nearest_candidate)
            if mix.size == 0:
                return None
            scores = await classify_fn(mix, nearest_candidate.sample_rate_hz)
            fused = fuse_label_scores({nearest_candidate.node_id: scores})
            if fused is None:
                return None
            return {
                "label": fused.label,
                "confidence": fused.confidence,
                "fusion_method": "nearest_node_omni",
                "best_node_id": nearest_candidate.node_id,
                "contributing_node_ids": [nearest_candidate.node_id],
                "per_node_confidence": fused.per_node_confidence,
                "canonical_beam": mix,
                "canonical_alias_cutoff_hz": None,
                "node_count": 1,
            }
        renders: dict[str, NodeBeamRender] = {}
        per_node_scores: dict[str, dict[str, float]] = {}
        for candidate, _distance in selected:
            render = render_node_beam(candidate, source_position_m, self.config)
            if render.samples.size == 0:
                continue
            renders[candidate.node_id] = render
            per_node_scores[candidate.node_id] = await classify_fn(
                render.samples, candidate.sample_rate_hz
            )
        fused = fuse_label_scores(per_node_scores)
        if fused is None:
            return None
        best_render = renders[fused.best_node_id]
        return {
            "label": fused.label,
            "confidence": fused.confidence,
            "fusion_method": "max_late_fusion",
            "best_node_id": fused.best_node_id,
            "contributing_node_ids": fused.contributing_node_ids,
            "per_node_confidence": fused.per_node_confidence,
            "canonical_beam": best_render.samples,
            "canonical_alias_cutoff_hz": best_render.alias_cutoff_hz,
            "node_count": len(per_node_scores),
        }


def eligible_distance_rank_weight(distance_m: float) -> float:
    """1/r² SNR proxy used for ranking (reuses the amplitude-range model)."""
    return 1.0 / max(distance_m, 1.0) ** 2


__all__ = [
    "CrossNodeBeamConfig",
    "CrossNodeBeamformer",
    "FusedLabelResult",
    "NodeAudioCandidate",
    "NodeBeamRender",
    "fuse_label_scores",
    "omni_mix",
    "position_source_trusted",
    "propagation_delay_s",
    "render_node_beam",
    "select_nodes",
    "world_to_node_local",
]
