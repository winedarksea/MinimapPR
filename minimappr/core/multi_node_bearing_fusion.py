"""Windowed multi-node bearing triangulation (Phase 4, tier b).

A single tetrahedral node is a *bearing* instrument: it resolves azimuth/elevation
well but not range. When two or more nodes independently detect the same source
within a short time window, intersecting their bearing rays recovers range —
without any cross-node correlation (that is tier c, Phase 5).

This module is server-side only and feature-flagged off by default
(``multi_node_bearing_fusion_enabled``). It is purely *opportunistic*: each
single-node cone is registered in a bounded, TTL-pruned store as it is produced; when
a later node's cone arrives and corroborates, the branch is upgraded in place. The
pipeline stays append-only — earlier detections are not retro-edited (Phase 3 track
association merges them), and the cadence is unchanged.

Fused covariance uses the closed-form bearing-intersection information matrix
``Σ wᵢ (I − dᵢdᵢᵀ) / σ²_lat,i``; its inverse is the position covariance. Degeneracy
(near-parallel bearings, e.g. a 1 km source seen by nodes 5 m apart) is rejected via
the information-matrix condition number and a minimum pairwise angular separation, so
the fusion never manufactures false range precision.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field

import numpy as np

from minimappr.core.localization_uncertainty import _stabilize_covariance

EPSILON = 1e-9
_SPEED_OF_SOUND_FALLBACK_MPS = 343.0


@dataclass(slots=True)
class BearingObservation:
    """One node's single-node cone, reduced to a bearing ray for triangulation."""

    node_id: str
    origin_m: np.ndarray  # contributing-sensor centroid (ray apex)
    direction: np.ndarray  # unit bearing from origin toward the source
    lateral_std_m: float  # angular (perpendicular) uncertainty of the cone
    confidence: float
    event_time_ns: int
    range_prior_m: float
    expiry_ns: int


@dataclass(slots=True)
class BearingFusionResult:
    position_m: np.ndarray
    covariance_m2: np.ndarray
    contributor_node_ids: list[str]
    range_m: float
    confidence: float
    reason: str = "fused"


def _unit(vector: np.ndarray) -> np.ndarray | None:
    v = np.asarray(vector, dtype=np.float64).reshape(-1)
    if v.size != 3 or not np.all(np.isfinite(v)):
        return None
    norm = float(np.linalg.norm(v))
    if norm < EPSILON:
        return None
    return v / norm


def _min_pairwise_separation_deg(directions: list[np.ndarray]) -> float:
    smallest = 180.0
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            dot = float(np.clip(np.dot(directions[i], directions[j]), -1.0, 1.0))
            angle_deg = math.degrees(math.acos(abs(dot)))
            smallest = min(smallest, angle_deg)
    return smallest


def fuse_bearings(
    observations: list[BearingObservation],
    *,
    min_separation_deg: float = 5.0,
    max_condition: float = 1e4,
    max_range_m: float = 1200.0,
    sound_speed_mps: float = _SPEED_OF_SOUND_FALLBACK_MPS,
    window_seconds: float = 1.5,
) -> BearingFusionResult | str:
    """Triangulate a source position from >=2 bearing observations.

    Returns a :class:`BearingFusionResult` on success, or a short reason string
    (``"insufficient_nodes"``, ``"near_parallel"``, ``"stale"``, ``"degenerate"``,
    ``"out_of_range"``) so the caller can attribute the right metric. Requires
    observations from at least two distinct nodes.
    """

    if len(observations) < 2:
        return "insufficient_nodes"
    # Distinct nodes only — repeat cones from one node cannot resolve range.
    by_node: dict[str, BearingObservation] = {}
    for obs in observations:
        prior = by_node.get(obs.node_id)
        if prior is None or obs.confidence > prior.confidence:
            by_node[obs.node_id] = obs
    if len(by_node) < 2:
        return "insufficient_nodes"
    selected = list(by_node.values())

    directions = [np.asarray(obs.direction, dtype=np.float64) for obs in selected]
    if _min_pairwise_separation_deg(directions) < min_separation_deg:
        return "near_parallel"  # range unobservable, do not fabricate precision

    # Time sanity: two nodes observing the same event can differ by at most the fusion
    # window plus the differential propagation delay across their baselines.
    speed = sound_speed_mps if sound_speed_mps > EPSILON else _SPEED_OF_SOUND_FALLBACK_MPS
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            a, b = selected[i], selected[j]
            baseline_m = float(np.linalg.norm(a.origin_m - b.origin_m))
            allowed_s = window_seconds + baseline_m / speed
            if abs(a.event_time_ns - b.event_time_ns) / 1e9 > allowed_s:
                return "stale"  # mismatched events

    # Closed-form bearing intersection: information = Σ wᵢ (I − dᵢdᵢᵀ) with
    # wᵢ = confidence / σ²_lat (tighter, more-confident cones weigh more).
    identity = np.eye(3, dtype=np.float64)
    information = np.zeros((3, 3), dtype=np.float64)
    target = np.zeros(3, dtype=np.float64)
    for obs, direction in zip(selected, directions):
        lateral_var = max(float(obs.lateral_std_m) ** 2, 1e-4)
        weight = max(float(obs.confidence), 0.05) / lateral_var
        projection = (identity - np.outer(direction, direction)) * weight
        information += projection
        target += projection @ np.asarray(obs.origin_m, dtype=np.float64)

    eigenvalues = np.linalg.eigvalsh(information)
    if not np.all(np.isfinite(eigenvalues)):
        return "degenerate"
    min_eig = float(np.min(eigenvalues))
    max_eig = float(np.max(eigenvalues))
    if min_eig <= EPSILON or (max_eig / max(min_eig, EPSILON)) > max_condition:
        return "degenerate"  # ill-conditioned intersection (parallax too small)

    try:
        position_m = np.linalg.solve(information, target)
        covariance_m2 = np.linalg.inv(information)
    except np.linalg.LinAlgError:
        return "degenerate"
    if not np.all(np.isfinite(position_m)) or not np.all(np.isfinite(covariance_m2)):
        return "degenerate"

    stabilized = _stabilize_covariance(covariance_m2, minimum_std_m=1.0)
    if stabilized is None:
        return "degenerate"

    origin_centroid = np.mean(
        np.vstack([np.asarray(obs.origin_m, dtype=np.float64) for obs in selected]), axis=0
    )
    range_m = float(np.linalg.norm(position_m - origin_centroid))
    if not math.isfinite(range_m) or range_m > max_range_m:
        return "out_of_range"

    combined_confidence = float(min(0.95, 1.0 - math.prod(1.0 - min(o.confidence, 0.99) for o in selected)))
    return BearingFusionResult(
        position_m=position_m,
        covariance_m2=np.asarray(stabilized, dtype=np.float64),
        contributor_node_ids=[obs.node_id for obs in selected],
        range_m=range_m,
        confidence=combined_confidence,
    )


class BearingFusionStore:
    """Bounded, TTL-pruned, asyncio-locked store of recent single-node cones."""

    def __init__(self, *, max_entries: int = 256, ttl_seconds: float = 4.0) -> None:
        self._max_entries = max(1, int(max_entries))
        self._ttl_ns = int(max(ttl_seconds, 0.1) * 1e9)
        self._observations: list[BearingObservation] = []
        self._lock = asyncio.Lock()

    async def register(self, observation: BearingObservation, *, now_ns: int) -> None:
        async with self._lock:
            self._prune_locked(now_ns)
            self._observations.append(observation)
            if len(self._observations) > self._max_entries:
                # Drop oldest by expiry.
                self._observations.sort(key=lambda obs: obs.expiry_ns)
                self._observations = self._observations[-self._max_entries :]

    async def corroborators(
        self,
        observation: BearingObservation,
        *,
        now_ns: int,
        window_seconds: float,
    ) -> list[BearingObservation]:
        """Return other-node observations within the event-time window (self excluded)."""

        window_ns = int(max(window_seconds, 0.0) * 1e9)
        async with self._lock:
            self._prune_locked(now_ns)
            matches: list[BearingObservation] = []
            for obs in self._observations:
                if obs.node_id == observation.node_id:
                    continue
                if abs(obs.event_time_ns - observation.event_time_ns) <= window_ns:
                    matches.append(obs)
            return matches

    async def size(self) -> int:
        async with self._lock:
            return len(self._observations)

    def _prune_locked(self, now_ns: int) -> None:
        self._observations = [obs for obs in self._observations if obs.expiry_ns > now_ns]
