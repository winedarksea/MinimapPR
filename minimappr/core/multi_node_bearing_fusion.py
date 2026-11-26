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

Lateral std model: a cone's lateral uncertainty is an *angular* part, stated as
``lateral_std_m`` at its solve range ``range_prior_m`` and growing linearly with
range, plus a range-independent node-position part ``position_std_m`` (a rigid
placement error shifts the ray, it does not widen its angle). At ray parameter t the
lateral std is ``max(sqrt((lateral_std_m·t/range_prior_m)² + position_std_m²),
min_lateral_std_m)``. The intersection is solved once with each cone's std at its
solve range, then re-weighted once with the std at the first fused range, so the
weights and the agreement gate use the same model.

Inputs carry no label or solve ID, so "same call" is judged geometrically: the fused
point must lie in front of every array (``behind_array``) and within
``max_miss_sigma`` lateral standard deviations of every ray (``rays_disagree``).
Fused confidence is discounted by ``exp(-0.5 · mean z²)`` of the per-ray normalized
misses so marginal agreement does not inherit the full noisy-OR confidence.

Selection: a node may have several cones in the window (two birds). Per node, the
cone that agrees best with the other nodes' chosen cones wins (coordinate descent on
the fused solve's rank, bounded by the node count). If the chosen set still fails
and more than two nodes remain, the node whose removal gives the best-ranked solve
is dropped and the rest re-solved — bounded leave-one-out outlier rejection, so one
node hearing a different bird cannot veto an otherwise consistent fix.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass

import numpy as np

from minimappr.core.localization_uncertainty import _stabilize_covariance

EPSILON = 1e-9
_SPEED_OF_SOUND_FALLBACK_MPS = 343.0
# Lateral variance floor (m²) inside the information weight.
LATERAL_VARIANCE_FLOOR_M2 = 1e-4
_MINIMUM_OBSERVATION_WEIGHT_CONFIDENCE = 0.05
_MAXIMUM_OBSERVATION_CONFIDENCE = 0.99
_MAXIMUM_COMBINED_CONFIDENCE = 0.95
_STABILIZED_MINIMUM_STD_M = 1.0
_MINIMUM_DISTINCT_NODES = 2
# Rank tiers for comparing candidate selections (lower is better).
_RANK_TIER_FUSED = 0
_RANK_TIER_RAYS_DISAGREE = 1
_RANK_TIER_OTHER_REJECTION = 2


@dataclass(slots=True)
class BearingObservation:
    """One node's single-node cone, reduced to a bearing ray for triangulation."""

    node_id: str
    origin_m: np.ndarray  # contributing-sensor centroid (ray apex)
    direction: np.ndarray  # unit bearing from origin toward the source
    lateral_std_m: float  # angular part of the lateral std, stated at range_prior_m
    confidence: float
    event_time_ns: int
    range_prior_m: float  # solve range at which lateral_std_m is stated
    expiry_ns: int
    position_std_m: float = 0.0  # range-independent lateral std (node-position error)


@dataclass(slots=True)
class BearingFusionResult:
    position_m: np.ndarray
    covariance_m2: np.ndarray
    contributor_node_ids: list[str]
    range_m: float
    confidence: float
    reason: str = "fused"


@dataclass(frozen=True, slots=True)
class _FusionGates:
    min_separation_deg: float
    max_condition: float
    max_range_m: float
    sound_speed_mps: float
    window_seconds: float
    max_miss_sigma: float
    min_lateral_std_m: float


@dataclass(slots=True)
class _SelectionSolve:
    result: BearingFusionResult | None
    reason: str
    max_normalized_miss: float = math.inf

    def rank(self) -> tuple[int, float]:
        if self.result is not None:
            return (_RANK_TIER_FUSED, self.max_normalized_miss)
        if self.reason == "rays_disagree":
            return (_RANK_TIER_RAYS_DISAGREE, self.max_normalized_miss)
        return (_RANK_TIER_OTHER_REJECTION, math.inf)


def _min_pairwise_line_separation_deg(directions: list[np.ndarray]) -> float:
    """Smallest angle between any two bearing *lines*, from the signed ray angle.

    Parallel (0°) and anti-parallel (180°) rays both leave range along the shared
    axis unobservable, so the separation from either end is what matters.
    Anti-parallel pairs that face away from each other are additionally caught by
    the ``behind_array`` test.
    """

    smallest = 180.0
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            signed_dot = float(np.clip(np.dot(directions[i], directions[j]), -1.0, 1.0))
            ray_angle_deg = math.degrees(math.acos(signed_dot))
            smallest = min(smallest, ray_angle_deg, 180.0 - ray_angle_deg)
    return smallest


def _lateral_std_at_ray_parameter_m(
    observation: BearingObservation, ray_parameter_m: float, min_lateral_std_m: float
) -> float | None:
    """Angular part scaled to ``ray_parameter_m`` in quadrature with the position part.

    ``None`` (degenerate) when the cone has no usable solve range or the std is not
    strictly positive and finite (e.g. zero-width cone with a zero floor).
    """

    range_prior_m = float(observation.range_prior_m)
    if not (math.isfinite(range_prior_m) and range_prior_m > EPSILON):
        return None
    angular_std_m = float(observation.lateral_std_m) * ray_parameter_m / range_prior_m
    position_std_m = float(observation.position_std_m)
    combined_std_m = math.sqrt(angular_std_m * angular_std_m + position_std_m * position_std_m)
    lateral_std_m = max(combined_std_m, min_lateral_std_m)
    if not (math.isfinite(lateral_std_m) and lateral_std_m > EPSILON):
        return None
    return lateral_std_m


def _weighted_intersection(
    selected: list[BearingObservation],
    directions: list[np.ndarray],
    lateral_stds_m: list[float],
    max_condition: float,
) -> tuple[np.ndarray, np.ndarray] | str:
    """Closed-form intersection with wᵢ = confidence / σ²_lat (tighter, surer cones weigh more)."""

    identity = np.eye(3, dtype=np.float64)
    information = np.zeros((3, 3), dtype=np.float64)
    target = np.zeros(3, dtype=np.float64)
    for obs, direction, lateral_std_m in zip(selected, directions, lateral_stds_m):
        lateral_var = max(lateral_std_m * lateral_std_m, LATERAL_VARIANCE_FLOOR_M2)
        weight = max(float(obs.confidence), _MINIMUM_OBSERVATION_WEIGHT_CONFIDENCE) / lateral_var
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
    return position_m, covariance_m2


def _ray_parameters_m(
    selected: list[BearingObservation], directions: list[np.ndarray], position_m: np.ndarray
) -> list[float] | None:
    """Distance along each ray to the foot of ``position_m``; ``None`` if any is not in front."""

    ray_parameters_m: list[float] = []
    for obs, direction in zip(selected, directions):
        ray_parameter_m = float(np.dot(position_m - np.asarray(obs.origin_m, dtype=np.float64), direction))
        if not ray_parameter_m > 0.0:
            return None
        ray_parameters_m.append(ray_parameter_m)
    return ray_parameters_m


def _lateral_stds_m(
    selected: list[BearingObservation], ray_parameters_m: list[float], min_lateral_std_m: float
) -> list[float] | None:
    stds: list[float] = []
    for obs, ray_parameter_m in zip(selected, ray_parameters_m):
        lateral_std_m = _lateral_std_at_ray_parameter_m(obs, ray_parameter_m, min_lateral_std_m)
        if lateral_std_m is None:
            return None
        stds.append(lateral_std_m)
    return stds


def _solve_selection(selected: list[BearingObservation], gates: _FusionGates) -> _SelectionSolve:
    """Fuse exactly one cone per node, or say why not."""

    directions = [np.asarray(obs.direction, dtype=np.float64) for obs in selected]
    if _min_pairwise_line_separation_deg(directions) < gates.min_separation_deg:
        return _SelectionSolve(None, "near_parallel")  # range unobservable

    # Time sanity: two nodes observing the same event can differ by at most the fusion
    # window plus the differential propagation delay across their baselines.
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            a, b = selected[i], selected[j]
            baseline_m = float(np.linalg.norm(a.origin_m - b.origin_m))
            allowed_s = gates.window_seconds + baseline_m / gates.sound_speed_mps
            if abs(a.event_time_ns - b.event_time_ns) / 1e9 > allowed_s:
                return _SelectionSolve(None, "stale")  # mismatched events

    prior_stds_m = _lateral_stds_m(selected, [float(obs.range_prior_m) for obs in selected], gates.min_lateral_std_m)
    if prior_stds_m is None:
        return _SelectionSolve(None, "degenerate")
    first_pass = _weighted_intersection(selected, directions, prior_stds_m, gates.max_condition)
    if isinstance(first_pass, str):
        return _SelectionSolve(None, first_pass)
    first_ray_parameters_m = _ray_parameters_m(selected, directions, first_pass[0])
    if first_ray_parameters_m is None:
        return _SelectionSolve(None, "behind_array")
    reweighted_stds_m = _lateral_stds_m(selected, first_ray_parameters_m, gates.min_lateral_std_m)
    if reweighted_stds_m is None:
        return _SelectionSolve(None, "degenerate")
    second_pass = _weighted_intersection(selected, directions, reweighted_stds_m, gates.max_condition)
    if isinstance(second_pass, str):
        return _SelectionSolve(None, second_pass)
    position_m, covariance_m2 = second_pass
    ray_parameters_m = _ray_parameters_m(selected, directions, position_m)
    if ray_parameters_m is None:
        return _SelectionSolve(None, "behind_array")
    gate_stds_m = _lateral_stds_m(selected, ray_parameters_m, gates.min_lateral_std_m)
    if gate_stds_m is None:
        return _SelectionSolve(None, "degenerate")

    normalized_misses: list[float] = []
    for obs, direction, ray_parameter_m, lateral_std_m in zip(selected, directions, ray_parameters_m, gate_stds_m):
        offset = position_m - np.asarray(obs.origin_m, dtype=np.float64)
        miss_distance_m = float(np.linalg.norm(offset - ray_parameter_m * direction))
        normalized_misses.append(miss_distance_m / lateral_std_m)
    max_normalized_miss = max(normalized_misses)
    if max_normalized_miss > gates.max_miss_sigma:
        # Lines cross but not where every cone points.
        return _SelectionSolve(None, "rays_disagree", max_normalized_miss)

    stabilized = _stabilize_covariance(covariance_m2, minimum_std_m=_STABILIZED_MINIMUM_STD_M)
    if stabilized is None:
        return _SelectionSolve(None, "degenerate")
    origin_centroid = np.mean(
        np.vstack([np.asarray(obs.origin_m, dtype=np.float64) for obs in selected]), axis=0
    )
    range_m = float(np.linalg.norm(position_m - origin_centroid))
    if not math.isfinite(range_m) or range_m > gates.max_range_m:
        return _SelectionSolve(None, "out_of_range")

    noisy_or_confidence = 1.0 - math.prod(
        1.0 - min(o.confidence, _MAXIMUM_OBSERVATION_CONFIDENCE) for o in selected
    )
    mean_squared_normalized_miss = sum(z * z for z in normalized_misses) / len(normalized_misses)
    agreement_factor = math.exp(-0.5 * mean_squared_normalized_miss)
    result = BearingFusionResult(
        position_m=position_m,
        covariance_m2=np.asarray(stabilized, dtype=np.float64),
        contributor_node_ids=[obs.node_id for obs in selected],
        range_m=range_m,
        confidence=float(min(_MAXIMUM_COMBINED_CONFIDENCE, noisy_or_confidence * agreement_factor)),
    )
    return _SelectionSolve(result, "fused", max_normalized_miss)


def _refine_cone_choice(
    cones_by_node: dict[str, list[BearingObservation]],
    selection: dict[str, BearingObservation],
    gates: _FusionGates,
) -> tuple[dict[str, BearingObservation], _SelectionSolve]:
    """Per node, swap in the cone that best agrees with the others' current choices.

    Only strict rank improvements are accepted, so it terminates; passes are capped
    at the node count.
    """

    solve = _solve_selection(list(selection.values()), gates)
    for _ in range(len(selection)):
        changed = False
        for node_id in list(selection):
            for cone in cones_by_node[node_id]:
                if cone is selection[node_id]:
                    continue
                trial = dict(selection)
                trial[node_id] = cone
                trial_solve = _solve_selection(list(trial.values()), gates)
                if trial_solve.rank() < solve.rank():
                    selection, solve, changed = trial, trial_solve, True
        if not changed:
            break
    return selection, solve


def fuse_bearings(
    observations: list[BearingObservation],
    *,
    min_separation_deg: float = 5.0,
    max_condition: float = 1e4,
    max_range_m: float = 1200.0,
    sound_speed_mps: float = _SPEED_OF_SOUND_FALLBACK_MPS,
    window_seconds: float = 1.5,
    max_miss_sigma: float = 3.0,
    min_lateral_std_m: float = 1.0,
) -> BearingFusionResult | str:
    """Triangulate a source position from bearing observations of >=2 distinct nodes.

    Returns a :class:`BearingFusionResult` on success, or a short reason string
    (``"insufficient_nodes"``, ``"near_parallel"``, ``"stale"``, ``"degenerate"``,
    ``"behind_array"``, ``"rays_disagree"``, ``"out_of_range"``) so the caller can
    attribute the right metric; on failure the reason describes the full node set
    after cone selection. ``contributor_node_ids`` may be a subset of the input
    nodes (outlier rejection), so a caller upgrading one node's cone must check its
    node is among them. ``min_lateral_std_m`` is a numerical floor on the combined
    lateral std (the node-position part is ``position_std_m``).
    """

    if len(observations) < _MINIMUM_DISTINCT_NODES:
        return "insufficient_nodes"
    cones_by_node: dict[str, list[BearingObservation]] = {}
    for obs in observations:
        cones_by_node.setdefault(obs.node_id, []).append(obs)
    # Repeat cones from one node cannot resolve range.
    if len(cones_by_node) < _MINIMUM_DISTINCT_NODES:
        return "insufficient_nodes"
    gates = _FusionGates(
        min_separation_deg=min_separation_deg,
        max_condition=max_condition,
        max_range_m=max_range_m,
        sound_speed_mps=sound_speed_mps if sound_speed_mps > EPSILON else _SPEED_OF_SOUND_FALLBACK_MPS,
        window_seconds=window_seconds,
        max_miss_sigma=max_miss_sigma,
        min_lateral_std_m=min_lateral_std_m,
    )
    selection: dict[str, BearingObservation] = {}
    for node_id, cones in cones_by_node.items():
        most_confident = cones[0]
        for cone in cones[1:]:
            if cone.confidence > most_confident.confidence:
                most_confident = cone
        selection[node_id] = most_confident

    selection, solve = _refine_cone_choice(cones_by_node, selection, gates)
    full_set_reason = solve.reason
    while solve.result is None and len(selection) > _MINIMUM_DISTINCT_NODES:
        best_drop_node_id: str | None = None
        best_drop_solve: _SelectionSolve | None = None
        for node_id in selection:
            remaining = [cone for other_id, cone in selection.items() if other_id != node_id]
            candidate_solve = _solve_selection(remaining, gates)
            if best_drop_solve is None or candidate_solve.rank() < best_drop_solve.rank():
                best_drop_node_id, best_drop_solve = node_id, candidate_solve
        selection = {node_id: cone for node_id, cone in selection.items() if node_id != best_drop_node_id}
        selection, solve = _refine_cone_choice(cones_by_node, selection, gates)
    if solve.result is None:
        return full_set_reason
    return solve.result


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
            # One cone per (node, event): an event solve and the render covering
            # it register the same cone, which must not corroborate itself twice.
            self._observations = [
                obs
                for obs in self._observations
                if not (obs.node_id == observation.node_id and obs.event_time_ns == observation.event_time_ns)
            ]
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
