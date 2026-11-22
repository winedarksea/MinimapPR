"""Scripted wildlife detection sequences for the S3/S6 parity scenarios.

Each builder returns pipeline steps (see ``tests/wildlife_pipeline_reference``)
for one seed: per-bout fixes jittered around perches on a 20 m three-node
triangle, with species evidence, unpositioned bouts, far-field bearing
evidence and maintenance ticks. The scenes mirror the server S gate's
(countersinging pair, rotating flock, hop with a ghost revert, two species
7 m apart, omni-then-localized, far bird on a bearing, expiry).
"""

from __future__ import annotations

import math

import numpy as np

SECOND_NS = 1_000_000_000
T0_NS = 1_700_000_000_000_000_000
NODES = {"tetra-a": [0.0, 0.0, 1.5], "tetra-b": [20.0, 0.0, 1.5], "tetra-c": [10.0, 17.3, 1.5]}
ROBIN, TANAGER, GROSBEAK, CHICKADEE, CARDINAL = (
    "american robin", "scarlet tanager", "rose-breasted grosbeak", "black-capped chickadee", "northern cardinal",
)
FIX_SIGMA_FLOOR_M = 0.4
FIX_SIGMA_PER_METRE_FROM_CENTRE = 0.03
LAYOUT_CENTRE_M = (10.0, 5.8)


class _Script:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.steps: list[dict] = []

    def _nearest_node(self, position) -> str:
        return min(NODES, key=lambda node: math.dist(NODES[node][:2], position[:2]))

    def bout(self, t_s: float, perch, label: str, confidence: float, *, mode: str = "localized",
             extra_evidence: list | None = None, sigma_scale: float = 1.0, node: str | None = None) -> None:
        perch = np.asarray(perch, dtype=float)
        offset = math.dist(perch[:2], LAYOUT_CENTRE_M)
        sigma = (FIX_SIGMA_FLOOR_M + FIX_SIGMA_PER_METRE_FROM_CENTRE * offset) * sigma_scale
        node = node or self._nearest_node(perch)
        position = (perch + self.rng.normal(scale=sigma, size=3)).tolist() if mode != "node_only" else None
        covariance = np.diag([sigma**2, sigma**2, (2 * sigma) ** 2]).tolist() if mode != "node_only" else None
        evidence = [[label, confidence], *(extra_evidence or [])] if label != "unknown" else list(extra_evidence or [])
        self.steps.append({
            "type": "detection", "detection_id": f"det-{len(self.steps) + 1}",
            "event_time_ns": T0_NS + int(round(t_s * SECOND_NS)), "label": label, "confidence": confidence,
            "species_evidence": evidence, "mode": mode, "node_id": node, "node_position_m": NODES[node],
            "position_m": position, "covariance_m2": covariance, "range_projection_mode": None,
        })

    def far_bout(self, t_s: float, true_position, label: str, confidence: float, *, solved_range_m: float,
                 bearing_projected: bool, mode: str = "localized") -> None:
        """A far bird's fix on the right bearing at the wrong range, covariance along the ray."""
        node = self._nearest_node(true_position)
        origin = np.asarray(NODES[node])
        direction = np.asarray(true_position, dtype=float) - origin
        direction[2] = 0.0
        direction /= np.linalg.norm(direction)
        position = (origin + direction * solved_range_m).tolist()
        radial_sigma, tangential_sigma = solved_range_m * 0.5, max(1.0, solved_range_m * 0.02)
        tangential = np.array([-direction[1], direction[0], 0.0])
        covariance = (radial_sigma**2 * np.outer(direction, direction) + tangential_sigma**2 * np.outer(tangential, tangential)
                      + np.diag([0.0, 0.0, 4.0])).tolist()
        self.steps.append({
            "type": "detection", "detection_id": f"det-{len(self.steps) + 1}",
            "event_time_ns": T0_NS + int(round(t_s * SECOND_NS)), "label": label, "confidence": confidence,
            "species_evidence": [[label, confidence]], "mode": mode, "node_id": node, "node_position_m": NODES[node],
            "position_m": position, "covariance_m2": covariance,
            "range_projection_mode": "range_bearing_projected" if bearing_projected else None,
        })

    def maintain(self, t_s: float) -> None:
        self.steps.append({"type": "maintain", "now_ns": T0_NS + int(round(t_s * SECOND_NS))})


def countersinging_pair(seed: int) -> list[dict]:
    script = _Script(seed)
    perches = ([4.0, 4.0, 5.0], [14.0, 11.7, 5.0])  # 12.6 m apart
    for bout in range(8):
        script.bout(1.0 + 1.5 * bout, perches[bout % 2], CARDINAL, float(script.rng.uniform(0.6, 0.9)))
        script.maintain(1.5 + 1.5 * bout)
    return script.steps


def rotating_flock(seed: int) -> list[dict]:
    script = _Script(seed)
    perches = ([3.0, 2.0, 4.0], [13.5, 2.0, 4.0], [8.0, 11.0, 4.0])  # 10-11 m apart
    for bout in range(12):
        script.bout(1.0 + 1.2 * bout, perches[bout % 3], CHICKADEE, float(script.rng.uniform(0.5, 0.9)))
    script.maintain(20.0)
    return script.steps


def hop_then_ghost_revert(seed: int) -> list[dict]:
    script = _Script(seed)
    old_perch, new_perch = [2.0, 2.0, 4.0], [17.0, 3.0, 4.0]  # 15 m hop
    for bout in range(3):
        script.bout(1.0 + 2.0 * bout, old_perch, ROBIN, 0.8)
    script.bout(10.0, new_perch, ROBIN, 0.8)  # silent 5 s: one bird hopping
    script.bout(11.0, old_perch, ROBIN, 0.8)  # back on the ghost: it was two birds
    script.bout(14.0, new_perch, ROBIN, 0.75)
    script.bout(16.0, old_perch, ROBIN, 0.75)
    script.maintain(18.0)
    return script.steps


def two_species_close(seed: int) -> list[dict]:
    script = _Script(seed)
    tanager, grosbeak = [14.0, 7.0, 3.0], [10.0, 13.0, 3.0]  # 7.2 m apart
    for bout in range(3):
        script.bout(1.0 + bout, tanager, TANAGER, 0.85)
    script.bout(4.5, grosbeak, GROSBEAK, 0.3)  # weak claim off the tanager's perch
    script.bout(5.5, grosbeak, GROSBEAK, 0.9)
    script.bout(6.5, tanager, GROSBEAK, 0.3, extra_evidence=[[TANAGER, 0.2]])  # misclassified bout on the perch
    script.bout(8.0, grosbeak, "unknown", 0.0)
    script.bout(9.0, tanager, TANAGER, 0.7)
    script.maintain(10.0)
    return script.steps


def omni_then_localized(seed: int) -> list[dict]:
    script = _Script(seed)
    wren = [11.0, 15.5, 2.0]  # 2.3 m from tetra-c
    script.bout(1.0, wren, "house wren", 0.6, mode="node_only", node="tetra-c")
    script.bout(3.0, wren, "house wren", 0.55, mode="node_only", node="tetra-c")
    script.bout(5.0, wren, "house wren", 0.9)
    script.bout(7.0, wren, "house wren", 0.85, mode="node_only", node="tetra-c")
    script.bout(9.0, wren, "house wren", 0.9)
    script.maintain(10.0)
    return script.steps


def far_bird_on_bearing(seed: int) -> list[dict]:
    script = _Script(seed)
    far = [60.0, 55.0, 6.0]
    script.bout(1.0, far, ROBIN, 0.8, sigma_scale=0.6)
    script.far_bout(4.0, far, ROBIN, 0.8, solved_range_m=391.0, bearing_projected=False)
    script.far_bout(7.0, far, ROBIN, 0.7, solved_range_m=160.0, bearing_projected=True)
    script.far_bout(9.0, [-70.0, -40.0, 5.0], TANAGER, 0.8, solved_range_m=90.0, bearing_projected=True, mode="bearing_only")
    script.far_bout(12.0, [-70.0, -40.0, 5.0], TANAGER, 0.8, solved_range_m=120.0, bearing_projected=True, mode="bearing_only")
    script.maintain(13.0)
    return script.steps


def expiry_and_step_back(seed: int) -> list[dict]:
    script = _Script(seed)
    perch = [6.0, 6.0, 4.0]
    script.bout(1.0, perch, ROBIN, 0.8)
    script.bout(3.0, perch, ROBIN, 0.8)
    for t_s in (100.0, 200.0, 370.0, 500.0, 760.0):
        script.maintain(t_s)
    script.bout(770.0, perch, ROBIN, 0.8, mode="node_only")
    script.maintain(1200.0)
    return script.steps


def random_mix(seed: int) -> list[dict]:
    script = _Script(1000 + seed)
    rng = script.rng
    birds = [(rng.uniform(-10, 30, size=3) * [1, 1, 0.2] + [0, 0, 2]).tolist() for _ in range(4)]
    labels = [ROBIN, ROBIN, TANAGER, CHICKADEE]
    t_s = 0.0
    for _ in range(18):
        t_s += float(rng.uniform(0.3, 4.0))
        index = int(rng.integers(0, len(birds)))
        roll = rng.uniform()
        mode = "node_only" if roll < 0.2 else "localized"
        if rng.uniform() < 0.1:
            birds[index] = (np.asarray(birds[index]) + rng.uniform(-15, 15, size=3) * [1, 1, 0]).tolist()
        script.bout(t_s, birds[index], labels[index], float(rng.uniform(0.2, 0.95)), mode=mode)
        if rng.uniform() < 0.3:
            script.maintain(t_s + 0.5)
    return script.steps


SCENARIO_BUILDERS = {
    "countersinging_pair": countersinging_pair,
    "rotating_flock": rotating_flock,
    "hop_then_ghost_revert": hop_then_ghost_revert,
    "two_species_close": two_species_close,
    "omni_then_localized": omni_then_localized,
    "far_bird_on_bearing": far_bird_on_bearing,
    "expiry_and_step_back": expiry_and_step_back,
    "random_mix": random_mix,
}
