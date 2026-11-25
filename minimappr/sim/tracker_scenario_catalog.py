"""Catalogue of tracker scenarios (see ``tracker_detection_scenarios``).

Each mirrors a row of the field-mode test strategy that the tracker owns:

- ``intermittent_perched_robin``: one bird, 8 bouts 6-14 s apart, 1.2 m fix
  jitter, weak and occasionally confused BirdNET scores ("tracks survive 10 s
  gaps without drifting", "three weak bouts confirm the species").
- ``robin_hops_15m``: 4 bouts on one perch, a 15 m hop, 4 bouts on the next
  ("a 15 m hop with a matching species stays one track").
- ``two_species_alternating``: robin and tanager 12 m apart, alternating, with
  robin bouts sometimes scored as tanager ("no label swaps on same-band
  two-species bundles").
- ``two_species_overlapping``: the same pair singing within 0.3 s, so the
  label-swap metric has eligible detections.
- ``weak_fix_weak_id``: plan example — a bird localized with a weak ID on one
  bout and heard omni-only with a better ID on the next.
- ``flock_same_species``: three robins on perches 20-25 m apart, alternating
  bouts ("a separate perched track for each active bird").
"""

from __future__ import annotations

import numpy as np

from minimappr.core.tracking_mode_config import TrackingMode, tracking_mode_by_category_with
from minimappr.sim.tracker_detection_scenarios import SyntheticBout, TrackerScenario

ROBIN = "american robin"
TANAGER = "scarlet tanager"
WREN = "house wren"
ROBIN_PERCH_M = (6.0, 12.0, 2.0)
TANAGER_PERCH_M = (18.0, 9.0, 3.0)
CATALOGUE_SEED = 7


def _gaps(rng: np.random.Generator, count: int, low_s: float, high_s: float) -> list[float]:
    return list(np.cumsum(rng.uniform(low_s, high_s, size=count)) - low_s)


def intermittent_perched_robin() -> TrackerScenario:
    rng = np.random.default_rng(CATALOGUE_SEED)
    times = _gaps(rng, 8, 6.0, 14.0)
    scores = [
        {ROBIN: 0.55}, {ROBIN: 0.32, WREN: 0.1}, {ROBIN: 0.45}, {WREN: 0.42, ROBIN: 0.3},
        {ROBIN: 0.38}, {ROBIN: 0.62}, {ROBIN: 0.28, WREN: 0.2}, {ROBIN: 0.5},
    ]
    bouts = [SyntheticBout(t, "robin-1", ROBIN_PERCH_M, s) for t, s in zip(times, scores)]
    return TrackerScenario("intermittent_perched_robin", {"robin-1": ROBIN}, bouts, fix_jitter_m=1.2)


def robin_hops_15m() -> TrackerScenario:
    second_perch = (19.0, 19.0, 2.0)
    bouts = []
    for index in range(8):
        perch = ROBIN_PERCH_M if index < 4 else second_perch
        bouts.append(SyntheticBout(8.0 * index, "robin-1", perch, {ROBIN: 0.5 + 0.05 * (index % 3)}))
    return TrackerScenario("robin_hops_15m", {"robin-1": ROBIN}, bouts)


def two_species_alternating() -> TrackerScenario:
    bouts = []
    for index in range(6):
        robin_scores = {TANAGER: 0.45, ROBIN: 0.35} if index in (2, 4) else {ROBIN: 0.6}
        bouts.append(SyntheticBout(8.0 * index, "robin-1", ROBIN_PERCH_M, robin_scores))
        bouts.append(SyntheticBout(8.0 * index + 4.0, "tanager-1", TANAGER_PERCH_M, {TANAGER: 0.6},
                                   reporting_node_id="node-b"))
    return TrackerScenario("two_species_alternating", {"robin-1": ROBIN, "tanager-1": TANAGER}, bouts)


def two_species_overlapping() -> TrackerScenario:
    """Same pair singing within 0.3 s of each other, so label swaps are scorable."""
    bouts = []
    for index in range(6):
        robin_scores = {TANAGER: 0.45, ROBIN: 0.35} if index in (1, 3, 4) else {ROBIN: 0.6}
        bouts.append(SyntheticBout(8.0 * index, "robin-1", ROBIN_PERCH_M, robin_scores))
        bouts.append(SyntheticBout(8.0 * index + 0.3, "tanager-1", TANAGER_PERCH_M, {TANAGER: 0.6},
                                   reporting_node_id="node-b"))
    return TrackerScenario("two_species_overlapping", {"robin-1": ROBIN, "tanager-1": TANAGER}, bouts)


def weak_fix_weak_id() -> TrackerScenario:
    bouts = [
        SyntheticBout(0.0, "robin-1", ROBIN_PERCH_M, {ROBIN: 0.3}),
        SyntheticBout(7.0, "robin-1", ROBIN_PERCH_M, {ROBIN: 0.55}, mode="omni"),
        SyntheticBout(15.0, "robin-1", ROBIN_PERCH_M, {ROBIN: 0.35}),
        SyntheticBout(22.0, "robin-1", ROBIN_PERCH_M, {ROBIN: 0.5}, mode="omni"),
        SyntheticBout(30.0, "robin-1", ROBIN_PERCH_M, {ROBIN: 0.33}),
    ]
    return TrackerScenario("weak_fix_weak_id", {"robin-1": ROBIN}, bouts)


def flock_same_species() -> TrackerScenario:
    perches = {"robin-1": (2.0, 10.0, 2.0), "robin-2": (24.0, 12.0, 2.0), "robin-3": (12.0, 32.0, 2.0)}
    rng = np.random.default_rng(CATALOGUE_SEED + 1)
    order = ["robin-1", "robin-2", "robin-3"] * 4
    times = _gaps(rng, len(order), 4.0, 8.0)
    bouts = [
        SyntheticBout(t, bird, perches[bird], {ROBIN: float(rng.uniform(0.45, 0.7))})
        for t, bird in zip(times, order)
    ]
    return TrackerScenario("flock_same_species", {bird: ROBIN for bird in perches}, bouts)


TRACKER_SCENARIOS = (
    intermittent_perched_robin,
    robin_hops_15m,
    two_species_alternating,
    two_species_overlapping,
    weak_fix_weak_id,
    flock_same_species,
)

# Wildlife held on its perch (stationary_hold, the shipped default since
# localization overhaul milestone 3) or moving under the constant-velocity filter.
WILDLIFE_STATIONARY_HOLD = {
    "tracking_mode_by_category": tracking_mode_by_category_with("wildlife", TrackingMode.STATIONARY_HOLD)
}
WILDLIFE_CONTINUOUS_MOTION = {
    "tracking_mode_by_category": tracking_mode_by_category_with("wildlife", TrackingMode.CONTINUOUS_MOTION)
}
# The tracker before milestone 3 changed the defaults: wildlife on the
# constant-velocity filter and no bird entities. "Legacy" rows compare to this.
LEGACY_TRACKER_SETTINGS = {**WILDLIFE_CONTINUOUS_MOTION, "entity_reporting_enabled": False}

ALL_TRACKER_FLAGS_ON = {
    **WILDLIFE_STATIONARY_HOLD,
    # Entity rows add it on top; these rows score the tracker alone.
    "entity_reporting_enabled": False,
    "tracking_species_log_odds_enabled": True,
    "tracking_partial_constraint_binding_enabled": True,
    "birdnet_two_threshold_enabled": True,
}
