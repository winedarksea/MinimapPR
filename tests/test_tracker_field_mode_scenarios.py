"""Scenario gates for the W2-tracker flags on synthetic detection sequences.

Replay bundles collapse fixes onto nodes today (server_baseline.md), so these
drive the real assembly track decision with realistic jittered fixes instead
(``minimappr/sim/tracker_detection_scenarios.py``). Flag-off numbers are pinned
too so a regression in the legacy path shows up here.
"""

from __future__ import annotations

import pytest

from minimappr.sim.tracker_detection_scenarios import score_tracker_scenario
from minimappr.sim.tracker_scenario_catalog import (
    ALL_TRACKER_FLAGS_ON,
    flock_same_species,
    intermittent_perched_robin,
    robin_hops_15m,
    two_species_alternating,
    two_species_overlapping,
    weak_fix_weak_id,
)

SEEDS = (0, 1, 2)


async def _scores(build, overrides):
    scenario = build()
    return [await score_tracker_scenario(scenario, overrides, seed=seed) for seed in SEEDS]


@pytest.mark.parametrize(
    "build",
    [intermittent_perched_robin, robin_hops_15m, two_species_alternating, two_species_overlapping,
     weak_fix_weak_id],
)
async def test_all_flags_on_full_recall_no_duplicates_no_swaps(build):
    for result in await _scores(build, ALL_TRACKER_FLAGS_ON):
        metrics = result.metrics
        assert metrics.duplicate_entities == 0, result
        assert metrics.entities_spanning_multiple_individuals == 0, result
        assert metrics.label_swap_rate in (None, 0.0), result
        assert result.surfaced_label_accuracy == 1.0, result
        floor = 0.8 if build is weak_fix_weak_id else 1.0
        assert metrics.localized_and_classified_recall >= floor, result


@pytest.mark.parametrize(
    "build", [intermittent_perched_robin, two_species_alternating, two_species_overlapping, weak_fix_weak_id]
)
async def test_flags_raise_recall_over_legacy(build):
    legacy = await _scores(build, {})
    flagged = await _scores(build, ALL_TRACKER_FLAGS_ON)
    for before, after in zip(legacy, flagged):
        assert after.metrics.localized_and_classified_recall > before.metrics.localized_and_classified_recall
        assert after.metrics.duplicate_entities <= before.metrics.duplicate_entities


async def test_legacy_two_species_fragments_and_mixes_individuals():
    """Pins the failure the perched filter fixes (CV coasting over 12 m)."""
    legacy = await _scores(two_species_alternating, {})
    assert all(r.metrics.duplicate_entities > 0 for r in legacy)
    perched = await _scores(two_species_alternating, {"tracking_perched_filter_enabled": True})
    assert all(r.metrics.duplicate_entities == 0 for r in perched)


async def test_flock_costs_at_most_one_relocation_mistake():
    for result in await _scores(flock_same_species, ALL_TRACKER_FLAGS_ON):
        assert result.distinct_visible_tracks == 3
        assert result.metrics.duplicate_entities <= 1
        assert result.metrics.entities_spanning_multiple_individuals <= 1
    legacy = await _scores(flock_same_species, {})
    assert all(r.metrics.duplicate_entities > 1 for r in legacy)
