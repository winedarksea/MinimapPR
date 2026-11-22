"""Field Mode W3-entity: entity metrics on the tracker detection scenarios.

Gate: turning entity reporting on never raises duplicates (or multi-individual
entities, or swaps) over the tracker-only view, and removes the omni/localized
duplicates the plan calls out.
"""

from __future__ import annotations

import pytest

from minimappr.sim.entity_scenario_catalog import ENTITY_SCENARIOS
from minimappr.sim.tracker_detection_scenarios import score_tracker_scenario
from minimappr.sim.tracker_scenario_catalog import ALL_TRACKER_FLAGS_ON, TRACKER_SCENARIOS

SEEDS = (0, 1, 2)
TRACKER_CONFIGURATIONS = {"off": {}, "all-on": ALL_TRACKER_FLAGS_ON}


def _mean(values) -> float:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else 0.0


async def _scores(build, overrides):
    scenario = build()
    return [await score_tracker_scenario(scenario, overrides, seed=seed) for seed in SEEDS]


@pytest.mark.parametrize("configuration", sorted(TRACKER_CONFIGURATIONS))
@pytest.mark.parametrize("build", [*TRACKER_SCENARIOS, *ENTITY_SCENARIOS], ids=lambda build: build.__name__)
async def test_entities_never_add_duplicates_over_tracker_only(build, configuration) -> None:
    tracker_only = TRACKER_CONFIGURATIONS[configuration]
    baseline = await _scores(build, tracker_only)
    with_entities = await _scores(build, {**tracker_only, "entity_reporting_enabled": True})
    assert _mean(r.metrics.duplicate_entities for r in with_entities) <= _mean(
        r.metrics.duplicate_entities for r in baseline
    )
    assert _mean(r.metrics.entities_spanning_multiple_individuals for r in with_entities) <= _mean(
        r.metrics.entities_spanning_multiple_individuals for r in baseline
    )
    assert _mean(r.metrics.label_swap_rate for r in with_entities) <= _mean(
        r.metrics.label_swap_rate for r in baseline
    )


@pytest.mark.parametrize("build", ENTITY_SCENARIOS, ids=lambda build: build.__name__)
async def test_omni_and_localized_bouts_are_one_entity_per_bird(build) -> None:
    results = await _scores(build, {**ALL_TRACKER_FLAGS_ON, "entity_reporting_enabled": True})
    for result in results:
        assert result.metrics.duplicate_entities == 0
        assert result.metrics.entities_spanning_multiple_individuals == 0
        assert result.metrics.localized_and_classified_recall == pytest.approx(1.0)
        assert result.distinct_entities == result.metrics.individual_count


async def test_flag_off_harness_output_is_unchanged() -> None:
    """No entity ids leak into the tracker-only path (flag off = old behaviour)."""
    from minimappr.sim.tracker_detection_scenarios import run_tracker_scenario

    detections, _ = await run_tracker_scenario(ENTITY_SCENARIOS[0](), ALL_TRACKER_FLAGS_ON, seed=0)
    assert detections and all("entity_id" not in detection for detection in detections)
