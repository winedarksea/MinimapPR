"""Flag-off vs flag-on entity metrics on the tracker detection scenarios.

    python scripts/measure_tracker_scenarios.py [--seeds 0 1 2]

Prints one markdown row per (scenario, configuration), averaged over seeds.
Results are recorded in docs/field_mode_progress/tracker.md.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimappr.sim.tracker_detection_scenarios import score_tracker_scenario  # noqa: E402
from minimappr.sim.tracker_scenario_catalog import (  # noqa: E402
    ALL_TRACKER_FLAGS_ON,
    TRACKER_SCENARIOS,
)

CONFIGURATIONS = {
    "off": {},
    "perched": {"tracking_perched_filter_enabled": True},
    "log-odds+2thr": {"tracking_species_log_odds_enabled": True, "birdnet_two_threshold_enabled": True},
    "all-on": ALL_TRACKER_FLAGS_ON,
}


def _mean(values):
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.2f}"


async def main_async(seeds: list[int]) -> None:
    print("| Scenario | Config | Surfaced dets | Visible tracks | L&C recall | Shown-label acc. | Swap rate | Duplicates | Multi-individual entities |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for build in TRACKER_SCENARIOS:
        scenario = build()
        for name, overrides in CONFIGURATIONS.items():
            results = [await score_tracker_scenario(scenario, overrides, seed=seed) for seed in seeds]
            print(
                f"| {scenario.name} | {name} "
                f"| {_fmt(_mean([r.surfaced_detections for r in results]))} "
                f"| {_fmt(_mean([r.distinct_visible_tracks for r in results]))} "
                f"| {_fmt(_mean([r.metrics.localized_and_classified_recall for r in results]))} "
                f"| {_fmt(_mean([r.surfaced_label_accuracy for r in results]))} "
                f"| {_fmt(_mean([r.metrics.label_swap_rate for r in results]))} "
                f"| {_fmt(_mean([r.metrics.duplicate_entities for r in results]))} "
                f"| {_fmt(_mean([r.metrics.entities_spanning_multiple_individuals for r in results]))} |"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    asyncio.run(main_async(parser.parse_args().seeds))


if __name__ == "__main__":
    main()
