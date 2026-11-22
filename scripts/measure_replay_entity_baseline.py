"""Measure the replay entity-metric baseline on the synthetic entity scenarios.

Renders each scenario in `minimappr/sim/synthetic_entity_scenarios.py` to a
ground-truth v2 bundle, replays it through a full FusionNode (dominant-
frequency stub classifier, Python ingest), and prints per-scenario metrics.
Results are recorded in docs/field_mode_progress/server_baseline.md.

    python scripts/measure_replay_entity_baseline.py [--seeds 0 1 2] [--radii-m 15 5] [--json out.json]
        [--set localization_onset_clustering_enabled=true ...]

Each replay is scored once per association radius (entity metrics are only as
strict as the spatial gate; a generous radius can hide collapsed fixes).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
import copy
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimappr.calibration.bundle import load_bundle  # noqa: E402
from minimappr.sim.replay import build_fusion_for_bundle, evaluate_bundle, replay_bundle  # noqa: E402
from minimappr.sim.synthetic_entity_scenarios import (  # noqa: E402
    SYNTHETIC_ENTITY_SCENARIOS,
    DominantFrequencySpeciesClassifier,
    render_scenario_bundle,
    species_bands_for,
)

# Same overrides as tests/test_calibration_harness_synthetic.py.
REPLAY_SETTINGS_OVERRIDES = {"trigger_rms": 0.006, "classification_window_seconds": 1.0}
DETECTION_SETTLE_SECONDS = 3.0
DETECTION_DEADLINE_SECONDS = 45.0
POLL_INTERVAL_SECONDS = 0.25


async def _collect_settled_detections(storage) -> list[dict]:
    """Poll until the detection count stops changing for DETECTION_SETTLE_SECONDS."""
    deadline = time.monotonic() + DETECTION_DEADLINE_SECONDS
    last_count, last_change = -1, time.monotonic()
    detections: list[dict] = []
    while time.monotonic() < deadline:
        detections = await storage.list_detections(limit=10_000)
        if len(detections) != last_count:
            last_count, last_change = len(detections), time.monotonic()
        elif detections and time.monotonic() - last_change >= DETECTION_SETTLE_SECONDS:
            break
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
    return detections


ENTITY_METRIC_KEYS = (
    "call_event_count",
    "localized_and_classified_recall",
    "label_swap_rate",
    "label_swap_count",
    "swap_eligible_detection_count",
    "duplicate_entities",
    "entity_count_by_individual",
    "echo_promotions",
    "unassociated_localized_entities",
    "entities_spanning_multiple_individuals",
)


def _entity_metrics_at_radius(detections, bundle, association_radius_m: float) -> dict:
    expectations = copy.deepcopy(bundle.expectations)
    expectations.setdefault("entities", {})["association_radius_m"] = association_radius_m
    entity = asdict(evaluate_bundle(detections, bundle, expectations).entity_metrics)
    return {key: entity[key] for key in ENTITY_METRIC_KEYS}


async def measure_scenario(
    scenario, seed: int, work_dir: Path, radii_m: list[float], extra_overrides: dict | None = None
) -> dict:
    bundle_path = render_scenario_bundle(scenario, work_dir / f"{scenario.name}-{seed}.zip", seed=seed)
    bundle = load_bundle(bundle_path)
    fusion, storage, _ = await build_fusion_for_bundle(
        bundle,
        work_dir / f"replay-{scenario.name}-{seed}",
        overrides={**REPLAY_SETTINGS_OVERRIDES, **(extra_overrides or {})},
        classifier=DominantFrequencySpeciesClassifier(species_bands_for(scenario)),
    )
    try:
        await replay_bundle(fusion, bundle)
        detections = await _collect_settled_detections(storage)
    finally:
        await fusion.stop()
        await storage.close()
    report = evaluate_bundle(detections, bundle)
    return {
        "scenario": scenario.name,
        "seed": seed,
        "detections": len(detections),
        "localized_detections": sum(
            1 for d in detections if d.get("reporting_modality") != "omni"
        ),
        "distinct_tracks": len({d.get("track_id") for d in detections if d.get("track_id")}),
        "labels": sorted({str(d.get("label")) for d in detections}),
        "event_recall": report.event_recall,
        "label_accuracy": report.label_accuracy,
        "median_position_error_m": report.median_position_error_m,
        "p90_position_error_m": report.p90_position_error_m,
        "legacy_errors": list(report.errors),
        "entity_metrics_by_radius_m": {
            str(radius_m): _entity_metrics_at_radius(detections, bundle, radius_m)
            for radius_m in radii_m
        },
    }


async def main_async(seeds: list[int], radii_m: list[float], extra_overrides: dict | None = None) -> list[dict]:
    results = []
    with tempfile.TemporaryDirectory(prefix="entity-baseline-") as tmp:
        for scenario in SYNTHETIC_ENTITY_SCENARIOS:
            for seed in seeds:
                result = await measure_scenario(scenario, seed, Path(tmp), radii_m, extra_overrides)
                print(json.dumps(result), flush=True)
                results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--radii-m", type=float, nargs="+", default=[15.0, 5.0])
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=JSON",
        help="extra Settings override, value parsed as JSON (e.g. localization_onset_clustering_enabled=true)",
    )
    args = parser.parse_args()
    extra = {key: json.loads(value) for key, value in (item.split("=", 1) for item in args.set)}
    results = asyncio.run(main_async(args.seeds, args.radii_m, extra))
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
