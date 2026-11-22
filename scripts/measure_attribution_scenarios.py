"""Attribution off vs on (onset clustering on) for the Field Mode simulator scenes.

Renders each `minimappr-sim/scenarios/onset_*.toml` scene, replays it with
`localization_onset_clustering_enabled=true` and `onset_attribution_enabled`
false then true (1 s reporting window, so ReportingFusionPolicy bucketing does
not hide second bouts), and prints one JSON row per (scene, attribution, run)
with label accuracy, entity metrics, attribution tiers and LOCO rerun cost.
Results: docs/field_mode_progress/attribution.md.

    python scripts/measure_attribution_scenarios.py [--only single_tetra] [--runs 2] [--json out.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimappr.sim.sim_scenario_scoring import SCENARIO_DIRECTORY, replay_and_score_scenario  # noqa: E402

ONSET_SCENARIO_GLOB = "onset_*.toml"
REPORTING_WINDOW_SECONDS = 1.0
ENTITY_KEYS = ("localized_and_classified_recall", "label_swap_rate", "duplicate_entities", "echo_promotions")


def _attribution_annotation(detection: dict) -> dict | None:
    features = detection.get("feature_summary") or {}
    for value in (features, *(v for v in features.values() if isinstance(v, dict))):
        if isinstance(value, dict) and isinstance(value.get("onset_attribution"), dict):
            return value["onset_attribution"]
    return None


async def measure(scenario_path: Path, attribution: bool, work_dir: Path, extra: dict) -> dict:
    report, detections = await replay_and_score_scenario(
        scenario_path,
        work_dir,
        settings_overrides={
            "localization_onset_clustering_enabled": True,
            "onset_attribution_enabled": attribution,
            "reporting_window_seconds": REPORTING_WINDOW_SECONDS,
            **extra,
        },
    )
    entity = asdict(report.entity_metrics)
    annotations = [a for a in (_attribution_annotation(d) for d in detections) if a]
    return {
        "scenario": scenario_path.stem,
        "attribution": attribution,
        "detections": len(detections),
        "labels": sorted(str(d.get("label")) for d in detections),
        "label_accuracy": report.label_accuracy,
        **{key: entity[key] for key in ENTITY_KEYS},
        "tiers": dict(Counter(a.get("tier") for a in annotations)),
        "loco_reruns": sum(1 for a in annotations if "loco_reason" in a),
    }


async def main_async(scenario_filter: str | None, runs: int, extra: dict) -> list[dict]:
    rows = []
    with tempfile.TemporaryDirectory(prefix="attribution-") as tmp:
        for scenario_path in sorted(SCENARIO_DIRECTORY.glob(ONSET_SCENARIO_GLOB)):
            if scenario_filter and scenario_filter not in scenario_path.stem:
                continue
            for run in range(runs):
                for attribution in (False, True):
                    work_dir = Path(tmp) / f"{scenario_path.stem}-{attribution}-{run}"
                    row = await measure(scenario_path, attribution, work_dir, extra)
                    row["run"] = run
                    print(json.dumps(row), flush=True)
                    rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", default=None, help="substring filter on scenario names")
    parser.add_argument("--runs", type=int, default=1, help="repeat each configuration (replay timing check)")
    parser.add_argument("--settings", default="{}", help="extra settings overrides as JSON")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    rows = asyncio.run(main_async(args.only, args.runs, json.loads(args.settings)))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
