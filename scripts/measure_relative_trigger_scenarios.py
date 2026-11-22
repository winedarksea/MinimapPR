"""Absolute vs noise-floor-relative ingest trigger on the Field Mode simulator scenes.

Replays each selected `minimappr-sim/scenarios/onset_*.toml` scene with onset
clustering on, the scene's `trigger_rms` overridden to `--trigger-rms` (a
level a quiet site would not hand-tune to), and
`trigger_noise_floor_relative_enabled` false then true. One JSON row per run.
Results: docs/field_mode_progress/attribution.md.

    python scripts/measure_relative_trigger_scenarios.py [--only faint] [--trigger-rms 0.001]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimappr.sim.sim_scenario_scoring import SCENARIO_DIRECTORY, replay_and_score_scenario  # noqa: E402

ONSET_SCENARIO_GLOB = "onset_*.toml"
REPORTING_WINDOW_SECONDS = 1.0


async def measure(scenario_path: Path, relative: bool, trigger_rms: float, work_dir: Path, extra: dict) -> dict:
    report, detections = await replay_and_score_scenario(
        scenario_path,
        work_dir,
        settings_overrides={
            "localization_onset_clustering_enabled": True,
            "reporting_window_seconds": REPORTING_WINDOW_SECONDS,
            "trigger_rms": trigger_rms,
            "trigger_noise_floor_relative_enabled": relative,
            **extra,
        },
    )
    entity = asdict(report.entity_metrics)
    return {
        "scenario": scenario_path.stem,
        "trigger_rms": trigger_rms,
        "relative_trigger": relative,
        "detections": len(detections),
        "labels": sorted(str(d.get("label")) for d in detections),
        "label_accuracy": report.label_accuracy,
        "localized_and_classified_recall": entity["localized_and_classified_recall"],
        "duplicate_entities": entity["duplicate_entities"],
        "echo_promotions": entity["echo_promotions"],
    }


async def main_async(scenario_filter: str | None, trigger_rms: float, extra: dict) -> list[dict]:
    rows = []
    with tempfile.TemporaryDirectory(prefix="relative-trigger-") as tmp:
        for scenario_path in sorted(SCENARIO_DIRECTORY.glob(ONSET_SCENARIO_GLOB)):
            if scenario_filter and scenario_filter not in scenario_path.stem:
                continue
            for relative in (False, True):
                work_dir = Path(tmp) / f"{scenario_path.stem}-{relative}"
                row = await measure(scenario_path, relative, trigger_rms, work_dir, extra)
                print(json.dumps(row), flush=True)
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", default=None, help="substring filter on scenario names")
    parser.add_argument("--trigger-rms", type=float, default=0.001)
    parser.add_argument("--settings", default="{}", help="extra settings overrides as JSON")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    rows = asyncio.run(main_async(args.only, args.trigger_rms, json.loads(args.settings)))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
