"""Flag-off vs flag-on entity metrics for the W2-onset-srp simulator scenarios.

Renders each `minimappr-sim/scenarios/onset_*.toml` scene (Rust simulator
binary, see minimappr/sim/sim_scenario_scoring.py), replays it with
`localization_onset_clustering_enabled` false and true, and prints one row per
(scene, flag, reporting window). Results: docs/field_mode_progress/onset_srp.md.

    python scripts/measure_onset_srp_scenarios.py [--reporting-windows 30 1] [--json out.json]

`--reporting-windows`: ReportingFusionPolicy merges same-label detections from
one node inside this window (30 s default). 1 s isolates per-call
localization + classification from that bucketing (W3-entity replaces it).
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
ENTITY_KEYS = (
    "localized_and_classified_recall",
    "label_swap_rate",
    "duplicate_entities",
    "echo_promotions",
    "unassociated_localized_entities",
)


async def measure(scenario_path: Path, flag: bool, reporting_window_s: float, work_dir: Path) -> dict:
    report, detections = await replay_and_score_scenario(
        scenario_path,
        work_dir,
        settings_overrides={
            "localization_onset_clustering_enabled": flag,
            "reporting_window_seconds": reporting_window_s,
        },
    )
    entity = asdict(report.entity_metrics)
    return {
        "scenario": scenario_path.stem,
        "onset_clustering": flag,
        "reporting_window_s": reporting_window_s,
        "detections": len(detections),
        "labels": sorted({str(d.get("label")) for d in detections}),
        "label_accuracy": report.label_accuracy,
        "median_position_error_m": report.median_position_error_m,
        "p90_position_error_m": report.p90_position_error_m,
        **{key: entity[key] for key in ENTITY_KEYS},
    }


async def main_async(windows: list[float], scenario_filter: str | None) -> list[dict]:
    rows = []
    with tempfile.TemporaryDirectory(prefix="onset-srp-") as tmp:
        for scenario_path in sorted(SCENARIO_DIRECTORY.glob(ONSET_SCENARIO_GLOB)):
            if scenario_filter and scenario_filter not in scenario_path.stem:
                continue
            for window in windows:
                for flag in (False, True):
                    row = await measure(
                        scenario_path, flag, window, Path(tmp) / f"{scenario_path.stem}-{flag}-{window}"
                    )
                    print(json.dumps(row), flush=True)
                    rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reporting-windows", type=float, nargs="+", default=[30.0, 1.0])
    parser.add_argument("--only", default=None, help="substring filter on scenario names")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    rows = asyncio.run(main_async(args.reporting_windows, args.only))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
