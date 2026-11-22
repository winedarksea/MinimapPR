#!/usr/bin/env python3
"""Server-track S gate: replay sim scenes under baseline / one-stream / all-on / all-on-minus-one.

Library and scoring rules: ``minimappr/sim/server_track_gate.py``. Results and
verdicts: ``docs/field_mode_progress/server_track_gate.md``.

    # run (scenes run sequentially; split scenes across processes to parallelise)
    python scripts/server_track_gate.py run --json out.json [--scenes a b] [--configs baseline all_on]
    # merge one or more run files, print tables + gate result + per-stream deltas
    python scripts/server_track_gate.py summarize out1.json out2.json [--markdown table.md]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimappr.sim import server_track_gate as gate  # noqa: E402

TABLE_COLUMNS = (
    ("surfaced_detections", "Dets"),
    ("localized_and_classified_individual_recall", "Indiv. recall"),
    ("localized_and_classified_recall", "Event recall"),
    ("label_swap_count", "Swaps"),
    ("duplicate_entities", "Dup"),
    ("echo_promotions", "Echo"),
    ("entities_spanning_multiple_individuals", "Merged"),
    ("unassociated_localized_entities", "Unassoc."),
    ("median_position_error_m", "Median err (m)"),
    ("p90_position_error_m", "P90 err (m)"),
)


async def run(scenes: list[str], configurations: list[str], work_dir: Path, json_path: Path | None) -> list[dict]:
    # Resume: keep rows already in the output file and skip their (scene, config).
    rows: list[dict] = json.loads(json_path.read_text()) if json_path and json_path.exists() else []
    done = {(r["scene"], r["configuration"]) for r in rows}
    for scene in scenes:
        for configuration in configurations:
            if (scene, configuration) in done:
                continue
            row = await gate.replay_scene_configuration(scene, configuration, work_dir)
            printable = {k: v for k, v in row.items() if k not in ("onset_clustering_s", "raw")}
            print(json.dumps(printable), flush=True)
            rows.append(row)
            if json_path:
                json_path.write_text(json.dumps(rows, default=str))
    return rows


def _cell(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def scene_table(rows: list[dict], configurations: list[str]) -> str:
    header = "| Scene | Config | " + " | ".join(label for _, label in TABLE_COLUMNS) + " |"
    lines = [header, "|" + " --- |" * (len(TABLE_COLUMNS) + 2)]
    for scene in dict.fromkeys(r["scene"] for r in rows):
        for configuration in configurations:
            match = [r for r in rows if r["scene"] == scene and r["configuration"] == configuration]
            if match:
                row = match[0]
                lines.append(f"| {scene} | {configuration} | " + " | ".join(_cell(row.get(k)) for k, _ in TABLE_COLUMNS) + " |")
    return "\n".join(lines)


def summary_table(rows: list[dict], configurations: list[str]) -> str:
    keys = ("individual_recall_sum", "event_recall_sum", "swaps", "duplicates", "echo_promotions",
            "merged_entities", "unassociated_localized", "median_of_scene_median_error_m")
    lines = ["| Config | " + " | ".join(keys) + " |", "|" + " --- |" * (len(keys) + 1)]
    for configuration in configurations:
        subset = [r for r in rows if r["configuration"] == configuration]
        if subset:
            summary = gate.configuration_summary(subset)
            lines.append(f"| {configuration} | " + " | ".join(_cell(summary[k]) for k in keys) + " |")
    return "\n".join(lines)


def cost_lines(rows: list[dict]) -> list[str]:
    lines = []
    for configuration in (gate.BASELINE, gate.ALL_ON):
        subset = [r for r in rows if r["configuration"] == configuration]
        if not subset:
            continue
        durations = [d for r in subset for d in r.get("onset_clustering_s", [])]
        reruns = sum(r["loco_reruns"] for r in subset)
        rerun_ms = sum(r["loco_rerun_ms"] for r in subset)
        audio = sum(r["audio_seconds"] for r in subset)
        wall = sum(r["replay_wall_s"] for r in subset)
        text = f"{configuration}: replay wall {wall:.0f} s for {audio:.0f} s of audio"
        if durations:
            text += (
                f"; onset clustering {len(durations)} windows, median {statistics.median(durations) * 1000:.0f} ms,"
                f" max {max(durations) * 1000:.0f} ms per window"
            )
        if reruns:
            text += f"; LOCO {reruns} reruns, mean {rerun_ms / reruns:.1f} ms"
        lines.append(text)
    return lines


def stream_deltas(rows: list[dict]) -> list[str]:
    """Per stream: alone vs baseline and all-on vs all-on-minus-stream, summed over scenes."""
    by_config: dict[str, list[dict]] = {}
    for row in rows:
        by_config.setdefault(row["configuration"], []).append(row)
    lines = ["| Stream | alone − baseline (recall / swaps / dup / echo / merged) | all-on − minus-stream (recall / swaps / dup / echo / merged) |",
             "| --- | --- | --- |"]

    def delta(a: list[dict], b: list[dict]) -> str:
        sa, sb = gate.configuration_summary(a), gate.configuration_summary(b)
        keys = ("individual_recall_sum", "swaps", "duplicates", "echo_promotions", "merged_entities")
        return " / ".join(f"{sa[k] - sb[k]:+.2f}" if isinstance(sa[k], float) else f"{sa[k] - sb[k]:+d}" for k in keys)

    for stream in gate.STREAM_FLAGS:
        alone = by_config.get(f"{gate.ALONE_PREFIX}{stream}")
        minus = by_config.get(f"{gate.LEAVE_ONE_OUT_PREFIX}{stream}")
        base, all_on = by_config.get(gate.BASELINE), by_config.get(gate.ALL_ON)
        left = delta(alone, base) if alone and base else "—"
        right = delta(all_on, minus) if all_on and minus else "—"
        lines.append(f"| {stream} | {left} | {right} |")
    return lines


def summarize(paths: list[Path], markdown_path: Path | None, *, rescore: bool = False) -> int:
    rows = [row for path in paths for row in json.loads(path.read_text())]
    if rescore:
        # Re-apply the current scoring rules to the stored replay rows (no replay).
        rows = [gate.rescore_row(row) for row in rows]
    configurations = list(dict.fromkeys(r["configuration"] for r in rows))
    parts = ["## Summary over scenes", summary_table(rows, configurations), "",
             "## Per-stream deltas", *stream_deltas(rows), "", "## Cost", *cost_lines(rows), "",
             "## Scenes × configs", scene_table(rows, configurations)]
    baseline = [r for r in rows if r["configuration"] == gate.BASELINE]
    all_on = [r for r in rows if r["configuration"] == gate.ALL_ON]
    if baseline and all_on:
        failures = gate.all_on_gate_failures(all_on, baseline)
        parts += ["", "## ALL-ON gate: " + ("PASS" if not failures else "FAIL"), *[f"- {f}" for f in failures]]
    text = "\n".join(parts)
    print(text)
    if markdown_path:
        markdown_path.write_text(text + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--scenes", nargs="+", default=list(gate.GATE_SCENES))
    run_parser.add_argument("--configs", nargs="+", default=gate.all_configuration_names())
    run_parser.add_argument("--work-dir", type=Path, default=None)
    run_parser.add_argument("--json", type=Path, default=None)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("paths", nargs="+", type=Path)
    summary_parser.add_argument("--markdown", type=Path, default=None)
    summary_parser.add_argument("--rescore", action="store_true", help="re-score stored rows with current rules")
    args = parser.parse_args()
    if args.command == "summarize":
        return summarize(args.paths, args.markdown, rescore=args.rescore)
    for configuration in args.configs:
        gate.configuration_flags(configuration)  # fail fast on a typo
    if args.work_dir is not None:
        asyncio.run(run(args.scenes, args.configs, args.work_dir, args.json))
    else:
        with tempfile.TemporaryDirectory(prefix="server-track-gate-") as tmp:
            asyncio.run(run(args.scenes, args.configs, Path(tmp), args.json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
