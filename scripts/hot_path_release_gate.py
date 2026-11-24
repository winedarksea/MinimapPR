#!/usr/bin/env python3
"""E10 hot-path release gate: measure a checkout, compare against a baseline.

The harness in *this* checkout drives the server of whichever checkout
``--checkout`` names (its own app + its own freshly built debug sidecar), so a
baseline commit can be measured from a worktree without the harness files:

    git worktree add /tmp/minimappr-baseline <baseline-commit>
    python scripts/hot_path_release_gate.py run --checkout /tmp/minimappr-baseline \\
        --repeat 3 --output /tmp/baseline.json
    python scripts/hot_path_release_gate.py run --checkout . --repeat 3 --output /tmp/candidate.json
    python scripts/hot_path_release_gate.py compare /tmp/baseline.json /tmp/candidate.json

The committed baseline (``tests/data/hot_path_baseline.json``) was recorded with
``run --checkout . --scenario full --repeat 3`` on the machine it names; latency
and RSS only compare meaningfully on the same machine.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimappr.sim.replay_gate_config import replay_gate_config_from_mapping  # noqa: E402
from tests.hot_path_release_gate import run_hot_path_release_gate, write_result  # noqa: E402
from tests.hot_path_release_gate_compare import compare_hot_path_results  # noqa: E402
from tests.hot_path_release_gate_scenario import SCENARIOS_BY_NAME  # noqa: E402


def _load_gate_config(path: Path | None):
    mapping = json.loads(path.read_text(encoding="utf-8")) if path else {}
    return replay_gate_config_from_mapping(mapping).hot_path


def _compare(baseline_path: Path, candidate_path: Path, gate_config_path: Path | None) -> int:
    comparison = compare_hot_path_results(
        json.loads(baseline_path.read_text(encoding="utf-8")),
        json.loads(candidate_path.read_text(encoding="utf-8")),
        _load_gate_config(gate_config_path),
    )
    print(comparison.format_table())
    print("PASS" if comparison.passed else "FAIL")
    return 0 if comparison.passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="measure one checkout")
    run_parser.add_argument("--checkout", type=Path, default=Path("."))
    run_parser.add_argument("--scenario", choices=sorted(SCENARIOS_BY_NAME), default="full")
    run_parser.add_argument("--repeat", type=int, default=3)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--work-dir", type=Path, help="keep per-run logs and endpoint snapshots here")
    run_parser.add_argument("--compare-to", type=Path, help="baseline result to compare against after the run")
    run_parser.add_argument("--gate-config", type=Path, help="JSON overrides for replay_gate_config")
    compare_parser = subparsers.add_parser("compare", help="compare two result files")
    compare_parser.add_argument("baseline", type=Path)
    compare_parser.add_argument("candidate", type=Path)
    compare_parser.add_argument("--gate-config", type=Path)
    args = parser.parse_args()

    if args.command == "compare":
        return _compare(args.baseline, args.candidate, args.gate_config)
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="minimappr-hot-path-gate-"))
    result = run_hot_path_release_gate(args.checkout, work_dir, SCENARIOS_BY_NAME[args.scenario], repeat_count=args.repeat)
    write_result(result, args.output)
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    print(f"wrote {args.output} (per-run logs in {work_dir})")
    if args.compare_to:
        return _compare(args.compare_to, args.output, args.gate_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
