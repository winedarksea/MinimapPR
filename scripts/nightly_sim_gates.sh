#!/usr/bin/env bash
# Nightly W4-sim-gates statistical suite on fresh seeds.
#
#   scripts/nightly_sim_gates.sh [SEED_UNITS] [START_SEED] [GATE]
#
# SEED_UNITS defaults to 200 (each gate scales it by its nightly fraction:
# coverage runs 200 sessions / trials, cheap gates thousands). START_SEED
# defaults to Unix time folded below the calibration training seeds (chosen
# by the runner and printed, so the run can be repeated; nightly never
# draws a training seed). An empty START_SEED ("") also means "pick one",
# so a GATE can be given alone. Exits non-zero if any gate failed; every
# failing seed prints its `MINIMAPPR_SIM_GATES_SEED=... cargo test` repro
# command, and aggregate failures print the nightly command plus
# contributing seeds.
set -euo pipefail
cd "$(dirname "$0")/.."
runner_arguments=(--seeds "${1:-200}")
if [[ -n "${2:-}" ]]; then
    runner_arguments+=(--start-seed "$2")
fi
if [[ -n "${3:-}" ]]; then
    runner_arguments+=(--gate "$3")
fi
exec cargo run --release -q -p minimappr-sim-gates --example nightly_sim_gates -- "${runner_arguments[@]}"
