#!/usr/bin/env bash
# Nightly W4-sim-gates statistical suite on fresh seeds.
#
#   scripts/nightly_sim_gates.sh [SEED_UNITS] [START_SEED] [GATE]
#
# SEED_UNITS defaults to 200 (each gate scales it by its nightly fraction:
# coverage runs 200 sessions / trials, cheap gates thousands). START_SEED
# defaults to Unix time modulo the calibration training start (printed, so
# the run can be repeated; nightly never draws a training seed). GATE limits
# the run to one gate. Exits non-zero if any gate failed; every failing seed
# prints its `MINIMAPPR_SIM_GATES_SEED=... cargo test` repro command, and
# aggregate failures print the nightly command plus contributing seeds.
set -euo pipefail
cd "$(dirname "$0")/.."
seed_units="${1:-200}"
start_seed="${2:-$(( $(date +%s) % 5000000 ))}"
gate_arguments=()
if [[ $# -ge 3 ]]; then
    gate_arguments=(--gate "$3")
fi
exec cargo run --release -q -p minimappr-sim-gates --example nightly_sim_gates -- \
    --seeds "$seed_units" --start-seed "$start_seed" "${gate_arguments[@]+"${gate_arguments[@]}"}"
