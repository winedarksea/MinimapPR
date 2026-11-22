#!/usr/bin/env bash
# Nightly Layer 2 mesh simulation: every scenario on fresh seeds.
#
#   scripts/nightly_mesh_sim.sh [SEED_COUNT] [START_SEED]
#
# SEED_COUNT defaults to 200; START_SEED defaults to the current Unix time
# (printed, so the run can be repeated). Exits non-zero if any run failed;
# each failure prints the seed and the `MINIMAPPR_SIM_SEED=... cargo test`
# command that reproduces it. After fixing, add the seed to
# PULL_REQUEST_SEEDS in minimappr-mesh-sim/src/seeded_random.rs.
set -euo pipefail
cd "$(dirname "$0")/.."
seed_count="${1:-200}"
start_seed="${2:-$(date +%s)}"
exec cargo run --release -q -p minimappr-mesh-sim --example nightly_mesh_seeds -- \
    --seeds "$seed_count" --start-seed "$start_seed"
