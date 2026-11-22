//! Layer 2 PR gate: every scenario on the 10 fixed seeds (or the single seed
//! in `MINIMAPPR_SIM_SEED`). One test per scenario so `cargo test` runs them
//! in parallel and a failure names the scenario and the seed to reproduce:
//! `MINIMAPPR_SIM_SEED=<seed> cargo test -p minimappr-mesh-sim <scenario>`.

use minimappr_mesh_sim::run_scenario_by_name;
use minimappr_mesh_sim::seeded_random::seeds_for_test;

fn run_all_seeds(name: &str) {
    let mut failures = Vec::new();
    for seed in seeds_for_test() {
        match run_scenario_by_name(name, seed) {
            Ok(report) => println!("{name} seed {seed}: {}", report.summary),
            Err(message) => failures.push(format!("{message}\n  reproduce: MINIMAPPR_SIM_SEED={seed} cargo test -p minimappr-mesh-sim {name}")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

macro_rules! scenario_tests {
    ($($name:ident),* $(,)?) => {
        $( #[test] fn $name() { run_all_seeds(stringify!($name)); } )*
    };
}

scenario_tests!(
    baseline_localization,
    host_death,
    network_faults,
    route_change,
    island_split_merge,
    offline_host_loss,
    walking_at_call,
    mirrored_layout,
    readiness_cap,
    mic_only_round_robin,
    muted_phone,
    transport_fallback,
    thermal_shedding,
    earbuds,
    constrained_link,
    faint_bird_one_classifier,
);

#[test]
fn every_catalogued_scenario_has_a_test() {
    assert_eq!(minimappr_mesh_sim::SCENARIO_NAMES.len(), 16);
}
