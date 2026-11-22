//! The gate catalogue. Each gate maps a seed list to a [`GateOutcome`];
//! `GATES` tells the test files and the nightly runner which `cargo test`
//! target reproduces a seed and how many fixed seeds the PR subset runs.

pub mod accuracy_gate;
pub mod anchor_gate;
pub mod chirality_gate;
pub mod coverage_gate;
pub mod degradation_gate;
pub mod invariance_gate;
pub mod island_isolation_gate;
pub mod loop_consistency_gate;
pub mod resource_gate;
pub mod srp_invariance_gate;

use crate::gate_config::SimGateConfig;
use crate::gate_seeds::GateOutcome;

/// One gate invocation.
#[derive(Debug, Clone)]
pub struct GateRun {
    pub seeds: Vec<u64>,
    pub threads: usize,
    pub config: SimGateConfig,
}

pub type GateFunction = fn(&GateRun) -> GateOutcome;

/// A catalogued gate.
pub struct GateEntry {
    pub name: &'static str,
    /// Integration test file (`tests/<target>.rs`) holding its PR test.
    pub test_target: &'static str,
    pub function: GateFunction,
    /// Fixed seeds in the `cargo test` subset (sized for < ~30 s total).
    pub pull_request_seed_count: usize,
    /// Default nightly seeds per requested "seed" unit (expensive gates run fewer).
    pub nightly_seed_fraction: f64,
}

pub const GATES: &[GateEntry] = &[
    GateEntry { name: "monte_carlo_accuracy", test_target: "statistical_accuracy", function: accuracy_gate::monte_carlo_accuracy, pull_request_seed_count: 200, nightly_seed_fraction: 2.0 },
    GateEntry { name: "graceful_degradation", test_target: "statistical_accuracy", function: degradation_gate::graceful_degradation, pull_request_seed_count: 100, nightly_seed_fraction: 2.0 },
    GateEntry { name: "mesh_contour_coverage", test_target: "statistical_coverage", function: coverage_gate::mesh_contour_coverage, pull_request_seed_count: 100, nightly_seed_fraction: 1.0 },
    GateEntry { name: "mesh_contour_coverage_by_class", test_target: "statistical_coverage", function: coverage_gate::mesh_contour_coverage_by_class, pull_request_seed_count: 100, nightly_seed_fraction: 1.0 },
    GateEntry { name: "srp_contour_coverage", test_target: "statistical_coverage", function: coverage_gate::srp_contour_coverage, pull_request_seed_count: 24, nightly_seed_fraction: 1.0 },
    GateEntry { name: "geometric_invariance", test_target: "statistical_geometry", function: invariance_gate::geometric_invariance, pull_request_seed_count: 100, nightly_seed_fraction: 2.0 },
    GateEntry { name: "srp_lm_invariance", test_target: "statistical_geometry", function: srp_invariance_gate::srp_lm_invariance, pull_request_seed_count: 4, nightly_seed_fraction: 0.1 },
    GateEntry { name: "chirality", test_target: "statistical_geometry", function: chirality_gate::chirality, pull_request_seed_count: 400, nightly_seed_fraction: 10.0 },
    GateEntry { name: "anchor_stability", test_target: "statistical_geometry", function: anchor_gate::anchor_stability, pull_request_seed_count: 30, nightly_seed_fraction: 1.0 },
    GateEntry { name: "anchor_persistent_jump", test_target: "statistical_geometry", function: anchor_gate::anchor_persistent_jump, pull_request_seed_count: 30, nightly_seed_fraction: 1.0 },
    GateEntry { name: "island_isolation", test_target: "statistical_mesh", function: island_isolation_gate::island_isolation, pull_request_seed_count: 20, nightly_seed_fraction: 0.5 },
    GateEntry { name: "loop_consistency", test_target: "statistical_mesh", function: loop_consistency_gate::loop_consistency, pull_request_seed_count: 300, nightly_seed_fraction: 10.0 },
    GateEntry { name: "resource_bounds", test_target: "statistical_resources", function: resource_gate::resource_bounds, pull_request_seed_count: 1, nightly_seed_fraction: 0.02 },
];

pub fn gate_by_name(name: &str) -> Option<&'static GateEntry> {
    GATES.iter().find(|gate| gate.name == name)
}

/// Run one catalogued gate on its fixed PR seeds (or the env-override
/// seed), print the report, and panic with it on failure.
pub fn assert_pull_request_gate(name: &str) {
    let gate = gate_by_name(name).unwrap_or_else(|| panic!("unknown gate {name}"));
    let run = GateRun {
        seeds: crate::gate_seeds::pull_request_seeds(gate.pull_request_seed_count),
        threads: crate::gate_seeds::default_thread_count(),
        config: SimGateConfig::default(),
    };
    let started = std::time::Instant::now();
    let outcome = (gate.function)(&run);
    let report = outcome.report(gate.test_target);
    println!("{report}\n  ({:.2} s)", started.elapsed().as_secs_f64());
    assert!(outcome.passed(), "{report}");
}
