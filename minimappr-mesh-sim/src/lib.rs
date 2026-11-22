//! Layer 2 deterministic mesh simulator for MinimapPR Field Mode
//! (`docs/field_mode_test_strategy.md`, Layer 2).
//!
//! The real sans-IO node and host state machines from
//! `minimappr_core::mesh` run over a seeded simulated network
//! ([`simulated_network`]) on a virtual clock ([`scenario_runner`]), against
//! an arrival-time acoustic world ([`acoustic_world`]) that uses
//! `minimappr-sim`'s physics constants and ISO 9613-1 absorption. One seed
//! drives geometry jitter, clocks, GPS error, latency, loss and timing
//! noise, so `MINIMAPPR_SIM_SEED=<seed> cargo test -p minimappr-mesh-sim
//! <scenario>` reproduces any failure exactly.

pub mod acoustic_world;
pub mod device_harness;
pub mod device_spec;
pub mod entity_metrics;
pub mod invariants;
pub mod run_metrics;
pub mod runner_actions;
pub mod scenario;
pub mod scenario_outcome;
pub mod scenario_runner;
pub mod scenarios;
pub mod seeded_random;
pub mod simulated_network;
pub mod synthetic_clip;
pub mod synthetic_clusters;

pub use scenario_outcome::ScenarioOutcome;
pub use scenario_runner::ScenarioRun;
pub use scenarios::{run_scenario_by_name, ScenarioReport, SCENARIO_NAMES};
