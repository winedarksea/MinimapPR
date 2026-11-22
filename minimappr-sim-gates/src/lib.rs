//! W4-sim-gates: the statistical gate suite from
//! `docs/field_mode_test_strategy.md` §What to assert.
//!
//! Each gate is a library function over a seed list returning a
//! [`gate_seeds::GateOutcome`], so the fixed-seed `cargo test` subset
//! (`tests/`) and the nightly many-seed runner
//! (`examples/nightly_sim_gates.rs`, `scripts/nightly_sim_gates.sh`) share one
//! implementation. Calibrated production parameters live in core configs:
//! `srp_layout::config::RobustScoringConfig::likelihood_temperature` and
//! `mesh::config::EntityContourConfig::contour_sigma_multiplier`, fitted by
//! `examples/calibrate_region_parameters.rs` on training seeds disjoint from
//! the gated ones.

pub mod arrival_trials;
pub mod gate_config;
pub mod gate_seeds;
pub mod heap_accounting;
pub mod mesh_region_trials;
pub mod srp_scene;
pub mod srp_trials;
pub mod statistics;
pub mod thread_cpu_time;
pub mod trial_geometry;
pub mod gates;
pub mod layout_chain;
