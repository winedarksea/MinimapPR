//! PR subset (fixed seeds) of the W4-sim-gates statistical suite; see
//! `minimappr_sim_gates::gates`. Reproduce one seed with
//! `MINIMAPPR_SIM_GATES_SEED=<seed> cargo test --release -p minimappr-sim-gates --test statistical_mesh <gate> -- --nocapture`.

use minimappr_sim_gates::gates::assert_pull_request_gate;

#[test]
fn island_isolation() {
    assert_pull_request_gate("island_isolation");
}

#[test]
fn loop_consistency() {
    assert_pull_request_gate("loop_consistency");
}
