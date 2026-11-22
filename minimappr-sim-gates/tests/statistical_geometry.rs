//! PR subset (fixed seeds) of the W4-sim-gates statistical suite; see
//! `minimappr_sim_gates::gates`. Reproduce one seed with
//! `MINIMAPPR_SIM_GATES_SEED=<seed> cargo test --release -p minimappr-sim-gates --test statistical_geometry <gate> -- --nocapture`.

use minimappr_sim_gates::gates::assert_pull_request_gate;

#[test]
fn geometric_invariance() {
    assert_pull_request_gate("geometric_invariance");
}

#[test]
fn srp_lm_invariance() {
    assert_pull_request_gate("srp_lm_invariance");
}

#[test]
fn chirality() {
    assert_pull_request_gate("chirality");
}

#[test]
fn anchor_stability() {
    assert_pull_request_gate("anchor_stability");
}
