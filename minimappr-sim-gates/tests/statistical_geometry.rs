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

/// KNOWN FAILURE (W4-sim-gates review finding, not loosened): the strategy
/// row "a 20 m GPS jump on one phone while the group is still moves the
/// anchor less than 1 m" fails when the jump persists: `MapAnchorFilter`
/// plus the Huber fit still follow a 3.3σ offset by ~2 m median, up to ~7 m.
/// `anchor_stability` gates only a 30 s transient jump. Nightly runs this.
#[test]
#[ignore = "known failure: persistent GPS jump moves the still anchor > 1 m (docs/field_mode_progress/sim_gates.md)"]
fn anchor_persistent_jump() {
    assert_pull_request_gate("anchor_persistent_jump");
}
