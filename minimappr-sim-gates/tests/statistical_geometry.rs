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

/// Strategy row read literally: a 20 m jump that persists while the group
/// is still moves the anchor < 1 m. Passes since closeout C1
/// (`FilteredMapAnchor` subtracts a still phone's jump from its own fixes).
/// Since closeout F2 the 1 m is an aggregate (≤ 1 % of seeds over it) with a
/// 2.5 m per-seed cap: the twins' chaotic per-round rotation fit put 23 of
/// 100 default nightly runs over a per-seed 1 m.
#[test]
fn anchor_persistent_jump() {
    assert_pull_request_gate("anchor_persistent_jump");
}
