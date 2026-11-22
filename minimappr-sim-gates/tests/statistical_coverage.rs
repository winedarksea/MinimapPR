//! PR subset (fixed seeds) of the W4-sim-gates statistical suite; see
//! `minimappr_sim_gates::gates`. Reproduce one seed with
//! `MINIMAPPR_SIM_GATES_SEED=<seed> cargo test --release -p minimappr-sim-gates --test statistical_coverage <gate> -- --nocapture`.

use minimappr_sim_gates::gates::assert_pull_request_gate;

#[test]
fn mesh_contour_coverage() {
    assert_pull_request_gate("mesh_contour_coverage");
}

/// KNOWN FAILURE (W4-sim-gates finding, not loosened): with realistic
/// ±10 cm / ±10 µs residuals no scalar likelihood temperature gives a
/// calibrated *and* sharp SRP region (23 % coverage at the shipped 0.002;
/// 95 % only at 0.0126, where inside regions become whole-grid fans). The
/// nightly runner still runs and reports it. Run it here with
/// `cargo test --release -p minimappr-sim-gates --test statistical_coverage srp_contour_coverage -- --ignored --nocapture`.
#[test]
#[ignore = "known failure: SRP likelihood region not calibratable by temperature (docs/field_mode_progress/sim_gates.md)"]
fn srp_contour_coverage() {
    assert_pull_request_gate("srp_contour_coverage");
}
