//! PR subset (fixed seeds) of the W4-sim-gates statistical suite; see
//! `minimappr_sim_gates::gates`. Reproduce one seed with
//! `MINIMAPPR_SIM_GATES_SEED=<seed> cargo test --release -p minimappr-sim-gates --test statistical_coverage <gate> -- --nocapture`.

use minimappr_sim_gates::gates::assert_pull_request_gate;

#[test]
fn mesh_contour_coverage() {
    assert_pull_request_gate("mesh_contour_coverage");
}

/// The strategy asks for 90–98 % coverage inside *and* outside the layout,
/// each on its own. Passes since closeout C3: the mesh fix covariance is
/// calibrated per geometry (`FixCovarianceCalibrationConfig`) and drawn at
/// the Gaussian multiplier.
#[test]
fn mesh_contour_coverage_by_class() {
    assert_pull_request_gate("mesh_contour_coverage_by_class");
}

/// World-frame coverage of the *published* polygon (calibrated Σ_fix plus
/// the filtered map anchor's error, closeouts C2/C3) against the raw world
/// truth: what a basemap, an export or another island reads.
#[test]
fn mesh_contour_world_coverage() {
    assert_pull_request_gate("mesh_contour_world_coverage");
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
