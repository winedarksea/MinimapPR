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

/// Drawn SRP region (contour or fan) coverage plus sharpness and accuracy.
/// Passes since closeout D3: pairs are max-pooled over ±k·σ_pair of their
/// declared layout/clock σ, and the temperature refitted on that surface.
/// Closeout F2 widened its coverage ceiling and inside-P95 bound to the
/// measured run spread (`CoverageGateConfig::srp_*`).
#[test]
fn srp_contour_coverage() {
    assert_pull_request_gate("srp_contour_coverage");
}
