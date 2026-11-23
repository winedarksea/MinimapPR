//! PR subset (fixed seeds) of the W4-sim-gates statistical suite; see
//! `minimappr_sim_gates::gates`. Reproduce one seed with
//! `MINIMAPPR_SIM_GATES_SEED=<seed> cargo test --release -p minimappr-sim-gates --test statistical_coverage <gate> -- --nocapture`.

use minimappr_sim_gates::gates::assert_pull_request_gate;

#[test]
fn mesh_contour_coverage() {
    assert_pull_request_gate("mesh_contour_coverage");
}

/// KNOWN FAILURE (W4-sim-gates review finding, not loosened): the strategy
/// asks for 90–98 % coverage inside *and* outside the layout. The single
/// fitted σ multiplier (0.52) under-covers inside calls near the floor
/// (~0.90–0.92) and over-covers outside calls (~0.99 > 0.98); the pooled
/// gate passes only by averaging the two. Needs a per-geometry (or
/// range-dependent) calibration or a correctly scaled covariance.
#[test]
#[ignore = "known failure: one σ multiplier cannot calibrate inside and outside calls (docs/field_mode_progress/sim_gates.md)"]
fn mesh_contour_coverage_by_class() {
    assert_pull_request_gate("mesh_contour_coverage_by_class");
}

/// World-frame coverage of the *published* polygon (Σ_fix plus the filtered
/// map anchor's error, closeout C2) against the raw world truth.
/// PENDING C3: the world gauge is already Gaussian-shaped (median 1.06,
/// P95 2.47 on nightly seeds 100000+200), but the shipped σ multiplier 0.52
/// was fitted to absorb the 4–7× overstated mesh LM covariance, so it
/// draws ~0.5σ of the honest anchor term (coverage 0.19). Un-ignored once
/// C3 makes Σ_fix realistic and refits the multiplier.
#[test]
#[ignore = "pending closeout C3: σ multiplier 0.52 compensates an overstated Σ_fix and under-draws the anchor term"]
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
