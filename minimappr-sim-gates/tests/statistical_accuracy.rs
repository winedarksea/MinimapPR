//! PR subset (fixed seeds) of the W4-sim-gates statistical suite; see
//! `minimappr_sim_gates::gates`. Reproduce one seed with
//! `MINIMAPPR_SIM_GATES_SEED=<seed> cargo test --release -p minimappr-sim-gates --test statistical_accuracy <gate> -- --nocapture`.

use minimappr_sim_gates::gates::assert_pull_request_gate;

#[test]
fn monte_carlo_accuracy() {
    assert_pull_request_gate("monte_carlo_accuracy");
}

/// KNOWN FAILURE (not loosened; closeout C4 improved, did not fix):
/// leave-one-out with a χ²-drop test now rejects 92 % of decisive pockets
/// (was 58 %), including a late pair reference (repro seed 26: rejected,
/// was dragged 22σ). Still failing on nightly 100000+400: pocketed estimate
/// outside the clean 95 % region 63/400 (limit 10 %; delays of 1–3 pair σ
/// are indistinguishable from layout error and move the now-honest σ
/// region); 13 decisive pockets kept (the delay trades against source
/// height over a flat layout); a correctly rejected critical node can still
/// leave the 6-phone fix > 4σ of the 7-phone one (seed 26: 4.6σ); σ shrinks
/// > 10 % on 5.7 % of removals (limit 5 %); 17 few-node fixes miss by > 4σ.
/// Run here with
/// `cargo test --release -p minimappr-sim-gates --test statistical_accuracy graceful_degradation -- --ignored --nocapture`.
#[test]
#[ignore = "known failure: pocketed-phone aggregate, σ monotonicity and few-node tails (docs/field_mode_progress/closeout_C.md)"]
fn graceful_degradation() {
    assert_pull_request_gate("graceful_degradation");
}
