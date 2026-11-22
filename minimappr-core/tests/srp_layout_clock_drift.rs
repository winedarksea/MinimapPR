//! In-clip clock-rate resampling: phones drifting ±60 ppm, clock models
//! anchored at session start, call 10 s later. With the rate applied the
//! fix stays at sim-clean accuracy; ignoring it leaves up to 1.2 ms of
//! inter-phone timing error (≈ 40 cm of path) and the fix degrades.

mod srp_layout_support;

use srp_layout_support::*;

const DRIFTS_PPM: [f64; 5] = [60.0, -60.0, 35.0, -45.0, 10.0];
const OFFSETS_S: [f64; 5] = [0.004, -0.011, 0.0, 0.017, -0.006];
/// Seconds between the clock anchor (sample 0) and the call.
const CALL_START_AFTER_ANCHOR_S: f64 = 10.0;
const CORRECTED_MEDIAN_ERROR_GATE_M: f64 = 0.5;
/// Uncorrected must be clearly worse, or the test is not exercising drift.
const MINIMUM_UNCORRECTED_TO_CORRECTED_ERROR_RATIO: f64 = 10.0;
const MINIMUM_UNCORRECTED_MEDIAN_ERROR_M: f64 = 0.5;

fn drifting_phone_layout() -> Vec<minimappr_sim::scenario::DeviceConfig> {
    let mut devices = near_phone_layout(0);
    for (index, device) in devices.iter_mut().enumerate() {
        device.clock = Some(fixed_clock(OFFSETS_S[index], DRIFTS_PPM[index]));
    }
    devices
}

#[test]
fn rate_correction_keeps_accuracy_under_sixty_ppm_drift() {
    let sources = [[7.0, 9.0, 6.0], [12.0, 12.0, 12.0], [2.0, 15.0, 4.0], [10.0, 4.0, 9.0]];
    let mut corrected_errors = Vec::new();
    let mut uncorrected_errors = Vec::new();
    for (index, source) in sources.iter().enumerate() {
        let scenario =
            scene_with_call_start(drifting_phone_layout(), *source, index, 800 + index as u64, CALL_START_AFTER_ANCHOR_S);
        let rendered = render(&scenario);
        let nodes = layout_nodes(&rendered.plan);
        let mut config = config_for(&rendered.plan);
        let corrected = localize_with(&rendered, &nodes, &config);
        config.apply_clock_rate_correction = false;
        let uncorrected = localize_with(&rendered, &nodes, &config);
        let corrected_error = horizontal_error_m(corrected.estimate.position_m, *source);
        let uncorrected_error = horizontal_error_m(uncorrected.estimate.position_m, *source);
        eprintln!(
            "drift source {source:?}: corrected {corrected_error:.3} m (score {:.4}), uncorrected {uncorrected_error:.3} m (score {:.4})",
            corrected.estimate.normalized_srp_score, uncorrected.estimate.normalized_srp_score
        );
        corrected_errors.push(corrected_error);
        uncorrected_errors.push(uncorrected_error);
    }
    let corrected_median = percentile(&corrected_errors, 0.5);
    let uncorrected_median = percentile(&uncorrected_errors, 0.5);
    eprintln!("median error: corrected {corrected_median:.3} m, uncorrected {uncorrected_median:.3} m");
    assert!(corrected_median < CORRECTED_MEDIAN_ERROR_GATE_M, "corrected median {corrected_median}");
    assert!(uncorrected_median > MINIMUM_UNCORRECTED_MEDIAN_ERROR_M, "uncorrected median {uncorrected_median}");
    assert!(
        uncorrected_median > MINIMUM_UNCORRECTED_TO_CORRECTED_ERROR_RATIO * corrected_median,
        "rate correction gave too little benefit"
    );
}
