use super::*;
use crate::sync::chirp_codes::SplitMix64;
use crate::sync::chirp_waveform::{generate_plain_chirp, SweepLaw, CHIRP_BAND_15_TO_18_KHZ};

const SAMPLE_RATE_HZ: f64 = 48_000.0;

fn template() -> Vec<f32> {
    generate_plain_chirp(
        CHIRP_BAND_15_TO_18_KHZ,
        0.05,
        SweepLaw::HyperbolicFm,
        SAMPLE_RATE_HZ,
    )
}

fn noise(length: usize, amplitude: f32, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    (0..length)
        .map(|_| {
            let uniform = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            ((uniform - 0.5) * 2.0) as f32 * amplitude
        })
        .collect()
}

fn add_copy(signal: &mut [f32], template: &[f32], start: usize, gain: f32) {
    for (index, value) in template.iter().enumerate() {
        signal[start + index] += gain * value;
    }
}

fn detector() -> FirstArrivalDetector {
    FirstArrivalDetector::new(&template(), FirstArrivalConfig::default(), SAMPLE_RATE_HZ)
}

#[test]
fn picks_weaker_direct_path_before_stronger_reflection() {
    let chirp = template();
    let mut signal = noise(12_000, 0.01, 1);
    add_copy(&mut signal, &chirp, 2_000, 0.5);
    add_copy(&mut signal, &chirp, 2_300, 1.0);
    let arrival = detector()
        .detect(&signal, 0..8_000, ArrivalContext::RemoteEmitter)
        .unwrap();
    assert!((arrival.sample_index - 2_000.0).abs() < 0.5, "{arrival:?}");
    assert!(arrival.lead_before_strongest_samples > 250.0);
}

#[test]
fn rejects_body_conduction_precursor_on_self_recording() {
    let chirp = template();
    // Far-mic class path: the precursor leads by ~39 samples, a separate peak.
    let path_length_m = 0.30;
    let lead_samples = ((path_length_m / 343.0 - path_length_m / SOUND_SPEED_IN_PHONE_BODY_MPS)
        * SAMPLE_RATE_HZ)
        .round() as usize;
    let airborne_start = 3_000;
    let mut signal = noise(12_000, 0.005, 2);
    add_copy(&mut signal, &chirp, airborne_start, 1.0);
    add_copy(&mut signal, &chirp, airborne_start - lead_samples, 0.4);

    let self_arrival = detector()
        .detect(
            &signal,
            0..8_000,
            ArrivalContext::SelfRecording {
                self_path_length_m: Some(path_length_m),
            },
        )
        .unwrap();
    assert!(self_arrival.precursor_rejected);
    assert!(
        (self_arrival.sample_index - airborne_start as f64).abs() < 1.0,
        "{self_arrival:?}"
    );

    // Treated as a remote arrival, the same precursor is (wrongly) picked.
    let remote_arrival = detector()
        .detect(&signal, 0..8_000, ArrivalContext::RemoteEmitter)
        .unwrap();
    // (The coherent sum with the airborne skirt moves the precursor peak a
    // few samples, so only check that the pick is well before the airborne path.)
    assert!(remote_arrival.sample_index < airborne_start as f64 - 0.5 * lead_samples as f64);
}

#[test]
fn noise_only_window_is_a_detection_failure() {
    let signal = noise(10_000, 0.05, 3);
    let result = detector().detect(&signal, 0..8_000, ArrivalContext::RemoteEmitter);
    assert!(matches!(
        result,
        Err(ArrivalDetectionFailure::BelowNoiseFloor { .. })
    ));
}

#[test]
fn empty_window_is_reported() {
    let signal = noise(1_000, 0.05, 3);
    let result = detector().detect(&signal, 2_000..3_000, ArrivalContext::RemoteEmitter);
    assert_eq!(result, Err(ArrivalDetectionFailure::EmptySearchWindow));
}

#[test]
fn sub_sample_lag_is_resolved() {
    // Exact 0.4-sample delay: synthesise at 10x the rate and decimate
    // with a 4-sample phase (sample n = fine sample 10n − 4).
    const OVERSAMPLING: usize = 10;
    const DELAY_FINE_SAMPLES: usize = 4;
    let fine = generate_plain_chirp(
        CHIRP_BAND_15_TO_18_KHZ,
        0.05,
        SweepLaw::HyperbolicFm,
        SAMPLE_RATE_HZ * OVERSAMPLING as f64,
    );
    let shifted: Vec<f32> = (0..fine.len() / OVERSAMPLING)
        .map(|n| {
            (n * OVERSAMPLING)
                .checked_sub(DELAY_FINE_SAMPLES)
                .map_or(0.0, |index| fine[index])
        })
        .collect();
    let mut signal = noise(10_000, 0.001, 4);
    add_copy(&mut signal, &shifted, 4_000, 1.0);
    let arrival = detector()
        .detect(&signal, 0..8_000, ArrivalContext::RemoteEmitter)
        .unwrap();
    assert!(
        (arrival.sample_index - 4_000.4).abs() < 0.25,
        "{}",
        arrival.sample_index
    );
}

#[test]
fn double_emission_is_flagged() {
    let chirp = template();
    let separation = (440e-6 * SAMPLE_RATE_HZ) as usize;
    let mut signal = noise(12_000, 0.005, 5);
    add_copy(&mut signal, &chirp, 2_000, 1.0);
    add_copy(&mut signal, &chirp, 2_000 + separation, 0.8);
    let arrival = detector()
        .detect(&signal, 0..8_000, ArrivalContext::RemoteEmitter)
        .unwrap();
    assert!(arrival.double_arrival_suspected, "{arrival:?}");

    let mut single = noise(12_000, 0.005, 6);
    add_copy(&mut single, &chirp, 2_000, 1.0);
    let clean = detector()
        .detect(&single, 0..8_000, ArrivalContext::RemoteEmitter)
        .unwrap();
    assert!(!clean.double_arrival_suspected);
}

#[test]
fn raw_threshold_onset_finds_first_large_sample() {
    let mut signal = vec![0.0f32; 100];
    signal[40] = 0.3;
    signal[60] = 1.0;
    assert_eq!(raw_threshold_onset(&signal, 0..100, 0.25), Some(40));
    assert_eq!(raw_threshold_onset(&signal, 0..100, 0.5), Some(60));
}

#[test]
fn self_pick_ignores_a_close_reflection_after_the_airborne_peak() {
    // Near mic (d_ii 5 cm); a hand or table reflection 0.4 ms after the
    // airborne path at −3 dB must not be mistaken for the airborne path.
    let chirp = template();
    let airborne_start = 3_000;
    let reflection_delay_samples = 19;
    let mut signal = noise(12_000, 0.005, 4);
    add_copy(&mut signal, &chirp, airborne_start, 1.0);
    add_copy(
        &mut signal,
        &chirp,
        airborne_start + reflection_delay_samples,
        0.7,
    );
    let arrival = detector()
        .detect(
            &signal,
            0..8_000,
            ArrivalContext::SelfRecording {
                self_path_length_m: Some(0.05),
            },
        )
        .unwrap();
    assert!(
        (arrival.sample_index - airborne_start as f64).abs() < 3.0,
        "{arrival:?}"
    );
}

#[test]
fn leading_edge_fraction_above_one_falls_back_to_the_peak() {
    let chirp = template();
    let mut signal = noise(12_000, 0.01, 5);
    add_copy(&mut signal, &chirp, 2_000, 1.0);
    let config = FirstArrivalConfig {
        timing: ArrivalTiming::LeadingEdge { fraction: 1.5 },
        ..FirstArrivalConfig::default()
    };
    let arrival = FirstArrivalDetector::new(&chirp, config, SAMPLE_RATE_HZ)
        .detect(&signal, 0..8_000, ArrivalContext::RemoteEmitter)
        .unwrap();
    assert!((arrival.sample_index - 2_000.0).abs() < 1.0, "{arrival:?}");
}

/// Two self-channel arrivals 24 samples apart (the Pixel 7a earpiece at
/// its bottom mic, 13–16 kHz): the strongest one is picked whatever d_ii
/// says, so d_ii no longer moves the pick.
#[test]
fn self_pick_is_the_strongest_peak_independent_of_d_ii() {
    let chirp = template();
    let separation = 24;
    for (early_gain, late_gain, expected_start) in [(1.0f32, 0.73f32, 3_000usize), (0.35, 1.0, 3_024)] {
        let mut signal = noise(12_000, 0.005, 7);
        add_copy(&mut signal, &chirp, 3_000, early_gain);
        add_copy(&mut signal, &chirp, 3_000 + separation, late_gain);
        for self_path_length_m in [Some(0.02), Some(0.20), None] {
            let arrival = detector()
                .detect(&signal, 0..8_000, ArrivalContext::SelfRecording { self_path_length_m })
                .unwrap();
            assert!(
                (arrival.sample_index - expected_start as f64).abs() < 3.0,
                "gains {early_gain}/{late_gain}, d_ii {self_path_length_m:?}: {arrival:?}"
            );
            assert_eq!(arrival.lead_before_strongest_samples, 0.0);
        }
    }
}

/// The pre-field policy skips a strongest early peak for a later one when
/// d_ii puts the early one inside the body-precursor window — the
/// 2026-11-26 r4 failure the default policy avoids.
#[test]
fn body_precursor_policy_skips_a_strongest_early_arrival() {
    let chirp = template();
    let mut signal = noise(12_000, 0.005, 8);
    add_copy(&mut signal, &chirp, 3_000, 1.0);
    add_copy(&mut signal, &chirp, 3_024, 0.73);
    let config = FirstArrivalConfig {
        self_recording_pick: SelfRecordingPick::EarliestOutsideBodyPrecursor,
        ..FirstArrivalConfig::default()
    };
    let arrival = FirstArrivalDetector::new(&chirp, config, SAMPLE_RATE_HZ)
        .detect(&signal, 0..8_000, ArrivalContext::SelfRecording { self_path_length_m: Some(0.20) })
        .unwrap();
    assert!(arrival.precursor_rejected);
    assert!(arrival.sample_index > 3_015.0, "{arrival:?}");
}

/// Envelope of one arrival at `peak` from the template's own
/// autocorrelation, with the leading first sidelobe scaled by `lift_db`.
fn envelope_with_lifted_leading_sidelobe(detector: &FirstArrivalDetector, peak: usize, lift_db: f64) -> Vec<f32> {
    let autocorrelation = &detector.autocorrelation_envelope;
    let main_lobe = detector.main_lobe_width_samples();
    let first_sidelobe_lag = main_lobe
        + (main_lobe..autocorrelation.len() - 1)
            .position(|lag| autocorrelation[lag + 1] < autocorrelation[lag])
            .unwrap();
    let mut envelope = vec![1e-4f32; peak * 2];
    for (lag, value) in autocorrelation.iter().enumerate().take(peak) {
        envelope[peak + lag] = envelope[peak + lag].max(*value);
        let lift = if (main_lobe..2 * first_sidelobe_lag - main_lobe).contains(&lag) {
            amplitude_ratio(lift_db) as f32
        } else {
            1.0
        };
        envelope[peak - lag] = envelope[peak - lag].max(*value * lift);
    }
    envelope
}

#[test]
fn remote_pick_ignores_a_transducer_lifted_first_sidelobe() {
    let detector = detector();
    let peak = 4_000;
    // +4.7 dB: the worst lift the 7a earpiece put on remote picks.
    let lifted = envelope_with_lifted_leading_sidelobe(&detector, peak, 4.7);
    let arrival = detector
        .detect_in_envelope(&lifted, 0..2 * peak, ArrivalContext::RemoteEmitter)
        .unwrap();
    assert_eq!(arrival.lead_before_strongest_samples, 0.0, "{arrival:?}");
    // The old 1.6× margin took the lifted sidelobe as an earlier arrival.
    let old_margin = FirstArrivalDetector::new(
        &template(),
        FirstArrivalConfig { sidelobe_rejection_margin: 1.6, ..FirstArrivalConfig::default() },
        SAMPLE_RATE_HZ,
    );
    let old = old_margin
        .detect_in_envelope(&lifted, 0..2 * peak, ArrivalContext::RemoteEmitter)
        .unwrap();
    assert!(old.lead_before_strongest_samples > 0.0, "{old:?}");
}
