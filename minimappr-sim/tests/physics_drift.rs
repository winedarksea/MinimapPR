//! Clock drift: a receiver running at (1 + α) fs stretches the arrival
//! spacing of two pulses by exactly α, and each arrival lands at
//! (t_arrival − θ) · fs · (1 + α) in its own sample index.

mod common;

use common::*;
use minimappr_sim::physics::constants::PARTS_PER_MILLION;
use minimappr_sim::physics::geometry::Trajectory;
use minimappr_sim::render::render_world;
use minimappr_sim::scenario::devices::ClockConfig;
use minimappr_sim::world::plan_world;

const PULSE_HALF_WIDTH_SAMPLES: usize = 32;
const PULSE_LENGTH_SAMPLES: usize = 4 * PULSE_HALF_WIDTH_SAMPLES + 1;
const PULSE_CUTOFF_FRACTION_OF_FS: f64 = 0.3;
const FIRST_PULSE_S: f64 = 0.5;
const SECOND_PULSE_S: f64 = 9.5;
const RANGE_M: f64 = 12.0;
/// Each arrival is estimated to ~1e-4 samples, i.e. ~0.0003 ppm over 9 s.
const MAX_DRIFT_ERROR_PPM: f64 = 0.01;
const MAX_ARRIVAL_ERROR_SAMPLES: f64 = 0.01;

fn arrival_positions(drift_ppm: f64, offset_s: f64) -> (f64, f64, f64, f64) {
    let clock = ClockConfig::Fixed {
        offset_s,
        drift_ppm,
    };
    let scenario = clean_scenario(10.0, vec![ideal_receiver("rx", [RANGE_M, 0.0, 1.0], clock)]);
    let mut plan = plan_world(&scenario).unwrap();
    let pulse = band_limited_impulse(PULSE_HALF_WIDTH_SAMPLES, PULSE_CUTOFF_FRACTION_OF_FS);
    // The drifting clock stretches the pulse about its own peak, so predict
    // where the peak lands and compare the correlation lag of the peak.
    let peak_index = (PULSE_LENGTH_SAMPLES / 2) as f64;
    for start in [FIRST_PULSE_S, SECOND_PULSE_S] {
        push_emission(
            &mut plan,
            pulse.clone(),
            start,
            Trajectory::stationary([0.0, 0.0, 1.0]),
        );
    }
    let received: Vec<f64> = render_world(&plan)[0].channels[0]
        .iter()
        .map(|s| *s as f64)
        .collect();
    let reference: Vec<f64> = pulse.iter().map(|s| *s as f64).collect();
    let effective_rate = SAMPLE_RATE_HZ * (1.0 + drift_ppm / PARTS_PER_MILLION);
    let travel_s = RANGE_M / plan.speed_of_sound_mps;
    let expected = |start: f64| {
        (start + peak_index / SAMPLE_RATE_HZ + travel_s - offset_s) * effective_rate - peak_index
    };
    let measure = |expected_position: f64| {
        // Correlate a local window so the other pulse can't interfere.
        let window_start = (expected_position as usize).saturating_sub(2 * PULSE_LENGTH_SAMPLES);
        let window: Vec<f64> =
            received[window_start..window_start + 6 * PULSE_LENGTH_SAMPLES].to_vec();
        let correlation = ContinuousCrossCorrelation::new(&reference, &window);
        let guess = (expected_position - window_start as f64).round() as i64;
        window_start as f64 + correlation.peak_lag(guess - 4, guess + 4)
    };
    (
        measure(expected(FIRST_PULSE_S)),
        measure(expected(SECOND_PULSE_S)),
        expected(FIRST_PULSE_S),
        expected(SECOND_PULSE_S),
    )
}

#[test]
fn drift_stretch_matches_ppm() {
    for (drift_ppm, offset_s) in [(60.0, 0.0), (-60.0, 0.0), (23.7, 0.004_2), (0.0, 0.0)] {
        let (first, second, expected_first, expected_second) =
            arrival_positions(drift_ppm, offset_s);
        let measured_ppm = ((second - first) / ((SECOND_PULSE_S - FIRST_PULSE_S) * SAMPLE_RATE_HZ)
            - 1.0)
            * PARTS_PER_MILLION;
        eprintln!(
            "drift {drift_ppm:+.1} ppm offset {offset_s}: measured {measured_ppm:+.4} ppm, arrival errors {:.2e} / {:.2e} samples",
            (first - expected_first).abs(),
            (second - expected_second).abs()
        );
        assert!(
            (measured_ppm - drift_ppm).abs() < MAX_DRIFT_ERROR_PPM,
            "measured {measured_ppm} ppm vs {drift_ppm}"
        );
        assert!((first - expected_first).abs() < MAX_ARRIVAL_ERROR_SAMPLES);
        assert!((second - expected_second).abs() < MAX_ARRIVAL_ERROR_SAMPLES);
    }
}

/// In-clip stretch: a 60 ppm clock fed a 2 s tone sees its frequency scaled
/// by 1/(1 + α) in sample-index terms.
#[test]
fn drift_scales_apparent_tone_frequency() {
    let drift_ppm = 60.0;
    let tone_hz = 4000.0;
    let clock = ClockConfig::Fixed {
        offset_s: 0.0,
        drift_ppm,
    };
    let scenario = clean_scenario(3.0, vec![ideal_receiver("rx", [5.0, 0.0, 1.0], clock)]);
    let mut plan = plan_world(&scenario).unwrap();
    push_emission(
        &mut plan,
        tone(tone_hz, 2.5, SAMPLE_RATE_HZ),
        0.2,
        Trajectory::stationary([0.0, 0.0, 1.0]),
    );
    let rendered = render_world(&plan);
    let steady = &rendered[0].channels[0][24_000..120_000];
    let measured = dominant_frequency_hz(steady, tone_hz, 5.0, SAMPLE_RATE_HZ);
    let expected = tone_hz / (1.0 + drift_ppm / PARTS_PER_MILLION);
    eprintln!("apparent tone {measured:.4} Hz, expected {expected:.4} Hz");
    assert!((measured - expected).abs() < 0.01);
}
