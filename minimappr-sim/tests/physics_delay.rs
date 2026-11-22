//! Fractional propagation delay must match the analytic band-limited delay
//! to < 0.01 sample (and the waveform to better than -60 dB).

mod common;

use common::*;
use minimappr_sim::physics::geometry::Trajectory;
use minimappr_sim::render::render_world;
use minimappr_sim::scenario::devices::ClockConfig;
use minimappr_sim::world::plan_world;

const EMISSION_START_S: f64 = 0.1;
const SIGNAL_LENGTH_SAMPLES: usize = 6000;
const SIGNAL_MARGIN_SAMPLES: usize = 64;
const SIGNAL_BAND_EDGE_FRACTION_OF_FS: f64 = 0.3;
const MAX_DELAY_ERROR_SAMPLES: f64 = 0.01;
const MAX_RELATIVE_WAVEFORM_ERROR: f64 = 1e-3;

struct DelayCase {
    range_m: f64,
    clock_offset_s: f64,
}

fn check_delay_case(case: &DelayCase) -> (f64, f64) {
    let clock = ClockConfig::Fixed {
        offset_s: case.clock_offset_s,
        drift_ppm: 0.0,
    };
    let receiver_position = [case.range_m, 0.0, 1.5];
    let scenario = clean_scenario(1.0, vec![ideal_receiver("rx", receiver_position, clock)]);
    let mut plan = plan_world(&scenario).unwrap();
    let signal = band_limited_noise(
        SIGNAL_LENGTH_SAMPLES,
        SIGNAL_MARGIN_SAMPLES,
        SIGNAL_BAND_EDGE_FRACTION_OF_FS,
        11,
    );
    push_emission(
        &mut plan,
        signal.clone(),
        EMISSION_START_S,
        Trajectory::stationary([0.0, 0.0, 1.5]),
    );
    let rendered = render_world(&plan);
    let received: Vec<f64> = rendered[0].channels[0].iter().map(|s| *s as f64).collect();

    let expected_delay_samples = (EMISSION_START_S + case.range_m / plan.speed_of_sound_mps
        - case.clock_offset_s)
        * SAMPLE_RATE_HZ;
    let gain = 1.0 / case.range_m / plan.devices[0].full_scale_pa;
    let analytic: Vec<f64> = analytic_delay(&signal, expected_delay_samples, received.len())
        .iter()
        .map(|v| v * gain)
        .collect();

    let peak = analytic.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let worst = received
        .iter()
        .zip(&analytic)
        .fold(0.0f64, |m, (r, a)| m.max((r - a).abs()));

    let reference: Vec<f64> = signal.iter().map(|s| *s as f64).collect();
    let correlation = ContinuousCrossCorrelation::new(&reference, &received);
    let rounded = expected_delay_samples.round() as i64;
    let estimated = correlation.peak_lag(rounded - 3, rounded + 3);
    ((estimated - expected_delay_samples).abs(), worst / peak)
}

#[test]
fn fractional_delay_matches_analytic_band_limited_delay() {
    let cases = [
        DelayCase {
            range_m: 1.0,
            clock_offset_s: 0.0,
        },
        DelayCase {
            range_m: 3.333_33,
            clock_offset_s: 0.0,
        },
        DelayCase {
            range_m: 10.07,
            clock_offset_s: 0.0,
        },
        DelayCase {
            range_m: 37.1234,
            clock_offset_s: 0.0,
        },
        DelayCase {
            range_m: 123.456,
            clock_offset_s: 0.0,
        },
        DelayCase {
            range_m: 17.5,
            clock_offset_s: 0.012_345_6,
        },
        DelayCase {
            range_m: 17.5,
            clock_offset_s: -0.003_210_9,
        },
    ];
    for case in &cases {
        let (delay_error, waveform_error) = check_delay_case(case);
        eprintln!(
            "range {:>8.3} m offset {:+.7} s: delay error {:.2e} samples, waveform error {:.2e}",
            case.range_m, case.clock_offset_s, delay_error, waveform_error
        );
        assert!(
            delay_error < MAX_DELAY_ERROR_SAMPLES,
            "delay error {delay_error} samples"
        );
        assert!(
            waveform_error < MAX_RELATIVE_WAVEFORM_ERROR,
            "waveform error {waveform_error}"
        );
    }
}
