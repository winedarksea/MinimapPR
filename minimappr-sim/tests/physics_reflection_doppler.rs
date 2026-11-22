//! Multipath timing (ground image source, point reflector) and Doppler
//! shift for moving sources/receivers.

mod common;

use common::*;
use minimappr_sim::physics::geometry::{distance, Trajectory};
use minimappr_sim::render::render_world;
use minimappr_sim::scenario::devices::ClockConfig;
use minimappr_sim::scenario::ReflectorConfig;
use minimappr_sim::world::plan_world;

const MAX_ARRIVAL_ERROR_SAMPLES: f64 = 0.02;
const MAX_RELATIVE_AMPLITUDE_ERROR: f64 = 0.01;

#[test]
fn ground_and_trunk_reflections_arrive_at_geometric_delays() {
    let source = [0.0, 0.0, 4.0];
    let receiver = [20.0, 0.0, 4.0];
    let trunk = [10.0, 6.0, 4.0];
    let (ground_coefficient, trunk_coefficient) = (0.6, 0.4);
    let clock = ClockConfig::Fixed {
        offset_s: 0.0,
        drift_ppm: 0.0,
    };
    let mut scenario = clean_scenario(0.5, vec![ideal_receiver("rx", receiver, clock)]);
    scenario.ground.enabled = true;
    scenario.ground.height_m = 0.0;
    scenario.ground.reflection_coefficient = ground_coefficient;
    scenario.reflectors.explicit.push(ReflectorConfig {
        reflector_id: None,
        position_m: trunk,
        reflection_coefficient: trunk_coefficient,
    });
    let mut plan = plan_world(&scenario).unwrap();
    let pulse = band_limited_impulse(32, 0.35);
    let emission_start_s = 0.05;
    push_emission(
        &mut plan,
        pulse.clone(),
        emission_start_s,
        Trajectory::stationary(source),
    );
    let received: Vec<f64> = render_world(&plan)[0].channels[0]
        .iter()
        .map(|s| *s as f64)
        .collect();
    let reference: Vec<f64> = pulse.iter().map(|s| *s as f64).collect();
    let correlation = ContinuousCrossCorrelation::new(&reference, &received);
    let autocorrelation_peak = ContinuousCrossCorrelation::new(&reference, &reference).at(0.0);

    let direct_length = distance(source, receiver);
    let image = [source[0], source[1], -source[2]];
    let paths = [
        ("direct", direct_length, 1.0),
        ("ground", distance(image, receiver), ground_coefficient),
        (
            "trunk",
            distance(source, trunk) + distance(trunk, receiver),
            trunk_coefficient,
        ),
    ];
    let full_scale = plan.devices[0].full_scale_pa;
    for (name, length, coefficient) in paths {
        let expected = (emission_start_s + length / plan.speed_of_sound_mps) * SAMPLE_RATE_HZ;
        let guess = expected.round() as i64;
        let measured = correlation.peak_lag(guess - 2, guess + 2);
        let amplitude = correlation.at(measured) / autocorrelation_peak * full_scale;
        let expected_amplitude = coefficient / length;
        eprintln!(
            "{name:>6}: arrival {measured:.4} vs {expected:.4} samples ({:+.2e}); amplitude {amplitude:.5} vs {expected_amplitude:.5}",
            measured - expected
        );
        assert!(
            (measured - expected).abs() < MAX_ARRIVAL_ERROR_SAMPLES,
            "{name} arrival"
        );
        assert!(
            (amplitude / expected_amplitude - 1.0).abs() < MAX_RELATIVE_AMPLITUDE_ERROR,
            "{name} amplitude"
        );
    }
}

const TONE_HZ: f64 = 2000.0;
const SPEED_MPS: f64 = 20.0;
const MAX_DOPPLER_ERROR_HZ: f64 = 0.05;

/// Source moving straight at a static receiver: f = f0 · c / (c − v).
#[test]
fn approaching_source_doppler_shift() {
    let clock = ClockConfig::Fixed {
        offset_s: 0.0,
        drift_ppm: 0.0,
    };
    let scenario = clean_scenario(3.5, vec![ideal_receiver("rx", [150.0, 0.0, 1.0], clock)]);
    let mut plan = plan_world(&scenario).unwrap();
    let source_path = Trajectory::from_rows(&[
        [0.0, -50.0, 0.0, 1.0],
        [5.0, -50.0 + SPEED_MPS * 5.0, 0.0, 1.0],
    ]);
    push_emission(
        &mut plan,
        tone(TONE_HZ, 2.0, SAMPLE_RATE_HZ),
        0.5,
        source_path,
    );
    let rendered = render_world(&plan);
    let c = plan.speed_of_sound_mps;
    // Steady segment well inside the received tone.
    let steady =
        &rendered[0].channels[0][(1.4 * SAMPLE_RATE_HZ) as usize..(2.2 * SAMPLE_RATE_HZ) as usize];
    let measured =
        dominant_frequency_hz(steady, TONE_HZ * c / (c - SPEED_MPS), 20.0, SAMPLE_RATE_HZ);
    let expected = TONE_HZ * c / (c - SPEED_MPS);
    eprintln!(
        "approaching source: {measured:.4} Hz, expected {expected:.4} Hz (shift {:.2} Hz)",
        expected - TONE_HZ
    );
    assert!((measured - expected).abs() < MAX_DOPPLER_ERROR_HZ);
}

/// Receiver moving away from a static source: f = f0 · (c − v) / c.
#[test]
fn receding_receiver_doppler_shift() {
    let clock = ClockConfig::Fixed {
        offset_s: 0.0,
        drift_ppm: 0.0,
    };
    let mut receiver = ideal_receiver("rx", [10.0, 0.0, 1.0], clock);
    receiver.trajectory = Some(vec![
        [0.0, 10.0, 0.0, 1.0],
        [4.0, 10.0 + SPEED_MPS * 4.0, 0.0, 1.0],
    ]);
    let scenario = clean_scenario(3.5, vec![receiver]);
    let mut plan = plan_world(&scenario).unwrap();
    push_emission(
        &mut plan,
        tone(TONE_HZ, 3.0, SAMPLE_RATE_HZ),
        0.0,
        Trajectory::stationary([0.0, 0.0, 1.0]),
    );
    let rendered = render_world(&plan);
    let c = plan.speed_of_sound_mps;
    let steady =
        &rendered[0].channels[0][(1.0 * SAMPLE_RATE_HZ) as usize..(2.0 * SAMPLE_RATE_HZ) as usize];
    let expected = TONE_HZ * (c - SPEED_MPS) / c;
    let measured = dominant_frequency_hz(steady, expected, 20.0, SAMPLE_RATE_HZ);
    eprintln!("receding receiver: {measured:.4} Hz, expected {expected:.4} Hz");
    assert!((measured - expected).abs() < MAX_DOPPLER_ERROR_HZ);
}
