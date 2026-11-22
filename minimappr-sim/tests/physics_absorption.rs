//! ISO 9613-1 atmospheric absorption: coefficient vs published table values
//! and end-to-end level after propagation (1/r spreading + absorption).

mod common;

use common::*;
use minimappr_sim::physics::atmosphere::AtmosphericConditions;
use minimappr_sim::physics::geometry::Trajectory;
use minimappr_sim::render::render_world;
use minimappr_sim::scenario::devices::ClockConfig;
use minimappr_sim::world::plan_world;

/// Published attenuation coefficients α (dB/km) at 101.325 kPa, computed by
/// ISO 9613-1 at exact octave midband frequencies (as tabulated in ISO
/// 9613-2:1996 Table 2), rounded to 0.1 dB/km. Columns: 63 Hz … 8 kHz
/// nominal; the standard evaluates at exact midbands 1000·10^(0.3·k) Hz.
const OCTAVE_INDICES_FROM_1_KHZ: [i32; 8] = [-4, -3, -2, -1, 0, 1, 2, 3];
const EXACT_MIDBAND_OCTAVE_EXPONENT_STEP: f64 = 0.3;
const MIDBAND_REFERENCE_HZ: f64 = 1000.0;
const ISO_TABLE_DB_PER_KM: [(f64, f64, [f64; 8]); 6] = [
    (10.0, 0.70, [0.1, 0.4, 1.0, 1.9, 3.7, 9.7, 32.8, 117.0]),
    (20.0, 0.70, [0.1, 0.3, 1.1, 2.8, 5.0, 9.0, 22.9, 76.6]),
    (30.0, 0.70, [0.1, 0.3, 1.0, 3.1, 7.4, 12.7, 23.1, 59.3]),
    (15.0, 0.20, [0.3, 0.6, 1.2, 2.7, 8.2, 28.2, 88.8, 202.0]),
    (15.0, 0.50, [0.1, 0.5, 1.2, 2.2, 4.2, 10.8, 36.2, 129.0]),
    (15.0, 0.80, [0.1, 0.3, 1.1, 2.4, 4.1, 8.3, 23.7, 82.8]),
];
/// Table values are rounded to 0.1 dB/km (larger ones to 3 significant figures).
const TABLE_ROUNDING_DB_PER_KM: f64 = 0.05;
const TABLE_RELATIVE_TOLERANCE: f64 = 0.005;
/// The octave table's 8 kHz entry is reported to the nearest 0.5–1 dB/km.
const LARGE_VALUE_ABSOLUTE_TOLERANCE_DB_PER_KM: f64 = 0.6;
const METERS_PER_KILOMETER: f64 = 1000.0;

#[test]
fn absorption_coefficient_matches_iso_9613_table() {
    for (temperature_c, humidity, expected_row) in ISO_TABLE_DB_PER_KM {
        let conditions = AtmosphericConditions {
            temperature_c,
            relative_humidity_fraction: humidity,
            ..Default::default()
        };
        for (octave, expected) in OCTAVE_INDICES_FROM_1_KHZ.iter().zip(expected_row) {
            let frequency = MIDBAND_REFERENCE_HZ
                * 10f64.powf(EXACT_MIDBAND_OCTAVE_EXPONENT_STEP * *octave as f64);
            let computed = conditions.absorption_db_per_m(frequency) * METERS_PER_KILOMETER;
            let tolerance = if expected >= 100.0 {
                LARGE_VALUE_ABSOLUTE_TOLERANCE_DB_PER_KM
            } else {
                TABLE_ROUNDING_DB_PER_KM + TABLE_RELATIVE_TOLERANCE * expected
            };
            eprintln!("{temperature_c} °C {:.0}% {frequency:>8.1} Hz: {computed:8.3} vs {expected:6.1} dB/km", humidity * 100.0);
            assert!(
                (computed - expected).abs() <= tolerance,
                "{temperature_c} °C {humidity} {frequency} Hz: {computed} vs {expected}"
            );
        }
    }
}

/// Render a tone at `range_m` and return its received amplitude (DTFT peak).
fn received_tone_amplitude(range_m: f64, absorption_enabled: bool, tone_hz: f64) -> (f64, f64) {
    let clock = ClockConfig::Fixed {
        offset_s: 0.0,
        drift_ppm: 0.0,
    };
    let mut scenario = clean_scenario(1.5, vec![ideal_receiver("rx", [range_m, 0.0, 1.0], clock)]);
    scenario.environment.absorption_enabled = absorption_enabled;
    let mut plan = plan_world(&scenario).unwrap();
    push_emission(
        &mut plan,
        tone(tone_hz, 1.0, SAMPLE_RATE_HZ),
        0.0,
        Trajectory::stationary([0.0, 0.0, 1.0]),
    );
    let rendered = render_world(&plan);
    let arrival = (range_m / plan.speed_of_sound_mps * SAMPLE_RATE_HZ) as usize;
    let steady = &rendered[0].channels[0][arrival + 4800..arrival + 43_200];
    let full_scale = plan.devices[0].full_scale_pa;
    (
        windowed_dtft_magnitude(steady, tone_hz, SAMPLE_RATE_HZ) * full_scale,
        plan.atmosphere.absorption_db_per_m(tone_hz),
    )
}

#[test]
fn rendered_level_follows_spreading_and_absorption() {
    let tone_hz = 8000.0;
    let (near, _) = received_tone_amplitude(10.0, false, tone_hz);
    let (far_dry, _) = received_tone_amplitude(100.0, false, tone_hz);
    let (far_wet, alpha_db_per_m) = received_tone_amplitude(100.0, true, tone_hz);
    let spreading_db = 20.0 * (near / far_dry).log10();
    let absorption_db = 20.0 * (far_dry / far_wet).log10();
    eprintln!("spreading 10→100 m: {spreading_db:.4} dB (expect 20); absorption over 100 m: {absorption_db:.4} dB (expect {:.4})", alpha_db_per_m * 100.0);
    assert!((spreading_db - 20.0).abs() < 0.01);
    assert!((absorption_db - alpha_db_per_m * 100.0).abs() < 0.02);
}
