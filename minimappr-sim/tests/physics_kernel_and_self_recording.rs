//! Interpolation kernel flatness up to the chirp band, and phone
//! self-recording (d_ii airborne path, body-conduction precursor, limiter).

mod common;

use common::*;
use minimappr_sim::physics::constants::SOUND_SPEED_IN_PHONE_BODY_MPS;
use minimappr_sim::physics::geometry::Trajectory;
use minimappr_sim::render::render_world;
use minimappr_sim::scenario::devices::{ClockConfig, Grip, LimiterConfig};
use minimappr_sim::scenario::ChirpScheduleConfig;
use minimappr_sim::signals::chirps::ChirpWaveformParams;
use minimappr_sim::world::plan_world;

/// 18–20 kHz chirps must not be coloured by the interpolator, whatever the
/// fractional delay.
const CHIRP_BAND_TOP_HZ: f64 = 20_000.0;
const MAX_KERNEL_RIPPLE_DB: f64 = 0.1;

#[test]
fn interpolation_is_flat_through_the_chirp_band() {
    for frequency_hz in [1000.0, 8000.0, 16_000.0, 18_000.0, CHIRP_BAND_TOP_HZ] {
        for range_m in [5.0, 5.001_7, 5.003_6, 5.005_1] {
            let clock = ClockConfig::Fixed {
                offset_s: 0.0,
                drift_ppm: 0.0,
            };
            let scenario =
                clean_scenario(0.5, vec![ideal_receiver("rx", [range_m, 0.0, 1.0], clock)]);
            let mut plan = plan_world(&scenario).unwrap();
            push_emission(
                &mut plan,
                tone(frequency_hz, 0.4, SAMPLE_RATE_HZ),
                0.0,
                Trajectory::stationary([0.0, 0.0, 1.0]),
            );
            let rendered = render_world(&plan);
            let steady = &rendered[0].channels[0][4800..16_800];
            let reference = tone(frequency_hz, 0.25, SAMPLE_RATE_HZ);
            let ratio = windowed_dtft_magnitude(steady, frequency_hz, SAMPLE_RATE_HZ)
                / windowed_dtft_magnitude(&reference[..steady.len()], frequency_hz, SAMPLE_RATE_HZ)
                * plan.devices[0].full_scale_pa
                * range_m;
            let ripple_db = 20.0 * ratio.log10();
            assert!(
                ripple_db.abs() < MAX_KERNEL_RIPPLE_DB,
                "{frequency_hz} Hz at {range_m} m: {ripple_db:.3} dB"
            );
        }
    }
}

#[test]
fn self_chirp_has_precursor_and_is_limited() {
    let speaker_distance_m = 0.12;
    let mut phone = ideal_receiver(
        "phone",
        [0.0, 0.0, 1.0],
        ClockConfig::Fixed {
            offset_s: 0.0,
            drift_ppm: 0.0,
        },
    );
    phone.speaker.speaker_to_mic_distance_m = Some(speaker_distance_m);
    phone.speaker.grip = Grip::HardTable;
    phone.limiter = Some(LimiterConfig {
        enabled: true,
        knee_full_scale: 0.5,
        ceiling_full_scale: 0.9,
    });
    let mut scenario = clean_scenario(0.5, vec![phone]);
    let chirp_start_s = 0.1;
    scenario.chirps = Some(ChirpScheduleConfig {
        round_start_times_s: vec![chirp_start_s],
        slot_duration_s: 0.25,
        waveform: ChirpWaveformParams {
            band_low_hz: 15_000.0,
            band_high_hz: 18_000.0,
            duration_s: 0.04,
            ..Default::default()
        },
        // Loud enough at 12 cm to drive the limiter.
        level_db_spl_at_1m: 110.0,
    });
    let plan = plan_world(&scenario).unwrap();
    let samples = &render_world(&plan)[0].channels[0];
    let c = plan.speed_of_sound_mps;
    let airborne_arrival = ((chirp_start_s + speaker_distance_m / c) * SAMPLE_RATE_HZ) as usize;
    let structural_arrival = ((chirp_start_s + speaker_distance_m / SOUND_SPEED_IN_PHONE_BODY_MPS)
        * SAMPLE_RATE_HZ) as usize;
    let peak_in =
        |range: std::ops::Range<usize>| samples[range].iter().fold(0f32, |m, s| m.max(s.abs()));
    let precursor_peak = peak_in(structural_arrival..airborne_arrival - 2);
    let silence_peak = peak_in(0..structural_arrival - 40);
    let overall_peak = peak_in(0..samples.len());
    eprintln!("precursor {precursor_peak:.4}, pre-silence {silence_peak:.2e}, overall peak {overall_peak:.4}");
    assert!(
        silence_peak < 1e-6,
        "nothing should arrive before the structural path"
    );
    assert!(precursor_peak > 1e-3, "body-conduction precursor missing");
    assert!(
        overall_peak <= 0.9 + 1e-6 && overall_peak > 0.5,
        "limiter should engage and cap at the ceiling"
    );
}
