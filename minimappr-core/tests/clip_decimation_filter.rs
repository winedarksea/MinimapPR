//! FIR response and exact-timing tests for the 48 -> 24 kHz clip decimator.

mod common;

use minimappr_core::clip_codec::decimation_filter_design::{
    design_kaiser_windowed_sinc_lowpass, fir_magnitude_response, measure_lowpass_response,
    FIELD_CLIP_ANTI_ALIAS_LOWPASS_SPECIFICATION,
};
use minimappr_core::clip_codec::{DecimationError, DecimationFilterId, FieldClipDecimator};

const PASSBAND_RIPPLE_LIMIT_DB: f64 = 0.01;
const STOPBAND_ATTENUATION_REQUIRED_DB: f64 = 90.0;
const RESPONSE_GRID_POINTS_PER_BAND: usize = 4000;
const IMPULSE_AMPLITUDE: i16 = 20_000;
/// Device-clock origin far from zero so index arithmetic is exercised.
const ROLLING_BUFFER_FIRST_SAMPLE_INDEX: u64 = 1_000_000_007;
const ROLLING_BUFFER_LENGTH: usize = 48_000;

fn both_filters() -> [DecimationFilterId; 2] {
    [
        DecimationFilterId::BirdBandWithRumbleCut,
        DecimationFilterId::FlatToDirectCurrentForDroneAndGunshot,
    ]
}

fn decimate_centre_window(
    decimator: &FieldClipDecimator,
    rolling_buffer: &[i16],
    window_offset_in_buffer: usize,
    window_length: usize,
) -> Vec<i16> {
    decimator
        .decimate_window_from_rolling_buffer(
            rolling_buffer,
            ROLLING_BUFFER_FIRST_SAMPLE_INDEX,
            ROLLING_BUFFER_FIRST_SAMPLE_INDEX + window_offset_in_buffer as u64,
            window_length,
        )
        .expect("window inside buffer")
        .samples_output_rate
}

#[test]
fn anti_alias_lowpass_meets_passband_ripple_and_stopband_attenuation() {
    let taps = design_kaiser_windowed_sinc_lowpass(&FIELD_CLIP_ANTI_ALIAS_LOWPASS_SPECIFICATION);
    let measured = measure_lowpass_response(
        &taps,
        &FIELD_CLIP_ANTI_ALIAS_LOWPASS_SPECIFICATION,
        RESPONSE_GRID_POINTS_PER_BAND,
    );
    println!(
        "anti-alias FIR: {} taps, group delay {} input samples, passband ripple {:.5} dB p-p, \
         stopband attenuation {:.2} dB",
        taps.len(),
        (taps.len() - 1) / 2,
        measured.passband_ripple_peak_to_peak_db,
        measured.minimum_stopband_attenuation_db
    );
    assert!(
        measured.passband_ripple_peak_to_peak_db <= PASSBAND_RIPPLE_LIMIT_DB,
        "{measured:?}"
    );
    assert!(
        measured.minimum_stopband_attenuation_db >= STOPBAND_ATTENUATION_REQUIRED_DB,
        "{measured:?}"
    );
    let direct_current_gain = fir_magnitude_response(&taps, 0.0, 48_000.0);
    assert!((direct_current_gain - 1.0).abs() < 1e-12);
}

#[test]
fn anti_alias_taps_are_odd_and_symmetric_so_group_delay_is_an_integer() {
    let taps = design_kaiser_windowed_sinc_lowpass(&FIELD_CLIP_ANTI_ALIAS_LOWPASS_SPECIFICATION);
    assert_eq!(taps.len() % 2, 1);
    for (tap_index, tap) in taps.iter().enumerate() {
        assert_eq!(
            *tap,
            taps[taps.len() - 1 - tap_index],
            "tap {tap_index} asymmetric"
        );
    }
}

#[test]
fn even_index_impulse_peaks_at_exact_output_index_with_symmetric_response() {
    for filter in both_filters() {
        let decimator = FieldClipDecimator::for_filter(filter);
        let window_offset = 20_000;
        let impulse_output_index = 1_000;
        let mut rolling_buffer = vec![0_i16; ROLLING_BUFFER_LENGTH];
        rolling_buffer[window_offset + 2 * impulse_output_index] = IMPULSE_AMPLITUDE;
        let output = decimate_centre_window(&decimator, &rolling_buffer, window_offset, 4_000);
        let peak_index = output
            .iter()
            .enumerate()
            .max_by_key(|(_, value)| value.unsigned_abs())
            .map(|(index, _)| index)
            .expect("non-empty");
        assert_eq!(
            peak_index, impulse_output_index,
            "{filter:?}: group delay not removed"
        );
        let symmetry_tolerance = match filter {
            DecimationFilterId::FlatToDirectCurrentForDroneAndGunshot => 0,
            // Rumble-cut prefix sums round differently on each side.
            DecimationFilterId::BirdBandWithRumbleCut => 1,
        };
        for offset in 1..200 {
            let before = i32::from(output[impulse_output_index - offset]);
            let after = i32::from(output[impulse_output_index + offset]);
            assert!(
                (before - after).abs() <= symmetry_tolerance,
                "{filter:?} offset {offset}"
            );
        }
    }
}

#[test]
fn odd_index_impulse_lands_exactly_between_two_output_samples() {
    let decimator =
        FieldClipDecimator::for_filter(DecimationFilterId::FlatToDirectCurrentForDroneAndGunshot);
    let window_offset = 20_000;
    let output_index_before = 700;
    let mut rolling_buffer = vec![0_i16; ROLLING_BUFFER_LENGTH];
    rolling_buffer[window_offset + 2 * output_index_before + 1] = IMPULSE_AMPLITUDE;
    let output = decimate_centre_window(&decimator, &rolling_buffer, window_offset, 4_000);
    assert_eq!(output[output_index_before], output[output_index_before + 1]);
    for offset in 1..100 {
        assert_eq!(
            output[output_index_before - offset],
            output[output_index_before + 1 + offset]
        );
    }
}

#[test]
fn passband_sine_keeps_phase_relative_to_device_clock() {
    let tone_hz = 1_000.0;
    let amplitude = 10_000.0;
    let rolling_buffer: Vec<i16> = (0..ROLLING_BUFFER_LENGTH)
        .map(|offset| {
            let device_index = ROLLING_BUFFER_FIRST_SAMPLE_INDEX + offset as u64;
            let phase =
                2.0 * std::f64::consts::PI * tone_hz * (device_index % 48_000) as f64 / 48_000.0;
            (amplitude * phase.sin()).round() as i16
        })
        .collect();
    for filter in both_filters() {
        let decimator = FieldClipDecimator::for_filter(filter);
        let window_offset = 12_345;
        let window = decimator
            .decimate_window_from_rolling_buffer(
                &rolling_buffer,
                ROLLING_BUFFER_FIRST_SAMPLE_INDEX,
                ROLLING_BUFFER_FIRST_SAMPLE_INDEX + window_offset,
                24_000,
            )
            .expect("inside");
        assert!(!window.filter_context_was_zero_padded);
        let worst_error = window
            .samples_output_rate
            .iter()
            .enumerate()
            .map(|(output_index, &value)| {
                let device_index = window.first_sample_index_device_clock + 2 * output_index as u64;
                let phase = 2.0 * std::f64::consts::PI * tone_hz * (device_index % 48_000) as f64
                    / 48_000.0;
                (f64::from(value) - amplitude * phase.sin()).abs()
            })
            .fold(0.0, f64::max);
        assert!(
            worst_error <= 4.0,
            "{filter:?}: worst error {worst_error} LSB"
        );
    }
}

#[test]
fn chirp_band_tone_does_not_alias_into_the_clip() {
    let chirp_tone_hz = 13_000.0;
    let rolling_buffer: Vec<i16> = (0..ROLLING_BUFFER_LENGTH)
        .map(|offset| {
            let phase = 2.0 * std::f64::consts::PI * chirp_tone_hz * offset as f64 / 48_000.0;
            (30_000.0 * phase.sin()).round() as i16
        })
        .collect();
    let decimator = FieldClipDecimator::for_filter(DecimationFilterId::BirdBandWithRumbleCut);
    let output = decimate_centre_window(&decimator, &rolling_buffer, 10_000, 20_000);
    let peak_alias = output
        .iter()
        .map(|value| value.unsigned_abs())
        .max()
        .unwrap_or(0);
    // 30000 * 10^(-88/20) ~ 1.2 LSB.
    assert!(peak_alias <= 2, "13 kHz chirp aliased at {peak_alias} LSB");
}

#[test]
fn drone_filter_keeps_sub_100_hz_while_bird_filter_cuts_rumble_and_dc() {
    let direct_current_offset = 1_000_i16;
    let low_tone_hz = 40.0;
    let low_tone_amplitude = 8_000.0;
    let rolling_buffer: Vec<i16> = (0..ROLLING_BUFFER_LENGTH)
        .map(|offset| {
            let phase = 2.0 * std::f64::consts::PI * low_tone_hz * offset as f64 / 48_000.0;
            direct_current_offset + (low_tone_amplitude * phase.sin()).round() as i16
        })
        .collect();
    let mean_and_tone_amplitude = |output: &[i16]| {
        let mean = output.iter().map(|&value| f64::from(value)).sum::<f64>() / output.len() as f64;
        let tone_rms = (output
            .iter()
            .map(|&value| (f64::from(value) - mean).powi(2))
            .sum::<f64>()
            / output.len() as f64)
            .sqrt();
        (mean, tone_rms * std::f64::consts::SQRT_2)
    };
    // 0.5 s window = 20 whole cycles of 40 Hz.
    let drone =
        FieldClipDecimator::for_filter(DecimationFilterId::FlatToDirectCurrentForDroneAndGunshot);
    let (drone_mean, drone_tone) = mean_and_tone_amplitude(&decimate_centre_window(
        &drone,
        &rolling_buffer,
        12_000,
        24_000,
    ));
    assert!(
        (drone_mean - f64::from(direct_current_offset)).abs() < 1.0,
        "drone DC {drone_mean}"
    );
    assert!(
        (drone_tone / low_tone_amplitude - 1.0).abs() < 0.002,
        "drone 40 Hz {drone_tone}"
    );

    let bird = FieldClipDecimator::for_filter(DecimationFilterId::BirdBandWithRumbleCut);
    let (bird_mean, bird_tone) = mean_and_tone_amplitude(&decimate_centre_window(
        &bird,
        &rolling_buffer,
        12_000,
        24_000,
    ));
    assert!(
        bird_mean.abs() < 1.0,
        "bird mode should remove DC, mean {bird_mean}"
    );
    // Grouse-drumming range survives within 1.5 dB.
    assert!(
        bird_tone / low_tone_amplitude > 0.84,
        "bird 40 Hz {bird_tone}"
    );
}

#[test]
fn context_flag_and_errors() {
    let decimator = FieldClipDecimator::for_filter(DecimationFilterId::BirdBandWithRumbleCut);
    let rolling_buffer = vec![0_i16; ROLLING_BUFFER_LENGTH];
    let at_buffer_start = decimator
        .decimate_window_from_rolling_buffer(&rolling_buffer, 0, 0, 1_001)
        .expect("overlaps");
    assert!(at_buffer_start.filter_context_was_zero_padded);
    assert_eq!(
        at_buffer_start.samples_output_rate.len(),
        501,
        "odd input length rounds up"
    );
    let context = decimator.required_context_input_samples_each_side() as u64;
    let with_context = decimator
        .decimate_window_from_rolling_buffer(&rolling_buffer, 0, context, 1_000)
        .expect("inside");
    assert!(!with_context.filter_context_was_zero_padded);
    assert_eq!(
        decimator.decimate_window_from_rolling_buffer(&rolling_buffer, 0, 0, 0),
        Err(DecimationError::EmptyWindow)
    );
    assert_eq!(
        decimator.decimate_window_from_rolling_buffer(
            &rolling_buffer,
            0,
            ROLLING_BUFFER_LENGTH as u64,
            10
        ),
        Err(DecimationError::WindowOutsideRollingBuffer)
    );
}

#[test]
fn full_scale_input_reports_clipping_instead_of_wrapping() {
    let decimator =
        FieldClipDecimator::for_filter(DecimationFilterId::FlatToDirectCurrentForDroneAndGunshot);
    // Full-scale square wave at 6 kHz overshoots (Gibbs) after lowpassing.
    let rolling_buffer: Vec<i16> = (0..ROLLING_BUFFER_LENGTH)
        .map(|offset| {
            if (offset / 4) % 2 == 0 {
                i16::MAX
            } else {
                i16::MIN
            }
        })
        .collect();
    let window = decimator
        .decimate_window_from_rolling_buffer(&rolling_buffer, 0, 10_000, 4_800)
        .expect("inside");
    assert!(window.clipped_output_sample_count > 0);
}
