//! Linear-phase FIR design for the Field Mode 48 -> 24 kHz clip decimator.
//!
//! Kaiser-windowed sinc: closed-form, deterministic, and its symmetric taps
//! give exactly linear phase, so the group delay is an integer number of input
//! samples (odd tap count) that the decimator removes by index arithmetic.
//! Coefficients are recomputed on each device from the constants below; tiny
//! libm differences across platforms change taps by ~1 ulp, which is far below
//! the 16-bit output quantum, so the filter id alone identifies the response.

use std::f64::consts::PI;

/// Device capture rate every Field Mode phone records at.
pub const FIELD_DEVICE_CLOCK_SAMPLE_RATE_HZ: u32 = 48_000;
/// Clip decimation factor (48 kHz -> 24 kHz).
pub const FIELD_CLIP_DECIMATION_FACTOR: usize = 2;
/// Output rate of decimated clips.
pub const FIELD_CLIP_OUTPUT_SAMPLE_RATE_HZ: u32 =
    FIELD_DEVICE_CLOCK_SAMPLE_RATE_HZ / FIELD_CLIP_DECIMATION_FACTOR as u32;

/// Kaiser empirical constants (Kaiser 1974; Oppenheim & Schafer eq. 7.62/7.63).
const KAISER_HIGH_ATTENUATION_THRESHOLD_DB: f64 = 50.0;
const KAISER_HIGH_ATTENUATION_SLOPE: f64 = 0.1102;
const KAISER_HIGH_ATTENUATION_OFFSET_DB: f64 = 8.7;
const KAISER_MID_ATTENUATION_THRESHOLD_DB: f64 = 21.0;
const KAISER_MID_ATTENUATION_POWER_COEFFICIENT: f64 = 0.5842;
const KAISER_MID_ATTENUATION_POWER: f64 = 0.4;
const KAISER_MID_ATTENUATION_LINEAR_COEFFICIENT: f64 = 0.07886;
const KAISER_ORDER_ATTENUATION_OFFSET_DB: f64 = 7.95;
const KAISER_ORDER_TRANSITION_SCALE: f64 = 14.36;
/// Extra taps over the Kaiser order estimate; the estimate is approximate and
/// a few taps of margin keep the measured stopband at or past the spec.
const KAISER_ORDER_SAFETY_MARGIN_TAPS: usize = 8;
/// Power-series truncation for the modified Bessel function I0.
const BESSEL_I0_SERIES_RELATIVE_TOLERANCE: f64 = 1e-17;
const BESSEL_I0_SERIES_MAX_TERMS: usize = 500;

/// Specification of a linear-phase lowpass FIR.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LowpassFirSpecification {
    pub input_sample_rate_hz: f64,
    pub passband_edge_hz: f64,
    pub stopband_edge_hz: f64,
    pub stopband_attenuation_db: f64,
}

/// Field clip anti-alias lowpass: flat to 10.5 kHz, >= 90 dB down from
/// 12.5 kHz, so 13 kHz+ ranging chirps cannot alias into the 8-12 kHz upper
/// bird band after decimation. Passes DC (sub-100 Hz drone energy survives).
pub const FIELD_CLIP_ANTI_ALIAS_LOWPASS_SPECIFICATION: LowpassFirSpecification =
    LowpassFirSpecification {
        input_sample_rate_hz: FIELD_DEVICE_CLOCK_SAMPLE_RATE_HZ as f64,
        passband_edge_hz: 10_500.0,
        stopband_edge_hz: 12_500.0,
        stopband_attenuation_db: 90.0,
    };

/// Kaiser shape parameter for a target stopband attenuation.
pub fn kaiser_beta_for_attenuation_db(stopband_attenuation_db: f64) -> f64 {
    if stopband_attenuation_db > KAISER_HIGH_ATTENUATION_THRESHOLD_DB {
        KAISER_HIGH_ATTENUATION_SLOPE
            * (stopband_attenuation_db - KAISER_HIGH_ATTENUATION_OFFSET_DB)
    } else if stopband_attenuation_db >= KAISER_MID_ATTENUATION_THRESHOLD_DB {
        let excess = stopband_attenuation_db - KAISER_MID_ATTENUATION_THRESHOLD_DB;
        KAISER_MID_ATTENUATION_POWER_COEFFICIENT * excess.powf(KAISER_MID_ATTENUATION_POWER)
            + KAISER_MID_ATTENUATION_LINEAR_COEFFICIENT * excess
    } else {
        0.0
    }
}

/// Odd tap count (integer group delay) meeting the specification.
pub fn kaiser_odd_tap_count(specification: &LowpassFirSpecification) -> usize {
    let normalized_transition_width = (specification.stopband_edge_hz
        - specification.passband_edge_hz)
        / specification.input_sample_rate_hz;
    let estimated_order = (specification.stopband_attenuation_db
        - KAISER_ORDER_ATTENUATION_OFFSET_DB)
        / (KAISER_ORDER_TRANSITION_SCALE * normalized_transition_width);
    let tap_count = estimated_order.ceil() as usize + 1 + KAISER_ORDER_SAFETY_MARGIN_TAPS;
    if tap_count.is_multiple_of(2) {
        tap_count + 1
    } else {
        tap_count
    }
}

/// Zeroth-order modified Bessel function of the first kind (power series).
pub fn modified_bessel_first_kind_order_zero(argument: f64) -> f64 {
    let half_argument_squared = (argument / 2.0).powi(2);
    let mut series_sum = 1.0;
    let mut term = 1.0;
    for term_index in 1..BESSEL_I0_SERIES_MAX_TERMS {
        term *= half_argument_squared / (term_index as f64).powi(2);
        series_sum += term;
        if term < series_sum * BESSEL_I0_SERIES_RELATIVE_TOLERANCE {
            break;
        }
    }
    series_sum
}

/// Symmetric, odd-length, unity-DC-gain Kaiser-windowed sinc lowpass.
pub fn design_kaiser_windowed_sinc_lowpass(specification: &LowpassFirSpecification) -> Vec<f64> {
    let tap_count = kaiser_odd_tap_count(specification);
    let beta = kaiser_beta_for_attenuation_db(specification.stopband_attenuation_db);
    let cutoff_fraction_of_sample_rate = (specification.passband_edge_hz
        + specification.stopband_edge_hz)
        / 2.0
        / specification.input_sample_rate_hz;
    let center_index = (tap_count - 1) as f64 / 2.0;
    let window_normalizer = modified_bessel_first_kind_order_zero(beta);
    let mut taps: Vec<f64> = (0..tap_count)
        .map(|tap_index| {
            let offset_from_center = tap_index as f64 - center_index;
            let ideal_sinc = if offset_from_center == 0.0 {
                2.0 * cutoff_fraction_of_sample_rate
            } else {
                (2.0 * PI * cutoff_fraction_of_sample_rate * offset_from_center).sin()
                    / (PI * offset_from_center)
            };
            let normalized_position = offset_from_center / center_index;
            let window_value = modified_bessel_first_kind_order_zero(
                beta * (1.0 - normalized_position * normalized_position)
                    .max(0.0)
                    .sqrt(),
            ) / window_normalizer;
            ideal_sinc * window_value
        })
        .collect();
    let direct_current_gain: f64 = taps.iter().sum();
    for tap in &mut taps {
        *tap /= direct_current_gain;
    }
    // Enforce exact symmetry so phase is exactly linear regardless of rounding.
    for tap_index in 0..tap_count / 2 {
        let mirrored_index = tap_count - 1 - tap_index;
        let symmetric_value = 0.5 * (taps[tap_index] + taps[mirrored_index]);
        taps[tap_index] = symmetric_value;
        taps[mirrored_index] = symmetric_value;
    }
    taps
}

/// |H(f)| of an FIR at `frequency_hz`.
pub fn fir_magnitude_response(taps: &[f64], frequency_hz: f64, sample_rate_hz: f64) -> f64 {
    let angular_step = 2.0 * PI * frequency_hz / sample_rate_hz;
    let (real_sum, imaginary_sum) =
        taps.iter()
            .enumerate()
            .fold((0.0, 0.0), |(real, imaginary), (tap_index, tap)| {
                let phase = angular_step * tap_index as f64;
                (real + tap * phase.cos(), imaginary - tap * phase.sin())
            });
    (real_sum * real_sum + imaginary_sum * imaginary_sum).sqrt()
}

/// Measured passband ripple (peak-to-peak dB) and minimum stopband
/// attenuation (dB) on a dense frequency grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasuredLowpassResponse {
    pub passband_ripple_peak_to_peak_db: f64,
    pub minimum_stopband_attenuation_db: f64,
}

pub fn measure_lowpass_response(
    taps: &[f64],
    specification: &LowpassFirSpecification,
    grid_points_per_band: usize,
) -> MeasuredLowpassResponse {
    let magnitude_db = |frequency_hz: f64| {
        20.0 * fir_magnitude_response(taps, frequency_hz, specification.input_sample_rate_hz)
            .max(f64::MIN_POSITIVE)
            .log10()
    };
    let grid = |start_hz: f64, end_hz: f64| {
        (0..=grid_points_per_band).map(move |point_index| {
            start_hz + (end_hz - start_hz) * point_index as f64 / grid_points_per_band as f64
        })
    };
    let (passband_min_db, passband_max_db) = grid(0.0, specification.passband_edge_hz)
        .map(magnitude_db)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), value| {
            (low.min(value), high.max(value))
        });
    let stopband_max_db = grid(
        specification.stopband_edge_hz,
        specification.input_sample_rate_hz / 2.0,
    )
    .map(magnitude_db)
    .fold(f64::NEG_INFINITY, f64::max);
    MeasuredLowpassResponse {
        passband_ripple_peak_to_peak_db: passband_max_db - passband_min_db,
        minimum_stopband_attenuation_db: -stopband_max_db,
    }
}
