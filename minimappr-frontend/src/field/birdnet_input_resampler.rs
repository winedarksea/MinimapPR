//! Capture-rate -> 32 kHz resampler for BirdNET V3 input.
//!
//! Kaiser-windowed sinc with a precomputed phase table, so any capture rate
//! (48 kHz requested; Safari may deliver 44.1 kHz) works without per-sample
//! trig. Linear interpolation (what the metal-detecting PWA used) aliases
//! 16-24 kHz energy into the 8-16 kHz band BirdNET scores, so it is not used.

use super::field_configuration::{
    RESAMPLER_CUTOFF_FRACTION_OF_NYQUIST, RESAMPLER_HALF_WIDTH_TAPS, RESAMPLER_KAISER_BETA,
    RESAMPLER_PHASE_COUNT,
};

pub struct WindowedSincResampler {
    input_rate_hz: f64,
    output_rate_hz: f64,
    /// `phase_kernels[phase][tap]`, tap 0 at offset `-(half_width - 1)`.
    phase_kernels: Vec<Vec<f32>>,
}

impl WindowedSincResampler {
    pub fn new(input_rate_hz: u32, output_rate_hz: u32) -> Self {
        let input_rate_hz = f64::from(input_rate_hz);
        let output_rate_hz = f64::from(output_rate_hz);
        // Cutoff in cycles per input sample; below both Nyquists.
        let cutoff = 0.5 * (output_rate_hz / input_rate_hz).min(1.0) * RESAMPLER_CUTOFF_FRACTION_OF_NYQUIST;
        let half_width = RESAMPLER_HALF_WIDTH_TAPS as f64;
        let phase_kernels = (0..RESAMPLER_PHASE_COUNT)
            .map(|phase| {
                let fraction = phase as f64 / RESAMPLER_PHASE_COUNT as f64;
                let taps: Vec<f64> = (0..2 * RESAMPLER_HALF_WIDTH_TAPS)
                    .map(|tap| {
                        let offset = tap as f64 - (half_width - 1.0) - fraction;
                        let window = kaiser_window(offset / half_width, RESAMPLER_KAISER_BETA);
                        2.0 * cutoff * normalized_sinc(2.0 * cutoff * offset) * window
                    })
                    .collect();
                // Unity DC gain per phase, so level is preserved exactly.
                let gain: f64 = taps.iter().sum();
                taps.iter().map(|tap| (tap / gain) as f32).collect()
            })
            .collect();
        Self {
            input_rate_hz,
            output_rate_hz,
            phase_kernels,
        }
    }

    pub fn resample(&self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }
        let step = self.input_rate_hz / self.output_rate_hz;
        let output_length = ((input.len() as f64) / step).floor() as usize;
        let last_input_index = input.len() as i64 - 1;
        let first_tap_offset = RESAMPLER_HALF_WIDTH_TAPS as i64 - 1;
        (0..output_length)
            .map(|output_index| {
                let position = output_index as f64 * step;
                let base = position.floor();
                let phase = (((position - base) * RESAMPLER_PHASE_COUNT as f64).round() as usize)
                    .min(RESAMPLER_PHASE_COUNT - 1);
                let kernel = &self.phase_kernels[phase];
                let base_index = base as i64;
                kernel
                    .iter()
                    .enumerate()
                    .map(|(tap, weight)| {
                        // Edge samples are held, not zeroed: zero padding
                        // would put a step transient into the window edge.
                        let source = (base_index - first_tap_offset + tap as i64).clamp(0, last_input_index);
                        input[source as usize] * weight
                    })
                    .sum()
            })
            .collect()
    }
}

fn normalized_sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
    }
}

/// Kaiser window at normalized position `x` in [-1, 1].
fn kaiser_window(x: f64, beta: f64) -> f64 {
    if x.abs() > 1.0 {
        return 0.0;
    }
    modified_bessel_first_kind_order_zero(beta * (1.0 - x * x).sqrt())
        / modified_bessel_first_kind_order_zero(beta)
}

fn modified_bessel_first_kind_order_zero(x: f64) -> f64 {
    const SERIES_CONVERGENCE_TOLERANCE: f64 = 1e-12;
    const MAXIMUM_SERIES_TERMS: u32 = 64;
    let half_x = x / 2.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    for order in 1..MAXIMUM_SERIES_TERMS {
        term *= (half_x / f64::from(order)).powi(2);
        sum += term;
        if term < sum * SERIES_CONVERGENCE_TOLERANCE {
            break;
        }
    }
    sum
}
