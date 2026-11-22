//! FFT matched filter producing the analytic correlation envelope.
//!
//! The envelope (magnitude of the analytic cross-correlation) is used for
//! arrival picking instead of the raw correlation: the raw correlation of a
//! 13–20 kHz chirp oscillates at the carrier, so its tallest lobe can jump
//! a whole carrier cycle (50–77 µs) between neighbouring peaks under small
//! amounts of multipath or limiting. The envelope peak is unambiguous.

use num_complex::Complex32;
use rustfft::FftPlanner;

/// Correlation of `signal` against a fixed template.
#[derive(Clone, Debug)]
pub struct MatchedFilter {
    template: Vec<f32>,
    template_energy: f32,
}

/// Matched-filter output over lags `0..signal.len()`: index k means "the
/// template starts at signal sample k".
#[derive(Clone, Debug)]
pub struct CorrelationOutput {
    /// |analytic correlation| / template energy (≈ received amplitude).
    pub envelope: Vec<f32>,
    /// Real correlation / template energy.
    pub real: Vec<f32>,
}

impl MatchedFilter {
    pub fn new(template: &[f32]) -> Self {
        let template_energy = template
            .iter()
            .map(|s| s * s)
            .sum::<f32>()
            .max(f32::EPSILON);
        Self {
            template: template.to_vec(),
            template_energy,
        }
    }

    pub fn template_length(&self) -> usize {
        self.template.len()
    }

    /// Analytic correlation: X(f)·conj(H(f)) with negative frequencies
    /// zeroed and positive ones doubled, then one inverse FFT.
    pub fn correlate(&self, signal: &[f32]) -> CorrelationOutput {
        if signal.is_empty() || self.template.is_empty() {
            return CorrelationOutput {
                envelope: vec![],
                real: vec![],
            };
        }
        let fft_length = (signal.len() + self.template.len()).next_power_of_two();
        let mut planner = FftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(fft_length);
        let inverse = planner.plan_fft_inverse(fft_length);

        let mut signal_spectrum: Vec<Complex32> = signal
            .iter()
            .map(|s| Complex32::new(*s, 0.0))
            .chain(std::iter::repeat(Complex32::new(0.0, 0.0)))
            .take(fft_length)
            .collect();
        let mut template_spectrum: Vec<Complex32> = self
            .template
            .iter()
            .map(|s| Complex32::new(*s, 0.0))
            .chain(std::iter::repeat(Complex32::new(0.0, 0.0)))
            .take(fft_length)
            .collect();
        forward.process(&mut signal_spectrum);
        forward.process(&mut template_spectrum);

        let nyquist_bin = fft_length / 2;
        for (bin, (x, h)) in signal_spectrum
            .iter_mut()
            .zip(template_spectrum.iter())
            .enumerate()
        {
            let product = *x * h.conj();
            *x = if bin == 0 || bin == nyquist_bin {
                product
            } else if bin < nyquist_bin {
                product * 2.0
            } else {
                Complex32::new(0.0, 0.0)
            };
        }
        inverse.process(&mut signal_spectrum);
        let scale = 1.0 / (fft_length as f32 * self.template_energy);
        let envelope = signal_spectrum[..signal.len()]
            .iter()
            .map(|c| c.norm() * scale)
            .collect();
        let real = signal_spectrum[..signal.len()]
            .iter()
            .map(|c| c.re * scale)
            .collect();
        CorrelationOutput { envelope, real }
    }

    pub fn correlation_envelope(&self, signal: &[f32]) -> Vec<f32> {
        self.correlate(signal).envelope
    }
}

/// Sub-sample peak position by fitting a parabola through `values[index-1..=index+1]`.
pub fn parabolic_peak_offset(values: &[f32], index: usize) -> f64 {
    if index == 0 || index + 1 >= values.len() {
        return 0.0;
    }
    let (left, centre, right) = (
        values[index - 1] as f64,
        values[index] as f64,
        values[index + 1] as f64,
    );
    let denominator = left - 2.0 * centre + right;
    if denominator.abs() < f64::EPSILON {
        return 0.0;
    }
    (0.5 * (left - right) / denominator).clamp(-0.5, 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::chirp_waveform::{generate_plain_chirp, SweepLaw, CHIRP_BAND_15_TO_18_KHZ};

    const SAMPLE_RATE_HZ: f64 = 48_000.0;

    #[test]
    fn envelope_peaks_at_template_start_with_unit_height() {
        let template = generate_plain_chirp(
            CHIRP_BAND_15_TO_18_KHZ,
            0.05,
            SweepLaw::LinearFm,
            SAMPLE_RATE_HZ,
        );
        let start = 3_000;
        let mut signal = vec![0.0f32; 10_000];
        for (index, value) in template.iter().enumerate() {
            signal[start + index] += 0.5 * value;
        }
        let envelope = MatchedFilter::new(&template).correlation_envelope(&signal);
        let (peak_index, peak_value) = envelope.iter().enumerate().fold(
            (0, 0.0f32),
            |best, (i, v)| if *v > best.1 { (i, *v) } else { best },
        );
        assert_eq!(peak_index, start);
        assert!((peak_value - 0.5).abs() < 0.01, "peak {peak_value}");
    }

    #[test]
    fn parabolic_offset_recovers_vertex() {
        let vertex = 5.3;
        let values: Vec<f32> = (0..10)
            .map(|i| 10.0 - (i as f32 - vertex as f32).powi(2))
            .collect();
        let offset = parabolic_peak_offset(&values, 5);
        assert!((5.0 + offset - vertex).abs() < 1e-6);
    }
}
