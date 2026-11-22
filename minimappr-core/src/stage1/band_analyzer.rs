//! Log-spaced sub-band energies of one Hann-windowed analysis frame.
//!
//! Energies are normalized so each band value is that band's contribution to
//! the frame mean-square (full-scale sine -> 0.5, i.e. -3 dBFS), which keeps
//! the quantized history in dBFS terms.

use std::sync::Arc;

use num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use super::configuration::Stage1Configuration;

/// One band's FFT bin range `[first_bin, end_bin)` and its edges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubBandLayout {
    pub low_edge_hz: f64,
    pub high_edge_hz: f64,
    pub first_bin: usize,
    pub end_bin: usize,
}

impl SubBandLayout {
    pub fn centre_frequency_hz(&self) -> f64 {
        (self.low_edge_hz * self.high_edge_hz).sqrt()
    }
}

pub struct SubBandEnergyAnalyzer {
    fft: Arc<dyn Fft<f32>>,
    hann_window: Vec<f32>,
    band_layouts: Vec<SubBandLayout>,
    energy_normalizer: f64,
    fft_buffer: Vec<Complex<f32>>,
    fft_scratch: Vec<Complex<f32>>,
}

impl SubBandEnergyAnalyzer {
    pub fn new(configuration: &Stage1Configuration) -> Self {
        let window_length = configuration.analysis_window_samples;
        let fft = FftPlanner::<f32>::new().plan_fft_forward(window_length);
        let hann_window: Vec<f32> = (0..window_length)
            .map(|sample_index| {
                let phase = 2.0 * std::f64::consts::PI * sample_index as f64 / window_length as f64;
                (0.5 - 0.5 * phase.cos()) as f32
            })
            .collect();
        let window_energy: f64 = hann_window
            .iter()
            .map(|&weight| f64::from(weight * weight))
            .sum();
        let band_layouts = log_spaced_band_layouts(configuration);
        let fft_scratch = vec![Complex::default(); fft.get_inplace_scratch_len()];
        Self {
            fft,
            hann_window,
            band_layouts,
            // Parseval: positive-frequency band power * 2 / (N * sum w^2) is
            // the band's share of the frame mean-square.
            energy_normalizer: 2.0 / (window_length as f64 * window_energy),
            fft_buffer: vec![Complex::default(); window_length],
            fft_scratch,
        }
    }

    pub fn band_layouts(&self) -> &[SubBandLayout] {
        &self.band_layouts
    }

    /// `frame` must be exactly `analysis_window_samples` long.
    pub fn band_energies(&mut self, frame: &[f32], band_energies_out: &mut Vec<f64>) {
        for ((slot, &sample), &weight) in
            self.fft_buffer.iter_mut().zip(frame).zip(&self.hann_window)
        {
            *slot = Complex::new(sample * weight, 0.0);
        }
        self.fft
            .process_with_scratch(&mut self.fft_buffer, &mut self.fft_scratch);
        band_energies_out.clear();
        band_energies_out.extend(self.band_layouts.iter().map(|layout| {
            let band_power: f64 = self.fft_buffer[layout.first_bin..layout.end_bin]
                .iter()
                .map(|bin| f64::from(bin.norm_sqr()))
                .sum();
            band_power * self.energy_normalizer
        }));
    }
}

/// Mel-like band edges: log-spaced, but never narrower than
/// `minimum_band_width_bins` (linear spacing takes over at the low end).
/// A band's energy is a chi-square over its bins; one- or two-bin bands
/// flicker by several dB in stationary noise and would false-trigger.
fn log_spaced_band_layouts(configuration: &Stage1Configuration) -> Vec<SubBandLayout> {
    let bin_width_hz = f64::from(configuration.input_sample_rate_hz)
        / configuration.analysis_window_samples as f64;
    let highest_usable_bin = configuration.analysis_window_samples / 2;
    let edge_ratio =
        configuration.history_highest_band_edge_hz / configuration.history_lowest_band_edge_hz;
    let minimum_band_width_hz = configuration.minimum_band_width_bins as f64 * bin_width_hz;
    let band_edge_hz = |edge_index: usize| {
        let logarithmic_edge = configuration.history_lowest_band_edge_hz
            * edge_ratio.powf(edge_index as f64 / configuration.band_count as f64);
        let linear_edge =
            configuration.history_lowest_band_edge_hz + edge_index as f64 * minimum_band_width_hz;
        logarithmic_edge.max(linear_edge)
    };
    (0..configuration.band_count)
        .map(|band_index| {
            let low_edge_hz = band_edge_hz(band_index);
            let high_edge_hz = band_edge_hz(band_index + 1);
            let mut first_bin =
                ((low_edge_hz / bin_width_hz).ceil() as usize).min(highest_usable_bin);
            let mut end_bin =
                ((high_edge_hz / bin_width_hz).ceil() as usize).min(highest_usable_bin);
            if end_bin <= first_bin {
                let centre_bin =
                    ((low_edge_hz * high_edge_hz).sqrt() / bin_width_hz).round() as usize;
                first_bin = centre_bin.min(highest_usable_bin - 1);
                end_bin = first_bin + 1;
            }
            SubBandLayout {
                low_edge_hz,
                high_edge_hz,
                first_bin,
                end_bin,
            }
        })
        .collect()
}
