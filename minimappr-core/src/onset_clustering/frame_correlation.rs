//! Delay-compensated, band-limited, soft-SNR-weighted GCC-PHAT for onset
//! frames (port of `minimappr/core/onset_frame_correlation.py`).
//!
//! The reference sensor's short template `[t0, t0 + L]` is correlated against
//! a search window `[t0 - tau_max, t0 + L + tau_max]` on the other sensor, so a
//! 50 ms note lands inside the other window however far apart the two are.
//! PHAT whitening is limited to the onset's band and each bin carries the
//! Knapp-Carter ML weight from the two sensors' own noise floors. Narrowband
//! frames correlate the envelope (fine structure cycle-skips under multipath).
//!
//! Why not `srp_layout::pair_correlation`: that path is f32 and whole-clip;
//! parity with the Python reference (1e-6 on curves feeding argmax decisions)
//! needs the f64 template-vs-search form below.

use num_complex::Complex64;

use super::numeric::{
    interp, interp_clamped, linspace, numpy_hanning, percentile, power_of_two_at_least,
    rfft_frequencies, SpectralTransforms,
};

pub const NOISE_FLOOR_PERCENTILE: f64 = 20.0;
pub const POWER_FLOOR: f64 = 1e-20;
/// Fraction of a segment tapered at each end before correlating (Tukey window).
pub const EDGE_TAPER_FRACTION: f64 = 0.1;
/// Segments shorter than this are not used for a noise spectrum.
const MINIMUM_NOISE_SEGMENT_SAMPLES: usize = 16;
/// Correlation needs at least this many template samples.
const MINIMUM_TEMPLATE_SAMPLES: usize = 8;

/// A single periodogram bin is exponentially distributed, so its p-th
/// percentile is `-ln(1 - p)` x the mean; dividing turns it back into the mean.
pub fn noise_percentile_to_mean() -> f64 {
    -(1.0 - NOISE_FLOOR_PERCENTILE / 100.0).ln()
}

#[derive(Debug, Clone, PartialEq)]
pub struct SensorNoiseSpectrum {
    pub frequencies_hz: Vec<f64>,
    pub power_density: Vec<f64>,
}

impl SensorNoiseSpectrum {
    pub fn at(&self, frequency_hz: f64) -> f64 {
        interp_clamped(frequency_hz, &self.frequencies_hz, &self.power_density)
    }
}

/// Correlation of `sensor_b` against `sensor_a`'s template; positive lag = b later.
#[derive(Debug, Clone, PartialEq)]
pub struct PairCorrelationCurve {
    pub sensor_a: String,
    pub sensor_b: String,
    pub lags_s: Vec<f64>,
    pub values: Vec<f64>,
    pub effective_bandwidth_hz: f64,
    pub used_envelope: bool,
    pub weight_sum: f64,
    /// Fine structure, kept when SRP uses the envelope: LM votes on the fine
    /// peak nearest the seed once SRP has picked the cycle.
    pub fine_values: Vec<f64>,
    pub centre_frequency_hz: f64,
}

impl PairCorrelationCurve {
    pub fn value_at(&self, lag_s: f64) -> f64 {
        interp(lag_s, &self.lags_s, &self.values, 0.0, 0.0)
    }

    /// Lag and value of the SRP curve maximum within `+-half_width_s` of `lag_s`.
    pub fn peak_lag_near(&self, lag_s: f64, half_width_s: f64) -> (f64, f64) {
        self.peak_near(&self.values, lag_s, half_width_s)
    }

    /// Fine-structure peak; for envelope curves the window is capped at half a
    /// carrier period so the vote cannot jump to an adjacent cycle.
    pub fn fine_peak_lag_near(&self, lag_s: f64, half_width_s: f64) -> (f64, f64) {
        if !self.used_envelope {
            return self.peak_lag_near(lag_s, half_width_s);
        }
        let mut half_width_s = half_width_s;
        if self.centre_frequency_hz > 0.0 {
            half_width_s = half_width_s.min(0.5 / self.centre_frequency_hz);
        }
        self.peak_near(&self.fine_values, lag_s, half_width_s)
    }

    fn peak_near(&self, curve: &[f64], lag_s: f64, half_width_s: f64) -> (f64, f64) {
        let mut best: Option<usize> = None;
        for (index, lag) in self.lags_s.iter().enumerate() {
            if (lag - lag_s).abs() <= half_width_s
                && best.is_none_or(|current| curve[index] > curve[current])
            {
                best = Some(index);
            }
        }
        match best {
            Some(index) => (self.lags_s[index], curve[index]),
            None => (lag_s, 0.0),
        }
    }
}

fn hann_density(
    segment: &[f64],
    fft_length: usize,
    transforms: &mut SpectralTransforms,
) -> Vec<f64> {
    let window = numpy_hanning(segment.len());
    let windowed: Vec<f64> = segment
        .iter()
        .zip(&window)
        .map(|(value, weight)| value * weight)
        .collect();
    let window_energy = window
        .iter()
        .map(|weight| weight * weight)
        .sum::<f64>()
        .max(POWER_FLOOR);
    transforms
        .rfft(&windowed, fft_length)
        .iter()
        .map(|bin| bin.norm().powi(2) / window_energy)
        .collect()
}

/// Low-percentile power density over non-overlapping Hann segments.
pub fn estimate_noise_spectrum(
    signal: &[f64],
    sample_rate_hz: u32,
    segment_samples: usize,
    transforms: &mut SpectralTransforms,
) -> SensorNoiseSpectrum {
    let segment_samples = MINIMUM_NOISE_SEGMENT_SAMPLES.max(segment_samples.min(signal.len()));
    let count = (signal.len() / segment_samples).max(1);
    let densities: Vec<Vec<f64>> = (0..count)
        .map(|index| {
            let stop = ((index + 1) * segment_samples).min(signal.len());
            hann_density(
                &signal[index * segment_samples..stop],
                segment_samples,
                transforms,
            )
        })
        .collect();
    let bins = densities[0].len();
    let scale = noise_percentile_to_mean();
    let power_density = (0..bins)
        .map(|bin| {
            percentile(
                &densities.iter().map(|row| row[bin]).collect::<Vec<_>>(),
                NOISE_FLOOR_PERCENTILE,
            ) / scale
                + POWER_FLOOR
        })
        .collect();
    SensorNoiseSpectrum {
        frequencies_hz: rfft_frequencies(segment_samples, sample_rate_hz as f64),
        power_density,
    }
}

fn tukey(length: usize) -> Vec<f64> {
    let taper = ((length as f64 * EDGE_TAPER_FRACTION) as usize).max(1);
    let mut window = vec![1.0; length];
    let ramp: Vec<f64> = linspace(0.0, std::f64::consts::PI, taper)
        .iter()
        .map(|angle| 0.5 - 0.5 * angle.cos())
        .collect();
    window[..taper].copy_from_slice(&ramp);
    for (offset, value) in ramp.iter().rev().enumerate() {
        window[length - taper + offset] = *value;
    }
    window
}

/// q / (1 + q): how much of each bin is call rather than noise.
fn presence_weight(joint_snr: f64) -> f64 {
    joint_snr / (1.0 + joint_snr)
}

/// q^2 / (1 + 2q): the Knapp-Carter ML GCC weight for two channels at SNR q.
fn maximum_likelihood_weight(joint_snr: f64) -> f64 {
    joint_snr * joint_snr / (1.0 + 2.0 * joint_snr)
}

pub fn rms_bandwidth_hz(frequencies_hz: &[f64], weights: &[f64]) -> f64 {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mean = weights
        .iter()
        .zip(frequencies_hz)
        .map(|(w, f)| w * f)
        .sum::<f64>()
        / total;
    2.0 * (weights
        .iter()
        .zip(frequencies_hz)
        .map(|(w, f)| w * (f - mean).powi(2))
        .sum::<f64>()
        / total)
        .sqrt()
}

/// One template-vs-search correlation request.
pub struct TemplateSearchPair<'signal> {
    pub sensor_a: &'signal str,
    pub sensor_b: &'signal str,
    pub template: &'signal [f64],
    /// Starts `max_lag_samples` before the template (length L + 2D).
    pub search: &'signal [f64],
    pub max_lag_samples: usize,
    pub template_noise: &'signal SensorNoiseSpectrum,
    pub search_noise: &'signal SensorNoiseSpectrum,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorrelationSettings {
    pub sample_rate_hz: u32,
    pub band_min_hz: f64,
    pub band_max_hz: f64,
    pub interpolation_factor: usize,
    pub narrowband_bandwidth_hz: f64,
    /// False keeps fine structure (compact pairs: lag span under a carrier period).
    pub allow_envelope: bool,
}

pub fn correlate_template_against_search(
    pair: &TemplateSearchPair<'_>,
    settings: &CorrelationSettings,
    transforms: &mut SpectralTransforms,
) -> Option<PairCorrelationCurve> {
    let template = pair.template;
    let search = pair.search;
    if template.len() < MINIMUM_TEMPLATE_SAMPLES
        || search.len() < template.len() + 2 * pair.max_lag_samples
    {
        return None;
    }
    let fft_length = power_of_two_at_least(search.len() + template.len());
    let frequencies = rfft_frequencies(fft_length, settings.sample_rate_hz as f64);
    let template_density = hann_density(template, fft_length, transforms);
    let search_density = hann_density(search, fft_length, transforms);
    let joint_snr: Vec<f64> = frequencies
        .iter()
        .enumerate()
        .map(|(bin, frequency)| {
            let template_snr =
                (template_density[bin] / pair.template_noise.at(*frequency) - 1.0).max(0.0);
            let search_snr =
                (search_density[bin] / pair.search_noise.at(*frequency) - 1.0).max(0.0);
            let in_band = *frequency >= settings.band_min_hz && *frequency <= settings.band_max_hz;
            if in_band {
                template_snr.min(search_snr)
            } else {
                0.0
            }
        })
        .collect();
    let weights: Vec<f64> = joint_snr
        .iter()
        .map(|q| maximum_likelihood_weight(*q))
        .collect();
    let weight_sum: f64 = weights.iter().sum();
    if weight_sum <= 0.0 {
        return None;
    }
    let tapered = |segment: &[f64]| -> Vec<f64> {
        segment
            .iter()
            .zip(tukey(segment.len()))
            .map(|(v, w)| v * w)
            .collect()
    };
    let template_spectrum = transforms.rfft(&tapered(template), fft_length);
    let search_spectrum = transforms.rfft(&tapered(search), fft_length);
    let whitened: Vec<Complex64> = (0..frequencies.len())
        .map(|bin| {
            let cross = search_spectrum[bin] * template_spectrum[bin].conj();
            cross * weights[bin] / cross.norm().max(POWER_FLOOR)
        })
        .collect();
    let presence: Vec<f64> = joint_snr.iter().map(|q| presence_weight(*q)).collect();
    let effective_bandwidth_hz = rms_bandwidth_hz(&frequencies, &presence);
    let use_envelope =
        settings.allow_envelope && effective_bandwidth_hz < settings.narrowband_bandwidth_hz;
    let interpolation = settings.interpolation_factor.max(1);
    let interpolated_length = fft_length * interpolation;
    // irfft normalisation: a perfectly coherent frame peaks at sum(weights) * 2 / n_fft.
    let normalisation = interpolated_length as f64 / (2.0 * weight_sum);
    let last_index = 2 * pair.max_lag_samples * interpolation;
    let fine = transforms.irfft(&whitened, interpolated_length);
    let fine_values: Vec<f64> = fine[..=last_index]
        .iter()
        .map(|value| value * normalisation)
        .collect();
    let values = if use_envelope {
        // Analytic correlation: one-sided spectrum -> complex IFFT; |.| is the envelope.
        let mut full = vec![Complex64::new(0.0, 0.0); interpolated_length];
        full[..whitened.len()].copy_from_slice(&whitened);
        for bin in full.iter_mut().take(whitened.len()).skip(1) {
            *bin *= 2.0;
        }
        let magnitudes = transforms.ifft_magnitudes(full);
        magnitudes[..=last_index]
            .iter()
            .map(|value| value * normalisation)
            .collect()
    } else {
        fine_values.clone()
    };
    let sample_rate = settings.sample_rate_hz as f64;
    let lags_s = (0..values.len())
        .map(|index| {
            (index as f64 / interpolation as f64 - pair.max_lag_samples as f64) / sample_rate
        })
        .collect();
    let presence_total: f64 = presence.iter().sum();
    let centre_frequency_hz = presence
        .iter()
        .zip(&frequencies)
        .map(|(w, f)| w * f)
        .sum::<f64>()
        / presence_total.max(POWER_FLOOR);
    Some(PairCorrelationCurve {
        sensor_a: pair.sensor_a.to_string(),
        sensor_b: pair.sensor_b.to_string(),
        lags_s,
        values,
        effective_bandwidth_hz,
        used_envelope: use_envelope,
        weight_sum,
        fine_values,
        centre_frequency_hz,
    })
}
