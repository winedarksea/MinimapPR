//! Note onsets from a sub-band spectral-flux history (port of
//! `minimappr/core/onset_detection.py`).
//!
//! A short onset frame, not a whole clip, is the unit of correlation: the
//! direct path leads reflections at a note's onset and two birds rarely start
//! a note at the same instant in the same band. Everything is relative to the
//! window's own per-band noise floor, so a faint far bird yields onsets exactly
//! like a loud near one (no absolute level gate).

use super::numeric::{
    geomspace, mean, median, numpy_hanning, percentile, population_std, python_round_to_i64,
    rfft_frequencies, SpectralTransforms,
};

/// Percentile of each sub-band's energy over the window taken as its noise floor.
pub const NOISE_FLOOR_PERCENTILE: f64 = 20.0;
/// Scale from MAD to a Gaussian standard deviation.
pub const MAD_TO_STD: f64 = 1.4826;
pub const POWER_FLOOR: f64 = 1e-20;
/// Hops before an onset that define the "before" level for the onset's band.
pub const ONSET_BASELINE_HOPS: usize = 5;
/// Frames are long enough that the narrowest sub-band spans this many FFT bins
/// (single-bin bands flicker by several dB on pure noise).
pub const MIN_BINS_PER_SUBBAND: f64 = 2.0;
/// A band's hop-to-hop rise counts towards flux only beyond this many of its
/// own noise-rise standard deviations.
pub const FLUX_RISE_NOISE_MULTIPLIER: f64 = 2.0;
/// Upper band edge is held below Nyquist by this fraction.
const NYQUIST_USABLE_FRACTION: f64 = 0.98;
/// Lower band edge is at most this fraction of the upper one.
const LOWER_EDGE_MAX_FRACTION_OF_UPPER: f64 = 0.5;

/// Per-hop log energy (dB) in log-spaced sub-bands plus each band's noise floor.
#[derive(Debug, Clone, PartialEq)]
pub struct SubbandHistory {
    /// `energy_db[hop][band]`.
    pub energy_db: Vec<Vec<f64>>,
    pub noise_floor_db: Vec<f64>,
    pub band_edges_hz: Vec<f64>,
    pub hop_samples: usize,
    pub frame_samples: usize,
}

impl SubbandHistory {
    pub fn hop_count(&self) -> usize {
        self.energy_db.len()
    }

    pub fn band_count(&self) -> usize {
        self.noise_floor_db.len()
    }

    pub fn band_centers_hz(&self) -> Vec<f64> {
        self.band_edges_hz
            .windows(2)
            .map(|edge| (edge[0] * edge[1]).sqrt())
            .collect()
    }

    pub fn snr_db(&self, hop: usize, band: usize) -> f64 {
        self.energy_db[hop][band] - self.noise_floor_db[band]
    }

    pub fn hop_start_sample(&self, hop: usize) -> usize {
        hop * self.hop_samples
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectedOnset {
    pub sample_index: usize,
    /// Flux in noise-normalised units (MADs above the median).
    pub strength: f64,
    pub band_min_hz: f64,
    pub band_max_hz: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubbandHistorySettings {
    pub sample_rate_hz: u32,
    pub hop_seconds: f64,
    pub band_min_hz: f64,
    pub band_max_hz: f64,
    pub band_count: usize,
}

/// Index of the sub-band holding `frequency` (`np.digitize(f, edges) - 1`); None outside.
fn band_index_of(frequency_hz: f64, edges: &[f64]) -> Option<usize> {
    let position = edges.partition_point(|edge| *edge <= frequency_hz);
    (position >= 1 && position < edges.len()).then(|| position - 1)
}

/// Short-time sub-band energies over Hann frames of at least two hops.
pub fn build_subband_history(
    signal: &[f64],
    settings: &SubbandHistorySettings,
    transforms: &mut SpectralTransforms,
) -> SubbandHistory {
    let sample_rate = settings.sample_rate_hz as f64;
    let hop_samples = python_round_to_i64(settings.hop_seconds * sample_rate).max(1) as usize;
    let nyquist_hz = 0.5 * sample_rate;
    let upper_hz = settings
        .band_max_hz
        .min(nyquist_hz * NYQUIST_USABLE_FRACTION);
    let lower_hz = settings
        .band_min_hz
        .min(upper_hz * LOWER_EDGE_MAX_FRACTION_OF_UPPER);
    let edges = geomspace(lower_hz, upper_hz, settings.band_count + 1);
    let narrowest_band_hz = edges
        .windows(2)
        .map(|edge| edge[1] - edge[0])
        .fold(f64::INFINITY, f64::min);
    let frame_samples = (2 * hop_samples)
        .max((MIN_BINS_PER_SUBBAND * sample_rate / narrowest_band_hz).ceil() as usize);
    let hop_count = if signal.len() >= frame_samples {
        1 + (signal.len() - frame_samples) / hop_samples
    } else {
        0
    };
    if hop_count == 0 {
        return SubbandHistory {
            energy_db: Vec::new(),
            noise_floor_db: vec![0.0; settings.band_count],
            band_edges_hz: edges,
            hop_samples,
            frame_samples,
        };
    }
    let window = numpy_hanning(frame_samples);
    let frequencies = rfft_frequencies(frame_samples, sample_rate);
    let bin_bands: Vec<Option<usize>> = frequencies
        .iter()
        .map(|frequency| band_index_of(*frequency, &edges))
        .collect();
    let mut nearest_bin_for_empty_band = vec![None; settings.band_count];
    for (band, slot) in nearest_bin_for_empty_band.iter_mut().enumerate() {
        if !bin_bands.contains(&Some(band)) {
            // Band narrower than a bin: take the nearest bin so it is not empty.
            let centre = (edges[band] * edges[band + 1]).sqrt();
            let distances: Vec<f64> = frequencies
                .iter()
                .map(|frequency| (frequency - centre).abs())
                .collect();
            *slot = Some(super::numeric::argmax(
                &distances.iter().map(|d| -d).collect::<Vec<_>>(),
            ));
        }
    }
    let mut energy_db = Vec::with_capacity(hop_count);
    let mut frame = vec![0.0; frame_samples];
    for hop in 0..hop_count {
        let start = hop * hop_samples;
        for (index, value) in frame.iter_mut().enumerate() {
            *value = signal[start + index] * window[index];
        }
        let spectrum = transforms.rfft(&frame, frame_samples);
        let power: Vec<f64> = spectrum.iter().map(|bin| bin.norm().powi(2)).collect();
        let mut band_power = vec![0.0; settings.band_count];
        for (bin, band) in bin_bands.iter().enumerate() {
            if let Some(band) = band {
                band_power[*band] += power[bin];
            }
        }
        for (band, nearest) in nearest_bin_for_empty_band.iter().enumerate() {
            if let Some(bin) = nearest {
                band_power[band] = power[*bin];
            }
        }
        energy_db.push(
            band_power
                .iter()
                .map(|value| 10.0 * (value + POWER_FLOOR).log10())
                .collect::<Vec<_>>(),
        );
    }
    let noise_floor_db = (0..settings.band_count)
        .map(|band| {
            percentile(
                &energy_db
                    .iter()
                    .map(|row: &Vec<f64>| row[band])
                    .collect::<Vec<_>>(),
                NOISE_FLOOR_PERCENTILE,
            )
        })
        .collect();
    SubbandHistory {
        energy_db,
        noise_floor_db,
        band_edges_hz: edges,
        hop_samples,
        frame_samples,
    }
}

/// Sum over active sub-bands of the positive dB rise beyond each band's rise noise.
pub fn positive_spectral_flux(history: &SubbandHistory, active_band_snr_db: f64) -> Vec<f64> {
    let hops = history.hop_count();
    if hops < 2 {
        return vec![0.0; hops];
    }
    let bands = history.band_count();
    let rise: Vec<Vec<f64>> = (0..hops)
        .map(|hop| {
            (0..bands)
                .map(|band| {
                    if hop == 0 {
                        0.0
                    } else {
                        history.energy_db[hop][band] - history.energy_db[hop - 1][band]
                    }
                })
                .collect()
        })
        .collect();
    let rise_noise_db: Vec<f64> = (0..bands)
        .map(|band| {
            let column: Vec<f64> = rise.iter().map(|row| row[band]).collect();
            let centre = median(&column);
            MAD_TO_STD
                * median(
                    &column
                        .iter()
                        .map(|value| (value - centre).abs())
                        .collect::<Vec<_>>(),
                )
        })
        .collect();
    (0..hops)
        .map(|hop| {
            (0..bands)
                .map(|band| {
                    let excess = (rise[hop][band]
                        - FLUX_RISE_NOISE_MULTIPLIER * rise_noise_db[band])
                        .max(0.0);
                    if history.snr_db(hop, band) >= active_band_snr_db {
                        excess
                    } else {
                        0.0
                    }
                })
                .sum()
        })
        .collect()
}

/// Frequency span of the sub-bands that gained energy at this onset.
fn onset_band_range(
    history: &SubbandHistory,
    hop: usize,
    frame_hops: usize,
    active_band_snr_db: f64,
) -> Option<(f64, f64)> {
    let stop = history.hop_count().min(hop + frame_hops.max(1));
    let before_start = hop.saturating_sub(ONSET_BASELINE_HOPS);
    let before_stop = hop.max(1);
    let mut rising = Vec::new();
    let mut active_bands = Vec::new();
    for band in 0..history.band_count() {
        let frame_rows: Vec<f64> = (hop..stop)
            .map(|row| history.energy_db[row][band])
            .collect();
        let before_rows: Vec<f64> = (before_start..before_stop)
            .map(|row| history.energy_db[row][band])
            .collect();
        let rise_db = mean(&frame_rows) - mean(&before_rows);
        let peak_snr = (hop..stop)
            .map(|row| history.snr_db(row, band))
            .fold(f64::NEG_INFINITY, f64::max);
        let active = peak_snr >= active_band_snr_db;
        if active {
            active_bands.push(band);
            if rise_db >= active_band_snr_db {
                rising.push(band);
            }
        }
    }
    let bands = if rising.is_empty() {
        active_bands
    } else {
        rising
    };
    let first = *bands.first()?;
    let last = *bands.last()?;
    Some((
        history.band_edges_hz[first],
        history.band_edges_hz[last + 1],
    ))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnsetDetectionSettings {
    pub sample_rate_hz: u32,
    pub flux_threshold_mad_multiplier: f64,
    pub min_separation_seconds: f64,
    pub max_onsets: usize,
    pub active_band_snr_db: f64,
    pub frame_seconds: f64,
}

/// Peaks of positive spectral flux above median + k * MAD, strongest first.
pub fn detect_onsets(
    history: &SubbandHistory,
    settings: &OnsetDetectionSettings,
) -> Vec<DetectedOnset> {
    let flux = positive_spectral_flux(history, settings.active_band_snr_db);
    if flux.len() < 3 {
        return Vec::new();
    }
    let centre = median(&flux);
    let mad_std = MAD_TO_STD
        * median(
            &flux
                .iter()
                .map(|value| (value - centre).abs())
                .collect::<Vec<_>>(),
        );
    // A mostly silent window has MAD 0; fall back to the flux std.
    let spread = if mad_std > 0.0 {
        mad_std
    } else {
        population_std(&flux)
    };
    if spread <= 0.0 {
        return Vec::new();
    }
    let normalized: Vec<f64> = flux.iter().map(|value| (value - centre) / spread).collect();
    let mut candidates: Vec<usize> = (1..normalized.len() - 1)
        .filter(|&hop| {
            normalized[hop] >= normalized[hop - 1]
                && normalized[hop] > normalized[hop + 1]
                && normalized[hop] >= settings.flux_threshold_mad_multiplier
        })
        .collect();
    let hop_samples = history.hop_samples as f64;
    let sample_rate = settings.sample_rate_hz as f64;
    let min_separation_hops =
        python_round_to_i64(settings.min_separation_seconds * sample_rate / hop_samples).max(1);
    let frame_hops =
        python_round_to_i64(settings.frame_seconds * sample_rate / hop_samples).max(1) as usize;
    // Stable sort, strongest first (ties keep ascending hop order, as Python's sorted()).
    candidates.sort_by(|a, b| normalized[*b].total_cmp(&normalized[*a]));
    let mut chosen: Vec<usize> = Vec::new();
    for hop in candidates {
        if chosen
            .iter()
            .all(|other| (hop as i64 - *other as i64).abs() >= min_separation_hops)
        {
            chosen.push(hop);
        }
        if chosen.len() >= settings.max_onsets {
            break;
        }
    }
    chosen
        .into_iter()
        .filter_map(|hop| {
            let (band_min_hz, band_max_hz) =
                onset_band_range(history, hop, frame_hops, settings.active_band_snr_db)?;
            Some(DetectedOnset {
                sample_index: history.hop_start_sample(hop),
                strength: normalized[hop],
                band_min_hz,
                band_max_hz,
            })
        })
        .collect()
}
