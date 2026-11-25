//! Arrival picks in a recording: the core `SplitPartDetector` (matched
//! filter, earliest-peak and precursor logic) in a window around a
//! predicted sample. When a bundle has no timestamps, a blind scan walks the
//! whole file; the report flags blind-scanned runs because a blind pick can
//! lock onto the other phone's chirp.

use minimappr_core::sync::chirp_waveform::{generate_plain_chirp, generate_up_down_chirp, ChirpBand, ChirpWaveform, SweepLaw};
use minimappr_core::sync::first_arrival::{ArrivalContext, FirstArrivalConfig};
use minimappr_core::sync::split_part_detector::SplitPartDetector;

use crate::bundle_schema::WaveformDescription;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScannedArrival {
    /// Waveform-start sample in the whole recording (fractional).
    pub sample_index: f64,
    pub snr_db: f64,
    pub doppler_skew_samples: f64,
    /// Largest |sample| over the chirp span (≈1 = clipped).
    pub peak_fraction: f32,
}

pub fn waveform_from_description(description: &WaveformDescription, sample_rate_hz: f64) -> Result<ChirpWaveform, String> {
    let band = ChirpBand { low_hz: description.band_low_hz, high_hz: description.band_high_hz };
    if band.low_hz <= 0.0 || band.high_hz <= band.low_hz || description.duration_s <= 0.0 {
        return Err(format!("invalid waveform description {description:?}"));
    }
    Ok(match description.kind.as_str() {
        "updown_hfm" => generate_up_down_chirp(band, description.duration_s, SweepLaw::HyperbolicFm, sample_rate_hz),
        "hfm" => ChirpWaveform::single_part(generate_plain_chirp(band, description.duration_s, SweepLaw::HyperbolicFm, sample_rate_hz)),
        "lfm" => ChirpWaveform::single_part(generate_plain_chirp(band, description.duration_s, SweepLaw::LinearFm, sample_rate_hz)),
        other => return Err(format!("unknown waveform kind {other:?}")),
    })
}

pub struct ArrivalScanner {
    detector: SplitPartDetector,
    waveform_length: usize,
}

impl ArrivalScanner {
    pub fn new(waveform: &ChirpWaveform, sound_speed_mps: f64, sample_rate_hz: f64) -> Self {
        let config = FirstArrivalConfig { air_sound_speed_mps: sound_speed_mps, ..FirstArrivalConfig::default() };
        Self { detector: SplitPartDetector::new(waveform, config, sample_rate_hz), waveform_length: waveform.samples.len() }
    }

    pub fn waveform_length(&self) -> usize {
        self.waveform_length
    }

    /// Search waveform starts within `centre ± half_window` samples.
    pub fn detect_near(&self, audio: &[f32], centre_sample: f64, half_window_samples: f64, context: ArrivalContext) -> Option<ScannedArrival> {
        let first = (centre_sample - half_window_samples).max(0.0) as usize;
        let lag_count = (2.0 * half_window_samples) as usize;
        self.detect_in_range(audio, first, lag_count, context)
    }

    pub fn detect_in_range(&self, audio: &[f32], first: usize, lag_count: usize, context: ArrivalContext) -> Option<ScannedArrival> {
        if first >= audio.len() {
            return None;
        }
        let end = (first + lag_count + self.waveform_length).min(audio.len());
        let slice = &audio[first..end];
        let usable_lags = slice.len().checked_sub(self.waveform_length)? + 1;
        let arrival = self.detector.detect(slice, 0..lag_count.min(usable_lags), context).ok()?;
        let span_start = (arrival.sample_index.max(0.0) as usize).min(slice.len());
        let span_end = (span_start + self.waveform_length).min(slice.len());
        Some(ScannedArrival {
            sample_index: first as f64 + arrival.sample_index,
            snr_db: arrival.snr_db,
            doppler_skew_samples: arrival.doppler_skew_samples,
            peak_fraction: slice[span_start..span_end].iter().fold(0.0f32, |peak, s| peak.max(s.abs())),
        })
    }

    /// Every chirp in the file with SNR ≥ `minimum_snr_db`, at least
    /// `minimum_spacing_samples` apart (strongest wins inside the spacing).
    pub fn blind_scan(&self, audio: &[f32], chunk_samples: usize, minimum_snr_db: f64, minimum_spacing_samples: usize) -> Vec<ScannedArrival> {
        let mut found: Vec<ScannedArrival> = Vec::new();
        let step = (chunk_samples / 2).max(1);
        let mut first = 0;
        while first + self.waveform_length < audio.len() {
            if let Some(arrival) = self.detect_in_range(audio, first, chunk_samples, ArrivalContext::RemoteEmitter) {
                if arrival.snr_db >= minimum_snr_db {
                    match found.iter_mut().find(|existing| (existing.sample_index - arrival.sample_index).abs() < minimum_spacing_samples as f64) {
                        Some(existing) if existing.snr_db < arrival.snr_db => *existing = arrival,
                        Some(_) => {}
                        None => found.push(arrival),
                    }
                }
            }
            first += step;
        }
        found.sort_by(|a, b| a.sample_index.total_cmp(&b.sample_index));
        found
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_an_embedded_chirp_near_its_prediction_and_blind() {
        let description = WaveformDescription { kind: "updown_hfm".into(), band_low_hz: 13_000.0, band_high_hz: 16_000.0, duration_s: 0.05 };
        let waveform = waveform_from_description(&description, 48_000.0).unwrap();
        let mut audio = vec![0.0f32; 96_000];
        for (index, value) in audio.iter_mut().enumerate() {
            *value = ((index * 7919 % 1000) as f32 / 1000.0 - 0.5) * 0.002;
        }
        for start in [20_000usize, 70_000] {
            for (offset, sample) in waveform.samples.iter().enumerate() {
                audio[start + offset] += 0.3 * sample;
            }
        }
        let scanner = ArrivalScanner::new(&waveform, 343.0, 48_000.0);
        let near = scanner.detect_near(&audio, 20_300.0, 1_000.0, ArrivalContext::RemoteEmitter).unwrap();
        assert!((near.sample_index - 20_000.0).abs() < 0.2, "{near:?}");
        let blind = scanner.blind_scan(&audio, 48_000, 20.0, 24_000);
        assert_eq!(blind.len(), 2, "{blind:?}");
        assert!((blind[1].sample_index - 70_000.0).abs() < 0.2);
    }
}
