//! Chirp arrival picking for the shell: the core `SplitPartDetector` on the
//! same waveform the phone plays. Chirp lab uses it for live self-SNR and
//! clipping readouts; the session listener (`chirp_round_listener`) reuses
//! `detect_in_f32_slice` on rolling-buffer windows.

use minimappr_core::sync::first_arrival::{ArrivalContext, FirstArrivalConfig};
use minimappr_core::sync::split_part_detector::SplitPartDetector;

use crate::chirp_waveforms::{chirp_waveform_from_spec, FieldChirpWaveformSpec, CHIRP_SAMPLE_RATE_HZ};
use crate::error::FieldFfiError;
use crate::pcm16_bytes::pcm16_samples_from_little_endian_bytes;

const PCM16_FULL_SCALE: f32 = i16::MAX as f32;

#[derive(Debug, Clone, Copy, PartialEq, uniffi::Record)]
pub struct FieldChirpDetection {
    /// Waveform-start lag in the analysed slice, fractional samples.
    pub sample_index: f64,
    /// Weakest part's SNR.
    pub snr_db: f64,
    /// Last part minus first part (0 for single-part waveforms).
    pub doppler_skew_samples: f64,
    /// Largest |sample| over the detected chirp span as a fraction of full
    /// scale: near 1 means the capture path clipped the chirp.
    pub peak_fraction: f32,
}

#[derive(uniffi::Object)]
pub struct FieldChirpDetector {
    detector: SplitPartDetector,
}

impl FieldChirpDetector {
    pub fn from_spec(spec: &FieldChirpWaveformSpec, sound_speed_mps: f64) -> Result<Self, FieldFfiError> {
        let waveform = chirp_waveform_from_spec(spec)?;
        let config = FirstArrivalConfig { air_sound_speed_mps: sound_speed_mps, ..FirstArrivalConfig::default() };
        Ok(Self { detector: SplitPartDetector::new(&waveform, config, CHIRP_SAMPLE_RATE_HZ) })
    }

    pub fn waveform_length(&self) -> usize {
        self.detector.waveform_length()
    }

    /// `samples` are full-scale-normalised; `None` when no part clears the
    /// noise floor or the window is empty.
    pub fn detect_in_f32_slice(
        &self,
        samples: &[f32],
        first_lag: usize,
        end_lag: usize,
        self_path_length_m: Option<f64>,
        is_self: bool,
    ) -> Option<FieldChirpDetection> {
        let end_lag = end_lag.min(samples.len().saturating_sub(self.detector.waveform_length()) + 1);
        if first_lag >= end_lag {
            return None;
        }
        let context = if is_self {
            ArrivalContext::SelfRecording { self_path_length_m }
        } else {
            ArrivalContext::RemoteEmitter
        };
        let arrival = self.detector.detect(samples, first_lag..end_lag, context).ok()?;
        let span_start = (arrival.sample_index.max(0.0) as usize).min(samples.len());
        let span_end = (span_start + self.detector.waveform_length()).min(samples.len());
        let peak_fraction = samples[span_start..span_end].iter().fold(0.0f32, |peak, sample| peak.max(sample.abs()));
        Some(FieldChirpDetection {
            sample_index: arrival.sample_index,
            snr_db: arrival.snr_db,
            doppler_skew_samples: arrival.doppler_skew_samples,
            peak_fraction,
        })
    }
}

pub fn pcm16_to_unit_f32(samples: &[i16]) -> Vec<f32> {
    samples.iter().map(|sample| *sample as f32 / PCM16_FULL_SCALE).collect()
}

#[uniffi::export]
impl FieldChirpDetector {
    #[uniffi::constructor]
    pub fn new(spec: FieldChirpWaveformSpec, sound_speed_mps: f64) -> Result<Self, FieldFfiError> {
        if !(300.0..=380.0).contains(&sound_speed_mps) {
            return Err(FieldFfiError::InvalidArgument { detail: format!("sound speed {sound_speed_mps} m/s out of range") });
        }
        Self::from_spec(&spec, sound_speed_mps)
    }

    /// Search waveform-start lags `first_lag..end_lag` of a mono PCM16 slice.
    pub fn detect(
        &self,
        pcm16_little_endian_bytes: Vec<u8>,
        first_lag: u32,
        end_lag: u32,
        is_self: bool,
        self_path_length_m: Option<f64>,
    ) -> Result<Option<FieldChirpDetection>, FieldFfiError> {
        let samples = pcm16_to_unit_f32(&pcm16_samples_from_little_endian_bytes(&pcm16_little_endian_bytes)?);
        Ok(self.detect_in_f32_slice(&samples, first_lag as usize, end_lag as usize, self_path_length_m, is_self))
    }

    pub fn template_frame_count(&self) -> u32 {
        self.detector.waveform_length() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chirp_waveforms::FieldChirpWaveformKind;
    use crate::pcm16_bytes::pcm16_little_endian_bytes;

    fn spec(kind: FieldChirpWaveformKind) -> FieldChirpWaveformSpec {
        FieldChirpWaveformSpec { kind, band_low_hz: 13_000.0, band_high_hz: 16_000.0, duration_s: 0.05 }
    }

    /// Deterministic low-level noise so the noise floor is not exactly zero.
    fn noise(length: usize, amplitude: f32) -> Vec<f32> {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        (0..length)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * amplitude
            })
            .collect()
    }

    #[test]
    fn inserted_chirp_is_recovered_within_a_tenth_of_a_sample() {
        for kind in [FieldChirpWaveformKind::UpDownHyperbolic, FieldChirpWaveformKind::HyperbolicFm, FieldChirpWaveformKind::LinearFm] {
            let waveform = chirp_waveform_from_spec(&spec(kind)).unwrap();
            let insert_at = 3_217;
            let mut recording = noise(12_000, 0.002);
            for (offset, sample) in waveform.samples.iter().enumerate() {
                recording[insert_at + offset] += 0.5 * sample;
            }
            let pcm: Vec<i16> = recording.iter().map(|s| (s * PCM16_FULL_SCALE) as i16).collect();
            let detector = FieldChirpDetector::new(spec(kind), 343.0).unwrap();
            let detection = detector
                .detect(pcm16_little_endian_bytes(&pcm), 0, 8_000, false, None)
                .unwrap()
                .expect("detected");
            assert!((detection.sample_index - insert_at as f64).abs() < 0.1, "{kind:?}: {detection:?}");
            assert!(detection.snr_db > 20.0);
            assert!((detection.peak_fraction - 0.5).abs() < 0.05);
        }
    }

    #[test]
    fn silence_yields_no_detection_and_empty_window_is_none() {
        let detector = FieldChirpDetector::new(spec(FieldChirpWaveformKind::UpDownHyperbolic), 343.0).unwrap();
        let quiet: Vec<i16> = noise(12_000, 0.002).iter().map(|s| (s * PCM16_FULL_SCALE) as i16).collect();
        assert!(detector.detect(pcm16_little_endian_bytes(&quiet), 0, 8_000, false, None).unwrap().is_none());
        assert!(detector.detect(pcm16_little_endian_bytes(&quiet), 9_000, 9_000, false, None).unwrap().is_none());
    }
}
