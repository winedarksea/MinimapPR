//! Arrival picking of multi-part waveforms (up/down "V" chirps and coded
//! up/down codes): each part is picked on its own and the part arrivals,
//! moved back to the waveform start, are averaged.
//!
//! Why: any FM sweep turns a Doppler shift into a timing error (range–
//! Doppler coupling). At 16.5 kHz, 1.2 m/s of relative motion shifts the
//! peak of a 50 ms, 3 kHz sweep by ~1 ms (≈ 33 cm of range); hyperbolic FM
//! keeps the peak tall under Doppler but still shifts it. An up-sweep and a
//! down-sweep shift in opposite directions, so the mean cancels the bias to
//! first order and the difference measures it (`doppler_skew_samples`).
//! A single-part waveform degenerates to the plain first-arrival pick.

use std::ops::Range;

use super::chirp_waveform::ChirpWaveform;
use super::first_arrival::{
    ArrivalContext, ArrivalDetectionFailure, ChirpArrival, FirstArrivalConfig, FirstArrivalDetector,
};

#[derive(Clone, Debug, PartialEq)]
pub struct SplitPartArrival {
    /// Mean part arrival, as a waveform-start lag in the recording slice.
    pub sample_index: f64,
    pub part_arrivals: Vec<ChirpArrival>,
    /// Last part's aligned arrival minus the first part's (0 for one part):
    /// twice the Doppler coupling shift for an up/down pair.
    pub doppler_skew_samples: f64,
    /// Weakest part's SNR.
    pub snr_db: f64,
}

/// Later parts are searched only this far (either way) from where the
/// first part puts the waveform start. The Doppler skew between an up and
/// a down half is ~2 ms at 2.4 m/s relative speed for 25 ms, 3 kHz halves;
/// a wider search lets a later part's template lock onto an earlier part
/// (short hopped segments are far from orthogonal).
pub const DEFAULT_MAXIMUM_PART_SKEW_S: f64 = 0.003;

#[derive(Clone, Debug)]
pub struct SplitPartDetector {
    parts: Vec<(usize, FirstArrivalDetector)>,
    waveform_length: usize,
    maximum_part_skew_samples: usize,
}

impl SplitPartDetector {
    pub fn new(waveform: &ChirpWaveform, config: FirstArrivalConfig, sample_rate_hz: f64) -> Self {
        Self {
            parts: waveform
                .parts
                .iter()
                .map(|part| {
                    (
                        part.start_sample,
                        FirstArrivalDetector::new(&part.samples, config.clone(), sample_rate_hz),
                    )
                })
                .collect(),
            waveform_length: waveform.samples.len(),
            maximum_part_skew_samples: (DEFAULT_MAXIMUM_PART_SKEW_S * sample_rate_hz) as usize,
        }
    }

    pub fn waveform_length(&self) -> usize {
        self.waveform_length
    }

    /// The first part's self-recording precursor window (all parts share the
    /// band, hence the main-lobe width that sets it).
    pub fn precursor_lead_window_samples(&self, self_path_length_m: Option<f64>) -> Option<f64> {
        self.parts.first().map(|(_, detector)| detector.precursor_lead_window_samples(self_path_length_m))
    }

    /// `search_lags` are waveform-start lags; every part must be detected.
    pub fn detect(
        &self,
        recording: &[f32],
        search_lags: Range<usize>,
        context: ArrivalContext,
    ) -> Result<SplitPartArrival, ArrivalDetectionFailure> {
        let mut part_arrivals = Vec::with_capacity(self.parts.len());
        let mut aligned: Vec<f64> = Vec::with_capacity(self.parts.len());
        for (offset, detector) in &self.parts {
            let lags = match aligned.first() {
                None => (search_lags.start + offset)..(search_lags.end + offset),
                Some(&first_start) => {
                    let centre = (first_start.max(0.0) as usize) + offset;
                    centre.saturating_sub(self.maximum_part_skew_samples)
                        ..centre + self.maximum_part_skew_samples + 1
                }
            };
            let arrival = detector.detect(recording, lags, context)?;
            aligned.push(arrival.sample_index - *offset as f64);
            part_arrivals.push(arrival);
        }
        let count = aligned.len().max(1) as f64;
        Ok(SplitPartArrival {
            sample_index: aligned.iter().sum::<f64>() / count,
            doppler_skew_samples: aligned.last().copied().unwrap_or(0.0)
                - aligned.first().copied().unwrap_or(0.0),
            snr_db: part_arrivals
                .iter()
                .map(|a| a.snr_db)
                .fold(f64::INFINITY, f64::min),
            part_arrivals,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::chirp_waveform::{
        generate_plain_chirp, generate_up_down_chirp, SweepLaw, CHIRP_BAND_15_TO_18_KHZ,
    };

    const SAMPLE_RATE_HZ: f64 = 48_000.0;
    const DURATION_S: f64 = 0.05;

    /// Time-compress a waveform by factor `eta` (> 1: approaching) through
    /// re-synthesis at 8× rate and decimation, i.e. s(η t).
    fn doppler_compressed(make: impl Fn(f64) -> Vec<f32>, eta: f64) -> Vec<f32> {
        const OVERSAMPLING: f64 = 8.0;
        let fine = make(SAMPLE_RATE_HZ * OVERSAMPLING);
        let length = (fine.len() as f64 / OVERSAMPLING / eta) as usize;
        (0..length)
            .map(|n| fine[((n as f64 * eta * OVERSAMPLING).round() as usize).min(fine.len() - 1)])
            .collect()
    }

    fn embed(signal: &[f32], start: usize) -> Vec<f32> {
        let mut recording = vec![0.0f32; 12_000];
        recording[start..start + signal.len()].copy_from_slice(signal);
        recording
    }

    #[test]
    fn up_down_pair_cancels_doppler_bias_that_shifts_a_single_sweep() {
        let relative_speed_mps = 2.4;
        let eta = 1.0 + relative_speed_mps / 343.0;
        let start = 4_000;
        let up_down = |fs| {
            generate_up_down_chirp(
                CHIRP_BAND_15_TO_18_KHZ,
                DURATION_S,
                SweepLaw::HyperbolicFm,
                fs,
            )
            .samples
        };
        let plain = |fs| {
            generate_plain_chirp(
                CHIRP_BAND_15_TO_18_KHZ,
                DURATION_S,
                SweepLaw::HyperbolicFm,
                fs,
            )
        };
        let config = FirstArrivalConfig::default();

        let single = SplitPartDetector::new(
            &ChirpWaveform::single_part(plain(SAMPLE_RATE_HZ)),
            config.clone(),
            SAMPLE_RATE_HZ,
        );
        let single_arrival = single
            .detect(
                &embed(&doppler_compressed(plain, eta), start),
                0..8_000,
                ArrivalContext::RemoteEmitter,
            )
            .unwrap();
        let single_bias = single_arrival.sample_index - start as f64;

        let waveform = generate_up_down_chirp(
            CHIRP_BAND_15_TO_18_KHZ,
            DURATION_S,
            SweepLaw::HyperbolicFm,
            SAMPLE_RATE_HZ,
        );
        let split = SplitPartDetector::new(&waveform, config, SAMPLE_RATE_HZ);
        let split_arrival = split
            .detect(
                &embed(&doppler_compressed(up_down, eta), start),
                0..8_000,
                ArrivalContext::RemoteEmitter,
            )
            .unwrap();
        let split_bias = split_arrival.sample_index - start as f64;
        assert!(single_bias.abs() > 10.0, "single sweep bias {single_bias}");
        assert!(
            split_bias.abs() < 0.2 * single_bias.abs(),
            "split {split_bias} vs single {single_bias}"
        );
        assert!(split_arrival.doppler_skew_samples.abs() > 10.0);
    }

    #[test]
    fn coded_up_down_pick_is_exact() {
        use crate::sync::chirp_codes::{SessionChirpCodebook, DEFAULT_CODE_SEGMENT_COUNT};
        let book = SessionChirpCodebook::new(
            99,
            CHIRP_BAND_15_TO_18_KHZ,
            DURATION_S,
            DEFAULT_CODE_SEGMENT_COUNT,
            SweepLaw::HyperbolicFm,
        );
        for slot in 0..6 {
            let waveform = book.waveform_for_slot(slot, SAMPLE_RATE_HZ);
            let detector =
                SplitPartDetector::new(&waveform, FirstArrivalConfig::default(), SAMPLE_RATE_HZ);
            let arrival = detector
                .detect(
                    &embed(&waveform.samples, 3_000),
                    0..8_000,
                    ArrivalContext::RemoteEmitter,
                )
                .unwrap();
            assert!(
                (arrival.sample_index - 3_000.0).abs() < 0.5,
                "slot {slot}: {arrival:?}"
            );
        }
    }

    #[test]
    fn stationary_up_down_pick_is_exact() {
        let waveform = generate_up_down_chirp(
            CHIRP_BAND_15_TO_18_KHZ,
            DURATION_S,
            SweepLaw::HyperbolicFm,
            SAMPLE_RATE_HZ,
        );
        let detector =
            SplitPartDetector::new(&waveform, FirstArrivalConfig::default(), SAMPLE_RATE_HZ);
        let arrival = detector
            .detect(
                &embed(&waveform.samples, 3_000),
                0..8_000,
                ArrivalContext::RemoteEmitter,
            )
            .unwrap();
        assert!((arrival.sample_index - 3_000.0).abs() < 0.2, "{arrival:?}");
        assert!(arrival.doppler_skew_samples.abs() < 0.2);
        assert_eq!(arrival.part_arrivals.len(), 2);
    }
}
