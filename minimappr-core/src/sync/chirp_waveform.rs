//! Ranging chirp waveforms: linear FM, hyperbolic FM and frequency-hopped
//! coded chirps, plus the stereo buffer that puts a chirp on one speaker.
//!
//! Every generator returns a unit-peak mono `Vec<f32>` that sweeps inside
//! `[band_low_hz, band_high_hz]` with a Tukey taper (hard edges splatter
//! below the band and drive the self-recording limiter harder). The same
//! buffer is the matched-filter template, so playback and detection never
//! disagree about the waveform.

use std::f64::consts::{PI, TAU};

/// Candidate chirp bands swept by the Phase 1 harness (plan, "Chirp design").
/// The floor sits at 13 kHz so the detection low-pass (stop band from
/// 12.5 kHz) removes every chirp.
pub const CHIRP_BAND_13_TO_16_KHZ: ChirpBand = ChirpBand {
    low_hz: 13_000.0,
    high_hz: 16_000.0,
};
pub const CHIRP_BAND_15_TO_18_KHZ: ChirpBand = ChirpBand {
    low_hz: 15_000.0,
    high_hz: 18_000.0,
};
pub const CHIRP_BAND_18_TO_20_KHZ: ChirpBand = ChirpBand {
    low_hz: 18_000.0,
    high_hz: 20_000.0,
};

/// Default fraction of the chirp spent in the Tukey ramps (both ends).
pub const DEFAULT_CHIRP_TAPER_FRACTION: f64 = 0.1;
/// Default chirp length: ~50 ms leaves a 0.20–0.25 s slot room for 100 ms
/// of flight at 35 m plus a reverberation tail (plan, "Schedule").
pub const DEFAULT_CHIRP_DURATION_S: f64 = 0.05;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChirpBand {
    pub low_hz: f64,
    pub high_hz: f64,
}

impl ChirpBand {
    pub fn bandwidth_hz(&self) -> f64 {
        self.high_hz - self.low_hz
    }

    pub fn centre_hz(&self) -> f64 {
        0.5 * (self.low_hz + self.high_hz)
    }

    /// Split into `count` equal contiguous sub-bands (low → high).
    pub fn split_into_sub_bands(&self, count: usize) -> Vec<ChirpBand> {
        let count = count.max(1);
        let width = self.bandwidth_hz() / count as f64;
        (0..count)
            .map(|index| ChirpBand {
                low_hz: self.low_hz + width * index as f64,
                high_hz: self.low_hz + width * (index + 1) as f64,
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SweepDirection {
    Up,
    Down,
}

/// Instantaneous-frequency law of one sweep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SweepLaw {
    /// f(t) linear in t. A Doppler shift becomes a timing error of
    /// Δf · T / B (range–Doppler coupling).
    LinearFm,
    /// Period 1/f(t) linear in t: Doppler-tolerant (the matched-filter peak
    /// keeps its height under time compression).
    HyperbolicFm,
}

/// One continuous sweep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SweepSpecification {
    pub band: ChirpBand,
    pub duration_s: f64,
    pub law: SweepLaw,
    pub direction: SweepDirection,
}

/// Phase φ(t) (radians) of an upward sweep from `f_start` to `f_end`.
fn sweep_phase_radians(law: SweepLaw, f_start: f64, f_end: f64, duration_s: f64, t: f64) -> f64 {
    match law {
        SweepLaw::LinearFm => TAU * (f_start * t + (f_end - f_start) * t * t / (2.0 * duration_s)),
        SweepLaw::HyperbolicFm => {
            // f(t) = f_start f_end T / (f_end T − (f_end − f_start) t); φ is its integral.
            let k = (f_end - f_start) / (f_end * duration_s);
            if k.abs() < f64::EPSILON {
                TAU * f_start * t
            } else {
                -TAU * f_start / k * (1.0 - k * t).ln()
            }
        }
    }
}

/// Raised-cosine (Tukey) gain for `index` of `length`.
pub fn tukey_gain(index: usize, length: usize, taper_fraction: f64) -> f64 {
    let taper_samples = ((length as f64) * taper_fraction / 2.0) as usize;
    if taper_samples == 0 || length == 0 {
        return 1.0;
    }
    let edge_distance = index.min(length - 1 - index);
    if edge_distance >= taper_samples {
        1.0
    } else {
        0.5 - 0.5 * (PI * edge_distance as f64 / taper_samples as f64).cos()
    }
}

/// Un-tapered samples of one sweep (unit amplitude).
fn untapered_sweep(sweep: &SweepSpecification, sample_rate_hz: f64) -> Vec<f64> {
    let length = (sweep.duration_s * sample_rate_hz).round() as usize;
    let duration = sweep.duration_s.max(f64::EPSILON);
    let (f_start, f_end) = match sweep.direction {
        SweepDirection::Up => (sweep.band.low_hz, sweep.band.high_hz),
        SweepDirection::Down => (sweep.band.high_hz, sweep.band.low_hz),
    };
    (0..length)
        .map(|index| {
            let t = index as f64 / sample_rate_hz;
            sweep_phase_radians(sweep.law, f_start, f_end, duration, t).sin()
        })
        .collect()
}

/// Unit-peak tapered single sweep.
pub fn generate_sweep(
    sweep: &SweepSpecification,
    taper_fraction: f64,
    sample_rate_hz: f64,
) -> Vec<f32> {
    let raw = untapered_sweep(sweep, sample_rate_hz);
    let length = raw.len();
    raw.iter()
        .enumerate()
        .map(|(index, value)| (value * tukey_gain(index, length, taper_fraction)) as f32)
        .collect()
}

/// Concatenate sweeps back to back, tapering each segment's edges so the
/// hops between sub-bands stay click-free. Unit peak.
pub fn generate_segmented_sweep(
    segments: &[SweepSpecification],
    segment_taper_fraction: f64,
    sample_rate_hz: f64,
) -> Vec<f32> {
    let mut output = Vec::new();
    for segment in segments {
        output.extend(generate_sweep(
            segment,
            segment_taper_fraction,
            sample_rate_hz,
        ));
    }
    output
}

/// Which waveform a session plays; per-phone codes live in `chirp_codes`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChirpWaveformKind {
    LinearFm,
    HyperbolicFm,
    /// Frequency-hopped segments with per-phone hop order and sweep
    /// directions (see `chirp_codes::SessionChirpCodebook`).
    SessionCodedHopping,
}

impl ChirpWaveformKind {
    pub fn name(&self) -> &'static str {
        match self {
            Self::LinearFm => "lfm",
            Self::HyperbolicFm => "hfm",
            Self::SessionCodedHopping => "coded_hfm_hop",
        }
    }
}

/// Plain (uncoded) up-sweep of the whole band.
pub fn generate_plain_chirp(
    band: ChirpBand,
    duration_s: f64,
    law: SweepLaw,
    sample_rate_hz: f64,
) -> Vec<f32> {
    generate_sweep(
        &SweepSpecification {
            band,
            duration_s,
            law,
            direction: SweepDirection::Up,
        },
        DEFAULT_CHIRP_TAPER_FRACTION,
        sample_rate_hz,
    )
}

/// Stereo output channel that carries the chirp.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StereoOutputChannel {
    Left,
    Right,
}

/// Interleaved stereo buffer `[L0, R0, L1, R1, …]` with the chirp on one
/// channel only. Phones play `USAGE_MEDIA` from the bottom speaker and the
/// earpiece together; a mono chirp would leave as two copies ~15 cm apart
/// (comb filtering, double arrivals, a spoiled d_ii). Phase 1 finds which
/// channel maps to the bottom speaker per model.
pub fn stereo_single_channel_buffer(mono: &[f32], channel: StereoOutputChannel) -> Vec<f32> {
    let mut interleaved = vec![0.0f32; mono.len() * 2];
    let channel_offset = match channel {
        StereoOutputChannel::Left => 0,
        StereoOutputChannel::Right => 1,
    };
    for (index, sample) in mono.iter().enumerate() {
        interleaved[2 * index + channel_offset] = *sample;
    }
    interleaved
}

/// Scale so the RMS over samples above 1% of peak is 1 (matches how the
/// simulator levels emissions; keeps "level dB SPL at 1 m" comparable
/// across waveforms).
pub fn normalize_to_unit_active_rms(samples: &mut [f32]) {
    const ACTIVE_FRACTION_OF_PEAK: f32 = 0.01;
    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    if peak <= 0.0 {
        return;
    }
    let (sum_squares, count) = samples
        .iter()
        .filter(|s| s.abs() > peak * ACTIVE_FRACTION_OF_PEAK)
        .fold((0.0f64, 0usize), |(acc, n), s| {
            (acc + (*s as f64).powi(2), n + 1)
        });
    if count == 0 {
        return;
    }
    let gain = (1.0 / (sum_squares / count as f64).sqrt()) as f32;
    samples.iter_mut().for_each(|s| *s *= gain);
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE_HZ: f64 = 48_000.0;

    /// Zero-crossing frequency estimate over a short window.
    fn zero_crossing_frequency(samples: &[f32], start: usize, length: usize) -> f64 {
        let window = &samples[start..start + length];
        let crossings = window
            .windows(2)
            .filter(|pair| (pair[0] <= 0.0) != (pair[1] <= 0.0))
            .count();
        crossings as f64 / 2.0 / (length as f64 / SAMPLE_RATE_HZ)
    }

    #[test]
    fn plain_sweeps_start_low_and_end_high_inside_band() {
        for law in [SweepLaw::LinearFm, SweepLaw::HyperbolicFm] {
            let chirp = generate_plain_chirp(CHIRP_BAND_13_TO_16_KHZ, 0.05, law, SAMPLE_RATE_HZ);
            assert_eq!(chirp.len(), 2400);
            let early = zero_crossing_frequency(&chirp, 200, 240);
            let late = zero_crossing_frequency(&chirp, 1960, 240);
            assert!(
                early > 12_800.0 && early < 13_800.0,
                "{law:?} early {early}"
            );
            assert!(late > 15_200.0 && late < 16_200.0, "{law:?} late {late}");
            assert!(chirp.iter().all(|s| s.abs() <= 1.0));
            assert!(chirp[0].abs() < 1e-6, "taper must start at zero");
        }
    }

    #[test]
    fn hyperbolic_period_is_linear_in_time() {
        let sweep = SweepSpecification {
            band: CHIRP_BAND_15_TO_18_KHZ,
            duration_s: 0.1,
            law: SweepLaw::HyperbolicFm,
            direction: SweepDirection::Up,
        };
        let samples = generate_sweep(&sweep, 0.0, SAMPLE_RATE_HZ);
        let window = 480;
        let periods: Vec<f64> = [0, 2000, 4000]
            .iter()
            .map(|start| 1.0 / zero_crossing_frequency(&samples, *start, window))
            .collect();
        let first_step = periods[1] - periods[0];
        let second_step = periods[2] - periods[1];
        assert!((first_step - second_step).abs() < 0.15 * first_step.abs());
    }

    #[test]
    fn down_sweep_reverses_direction() {
        let sweep = SweepSpecification {
            band: CHIRP_BAND_13_TO_16_KHZ,
            duration_s: 0.05,
            law: SweepLaw::LinearFm,
            direction: SweepDirection::Down,
        };
        let samples = generate_sweep(&sweep, 0.0, SAMPLE_RATE_HZ);
        assert!(
            zero_crossing_frequency(&samples, 0, 240)
                > zero_crossing_frequency(&samples, 2100, 240)
        );
    }

    #[test]
    fn stereo_buffer_puts_chirp_on_one_channel_only() {
        let mono = vec![0.5f32, -0.25, 1.0];
        let right = stereo_single_channel_buffer(&mono, StereoOutputChannel::Right);
        assert_eq!(right, vec![0.0, 0.5, 0.0, -0.25, 0.0, 1.0]);
        let left = stereo_single_channel_buffer(&mono, StereoOutputChannel::Left);
        assert_eq!(left, vec![0.5, 0.0, -0.25, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn sub_bands_tile_the_band() {
        let parts = CHIRP_BAND_15_TO_18_KHZ.split_into_sub_bands(4);
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0].low_hz, 15_000.0);
        assert!((parts[3].high_hz - 18_000.0).abs() < 1e-9);
    }

    #[test]
    fn active_rms_normalization_gives_unit_rms() {
        let mut chirp = generate_plain_chirp(
            CHIRP_BAND_18_TO_20_KHZ,
            0.05,
            SweepLaw::LinearFm,
            SAMPLE_RATE_HZ,
        );
        normalize_to_unit_active_rms(&mut chirp);
        let rms =
            (chirp.iter().map(|s| (*s as f64).powi(2)).sum::<f64>() / chirp.len() as f64).sqrt();
        assert!(rms > 0.9 && rms < 1.05, "rms {rms}");
    }
}
