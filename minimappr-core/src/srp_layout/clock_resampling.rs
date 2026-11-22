//! In-clip clock-rate resampling: maps every node's clip onto one shared
//! island timeline before correlation.
//!
//! Why: a phone crystal off by 60 ppm drifts 0.6 ms in 10 s since its last
//! sync anchor (20 cm of path) and smears a 1 s correlation by 60 µs. The
//! clock filter delivers offset + rate; applying only the offset leaves both
//! errors. One windowed-sinc pass per channel applies offset and rate
//! together, so every downstream lag is in island time.

use std::sync::OnceLock;

/// Taps on each side of the interpolation point. 16 keeps passband ripple
/// well under 0.01 dB below 0.9 × Nyquist with the Kaiser window below.
const SINC_HALF_WIDTH_TAPS: usize = 16;
/// Tabulated fractional phases; 1/1024 sample ≈ 20 ns at 48 kHz.
const SINC_PHASE_COUNT: usize = 1024;
/// Kaiser β ≈ 8.6 trades ~90 dB stopband for a slightly wider transition.
const KAISER_BETA: f64 = 8.6;
/// Low-pass cutoff as a fraction of Nyquist; the localization band ends far
/// below it, and the margin keeps the windowed sinc's transition out of band.
const SINC_CUTOFF_FRACTION_OF_NYQUIST: f64 = 0.92;
/// Below this residual the shift is treated as an exact integer copy, so
/// PPS-aligned tetra channels stay bit-identical.
const INTEGER_SHIFT_TOLERANCE_SAMPLES: f64 = 1.0e-9;
/// Series terms for the modified Bessel I0 used by the Kaiser window.
const BESSEL_I0_SERIES_TERMS: usize = 32;

/// Mapping from a node's local sample index to island time:
/// `t(n) = anchor_island_time_s + (n − anchor_sample_index) / (fs · rate_ratio)`.
/// `rate_ratio` = node sample rate / nominal (1 + drift); the sim's
/// `ClockTruth { offset_s, drift_ppm }` is `{0, offset_s, 1 + drift·1e-6}`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeClockModel {
    pub anchor_sample_index: f64,
    pub anchor_island_time_s: f64,
    pub rate_ratio: f64,
}

impl NodeClockModel {
    /// A disciplined clock whose sample 0 sits at island time 0.
    pub fn disciplined() -> Self {
        Self {
            anchor_sample_index: 0.0,
            anchor_island_time_s: 0.0,
            rate_ratio: 1.0,
        }
    }

    /// Same anchor, rate assumed nominal (for uncorrected A/B runs).
    pub fn without_rate_correction(self) -> Self {
        Self {
            rate_ratio: 1.0,
            ..self
        }
    }

    pub fn local_sample_index_at(&self, island_time_s: f64, nominal_sample_rate_hz: f64) -> f64 {
        self.anchor_sample_index
            + (island_time_s - self.anchor_island_time_s) * nominal_sample_rate_hz * self.rate_ratio
    }

    pub fn island_time_of_sample(&self, local_sample_index: f64, nominal_sample_rate_hz: f64) -> f64 {
        self.anchor_island_time_s
            + (local_sample_index - self.anchor_sample_index)
                / (nominal_sample_rate_hz * self.rate_ratio)
    }
}

/// One channel resampled onto `[window_start_s, window_start_s + len / fs)`.
#[derive(Clone, Debug)]
pub struct IslandTimelineChannel {
    pub samples: Vec<f32>,
    /// Fraction of output samples that fell inside the source clip.
    pub coverage_fraction: f64,
}

/// Resample `clip` (first sample at local index `clip_first_sample_index`)
/// onto the island timeline. Output samples outside the clip are zero.
pub fn resample_onto_island_timeline(
    clip: &[f32],
    clip_first_sample_index: i64,
    clock: &NodeClockModel,
    window_start_island_time_s: f64,
    output_length: usize,
    nominal_sample_rate_hz: f64,
) -> IslandTimelineChannel {
    let first_position = clock.local_sample_index_at(window_start_island_time_s, nominal_sample_rate_hz)
        - clip_first_sample_index as f64;
    let step = clock.rate_ratio;
    let mut samples = vec![0.0_f32; output_length];
    let mut covered = 0_usize;

    let is_integer_copy = (step - 1.0).abs() < f64::EPSILON
        && (first_position - first_position.round()).abs() < INTEGER_SHIFT_TOLERANCE_SAMPLES;
    if is_integer_copy {
        let offset = first_position.round() as i64;
        for (output_index, slot) in samples.iter_mut().enumerate() {
            let source_index = offset + output_index as i64;
            if source_index >= 0 && (source_index as usize) < clip.len() {
                *slot = clip[source_index as usize];
                covered += 1;
            }
        }
    } else {
        let table = sinc_table();
        let half_width = SINC_HALF_WIDTH_TAPS as i64;
        for (output_index, slot) in samples.iter_mut().enumerate() {
            let position = first_position + output_index as f64 * step;
            let base_index = position.floor();
            let fraction = position - base_index;
            let base_index = base_index as i64;
            if base_index < 0 || base_index >= clip.len() as i64 {
                continue;
            }
            covered += 1;
            let phase_row = &table[(fraction * SINC_PHASE_COUNT as f64).round() as usize];
            let first_tap_index = base_index - half_width + 1;
            let mut accumulator = 0.0_f32;
            for (tap, coefficient) in phase_row.iter().enumerate() {
                let source_index = first_tap_index + tap as i64;
                if source_index >= 0 && (source_index as usize) < clip.len() {
                    accumulator += clip[source_index as usize] * coefficient;
                }
            }
            *slot = accumulator;
        }
    }
    IslandTimelineChannel {
        samples,
        coverage_fraction: if output_length == 0 {
            0.0
        } else {
            covered as f64 / output_length as f64
        },
    }
}

/// Row p holds taps for fractional offset p / SINC_PHASE_COUNT; tap j weights
/// source sample `floor(position) − half_width + 1 + j`.
fn sinc_table() -> &'static Vec<Vec<f32>> {
    static TABLE: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let tap_count = 2 * SINC_HALF_WIDTH_TAPS;
        let half_width = SINC_HALF_WIDTH_TAPS as f64;
        let window_normalizer = bessel_i0(KAISER_BETA);
        (0..=SINC_PHASE_COUNT)
            .map(|phase| {
                let fraction = phase as f64 / SINC_PHASE_COUNT as f64;
                let mut row: Vec<f64> = (0..tap_count)
                    .map(|tap| {
                        let distance = (tap as f64 - half_width + 1.0) - fraction;
                        let normalized = distance / half_width;
                        if normalized.abs() >= 1.0 {
                            return 0.0;
                        }
                        let window = bessel_i0(KAISER_BETA * (1.0 - normalized * normalized).sqrt())
                            / window_normalizer;
                        let argument = std::f64::consts::PI * SINC_CUTOFF_FRACTION_OF_NYQUIST * distance;
                        let sinc = if argument.abs() < f64::EPSILON {
                            1.0
                        } else {
                            argument.sin() / argument
                        };
                        SINC_CUTOFF_FRACTION_OF_NYQUIST * sinc * window
                    })
                    .collect();
                // Unit DC gain per phase so resampling never modulates level.
                let sum: f64 = row.iter().sum();
                row.iter_mut().for_each(|value| *value /= sum);
                row.into_iter().map(|value| value as f32).collect()
            })
            .collect()
    })
}

fn bessel_i0(x: f64) -> f64 {
    let quarter_x_squared = 0.25 * x * x;
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..BESSEL_I0_SERIES_TERMS {
        term *= quarter_x_squared / (k * k) as f64;
        sum += term;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE_HZ: f64 = 48_000.0;

    fn tone(frequency_hz: f64, time_of_sample: impl Fn(usize) -> f64, length: usize) -> Vec<f32> {
        (0..length)
            .map(|n| (std::f64::consts::TAU * frequency_hz * time_of_sample(n)).sin() as f32)
            .collect()
    }

    #[test]
    fn rate_and_offset_mapping_recovers_island_time_tone() {
        let drift_ratio = 1.0 + 60.0e-6;
        let clock = NodeClockModel {
            anchor_sample_index: 0.0,
            anchor_island_time_s: 0.0123,
            rate_ratio: drift_ratio,
        };
        let length = 48_000 * 3;
        let tone_hz = 3_000.0;
        let clip = tone(tone_hz, |n| clock.island_time_of_sample(n as f64, RATE_HZ), length);
        let window_start_s = 2.0;
        let output = resample_onto_island_timeline(&clip, 0, &clock, window_start_s, 4_800, RATE_HZ);
        assert!((output.coverage_fraction - 1.0).abs() < 1e-12);
        let worst = output
            .samples
            .iter()
            .enumerate()
            .map(|(k, value)| {
                let expected = (std::f64::consts::TAU * tone_hz * (window_start_s + k as f64 / RATE_HZ)).sin();
                (*value as f64 - expected).abs()
            })
            .fold(0.0_f64, f64::max);
        assert!(worst < 2.0e-3, "worst resampling error {worst}");
    }

    #[test]
    fn integer_aligned_disciplined_clip_is_copied_exactly() {
        let clip: Vec<f32> = (0..1000).map(|n| (n as f32 * 0.37).sin()).collect();
        let clock = NodeClockModel::disciplined();
        let output = resample_onto_island_timeline(&clip, 100, &clock, 150.0 / RATE_HZ, 200, RATE_HZ);
        assert_eq!(&output.samples[..], &clip[50..250]);
    }
}
