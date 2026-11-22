//! Band-limited GCC-PHAT coherence curves for arbitrary sensor pairs, plus a
//! max-pooled pyramid so a coarse cell can ask "the best this pair offers
//! anywhere in my TDOA interval" in O(1).
//!
//! Why the pyramid: a band-limited PHAT peak is ~0.1–0.2 ms wide, while a
//! 2 m cell spans up to ~6 ms of TDOA at the layout's edge. Sampling the
//! curve at the cell centre misses sharp peaks entirely; taking the maximum
//! over the cell's interval makes the coarse score an upper envelope.
//!
//! Curves hold absolute coherence c(τ) = (1/K) Σ Re{Wₖ e^{jωτ}} over the K
//! whitened in-band bins, so perfect coherence is 1 and a noise-only pair
//! sits near 1/√K everywhere (contributing a constant, not a bias).

use num_complex::Complex32;
use rustfft::FftPlanner;

use super::config::PairCorrelationConfig;

/// Sign convention (matches `gcc_phat.rs`): lag τ > 0 means sensor a hears
/// the sound later than sensor b, i.e. τ_ab = (|x − a| − |x − b|) / c.
#[derive(Clone, Debug)]
pub struct PairCoherenceCurve {
    pub sensor_a: usize,
    pub sensor_b: usize,
    /// Lag spacing of `values` (seconds).
    pub lag_step_s: f64,
    /// Index of zero lag in `values`.
    pub zero_lag_index: usize,
    pub values: Vec<f32>,
    /// `segment_maxima[i]`: max of the interpolant over [i, i + 1] steps.
    segment_maxima: Vec<f32>,
    /// `segment_maxima_pyramid[k−1]` is level k: stride 2^(k−1), window
    /// half-width 2^k − 1 segments.
    segment_maxima_pyramid: Vec<Vec<f32>>,
    /// Whitened in-band bin count K (both spectrum halves).
    pub in_band_bin_count: usize,
    /// Peak coherence over the lag range.
    pub peak_coherence: f32,
}

/// Intervals spanning fewer segments than this are scanned directly; a
/// pyramid lookup only pays off for wider ones.
const DIRECT_SCAN_SEGMENT_COUNT: usize = 8;

impl PairCoherenceCurve {
    /// Build a curve (and its envelope tables) from sampled coherence values.
    pub fn from_values(
        sensor_a: usize,
        sensor_b: usize,
        lag_step_s: f64,
        zero_lag_index: usize,
        values: Vec<f32>,
        in_band_bin_count: usize,
    ) -> Self {
        let peak_coherence = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let segment_maxima = catmull_rom_segment_maxima(&values);
        Self {
            sensor_a,
            sensor_b,
            lag_step_s,
            zero_lag_index,
            segment_maxima_pyramid: build_max_pyramid(&segment_maxima),
            segment_maxima,
            values,
            in_band_bin_count,
            peak_coherence,
        }
    }

    fn full_resolution_position(&self, lag_s: f64) -> f64 {
        lag_s / self.lag_step_s + self.zero_lag_index as f64
    }

    /// Catmull-Rom interpolated coherence at `lag_s` (clamped to range).
    pub fn coherence_at(&self, lag_s: f64) -> f32 {
        let last = self.values.len() - 1;
        let position = self.full_resolution_position(lag_s).clamp(0.0, last as f64);
        let base = position.floor() as usize;
        let t = (position - base as f64) as f32;
        let sample = |offset: isize| -> f32 {
            let index = (base as isize + offset).clamp(0, last as isize) as usize;
            self.values[index]
        };
        let (p0, p1, p2, p3) = (sample(-1), sample(0), sample(1), sample(2));
        let t2 = t * t;
        let t3 = t2 * t;
        0.5 * ((2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
    }

    /// Upper envelope: max of the interpolated curve (`coherence_at`) over
    /// [center − half_width, center + half_width]. Exact up to f32 rounding:
    /// it pools per-segment maxima of the Catmull-Rom cubic, which can
    /// overshoot the stored samples near a peak. Bounding raw samples instead
    /// would let a cell's envelope fall below its own exact score, so
    /// branch-and-bound could prune the cell holding the true source.
    pub fn max_coherence_over_interval(&self, center_lag_s: f64, half_width_s: f64) -> f32 {
        let half_width_s = half_width_s.max(0.0);
        let last_segment = self.segment_maxima.len() - 1;
        let segment_of = |lag_s: f64| {
            let position = self.full_resolution_position(lag_s);
            if position.is_nan() {
                return 0;
            }
            position.floor().clamp(0.0, last_segment as f64) as usize
        };
        let low = segment_of(center_lag_s - half_width_s);
        let high = segment_of(center_lag_s + half_width_s).max(low);
        if high - low < DIRECT_SCAN_SEGMENT_COUNT {
            return self.segment_maxima[low..=high].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        }
        // Level k (stride s = 2^(k−1)) stores, at index j, the max over
        // segments [j·s − (2^k − 1), j·s + (2^k − 1)].
        let middle = (low + high) / 2;
        for (level_offset, level) in self.segment_maxima_pyramid.iter().enumerate() {
            let level_number = level_offset + 1;
            let stride = 1_usize << (level_number - 1);
            let half_window = (1_usize << level_number) - 1;
            let index = ((middle + stride / 2) / stride).min(level.len() - 1);
            let covered_center = index * stride;
            if covered_center.saturating_sub(half_window) <= low && covered_center + half_window >= high {
                return level[index];
            }
        }
        self.segment_maxima.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Strongest local peak within `half_window_s` of `predicted_lag_s`,
    /// parabolically refined: (lag seconds, coherence).
    pub fn local_peak_near(&self, predicted_lag_s: f64, half_window_s: f64) -> (f64, f32) {
        let last = self.values.len() - 1;
        let low = self
            .full_resolution_position(predicted_lag_s - half_window_s)
            .floor()
            .clamp(0.0, last as f64) as usize;
        let high = self
            .full_resolution_position(predicted_lag_s + half_window_s)
            .ceil()
            .clamp(0.0, last as f64) as usize;
        let (best_index, best_value) = (low..=high)
            .map(|index| (index, self.values[index]))
            .fold((low, f32::NEG_INFINITY), |best, candidate| {
                if candidate.1 > best.1 {
                    candidate
                } else {
                    best
                }
            });
        let mut offset = 0.0_f64;
        if best_index > 0 && best_index < last {
            let (left, center, right) = (
                self.values[best_index - 1] as f64,
                best_value as f64,
                self.values[best_index + 1] as f64,
            );
            let denominator = left - 2.0 * center + right;
            if denominator < 0.0 {
                offset = (0.5 * (left - right) / denominator).clamp(-0.5, 0.5);
            }
        }
        let lag_s = (best_index as f64 + offset - self.zero_lag_index as f64) * self.lag_step_s;
        (lag_s, best_value)
    }
}

/// What a pair is, before audio: which sensors and how far its lags can go.
#[derive(Clone, Copy, Debug)]
pub struct SensorPairSpec {
    pub sensor_a: usize,
    pub sensor_b: usize,
    pub maximum_lag_s: f64,
}

/// Compute coherence curves for `pairs` over equal-length island-aligned
/// channels (one per sensor).
pub fn compute_pair_coherence_curves(
    aligned_channels: &[Vec<f32>],
    pairs: &[SensorPairSpec],
    sample_rate_hz: u32,
    config: &PairCorrelationConfig,
) -> Vec<PairCoherenceCurve> {
    let sample_count = aligned_channels.iter().map(Vec::len).min().unwrap_or(0);
    let longest_lag_samples = pairs
        .iter()
        .map(|pair| (pair.maximum_lag_s * sample_rate_hz as f64).ceil() as usize)
        .max()
        .unwrap_or(0);
    // Linear (not circular) correlation over every lag we will read.
    let fft_length = (sample_count + longest_lag_samples + 1).next_power_of_two();
    let upsample_factor = config.upsample_factor.max(1);
    let upsampled_length = fft_length * upsample_factor;

    let mut planner = FftPlanner::<f32>::new();
    let forward = planner.plan_fft_forward(fft_length);
    let inverse = planner.plan_fft_inverse(upsampled_length);

    let spectra: Vec<Vec<Complex32>> = aligned_channels
        .iter()
        .map(|channel| {
            let mut buffer: Vec<Complex32> = channel[..sample_count]
                .iter()
                .map(|&sample| Complex32::new(sample, 0.0))
                .chain(std::iter::repeat_n(Complex32::new(0.0, 0.0), fft_length - sample_count))
                .collect();
            forward.process(&mut buffer);
            buffer
        })
        .collect();

    let nyquist_hz = sample_rate_hz as f32 * 0.5;
    let bin_hz = sample_rate_hz as f32 / fft_length as f32;
    let low_bin = ((config.band_hz[0].max(0.0) / bin_hz).ceil() as usize).max(1);
    let high_bin = ((config.band_hz[1].min(nyquist_hz) / bin_hz).floor() as usize).min(fft_length / 2 - 1);

    pairs
        .iter()
        .map(|pair| {
            let spectrum_a = &spectra[pair.sensor_a];
            let spectrum_b = &spectra[pair.sensor_b];
            let strongest = (low_bin..=high_bin)
                .map(|bin| (spectrum_a[bin] * spectrum_b[bin].conj()).norm())
                .fold(0.0_f32, f32::max);
            let floor = strongest * config.whitening_floor_fraction;
            let mut whitened = vec![Complex32::new(0.0, 0.0); upsampled_length];
            let mut positive_bins = 0_usize;
            for bin in low_bin..=high_bin {
                let cross = spectrum_a[bin] * spectrum_b[bin].conj();
                let magnitude = cross.norm();
                if magnitude <= floor || magnitude == 0.0 {
                    continue;
                }
                let phasor = cross / magnitude;
                // Zero-padded Hermitian placement = band-limited upsampling.
                whitened[bin] = phasor;
                whitened[upsampled_length - bin] = phasor.conj();
                positive_bins += 1;
            }
            inverse.process(&mut whitened);
            let in_band_bin_count = 2 * positive_bins;
            let normalizer = 1.0 / in_band_bin_count.max(1) as f32;
            let lag_step_s = 1.0 / (sample_rate_hz as f64 * upsample_factor as f64);
            let lag_steps = ((pair.maximum_lag_s / lag_step_s).ceil() as usize)
                .clamp(1, upsampled_length / 2 - 1);
            let values: Vec<f32> = (0..=2 * lag_steps)
                .map(|index| {
                    let signed = index as isize - lag_steps as isize;
                    let wrapped = signed.rem_euclid(upsampled_length as isize) as usize;
                    whitened[wrapped].re * normalizer
                })
                .collect();
            PairCoherenceCurve::from_values(
                pair.sensor_a,
                pair.sensor_b,
                lag_step_s,
                lag_steps,
                values,
                in_band_bin_count,
            )
        })
        .collect()
}

/// Max of the Catmull-Rom cubic `coherence_at` uses on each segment
/// [i, i + 1] (endpoints plus interior stationary points), clamping the
/// neighbour samples at the curve ends exactly as `coherence_at` does.
fn catmull_rom_segment_maxima(values: &[f32]) -> Vec<f32> {
    if values.len() < 2 {
        return values.to_vec();
    }
    let last = values.len() - 1;
    let sample = |index: isize| values[index.clamp(0, last as isize) as usize] as f64;
    (0..last)
        .map(|segment| {
            let base = segment as isize;
            let (p0, p1, p2, p3) = (sample(base - 1), sample(base), sample(base + 1), sample(base + 2));
            // f(t) = 0.5 (2 p1 + c1 t + c2 t² + c3 t³), same form as `coherence_at`.
            let (c1, c2, c3) = (p2 - p0, 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3, -p0 + 3.0 * p1 - 3.0 * p2 + p3);
            let cubic = |t: f64| 0.5 * (2.0 * p1 + c1 * t + c2 * t * t + c3 * t * t * t);
            let mut best = p1.max(p2);
            // f'(t) ∝ c1 + 2 c2 t + 3 c3 t².
            let (a, b, c) = (3.0 * c3, 2.0 * c2, c1);
            let mut consider = |t: f64| {
                if t > 0.0 && t < 1.0 {
                    best = best.max(cubic(t));
                }
            };
            if a.abs() > f64::EPSILON {
                let discriminant = b * b - 4.0 * a * c;
                if discriminant >= 0.0 {
                    let root = discriminant.sqrt();
                    consider((-b + root) / (2.0 * a));
                    consider((-b - root) / (2.0 * a));
                }
            } else if b.abs() > f64::EPSILON {
                consider(-c / b);
            }
            // Pad by a few f32 ulps: `coherence_at` evaluates in f32, so
            // its rounding can sit slightly above the f64 maximum.
            let stored = best as f32;
            stored + ENVELOPE_ROUNDING_PAD_ULPS * f32::EPSILON * stored.abs().max(1.0)
        })
        .collect()
}

/// f32 ulps added to each segment maximum (see `catmull_rom_segment_maxima`).
const ENVELOPE_ROUNDING_PAD_ULPS: f32 = 8.0;

/// Dyadic max-pool levels until one window spans the whole curve.
fn build_max_pyramid(values: &[f32]) -> Vec<Vec<f32>> {
    let length = values.len();
    let mut levels = Vec::new();
    let mut previous = values.to_vec();
    let mut level_number = 1_usize;
    loop {
        let offset = 1_usize << (level_number - 1);
        let current: Vec<f32> = (0..length)
            .map(|index| {
                let mut best = previous[index];
                if index >= offset {
                    best = best.max(previous[index - offset]);
                }
                if index + offset < length {
                    best = best.max(previous[index + offset]);
                }
                best
            })
            .collect();
        let stride = offset;
        levels.push(current.iter().step_by(stride).copied().collect());
        let half_window = (1_usize << level_number) - 1;
        if half_window >= length {
            break;
        }
        previous = current;
        level_number += 1;
    }
    levels
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_noise(length: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..length)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect()
    }

    #[test]
    fn integer_delay_peaks_at_expected_lag_with_unit_coherence() {
        let rate = 48_000;
        let source = deterministic_noise(9_000, 7);
        let delay = 37_usize;
        let sensor_b: Vec<f32> = source[delay..delay + 8_000].to_vec();
        let sensor_a: Vec<f32> = source[..8_000].to_vec();
        // a = source(t), b = source(t + delay) → b leads a → τ_ab > 0.
        let curves = compute_pair_coherence_curves(
            &[sensor_a, sensor_b],
            &[SensorPairSpec { sensor_a: 0, sensor_b: 1, maximum_lag_s: 0.002 }],
            rate,
            &PairCorrelationConfig::default(),
        );
        let (lag_s, peak) = curves[0].local_peak_near(0.0, 0.002);
        assert!((lag_s * rate as f64 - delay as f64).abs() < 0.05, "lag {}", lag_s * rate as f64);
        assert!(peak > 0.8, "peak coherence {peak}");
    }


    /// Dense probes of `coherence_at` across the interval: the envelope must
    /// bound every one (admissibility of branch-and-bound pruning).
    fn assert_envelope_bounds_interpolant(curve: &PairCoherenceCurve, center: f64, half_width: f64) {
        let envelope = curve.max_coherence_over_interval(center, half_width);
        let probe_count = 64 + (8.0 * half_width) as usize;
        for probe in 0..=probe_count {
            let lag = center - half_width + 2.0 * half_width * probe as f64 / probe_count as f64;
            let exact = curve.coherence_at(lag);
            assert!(envelope >= exact, "center {center} hw {half_width} lag {lag}: {envelope} < {exact}");
        }
    }

    #[test]
    fn interval_envelope_never_undercuts_interpolated_curve() {
        let noise = PairCoherenceCurve::from_values(0, 1, 1.0, 2_000, deterministic_noise(4_001, 3), 1);
        for (center, half_width) in [(0.0_f64, 3.0_f64), (100.4, 17.2), (-1500.0, 300.0), (900.7, 1.5), (3.3, 0.2)] {
            assert_envelope_bounds_interpolant(&noise, center, half_width);
        }
        // A sharp peak between samples: Catmull-Rom overshoots both stored
        // neighbours, so a sample-only bound (the old one) undercut it.
        let peak_values: Vec<f32> = (0..41).map(|index| (-((index as f32 - 20.3) / 1.2).powi(2)).exp()).collect();
        let peak = PairCoherenceCurve::from_values(0, 1, 1.0, 20, peak_values, 1);
        for center_tenths in -30..=30 {
            for half_width in [0.0_f64, 0.05, 0.3, 0.6, 0.95, 4.0, 12.5] {
                assert_envelope_bounds_interpolant(&peak, center_tenths as f64 * 0.1, half_width);
            }
        }
    }

    #[test]
    fn interval_envelope_is_tight_on_a_wide_interval_away_from_the_peak() {
        let values: Vec<f32> = (0..201).map(|index| (-((index as f32 - 150.0) / 2.0).powi(2)).exp()).collect();
        let curve = PairCoherenceCurve::from_values(0, 1, 1.0, 100, values, 1);
        // Interval [-80, -20] relative to zero lag never reaches the peak at +50.
        assert!(curve.max_coherence_over_interval(-50.0, 30.0) < 1.0e-3);
        assert!(curve.max_coherence_over_interval(50.0, 30.0) > 0.99);
    }
}
