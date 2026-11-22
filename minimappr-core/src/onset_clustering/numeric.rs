//! NumPy-compatible f64 helpers for the onset-clustering port.
//!
//! Python (`minimappr/core/onset_*.py`) is the parity reference, so these
//! reproduce NumPy's exact formulas (`np.hanning`'s cosine form, linear
//! `np.percentile`, `np.interp`'s exact-knot rule, `np.geomspace` end pinning,
//! `np.arange` length, `irfft` / analytic-signal conventions) rather than the
//! textbook equivalents, which differ in the last bits and can flip argmax or
//! threshold decisions downstream.

use std::sync::Arc;

use num_complex::Complex64;
use rustfft::{Fft, FftPlanner};

/// Python's `int(round(x))`: round half to even.
pub fn python_round_to_i64(value: f64) -> i64 {
    value.round_ties_even() as i64
}

/// Python floor division for a positive divisor.
pub fn floor_divide(numerator: i64, divisor: i64) -> i64 {
    numerator.div_euclid(divisor)
}

/// `np.arange(start, stop, step)` (float): `ceil((stop - start) / step)` values.
pub fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let count = ((stop - start) / step).ceil().max(0.0) as usize;
    (0..count)
        .map(|index| start + index as f64 * step)
        .collect()
}

/// `np.linspace(start, stop, count)` with the endpoint pinned.
pub fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    match count {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let step = (stop - start) / (count - 1) as f64;
            let mut values: Vec<f64> = (0..count)
                .map(|index| index as f64 * step + start)
                .collect();
            values[count - 1] = stop;
            values
        }
    }
}

/// `np.geomspace(start, stop, count)` (positive endpoints, pinned exactly).
pub fn geomspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    let exponents = linspace(start.log10(), stop.log10(), count);
    let mut values: Vec<f64> = exponents
        .iter()
        .map(|exponent| 10f64.powf(*exponent))
        .collect();
    if let Some(first) = values.first_mut() {
        *first = start;
    }
    if count > 1 {
        values[count - 1] = stop;
    }
    values
}

/// `np.hanning(length)`: `0.5 + 0.5 cos(pi n / (M - 1))`, n = 1-M, 3-M, ..., M-1.
pub fn numpy_hanning(length: usize) -> Vec<f64> {
    match length {
        0 => Vec::new(),
        1 => vec![1.0],
        _ => {
            let denominator = (length - 1) as f64;
            (0..length)
                .map(|index| {
                    let n = 1.0 - length as f64 + 2.0 * index as f64;
                    0.5 + 0.5 * (std::f64::consts::PI * n / denominator).cos()
                })
                .collect()
        }
    }
}

/// Linear-interpolated percentile of unsorted values (`np.percentile`, default method).
pub fn percentile(values: &[f64], percent: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    percentile_of_sorted(&sorted, percent)
}

pub fn percentile_of_sorted(sorted: &[f64], percent: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let virtual_index = percent / 100.0 * (sorted.len() - 1) as f64;
    let lower = virtual_index.floor() as usize;
    let upper = (lower + 1).min(sorted.len() - 1);
    let fraction = virtual_index - lower as f64;
    let below = sorted[lower];
    let above = sorted[upper];
    // NumPy's `_lerp`: switch formula at t >= 0.5 for monotonic rounding.
    let difference = above - below;
    if fraction >= 0.5 {
        above - difference * (1.0 - fraction)
    } else {
        below + difference * fraction
    }
}

pub fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let count = sorted.len();
    if count == 0 {
        return f64::NAN;
    }
    if count % 2 == 1 {
        sorted[count / 2]
    } else {
        // np.median takes the mean of the two middle values.
        (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
    }
}

pub fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Population standard deviation (`np.std`, ddof 0).
pub fn population_std(values: &[f64]) -> f64 {
    let average = mean(values);
    (values
        .iter()
        .map(|value| (value - average).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

/// Index of the first maximum (`np.argmax`; NaN-free inputs).
pub fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

/// `np.interp(x, xp, fp, left, right)` for increasing `xp`.
pub fn interp(x: f64, xp: &[f64], fp: &[f64], left: f64, right: f64) -> f64 {
    let last = xp.len() - 1;
    if x.is_nan() {
        return x;
    }
    if x < xp[0] {
        return left;
    }
    if x > xp[last] {
        return right;
    }
    if x == xp[last] {
        return fp[last];
    }
    // Largest j with xp[j] <= x (x < xp[last] here).
    let j = xp.partition_point(|knot| *knot <= x) - 1;
    if xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    slope * (x - xp[j]) + fp[j]
}

/// `np.interp` with edge values held outside the knots.
pub fn interp_clamped(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    interp(x, xp, fp, fp[0], fp[fp.len() - 1])
}

/// `np.fft.rfftfreq(n, 1 / sample_rate)`.
pub fn rfft_frequencies(length: usize, sample_rate_hz: f64) -> Vec<f64> {
    let spacing = 1.0 / (length as f64 * (1.0 / sample_rate_hz));
    (0..length / 2 + 1)
        .map(|bin| bin as f64 * spacing)
        .collect()
}

/// Smallest power of two `>= value` as NumPy computes it (`2 ** ceil(log2(value))`).
pub fn power_of_two_at_least(value: usize) -> usize {
    2usize.pow((value as f64).log2().ceil() as u32)
}

/// FFT plans shared by one clustering call (sans-IO: owned by the caller's stack).
pub struct SpectralTransforms {
    planner: FftPlanner<f64>,
}

impl Default for SpectralTransforms {
    fn default() -> Self {
        Self {
            planner: FftPlanner::new(),
        }
    }
}

impl SpectralTransforms {
    fn forward(&mut self, length: usize) -> Arc<dyn Fft<f64>> {
        self.planner.plan_fft_forward(length)
    }

    fn inverse(&mut self, length: usize) -> Arc<dyn Fft<f64>> {
        self.planner.plan_fft_inverse(length)
    }

    /// `np.fft.rfft(signal, length)` (zero-padded or truncated to `length`).
    pub fn rfft(&mut self, signal: &[f64], length: usize) -> Vec<Complex64> {
        let mut buffer: Vec<Complex64> = (0..length)
            .map(|index| Complex64::new(signal.get(index).copied().unwrap_or(0.0), 0.0))
            .collect();
        self.forward(length).process(&mut buffer);
        buffer.truncate(length / 2 + 1);
        buffer
    }

    /// `np.fft.irfft(spectrum, length)` (backward normalisation).
    pub fn irfft(&mut self, spectrum: &[Complex64], length: usize) -> Vec<f64> {
        let half = length / 2;
        let mut buffer = vec![Complex64::new(0.0, 0.0); length];
        for bin in 0..=half {
            if let Some(value) = spectrum.get(bin) {
                buffer[bin] = *value;
            }
        }
        // c2r ignores the imaginary part of DC and (even length) Nyquist.
        buffer[0].im = 0.0;
        if length % 2 == 0 {
            buffer[half].im = 0.0;
        }
        let upper = if length % 2 == 0 { half } else { half + 1 };
        for bin in 1..upper {
            buffer[length - bin] = buffer[bin].conj();
        }
        self.inverse(length).process(&mut buffer);
        buffer
            .iter()
            .map(|value| value.re / length as f64)
            .collect()
    }

    /// `np.fft.ifft(full)` magnitudes (backward normalisation).
    pub fn ifft_magnitudes(&mut self, mut full: Vec<Complex64>) -> Vec<f64> {
        let length = full.len();
        self.inverse(length).process(&mut full);
        full.iter()
            .map(|value| (value / length as f64).norm())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_matches_numpy_linear_rule() {
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 20.0), 1.6);
        assert_eq!(median(&[3.0, 1.0, 2.0, 4.0]), 2.5);
    }

    #[test]
    fn interp_holds_edges_and_hits_knots() {
        let xp = [0.0, 1.0, 2.0];
        let fp = [0.0, 10.0, 20.0];
        assert_eq!(interp(-1.0, &xp, &fp, 0.0, 0.0), 0.0);
        assert_eq!(interp(2.0, &xp, &fp, 0.0, 0.0), 20.0);
        assert_eq!(interp(0.5, &xp, &fp, 0.0, 0.0), 5.0);
        assert_eq!(interp_clamped(5.0, &xp, &fp), 20.0);
    }

    #[test]
    fn irfft_inverts_rfft() {
        let mut transforms = SpectralTransforms::default();
        let signal = [1.0, -2.0, 0.5, 3.0, 0.0, 1.5];
        let spectrum = transforms.rfft(&signal, 8);
        let restored = transforms.irfft(&spectrum, 8);
        for (index, value) in signal.iter().enumerate() {
            assert!((restored[index] - value).abs() < 1e-12);
        }
    }

    #[test]
    fn arange_and_geomspace_match_numpy_lengths() {
        assert_eq!(arange(0.0, 360.0, 2.0).len(), 180);
        assert_eq!(arange(-30.0, 60.0 + 1e-12, 10.0).len(), 10);
        let rings = geomspace(1.0, 300.0, 40);
        assert_eq!(rings[0], 1.0);
        assert_eq!(rings[39], 300.0);
    }
}
