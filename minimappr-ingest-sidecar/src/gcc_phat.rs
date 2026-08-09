use std::cmp::Ordering;

use num_complex::Complex32;
use rustfft::FftPlanner;

/// TDOA result from one microphone pair.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TdoaResult {
    /// Fractional sample delay (positive = ch2 leads ch1).
    pub delay_samples: f32,
    /// Delay converted to seconds.
    pub lag_seconds: f32,
    /// Confidence: peak-to-sidelobe ratio in [0.0, 1.0].
    pub confidence: f32,
    /// Absolute GCC-PHAT peak magnitude used in SRP confidence scoring.
    #[serde(default)]
    pub peak_value: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct GccPhatCorrelation {
    pub lags_seconds: Vec<f32>,
    pub magnitudes: Vec<f32>,
    pub tdoa: TdoaResult,
}

/// Compute the GCC-PHAT inter-channel time delay for one microphone pair.
///
/// `max_lag_samples` bounds the search window; for the tetrahedral array
/// (50 mm spacing, 340 m/s, 16 kHz) use 4.
#[allow(dead_code)]
pub fn gcc_phat(
    ch1: &[f32],
    ch2: &[f32],
    sample_rate_hz: u32,
    max_lag_samples: usize,
) -> TdoaResult {
    phat_correlation(
        ch1,
        ch2,
        sample_rate_hz,
        max_lag_samples as f32 / sample_rate_hz.max(1) as f32,
        None,
        4,
    )
    .tdoa
}

pub(crate) fn phat_correlation(
    ch1: &[f32],
    ch2: &[f32],
    sample_rate_hz: u32,
    max_tau_s: f32,
    band_hz: Option<[f32; 2]>,
    interp_factor: usize,
) -> GccPhatCorrelation {
    const EPSILON: f32 = 1e-9;

    let n = ch1.len().min(ch2.len());
    if n == 0 || sample_rate_hz == 0 {
        return GccPhatCorrelation {
            lags_seconds: vec![0.0],
            magnitudes: vec![0.0],
            tdoa: TdoaResult {
                delay_samples: 0.0,
                lag_seconds: 0.0,
                confidence: 0.0,
                peak_value: 0.0,
            },
        };
    }

    let fft_len = next_pow2(2 * n.max(1));

    thread_local! {
        static PLANNER: std::cell::RefCell<FftPlanner<f32>> = std::cell::RefCell::new(FftPlanner::new());
    }

    let fft = PLANNER.with(|planner| {
        let mut p = planner.borrow_mut();
        p.plan_fft_forward(fft_len)
    });

    let mut x1: Vec<Complex32> = ch1[..n]
        .iter()
        .map(|&sample| Complex32::new(sample, 0.0))
        .chain(std::iter::repeat_n(Complex32::new(0.0, 0.0), fft_len - n))
        .collect();
    let mut x2: Vec<Complex32> = ch2[..n]
        .iter()
        .map(|&sample| Complex32::new(sample, 0.0))
        .chain(std::iter::repeat_n(Complex32::new(0.0, 0.0), fft_len - n))
        .collect();

    fft.process(&mut x1);
    fft.process(&mut x2);

    let effective_band = normalize_band_hz(band_hz, sample_rate_hz);
    let cross_spectrum: Vec<Complex32> = x1
        .iter()
        .zip(x2.iter())
        .enumerate()
        .map(|(bin, (a, b))| {
            if !bin_in_band(bin, fft_len, sample_rate_hz, effective_band) {
                return Complex32::new(0.0, 0.0);
            }
            let cross = a * b.conj();
            let magnitude = cross.norm();
            if magnitude <= EPSILON {
                Complex32::new(0.0, 0.0)
            } else {
                cross / magnitude
            }
        })
        .collect();

    let interp_factor = interp_factor.max(1);
    let upsampled_fft_len = fft_len.saturating_mul(interp_factor);
    let mut cross_correlation = zero_pad_hermitian_spectrum(&cross_spectrum, upsampled_fft_len);
    let ifft = PLANNER.with(|planner| planner.borrow_mut().plan_fft_inverse(upsampled_fft_len));
    ifft.process(&mut cross_correlation);

    let scale = 1.0 / upsampled_fft_len as f32;
    let lag = ((max_tau_s.max(1.0 / sample_rate_hz as f32))
        * sample_rate_hz as f32
        * interp_factor as f32)
        .ceil() as usize;
    let lag = lag.clamp(1, upsampled_fft_len / 2);

    let mut lags_seconds = Vec::with_capacity((2 * lag) + 1);
    let mut magnitudes = Vec::with_capacity((2 * lag) + 1);

    for delay in (1..=lag).rev() {
        lags_seconds.push(-(delay as f32) / (sample_rate_hz as f32 * interp_factor as f32));
        magnitudes.push((cross_correlation[upsampled_fft_len - delay].re * scale).abs());
    }
    lags_seconds.push(0.0);
    magnitudes.push((cross_correlation[0].re * scale).abs());
    for (delay, cross_val) in cross_correlation.iter().enumerate().skip(1).take(lag) {
        lags_seconds.push(delay as f32 / (sample_rate_hz as f32 * interp_factor as f32));
        magnitudes.push((cross_val.re * scale).abs());
    }

    let (peak_index, peak_value) = magnitudes
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.partial_cmp(right).unwrap_or(Ordering::Equal))
        .unwrap_or((lag, 0.0));
    let fractional_peak_offset = if peak_index > 0 && peak_index + 1 < magnitudes.len() {
        let left_peak = magnitudes[peak_index - 1];
        let center_peak = magnitudes[peak_index];
        let right_peak = magnitudes[peak_index + 1];
        let denominator = left_peak - (2.0 * center_peak) + right_peak;
        if denominator.abs() > 1.0e-20 {
            (0.5 * (left_peak - right_peak) / denominator).clamp(-0.5, 0.5)
        } else {
            0.0
        }
    } else {
        0.0
    };
    let mut delay_samples =
        (peak_index as f32 + fractional_peak_offset - lag as f32) / interp_factor as f32;

    // Phase-slope refinement — more accurate than parabolic interpolation for
    // compact arrays where the inter-element delay is a small fraction of one
    // sample (e.g. the 5 cm Sirith tetrahedral). Mirrors the Python estimator in
    // `minimappr/core/localization.py::_phase_slope_tau`. Restricted to
    // |τ| ≤ 200 μs (covers a 5 cm tetra; excludes larger baselines where deep
    // phase wrapping degrades the fit) and only accepted when it agrees with the
    // parabolic estimate to within one sample.
    const PHASE_SLOPE_MAX_TAU_S: f32 = 200e-6;
    let parabolic_tau_s = delay_samples / sample_rate_hz as f32;
    if parabolic_tau_s.abs() <= PHASE_SLOPE_MAX_TAU_S {
        if let Some(phase_tau_s) =
            phase_slope_tau(&x1, &x2, fft_len, sample_rate_hz, effective_band, max_tau_s)
        {
            if (phase_tau_s - parabolic_tau_s).abs() < 1.0 / sample_rate_hz as f32 {
                delay_samples = phase_tau_s * sample_rate_hz as f32;
            }
        }
    }

    let mut sorted_peaks = magnitudes.clone();
    sorted_peaks.sort_by(|left, right| right.partial_cmp(left).unwrap_or(Ordering::Equal));
    let sidelobe_count = sorted_peaks.len().saturating_sub(1).min(3);
    let sidelobe_mean = if sidelobe_count > 0 {
        sorted_peaks[1..=sidelobe_count].iter().sum::<f32>() / sidelobe_count as f32
    } else {
        0.0
    };
    let confidence = if peak_value > 0.0 && sidelobe_mean > 0.0 {
        (peak_value / (peak_value + sidelobe_mean)).clamp(0.0, 1.0)
    } else if peak_value > 0.0 {
        1.0
    } else {
        0.0
    };

    GccPhatCorrelation {
        lags_seconds,
        magnitudes,
        tdoa: TdoaResult {
            delay_samples,
            lag_seconds: delay_samples / sample_rate_hz as f32,
            confidence,
            peak_value,
        },
    }
}

/// Frequency-domain phase-slope delay estimator. Fits a linear slope to the
/// unwrapped phase of the *raw* cross-spectrum (`x1·conj(x2)`) over in-band bins
/// that carry meaningful energy. Mirrors the Python `_phase_slope_tau` in
/// `minimappr/core/localization.py`.
///
/// NOTE: this operates on the raw cross product, not the PHAT-normalised
/// spectrum used for the IFFT — the magnitude mask needs the true signal energy
/// per bin. Returns `None` when the delay is too large for reliable unwrapping
/// (dense wrapping) or the spectrum lacks broadband energy.
fn phase_slope_tau(
    x1: &[Complex32],
    x2: &[Complex32],
    fft_len: usize,
    sample_rate_hz: u32,
    band_hz: Option<[f32; 2]>,
    max_tau_s: f32,
) -> Option<f32> {
    if fft_len == 0 || sample_rate_hz == 0 {
        return None;
    }
    let half = (fft_len / 2).min(x1.len().min(x2.len()).saturating_sub(1));
    let mut freqs: Vec<f32> = Vec::with_capacity(half + 1);
    let mut magnitudes: Vec<f32> = Vec::with_capacity(half + 1);
    let mut phases: Vec<f32> = Vec::with_capacity(half + 1);
    for bin in 0..=half {
        if !bin_in_band(bin, fft_len, sample_rate_hz, band_hz) {
            continue;
        }
        let cross = x1[bin] * x2[bin].conj();
        freqs.push(bin as f32 * sample_rate_hz as f32 / fft_len as f32);
        magnitudes.push(cross.norm());
        phases.push(cross.arg());
    }

    let peak_magnitude = magnitudes.iter().copied().fold(0.0_f32, f32::max);
    if peak_magnitude < 1e-30 {
        return None;
    }
    // Restrict to frequency bins with meaningful cross-spectrum energy.
    let threshold = 0.05 * peak_magnitude;
    let mut selected_freqs: Vec<f32> = Vec::new();
    let mut selected_phases: Vec<f32> = Vec::new();
    for index in 0..magnitudes.len() {
        if magnitudes[index] > threshold {
            selected_freqs.push(freqs[index]);
            selected_phases.push(phases[index]);
        }
    }
    if selected_freqs.len() < 8 {
        return None;
    }
    let max_freq = selected_freqs.iter().copied().fold(0.0_f32, f32::max);
    if max_freq < 100.0 {
        return None;
    }

    // Unwrap phase (cumulative ±2π corrections); reject if any adjacent step
    // still jumps > 0.9π — the delay is too large for reliable unwrapping.
    let pi = std::f32::consts::PI;
    let mut unwrapped: Vec<f32> = Vec::with_capacity(selected_phases.len());
    unwrapped.push(selected_phases[0]);
    for index in 1..selected_phases.len() {
        let mut delta = selected_phases[index] - selected_phases[index - 1];
        while delta > pi {
            delta -= 2.0 * pi;
        }
        while delta < -pi {
            delta += 2.0 * pi;
        }
        if delta.abs() > 0.9 * pi {
            return None;
        }
        unwrapped.push(unwrapped[index - 1] + delta);
    }

    // Closed-form degree-1 least-squares fit: phase(f) ≈ slope·f + intercept.
    let n = selected_freqs.len() as f32;
    let sum_x: f32 = selected_freqs.iter().sum();
    let sum_y: f32 = unwrapped.iter().sum();
    let sum_xx: f32 = selected_freqs.iter().map(|f| f * f).sum();
    let sum_xy: f32 = selected_freqs
        .iter()
        .zip(unwrapped.iter())
        .map(|(f, p)| f * p)
        .sum();
    let denominator = n * sum_xx - sum_x * sum_x;
    if denominator.abs() < 1e-12 {
        return None;
    }
    let slope = (n * sum_xy - sum_x * sum_y) / denominator;
    // phase(f) ≈ -2π·f·τ  →  slope = -2π·τ  →  τ = -slope / (2π)
    let tau = -slope / (2.0 * pi);
    if !tau.is_finite() || tau.abs() > max_tau_s {
        return None;
    }
    Some(tau)
}

/// Power-weighted mean frequency (Hz) of a real signal. Mirrors the Python
/// `dominant_frequency_hz` in `minimappr/core/localization.py`: mean-centre,
/// apply a Hann taper, take the magnitude-squared spectrum, drop the DC bin, and
/// return Σ(f·power)/Σ(power). Returns 0.0 for empty/silent input.
pub(crate) fn dominant_frequency_hz(window: &[f32], sample_rate_hz: u32) -> f32 {
    let n = window.len();
    if n <= 1 || sample_rate_hz == 0 {
        return 0.0;
    }
    let mean = window.iter().copied().sum::<f32>() / n as f32;
    let centered: Vec<f32> = window.iter().map(|&sample| sample - mean).collect();
    if !centered.iter().any(|&value| value.abs() > 1e-12) {
        return 0.0;
    }
    // np.hanning(n): 0.5 - 0.5·cos(2π·i/(n-1))
    let pi = std::f32::consts::PI;
    let mut buffer: Vec<Complex32> = centered
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let taper = 0.5 - 0.5 * (2.0 * pi * index as f32 / (n as f32 - 1.0)).cos();
            Complex32::new(value * taper, 0.0)
        })
        .collect();

    thread_local! {
        static PLANNER: std::cell::RefCell<FftPlanner<f32>> = std::cell::RefCell::new(FftPlanner::new());
    }
    let fft = PLANNER.with(|planner| planner.borrow_mut().plan_fft_forward(n));
    fft.process(&mut buffer);

    // rfft-equivalent positive-frequency bins, dropping DC (bin 0).
    let half = n / 2;
    let mut total_power = 0.0_f64;
    let mut weighted_sum = 0.0_f64;
    for (bin, value) in buffer.iter().enumerate().take(half + 1).skip(1) {
        let power = (value.norm() as f64).powi(2);
        let freq = bin as f64 * sample_rate_hz as f64 / n as f64;
        total_power += power;
        weighted_sum += freq * power;
    }
    if total_power <= 1e-12 {
        return 0.0;
    }
    (weighted_sum / total_power) as f32
}

pub(crate) fn pair_max_tau_s(
    position_a_m: [f32; 3],
    position_b_m: [f32; 3],
    sample_rate_hz: u32,
    sound_speed_mps: f32,
) -> f32 {
    let distance_m = euclidean_distance(position_a_m, position_b_m);
    ((distance_m / sound_speed_mps.max(1.0)) + (1.0 / sample_rate_hz.max(1) as f32))
        .max(1.0 / sample_rate_hz.max(1) as f32)
}

/// All 6 pairwise TDOA results for a 4-microphone tetrahedral array.
/// Channel order follows the Sirith node: [MK1, MK2, MK3, MK4].
#[allow(dead_code)]
pub fn tetrahedral_gcc_phat(
    channels: &[Vec<f32>; 4],
    sample_rate_hz: u32,
    band_hz: Option<[f32; 2]>,
) -> [TdoaResult; 6] {
    const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    // Keep this helper permissive; the worker uses per-pair geometric limits.
    let max_tau_s = (0.07_f32 / 343.2) + (1.0 / sample_rate_hz.max(1) as f32);
    PAIRS.map(|(a, b)| {
        phat_correlation(
            &channels[a],
            &channels[b],
            sample_rate_hz,
            max_tau_s,
            band_hz,
            4,
        )
        .tdoa
    })
}

fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p = 1_usize;
    while p < n {
        p <<= 1;
    }
    p
}

fn zero_pad_hermitian_spectrum(spectrum: &[Complex32], target_len: usize) -> Vec<Complex32> {
    if target_len <= spectrum.len() {
        return spectrum.to_vec();
    }

    let original_len = spectrum.len();
    let half_len = original_len / 2;
    let mut padded = vec![Complex32::new(0.0, 0.0); target_len];
    padded[0] = spectrum[0];

    if original_len.is_multiple_of(2) {
        padded[1..half_len].copy_from_slice(&spectrum[1..half_len]);
        padded[target_len - (original_len - half_len - 1)..]
            .copy_from_slice(&spectrum[half_len + 1..]);
        padded[half_len] = spectrum[half_len] * 0.5;
        padded[target_len - half_len] = spectrum[half_len] * 0.5;
    } else {
        padded[1..=half_len].copy_from_slice(&spectrum[1..=half_len]);
        padded[target_len - half_len..].copy_from_slice(&spectrum[half_len + 1..]);
    }

    padded
}

fn normalize_band_hz(band_hz: Option<[f32; 2]>, sample_rate_hz: u32) -> Option<[f32; 2]> {
    let [low_hz, high_hz] = band_hz?;
    let nyquist_hz = sample_rate_hz as f32 * 0.5;
    let clamped_low_hz = low_hz.max(0.0).min(nyquist_hz);
    // A non-positive maximum means "no ceiling", i.e. up to Nyquist, so a
    // min-only band becomes a pure high-pass instead of disabling the mask
    // entirely. Mirrors create_localization_preprocessor on the Python side.
    let clamped_high_hz = if high_hz <= 0.0 {
        nyquist_hz
    } else {
        high_hz.min(nyquist_hz)
    };
    (clamped_high_hz > clamped_low_hz).then_some([clamped_low_hz, clamped_high_hz])
}

fn bin_in_band(bin: usize, fft_len: usize, sample_rate_hz: u32, band_hz: Option<[f32; 2]>) -> bool {
    let Some([low_hz, high_hz]) = band_hz else {
        return true;
    };
    let folded_bin = bin.min(fft_len.saturating_sub(bin));
    let frequency_hz = folded_bin as f32 * sample_rate_hz as f32 / fft_len as f32;
    frequency_hz >= low_hz && frequency_hz <= high_hz
}

fn euclidean_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic white-spectrum signal via LCG — gives a clean PHAT correlation peak.
    fn pseudo_random(n: usize) -> Vec<f32> {
        let mut x = 0x12345678_u32;
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (x as i32 as f32) / (i32::MAX as f32)
            })
            .collect()
    }

    #[test]
    fn min_only_band_becomes_a_highpass_up_to_nyquist() {
        // A non-positive maximum means "no ceiling", not "no filter": the mask
        // must run from the minimum up to Nyquist.
        let normalized = normalize_band_hz(Some([50.0, 0.0]), 16_000);
        assert_eq!(normalized, Some([50.0, 8_000.0]));

        let fft_len = 1_024;
        assert!(!bin_in_band(1, fft_len, 16_000, normalized), "15.6 Hz is below the high-pass");
        assert!(bin_in_band(64, fft_len, 16_000, normalized), "1 kHz must pass");
        assert!(bin_in_band(400, fft_len, 16_000, normalized), "6.25 kHz has no ceiling");
    }

    #[test]
    fn fully_disabled_band_masks_nothing() {
        assert_eq!(normalize_band_hz(Some([0.0, 0.0]), 16_000), Some([0.0, 8_000.0]));
        assert!(bin_in_band(1, 1_024, 16_000, normalize_band_hz(Some([0.0, 0.0]), 16_000)));
        assert!(bin_in_band(0, 1_024, 16_000, None));
    }

    #[test]
    fn both_bounds_still_build_a_bandpass() {
        let normalized = normalize_band_hz(Some([300.0, 3_500.0]), 16_000);
        assert_eq!(normalized, Some([300.0, 3_500.0]));
        assert!(!bin_in_band(400, 1_024, 16_000, normalized), "6.25 kHz is above the ceiling");
        assert!(bin_in_band(64, 1_024, 16_000, normalized));
    }

    #[test]
    fn zero_delay_gives_zero_lag() {
        let sr = 16_000;
        let signal = pseudo_random(512);
        let result = gcc_phat(&signal, &signal, sr, 4);
        assert!(
            result.delay_samples.abs() < 0.2,
            "expected ~0 lag, got {}",
            result.delay_samples
        );
        assert!(result.confidence > 0.5);
        assert!(result.peak_value > 0.0);
    }

    #[test]
    fn integer_delay_recovered_accurately() {
        let sr = 16_000;
        let n = 512;
        let signal = pseudo_random(n + 4);
        let ch1 = signal[..n].to_vec();
        let ch2 = signal[2..n + 2].to_vec(); // ch2 = ch1 shifted left by 2 → peak at +2
        let result = gcc_phat(&ch1, &ch2, sr, 4);
        assert!(
            (result.delay_samples - 2.0).abs() < 0.15,
            "expected 2.0 ± 0.15, got {}",
            result.delay_samples
        );
    }

    #[test]
    fn sub_sample_delay_within_quarter_sample_of_truth() {
        let sr = 16_000;
        let n = 512;
        let signal = pseudo_random(n + 2);
        let ch1 = signal[..n].to_vec();
        let ch2 = signal[1..n + 1].to_vec(); // 1-sample delay; parabolic interp should stay close
        let result = gcc_phat(&ch1, &ch2, sr, 4);
        assert!(
            (result.delay_samples - 1.0).abs() < 0.25,
            "expected 1.0 ± 0.25, got {}",
            result.delay_samples
        );
    }

    #[test]
    fn tetrahedral_returns_six_results() {
        let sr = 16_000;
        let signal = pseudo_random(512);
        let channels = [
            signal.clone(),
            signal.clone(),
            signal.clone(),
            signal.clone(),
        ];
        let results = tetrahedral_gcc_phat(&channels, sr, None);
        assert_eq!(results.len(), 6);
        for r in &results {
            assert!(r.delay_samples.abs() < 0.3);
        }
    }

    #[test]
    fn localization_band_suppresses_out_of_band_delay() {
        let sr = 16_000;
        let len = 1_024;
        let low_component = moving_average(&pseudo_random_with_seed(0xA5A5_A5A5, len + 8), 31);
        let ch1 = low_component[..len].to_vec();
        let ch2 = low_component[4..len + 4].to_vec();

        let unbanded = phat_correlation(&ch1, &ch2, sr, 6.0 / sr as f32, None, 4).tdoa;
        let banded =
            phat_correlation(&ch1, &ch2, sr, 6.0 / sr as f32, Some([1_000.0, 3_200.0]), 4).tdoa;

        assert!(
            (unbanded.delay_samples - 4.0).abs() < 0.45,
            "expected unbanded low-band lag near 4 samples, got {}",
            unbanded.delay_samples
        );
        assert!(
            banded.peak_value < (unbanded.peak_value * 0.35),
            "expected banded path to suppress the out-of-band correlation, got banded_peak={} unbanded_peak={}",
            banded.peak_value,
            unbanded.peak_value
        );
        assert!(
            banded.confidence < unbanded.confidence,
            "expected banded path to lower confidence for an out-of-band signal, got banded_confidence={} unbanded_confidence={}",
            banded.confidence,
            unbanded.confidence
        );
    }

    #[test]
    fn localization_band_preserves_in_band_delay() {
        let sr = 16_000;
        let len = 1_024;
        let high_raw = pseudo_random_with_seed(0x5A5A_5A5A, len + 8);
        let high_component = high_raw
            .iter()
            .zip(moving_average(&high_raw, 9).iter())
            .map(|(raw, smooth)| raw - smooth)
            .collect::<Vec<_>>();

        let ch1 = high_component[..len].to_vec();
        let ch2 = high_component[1..len + 1].to_vec();
        let banded =
            phat_correlation(&ch1, &ch2, sr, 6.0 / sr as f32, Some([1_000.0, 3_200.0]), 4).tdoa;

        assert!(
            (banded.delay_samples - 1.0).abs() < 0.45,
            "expected banded in-band lag near 1 sample, got {}",
            banded.delay_samples
        );
        assert!(banded.peak_value > 0.0);
    }

    #[test]
    fn fractional_delay_recovers_three_halves_sample_shift() {
        let sr = 16_000;
        let signal = pseudo_random(640);
        let ch1 = signal[..512].to_vec();
        let ch2 = fractional_delay(&signal, 1.5, 512);
        let result = phat_correlation(&ch1, &ch2, sr, 4.0 / sr as f32, None, 4).tdoa;
        assert!(
            (result.delay_samples + 1.5).abs() < 0.2,
            "expected -1.5 ± 0.2, got {}",
            result.delay_samples
        );
    }

    #[test]
    fn phase_slope_tau_recovers_subsample_delay() {
        use rustfft::FftPlanner;
        let sr = 48_000u32;
        let n = 1_024usize;
        let signal = pseudo_random(n);
        let fft_len = next_pow2(2 * n);
        let fft = FftPlanner::<f32>::new().plan_fft_forward(fft_len);
        let mut x1: Vec<Complex32> = signal
            .iter()
            .map(|&s| Complex32::new(s, 0.0))
            .chain(std::iter::repeat_n(Complex32::new(0.0, 0.0), fft_len - n))
            .collect();
        fft.process(&mut x1);

        // Construct x2 as a *true* fractional delay of x1 in the frequency domain:
        // x2(f) = x1(f)·exp(-j2πf·d/sr) lags x1 by d samples. With cross = x1·conj(x2)
        // the phase slope then encodes -d samples (positive delay_samples = ch2
        // leads ch1). Only the positive-freq half read by phase_slope_tau matters.
        let lag_samples = 0.4_f32;
        let half = fft_len / 2;
        let mut x2 = vec![Complex32::new(0.0, 0.0); fft_len];
        for bin in 0..=half {
            let freq = bin as f32 * sr as f32 / fft_len as f32;
            let theta = -2.0 * std::f32::consts::PI * freq * lag_samples / sr as f32;
            x2[bin] = x1[bin] * Complex32::new(theta.cos(), theta.sin());
        }

        let tau = phase_slope_tau(&x1, &x2, fft_len, sr, None, 8.0 / sr as f32)
            .expect("phase slope should resolve a clean broadband sub-sample delay");
        let recovered_samples = tau * sr as f32;
        assert!(
            (recovered_samples + lag_samples).abs() < 0.02,
            "expected ~{} samples, got {}",
            -lag_samples,
            recovered_samples
        );
    }

    #[test]
    fn phase_slope_returns_none_without_enough_energy_bins() {
        let sr = 48_000u32;
        let fft_len = 2_048usize;
        // Energy in only three bins → below the ≥ 8 broadband-energy-bin gate.
        let mut x1 = vec![Complex32::new(0.0, 0.0); fft_len];
        for &bin in &[40usize, 41, 42] {
            x1[bin] = Complex32::new(1.0, 0.5);
        }
        let x2 = x1.clone();
        let tau = phase_slope_tau(&x1, &x2, fft_len, sr, None, 8.0 / sr as f32);
        assert!(
            tau.is_none(),
            "expected None for narrowband energy, got {tau:?}"
        );
    }

    #[test]
    fn dominant_frequency_recovers_tone() {
        let sr = 48_000u32;
        let n = 4_096usize;
        let freq = 3_000.0f32;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
            .collect();
        let dominant = dominant_frequency_hz(&signal, sr);
        assert!(
            (dominant - freq).abs() < 100.0,
            "expected ~{freq} Hz, got {dominant}"
        );
    }

    #[test]
    fn dominant_frequency_zero_for_silence_or_empty() {
        assert_eq!(dominant_frequency_hz(&vec![0.0; 1_024], 48_000), 0.0);
        assert_eq!(dominant_frequency_hz(&[], 48_000), 0.0);
        assert_eq!(dominant_frequency_hz(&[1.0], 48_000), 0.0);
    }

    fn pseudo_random_with_seed(mut seed: u32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed as i32 as f32) / (i32::MAX as f32)
            })
            .collect()
    }

    fn moving_average(samples: &[f32], window_len: usize) -> Vec<f32> {
        let mut smoothed = Vec::with_capacity(samples.len());
        let radius = window_len.max(1) / 2;
        for index in 0..samples.len() {
            let start = index.saturating_sub(radius);
            let end = (index + radius + 1).min(samples.len());
            let mean = samples[start..end].iter().sum::<f32>() / (end - start) as f32;
            smoothed.push(mean);
        }
        smoothed
    }

    fn fractional_delay(source: &[f32], delay_samples: f32, output_len: usize) -> Vec<f32> {
        (0..output_len)
            .map(|sample_index| {
                let source_index = sample_index as f32 - delay_samples;
                if source_index < 0.0 || source_index + 1.0 >= source.len() as f32 {
                    return 0.0;
                }
                let lower_index = source_index.floor() as usize;
                let upper_index = lower_index + 1;
                let fraction = source_index - lower_index as f32;
                (source[lower_index] * (1.0 - fraction)) + (source[upper_index] * fraction)
            })
            .collect()
    }
}
