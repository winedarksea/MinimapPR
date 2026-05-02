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
    )
    .tdoa
}

pub(crate) fn phat_correlation(
    ch1: &[f32],
    ch2: &[f32],
    sample_rate_hz: u32,
    max_tau_s: f32,
    band_hz: Option<[f32; 2]>,
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

    let (fft, ifft) = PLANNER.with(|planner| {
        let mut p = planner.borrow_mut();
        (p.plan_fft_forward(fft_len), p.plan_fft_inverse(fft_len))
    });

    let mut x1: Vec<Complex32> = ch1[..n]
        .iter()
        .map(|&sample| Complex32::new(sample, 0.0))
        .chain(std::iter::repeat(Complex32::new(0.0, 0.0)).take(fft_len - n))
        .collect();
    let mut x2: Vec<Complex32> = ch2[..n]
        .iter()
        .map(|&sample| Complex32::new(sample, 0.0))
        .chain(std::iter::repeat(Complex32::new(0.0, 0.0)).take(fft_len - n))
        .collect();

    fft.process(&mut x1);
    fft.process(&mut x2);

    let effective_band = normalize_band_hz(band_hz, sample_rate_hz);
    let mut cross_spectrum: Vec<Complex32> = x1
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

    ifft.process(&mut cross_spectrum);

    let scale = 1.0 / fft_len as f32;
    let lag =
        ((max_tau_s.max(1.0 / sample_rate_hz as f32)) * sample_rate_hz as f32).ceil() as usize;
    let lag = lag.clamp(1, fft_len / 2);

    let mut lags_seconds = Vec::with_capacity((2 * lag) + 1);
    let mut magnitudes = Vec::with_capacity((2 * lag) + 1);

    for delay in (1..=lag).rev() {
        lags_seconds.push(-(delay as f32) / sample_rate_hz as f32);
        magnitudes.push((cross_spectrum[fft_len - delay].re * scale).abs());
    }
    lags_seconds.push(0.0);
    magnitudes.push((cross_spectrum[0].re * scale).abs());
    for delay in 1..=lag {
        lags_seconds.push(delay as f32 / sample_rate_hz as f32);
        magnitudes.push((cross_spectrum[delay].re * scale).abs());
    }

    let (peak_index, peak_value) = magnitudes
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.partial_cmp(right).unwrap_or(Ordering::Equal))
        .unwrap_or((lag, 0.0));
    let peak_offset = parabolic_peak_offset(&magnitudes, peak_index);
    let delay_samples = (peak_index as f32 - lag as f32) + peak_offset;

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

fn parabolic_peak_offset(magnitudes: &[f32], peak_index: usize) -> f32 {
    if peak_index == 0 || peak_index + 1 >= magnitudes.len() {
        return 0.0;
    }
    let left = magnitudes[peak_index - 1];
    let center = magnitudes[peak_index];
    let right = magnitudes[peak_index + 1];
    let denominator = left - (2.0 * center) + right;
    if denominator.abs() < 1e-12 {
        return 0.0;
    }
    (0.5 * (left - right) / denominator).clamp(-1.0, 1.0)
}

fn normalize_band_hz(band_hz: Option<[f32; 2]>, sample_rate_hz: u32) -> Option<[f32; 2]> {
    let [low_hz, high_hz] = band_hz?;
    let nyquist_hz = sample_rate_hz as f32 * 0.5;
    let clamped_low_hz = low_hz.max(0.0).min(nyquist_hz);
    let clamped_high_hz = high_hz.max(0.0).min(nyquist_hz);
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

        let unbanded = phat_correlation(&ch1, &ch2, sr, 6.0 / sr as f32, None).tdoa;
        let banded =
            phat_correlation(&ch1, &ch2, sr, 6.0 / sr as f32, Some([1_000.0, 3_200.0])).tdoa;

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
            phat_correlation(&ch1, &ch2, sr, 6.0 / sr as f32, Some([1_000.0, 3_200.0])).tdoa;

        assert!(
            (banded.delay_samples - 1.0).abs() < 0.45,
            "expected banded in-band lag near 1 sample, got {}",
            banded.delay_samples
        );
        assert!(banded.peak_value > 0.0);
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
}
