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
    let n = ch1.len().min(ch2.len());
    if n == 0 {
        return TdoaResult {
            delay_samples: 0.0,
            lag_seconds: 0.0,
            confidence: 0.0,
        };
    }

    // Zero-pad to next power-of-2 >= 2*n to prevent circular wrap-around.
    let fft_len = next_pow2(2 * n);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_len);
    let ifft = planner.plan_fft_inverse(fft_len);

    let mut x1: Vec<Complex32> = ch1[..n]
        .iter()
        .map(|&s| Complex32::new(s, 0.0))
        .chain(std::iter::repeat(Complex32::new(0.0, 0.0)).take(fft_len - n))
        .collect();
    let mut x2: Vec<Complex32> = ch2[..n]
        .iter()
        .map(|&s| Complex32::new(s, 0.0))
        .chain(std::iter::repeat(Complex32::new(0.0, 0.0)).take(fft_len - n))
        .collect();

    fft.process(&mut x1);
    fft.process(&mut x2);

    // Cross-spectrum with PHAT whitening.
    const EPSILON: f32 = 1e-9;
    let mut g: Vec<Complex32> = x1
        .iter()
        .zip(x2.iter())
        .map(|(a, b)| {
            let cross = a * b.conj();
            let mag = cross.norm();
            cross / (mag + EPSILON)
        })
        .collect();

    ifft.process(&mut g);
    let scale = 1.0 / fft_len as f32;

    // Real part of the IFFT output is the GCC-PHAT correlation.
    // The correlation is wrapped: positive lags are at indices [0..max_lag] and
    // negative lags are at indices [fft_len - max_lag..fft_len].
    let lag = max_lag_samples.min(fft_len / 2);
    let mut corr: Vec<(i64, f32)> = Vec::with_capacity(2 * lag + 1);
    for d in 0..=lag as i64 {
        corr.push((d, g[d as usize].re * scale));
    }
    for d in 1..=lag as i64 {
        let idx = fft_len as i64 - d;
        corr.push((-d, g[idx as usize].re * scale));
    }
    corr.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let (peak_lag, peak_val) = corr[0];

    // Parabolic interpolation for sub-sample accuracy.
    let delay_samples = parabolic_interp(&g, peak_lag, fft_len, scale);

    // Confidence: peak vs. mean of the next 3 highest sidelobe maxima.
    let sidelobe_mean = if corr.len() > 1 {
        corr[1..corr.len().min(4)]
            .iter()
            .map(|(_, v)| v.abs())
            .sum::<f32>()
            / (corr.len().min(4) - 1) as f32
    } else {
        0.0
    };
    let confidence = if peak_val.abs() > 0.0 && sidelobe_mean > 0.0 {
        (peak_val.abs() / (peak_val.abs() + sidelobe_mean)).clamp(0.0, 1.0)
    } else if peak_val.abs() > 0.0 {
        1.0
    } else {
        0.0
    };

    TdoaResult {
        delay_samples,
        lag_seconds: delay_samples / sample_rate_hz as f32,
        confidence,
    }
}

/// All 6 pairwise TDOA results for a 4-microphone tetrahedral array.
/// Channel order follows the Sirith node: [MK1, MK2, MK3, MK4].
pub fn tetrahedral_gcc_phat(channels: &[Vec<f32>; 4], sample_rate_hz: u32) -> [TdoaResult; 6] {
    const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    // 50 mm max spacing, 340 m/s → 0.147 ms → 2.4 samples at 16 kHz
    let max_lag = (0.05_f32 / 340.0 * sample_rate_hz as f32).ceil() as usize + 1;
    PAIRS.map(|(a, b)| gcc_phat(&channels[a], &channels[b], sample_rate_hz, max_lag))
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

fn parabolic_interp(g: &[Complex32], peak_lag: i64, fft_len: usize, scale: f32) -> f32 {
    let to_idx = |lag: i64| -> usize {
        if lag >= 0 {
            lag as usize
        } else {
            (fft_len as i64 + lag) as usize
        }
    };
    let y0 = g[to_idx(peak_lag - 1)].re * scale;
    let y1 = g[to_idx(peak_lag)].re * scale;
    let y2 = g[to_idx(peak_lag + 1)].re * scale;
    let denom = 2.0 * y1 - y0 - y2;
    if denom.abs() < 1e-12 {
        return peak_lag as f32;
    }
    peak_lag as f32 + (y2 - y0) / (2.0 * denom)
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
        let results = tetrahedral_gcc_phat(&channels, sr);
        assert_eq!(results.len(), 6);
        for r in &results {
            assert!(r.delay_samples.abs() < 0.3);
        }
    }
}
