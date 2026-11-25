//! Chirp energy that leaks into the bird band (plan A9): the speaker and
//! amplifier non-linearities fold some of a 13–20 kHz chirp down below
//! 11 kHz, where the Stage-1 detector and classifier listen. Measured per
//! chirp as band power during the chirp over band power just before it.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::pair_ranging_series::median_of;

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct LeakageReport {
    pub band_hz: [f64; 2],
    pub chirp_count: usize,
    /// Median over chirps of 10·log10(band power in chirp / before chirp).
    pub median_over_floor_db: Option<f64>,
    pub max_over_floor_db: Option<f64>,
}

fn band_power(samples: &[f32], band_hz: [f64; 2], sample_rate_hz: f64, planner: &mut FftPlanner<f32>) -> f64 {
    let length = samples.len();
    if length == 0 {
        return 0.0;
    }
    let fft = planner.plan_fft_forward(length);
    let mut buffer: Vec<Complex<f32>> = samples
        .iter()
        .enumerate()
        .map(|(index, sample)| {
            let hann = 0.5 - 0.5 * (std::f32::consts::TAU * index as f32 / length as f32).cos();
            Complex::new(sample * hann, 0.0)
        })
        .collect();
    fft.process(&mut buffer);
    let bin_hz = sample_rate_hz / length as f64;
    let first_bin = (band_hz[0] / bin_hz).ceil() as usize;
    let last_bin = ((band_hz[1] / bin_hz).floor() as usize).min(length / 2);
    (first_bin..=last_bin).map(|bin| buffer[bin].norm_sqr() as f64).sum()
}

/// `chirp_starts`: waveform-start samples of chirps in this recording.
pub fn leakage_report(audio: &[f32], chirp_starts: &[f64], chirp_length: usize, band_hz: [f64; 2], sample_rate_hz: f64) -> LeakageReport {
    let mut planner = FftPlanner::new();
    let ratios_db: Vec<f64> = chirp_starts
        .iter()
        .filter_map(|start| {
            let start = start.round() as usize;
            // Noise window: one chirp length ending a chirp length before the chirp.
            let noise_start = start.checked_sub(2 * chirp_length)?;
            let chirp = audio.get(start..start + chirp_length)?;
            let noise = audio.get(noise_start..noise_start + chirp_length)?;
            let noise_power = band_power(noise, band_hz, sample_rate_hz, &mut planner);
            let chirp_power = band_power(chirp, band_hz, sample_rate_hz, &mut planner);
            (noise_power > 0.0).then(|| 10.0 * (chirp_power / noise_power).log10())
        })
        .collect();
    LeakageReport {
        band_hz,
        chirp_count: ratios_db.len(),
        median_over_floor_db: median_of(&ratios_db),
        max_over_floor_db: ratios_db.iter().copied().reduce(f64::max),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_band_tone_during_chirp_reads_as_leakage_and_clean_chirp_does_not() {
        let rate = 48_000.0;
        let length = 2_400;
        let mut audio: Vec<f32> = (0..20_000).map(|i| ((i * 7919 % 1000) as f32 / 1000.0 - 0.5) * 0.001).collect();
        for i in 10_000..10_000 + length {
            audio[i] += 0.3 * (std::f32::consts::TAU * 15_000.0 * i as f32 / rate as f32).sin();
        }
        let clean = leakage_report(&audio, &[10_000.0], length, [2_000.0, 11_000.0], rate);
        assert!(clean.median_over_floor_db.unwrap() < 3.0, "{clean:?}");
        for i in 10_000..10_000 + length {
            audio[i] += 0.05 * (std::f32::consts::TAU * 5_000.0 * i as f32 / rate as f32).sin();
        }
        let leaky = leakage_report(&audio, &[10_000.0], length, [2_000.0, 11_000.0], rate);
        assert!(leaky.median_over_floor_db.unwrap() > 20.0, "{leaky:?}");
    }
}
