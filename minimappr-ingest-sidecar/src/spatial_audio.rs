//! First-order ambisonics encoders for offline capture rendering.

use std::f64::consts::PI;

use num_complex::Complex64;
use rustfft::{FftDirection, FftPlanner};
use serde::Deserialize;

pub const SPEED_OF_SOUND_MPS: f64 = 343.2;
pub const TIKHONOV_LAMBDA_0: f64 = 1e-3;
pub const TIKHONOV_F_REF_HZ: f64 = 120.0;

#[derive(Clone, Debug)]
pub struct AmbisonicsRenderRequest {
    pub channels: Vec<Vec<f32>>,
    pub mic_positions_m: Vec<[f64; 3]>,
    pub sample_rate_hz: u32,
    pub block_size: usize,
    pub hop_size: usize,
}

#[derive(Clone, Debug)]
pub struct AmbisonicsRenderOutput {
    pub bformat: Vec<Vec<f32>>,
    pub sample_rate_hz: u32,
}

#[derive(Clone, Debug)]
pub struct AmbisonicsProfile {
    pub frame_duration_ms: f64,
    pub hop_fraction: f64,
    pub min_parametric_hz: f64,
    pub max_parametric_fraction_of_nyquist: f64,
    pub intensity_smoothing_ms: f64,
    pub diffuseness_smoothing_ms: f64,
    pub max_parametric_blend: f64,
    pub min_confidence_for_blend: f64,
    pub output_peak_target: f64,
}

impl AmbisonicsProfile {
    pub fn from_name(name: &str) -> Result<Self, String> {
        let profiles: std::collections::HashMap<String, AmbisonicsProfileJson> =
            serde_json::from_str(include_str!(
                "../../minimappr/spatial_audio/ambisonics_profiles.json"
            ))
            .map_err(|error| format!("failed to parse ambisonics profile JSON: {error}"))?;
        profiles
            .get(name)
            .map(AmbisonicsProfileJson::to_profile)
            .ok_or_else(|| format!("unknown ambisonics profile {name}"))
    }
}

#[derive(Clone, Debug, Deserialize)]
struct AmbisonicsProfileJson {
    frame_duration_ms: f64,
    hop_fraction: f64,
    min_parametric_hz: f64,
    max_parametric_fraction_of_nyquist: f64,
    intensity_smoothing_ms: f64,
    diffuseness_smoothing_ms: f64,
    max_parametric_blend: f64,
    min_confidence_for_blend: f64,
    output_peak_target: f64,
}

impl AmbisonicsProfileJson {
    fn to_profile(&self) -> AmbisonicsProfile {
        AmbisonicsProfile {
            frame_duration_ms: self.frame_duration_ms,
            hop_fraction: self.hop_fraction,
            min_parametric_hz: self.min_parametric_hz,
            max_parametric_fraction_of_nyquist: self.max_parametric_fraction_of_nyquist,
            intensity_smoothing_ms: self.intensity_smoothing_ms,
            diffuseness_smoothing_ms: self.diffuseness_smoothing_ms,
            max_parametric_blend: self.max_parametric_blend,
            min_confidence_for_blend: self.min_confidence_for_blend,
            output_peak_target: self.output_peak_target,
        }
    }
}

pub fn encode_ambisonics(
    request: AmbisonicsRenderRequest,
    profile: &AmbisonicsProfile,
) -> AmbisonicsRenderOutput {
    let frame_size = frame_size_for_rate(request.sample_rate_hz, profile.frame_duration_ms);
    let hop_size = (frame_size / 4).max(1);
    let linear = atob_foa_linear(AmbisonicsRenderRequest {
        channels: request.channels,
        mic_positions_m: request.mic_positions_m,
        sample_rate_hz: request.sample_rate_hz,
        block_size: frame_size,
        hop_size,
    });
    if profile.max_parametric_blend <= 0.0 {
        return AmbisonicsRenderOutput {
            bformat: scale_true_peak(linear.bformat, profile.output_peak_target),
            sample_rate_hz: linear.sample_rate_hz,
        };
    }
    let enhanced = enhance_foa_parametric(&linear.bformat, linear.sample_rate_hz, profile, frame_size);
    AmbisonicsRenderOutput {
        bformat: scale_true_peak(enhanced, profile.output_peak_target),
        sample_rate_hz: linear.sample_rate_hz,
    }
}

pub fn atob_foa_linear(request: AmbisonicsRenderRequest) -> AmbisonicsRenderOutput {
    let mic_count = request.channels.len().min(request.mic_positions_m.len());
    if mic_count < 4 || request.channels.is_empty() {
        return AmbisonicsRenderOutput {
            bformat: vec![Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            sample_rate_hz: request.sample_rate_hz,
        };
    }
    let n_samples = request.channels[0].len();
    if n_samples == 0 {
        return AmbisonicsRenderOutput {
            bformat: vec![Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            sample_rate_hz: request.sample_rate_hz,
        };
    }
    for channel in &request.channels {
        assert_eq!(channel.len(), n_samples, "all channels must have equal length");
    }

    let block_size = request.block_size.max(2).next_power_of_two();
    let hop_size = request.hop_size.max(1).min(block_size);
    let bin_count = (block_size / 2) + 1;
    let positions = centroid_corrected_positions(&request.mic_positions_m[..mic_count]);
    let alias_cutoff_hz = alias_cutoff_from_positions(&request.mic_positions_m[..mic_count]);
    let freqs: Vec<f64> = (0..bin_count)
        .map(|bin| bin as f64 * request.sample_rate_hz as f64 / block_size as f64)
        .collect();
    let window = hanning_window(block_size);

    let mut output = vec![vec![0.0f64; n_samples]; 4];
    let mut norm = vec![0.0f64; n_samples];
    let n_blocks = n_samples.div_ceil(hop_size);

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft(block_size, FftDirection::Forward);
    let ifft = planner.plan_fft(block_size, FftDirection::Inverse);

    for block_index in 0..n_blocks {
        let start = block_index * hop_size;
        let end = (start + block_size).min(n_samples);
        let actual = end - start;

        let mut a_freq = vec![vec![Complex64::new(0.0, 0.0); bin_count]; mic_count];
        for channel_index in 0..mic_count {
            let mut buffer = vec![Complex64::new(0.0, 0.0); block_size];
            for sample_index in 0..actual {
                buffer[sample_index].re =
                    request.channels[channel_index][start + sample_index] as f64
                        * window[sample_index];
            }
            fft.process(&mut buffer);
            a_freq[channel_index][..bin_count].copy_from_slice(&buffer[..bin_count]);
        }

        let mut b_freq = apply_atob_matrix(&a_freq, &positions, &freqs);
        apply_alias_lowpass(&mut b_freq, &freqs, alias_cutoff_hz);

        for component_index in 0..4 {
            let mut spectrum = vec![Complex64::new(0.0, 0.0); block_size];
            spectrum[..bin_count].copy_from_slice(&b_freq[component_index][..bin_count]);
            for bin in 1..(block_size / 2) {
                spectrum[block_size - bin] = spectrum[bin].conj();
            }
            ifft.process(&mut spectrum);
            for sample_index in 0..actual {
                let value = spectrum[sample_index].re / block_size as f64;
                output[component_index][start + sample_index] += value * window[sample_index];
            }
        }
        for sample_index in 0..actual {
            norm[start + sample_index] += window[sample_index] * window[sample_index];
        }
    }

    let bformat = output
        .into_iter()
        .map(|mut component| {
            for (sample, normalizer) in component.iter_mut().zip(norm.iter()) {
                if *normalizer > 1e-12 {
                    *sample /= *normalizer;
                }
                *sample = sample.clamp(-1.0, 1.0);
            }
            component.into_iter().map(|sample| sample as f32).collect()
        })
        .collect();

    AmbisonicsRenderOutput {
        bformat,
        sample_rate_hz: request.sample_rate_hz,
    }
}

fn enhance_foa_parametric(
    foa_linear: &[Vec<f32>],
    sample_rate_hz: u32,
    profile: &AmbisonicsProfile,
    frame_size: usize,
) -> Vec<Vec<f32>> {
    if foa_linear.len() != 4 || foa_linear[0].is_empty() {
        return foa_linear.to_vec();
    }
    let hop_size = ((frame_size as f64 * profile.hop_fraction).round() as usize).max(1);
    let window = sqrt_hann_window(frame_size);
    let spectra = stft_channels(foa_linear, frame_size, hop_size, &window);
    let frame_count = spectra[0].len();
    let bin_count = spectra[0][0].len();
    let freqs: Vec<f64> = (0..bin_count)
        .map(|bin| bin as f64 * sample_rate_hz as f64 / frame_size as f64)
        .collect();
    let directions = smoothed_intensity_directions(
        &spectra,
        hop_size,
        sample_rate_hz,
        profile.intensity_smoothing_ms,
    );
    let diffuseness = smoothed_diffuseness(
        &spectra,
        &directions,
        hop_size,
        sample_rate_hz,
        profile.diffuseness_smoothing_ms,
    );

    let mut output_spectra = spectra.clone();
    let mut energy_values = Vec::with_capacity(frame_count * bin_count);
    for frame in 0..frame_count {
        for bin in 0..bin_count {
            let mut energy = spectra[0][frame][bin].norm_sqr();
            for component in spectra.iter().take(4).skip(1) {
                energy += component[frame][bin].norm_sqr();
            }
            energy_values.push(energy);
        }
    }
    let energy_floor = percentile(&mut energy_values, 35.0);
    let low_hz = profile.min_parametric_hz;
    let high_hz = 0.5 * sample_rate_hz as f64 * profile.max_parametric_fraction_of_nyquist;

    for frame in 0..frame_count {
        for (bin, freq_hz) in freqs.iter().enumerate() {
            let mut energy = spectra[0][frame][bin].norm_sqr();
            for component in spectra.iter().take(4).skip(1) {
                energy += component[frame][bin].norm_sqr();
            }
            let confidence = energy / (energy + energy_floor + 1e-12);
            let mut blend =
                profile.max_parametric_blend * (1.0 - diffuseness[frame][bin]) * confidence;
            if confidence < profile.min_confidence_for_blend || *freq_hz < low_hz || *freq_hz > high_hz {
                blend = 0.0;
            }
            blend = blend.clamp(0.0, profile.max_parametric_blend);
            output_spectra[0][frame][bin] = spectra[0][frame][bin];
            for axis in 0..3 {
                let parametric_xyz = spectra[0][frame][bin] * (2.0f64.sqrt() * directions[frame][axis]);
                output_spectra[axis + 1][frame][bin] =
                    spectra[axis + 1][frame][bin] * (1.0 - blend) + parametric_xyz * blend;
            }
        }
    }

    istft_channels(
        &output_spectra,
        frame_size,
        hop_size,
        foa_linear[0].len(),
        &window,
    )
}

fn hanning_window(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }
    (0..n)
        .map(|i| 0.5 - (0.5 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()))
        .collect()
}

fn sqrt_hann_window(n: usize) -> Vec<f64> {
    hanning_window(n).into_iter().map(f64::sqrt).collect()
}

fn stft_channels(
    channels: &[Vec<f32>],
    frame_size: usize,
    hop_size: usize,
    window: &[f64],
) -> Vec<Vec<Vec<Complex64>>> {
    let channel_count = channels.len();
    let n_samples = channels[0].len();
    let frame_count = ((n_samples.saturating_sub(frame_size)).max(1) as f64 / hop_size as f64)
        .ceil() as usize
        + 1;
    let bin_count = (frame_size / 2) + 1;
    let mut spectra =
        vec![vec![vec![Complex64::new(0.0, 0.0); bin_count]; frame_count]; channel_count];
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft(frame_size, FftDirection::Forward);
    for channel_index in 0..channel_count {
        for frame_index in 0..frame_count {
            let start = frame_index * hop_size;
            let end = (start + frame_size).min(n_samples);
            let actual = end.saturating_sub(start);
            let mut buffer = vec![Complex64::new(0.0, 0.0); frame_size];
            for sample_index in 0..actual {
                buffer[sample_index].re =
                    channels[channel_index][start + sample_index] as f64 * window[sample_index];
            }
            fft.process(&mut buffer);
            spectra[channel_index][frame_index][..bin_count].copy_from_slice(&buffer[..bin_count]);
        }
    }
    spectra
}

fn istft_channels(
    spectra: &[Vec<Vec<Complex64>>],
    frame_size: usize,
    hop_size: usize,
    n_samples: usize,
    window: &[f64],
) -> Vec<Vec<f32>> {
    let channel_count = spectra.len();
    let frame_count = spectra[0].len();
    let bin_count = (frame_size / 2) + 1;
    let mut output = vec![vec![0.0f64; n_samples]; channel_count];
    let mut norm = vec![0.0f64; n_samples];
    let mut planner = FftPlanner::<f64>::new();
    let ifft = planner.plan_fft(frame_size, FftDirection::Inverse);

    for frame_index in 0..frame_count {
        let start = frame_index * hop_size;
        let end = (start + frame_size).min(n_samples);
        let actual = end.saturating_sub(start);
        if actual == 0 {
            continue;
        }
        for channel_index in 0..channel_count {
            let mut spectrum = vec![Complex64::new(0.0, 0.0); frame_size];
            spectrum[..bin_count].copy_from_slice(&spectra[channel_index][frame_index][..bin_count]);
            for bin in 1..(frame_size / 2) {
                spectrum[frame_size - bin] = spectrum[bin].conj();
            }
            ifft.process(&mut spectrum);
            for sample_index in 0..actual {
                output[channel_index][start + sample_index] +=
                    (spectrum[sample_index].re / frame_size as f64) * window[sample_index];
            }
        }
        for sample_index in 0..actual {
            norm[start + sample_index] += window[sample_index] * window[sample_index];
        }
    }

    output
        .into_iter()
        .map(|mut channel| {
            for (sample, normalizer) in channel.iter_mut().zip(norm.iter()) {
                if *normalizer > 1e-12 {
                    *sample /= *normalizer;
                }
            }
            channel.into_iter().map(|sample| sample as f32).collect()
        })
        .collect()
}

fn smoothed_intensity_directions(
    spectra: &[Vec<Vec<Complex64>>],
    hop_size: usize,
    sample_rate_hz: u32,
    smoothing_ms: f64,
) -> Vec<[f64; 3]> {
    let frame_count = spectra[0].len();
    let bin_count = spectra[0][0].len();
    let alpha = ema_alpha(hop_size, sample_rate_hz, smoothing_ms);
    let mut previous = [1.0, 0.0, 0.0];
    let mut directions = vec![[1.0, 0.0, 0.0]; frame_count];
    for frame in 0..frame_count {
        let mut vector = [0.0, 0.0, 0.0];
        for bin in 0..bin_count {
            let w_conj = spectra[0][frame][bin].conj();
            for axis in 0..3 {
                vector[axis] += (w_conj * spectra[axis + 1][frame][bin]).re;
            }
        }
        if vector_norm(vector) > 1e-12 {
            previous = [
                alpha * previous[0] + (1.0 - alpha) * vector[0],
                alpha * previous[1] + (1.0 - alpha) * vector[1],
                alpha * previous[2] + (1.0 - alpha) * vector[2],
            ];
        }
        directions[frame] = normalize_vec3_f64(previous);
    }
    directions
}

fn smoothed_diffuseness(
    spectra: &[Vec<Vec<Complex64>>],
    directions: &[[f64; 3]],
    hop_size: usize,
    sample_rate_hz: u32,
    smoothing_ms: f64,
) -> Vec<Vec<f64>> {
    let frame_count = spectra[0].len();
    let bin_count = spectra[0][0].len();
    let alpha = ema_alpha(hop_size, sample_rate_hz, smoothing_ms);
    let mut raw = vec![vec![0.0; bin_count]; frame_count];
    for frame in 0..frame_count {
        for bin in 0..bin_count {
            let projected = spectra[1][frame][bin] * directions[frame][0]
                + spectra[2][frame][bin] * directions[frame][1]
                + spectra[3][frame][bin] * directions[frame][2];
            let directional_energy = projected.norm_sqr();
            let total_velocity_energy = spectra[1][frame][bin].norm_sqr()
                + spectra[2][frame][bin].norm_sqr()
                + spectra[3][frame][bin].norm_sqr();
            raw[frame][bin] = (1.0 - (directional_energy / (total_velocity_energy + 1e-12)))
                .clamp(0.0, 1.0);
        }
    }
    let mut previous = raw[0].clone();
    let mut smoothed = vec![vec![0.0; bin_count]; frame_count];
    for frame in 0..frame_count {
        for bin in 0..bin_count {
            previous[bin] = alpha * previous[bin] + (1.0 - alpha) * raw[frame][bin];
            smoothed[frame][bin] = previous[bin];
        }
    }
    smoothed
}

fn ema_alpha(hop_size: usize, sample_rate_hz: u32, smoothing_ms: f64) -> f64 {
    let tau_s = (smoothing_ms / 1000.0).max(1e-3);
    let hop_s = hop_size as f64 / sample_rate_hz as f64;
    (-hop_s / tau_s).exp()
}

fn percentile(values: &mut [f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let position = (values.len() - 1) as f64 * (percentile / 100.0);
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    if low == high {
        values[low]
    } else {
        let fraction = position - low as f64;
        values[low] * (1.0 - fraction) + values[high] * fraction
    }
}

fn vector_norm(vector: [f64; 3]) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

fn normalize_vec3_f64(vector: [f64; 3]) -> [f64; 3] {
    let norm = vector_norm(vector);
    if norm <= 1e-12 {
        [1.0, 0.0, 0.0]
    } else {
        [vector[0] / norm, vector[1] / norm, vector[2] / norm]
    }
}

fn frame_size_for_rate(sample_rate_hz: u32, frame_duration_ms: f64) -> usize {
    let target = (sample_rate_hz as f64 * frame_duration_ms / 1000.0)
        .round()
        .max(256.0) as usize;
    target.next_power_of_two()
}

fn scale_true_peak(mut channels: Vec<Vec<f32>>, target_peak: f64) -> Vec<Vec<f32>> {
    let peak = channels
        .iter()
        .flatten()
        .map(|sample| sample.abs() as f64)
        .fold(0.0f64, f64::max);
    if peak <= target_peak || peak <= 1e-12 {
        return channels;
    }
    let scale = (target_peak / peak) as f32;
    for sample in channels.iter_mut().flatten() {
        *sample *= scale;
    }
    channels
}

fn centroid_corrected_positions(positions: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let inv_count = 1.0 / positions.len().max(1) as f64;
    let centroid = positions.iter().fold([0.0; 3], |mut acc, position| {
        acc[0] += position[0] * inv_count;
        acc[1] += position[1] * inv_count;
        acc[2] += position[2] * inv_count;
        acc
    });
    positions
        .iter()
        .map(|position| {
            [
                position[0] - centroid[0],
                position[1] - centroid[1],
                position[2] - centroid[2],
            ]
        })
        .collect()
}

fn alias_cutoff_from_positions(positions: &[[f64; 3]]) -> f64 {
    let mut max_baseline_m = 0.0f64;
    for first_index in 0..positions.len() {
        for second_index in (first_index + 1)..positions.len() {
            let first = positions[first_index];
            let second = positions[second_index];
            let dx = first[0] - second[0];
            let dy = first[1] - second[1];
            let dz = first[2] - second[2];
            max_baseline_m = max_baseline_m.max((dx * dx + dy * dy + dz * dz).sqrt());
        }
    }
    if max_baseline_m < 1e-6 {
        SPEED_OF_SOUND_MPS / (2.0 * 0.05)
    } else {
        SPEED_OF_SOUND_MPS / (2.0 * max_baseline_m)
    }
}

fn apply_atob_matrix(
    a_freq: &[Vec<Complex64>],
    corrected_positions_m: &[[f64; 3]],
    freqs: &[f64],
) -> Vec<Vec<Complex64>> {
    let mic_count = corrected_positions_m.len();
    let bin_count = freqs.len();
    let mut b_freq = vec![vec![Complex64::new(0.0, 0.0); bin_count]; 4];
    for (bin_index, freq_hz) in freqs.iter().enumerate() {
        let encoding = build_pressure_gradient_matrix(corrected_positions_m, *freq_hz);
        let lambda = tikhonov_lambda(*freq_hz);
        let mut regularized = vec![vec![Complex64::new(0.0, 0.0); mic_count]; mic_count];
        for row in 0..mic_count {
            for col in 0..mic_count {
                let mut sum = Complex64::new(0.0, 0.0);
                for component in 0..4 {
                    sum += encoding[row][component] * encoding[col][component].conj();
                }
                if row == col {
                    sum += Complex64::new(lambda, 0.0);
                }
                regularized[row][col] = sum;
            }
        }
        let rhs: Vec<Complex64> = (0..mic_count).map(|mic| a_freq[mic][bin_index]).collect();
        let solved = solve_complex_linear_system(regularized, rhs);
        for component in 0..4 {
            let mut sum = Complex64::new(0.0, 0.0);
            for mic in 0..mic_count {
                sum += encoding[mic][component].conj() * solved[mic];
            }
            b_freq[component][bin_index] = sum;
        }
    }
    b_freq
}

fn build_pressure_gradient_matrix(
    corrected_positions_m: &[[f64; 3]],
    freq_hz: f64,
) -> Vec<[Complex64; 4]> {
    let wavenumber = 2.0 * PI * freq_hz.max(0.0) / SPEED_OF_SOUND_MPS;
    corrected_positions_m
        .iter()
        .map(|position| {
            [
                Complex64::new(2.0f64.sqrt(), 0.0),
                Complex64::new(0.0, -wavenumber * position[0]),
                Complex64::new(0.0, -wavenumber * position[1]),
                Complex64::new(0.0, -wavenumber * position[2]),
            ]
        })
        .collect()
}

fn tikhonov_lambda(freq_hz: f64) -> f64 {
    let clamped = freq_hz.abs().max(TIKHONOV_F_REF_HZ);
    TIKHONOV_LAMBDA_0 * (TIKHONOV_F_REF_HZ / clamped).powi(2)
}

fn apply_alias_lowpass(b_freq: &mut [Vec<Complex64>], freqs: &[f64], cutoff_hz: f64) {
    for (bin_index, freq_hz) in freqs.iter().enumerate() {
        let ratio = freq_hz.abs() / cutoff_hz.max(1e-9);
        let gain = 1.0 / (1.0 + ratio.powi(8)).sqrt();
        for component in b_freq.iter_mut().take(4).skip(1) {
            component[bin_index] *= gain;
        }
    }
}

fn solve_complex_linear_system(
    mut matrix: Vec<Vec<Complex64>>,
    mut rhs: Vec<Complex64>,
) -> Vec<Complex64> {
    let n = rhs.len();
    for pivot_index in 0..n {
        let mut best_row = pivot_index;
        let mut best_norm = matrix[pivot_index][pivot_index].norm_sqr();
        for row in (pivot_index + 1)..n {
            let norm = matrix[row][pivot_index].norm_sqr();
            if norm > best_norm {
                best_norm = norm;
                best_row = row;
            }
        }
        if best_norm < 1e-24 {
            continue;
        }
        if best_row != pivot_index {
            matrix.swap(best_row, pivot_index);
            rhs.swap(best_row, pivot_index);
        }
        let pivot = matrix[pivot_index][pivot_index];
        let pivot_row = matrix[pivot_index].clone();
        let pivot_rhs = rhs[pivot_index];
        for row in (pivot_index + 1)..n {
            let factor = matrix[row][pivot_index] / pivot;
            for col in pivot_index..n {
                matrix[row][col] -= factor * pivot_row[col];
            }
            rhs[row] -= factor * pivot_rhs;
        }
    }

    let mut output = vec![Complex64::new(0.0, 0.0); n];
    for row in (0..n).rev() {
        let mut sum = rhs[row];
        for col in (row + 1)..n {
            sum -= matrix[row][col] * output[col];
        }
        let pivot = matrix[row][row];
        output[row] = if pivot.norm_sqr() > 1e-24 {
            sum / pivot
        } else {
            Complex64::new(0.0, 0.0)
        };
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_encoder_returns_foa_shape() {
        let n = 2048;
        let request = AmbisonicsRenderRequest {
            channels: vec![vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]],
            mic_positions_m: vec![
                [0.0, 0.050, 0.0],
                [0.0433, 0.025, 0.0],
                [0.0, 0.0, 0.0],
                [0.02165, 0.025, 0.04082],
            ],
            sample_rate_hz: 16_000,
            block_size: 1024,
            hop_size: 512,
        };
        let output = atob_foa_linear(request);
        assert_eq!(output.bformat.len(), 4);
        assert_eq!(output.bformat[0].len(), n);
        assert!(output.bformat.iter().flatten().all(|sample| sample.is_finite()));
    }

    #[test]
    fn parametric_encoder_returns_foa_shape() {
        let n = 2048;
        let request = AmbisonicsRenderRequest {
            channels: vec![vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]],
            mic_positions_m: vec![
                [0.0, 0.050, 0.0],
                [0.0433, 0.025, 0.0],
                [0.0, 0.0, 0.0],
                [0.02165, 0.025, 0.04082],
            ],
            sample_rate_hz: 16_000,
            block_size: 1024,
            hop_size: 512,
        };
        let output = encode_ambisonics(
            request,
            &AmbisonicsProfile::from_name("parametric_v2").unwrap(),
        );
        assert_eq!(output.bformat.len(), 4);
        assert_eq!(output.bformat[0].len(), n);
        assert!(output.bformat.iter().flatten().all(|sample| sample.is_finite()));
    }
}
