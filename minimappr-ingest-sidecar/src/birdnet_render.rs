use num_complex::Complex32;
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug)]
pub struct HybridRenderConfig {
    pub spatial_band_hz: [f32; 2],
    pub pre_blend_highpass_hz: f32,
    pub sound_speed_mps: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ClassifierRenderPayload {
    pub render_id: String,
    pub render_kind: String,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub sample_count: usize,
    pub sample_format: String,
    pub effective_spatial_band: Option<[f32; 2]>,
    pub source_channel_count: usize,
    pub fallback_reason: Option<String>,
}

pub fn render_omni_pcm16le(channels: &[Vec<f32>]) -> Vec<u8> {
    let mono = omni_mix(channels);
    f32_to_pcm16le(&mono)
}

pub fn render_hybrid_pcm16le(
    channels: &[Vec<f32>],
    steering_direction: [f32; 3],
    mic_positions_m: &[[f32; 3]; 4],
    sample_rate_hz: u32,
    config: HybridRenderConfig,
) -> Vec<u8> {
    let omni = omni_mix(channels);
    let mut steered = delay_and_sum(
        channels,
        steering_direction,
        mic_positions_m,
        sample_rate_hz,
        config,
    );
    if config.pre_blend_highpass_hz > 0.0 {
        apply_one_pole_highpass(&mut steered, sample_rate_hz, config.pre_blend_highpass_hz);
    }
    let blended = frequency_blend(&omni, &steered, sample_rate_hz, config.spatial_band_hz);
    f32_to_pcm16le(&blended)
}

fn omni_mix(channels: &[Vec<f32>]) -> Vec<f32> {
    let frame_count = channels.iter().map(Vec::len).min().unwrap_or(0);
    if frame_count == 0 || channels.is_empty() {
        return Vec::new();
    }
    let scale = 1.0 / channels.len() as f32;
    let mut mono = vec![0.0_f32; frame_count];
    for channel in channels {
        for (out, sample) in mono.iter_mut().zip(channel.iter()) {
            *out += *sample * scale;
        }
    }
    mono
}

fn delay_and_sum(
    channels: &[Vec<f32>],
    steering_direction: [f32; 3],
    mic_positions_m: &[[f32; 3]; 4],
    sample_rate_hz: u32,
    config: HybridRenderConfig,
) -> Vec<f32> {
    let frame_count = channels.iter().take(4).map(Vec::len).min().unwrap_or(0);
    if frame_count == 0 {
        return Vec::new();
    }
    let arrival_offsets_s = mic_positions_m
        .map(|position| -dot(position, steering_direction) / config.sound_speed_mps.max(1.0));
    let earliest = arrival_offsets_s
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let delay_samples =
        arrival_offsets_s.map(|offset| ((offset - earliest) * sample_rate_hz as f32).max(0.0));

    let mut rendered = vec![0.0_f32; frame_count];
    for channel_index in 0..4 {
        for (sample_index, out) in rendered.iter_mut().enumerate() {
            *out += sample_at_fractional(
                &channels[channel_index],
                sample_index as f32 - delay_samples[channel_index],
            );
        }
    }
    for sample in &mut rendered {
        *sample *= 0.25;
    }
    rendered
}

fn frequency_blend(
    omni: &[f32],
    steered: &[f32],
    sample_rate_hz: u32,
    spatial_band_hz: [f32; 2],
) -> Vec<f32> {
    let n = omni.len().min(steered.len());
    if n == 0 {
        return Vec::new();
    }
    let fft_len = next_pow2(n);

    // Cache the FFT planner per Rayon thread to avoid heavy re-computation in the hot loop.
    thread_local! {
        static PLANNER: std::cell::RefCell<FftPlanner<f32>> = std::cell::RefCell::new(FftPlanner::new());
    }

    let (fft, ifft) = PLANNER.with(|planner| {
        let mut p = planner.borrow_mut();
        (p.plan_fft_forward(fft_len), p.plan_fft_inverse(fft_len))
    });

    let mut omni_spectrum = real_to_complex_padded(&omni[..n], fft_len);
    let mut steered_spectrum = real_to_complex_padded(&steered[..n], fft_len);
    fft.process(&mut omni_spectrum);
    fft.process(&mut steered_spectrum);

    let band_min = spatial_band_hz[0].max(0.0);
    let band_max = spatial_band_hz[1].max(band_min);
    for bin in 0..fft_len {
        let mirrored_bin = bin.min(fft_len - bin);
        let frequency_hz = mirrored_bin as f32 * sample_rate_hz as f32 / fft_len as f32;
        if frequency_hz >= band_min && frequency_hz <= band_max {
            omni_spectrum[bin] = steered_spectrum[bin];
        }
    }
    ifft.process(&mut omni_spectrum);
    let scale = 1.0 / fft_len as f32;
    omni_spectrum[..n]
        .iter()
        .map(|value| value.re * scale)
        .collect()
}

fn apply_one_pole_highpass(samples: &mut [f32], sample_rate_hz: u32, cutoff_hz: f32) {
    if samples.is_empty() || cutoff_hz <= 0.0 {
        return;
    }
    let dt = 1.0 / sample_rate_hz as f32;
    let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
    let alpha = rc / (rc + dt);
    let mut previous_input = samples[0];
    let mut previous_output = samples[0];
    for sample in samples.iter_mut().skip(1) {
        let input = *sample;
        let output = alpha * (previous_output + input - previous_input);
        *sample = output;
        previous_input = input;
        previous_output = output;
    }
}

fn f32_to_pcm16le(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        let pcm = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        bytes.extend_from_slice(&pcm.to_le_bytes());
    }
    bytes
}

fn real_to_complex_padded(samples: &[f32], fft_len: usize) -> Vec<Complex32> {
    samples
        .iter()
        .map(|sample| Complex32::new(*sample, 0.0))
        .chain(std::iter::repeat_n(Complex32::new(0.0, 0.0), fft_len - samples.len()))
        .collect()
}

fn sample_at_fractional(samples: &[f32], index: f32) -> f32 {
    if index <= 0.0 {
        return samples.first().copied().unwrap_or(0.0);
    }
    let lower = index.floor() as usize;
    let upper = lower + 1;
    if upper >= samples.len() {
        return samples.last().copied().unwrap_or(0.0);
    }
    let frac = index - lower as f32;
    samples[lower] * (1.0 - frac) + samples[upper] * frac
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

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omni_render_preserves_mono_pcm_shape() {
        let bytes = render_omni_pcm16le(&[vec![0.0, 0.5], vec![0.0, -0.5]]);
        assert_eq!(bytes.len(), 4);
        assert_eq!(i16::from_le_bytes([bytes[0], bytes[1]]), 0);
        assert_eq!(i16::from_le_bytes([bytes[2], bytes[3]]), 0);
    }
}
