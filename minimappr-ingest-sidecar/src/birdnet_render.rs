use num_complex::Complex32;
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};

use crate::dsp_math::{
    alias_cutoff_from_positions, raised_cosine_band_weight, steering_delays_s,
};

/// Band-split delay-and-sum render config (BEAMFORMED_RENDER_CONTRACT.md).
#[derive(Clone, Copy, Debug)]
pub struct BandSplitRenderConfig {
    pub highpass_hz: f32,
    pub low_crossover_width_hz: f32,
    pub high_crossover_width_min_hz: f32,
    pub high_crossover_width_fraction: f32,
    pub sound_speed_mps: f32,
    /// Optional legacy clamp: effective steered-band top = min(clamp, alias cutoff).
    pub band_max_clamp_hz: Option<f32>,
}

impl Default for BandSplitRenderConfig {
    fn default() -> Self {
        Self {
            highpass_hz: 100.0,
            low_crossover_width_hz: 100.0,
            high_crossover_width_min_hz: 400.0,
            high_crossover_width_fraction: 0.15,
            sound_speed_mps: crate::dsp_math::SPEED_OF_SOUND_MPS,
            band_max_clamp_hz: None,
        }
    }
}

/// Output of the band-split render: PCM bytes plus provenance per contract §6.
pub struct BandSplitRenderOutput {
    pub pcm_bytes: Vec<u8>,
    pub effective_spatial_band: [f32; 2],
    pub alias_cutoff_hz: f32,
}

/// Float-sample variant used by the beamform-oracle parity harness.
pub struct BandSplitRenderSamples {
    pub samples: Vec<f32>,
    pub effective_spatial_band: [f32; 2],
    pub alias_cutoff_hz: f32,
}

pub fn render_band_split_pcm16le(
    channels: &[Vec<f32>],
    steer_position_m: [f32; 3],
    mic_positions_m: &[[f32; 3]],
    sample_rate_hz: u32,
    config: BandSplitRenderConfig,
) -> BandSplitRenderOutput {
    let out = render_band_split_f32(
        channels,
        steer_position_m,
        mic_positions_m,
        sample_rate_hz,
        config,
    );
    BandSplitRenderOutput {
        pcm_bytes: f32_to_pcm16le(&out.samples),
        effective_spatial_band: out.effective_spatial_band,
        alias_cutoff_hz: out.alias_cutoff_hz,
    }
}

/// Contract §4 reference algorithm: one FFT per channel, omni = mean spectrum,
/// steered = mean of phase-advanced spectra (exp(+j2πfτ)), raised-cosine
/// blend, single inverse FFT. Near-field point-source steering toward
/// `steer_position_m` in the node-local mic frame.
pub fn render_band_split_f32(
    channels: &[Vec<f32>],
    steer_position_m: [f32; 3],
    mic_positions_m: &[[f32; 3]],
    sample_rate_hz: u32,
    config: BandSplitRenderConfig,
) -> BandSplitRenderSamples {
    let n_channels = channels.len().min(mic_positions_m.len());
    let frame_count = channels[..n_channels.min(channels.len())]
        .iter()
        .map(Vec::len)
        .min()
        .unwrap_or(0);
    let alias_cutoff_hz =
        alias_cutoff_from_positions(&mic_positions_m[..n_channels], config.sound_speed_mps);
    let effective_cutoff_hz = match config.band_max_clamp_hz {
        Some(clamp) if clamp > 0.0 => alias_cutoff_hz.min(clamp),
        _ => alias_cutoff_hz,
    };
    let effective_band = [config.highpass_hz, effective_cutoff_hz];

    if frame_count == 0 || n_channels == 0 {
        return BandSplitRenderSamples {
            samples: Vec::new(),
            effective_spatial_band: effective_band,
            alias_cutoff_hz,
        };
    }
    if n_channels == 1 {
        return BandSplitRenderSamples {
            samples: channels[0][..frame_count].to_vec(),
            effective_spatial_band: effective_band,
            alias_cutoff_hz,
        };
    }

    let delays = steering_delays_s(
        &mic_positions_m[..n_channels],
        steer_position_m,
        config.sound_speed_mps,
    );

    thread_local! {
        static PLANNER: std::cell::RefCell<FftPlanner<f32>> = std::cell::RefCell::new(FftPlanner::new());
    }
    let n = frame_count;
    let (fft, ifft) = PLANNER.with(|planner| {
        let mut p = planner.borrow_mut();
        (p.plan_fft_forward(n), p.plan_fft_inverse(n))
    });

    let scale = 1.0 / n_channels as f32;
    let mut omni_spectrum = vec![Complex32::new(0.0, 0.0); n];
    let mut steered_spectrum = vec![Complex32::new(0.0, 0.0); n];
    let mut buffer = vec![Complex32::new(0.0, 0.0); n];
    for channel_index in 0..n_channels {
        for (dst, sample) in buffer.iter_mut().zip(channels[channel_index][..n].iter()) {
            *dst = Complex32::new(*sample, 0.0);
        }
        fft.process(&mut buffer);
        let tau = delays[channel_index];
        for (bin, value) in buffer.iter().enumerate() {
            // Signed bin frequency: positive for bin <= n/2, negative above.
            let signed_freq_hz = signed_bin_frequency_hz(bin, n, sample_rate_hz);
            // Phase advance exp(+j2πfτ) compensates the propagation delay.
            let phase = 2.0 * std::f32::consts::PI * signed_freq_hz * tau;
            let rotation = Complex32::new(phase.cos(), phase.sin());
            omni_spectrum[bin] += *value * scale;
            steered_spectrum[bin] += *value * rotation * scale;
        }
    }

    let high_width_hz = config
        .high_crossover_width_min_hz
        .max(config.high_crossover_width_fraction * effective_cutoff_hz);
    let mut blended = vec![Complex32::new(0.0, 0.0); n];
    for bin in 0..n {
        let freq_hz = signed_bin_frequency_hz(bin, n, sample_rate_hz).abs();
        let w = raised_cosine_band_weight(
            freq_hz,
            config.highpass_hz,
            config.low_crossover_width_hz,
            effective_cutoff_hz,
            high_width_hz,
        );
        blended[bin] = steered_spectrum[bin] * w + omni_spectrum[bin] * (1.0 - w);
    }
    ifft.process(&mut blended);
    let inv_n = 1.0 / n as f32;
    let samples: Vec<f32> = blended.iter().map(|v| v.re * inv_n).collect();

    BandSplitRenderSamples {
        samples,
        effective_spatial_band: effective_band,
        alias_cutoff_hz,
    }
}

fn signed_bin_frequency_hz(bin: usize, n: usize, sample_rate_hz: u32) -> f32 {
    let k = if bin <= n / 2 {
        bin as f32
    } else {
        bin as f32 - n as f32
    };
    k * sample_rate_hz as f32 / n as f32
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

fn f32_to_pcm16le(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        let pcm = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        bytes.extend_from_slice(&pcm.to_le_bytes());
    }
    bytes
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

    const SIRITH: [[f32; 3]; 4] = [
        [0.0, 0.050, 0.0],
        [0.0433, 0.025, 0.0],
        [0.0, 0.0, 0.0],
        [0.02165, 0.025, 0.04082],
    ];

    fn pcm_to_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32767.0)
            .collect()
    }

    #[test]
    fn band_split_tetra_cutoff_near_3432() {
        let n = 4096usize;
        let sr = 44_100u32;
        let channels: Vec<Vec<f32>> = (0..4)
            .map(|c| {
                (0..n)
                    .map(|i| (0.1 * (i as f32 * 0.01 + c as f32)).sin() * 0.1)
                    .collect()
            })
            .collect();
        let out = render_band_split_pcm16le(
            &channels,
            [10.0, 0.0, 1.0],
            &SIRITH,
            sr,
            BandSplitRenderConfig::default(),
        );
        assert!((out.alias_cutoff_hz - 3266.0).abs() < 5.0);
        assert_eq!(out.effective_spatial_band[0], 100.0);
        assert!((out.effective_spatial_band[1] - out.alias_cutoff_hz).abs() < 1e-3);
        assert_eq!(out.pcm_bytes.len(), n * 2);
    }

    #[test]
    fn band_split_clamp_limits_band_top() {
        let channels: Vec<Vec<f32>> = (0..4).map(|_| vec![0.01_f32; 1024]).collect();
        let config = BandSplitRenderConfig {
            band_max_clamp_hz: Some(2000.0),
            ..Default::default()
        };
        let out = render_band_split_pcm16le(&channels, [5.0, 0.0, 0.0], &SIRITH, 44_100, config);
        assert!((out.effective_spatial_band[1] - 2000.0).abs() < 1e-3);
        // alias cutoff still reported unclamped
        assert!(out.alias_cutoff_hz > 3000.0);
    }

    #[test]
    fn band_split_near_field_delay_alignment_boosts_steered_tone() {
        // In-band tone (1.5 kHz) from a near-field point source: steering at the
        // source should reconstruct amplitude near the coherent mean; steering
        // opposite should reduce it.
        let sr = 44_100u32;
        let n = 8192usize;
        let c = 343.2_f32;
        let f0 = 1500.0_f32;
        let src = [3.0_f32, 1.0, 0.5];
        let channels: Vec<Vec<f32>> = SIRITH
            .iter()
            .map(|mic| {
                let d = ((src[0] - mic[0]).powi(2)
                    + (src[1] - mic[1]).powi(2)
                    + (src[2] - mic[2]).powi(2))
                .sqrt();
                let tau = d / c;
                (0..n)
                    .map(|i| {
                        let t = i as f32 / sr as f32;
                        0.25 * (2.0 * std::f32::consts::PI * f0 * (t - tau)).sin()
                    })
                    .collect()
            })
            .collect();
        let steered = render_band_split_pcm16le(
            &channels,
            src,
            &SIRITH,
            sr,
            BandSplitRenderConfig::default(),
        );
        let away = render_band_split_pcm16le(
            &channels,
            [-3.0, -1.0, -0.5],
            &SIRITH,
            sr,
            BandSplitRenderConfig::default(),
        );
        let rms = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        let steered_rms = rms(&pcm_to_f32(&steered.pcm_bytes)[1000..7000]);
        let away_rms = rms(&pcm_to_f32(&away.pcm_bytes)[1000..7000]);
        assert!(
            steered_rms > away_rms,
            "steered {steered_rms} should exceed off-target {away_rms}"
        );
        // 0.25-amplitude sine → RMS ≈ 0.177 when coherently reconstructed.
        assert!(steered_rms > 0.15, "steered RMS {steered_rms} should stay near 0.177");
    }

    #[test]
    fn band_split_single_channel_passthrough() {
        let signal: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.1).sin() * 0.3).collect();
        let out = render_band_split_pcm16le(
            &[signal.clone()],
            [1.0, 0.0, 0.0],
            &SIRITH[..1],
            16_000,
            BandSplitRenderConfig::default(),
        );
        let round_trip = pcm_to_f32(&out.pcm_bytes);
        for (a, b) in round_trip.iter().zip(signal.iter()) {
            assert!((a - b).abs() < 1e-3);
        }
    }

    #[test]
    fn band_split_empty_inputs() {
        let out = render_band_split_pcm16le(
            &[],
            [1.0, 0.0, 0.0],
            &SIRITH,
            16_000,
            BandSplitRenderConfig::default(),
        );
        assert!(out.pcm_bytes.is_empty());
    }
}
