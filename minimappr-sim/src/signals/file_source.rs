//! File-based dry sources: any-rate WAV (int or float, any channel count)
//! → mono → sim rate via band-limited windowed-sinc resampling.

use std::path::Path;

use crate::physics::fractional_delay::windowed_sinc;

/// Resampler kernel: taps per side at unity cutoff and Kaiser β.
const RESAMPLER_HALF_WIDTH_TAPS: f64 = 24.0;
const RESAMPLER_KAISER_BETA: f64 = 9.0;
/// Anti-alias cutoff margin below the lower of the two Nyquist rates.
const RESAMPLER_CUTOFF_FRACTION: f64 = 0.95;

/// Load `path`, average channels to mono, resample to `target_rate_hz`.
pub fn load_wav_mono_resampled(path: &Path, target_rate_hz: f64) -> Result<Vec<f32>, String> {
    let reader =
        hound::WavReader::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(|e| format!("read {}: {e}", path.display()))?,
        hound::SampleFormat::Int => {
            let full_scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 / full_scale))
                .collect::<Result<_, _>>()
                .map_err(|e| format!("read {}: {e}", path.display()))?
        }
    };
    let mono: Vec<f32> = interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    Ok(resample(&mono, spec.sample_rate as f64, target_rate_hz))
}

/// Arbitrary-ratio band-limited resampling (direct form; sources are short).
pub fn resample(input: &[f32], input_rate_hz: f64, output_rate_hz: f64) -> Vec<f32> {
    if (input_rate_hz - output_rate_hz).abs() < f64::EPSILON || input.is_empty() {
        return input.to_vec();
    }
    let ratio = output_rate_hz / input_rate_hz;
    let cutoff = RESAMPLER_CUTOFF_FRACTION * ratio.min(1.0);
    // Kernel stretches in input samples when downsampling.
    let half_width_input = RESAMPLER_HALF_WIDTH_TAPS / cutoff;
    let output_length = (input.len() as f64 * ratio).floor() as usize;
    (0..output_length)
        .map(|output_index| {
            let center = output_index as f64 / ratio;
            let first = (center - half_width_input).ceil().max(0.0) as usize;
            let last = ((center + half_width_input).floor() as usize).min(input.len() - 1);
            (first..=last)
                .map(|input_index| {
                    let offset = center - input_index as f64;
                    input[input_index] as f64
                        * cutoff
                        * windowed_sinc(
                            offset * cutoff,
                            RESAMPLER_HALF_WIDTH_TAPS,
                            1.0,
                            RESAMPLER_KAISER_BETA,
                        )
                })
                .sum::<f64>() as f32
        })
        .collect()
}
