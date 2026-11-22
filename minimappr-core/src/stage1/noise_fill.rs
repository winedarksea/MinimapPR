//! Noise-fill blanking: replace a span (own chirp slot, handling transient,
//! LOCO mask) with noise at the running noise floor, cross-faded with a
//! raised cosine. Never zeros: hard zeros put broadband clicks in the
//! classifier spectrogram, and BirdNET+ V3.0 returns NaN on zero variance.
//!
//! Randomness is a seeded xorshift so the caller (and tests) control it.

use std::f32::consts::PI;
use std::ops::Range;

/// Uniform noise on [-a, a] has RMS a / sqrt(3).
const UNIFORM_AMPLITUDE_PER_UNIT_RMS: f32 = 1.732_050_8;
/// Any non-zero constant works; xorshift has a fixed point at zero.
const XORSHIFT_NONZERO_SEED_REPLACEMENT: u64 = 0x9E37_79B9_7F4A_7C15;
const XORSHIFT_STAR_MULTIPLIER: u64 = 0x2545_F491_4F6C_DD1D;
const UNIT_INTERVAL_MANTISSA_BITS: u32 = 24;

/// Deterministic xorshift64* noise source.
#[derive(Debug, Clone)]
pub struct DeterministicNoiseSource {
    state: u64,
}

impl DeterministicNoiseSource {
    pub fn from_seed(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                XORSHIFT_NONZERO_SEED_REPLACEMENT
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state.wrapping_mul(XORSHIFT_STAR_MULTIPLIER)
    }

    /// Uniform in [-1, 1).
    pub fn next_symmetric_unit(&mut self) -> f32 {
        let mantissa = (self.next_u64() >> (u64::BITS - UNIT_INTERVAL_MANTISSA_BITS)) as f32;
        2.0 * mantissa / (1_u32 << UNIT_INTERVAL_MANTISSA_BITS) as f32 - 1.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseFillConfiguration {
    /// Raised-cosine crossfade length at each end, inside the span.
    pub crossfade_samples: usize,
    /// Fill RMS never goes below this (1 PCM16 LSB in float units).
    pub minimum_fill_rms: f32,
}

/// 2.5 ms crossfade at 48 kHz: long enough to avoid a click, short enough
/// to keep a 50-100 ms chirp slot mostly fill.
pub const DEFAULT_NOISE_FILL_CONFIGURATION: NoiseFillConfiguration = NoiseFillConfiguration {
    crossfade_samples: 120,
    minimum_fill_rms: 1.0 / 32_768.0,
};

/// Blanks `span` of `samples` with noise at `noise_floor_rms` (full scale
/// 1.0). The span is clamped to the buffer; crossfades shrink to fit.
pub fn blank_span_with_noise_fill(
    samples: &mut [f32],
    span: Range<usize>,
    noise_floor_rms: f32,
    configuration: &NoiseFillConfiguration,
    noise_source: &mut DeterministicNoiseSource,
) {
    let span_start = span.start.min(samples.len());
    let span_end = span.end.min(samples.len()).max(span_start);
    let span_length = span_end - span_start;
    if span_length == 0 {
        return;
    }
    let fill_amplitude =
        noise_floor_rms.max(configuration.minimum_fill_rms) * UNIFORM_AMPLITUDE_PER_UNIT_RMS;
    let crossfade_length = configuration.crossfade_samples.min(span_length / 2);
    for offset in 0..span_length {
        let distance_to_edge = offset.min(span_length - 1 - offset);
        let fill_weight = if distance_to_edge < crossfade_length {
            let fade_position = (distance_to_edge as f32 + 0.5) / crossfade_length as f32;
            0.5 - 0.5 * (PI * fade_position).cos()
        } else {
            1.0
        };
        let fill_sample = fill_amplitude * noise_source.next_symmetric_unit();
        let slot = &mut samples[span_start + offset];
        *slot = (1.0 - fill_weight) * *slot + fill_weight * fill_sample;
    }
}
