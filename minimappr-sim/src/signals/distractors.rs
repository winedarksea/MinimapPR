//! Non-bird sources: synthetic speech babble and hand claps. Real recorded
//! conversation can replace the babble via a WAV path on the speech knob.

use std::f64::consts::TAU;

use rand::Rng;
use serde::{Deserialize, Serialize};

use super::calls::{edge_ramp_gain, normalize_active_rms};
use super::noise::{standard_normal_approx, uniform_in_range};
use crate::physics::biquad::{Biquad, BiquadCascade};

/// Vowel formant table (F1, F2) in Hz — cycles through /a/, /i/, /u/, /e/, /o/.
const VOWEL_FORMANTS_HZ: [(f64, f64); 5] = [
    (730.0, 1090.0),
    (270.0, 2290.0),
    (300.0, 870.0),
    (530.0, 1840.0),
    (570.0, 840.0),
];
const FORMANT_QUALITY_FACTOR: f64 = 6.0;
const GLOTTAL_HARMONIC_COUNT: usize = 30;
const SYLLABLE_EDGE_RAMP_FRACTION: f64 = 0.25;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpeechBabbleParams {
    pub duration_s: f64,
    pub pitch_low_hz: f64,
    pub pitch_high_hz: f64,
    pub syllables_per_second: f64,
    /// Fraction of syllable slots left silent (pauses between phrases).
    pub pause_fraction: f64,
}

impl Default for SpeechBabbleParams {
    fn default() -> Self {
        Self {
            duration_s: 8.0,
            pitch_low_hz: 100.0,
            pitch_high_hz: 220.0,
            syllables_per_second: 4.5,
            pause_fraction: 0.25,
        }
    }
}

/// Voiced babble: glottal harmonic stack through two formant band-passes,
/// syllable-gated. Spectrally speech-like enough to stress classifiers and
/// the "people nearby" path without shipping real voices.
pub fn speech_babble<R: Rng>(params: &SpeechBabbleParams, fs: f64, rng: &mut R) -> Vec<f32> {
    let total = (params.duration_s * fs) as usize;
    let syllable_length = (fs / params.syllables_per_second.max(f64::EPSILON)) as usize;
    let mut output = vec![0f32; total];
    let mut start = 0;
    let mut syllable_index = 0usize;
    while start < total {
        let length = syllable_length.min(total - start);
        if uniform_in_range(rng, 0.0, 1.0) >= params.pause_fraction {
            let pitch = uniform_in_range(rng, params.pitch_low_hz, params.pitch_high_hz);
            let (f1, f2) = VOWEL_FORMANTS_HZ[syllable_index % VOWEL_FORMANTS_HZ.len()];
            let mut voiced: Vec<f32> = (0..length)
                .map(|i| {
                    let t = i as f64 / fs;
                    (1..=GLOTTAL_HARMONIC_COUNT)
                        .filter(|h| pitch * (*h as f64) < fs / 2.0)
                        .map(|h| (TAU * pitch * h as f64 * t).sin() / h as f64)
                        .sum::<f64>() as f32
                })
                .collect();
            let mut first = voiced.clone();
            BiquadCascade::new(vec![Biquad::bandpass(f1, FORMANT_QUALITY_FACTOR, fs)])
                .process_in_place(&mut first);
            BiquadCascade::new(vec![Biquad::bandpass(f2, FORMANT_QUALITY_FACTOR, fs)])
                .process_in_place(&mut voiced);
            let ramp = (length as f64 * SYLLABLE_EDGE_RAMP_FRACTION) as usize;
            for i in 0..length {
                output[start + i] =
                    ((first[i] + voiced[i]) as f64 * edge_ramp_gain(i, length, ramp)) as f32;
            }
        }
        start += syllable_length.max(1);
        syllable_index += 1;
    }
    normalize_active_rms(&mut output);
    output
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClapParams {
    /// Exponential decay time constant of the noise burst.
    pub decay_time_constant_s: f64,
    pub duration_s: f64,
}

impl Default for ClapParams {
    fn default() -> Self {
        Self {
            decay_time_constant_s: 0.004,
            duration_s: 0.03,
        }
    }
}

/// Broadband impulse: exponentially decaying white-noise burst.
pub fn clap<R: Rng>(params: &ClapParams, fs: f64, rng: &mut R) -> Vec<f32> {
    let length = (params.duration_s * fs).max(1.0) as usize;
    let mut output: Vec<f32> = (0..length)
        .map(|i| {
            let t = i as f64 / fs;
            standard_normal_approx(rng) * (-t / params.decay_time_constant_s).exp() as f32
        })
        .collect();
    normalize_active_rms(&mut output);
    output
}
