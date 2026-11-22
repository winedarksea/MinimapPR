//! Synthetic dry bird-call generators. Stand-ins until the curated
//! xeno-canto fixtures land in `tests/fixtures/field_audio/` (use
//! `CallGenerator::WavFile` for those). Tonal whistles matter most: they
//! expose GCC-PHAT cycle skipping that broadband pings never trigger.
//!
//! Every generator returns mono samples at the sim rate normalised to unit
//! RMS over the active (non-gap) samples; the caller scales to pascals.

use std::f64::consts::TAU;
use std::path::Path;

use rand::Rng;
use serde::{Deserialize, Serialize};

use super::file_source::load_wav_mono_resampled;
use super::noise::standard_normal_approx;
use crate::physics::biquad::BiquadCascade;

/// Raised-cosine edge ramp on every note so onsets are click-free yet sharp.
pub const DEFAULT_NOTE_EDGE_RAMP_S: f64 = 0.005;
const BAND_LIMIT_FILTER_ORDER: usize = 4;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TonalWhistleParams {
    pub start_frequency_hz: f64,
    pub end_frequency_hz: f64,
    pub note_duration_s: f64,
    pub note_count: usize,
    pub note_gap_s: f64,
    /// Each successive note's frequencies multiply by this ratio.
    pub frequency_step_ratio: f64,
    pub vibrato_depth_hz: f64,
    pub vibrato_rate_hz: f64,
    pub edge_ramp_s: f64,
}

impl Default for TonalWhistleParams {
    fn default() -> Self {
        // Roughly a White-throated Sparrow "Old Sam Peabody" opening.
        Self {
            start_frequency_hz: 3200.0,
            end_frequency_hz: 3200.0,
            note_duration_s: 0.6,
            note_count: 3,
            note_gap_s: 0.08,
            frequency_step_ratio: 1.12,
            vibrato_depth_hz: 0.0,
            vibrato_rate_hz: 0.0,
            edge_ramp_s: DEFAULT_NOTE_EDGE_RAMP_S,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrillParams {
    pub carrier_frequency_hz: f64,
    /// Each element sweeps down by this much, centred on the carrier.
    pub element_sweep_hz: f64,
    pub element_rate_hz: f64,
    /// Fraction of each element period that is sounding.
    pub element_duty: f64,
    pub duration_s: f64,
    pub edge_ramp_s: f64,
}

impl Default for TrillParams {
    fn default() -> Self {
        Self {
            carrier_frequency_hz: 4500.0,
            element_sweep_hz: 1800.0,
            element_rate_hz: 16.0,
            element_duty: 0.6,
            duration_s: 1.6,
            edge_ramp_s: 0.002,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FmSweepParams {
    pub start_frequency_hz: f64,
    pub end_frequency_hz: f64,
    pub duration_s: f64,
    /// Exponential (log-frequency) instead of linear sweep.
    pub exponential: bool,
    pub edge_ramp_s: f64,
}

impl Default for FmSweepParams {
    fn default() -> Self {
        Self {
            start_frequency_hz: 6500.0,
            end_frequency_hz: 2500.0,
            duration_s: 0.25,
            exponential: false,
            edge_ramp_s: DEFAULT_NOTE_EDGE_RAMP_S,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HarshParams {
    /// Harmonic stack fundamental (jay/crow-like rasp).
    pub fundamental_hz: f64,
    pub harmonic_count: usize,
    pub noise_band_low_hz: f64,
    pub noise_band_high_hz: f64,
    /// 0 = pure harmonic stack, 1 = pure broadband noise.
    pub noise_mix: f64,
    /// Amplitude-modulation rate giving the rasp.
    pub roughness_modulation_hz: f64,
    pub duration_s: f64,
    pub edge_ramp_s: f64,
}

impl Default for HarshParams {
    fn default() -> Self {
        Self {
            fundamental_hz: 1100.0,
            harmonic_count: 7,
            noise_band_low_hz: 800.0,
            noise_band_high_hz: 9000.0,
            noise_mix: 0.4,
            roughness_modulation_hz: 70.0,
            duration_s: 0.45,
            edge_ramp_s: 0.003,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WavFileParams {
    /// Relative to the scenario file's directory (or absolute).
    pub path: String,
    #[serde(default)]
    pub trim_start_s: f64,
    #[serde(default)]
    pub max_duration_s: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CallGenerator {
    TonalWhistle(TonalWhistleParams),
    Trill(TrillParams),
    FmSweep(FmSweepParams),
    Harsh(HarshParams),
    WavFile(WavFileParams),
}

impl CallGenerator {
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::TonalWhistle(_) => "tonal_whistle",
            Self::Trill(_) => "trill",
            Self::FmSweep(_) => "fm_sweep",
            Self::Harsh(_) => "harsh",
            Self::WavFile(_) => "wav_file",
        }
    }

    /// Render at `sample_rate_hz`, unit RMS over active samples.
    pub fn generate<R: Rng>(
        &self,
        sample_rate_hz: f64,
        scenario_directory: &Path,
        rng: &mut R,
    ) -> Result<Vec<f32>, String> {
        let mut samples = match self {
            Self::TonalWhistle(p) => tonal_whistle(p, sample_rate_hz),
            Self::Trill(p) => trill(p, sample_rate_hz),
            Self::FmSweep(p) => fm_sweep(p, sample_rate_hz),
            Self::Harsh(p) => harsh(p, sample_rate_hz, rng),
            Self::WavFile(p) => {
                let path = scenario_directory.join(&p.path);
                let mut loaded = load_wav_mono_resampled(&path, sample_rate_hz)?;
                let skip = ((p.trim_start_s * sample_rate_hz) as usize).min(loaded.len());
                loaded.drain(..skip);
                if let Some(max_s) = p.max_duration_s {
                    loaded.truncate((max_s * sample_rate_hz) as usize);
                }
                loaded
            }
        };
        normalize_active_rms(&mut samples);
        Ok(samples)
    }
}

/// Raised-cosine gain for sample `index` of an `length`-sample note.
pub fn edge_ramp_gain(index: usize, length: usize, ramp_samples: usize) -> f64 {
    if ramp_samples == 0 || length == 0 {
        return 1.0;
    }
    let ramp = ramp_samples.min(length / 2).max(1);
    let distance_to_edge = index.min(length - 1 - index);
    if distance_to_edge >= ramp {
        1.0
    } else {
        0.5 - 0.5 * (std::f64::consts::PI * distance_to_edge as f64 / ramp as f64).cos()
    }
}

/// Scale so RMS over non-zero samples is 1 (gaps between notes don't count).
pub fn normalize_active_rms(samples: &mut [f32]) {
    let (sum_squares, active) = samples
        .iter()
        .filter(|s| **s != 0.0)
        .fold((0f64, 0usize), |(sum, count), s| {
            (sum + (*s as f64).powi(2), count + 1)
        });
    if active == 0 || sum_squares <= 0.0 {
        return;
    }
    let gain = (active as f64 / sum_squares).sqrt() as f32;
    samples.iter_mut().for_each(|s| *s *= gain);
}

/// Sinusoid with an arbitrary instantaneous-frequency law and edge ramps.
fn swept_tone(
    length: usize,
    sample_rate_hz: f64,
    ramp_samples: usize,
    frequency_at: impl Fn(f64) -> f64,
) -> Vec<f32> {
    let mut phase = 0.0;
    (0..length)
        .map(|index| {
            let time_s = index as f64 / sample_rate_hz;
            phase += TAU * frequency_at(time_s) / sample_rate_hz;
            (phase.sin() * edge_ramp_gain(index, length, ramp_samples)) as f32
        })
        .collect()
}

fn tonal_whistle(p: &TonalWhistleParams, fs: f64) -> Vec<f32> {
    let note_length = (p.note_duration_s * fs) as usize;
    let gap_length = (p.note_gap_s * fs) as usize;
    let ramp = (p.edge_ramp_s * fs) as usize;
    let mut output = Vec::new();
    for note in 0..p.note_count.max(1) {
        let ratio = p.frequency_step_ratio.powi(note as i32);
        let (start, end) = (p.start_frequency_hz * ratio, p.end_frequency_hz * ratio);
        let duration = p.note_duration_s.max(f64::EPSILON);
        output.extend(swept_tone(note_length, fs, ramp, |t| {
            start
                + (end - start) * t / duration
                + p.vibrato_depth_hz * (TAU * p.vibrato_rate_hz * t).sin()
        }));
        if note + 1 < p.note_count {
            output.extend(std::iter::repeat_n(0.0, gap_length));
        }
    }
    output
}

fn trill(p: &TrillParams, fs: f64) -> Vec<f32> {
    let total = (p.duration_s * fs) as usize;
    let period = (fs / p.element_rate_hz.max(f64::EPSILON)) as usize;
    let sounding = ((period as f64) * p.element_duty.clamp(0.0, 1.0)) as usize;
    let ramp = (p.edge_ramp_s * fs) as usize;
    let mut output = vec![0f32; total];
    let mut element_start = 0;
    while element_start < total {
        let length = sounding.min(total - element_start);
        let element_duration = length as f64 / fs;
        let high = p.carrier_frequency_hz + p.element_sweep_hz / 2.0;
        let element = swept_tone(length, fs, ramp, |t| {
            high - p.element_sweep_hz * t / element_duration.max(f64::EPSILON)
        });
        output[element_start..element_start + length].copy_from_slice(&element);
        element_start += period.max(1);
    }
    output
}

fn fm_sweep(p: &FmSweepParams, fs: f64) -> Vec<f32> {
    let length = (p.duration_s * fs) as usize;
    let duration = p.duration_s.max(f64::EPSILON);
    let ramp = (p.edge_ramp_s * fs) as usize;
    let (start, end) = (p.start_frequency_hz, p.end_frequency_hz);
    if p.exponential {
        let ratio = end / start;
        swept_tone(length, fs, ramp, |t| start * ratio.powf(t / duration))
    } else {
        swept_tone(length, fs, ramp, |t| start + (end - start) * t / duration)
    }
}

fn harsh<R: Rng>(p: &HarshParams, fs: f64, rng: &mut R) -> Vec<f32> {
    let length = (p.duration_s * fs) as usize;
    let ramp = (p.edge_ramp_s * fs) as usize;
    let nyquist_guard = fs / 2.0;
    let mut noise: Vec<f32> = (0..length).map(|_| standard_normal_approx(rng)).collect();
    let mut band =
        BiquadCascade::butterworth_highpass(BAND_LIMIT_FILTER_ORDER, p.noise_band_low_hz, fs);
    band.extend(BiquadCascade::butterworth_lowpass(
        BAND_LIMIT_FILTER_ORDER,
        p.noise_band_high_hz.min(nyquist_guard * 0.95),
        fs,
    ));
    band.process_in_place(&mut noise);
    normalize_active_rms(&mut noise);
    let harmonic_norm = (2.0 / p.harmonic_count.max(1) as f64).sqrt();
    (0..length)
        .map(|index| {
            let t = index as f64 / fs;
            let stack: f64 = (1..=p.harmonic_count)
                .filter(|h| (*h as f64) * p.fundamental_hz < nyquist_guard)
                .map(|h| (TAU * p.fundamental_hz * h as f64 * t).sin() / (h as f64).sqrt())
                .sum::<f64>()
                * harmonic_norm;
            let roughness = 0.5 + 0.5 * (TAU * p.roughness_modulation_hz * t).sin();
            let mixed = (1.0 - p.noise_mix) * stack + p.noise_mix * noise[index] as f64;
            (mixed * roughness * edge_ramp_gain(index, length, ramp)) as f32
        })
        .collect()
}
