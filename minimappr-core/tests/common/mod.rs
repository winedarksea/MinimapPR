//! Shared helpers for codec / Stage-1 integration tests.
#![allow(dead_code)]

use std::path::PathBuf;

pub const DEVICE_SAMPLE_RATE_HZ: f64 = 48_000.0;

pub fn field_audio_fixture_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("field_audio")
}

/// All committed field-audio fixture WAVs (48 kHz mono PCM16), sorted.
pub fn field_audio_fixture_paths() -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(field_audio_fixture_directory())
        .expect("tests/fixtures/field_audio must exist (committed fixtures)")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|extension| extension == "wav"))
        .collect();
    paths.sort();
    paths
}

pub fn load_pcm16_mono_wav(path: &std::path::Path) -> Vec<i16> {
    let mut reader = hound::WavReader::open(path).expect("fixture WAV opens");
    let specification = reader.spec();
    assert_eq!(specification.channels, 1, "{path:?} must be mono");
    assert_eq!(specification.sample_rate, 48_000, "{path:?} must be 48 kHz");
    assert_eq!(specification.bits_per_sample, 16, "{path:?} must be PCM16");
    reader
        .samples::<i16>()
        .map(|sample| sample.expect("valid sample"))
        .collect()
}

/// Deterministic Gaussian noise (xorshift64* + Box-Muller).
pub struct SeededGaussianNoise {
    state: u64,
}

impl SeededGaussianNoise {
    pub fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_unit_open(&mut self) -> f64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let value = self.state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (value as f64 + 0.5) / (1_u64 << 53) as f64
    }

    pub fn next_standard_normal(&mut self) -> f64 {
        let radius = (-2.0 * self.next_unit_open().ln()).sqrt();
        radius * (2.0 * std::f64::consts::PI * self.next_unit_open()).cos()
    }

    pub fn samples(&mut self, count: usize, rms: f64) -> Vec<f64> {
        (0..count)
            .map(|_| rms * self.next_standard_normal())
            .collect()
    }
}

pub fn decibels_to_amplitude(decibels: f64) -> f64 {
    10_f64.powf(decibels / 20.0)
}

pub fn root_mean_square(samples: &[f64]) -> f64 {
    (samples.iter().map(|value| value * value).sum::<f64>() / samples.len() as f64).sqrt()
}
