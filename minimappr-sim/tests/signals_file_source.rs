//! File-based sources: any-rate, multi-channel WAV → mono at the sim rate.

mod common;

use std::f64::consts::TAU;
use std::path::Path;

use common::{dominant_frequency_hz, SAMPLE_RATE_HZ};
use minimappr_sim::signals::calls::{CallGenerator, WavFileParams};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const FILE_RATE_HZ: u32 = 44_100;
const TONE_HZ: f64 = 3000.0;
const FILE_DURATION_S: f64 = 0.5;

#[test]
fn stereo_44k1_int_wav_loads_as_resampled_unit_rms_mono() {
    let directory = Path::new(env!("CARGO_TARGET_TMPDIR"));
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: FILE_RATE_HZ,
        bits_per_sample: 24,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(directory.join("tone_44k1.wav"), spec).unwrap();
    let frames = (FILE_DURATION_S * FILE_RATE_HZ as f64) as usize;
    let full_scale = (1 << 23) as f64 * 0.5;
    for i in 0..frames {
        let value = ((TAU * TONE_HZ * i as f64 / FILE_RATE_HZ as f64).sin() * full_scale) as i32;
        writer.write_sample(value).unwrap();
        writer.write_sample(value).unwrap();
    }
    writer.finalize().unwrap();

    let generator = CallGenerator::WavFile(WavFileParams {
        path: "tone_44k1.wav".into(),
        trim_start_s: 0.0,
        max_duration_s: None,
    });
    let samples = generator
        .generate(SAMPLE_RATE_HZ, directory, &mut ChaCha8Rng::seed_from_u64(0))
        .unwrap();
    let expected_length = (FILE_DURATION_S * SAMPLE_RATE_HZ) as usize;
    assert!((samples.len() as i64 - expected_length as i64).abs() <= 1);
    let middle = &samples[samples.len() / 4..3 * samples.len() / 4];
    let rms =
        (middle.iter().map(|s| (*s as f64).powi(2)).sum::<f64>() / middle.len() as f64).sqrt();
    assert!((rms - 1.0).abs() < 0.01, "unit RMS, got {rms}");
    let frequency = dominant_frequency_hz(middle, TONE_HZ, 20.0, SAMPLE_RATE_HZ);
    assert!(
        (frequency - TONE_HZ).abs() < 0.05,
        "tone preserved, got {frequency}"
    );
}
