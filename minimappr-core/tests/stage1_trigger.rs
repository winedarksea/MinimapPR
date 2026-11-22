//! Stage-1 trigger: fires on faint calls relative to the noise floor, stays
//! quiet on stationary noise at any absolute level, respects mode bands, and
//! noise fill never writes zeros.

mod common;

use common::{decibels_to_amplitude, SeededGaussianNoise, DEVICE_SAMPLE_RATE_HZ};
use minimappr_core::stage1::{
    blank_span_with_noise_fill, DeterministicNoiseSource, Stage1Configuration, Stage1Counters,
    Stage1DetectionMode, Stage1Detector, Stage1Event, DEFAULT_NOISE_FILL_CONFIGURATION,
};

const NOISE_RMS_DBFS: f64 = -50.0;
const BLOCK_SAMPLES: usize = 480;
const STREAM_FIRST_SAMPLE_INDEX: u64 = 77_000_000;
const HOP_SAMPLES: u64 = 960;

struct RunResult {
    events: Vec<Stage1Event>,
    counters: Stage1Counters,
}

fn run_detector(mode: Stage1DetectionMode, signal: &[f64]) -> RunResult {
    let mut detector = Stage1Detector::new(Stage1Configuration::for_mode(mode));
    let mut events = Vec::new();
    for (block_index, block) in signal.chunks(BLOCK_SAMPLES).enumerate() {
        let samples: Vec<f32> = block.iter().map(|&value| value as f32).collect();
        events.extend(detector.process_block(
            STREAM_FIRST_SAMPLE_INDEX + (block_index * BLOCK_SAMPLES) as u64,
            &samples,
        ));
    }
    RunResult {
        events,
        counters: detector.counters(),
    }
}

fn trigger_open_sample_indices(events: &[Stage1Event]) -> Vec<u64> {
    events
        .iter()
        .filter_map(|event| match event {
            Stage1Event::TriggerOpened {
                device_sample_index,
                ..
            } => Some(*device_sample_index),
            _ => None,
        })
        .collect()
}

fn onset_sample_indices(events: &[Stage1Event]) -> Vec<u64> {
    events
        .iter()
        .filter_map(|event| match event {
            Stage1Event::Onset(onset) => Some(onset.device_sample_index),
            _ => None,
        })
        .collect()
}

fn stationary_noise(seconds: f64, rms_dbfs: f64, seed: u64) -> Vec<f64> {
    SeededGaussianNoise::new(seed).samples(
        (seconds * DEVICE_SAMPLE_RATE_HZ) as usize,
        decibels_to_amplitude(rms_dbfs),
    )
}

/// Adds tone bursts; returns burst start offsets (samples).
fn add_tone_bursts(
    signal: &mut [f64],
    tone_hz: f64,
    amplitude: f64,
    first_start_seconds: f64,
    burst_seconds: f64,
    period_seconds: f64,
    burst_count: usize,
) -> Vec<usize> {
    let burst_length = (burst_seconds * DEVICE_SAMPLE_RATE_HZ) as usize;
    let ramp_length = (0.005 * DEVICE_SAMPLE_RATE_HZ) as usize;
    (0..burst_count)
        .map(|burst_index| {
            let start = ((first_start_seconds + burst_index as f64 * period_seconds)
                * DEVICE_SAMPLE_RATE_HZ) as usize;
            for offset in 0..burst_length {
                let ramp =
                    (offset.min(burst_length - 1 - offset) as f64 / ramp_length as f64).min(1.0);
                let phase =
                    2.0 * std::f64::consts::PI * tone_hz * offset as f64 / DEVICE_SAMPLE_RATE_HZ;
                signal[start + offset] += ramp * amplitude * phase.sin();
            }
            start
        })
        .collect()
}

#[test]
fn stationary_noise_never_triggers_at_any_absolute_level() {
    for (rms_dbfs, seed) in [(-30.0, 1), (NOISE_RMS_DBFS, 2), (-90.0, 3)] {
        for mode in [Stage1DetectionMode::Birds, Stage1DetectionMode::Drone] {
            let result = run_detector(mode, &stationary_noise(20.0, rms_dbfs, seed));
            assert_eq!(
                result.counters.triggers_opened, 0,
                "{mode:?} at {rms_dbfs} dBFS: {:?}",
                result.counters
            );
            assert!(result.counters.frames_rejected > 900);
            assert!(
                result.counters.onsets_detected <= 2,
                "{mode:?} at {rms_dbfs} dBFS: {} onsets in noise",
                result.counters.onsets_detected
            );
        }
    }
}

/// 4 kHz tone bursts whose RMS is `call_below_noise_db` below the broadband
/// noise RMS, run at an overall gain; returns (burst starts, result).
fn tone_bursts_in_noise(call_below_noise_db: f64, overall_gain_db: f64) -> (Vec<usize>, RunResult) {
    let call_amplitude =
        decibels_to_amplitude(NOISE_RMS_DBFS - call_below_noise_db) * std::f64::consts::SQRT_2;
    let mut signal = stationary_noise(8.0, NOISE_RMS_DBFS, 11);
    let burst_starts = add_tone_bursts(&mut signal, 4_000.0, call_amplitude, 3.0, 0.15, 1.0, 4);
    let gain = decibels_to_amplitude(overall_gain_db);
    signal.iter_mut().for_each(|value| *value *= gain);
    (
        burst_starts,
        run_detector(Stage1DetectionMode::Birds, &signal),
    )
}

#[test]
fn faint_call_below_broadband_noise_rms_triggers_at_any_absolute_level() {
    // Call RMS 10 dB *below* the broadband noise RMS: only ~9 dB above its
    // own sub-band floor, as a distant bird is. The -40 dB run proves there
    // is no absolute level gate.
    for overall_gain_db in [0.0, -40.0] {
        let (burst_starts, result) = tone_bursts_in_noise(10.0, overall_gain_db);
        let opens = trigger_open_sample_indices(&result.events);
        assert!(
            !opens.is_empty(),
            "gain {overall_gain_db} dB: faint call did not trigger"
        );
        let first_burst_index = STREAM_FIRST_SAMPLE_INDEX + burst_starts[0] as u64;
        assert!(
            opens[0] >= first_burst_index && opens[0] <= first_burst_index + 6 * HOP_SAMPLES,
            "trigger at {} vs burst {first_burst_index}",
            opens[0]
        );
    }
}

#[test]
fn moderate_calls_get_one_onset_per_note() {
    for overall_gain_db in [0.0, -40.0] {
        let (burst_starts, result) = tone_bursts_in_noise(5.0, overall_gain_db);
        let onsets = onset_sample_indices(&result.events);
        for &burst_start in &burst_starts {
            let burst_index = STREAM_FIRST_SAMPLE_INDEX + burst_start as u64;
            assert!(
                onsets.iter().any(|&onset| onset + 2 * HOP_SAMPLES >= burst_index && onset <= burst_index + 2 * HOP_SAMPLES),
                "gain {overall_gain_db}: no onset within 2 hops of burst at {burst_index}; onsets {onsets:?}"
            );
        }
        assert!(
            onsets.len() <= burst_starts.len() + 2,
            "spurious onsets: {onsets:?}"
        );
    }
}

#[test]
fn opened_trigger_reports_the_active_band_of_the_call() {
    let call_amplitude = decibels_to_amplitude(NOISE_RMS_DBFS - 5.0) * std::f64::consts::SQRT_2;
    let mut signal = stationary_noise(5.0, NOISE_RMS_DBFS, 12);
    add_tone_bursts(&mut signal, 6_000.0, call_amplitude, 3.0, 0.3, 1.0, 1);
    let result = run_detector(Stage1DetectionMode::Birds, &signal);
    let (low, high) = result
        .events
        .iter()
        .find_map(|event| match event {
            Stage1Event::TriggerOpened {
                active_band_low_hz,
                active_band_high_hz,
                ..
            } => Some((*active_band_low_hz, *active_band_high_hz)),
            _ => None,
        })
        .expect("triggered");
    assert!(
        low <= 6_000.0 && high >= 6_000.0 && high - low < 2_000.0,
        "band {low}..{high}"
    );
}

#[test]
fn fixture_whistle_mixed_faintly_into_noise_triggers_inside_the_song() {
    let whistle = common::load_pcm16_mono_wav(
        &common::field_audio_fixture_directory().join("wtsp_whistle_xc569523.wav"),
    );
    let whistle_float: Vec<f64> = whistle
        .iter()
        .map(|&sample| f64::from(sample) / 32_768.0)
        .collect();
    let whistle_rms = common::root_mean_square(&whistle_float);
    // Song RMS 20 dB below the noise RMS.
    let whistle_gain = decibels_to_amplitude(NOISE_RMS_DBFS - 20.0) / whistle_rms;
    let lead_in_samples = (4.0 * DEVICE_SAMPLE_RATE_HZ) as usize;
    let mut signal = stationary_noise(
        4.0 + whistle_float.len() as f64 / DEVICE_SAMPLE_RATE_HZ,
        NOISE_RMS_DBFS,
        13,
    );
    for (offset, &sample) in whistle_float.iter().enumerate() {
        signal[lead_in_samples + offset] += whistle_gain * sample;
    }
    let result = run_detector(Stage1DetectionMode::Birds, &signal);
    let opens = trigger_open_sample_indices(&result.events);
    let song_start = STREAM_FIRST_SAMPLE_INDEX + lead_in_samples as u64;
    assert!(
        !opens.is_empty(),
        "faint whistle missed: {:?}",
        result.counters
    );
    assert!(
        opens.iter().all(|&open| open >= song_start),
        "trigger before the song: {opens:?}"
    );
}

#[test]
fn mode_bands_separate_drone_rotor_tones_from_birds() {
    let mut signal = stationary_noise(8.0, NOISE_RMS_DBFS, 14);
    let harmonic_amplitude = decibels_to_amplitude(NOISE_RMS_DBFS - 12.0);
    for harmonic in 1..=4 {
        add_tone_bursts(
            &mut signal,
            150.0 * harmonic as f64,
            harmonic_amplitude,
            3.0,
            3.0,
            10.0,
            1,
        );
    }
    let drone = run_detector(Stage1DetectionMode::Drone, &signal);
    assert!(
        drone.counters.triggers_opened >= 1,
        "drone mode missed rotor tones: {:?}",
        drone.counters
    );
    let birds = run_detector(Stage1DetectionMode::Birds, &signal);
    assert_eq!(
        birds.counters.triggers_opened, 0,
        "bird mode should ignore 150-600 Hz: {:?}",
        birds.counters
    );
}

#[test]
fn history_holds_thirty_seconds_of_quantized_bands_and_onsets_are_queryable() {
    let call_amplitude = decibels_to_amplitude(NOISE_RMS_DBFS - 3.0);
    let mut signal = stationary_noise(35.0, NOISE_RMS_DBFS, 15);
    let burst_starts = add_tone_bursts(&mut signal, 3_000.0, call_amplitude, 33.0, 0.1, 0.5, 2);
    let configuration = Stage1Configuration::for_mode(Stage1DetectionMode::Birds);
    assert!((32..=48).contains(&configuration.band_count));
    let mut detector = Stage1Detector::new(configuration.clone());
    for (block_index, block) in signal.chunks(BLOCK_SAMPLES).enumerate() {
        let samples: Vec<f32> = block.iter().map(|&value| value as f32).collect();
        detector.process_block(
            STREAM_FIRST_SAMPLE_INDEX + (block_index * BLOCK_SAMPLES) as u64,
            &samples,
        );
    }
    let history = detector.history();
    let held_frames = history.next_frame_index() - history.oldest_frame_index();
    assert_eq!(
        held_frames as usize,
        configuration.history_capacity_frames()
    );
    assert_eq!(held_frames, 1_500, "30 s at 20 ms hops");
    let newest = history
        .frame_levels(history.next_frame_index() - 1)
        .expect("newest held");
    assert_eq!(newest.len(), configuration.band_count);
    assert!(history.frame_levels(0).is_none(), "oldest frames evicted");
    let call_start = STREAM_FIRST_SAMPLE_INDEX + burst_starts[0] as u64;
    let onsets = detector.onsets_in_device_range(call_start - 2 * HOP_SAMPLES, call_start + 48_000);
    assert_eq!(onsets.len(), 2, "{onsets:?}");
}

#[test]
fn noise_fill_matches_floor_and_never_writes_zeros() {
    let noise_rms = decibels_to_amplitude(NOISE_RMS_DBFS);
    let mut detector =
        Stage1Detector::new(Stage1Configuration::for_mode(Stage1DetectionMode::Birds));
    let noise = stationary_noise(3.0, NOISE_RMS_DBFS, 16);
    let noise_f32: Vec<f32> = noise.iter().map(|&value| value as f32).collect();
    detector.process_block(0, &noise_f32);
    let fill_rms = detector.noise_fill_rms();
    assert!(
        (f64::from(fill_rms) / noise_rms - 1.0).abs() < 0.15,
        "floor rms {fill_rms} vs {noise_rms}"
    );

    // A loud own-chirp slot (14 kHz, inside the span's crossfade margins)
    // blanked out of the buffer.
    let mut buffer: Vec<f32> = noise_f32[..9_600].to_vec();
    let crossfade = DEFAULT_NOISE_FILL_CONFIGURATION.crossfade_samples;
    for (offset, sample) in buffer[2_400 + crossfade..7_200 - crossfade]
        .iter_mut()
        .enumerate()
    {
        *sample += 0.5 * (2.0 * std::f32::consts::PI * 14_000.0 * offset as f32 / 48_000.0).sin();
    }
    let mut noise_source = DeterministicNoiseSource::from_seed(99);
    blank_span_with_noise_fill(
        &mut buffer,
        2_400..7_200,
        fill_rms,
        &DEFAULT_NOISE_FILL_CONFIGURATION,
        &mut noise_source,
    );
    let interior: Vec<f64> = buffer[2_600..7_000]
        .iter()
        .map(|&value| f64::from(value))
        .collect();
    assert!(interior.iter().all(|&value| value != 0.0));
    assert!((common::root_mean_square(&interior) / f64::from(fill_rms) - 1.0).abs() < 0.1);
    let largest_step = buffer
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        largest_step < 0.05,
        "crossfade should prevent clicks, step {largest_step}"
    );

    // Digital silence: fill still has non-zero variance (BirdNET NaN guard).
    let mut silent = vec![0.0_f32; 4_800];
    blank_span_with_noise_fill(
        &mut silent,
        0..4_800,
        0.0,
        &DEFAULT_NOISE_FILL_CONFIGURATION,
        &mut noise_source,
    );
    let silent_f64: Vec<f64> = silent.iter().map(|&value| f64::from(value)).collect();
    assert!(common::root_mean_square(&silent_f64) > 0.5 / 32_768.0);
}

#[test]
fn stream_gap_is_counted_and_does_not_trigger() {
    let noise = stationary_noise(4.0, NOISE_RMS_DBFS, 17);
    let noise_f32: Vec<f32> = noise.iter().map(|&value| value as f32).collect();
    let mut detector =
        Stage1Detector::new(Stage1Configuration::for_mode(Stage1DetectionMode::Birds));
    detector.process_block(0, &noise_f32[..96_000]);
    let events = detector.process_block(96_000 + 4_800, &noise_f32[96_000..]);
    assert_eq!(detector.counters().stream_discontinuities, 1);
    assert!(trigger_open_sample_indices(&events).is_empty());
}
