//! Renders a two-phone chirp-lab run to disk (WAV, session.json,
//! events.jsonl, run_manifest.json) with known ground truth: unrelated
//! monotonic epochs, ±60 ppm audio crystals, a 7.3 m separation, d_ii,
//! a body-conducted precursor on the self path, noise, timestamp jitter
//! and NTP-level wall-clock error. The analyser must recover the range to
//! 5 mm and the clock offset to a linear model within 2 µs.

use std::fs;
use std::path::{Path, PathBuf};

use minimappr_core::sync::chirp_waveform::{generate_up_down_chirp, ChirpBand, SweepLaw};
use minimappr_sync_spike::analysis_config::AnalysisConfig;
use minimappr_sync_spike::bundle_schema::BundleEvent;
use minimappr_sync_spike::bundle_validation::validate_run;
use minimappr_sync_spike::gate_evaluation::evaluate_run;
use minimappr_sync_spike::report_writer::write_run_outputs;
use minimappr_sync_spike::run_analysis::analyse_run;

const RATE_HZ: f64 = 48_000.0;
const OVERSAMPLING: f64 = 16.0;
const DISTANCE_M: f64 = 7.3;
const SELF_PATH_M: f64 = 0.02;
const TEMPERATURE_C: f64 = 20.0;
const PERIOD_S: f64 = 4.0;
const RUN_S: f64 = 130.0;
const BODY_SOUND_SPEED_MPS: f64 = 5_000.0;
/// Body precursor level relative to the airborne self arrival (hard table).
const PRECURSOR_GAIN: f32 = 0.2;

struct SyntheticPhone {
    directory_name: &'static str,
    role: &'static str,
    platform: &'static str,
    capture_start_true_s: f64,
    playback_start_true_s: f64,
    rate_ppm: f64,
    monotonic_offset_ns: i64,
    wall_error_ms: f64,
    schedule_offset_s: f64,
    truncate_wav_header: bool,
}

impl SyntheticPhone {
    fn rate_hz(&self) -> f64 {
        RATE_HZ * (1.0 + self.rate_ppm * 1e-6)
    }

    /// True time output frame `frame` leaves the speaker (shared crystal).
    fn playback_true_s(&self, frame: u64) -> f64 {
        self.playback_start_true_s + frame as f64 / self.rate_hz()
    }

    fn emission_frames(&self) -> Vec<u64> {
        let first = self.schedule_offset_s + 1.0;
        (0..).map(|k| first + k as f64 * PERIOD_S).take_while(|t| *t < RUN_S - 2.0).map(|t| (t * RATE_HZ) as u64).collect()
    }
}

fn speed_of_sound() -> f64 {
    331.3 * (1.0 + TEMPERATURE_C / 273.15).sqrt()
}

fn deterministic_noise(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.004
}

fn write_wav(path: &Path, samples: &[f32], truncate_header: bool) {
    let data_bytes = (samples.len() * 2) as u32;
    let declared = if truncate_header { 0 } else { data_bytes };
    let mut bytes = Vec::with_capacity(44 + samples.len() * 2);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(declared.wrapping_add(36)).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&48_000u32.to_le_bytes());
    bytes.extend_from_slice(&96_000u32.to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&declared.to_le_bytes());
    for sample in samples {
        bytes.extend_from_slice(&((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16).to_le_bytes());
    }
    fs::write(path, bytes).unwrap();
}

/// (true start s, path m, gain) of every chirp copy reaching `listener`.
fn arrivals_at(listener: usize, phones: &[SyntheticPhone], c: f64) -> Vec<(f64, f32)> {
    let mut arrivals = Vec::new();
    for (emitter_index, emitter) in phones.iter().enumerate() {
        for frame in emitter.emission_frames() {
            let emitted = emitter.playback_true_s(frame);
            if emitter_index == listener {
                arrivals.push((emitted + SELF_PATH_M / c, 0.5));
                arrivals.push((emitted + SELF_PATH_M / BODY_SOUND_SPEED_MPS, 0.5 * PRECURSOR_GAIN));
            } else {
                arrivals.push((emitted + DISTANCE_M / c, 0.06));
            }
        }
    }
    arrivals.sort_by(|a, b| a.0.total_cmp(&b.0));
    arrivals
}

fn render_phone(directory: &Path, index: usize, phones: &[SyntheticPhone], fine_template: &[f32]) {
    let phone = &phones[index];
    let c = speed_of_sound();
    let arrivals = arrivals_at(index, phones, c);
    let sample_count = ((RUN_S - phone.capture_start_true_s) * phone.rate_hz()) as usize;
    let template_s = fine_template.len() as f64 / (RATE_HZ * OVERSAMPLING);
    let mut noise_state = 0x9E37_79B9_7F4A_7C15 ^ index as u64;
    let mut samples = Vec::with_capacity(sample_count);
    let mut first_active = 0;
    for sample_index in 0..sample_count {
        let true_s = phone.capture_start_true_s + sample_index as f64 / phone.rate_hz();
        while first_active < arrivals.len() && arrivals[first_active].0 + template_s < true_s {
            first_active += 1;
        }
        let mut value = deterministic_noise(&mut noise_state);
        for (start, gain) in arrivals[first_active..].iter().take_while(|(start, _)| *start <= true_s) {
            let fine = ((true_s - start) * RATE_HZ * OVERSAMPLING).round() as usize;
            if fine < fine_template.len() {
                value += gain * fine_template[fine];
            }
        }
        samples.push(value);
    }
    let phone_directory = directory.join(phone.directory_name);
    fs::create_dir_all(&phone_directory).unwrap();
    write_wav(&phone_directory.join("capture.wav"), &samples, phone.truncate_wav_header);
    let true_to_local_ns = |true_s: f64| (true_s * 1e9) as i64 + phone.monotonic_offset_ns;
    let mut events = Vec::new();
    let mut jitter_state = 0x1234_5678 ^ index as u64;
    for frame in (0..sample_count as u64).step_by(48_000) {
        let true_s = phone.capture_start_true_s + frame as f64 / phone.rate_hz();
        let jitter_ns = (deterministic_noise(&mut jitter_state) * 5e7) as i64; // ±100 µs
        events.push(BundleEvent::CaptureTimestamp {
            t_ns: true_to_local_ns(true_s),
            frame_position: frame,
            nano_time: true_to_local_ns(true_s) + jitter_ns,
            wall_ms: Some((true_s * 1e3 + 1.7e12 + phone.wall_error_ms) as i64),
        });
    }
    for (emission_index, frame) in phone.emission_frames().into_iter().enumerate() {
        let presented = phone.playback_true_s(frame);
        events.push(BundleEvent::Emission {
            t_ns: true_to_local_ns(presented - 0.3),
            emission_index: emission_index as u64,
            output_frame_index: frame,
            target_ns: Some(true_to_local_ns(presented) + 400_000),
            played: true,
            channel: Some("right".into()),
        });
    }
    events.sort_by_key(BundleEvent::t_ns);
    let lines: Vec<String> = events.iter().map(BundleEvent::to_json_line).collect();
    fs::write(phone_directory.join("events.jsonl"), lines.join("\n") + "\n").unwrap();
    let session = serde_json::json!({
        "run_id": "synthetic", "role": phone.role, "mode": "chirp_lab", "platform_name": phone.platform,
        "device": {"manufacturer": "synthetic", "model": phone.platform, "sdk_int": 37, "fingerprint": "test"},
        "schedule": {"period_ms": (PERIOD_S * 1e3) as u64, "offset_ms": (phone.schedule_offset_s * 1e3) as u64, "lead_ms": 300,
            "waveform": {"kind": "updown_hfm", "band_low_hz": 13000.0, "band_high_hz": 16000.0, "duration_s": 0.05},
            "channel": "right", "level_dbfs": -6.0},
        "capture": {"requested_source": "unprocessed", "actual_source": "unprocessed", "buffer_size_frames": 960, "sample_rate_hz": 48000},
        "started_ns": true_to_local_ns(phone.capture_start_true_s),
    });
    fs::write(phone_directory.join("session.json"), serde_json::to_string_pretty(&session).unwrap()).unwrap();
}

fn render_run() -> PathBuf {
    let directory = std::env::temp_dir().join(format!("sync_spike_synthetic_{}", std::process::id()));
    let _ = fs::remove_dir_all(&directory);
    fs::create_dir_all(&directory).unwrap();
    let phones = [
        SyntheticPhone { directory_name: "pixel7a", role: "A", platform: "pixel_7a", capture_start_true_s: 0.21, playback_start_true_s: 0.43,
            rate_ppm: 60.0, monotonic_offset_ns: 9_876_543_210_123, wall_error_ms: 18.0, schedule_offset_s: 0.0, truncate_wav_header: false },
        SyntheticPhone { directory_name: "sm_a037f", role: "B", platform: "sm-a037f", capture_start_true_s: 0.87, playback_start_true_s: 1.02,
            rate_ppm: -60.0, monotonic_offset_ns: 123_456_789_000, wall_error_ms: -25.0, schedule_offset_s: 0.5 * PERIOD_S, truncate_wav_header: true },
    ];
    let band = ChirpBand { low_hz: 13_000.0, high_hz: 16_000.0 };
    let fine_template = generate_up_down_chirp(band, 0.05, SweepLaw::HyperbolicFm, RATE_HZ * OVERSAMPLING).samples;
    for index in 0..phones.len() {
        render_phone(&directory, index, &phones, &fine_template);
    }
    let manifest = serde_json::json!({
        "label": "synthetic_7m3", "matrix_run": "A5", "tape_distance_m": DISTANCE_M, "temperature_c": TEMPERATURE_C,
        "roles": {
            "A": {"serial": "synthetic-a", "dir": "pixel7a", "self_path_m": SELF_PATH_M},
            "B": {"serial": "synthetic-b", "dir": "sm_a037f", "self_path_m": SELF_PATH_M},
        },
    });
    fs::write(directory.join("run_manifest.json"), serde_json::to_string_pretty(&manifest).unwrap()).unwrap();
    directory
}

#[test]
fn synthetic_pair_recovers_range_and_clock() {
    let directory = render_run();
    let validation = validate_run(&directory);
    assert!(validation.is_usable(), "{validation:?}");
    assert!(validation.warnings.iter().any(|w| w.contains("WAV header")), "truncated header must be flagged: {validation:?}");

    let config = AnalysisConfig::default();
    let report = analyse_run(&directory, &config).expect("analysis");
    assert!(report.round_count >= 25, "only {} rounds", report.round_count);
    for phone in &report.phones {
        assert_eq!(phone.onset.missed, 0, "{} missed self chirps", phone.platform_name);
        assert!(!phone.blind_scanned);
    }
    for mode in ["double_sided", "tracked_rate_ratio"] {
        let stats = report.distance.iter().find(|d| d.mode == mode).expect(mode);
        let error = stats.error_vs_tape_m.unwrap();
        assert!(error.abs() < 0.005, "{mode}: range error {error} m ({stats:?})");
    }
    let uncompensated = report.distance.iter().find(|d| d.mode == "none").unwrap();
    assert!(uncompensated.error_vs_tape_m.unwrap().abs() > 0.005, "120 ppm over 2 s must bias the uncompensated range: {uncompensated:?}");
    let clock = report.clock.as_ref().expect("clock report");
    let linear_rms = clock.linear_fit.as_ref().unwrap().rms_s;
    assert!(linear_rms < 2e-6, "θ linear residual {} µs", linear_rms * 1e6);
    assert!((clock.rate_offset_ppm.unwrap() + 120.0).abs() < 0.5, "rate {:?} ppm", clock.rate_offset_ppm);
    assert!((report.implied_mean_self_path_m.unwrap() - SELF_PATH_M).abs() < 0.005);

    let gates = evaluate_run(&report, &config.gates);
    write_run_outputs(&report, &gates, &directory.join("analysis")).unwrap();
    assert!(directory.join("analysis/report.md").exists());
    if std::env::var_os("SYNC_SPIKE_PRINT_REPORT").is_some() {
        println!("{}", fs::read_to_string(directory.join("analysis/report.md")).unwrap());
    }
    fs::remove_dir_all(directory).ok();
}
