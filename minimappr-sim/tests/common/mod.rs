//! Shared helpers for physics tests: clean single-path scenes, band-limited
//! test signals, analytic fractional delays and sub-sample delay/frequency
//! estimators that do not share code with the renderer.
#![allow(dead_code)]

use std::f64::consts::TAU;
use std::sync::Arc;

use minimappr_sim::physics::geometry::Trajectory;
use minimappr_sim::scenario::devices::{
    ClockConfig, DeviceConfig, DeviceKind, LimiterConfig, MicResponseConfig,
};
use minimappr_sim::scenario::Scenario;
use minimappr_sim::world::{BirdCallRecord, EmissionRecord, PlannedEmission, WorldPlan};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rustfft::num_complex::Complex64;
use rustfft::FftPlanner;

pub const SAMPLE_RATE_HZ: f64 = 48_000.0;
pub const GOLDEN_RATIO_CONJUGATE: f64 = 0.618_033_988_749_894_9;

/// A phone with an ideal, noiseless, unclipped, flat receive chain.
pub fn ideal_receiver(device_id: &str, position_m: [f64; 3], clock: ClockConfig) -> DeviceConfig {
    DeviceConfig {
        device_id: device_id.into(),
        kind: DeviceKind::Phone,
        position_m,
        trajectory: None,
        clock: Some(clock),
        mic_response: Some(MicResponseConfig {
            highpass_hz: None,
            lowpass_hz: None,
            self_noise_db_spl: -200.0,
            ..MicResponseConfig::default()
        }),
        second_mic: None,
        speaker: Default::default(),
        limiter: Some(LimiterConfig {
            enabled: false,
            ..LimiterConfig::default()
        }),
    }
}

/// Scene with no ambient noise, no ground, no absorption (opt back in per test).
pub fn clean_scenario(duration_s: f64, devices: Vec<DeviceConfig>) -> Scenario {
    let mut scenario = Scenario {
        duration_s,
        devices,
        ..Scenario::default()
    };
    scenario.ambient_noise.enabled = false;
    scenario.ground.enabled = false;
    scenario.environment.absorption_enabled = false;
    // Exact per-path absorption filters (no length-bucket sharing).
    scenario.render.absorption_sharing_tolerance_db = 0.0;
    scenario
}

/// Inject a custom source waveform (Pa at 1 m) into a planned world.
pub fn push_emission(
    plan: &mut WorldPlan,
    samples: Vec<f32>,
    start_time_s: f64,
    trajectory: Trajectory,
) {
    let index = plan.emissions.len();
    plan.emissions.push(PlannedEmission {
        emission_id: format!("test-{index}"),
        samples_pa_at_1m: Arc::new(samples),
        start_time_s,
        trajectory,
        emitting_device_index: None,
        record: EmissionRecord::BirdCall(BirdCallRecord {
            event_id: format!("test-{index}"),
            individual_id: "test".into(),
            species: vec!["test".into()],
            label: "test".into(),
            label_category: "bird".into(),
            generator_kind: "test".into(),
            level_db_spl_at_1m: 0.0,
            origin: "test",
        }),
    });
}

fn fft(buffer: &mut [Complex64], inverse: bool) {
    let mut planner = FftPlanner::<f64>::new();
    let plan = if inverse {
        planner.plan_fft_inverse(buffer.len())
    } else {
        planner.plan_fft_forward(buffer.len())
    };
    plan.process(buffer);
}

/// White noise strictly band-limited to (0, band_edge_fraction × fs] with
/// zero-padded silent margins of `margin` samples each side.
pub fn band_limited_noise(
    length: usize,
    margin: usize,
    band_edge_fraction_of_fs: f64,
    seed: u64,
) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let total = (length + 2 * margin).next_power_of_two();
    let mut buffer: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); total];
    for slot in buffer.iter_mut().skip(margin).take(length) {
        *slot = Complex64::new(
            minimappr_sim::signals::noise::standard_normal_approx(&mut rng) as f64,
            0.0,
        );
    }
    // Hann-taper the burst so it is smooth in time as well.
    for i in 0..length {
        let w = 0.5 - 0.5 * (TAU * i as f64 / (length - 1) as f64).cos();
        buffer[margin + i] *= w;
    }
    fft(&mut buffer, false);
    let edge_bin = (band_edge_fraction_of_fs * total as f64) as usize;
    for (bin, value) in buffer.iter_mut().enumerate() {
        let frequency_bin = bin.min(total - bin);
        if frequency_bin > edge_bin || frequency_bin == 0 {
            *value = Complex64::new(0.0, 0.0);
        }
    }
    fft(&mut buffer, true);
    buffer
        .iter()
        .map(|c| (c.re / total as f64) as f32)
        .collect()
}

/// Exact (circular) fractional delay of a band-limited signal, output length
/// `output_length`, via frequency-domain phase shift.
pub fn analytic_delay(signal: &[f32], delay_samples: f64, output_length: usize) -> Vec<f64> {
    let total = (signal.len().max(output_length) * 2).next_power_of_two();
    let mut buffer: Vec<Complex64> = (0..total)
        .map(|i| Complex64::new(signal.get(i).copied().unwrap_or(0.0) as f64, 0.0))
        .collect();
    fft(&mut buffer, false);
    for (bin, value) in buffer.iter_mut().enumerate() {
        let signed_bin = if bin <= total / 2 {
            bin as f64
        } else {
            bin as f64 - total as f64
        };
        let phase = -TAU * signed_bin * delay_samples / total as f64;
        *value *= Complex64::from_polar(1.0, phase);
    }
    fft(&mut buffer, true);
    buffer
        .iter()
        .take(output_length)
        .map(|c| c.re / total as f64)
        .collect()
}

/// Continuous cross-correlation R(τ) = Σ_k Re{X*(k) Y(k) e^{j2πkτ/N}}, the
/// band-limited interpolation of the sampled correlation.
pub struct ContinuousCrossCorrelation {
    cross_spectrum: Vec<Complex64>,
}

impl ContinuousCrossCorrelation {
    pub fn new(reference: &[f64], received: &[f64]) -> Self {
        let total = (reference.len().max(received.len()) * 2).next_power_of_two();
        let mut x: Vec<Complex64> = (0..total)
            .map(|i| Complex64::new(reference.get(i).copied().unwrap_or(0.0), 0.0))
            .collect();
        let mut y: Vec<Complex64> = (0..total)
            .map(|i| Complex64::new(received.get(i).copied().unwrap_or(0.0), 0.0))
            .collect();
        fft(&mut x, false);
        fft(&mut y, false);
        Self {
            cross_spectrum: x.iter().zip(&y).map(|(a, b)| a.conj() * b).collect(),
        }
    }

    pub fn at(&self, lag_samples: f64) -> f64 {
        let total = self.cross_spectrum.len();
        self.cross_spectrum
            .iter()
            .enumerate()
            .map(|(bin, value)| {
                let signed_bin = if bin <= total / 2 {
                    bin as f64
                } else {
                    bin as f64 - total as f64
                };
                (value * Complex64::from_polar(1.0, TAU * signed_bin * lag_samples / total as f64))
                    .re
            })
            .sum::<f64>()
            / total as f64
    }

    /// Integer-lag argmax over [low, high], then golden-section refinement.
    pub fn peak_lag(&self, low: i64, high: i64) -> f64 {
        let best = (low..=high)
            .max_by(|a, b| self.at(*a as f64).total_cmp(&self.at(*b as f64)))
            .unwrap();
        golden_section_maximum(
            |lag| self.at(lag),
            best as f64 - 1.0,
            best as f64 + 1.0,
            1e-6,
        )
    }
}

pub fn golden_section_maximum(
    f: impl Fn(f64) -> f64,
    mut low: f64,
    mut high: f64,
    tolerance: f64,
) -> f64 {
    let mut a = high - GOLDEN_RATIO_CONJUGATE * (high - low);
    let mut b = low + GOLDEN_RATIO_CONJUGATE * (high - low);
    let (mut fa, mut fb) = (f(a), f(b));
    while (high - low).abs() > tolerance {
        if fa > fb {
            high = b;
            b = a;
            fb = fa;
            a = high - GOLDEN_RATIO_CONJUGATE * (high - low);
            fa = f(a);
        } else {
            low = a;
            a = b;
            fa = fb;
            b = low + GOLDEN_RATIO_CONJUGATE * (high - low);
            fb = f(b);
        }
    }
    0.5 * (low + high)
}

/// Hann-windowed DTFT magnitude of `samples` at `frequency_hz`.
pub fn windowed_dtft_magnitude(samples: &[f32], frequency_hz: f64, sample_rate_hz: f64) -> f64 {
    let n = samples.len() as f64;
    let (mut re, mut im) = (0.0, 0.0);
    for (i, sample) in samples.iter().enumerate() {
        let window = 0.5 - 0.5 * (TAU * i as f64 / (n - 1.0)).cos();
        let phase = TAU * frequency_hz * i as f64 / sample_rate_hz;
        re += *sample as f64 * window * phase.cos();
        im -= *sample as f64 * window * phase.sin();
    }
    re.hypot(im)
}

/// Frequency of the dominant tone near `guess_hz` (± `search_hz`).
pub fn dominant_frequency_hz(
    samples: &[f32],
    guess_hz: f64,
    search_hz: f64,
    sample_rate_hz: f64,
) -> f64 {
    let steps = 400;
    let coarse = (0..=steps)
        .map(|i| guess_hz - search_hz + 2.0 * search_hz * i as f64 / steps as f64)
        .max_by(|a, b| {
            windowed_dtft_magnitude(samples, *a, sample_rate_hz)
                .total_cmp(&windowed_dtft_magnitude(samples, *b, sample_rate_hz))
        })
        .unwrap();
    let step = 2.0 * search_hz / steps as f64;
    golden_section_maximum(
        |f| windowed_dtft_magnitude(samples, f, sample_rate_hz),
        coarse - step,
        coarse + step,
        1e-4,
    )
}

/// Hann-windowed sinc pulse (low-pass at `cutoff_fraction_of_fs`) with a
/// compact, sidelobe-free autocorrelation, centred in a 4×half-width buffer.
pub fn band_limited_impulse(half_width_samples: usize, cutoff_fraction_of_fs: f64) -> Vec<f32> {
    let length = 4 * half_width_samples + 1;
    let center = (length / 2) as f64;
    (0..length)
        .map(|i| {
            let x = i as f64 - center;
            if x.abs() > half_width_samples as f64 {
                return 0.0;
            }
            let argument = TAU * cutoff_fraction_of_fs * x;
            let sinc = if x == 0.0 {
                1.0
            } else {
                argument.sin() / argument
            };
            let window = 0.5 + 0.5 * (std::f64::consts::PI * x / half_width_samples as f64).cos();
            (sinc * window) as f32
        })
        .collect()
}

pub fn tone(frequency_hz: f64, duration_s: f64, sample_rate_hz: f64) -> Vec<f32> {
    let length = (duration_s * sample_rate_hz) as usize;
    (0..length)
        .map(|i| (TAU * frequency_hz * i as f64 / sample_rate_hz).sin() as f32)
        .collect()
}
