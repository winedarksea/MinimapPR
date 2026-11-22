//! Arrival-time acoustic world: true positions, clocks, calls and chirps.
//!
//! This is the cheaper abstraction Layer 2 uses instead of rendering audio
//! (Layer 1): every call and chirp is reduced to a true arrival time per
//! device (straight-line flight at the speed of sound), mapped onto that
//! device's own sample clock (offset θ, drift α, minus samples lost in
//! capture stalls), plus an in-band SNR from 1/r spreading, ISO 9613-1
//! absorption (from `minimappr-sim`) and ambient noise. Thresholds on SNR
//! stand in for Stage-1, the classifier and clip correlation; timing noise
//! shrinks with SNR. Multipath, Doppler and waveform effects are Layer 1's.

use minimappr_sim::physics::atmosphere::AtmosphericConditions;
use minimappr_sim::physics::constants::speed_of_sound_mps;

use crate::device_spec::DeviceSpec;
use crate::seeded_random::{hashed_gaussian, hashed_uniform};

pub const DEVICE_SAMPLE_RATE_HZ: f64 = 48_000.0;
const PARTS_PER_MILLION: f64 = 1e6;
const MINIMUM_SPREADING_DISTANCE_M: f64 = 1.0;
const DECIBELS_PER_DECADE_SPHERICAL: f64 = 20.0;
const DECIBELS_PER_POWER_DECADE: f64 = 10.0;

#[derive(Debug, Clone, PartialEq)]
pub struct WorldConfig {
    pub atmosphere: AtmosphericConditions,
    pub ambient_bird_band_noise_db_spl: f64,
    pub ambient_chirp_band_noise_db_spl: f64,
    pub chirp_level_db_spl_at_1m: f64,
    pub chirp_centre_frequency_hz: f64,
    pub chirp_detection_minimum_snr_db: f64,
    pub chirp_pick_std_at_reference_s: f64,
    pub clip_timing_std_at_reference_s: f64,
    pub timing_reference_snr_db: f64,
    pub maximum_timing_std_s: f64,
    /// Stage-1 opens at ~7 dB contrast in-band (core stage1 notes).
    pub stage1_trigger_snr_db: f64,
    pub classifier_detection_snr_db: f64,
    /// Below this a clip gives no usable correlation peak.
    pub clip_correlation_minimum_snr_db: f64,
    pub classification_window_s: f64,
    pub classification_latency_after_arrival_s: f64,
    pub rolling_buffer_s: f64,
    pub maximum_clock_offset_s: f64,
    pub maximum_clock_drift_ppm: f64,
    pub walking_speed_threshold_mps: f64,
    pub walking_linear_acceleration_m_per_s2: f32,
    pub still_linear_acceleration_m_per_s2: f32,
    pub gnss_fix_interval_s: f64,
    pub sensor_frame_interval_s: f64,
    /// 24 kHz PCM16 after 1.75× FLAC ≈ 0.57 bytes per 48 kHz device sample.
    pub clip_bytes_per_device_sample: f64,
    /// Stereo phone heading (compass) error, common to all its bearings in a round.
    pub chirp_direction_heading_std_deg: f64,
    /// Independent DOA noise per bearing.
    pub chirp_direction_noise_std_deg: f64,
    /// Below this headroom the capture thread stalls now and then.
    pub stall_headroom_threshold: f64,
    pub stall_interval_range_s: [f64; 2],
    pub stall_duration_range_s: [f64; 2],
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            atmosphere: AtmosphericConditions::default(),
            ambient_bird_band_noise_db_spl: 30.0,
            ambient_chirp_band_noise_db_spl: 20.0,
            chirp_level_db_spl_at_1m: 85.0,
            chirp_centre_frequency_hz: 14_500.0,
            chirp_detection_minimum_snr_db: 6.0,
            chirp_pick_std_at_reference_s: 10e-6,
            clip_timing_std_at_reference_s: 20e-6,
            timing_reference_snr_db: 20.0,
            maximum_timing_std_s: 400e-6,
            stage1_trigger_snr_db: 10.0,
            classifier_detection_snr_db: 15.0,
            clip_correlation_minimum_snr_db: 0.0,
            classification_window_s: 3.0,
            classification_latency_after_arrival_s: 2.0,
            rolling_buffer_s: 30.0,
            maximum_clock_offset_s: 0.02,
            maximum_clock_drift_ppm: 60.0,
            walking_speed_threshold_mps: 0.3,
            walking_linear_acceleration_m_per_s2: 1.5,
            still_linear_acceleration_m_per_s2: 0.05,
            gnss_fix_interval_s: 5.0,
            sensor_frame_interval_s: 0.25,
            clip_bytes_per_device_sample: 0.57,
            chirp_direction_heading_std_deg: 20.0,
            chirp_direction_noise_std_deg: 5.0,
            stall_headroom_threshold: 0.05,
            stall_interval_range_s: [8.0, 20.0],
            stall_duration_range_s: [0.05, 0.3],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaptureStall {
    pub time_s: f64,
    pub duration_s: f64,
    /// Reported by an overrun counter, or silent (caught by the clock filter).
    pub reported: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScriptedCall {
    pub time_s: f64,
    pub position_m: [f64; 3],
    pub level_db_spl_at_1m: f64,
    pub species_label: u32,
    pub classifier_score: f32,
    pub centre_frequency_hz: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseKind {
    Speech,
    Handling,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseInterval {
    pub device: usize,
    pub start_s: f64,
    pub end_s: f64,
    pub kind: NoiseKind,
}

#[derive(Debug, Clone)]
pub struct DeviceTruth {
    pub spec: DeviceSpec,
    pub clock_offset_s: f64,
    pub clock_drift: f64,
    pub gps_bias_m: [f64; 2],
    pub stalls: Vec<CaptureStall>,
}

impl DeviceTruth {
    fn lost_seconds_before(&self, time_s: f64) -> f64 {
        self.stalls.iter().filter(|stall| stall.time_s < time_s).map(|stall| stall.duration_s).sum()
    }

    /// Sample n is taken at θ + n / (fs (1 + α)), shifted by lost stalls.
    pub fn sample_index_at(&self, time_s: f64) -> f64 {
        (time_s - self.clock_offset_s - self.lost_seconds_before(time_s)) * DEVICE_SAMPLE_RATE_HZ * (1.0 + self.clock_drift)
    }

    pub fn true_time_of_sample(&self, sample_index: f64) -> f64 {
        let base = sample_index / (DEVICE_SAMPLE_RATE_HZ * (1.0 + self.clock_drift)) + self.clock_offset_s;
        let mut time_s = base;
        for _ in 0..self.stalls.len() + 1 {
            time_s = base + self.lost_seconds_before(time_s);
        }
        time_s
    }
}

#[derive(Debug, Clone)]
pub struct AcousticWorld {
    pub seed: u64,
    pub config: WorldConfig,
    pub devices: Vec<DeviceTruth>,
    pub calls: Vec<ScriptedCall>,
    pub noise_intervals: Vec<NoiseInterval>,
    pub sound_speed_mps: f64,
}

pub fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn uniform_in(seed: u64, purpose: &str, a: u64, b: u64, range: [f64; 2]) -> f64 {
    range[0] + (range[1] - range[0]) * hashed_uniform(seed, purpose, a, b)
}

impl AcousticWorld {
    pub fn new(seed: u64, config: WorldConfig, specs: Vec<DeviceSpec>, calls: Vec<ScriptedCall>, noise_intervals: Vec<NoiseInterval>, duration_s: f64) -> Self {
        let devices = specs
            .into_iter()
            .enumerate()
            .map(|(index, spec)| {
                let index = index as u64;
                let symmetric = |purpose: &str, limit: f64| limit * (2.0 * hashed_uniform(seed, purpose, index, 0) - 1.0);
                let mut stalls = Vec::new();
                if let Some(ramp) = spec.thermal_ramp {
                    let mut time_s = ramp.start_s;
                    let mut stall_index = 0u64;
                    while time_s < duration_s {
                        time_s += uniform_in(seed, "stall_gap", index, stall_index, config.stall_interval_range_s);
                        if ramp.headroom_at(time_s) < config.stall_headroom_threshold && time_s < duration_s {
                            stalls.push(CaptureStall {
                                time_s,
                                duration_s: uniform_in(seed, "stall_length", index, stall_index, config.stall_duration_range_s),
                                reported: stall_index.is_multiple_of(2),
                            });
                        }
                        stall_index += 1;
                    }
                }
                DeviceTruth {
                    clock_offset_s: symmetric("clock_offset", config.maximum_clock_offset_s),
                    clock_drift: symmetric("clock_drift", config.maximum_clock_drift_ppm) / PARTS_PER_MILLION,
                    gps_bias_m: [
                        spec.gps_error_sigma_m * hashed_gaussian(seed, "gps_bias_east", index, 0),
                        spec.gps_error_sigma_m * hashed_gaussian(seed, "gps_bias_north", index, 0),
                    ],
                    spec,
                    stalls,
                }
            })
            .collect();
        Self {
            seed,
            sound_speed_mps: speed_of_sound_mps(config.atmosphere.temperature_c, config.atmosphere.relative_humidity_fraction),
            config,
            devices,
            calls,
            noise_intervals,
        }
    }

    pub fn position(&self, device: usize, time_s: f64) -> [f64; 3] {
        self.devices[device].spec.trajectory.position_at(time_s)
    }

    pub fn is_walking(&self, device: usize, time_s: f64) -> bool {
        self.devices[device].spec.trajectory.speed_at(time_s) > self.config.walking_speed_threshold_mps
    }

    /// Flight from a fixed source to a possibly moving receiver.
    pub fn arrival_time(&self, emission_time_s: f64, source_m: [f64; 3], device: usize) -> f64 {
        let mut arrival = emission_time_s + distance(source_m, self.position(device, emission_time_s)) / self.sound_speed_mps;
        for _ in 0..2 {
            arrival = emission_time_s + distance(source_m, self.position(device, arrival)) / self.sound_speed_mps;
        }
        arrival
    }

    fn received_snr_db(&self, level_db: f64, frequency_hz: f64, range_m: f64, noise_db: f64, extra_attenuation_db: f64) -> f64 {
        level_db
            - DECIBELS_PER_DECADE_SPHERICAL * range_m.max(MINIMUM_SPREADING_DISTANCE_M).log10()
            - self.config.atmosphere.absorption_db_per_m(frequency_hz) * range_m
            - noise_db
            - extra_attenuation_db
    }

    pub fn call_snr_db(&self, call: usize, device: usize) -> f64 {
        let call = &self.calls[call];
        let range = distance(call.position_m, self.position(device, call.time_s));
        self.received_snr_db(
            call.level_db_spl_at_1m,
            call.centre_frequency_hz,
            range,
            self.config.ambient_bird_band_noise_db_spl,
            self.devices[device].spec.placement_attenuation_db,
        )
    }

    pub fn chirp_snr_db(&self, emitter: usize, listener: usize, time_s: f64) -> f64 {
        if self.devices[emitter].spec.muted_speaker {
            return f64::NEG_INFINITY;
        }
        let range = if emitter == listener {
            self.devices[emitter].spec.self_path_length_m
        } else {
            distance(self.position(emitter, time_s), self.position(listener, time_s))
        };
        self.received_snr_db(
            self.config.chirp_level_db_spl_at_1m,
            self.config.chirp_centre_frequency_hz,
            range,
            self.config.ambient_chirp_band_noise_db_spl,
            self.devices[listener].spec.placement_attenuation_db,
        )
    }

    /// Timing σ that shrinks as 1/√SNR from its value at the reference SNR.
    pub fn timing_std_s(&self, snr_db: f64, std_at_reference_s: f64) -> f64 {
        let ratio = DECIBELS_PER_POWER_DECADE.powf((self.config.timing_reference_snr_db - snr_db) / (2.0 * DECIBELS_PER_POWER_DECADE));
        (std_at_reference_s * ratio).min(self.config.maximum_timing_std_s)
    }

    pub fn noise_overlaps(&self, device: usize, start_s: f64, end_s: f64, kind: NoiseKind) -> bool {
        self.noise_intervals
            .iter()
            .any(|interval| interval.device == device && interval.kind == kind && interval.start_s < end_s && interval.end_s > start_s)
    }
}
