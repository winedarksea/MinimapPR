//! One simulated device: the core node state machine, an optional host
//! state machine (the device that currently hosts), and the shell-side
//! glue a phone app would provide — its rolling-buffer view of the
//! arrival-time world (`NodeAudioWindowSource`) and its chirp arrivals.

use std::collections::{BTreeMap, BTreeSet};

use minimappr_core::mesh::host_state_machine::FieldHostStateMachine;
use minimappr_core::mesh::node_state_machine::{EncodedLocalClip, FieldNodeStateMachine, LocalAudioReadiness, NodeAudioWindowSource};
use minimappr_core::protocol::detection_messages::SensorFrameSample;
use minimappr_core::protocol::session_messages::ChirpArrival;
use minimappr_core::protocol::wire_codes::{FrequencyBandHz, MotionState};

use crate::acoustic_world::{AcousticWorld, NoiseKind, DEVICE_SAMPLE_RATE_HZ};
use crate::seeded_random::hashed_gaussian;
use crate::synthetic_clip::{encode_synthetic_clip, SyntheticArrival};

/// In-band level reported for a window with no call in it (dB over floor).
const EMPTY_WINDOW_LEVEL_DB: f32 = -10.0;
const IDENTITY_QUATERNION_WXYZ: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

pub struct SimulatedDevice {
    pub index: usize,
    pub node: FieldNodeStateMachine,
    pub host: Option<FieldHostStateMachine>,
    pub reported_link_up: Option<bool>,
    pub reported_motion: Option<MotionState>,
    pub reported_headroom: Option<f64>,
    pub reported_stall_count: usize,
    pub last_gnss_fix_s: Option<f64>,
    pub gnss_fix_count: u64,
    pub own_emissions: BTreeMap<u32, f64>,
    pub listening_rounds: BTreeSet<u32>,
    pub round_arrivals: BTreeMap<u32, Vec<ChirpArrival>>,
    pub chirp_direction_enabled: bool,
}

impl SimulatedDevice {
    pub fn new(index: usize, node: FieldNodeStateMachine) -> Self {
        Self {
            index,
            node,
            host: None,
            reported_link_up: None,
            reported_motion: None,
            reported_headroom: None,
            reported_stall_count: 0,
            last_gnss_fix_s: None,
            gnss_fix_count: 0,
            own_emissions: BTreeMap::new(),
            listening_rounds: BTreeSet::new(),
            round_arrivals: BTreeMap::new(),
            chirp_direction_enabled: false,
        }
    }
}

/// The device's rolling buffer, seen through the arrival-time world.
pub struct DeviceAudioView<'world> {
    pub world: &'world AcousticWorld,
    pub device: usize,
    pub now_s: f64,
}

impl DeviceAudioView<'_> {
    fn window_true_times(&self, first_sample: f64, last_sample: f64) -> (f64, f64) {
        let truth = &self.world.devices[self.device];
        (truth.true_time_of_sample(first_sample), truth.true_time_of_sample(last_sample))
    }

    fn buffered(&self, start_s: f64, end_s: f64) -> bool {
        end_s <= self.now_s && self.now_s - start_s <= self.world.config.rolling_buffer_s
    }

    fn calls_in_window(&self, start_s: f64, end_s: f64) -> Vec<(usize, f64, f64)> {
        (0..self.world.calls.len())
            .filter_map(|call| {
                let spec = &self.world.calls[call];
                let arrival = self.world.arrival_time(spec.time_s, spec.position_m, self.device);
                (arrival >= start_s && arrival <= end_s).then(|| (call, arrival, self.world.call_snr_db(call, self.device)))
            })
            .collect()
    }
}

impl NodeAudioWindowSource for DeviceAudioView<'_> {
    fn readiness(&self, centre_device_sample_index: u64, half_window_samples: u32, _band: FrequencyBandHz) -> LocalAudioReadiness {
        let centre = centre_device_sample_index as f64;
        let half = half_window_samples as f64;
        let (start_s, end_s) = self.window_true_times(centre - half, centre + half);
        let level = self
            .calls_in_window(start_s, end_s)
            .iter()
            .map(|(_, _, snr)| *snr as f32)
            .fold(EMPTY_WINDOW_LEVEL_DB, f32::max);
        LocalAudioReadiness {
            audio_still_buffered: self.buffered(start_s, end_s),
            speech_free: !self.world.noise_overlaps(self.device, start_s, end_s, NoiseKind::Speech),
            handling_noise_free: !self.world.noise_overlaps(self.device, start_s, end_s, NoiseKind::Handling),
            in_band_level_over_floor_db: level,
        }
    }

    fn encode_clip(&self, window_first_device_sample_index: u64, window_sample_count: u32, _decimation_filter_id: u8) -> Option<EncodedLocalClip> {
        let first = window_first_device_sample_index as f64;
        let last = first + window_sample_count as f64;
        let (start_s, end_s) = self.window_true_times(first, last);
        if !self.buffered(start_s, end_s) {
            return None;
        }
        let truth = &self.world.devices[self.device];
        let config = &self.world.config;
        let arrivals: Vec<SyntheticArrival> = self
            .calls_in_window(start_s, end_s)
            .into_iter()
            .filter(|(_, _, snr)| *snr >= config.clip_correlation_minimum_snr_db)
            .map(|(call, arrival_s, snr)| {
                let timing_std_s = self.world.timing_std_s(snr, config.clip_timing_std_at_reference_s);
                let noise = hashed_gaussian(self.world.seed, "clip_timing", self.device as u64, call as u64);
                SyntheticArrival {
                    call_index: call as u32,
                    arrival_device_sample: truth.sample_index_at(arrival_s) + noise * timing_std_s * DEVICE_SAMPLE_RATE_HZ,
                    snr_db: snr as f32,
                    timing_std_s: timing_std_s as f32,
                    species_label: self.world.calls[call].species_label,
                    classifier_score: self.world.calls[call].classifier_score,
                    centre_frequency_hz: self.world.calls[call].centre_frequency_hz as f32,
                }
            })
            .collect();
        let padded_size = (window_sample_count as f64 * config.clip_bytes_per_device_sample) as usize;
        let mut sensor_frames = Vec::new();
        let mut frame_time = start_s;
        while frame_time <= end_s {
            let acceleration = if self.world.is_walking(self.device, frame_time) {
                config.walking_linear_acceleration_m_per_s2
            } else {
                config.still_linear_acceleration_m_per_s2
            };
            sensor_frames.push(SensorFrameSample {
                device_sample_index: truth.sample_index_at(frame_time).max(0.0) as u64,
                orientation_quaternion: IDENTITY_QUATERNION_WXYZ,
                linear_acceleration_m_per_s2: [acceleration, 0.0, 0.0],
                pressure_hectopascal: None,
            });
            frame_time += config.sensor_frame_interval_s;
        }
        Some(EncodedLocalClip {
            clip_container: encode_synthetic_clip(&arrivals, padded_size),
            onsets: Vec::new(),
            sensor_frames,
        })
    }
}
