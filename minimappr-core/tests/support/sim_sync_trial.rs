//! One simulated chirp round through the whole sync pipeline.
//!
//! Builds a `minimappr-sim` scene of phones, replaces the simulator's chirp
//! emissions with `minimappr-core` waveforms (so the matched-filter
//! template is exactly what was played; per-phone codes are possible),
//! renders it, then runs first-arrival picking → BeepBeep ranging (three
//! rate compensations) → clock graph → layout and scores each stage
//! against simulator ground truth. Shared by the sync scenario tests and
//! the Phase 1 pre-study example (included there with `#[path]`).
#![allow(dead_code)]

use std::sync::Arc;

use minimappr_core::sync::chirp_codes::SessionChirpCodebook;
use minimappr_core::sync::chirp_waveform::{
    generate_plain_chirp, generate_up_down_chirp, normalize_to_unit_active_rms, ChirpBand,
    ChirpWaveform, ChirpWaveformKind, SweepLaw,
};
use minimappr_core::sync::first_arrival::{
    raw_threshold_onset, ArrivalContext, FirstArrivalConfig,
};
use minimappr_core::sync::split_part_detector::SplitPartDetector;
use minimappr_sim::physics::constants::pascals_from_db_spl;
use minimappr_sim::physics::geometry::{add, distance, Vec3};
use minimappr_sim::scenario::devices::{
    ClockConfig, DeviceConfig, DeviceKind, Grip, SecondMicConfig, SpeakerConfig,
};
use minimappr_sim::scenario::{ChirpScheduleConfig, RandomTrunksConfig, Scenario};
use minimappr_sim::world::{plan_world, PlannedEmission, WorldPlan};

/// Session seed for coded chirps in every trial (codes, not geometry, vary by it).
pub const TRIAL_SESSION_SEED: u64 = 0x5EED_C0DE;
/// The phone's coarse knowledge of when a slot starts (clock anchor), ±s.
pub const COARSE_SCHEDULE_UNCERTAINTY_S: f64 = 0.005;
/// Sound needs this long after the last chirp before the scene ends.
const SCENE_TAIL_S: f64 = 0.4;
const ROUND_START_S: f64 = 0.3;
/// Far mic sits this far beyond the primary mic, away from the speaker.
pub const FAR_MIC_EXTRA_DISTANCE_M: f64 = 0.15;
/// Fixed-point iterations for the retarded-time truth.
const TRUTH_ITERATIONS: usize = 6;
/// Raw-threshold onset baseline: first sample above this fraction of peak.
pub const RAW_ONSET_FRACTION_OF_PEAK: f32 = 0.2;

#[derive(Clone, Debug)]
pub struct SyncTrialConfig {
    pub seed: u64,
    pub phone_positions_m: Vec<Vec3>,
    /// Constant velocity of each phone during the round (zero = still).
    pub phone_velocities_mps: Vec<Vec3>,
    pub band: ChirpBand,
    pub waveform: ChirpWaveformKind,
    pub chirp_duration_s: f64,
    pub slot_duration_s: f64,
    pub chirp_level_db_spl_at_1m: f64,
    pub closing_chirp: bool,
    pub use_far_mic: bool,
    pub grip: Grip,
    pub ambient_noise_db_spl: f64,
    pub random_trunks: bool,
    /// Phones' mic low-pass corner (weak OEM paths roll off at ~16 kHz).
    pub mic_lowpass_hz: Option<f64>,
    pub mic_lowpass_order: usize,
}

impl SyncTrialConfig {
    pub fn baseline(seed: u64, phone_positions_m: Vec<Vec3>) -> Self {
        let count = phone_positions_m.len();
        Self {
            seed,
            phone_positions_m,
            phone_velocities_mps: vec![[0.0; 3]; count],
            band: minimappr_core::sync::chirp_waveform::CHIRP_BAND_15_TO_18_KHZ,
            waveform: ChirpWaveformKind::HyperbolicFm,
            chirp_duration_s: 0.05,
            slot_duration_s: 0.25,
            chirp_level_db_spl_at_1m: 85.0,
            closing_chirp: true,
            use_far_mic: false,
            grip: Grip::Hand,
            ambient_noise_db_spl: 35.0,
            random_trunks: true,
            mic_lowpass_hz: Some(20_000.0),
            mic_lowpass_order: 4,
        }
    }

    pub fn phone_count(&self) -> usize {
        self.phone_positions_m.len()
    }

    fn emission_order(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.phone_count()).collect();
        if self.closing_chirp {
            order.push(0);
        }
        order
    }

    pub fn round_mid_time_s(&self) -> f64 {
        ROUND_START_S + 0.5 * self.slot_duration_s * self.emission_order().len() as f64
    }
}

/// Waveform per phone (coded chirps differ by slot).
pub fn templates_for(config: &SyncTrialConfig, sample_rate_hz: f64) -> Vec<ChirpWaveform> {
    let codebook = SessionChirpCodebook::new(
        TRIAL_SESSION_SEED,
        config.band,
        config.chirp_duration_s,
        minimappr_core::sync::chirp_codes::DEFAULT_CODE_SEGMENT_COUNT,
        SweepLaw::HyperbolicFm,
    );
    (0..config.phone_count())
        .map(|phone| match config.waveform {
            ChirpWaveformKind::LinearFm => ChirpWaveform::single_part(generate_plain_chirp(
                config.band,
                config.chirp_duration_s,
                SweepLaw::LinearFm,
                sample_rate_hz,
            )),
            ChirpWaveformKind::HyperbolicFm => ChirpWaveform::single_part(generate_plain_chirp(
                config.band,
                config.chirp_duration_s,
                SweepLaw::HyperbolicFm,
                sample_rate_hz,
            )),
            ChirpWaveformKind::UpDownHyperbolic => generate_up_down_chirp(
                config.band,
                config.chirp_duration_s,
                SweepLaw::HyperbolicFm,
                sample_rate_hz,
            ),
            ChirpWaveformKind::SessionCodedUpDown => {
                codebook.waveform_for_slot(phone, sample_rate_hz)
            }
        })
        .collect()
}

fn build_scenario(config: &SyncTrialConfig) -> Scenario {
    let emissions = config.emission_order().len();
    let duration_s = ROUND_START_S + config.slot_duration_s * emissions as f64 + SCENE_TAIL_S;
    let devices = config
        .phone_positions_m
        .iter()
        .zip(&config.phone_velocities_mps)
        .enumerate()
        .map(|(index, (position, velocity))| {
            let end = add(
                *position,
                [
                    velocity[0] * duration_s,
                    velocity[1] * duration_s,
                    velocity[2] * duration_s,
                ],
            );
            let mut mic_response = minimappr_sim::scenario::devices::MicResponseConfig::default();
            mic_response.lowpass_hz = config.mic_lowpass_hz;
            mic_response.lowpass_order = config.mic_lowpass_order;
            DeviceConfig {
                device_id: format!("phone{index}"),
                kind: DeviceKind::Phone,
                position_m: *position,
                trajectory: Some(vec![
                    [0.0, position[0], position[1], position[2]],
                    [duration_s, end[0], end[1], end[2]],
                ]),
                clock: Some(ClockConfig::Random {
                    max_offset_s: 0.02,
                    max_drift_ppm: 60.0,
                }),
                mic_response: Some(mic_response),
                second_mic: config.use_far_mic.then(|| SecondMicConfig {
                    offset_m: [0.0, FAR_MIC_EXTRA_DISTANCE_M, 0.0],
                    ..SecondMicConfig::default()
                }),
                speaker: SpeakerConfig {
                    grip: config.grip,
                    ..SpeakerConfig::default()
                },
                limiter: None,
            }
        })
        .collect();
    let mut scenario = Scenario {
        name: "sync_trial".into(),
        seed: config.seed,
        duration_s,
        devices,
        chirps: Some(ChirpScheduleConfig {
            round_start_times_s: vec![ROUND_START_S],
            slot_duration_s: config.slot_duration_s,
            level_db_spl_at_1m: config.chirp_level_db_spl_at_1m,
            ..ChirpScheduleConfig::default()
        }),
        ..Scenario::default()
    };
    scenario.ambient_noise.level_db_spl = config.ambient_noise_db_spl;
    if config.random_trunks {
        scenario.reflectors.random_trunks = Some(RandomTrunksConfig::default());
    }
    scenario
}

/// Swap chirp samples for core templates; append the closing chirp.
fn install_core_chirps(
    plan: &mut WorldPlan,
    config: &SyncTrialConfig,
    templates: &[ChirpWaveform],
) {
    let gain = pascals_from_db_spl(config.chirp_level_db_spl_at_1m) as f32;
    let leveled: Vec<Arc<Vec<f32>>> = templates
        .iter()
        .map(|t| {
            let mut samples = t.samples.clone();
            normalize_to_unit_active_rms(&mut samples);
            Arc::new(samples.iter().map(|s| s * gain).collect())
        })
        .collect();
    for emission in plan.emissions.iter_mut() {
        if let Some(phone) = emission.emitting_device_index {
            emission.samples_pa_at_1m = leveled[phone].clone();
        }
    }
    if config.closing_chirp {
        let opener = plan
            .emissions
            .iter()
            .find(|e| e.emitting_device_index == Some(0))
            .cloned();
        if let Some(mut closing) = opener {
            closing.emission_id.push_str("-closing");
            closing.start_time_s += config.slot_duration_s * config.phone_count() as f64;
            plan.emissions.push(closing);
        }
    }
}

/// Retarded-time true arrival of `emission` at `phone`'s ranging mic, as a
/// fractional sample on the phone's clock, plus the path length.
fn true_arrival(
    plan: &WorldPlan,
    emission: &PlannedEmission,
    phone: usize,
    mic_offset: Vec3,
) -> (f64, f64) {
    let device = &plan.devices[phone];
    let source = emission.trajectory.position_at(emission.start_time_s);
    let c = plan.speed_of_sound_mps;
    let mut arrival_time = emission.start_time_s;
    let mut length = 0.0;
    for _ in 0..TRUTH_ITERATIONS {
        length = distance(
            source,
            add(device.trajectory.position_at(arrival_time), mic_offset),
        );
        arrival_time = emission.start_time_s + length / c;
    }
    (
        device
            .clock
            .sample_of_true_time(arrival_time, plan.sample_rate_hz),
        length,
    )
}

/// Per-arrival pick record.
#[derive(Clone, Debug)]
pub struct ArrivalPick {
    pub receiver: usize,
    pub emission: usize,
    pub is_self: bool,
    pub detected_sample: Option<f64>,
    pub true_sample: f64,
    pub raw_onset_sample: Option<f64>,
    pub snr_db: Option<f64>,
}

impl ArrivalPick {
    pub fn error_us(&self, sample_rate_hz: f64) -> Option<f64> {
        self.detected_sample
            .map(|d| (d - self.true_sample) / sample_rate_hz * 1e6)
    }
}

/// Everything a trial produced, before scoring.
pub struct RenderedRound {
    /// Speaker position relative to the ranging mic, world frame, per phone.
    pub speaker_offsets_from_mic_m: Vec<Vec3>,
    pub config: SyncTrialConfig,
    pub plan: WorldPlan,
    pub picks: Vec<ArrivalPick>,
    pub emission_emitters: Vec<usize>,
    pub self_path_lengths_m: Vec<f64>,
    /// `path_lengths[receiver][emission]` true path lengths.
    pub path_lengths_m: Vec<Vec<f64>>,
}

pub fn render_and_pick(
    config: &SyncTrialConfig,
    detector_config: &FirstArrivalConfig,
) -> RenderedRound {
    let scenario = build_scenario(config);
    let mut plan = plan_world(&scenario).expect("scenario plans");
    let fs = plan.sample_rate_hz;
    let templates = templates_for(config, fs);
    install_core_chirps(&mut plan, config, &templates);
    let rendered = minimappr_sim::render::render_world(&plan);
    let mut detector_config = detector_config.clone();
    detector_config.air_sound_speed_mps = plan.speed_of_sound_mps;
    let detectors: Vec<SplitPartDetector> = templates
        .iter()
        .map(|t| SplitPartDetector::new(t, detector_config.clone(), fs))
        .collect();
    let emitters = config.emission_order();
    let mic_channel = usize::from(config.use_far_mic);
    let mic_offset = |phone: usize| plan.devices[phone].mic_offsets_m[mic_channel];
    let self_path_lengths_m: Vec<f64> = (0..config.phone_count())
        .map(|phone| {
            plan.devices[phone].speaker_to_mic_distance_m.unwrap_or(0.0)
                + if config.use_far_mic {
                    FAR_MIC_EXTRA_DISTANCE_M
                } else {
                    0.0
                }
        })
        .collect();
    let mut rng = minimappr_core::sync::chirp_codes::SplitMix64::new(config.seed ^ 0xA5A5);
    let mut picks = Vec::new();
    let mut path_lengths_m = vec![vec![0.0; emitters.len()]; config.phone_count()];
    for receiver in 0..config.phone_count() {
        let channel = &rendered[receiver].channels[mic_channel];
        for (emission_index, &emitter) in emitters.iter().enumerate() {
            let emission = plan
                .emissions
                .iter()
                .filter(|e| e.emitting_device_index == Some(emitter))
                .nth(usize::from(emission_index > 0 && emitter == 0))
                .expect("emission");
            let (true_sample, length) =
                true_arrival(&plan, emission, receiver, mic_offset(receiver));
            path_lengths_m[receiver][emission_index] = length;
            // The phone knows its slot only through the coarse clock anchor.
            let coarse_error_s = ((rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
                * 2.0
                * COARSE_SCHEDULE_UNCERTAINTY_S;
            let expected_start = plan.devices[receiver]
                .clock
                .sample_of_true_time(emission.start_time_s + coarse_error_s, fs);
            let maximum_flight_s = config.slot_duration_s
                - config.chirp_duration_s
                - 2.0 * COARSE_SCHEDULE_UNCERTAINTY_S;
            let window_start =
                (expected_start - 2.0 * COARSE_SCHEDULE_UNCERTAINTY_S * fs).max(0.0) as usize;
            let search_length =
                ((maximum_flight_s + 2.0 * COARSE_SCHEDULE_UNCERTAINTY_S) * fs) as usize;
            let slice_end = (window_start + search_length + templates[emitter].samples.len())
                .min(channel.len());
            let slice = &channel[window_start..slice_end];
            let context = if receiver == emitter {
                ArrivalContext::SelfRecording {
                    self_path_length_m: Some(self_path_lengths_m[receiver]),
                }
            } else {
                ArrivalContext::RemoteEmitter
            };
            let detection = detectors[emitter]
                .detect(slice, 0..search_length, context)
                .ok();
            let raw_onset =
                raw_threshold_onset(slice, 0..search_length, RAW_ONSET_FRACTION_OF_PEAK)
                    .map(|o| (window_start + o) as f64);
            picks.push(ArrivalPick {
                receiver,
                emission: emission_index,
                is_self: receiver == emitter,
                detected_sample: detection
                    .as_ref()
                    .map(|d| window_start as f64 + d.sample_index),
                true_sample,
                raw_onset_sample: raw_onset,
                snr_db: detection.map(|d| d.snr_db),
            });
        }
    }
    let speaker_offsets_from_mic_m = (0..config.phone_count())
        .map(|phone| {
            let speaker = plan.devices[phone].speakers[0].offset_m;
            let mic = mic_offset(phone);
            [
                speaker[0] - mic[0],
                speaker[1] - mic[1],
                speaker[2] - mic[2],
            ]
        })
        .collect();
    RenderedRound {
        speaker_offsets_from_mic_m,
        config: config.clone(),
        plan,
        picks,
        emission_emitters: emitters,
        self_path_lengths_m,
        path_lengths_m,
    }
}
