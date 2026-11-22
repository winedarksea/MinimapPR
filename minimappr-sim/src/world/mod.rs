//! World planning: resolves a `Scenario` (plus its seeded random draws and
//! chaos knobs) into concrete devices, emissions and reflectors. The
//! renderer consumes only a `WorldPlan`, so every random choice is made —
//! and recorded for ground truth — exactly once, here.

mod device_planning;
mod emission_planning;
mod knob_sources;
pub mod motion;

use std::sync::Arc;

use crate::physics::atmosphere::AtmosphericConditions;
use crate::physics::biquad::BiquadCascade;
use crate::physics::constants::{speed_of_sound_mps, PARTS_PER_MILLION};
use crate::physics::geometry::{Trajectory, Vec3};
use crate::scenario::devices::{DeviceKind, LimiterConfig};
use crate::scenario::{RenderConfig, Scenario};
use crate::signals::chirps::ChirpWaveformParams;

/// A device's sample clock truth: sample n is taken at true time
/// `offset_s + n / (fs * (1 + drift_ppm * 1e-6))`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClockTruth {
    pub offset_s: f64,
    pub drift_ppm: f64,
}

impl ClockTruth {
    pub fn effective_sample_rate_hz(&self, nominal_rate_hz: f64) -> f64 {
        nominal_rate_hz * (1.0 + self.drift_ppm / PARTS_PER_MILLION)
    }

    pub fn true_time_of_sample(&self, sample_index: f64, nominal_rate_hz: f64) -> f64 {
        self.offset_s + sample_index / self.effective_sample_rate_hz(nominal_rate_hz)
    }

    pub fn sample_of_true_time(&self, true_time_s: f64, nominal_rate_hz: f64) -> f64 {
        (true_time_s - self.offset_s) * self.effective_sample_rate_hz(nominal_rate_hz)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DropoutCause {
    SampleDrop,
    ThermalStall,
}

impl DropoutCause {
    pub fn name(&self) -> &'static str {
        match self {
            Self::SampleDrop => "sample_drop",
            Self::ThermalStall => "thermal_stall",
        }
    }
}

/// Missing capture samples, in the device's original (pre-drop) indices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dropout {
    pub start_sample: usize,
    pub length_samples: usize,
    /// Reported → zero-filled in place; silent → removed from the stream.
    pub reported: bool,
    pub cause: DropoutCause,
}

/// Receiver filter active only over a device-sample window (pocket, route change).
#[derive(Clone, Debug)]
pub struct GatedReceiverFilter {
    pub kind: &'static str,
    pub start_sample: usize,
    pub end_sample: Option<usize>,
    pub start_s: f64,
    pub end_s: Option<f64>,
    pub filter: BiquadCascade,
}

/// One emitting speaker on a phone.
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceSpeaker {
    pub name: &'static str,
    pub offset_m: Vec3,
    pub relative_gain: f64,
}

#[derive(Clone, Debug)]
pub struct PlannedDevice {
    pub device_id: String,
    pub kind: DeviceKind,
    /// True path of the device reference point (tetra centroid / phone primary mic).
    pub trajectory: Trajectory,
    pub mic_offsets_m: Vec<Vec3>,
    /// Extra per-capsule colouring (phone-body shadow on the 2nd mic).
    pub per_mic_filters: Vec<BiquadCascade>,
    pub mic_response: BiquadCascade,
    pub full_scale_pa: f64,
    pub self_noise_rms_pa: f64,
    pub limiter: LimiterConfig,
    pub clock: ClockTruth,
    pub sample_count: usize,
    pub gated_filters: Vec<GatedReceiverFilter>,
    pub dropouts: Vec<Dropout>,
    pub speakers: Vec<DeviceSpeaker>,
    pub speaker_to_mic_distance_m: Option<f64>,
    /// Structural precursor amplitude relative to the airborne self path.
    pub body_conduction_relative_gain: f64,
    pub speaker_muted: bool,
    /// Position written to the manifest (true start + GPS error, if any).
    pub reported_position_m: Vec3,
    /// `[t_s, dx, dy, dz]` GPS error samples (empty without the canopy knob).
    pub gps_error_track: Vec<[f64; 4]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BirdCallRecord {
    pub event_id: String,
    pub individual_id: String,
    pub species: Vec<String>,
    pub label: String,
    pub label_category: String,
    pub generator_kind: String,
    pub level_db_spl_at_1m: f64,
    pub origin: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChirpRecord {
    pub device_id: String,
    pub round_index: usize,
    pub speaker: &'static str,
    pub muted: bool,
    pub waveform: ChirpWaveformParams,
    pub level_db_spl_at_1m: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DistractorRecord {
    pub kind: &'static str,
    pub near_device_id: Option<String>,
    pub level_db_spl_at_1m: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EmissionRecord {
    BirdCall(BirdCallRecord),
    Chirp(ChirpRecord),
    Distractor(DistractorRecord),
}

#[derive(Clone, Debug)]
pub struct PlannedEmission {
    pub emission_id: String,
    /// Source pressure waveform referenced to 1 m (Pa), at the sim rate.
    pub samples_pa_at_1m: Arc<Vec<f32>>,
    /// True session time of `samples_pa_at_1m[0]` leaving the emitter.
    pub start_time_s: f64,
    pub trajectory: Trajectory,
    pub emitting_device_index: Option<usize>,
    pub record: EmissionRecord,
}

impl PlannedEmission {
    pub fn duration_s(&self, sample_rate_hz: f64) -> f64 {
        self.samples_pa_at_1m.len() as f64 / sample_rate_hz
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlannedReflector {
    pub reflector_id: String,
    pub position_m: Vec3,
    pub reflection_coefficient: f64,
}

#[derive(Clone, Debug)]
pub struct WorldPlan {
    pub scenario_name: String,
    pub seed: u64,
    pub sample_rate_hz: f64,
    pub duration_s: f64,
    pub speed_of_sound_mps: f64,
    pub atmosphere: AtmosphericConditions,
    pub absorption_enabled: bool,
    /// (ground height, reflection coefficient) when the ground path is on.
    pub ground: Option<(f64, f64)>,
    pub reflectors: Vec<PlannedReflector>,
    pub ambient_noise_rms_pa: f64,
    pub ambient_tilt_corner_hz: f64,
    pub devices: Vec<PlannedDevice>,
    pub emissions: Vec<PlannedEmission>,
    pub render: RenderConfig,
}

/// Ground-truth loaders reject duplicate `event_id`s; catch them before a
/// full render (two sources sharing an `individual_id` generate the same
/// default ids, and explicit `event_id`s may collide with generated ones).
fn ensure_unique_event_ids(emissions: &[PlannedEmission]) -> Result<(), String> {
    let mut seen = std::collections::HashSet::new();
    for emission in emissions {
        if let EmissionRecord::BirdCall(record) = &emission.record {
            if !seen.insert(record.event_id.as_str()) {
                return Err(format!(
                    "duplicate ground-truth event_id {}",
                    record.event_id
                ));
            }
        }
    }
    Ok(())
}

/// Resolve every random draw and knob of `scenario` into a `WorldPlan`.
pub fn plan_world(scenario: &Scenario) -> Result<WorldPlan, String> {
    scenario.validate()?;
    let devices = device_planning::plan_devices(scenario)?;
    let reflectors = device_planning::plan_reflectors(scenario, &devices);
    let emissions = emission_planning::plan_emissions(scenario, &devices)?;
    ensure_unique_event_ids(&emissions)?;
    let ambient = &scenario.ambient_noise;
    Ok(WorldPlan {
        scenario_name: scenario.name.clone(),
        seed: scenario.seed,
        sample_rate_hz: scenario.sample_rate(),
        duration_s: scenario.duration_s,
        speed_of_sound_mps: speed_of_sound_mps(
            scenario.environment.temperature_c,
            scenario.environment.humidity_fraction,
        ),
        atmosphere: scenario.environment.atmospheric_conditions(),
        absorption_enabled: scenario.environment.absorption_enabled,
        ground: scenario.ground.enabled.then_some((
            scenario.ground.height_m,
            scenario.ground.reflection_coefficient,
        )),
        reflectors,
        ambient_noise_rms_pa: if ambient.enabled {
            crate::physics::constants::pascals_from_db_spl(ambient.level_db_spl)
        } else {
            0.0
        },
        ambient_tilt_corner_hz: ambient.spectral_tilt_corner_hz,
        devices,
        emissions,
        render: scenario.render.clone(),
    })
}
