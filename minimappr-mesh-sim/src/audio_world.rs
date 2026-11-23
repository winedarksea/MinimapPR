//! Rendered-audio mode (closeout D2): the scenario's calls rendered by the
//! Layer 1 simulator (`minimappr-sim`) onto every device's own drifting
//! sample clock, so clips carry real FLAC audio and the host runs the real
//! `OnsetClipSetLocalizer` instead of the arrival-time stand-in.
//!
//! Only bird-call clips become audio. Chirp sync, Stage-1 triggering,
//! readiness and the classifier stand-in stay on the arrival-time world
//! (the classifier is not what this mode tests; clustering and
//! localization are). Dependency direction is unchanged: mesh-sim already
//! depends on minimappr-sim for its physics, and neither depends on it.

use std::sync::Arc;

use minimappr_sim::render::render_world;
use minimappr_sim::scenario::devices::{ClockConfig, DeviceConfig, DeviceKind};
use minimappr_sim::scenario::sources::{CallConfig, SourceConfig};
use minimappr_sim::scenario::{EnvironmentConfig, Scenario};
use minimappr_sim::signals::calls::{CallGenerator, FmSweepParams, HarshParams, TrillParams};
use minimappr_sim::world::plan_world;

use crate::acoustic_world::AcousticWorld;

const PARTS_PER_MILLION: f64 = 1e6;
const PCM16_FULL_SCALE: f32 = 32_767.0;
const DECIBELS_PER_POWER_DECADE: f64 = 10.0;
/// Frequency samples for the in-band share of the ambient spectrum.
const AMBIENT_SPECTRUM_INTEGRATION_STEPS: usize = 8_192;
/// A trill's elements sweep this fraction of the carrier.
const TRILL_ELEMENT_SWEEP_FRACTION_OF_CARRIER: f64 = 0.4;

/// Which acoustic world a run's clips come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AcousticMode {
    /// Clips carry arrival times; `ArrivalClipLocalizer` stands in for clip SRP.
    #[default]
    ArrivalTimes,
    /// Clips carry Layer 1 audio; the real onset-clustering localizer runs.
    RenderedAudio,
}

/// Layer 1 waveform for every scripted call (all centred on the call's
/// frequency; the stand-in only knows a centre frequency and a band).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AudioCallWaveform {
    /// Linear FM down-sweep over `sweep_band_fraction` of the call band.
    #[default]
    FmSweep,
    /// Repeated down-swept elements on the call frequency (Layer 1 trill).
    Trill,
    /// Harmonic stack plus noise band (jay/crow-like rasp, Layer 1 default spectrum).
    Harsh,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioModeConfig {
    pub call_waveform: AudioCallWaveform,
    pub call_duration_s: f64,
    /// Fraction of the stand-in band the sweep covers, centred on the call
    /// frequency in log terms (1 = the whole 2-octave band).
    pub sweep_band_fraction: f64,
    /// Layer 1 broadband ambient (forest tilt), dB SPL. None: chosen so its
    /// share in the call band equals the arrival world's in-band ambient,
    /// which the Stage-1 / classifier / readiness SNRs are computed against.
    pub ambient_noise_db_spl: Option<f64>,
    pub ground_reflection_enabled: bool,
    /// Scoring only: a cluster's led onset is matched to a scripted call
    /// sounding at that node within this slack (10 ms onset hops plus a
    /// 10 ms frame pre-roll and rise time).
    pub onset_match_tolerance_s: f64,
}

impl Default for AudioModeConfig {
    fn default() -> Self {
        Self {
            call_waveform: AudioCallWaveform::default(),
            call_duration_s: 0.3,
            sweep_band_fraction: 0.5,
            ambient_noise_db_spl: None,
            ground_reflection_enabled: true,
            onset_match_tolerance_s: 0.05,
        }
    }
}

/// Every device's rendered stream (mic 0) as PCM16 on its own sample clock.
pub struct RenderedSessionAudio {
    pub device_streams: Vec<Vec<i16>>,
}

impl std::fmt::Debug for RenderedSessionAudio {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lengths: Vec<usize> = self.device_streams.iter().map(Vec::len).collect();
        formatter.debug_struct("RenderedSessionAudio").field("stream_lengths", &lengths).finish()
    }
}

fn call_generator(centre_frequency_hz: f64, config: &AudioModeConfig) -> CallGenerator {
    use crate::runner_actions::{CALL_BAND_HIGH_FRACTION, CALL_BAND_LOW_FRACTION};
    let half_log_span = 0.5 * config.sweep_band_fraction * (CALL_BAND_HIGH_FRACTION / CALL_BAND_LOW_FRACTION).ln();
    let geometric_centre = centre_frequency_hz * (CALL_BAND_LOW_FRACTION * CALL_BAND_HIGH_FRACTION).sqrt();
    match config.call_waveform {
        AudioCallWaveform::FmSweep => CallGenerator::FmSweep(FmSweepParams {
            start_frequency_hz: geometric_centre * half_log_span.exp(),
            end_frequency_hz: geometric_centre * (-half_log_span).exp(),
            duration_s: config.call_duration_s,
            ..FmSweepParams::default()
        }),
        AudioCallWaveform::Trill => CallGenerator::Trill(TrillParams {
            carrier_frequency_hz: geometric_centre,
            element_sweep_hz: TRILL_ELEMENT_SWEEP_FRACTION_OF_CARRIER * geometric_centre,
            duration_s: config.call_duration_s,
            ..TrillParams::default()
        }),
        AudioCallWaveform::Harsh => CallGenerator::Harsh(HarshParams { duration_s: config.call_duration_s, ..HarshParams::default() }),
    }
}

/// Share of Layer 1's tilted ambient power inside `[low_hz, high_hz]`: the
/// one-pole spectrum |1 / (1 − p e^{−jω})|² integrated numerically.
fn tilted_ambient_band_share(tilt_corner_hz: f64, sample_rate_hz: f64, low_hz: f64, high_hz: f64) -> f64 {
    let pole = (-std::f64::consts::TAU * tilt_corner_hz / sample_rate_hz).exp();
    let nyquist = 0.5 * sample_rate_hz;
    let (mut in_band, mut total) = (0.0, 0.0);
    for step in 0..AMBIENT_SPECTRUM_INTEGRATION_STEPS {
        let frequency = (step as f64 + 0.5) / AMBIENT_SPECTRUM_INTEGRATION_STEPS as f64 * nyquist;
        let omega = std::f64::consts::TAU * frequency / sample_rate_hz;
        let density = 1.0 / (1.0 - 2.0 * pole * omega.cos() + pole * pole);
        total += density;
        if (low_hz..=high_hz).contains(&frequency) {
            in_band += density;
        }
    }
    in_band / total.max(f64::MIN_POSITIVE)
}

/// Broadband level whose share in the median call's band is the arrival
/// world's in-band ambient level.
fn matched_ambient_db_spl(world: &AcousticWorld, scenario: &Scenario) -> f64 {
    use crate::runner_actions::{CALL_BAND_HIGH_FRACTION, CALL_BAND_LOW_FRACTION};
    let mut centres: Vec<f64> = world.calls.iter().map(|call| call.centre_frequency_hz).collect();
    centres.sort_by(f64::total_cmp);
    let centre = centres.get(centres.len() / 2).copied().unwrap_or(crate::scenario::DEFAULT_CALL_FREQUENCY_HZ);
    let share = tilted_ambient_band_share(
        scenario.ambient_noise.spectral_tilt_corner_hz,
        f64::from(scenario.sample_rate_hz),
        centre * CALL_BAND_LOW_FRACTION,
        centre * CALL_BAND_HIGH_FRACTION,
    );
    world.config.ambient_bird_band_noise_db_spl - DECIBELS_PER_POWER_DECADE * share.log10()
}

/// The Layer 1 scenario equivalent to `world`: same devices, trajectories,
/// clocks and calls. Capture stalls and placement attenuation have no
/// Layer 1 equivalent here, so a world using them is refused rather than
/// rendered differently from what the arrival-time side assumes.
pub fn layer_one_scenario(world: &AcousticWorld, duration_s: f64, config: &AudioModeConfig) -> Result<Scenario, String> {
    let mut devices = Vec::new();
    for (index, truth) in world.devices.iter().enumerate() {
        if !truth.stalls.is_empty() || truth.spec.placement_attenuation_db != 0.0 {
            return Err(format!("device {index}: capture stalls / placement attenuation are not rendered in audio mode"));
        }
        let waypoints = &truth.spec.trajectory.waypoints;
        devices.push(DeviceConfig {
            device_id: format!("phone-{index}"),
            kind: DeviceKind::Phone,
            position_m: waypoints.first().map_or([0.0; 3], |(_, position)| *position),
            trajectory: (waypoints.len() > 1).then(|| waypoints.iter().map(|(time, p)| [*time, p[0], p[1], p[2]]).collect()),
            clock: Some(ClockConfig::Fixed { offset_s: truth.clock_offset_s, drift_ppm: truth.clock_drift * PARTS_PER_MILLION }),
            mic_response: None,
            second_mic: None,
            speaker: Default::default(),
            limiter: None,
        });
    }
    let sources = world
        .calls
        .iter()
        .enumerate()
        .map(|(index, call)| SourceConfig {
            individual_id: format!("call-{index}"),
            species: call.species_label.to_string(),
            label: None,
            label_category: "bird".into(),
            position_m: call.position_m,
            trajectory: None,
            calls: vec![CallConfig {
                start_s: call.time_s,
                level_db_spl_at_1m: call.level_db_spl_at_1m,
                generator: call_generator(call.centre_frequency_hz, config),
                event_id: Some(format!("call-{index}")),
            }],
        })
        .collect();
    let atmosphere = world.config.atmosphere;
    let mut scenario = Scenario {
        name: "mesh_sim_audio".into(),
        seed: world.seed,
        duration_s,
        devices,
        sources,
        environment: EnvironmentConfig {
            temperature_c: atmosphere.temperature_c,
            humidity_fraction: atmosphere.relative_humidity_fraction,
            pressure_kpa: atmosphere.pressure_kpa,
            absorption_enabled: true,
        },
        ..Scenario::default()
    };
    scenario.ground.enabled = config.ground_reflection_enabled;
    scenario.ambient_noise.level_db_spl = config.ambient_noise_db_spl.unwrap_or_else(|| matched_ambient_db_spl(world, &scenario));
    Ok(scenario)
}

/// Render the session once; clips are cut from these streams.
pub fn render_session_audio(world: &AcousticWorld, duration_s: f64, config: &AudioModeConfig) -> Result<Arc<RenderedSessionAudio>, String> {
    let plan = plan_world(&layer_one_scenario(world, duration_s, config)?)?;
    let device_streams = render_world(&plan)
        .into_iter()
        .map(|device| {
            device.channels.first().map_or_else(Vec::new, |channel| {
                channel.iter().map(|sample| (sample * PCM16_FULL_SCALE).round().clamp(-PCM16_FULL_SCALE - 1.0, PCM16_FULL_SCALE) as i16).collect()
            })
        })
        .collect();
    Ok(Arc::new(RenderedSessionAudio { device_streams }))
}
