//! Seeded point-only scenes for the localizer bake-off (E8): 3–8 single-mic
//! nodes over a 20 m – 1 km site, one or two simultaneous birds or one
//! continuous source, realistic clocks and positions, rendered by the
//! Layer 1 simulator (spherical spreading, ISO 9613-1 absorption, ground
//! reflection, per-device ambient and self noise).
//!
//! Every node's stream is on its own clock: sample n was taken at true time
//! `offset + n / (fs (1 + drift))` but the node stamps it `n / fs`, so a
//! window cut on the shared stamped timeline carries the real clock error.
//! Solvers see measured positions and declared uncertainties only; truth is
//! kept for scoring.

use rand::Rng;
use rand_chacha::ChaCha8Rng;

use minimappr_sim::physics::constants::speed_of_sound_mps;
use minimappr_sim::render::render_world;
use minimappr_sim::scenario::devices::{ClockConfig, DeviceConfig, DeviceKind, MicResponseConfig};
use minimappr_sim::scenario::sources::{CallConfig, SourceConfig};
use minimappr_sim::scenario::{EnvironmentConfig, Scenario};
use minimappr_sim::signals::calls::{CallGenerator, FmSweepParams, HarshParams, TonalWhistleParams, TrillParams};
use minimappr_sim::world::plan_world;

use crate::bakeoff_config::{BirdWaveform, NodeClockClass, SceneDrawConfig};
use crate::seeded_random::purpose_stream;

const PARTS_PER_MILLION: f64 = 1e6;
const SCENE_TEMPERATURE_C: f64 = 20.0;
const SCENE_HUMIDITY_FRACTION: f64 = 0.5;
/// Draw attempts before a node or caller placement gives up on its spacing rule.
const PLACEMENT_ATTEMPTS: usize = 200;
/// Point-node microphone: Layer 1 phone capsule without the 20 kHz roll-off
/// (above Nyquist at the bake-off rate).
const POINT_MIC_HIGHPASS_HZ: f64 = 80.0;
const POINT_MIC_HIGHPASS_ORDER: usize = 2;
const POINT_MIC_SELF_NOISE_DB_SPL: f64 = 32.0;
const POINT_MIC_FULL_SCALE_DB_SPL: f64 = 120.0;
/// Bird waveform centre frequencies (Hz) and shapes.
const BIRD_CENTRE_FREQUENCY_HZ: [f64; 2] = [2_000.0, 5_500.0];
const FM_SWEEP_HALF_SPAN_FRACTION: f64 = 0.35;
const TRILL_ELEMENT_SWEEP_FRACTION: f64 = 0.4;
const HARSH_FUNDAMENTAL_HZ: [f64; 2] = [800.0, 1_500.0];
const CONTINUOUS_SOURCE_HARMONIC_COUNT: usize = 12;
const CONTINUOUS_SOURCE_NOISE_MIX: f64 = 0.3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BakeoffEventKind {
    SingleBird,
    TwoBirds,
    ContinuousSource,
}

impl BakeoffEventKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::SingleBird => "single_bird",
            Self::TwoBirds => "two_birds",
            Self::ContinuousSource => "continuous_source",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ClockRegime {
    AllGps,
    AllNtp,
    Mixed,
}

impl ClockRegime {
    pub fn label(self) -> &'static str {
        match self {
            Self::AllGps => "all_gps",
            Self::AllNtp => "all_ntp",
            Self::Mixed => "mixed_gps_ntp_free",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BakeoffNode {
    pub true_position_m: [f64; 3],
    /// What the solvers are given (hand-measured nodes carry a real error).
    pub measured_position_m: [f64; 3],
    pub stated_position_std_m: f64,
    pub hand_measured: bool,
    pub clock_class: NodeClockClass,
    pub true_clock_offset_s: f64,
    pub clock_drift_fraction: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BakeoffCall {
    pub position_m: [f64; 3],
    pub start_s: f64,
    pub duration_s: f64,
    /// `BirdWaveform::label`, or the continuous source's.
    pub waveform_label: &'static str,
}

pub const CONTINUOUS_SOURCE_WAVEFORM_LABEL: &str = "continuous_harmonic";

pub struct BakeoffScene {
    pub seed: u64,
    pub event_kind: BakeoffEventKind,
    pub clock_regime: ClockRegime,
    pub nodes: Vec<BakeoffNode>,
    pub calls: Vec<BakeoffCall>,
    pub sample_rate_hz: u32,
    pub sound_speed_mps: f64,
    /// Largest distance between measured node positions.
    pub aperture_m: f64,
    pub lead_in_noise_s: f64,
    /// Node i's mic samples on its own clock, full-scale f32.
    pub streams: Vec<Vec<f32>>,
}

impl std::fmt::Debug for BakeoffScene {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BakeoffScene")
            .field("seed", &self.seed)
            .field("event_kind", &self.event_kind)
            .field("clock_regime", &self.clock_regime)
            .field("node_count", &self.nodes.len())
            .field("aperture_m", &self.aperture_m)
            .finish()
    }
}

pub fn distance_m(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

pub fn horizontal_distance_m(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

pub fn largest_pairwise_distance_m(points: &[[f64; 3]]) -> f64 {
    points.iter().flat_map(|a| points.iter().map(move |b| distance_m(a, b))).fold(0.0, f64::max)
}

impl BakeoffScene {
    /// Node `node`'s stamped stream time for true time `true_time_s`.
    pub fn stream_time_of_true_time(&self, node: usize, true_time_s: f64) -> f64 {
        let truth = &self.nodes[node];
        (true_time_s - truth.true_clock_offset_s) * (1.0 + truth.clock_drift_fraction)
    }

    /// Direct-path arrival of `call` at `node`, on the node's stamped timeline.
    pub fn arrival_stream_time_s(&self, call: usize, node: usize) -> f64 {
        let call_truth = &self.calls[call];
        let true_arrival_s = call_truth.start_s + distance_m(&call_truth.position_m, &self.nodes[node].true_position_m) / self.sound_speed_mps;
        self.stream_time_of_true_time(node, true_arrival_s)
    }

    pub fn measured_positions_m(&self) -> Vec<[f64; 3]> {
        self.nodes.iter().map(|node| node.measured_position_m).collect()
    }

    pub fn largest_layout_lag_s(&self) -> f64 {
        self.aperture_m / self.sound_speed_mps
    }

    pub fn has_hand_measured_node(&self) -> bool {
        self.nodes.iter().any(|node| node.hand_measured)
    }
}

fn uniform(rng: &mut ChaCha8Rng, range: [f64; 2]) -> f64 {
    if range[1] <= range[0] {
        return range[0];
    }
    rng.random_range(range[0]..range[1])
}

fn standard_normal(rng: &mut ChaCha8Rng) -> f64 {
    let first: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let second: f64 = rng.random();
    (-2.0 * first.ln()).sqrt() * (std::f64::consts::TAU * second).cos()
}

fn point_in_disc(rng: &mut ChaCha8Rng, radius_m: f64) -> [f64; 2] {
    let radius = radius_m * rng.random::<f64>().sqrt();
    let angle = std::f64::consts::TAU * rng.random::<f64>();
    [radius * angle.cos(), radius * angle.sin()]
}

fn draw_clock_regime(rng: &mut ChaCha8Rng, config: &SceneDrawConfig) -> ClockRegime {
    let draw: f64 = rng.random();
    if draw < config.all_gps_probability {
        ClockRegime::AllGps
    } else if draw < config.all_gps_probability + config.all_ntp_probability {
        ClockRegime::AllNtp
    } else {
        ClockRegime::Mixed
    }
}

fn draw_clock_classes(rng: &mut ChaCha8Rng, regime: ClockRegime, node_count: usize, config: &SceneDrawConfig) -> Vec<NodeClockClass> {
    match regime {
        ClockRegime::AllGps => vec![NodeClockClass::GpsLocked; node_count],
        ClockRegime::AllNtp => vec![NodeClockClass::NtpDisciplined; node_count],
        ClockRegime::Mixed => {
            let mut classes: Vec<NodeClockClass> =
                (0..node_count).map(|_| if rng.random_bool(0.5) { NodeClockClass::GpsLocked } else { NodeClockClass::NtpDisciplined }).collect();
            if rng.random::<f64>() < config.mixed_site_free_running_node_probability {
                let index = rng.random_range(0..node_count);
                classes[index] = NodeClockClass::FreeRunning;
            }
            classes
        }
    }
}

fn draw_nodes(rng: &mut ChaCha8Rng, regime: ClockRegime, spread_m: f64, config: &SceneDrawConfig) -> Vec<BakeoffNode> {
    let node_count = rng.random_range(config.node_count_range[0]..=config.node_count_range[1]);
    let minimum_separation_m = config.minimum_node_separation_spread_fraction * spread_m;
    let mut horizontal: Vec<[f64; 2]> = Vec::new();
    for _ in 0..node_count {
        let mut candidate = point_in_disc(rng, 0.5 * spread_m);
        for _ in 0..PLACEMENT_ATTEMPTS {
            if horizontal.iter().all(|other| (other[0] - candidate[0]).hypot(other[1] - candidate[1]) >= minimum_separation_m) {
                break;
            }
            candidate = point_in_disc(rng, 0.5 * spread_m);
        }
        horizontal.push(candidate);
    }
    let classes = draw_clock_classes(rng, regime, node_count, config);
    let site_has_hand_measured_nodes = rng.random::<f64>() < config.hand_measured_site_probability;
    horizontal
        .into_iter()
        .zip(classes)
        .map(|([x, y], clock_class)| {
            let true_position_m = [x, y, uniform(rng, config.node_height_m)];
            let hand_measured = site_has_hand_measured_nodes && rng.random::<f64>() < config.hand_measured_node_probability;
            let stated_position_std_m = if hand_measured { uniform(rng, config.hand_measured_stated_std_m) } else { config.surveyed_position_std_m };
            // Stated std is per horizontal axis; height is measured twice as well.
            let measured_position_m = [
                true_position_m[0] + stated_position_std_m * standard_normal(rng),
                true_position_m[1] + stated_position_std_m * standard_normal(rng),
                true_position_m[2] + 0.5 * stated_position_std_m * standard_normal(rng),
            ];
            let drift_ppm_max = match clock_class {
                NodeClockClass::GpsLocked => 0.0,
                NodeClockClass::NtpDisciplined => config.ntp_drift_ppm_max,
                NodeClockClass::FreeRunning => config.free_running_drift_ppm_max,
            };
            BakeoffNode {
                true_position_m,
                measured_position_m,
                stated_position_std_m,
                hand_measured,
                clock_class,
                // Honest declared uncertainty: offsets drawn at the class's declared sigma.
                true_clock_offset_s: clock_class.declared_offset_std_s() * standard_normal(rng),
                clock_drift_fraction: uniform(rng, [-drift_ppm_max, drift_ppm_max]) / PARTS_PER_MILLION,
            }
        })
        .collect()
}

fn bird_generator(rng: &mut ChaCha8Rng, waveform: BirdWaveform, duration_s: f64) -> CallGenerator {
    let centre_hz = uniform(rng, BIRD_CENTRE_FREQUENCY_HZ);
    match waveform {
        BirdWaveform::FmSweep => CallGenerator::FmSweep(FmSweepParams {
            start_frequency_hz: centre_hz * (1.0 + FM_SWEEP_HALF_SPAN_FRACTION),
            end_frequency_hz: centre_hz * (1.0 - FM_SWEEP_HALF_SPAN_FRACTION),
            duration_s,
            ..FmSweepParams::default()
        }),
        BirdWaveform::TonalWhistle => CallGenerator::TonalWhistle(TonalWhistleParams {
            start_frequency_hz: centre_hz,
            end_frequency_hz: centre_hz,
            note_duration_s: duration_s,
            note_count: 1,
            ..TonalWhistleParams::default()
        }),
        BirdWaveform::Trill => CallGenerator::Trill(TrillParams {
            carrier_frequency_hz: centre_hz,
            element_sweep_hz: TRILL_ELEMENT_SWEEP_FRACTION * centre_hz,
            duration_s,
            ..TrillParams::default()
        }),
        BirdWaveform::Harsh => CallGenerator::Harsh(HarshParams { fundamental_hz: uniform(rng, HARSH_FUNDAMENTAL_HZ), duration_s, ..HarshParams::default() }),
    }
}

fn continuous_source_generator(rng: &mut ChaCha8Rng, config: &SceneDrawConfig) -> CallGenerator {
    let fundamental_hz = uniform(rng, config.continuous_source_fundamental_hz);
    CallGenerator::Harsh(HarshParams {
        fundamental_hz,
        harmonic_count: CONTINUOUS_SOURCE_HARMONIC_COUNT,
        noise_mix: CONTINUOUS_SOURCE_NOISE_MIX,
        // Blade-pass amplitude modulation at the fundamental.
        roughness_modulation_hz: fundamental_hz,
        duration_s: config.continuous_source_duration_s,
        edge_ramp_s: config.continuous_source_edge_ramp_s,
        ..HarshParams::default()
    })
}

struct DrawnSource {
    call: BakeoffCall,
    level_db_spl_at_1m: f64,
    generator: CallGenerator,
}

fn draw_bird_position(rng: &mut ChaCha8Rng, spread_m: f64, config: &SceneDrawConfig, avoid: Option<&[f64; 3]>) -> [f64; 3] {
    let radius_m = config.source_radius_spread_fraction * 0.5 * spread_m;
    let mut position = point_in_disc(rng, radius_m);
    if let Some(other) = avoid {
        let minimum_m = config.second_caller_minimum_separation_spread_fraction * spread_m;
        for _ in 0..PLACEMENT_ATTEMPTS {
            if (position[0] - other[0]).hypot(position[1] - other[1]) >= minimum_m {
                break;
            }
            position = point_in_disc(rng, radius_m);
        }
    }
    [position[0], position[1], uniform(rng, config.bird_height_m)]
}

fn draw_sources(rng: &mut ChaCha8Rng, kind: BakeoffEventKind, spread_m: f64, event_start_s: f64, config: &SceneDrawConfig) -> Vec<DrawnSource> {
    let bird = |rng: &mut ChaCha8Rng, start_s: f64, avoid: Option<&[f64; 3]>| {
        let duration_s = uniform(rng, config.bird_call_duration_s);
        let waveform = config.bird_waveforms[rng.random_range(0..config.bird_waveforms.len())];
        DrawnSource {
            call: BakeoffCall { position_m: draw_bird_position(rng, spread_m, config, avoid), start_s, duration_s, waveform_label: waveform.label() },
            level_db_spl_at_1m: uniform(rng, config.bird_level_db_spl_at_1m),
            generator: bird_generator(rng, waveform, duration_s),
        }
    };
    match kind {
        BakeoffEventKind::SingleBird => vec![bird(rng, event_start_s, None)],
        BakeoffEventKind::TwoBirds => {
            let first = bird(rng, event_start_s, None);
            let second_start_s = event_start_s + uniform(rng, config.second_caller_delay_s);
            let second = bird(rng, second_start_s, Some(&first.call.position_m));
            vec![first, second]
        }
        BakeoffEventKind::ContinuousSource => {
            let horizontal = point_in_disc(rng, config.source_radius_spread_fraction * 0.5 * spread_m);
            vec![DrawnSource {
                call: BakeoffCall {
                    position_m: [horizontal[0], horizontal[1], uniform(rng, config.continuous_source_height_m)],
                    start_s: event_start_s,
                    duration_s: config.continuous_source_duration_s,
                    waveform_label: CONTINUOUS_SOURCE_WAVEFORM_LABEL,
                },
                level_db_spl_at_1m: uniform(rng, config.continuous_source_level_db_spl_at_1m),
                generator: continuous_source_generator(rng, config),
            }]
        }
    }
}

fn draw_event_kind(rng: &mut ChaCha8Rng, config: &SceneDrawConfig) -> BakeoffEventKind {
    let draw: f64 = rng.random();
    if draw < config.single_bird_probability {
        BakeoffEventKind::SingleBird
    } else if draw < config.single_bird_probability + config.two_bird_probability {
        BakeoffEventKind::TwoBirds
    } else {
        BakeoffEventKind::ContinuousSource
    }
}

fn point_mic_response() -> MicResponseConfig {
    MicResponseConfig {
        highpass_hz: Some(POINT_MIC_HIGHPASS_HZ),
        highpass_order: POINT_MIC_HIGHPASS_ORDER,
        lowpass_hz: None,
        lowpass_order: POINT_MIC_HIGHPASS_ORDER,
        self_noise_db_spl: POINT_MIC_SELF_NOISE_DB_SPL,
        full_scale_db_spl: POINT_MIC_FULL_SCALE_DB_SPL,
    }
}

/// Draw and render the scene for `seed`.
pub fn build_bakeoff_scene(seed: u64, config: &SceneDrawConfig) -> Result<BakeoffScene, String> {
    let mut rng = purpose_stream(seed, "point_bakeoff_scene");
    let spread_m = (uniform(&mut rng, [config.site_spread_range_m[0].ln(), config.site_spread_range_m[1].ln()])).exp();
    let clock_regime = draw_clock_regime(&mut rng, config);
    let nodes = draw_nodes(&mut rng, clock_regime, spread_m, config);
    let event_kind = draw_event_kind(&mut rng, config);
    let sound_speed_mps = speed_of_sound_mps(SCENE_TEMPERATURE_C, SCENE_HUMIDITY_FRACTION);
    let true_positions: Vec<[f64; 3]> = nodes.iter().map(|node| node.true_position_m).collect();
    // The event starts one full layout lag after the noise lead-in, so a
    // window reaching back by the layout lag still holds only noise.
    let event_start_s = config.lead_in_noise_s + largest_pairwise_distance_m(&true_positions) / sound_speed_mps;
    let sources = draw_sources(&mut rng, event_kind, spread_m, event_start_s, config);
    let last_arrival_s = sources
        .iter()
        .flat_map(|source| {
            true_positions.iter().map(move |node| source.call.start_s + source.call.duration_s + distance_m(&source.call.position_m, node) / sound_speed_mps)
        })
        .fold(0.0, f64::max);
    let ambient_noise_db_spl = uniform(&mut rng, config.ambient_noise_db_spl);
    let duration_s = last_arrival_s + config.tail_after_last_arrival_s + largest_pairwise_distance_m(&true_positions) / sound_speed_mps;
    let scenario = layer_one_scenario(seed, &nodes, &sources, duration_s, ambient_noise_db_spl, config.sample_rate_hz);
    let streams = render_world(&plan_world(&scenario)?).into_iter().map(|device| device.channels.into_iter().next().unwrap_or_default()).collect();
    let measured: Vec<[f64; 3]> = nodes.iter().map(|node| node.measured_position_m).collect();
    Ok(BakeoffScene {
        seed,
        event_kind,
        clock_regime,
        aperture_m: largest_pairwise_distance_m(&measured),
        nodes,
        calls: sources.into_iter().map(|source| source.call).collect(),
        sample_rate_hz: config.sample_rate_hz,
        sound_speed_mps,
        lead_in_noise_s: config.lead_in_noise_s,
        streams,
    })
}

fn layer_one_scenario(seed: u64, nodes: &[BakeoffNode], sources: &[DrawnSource], duration_s: f64, ambient_noise_db_spl: f64, sample_rate_hz: u32) -> Scenario {
    let devices = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| DeviceConfig {
            device_id: format!("point-{index}"),
            kind: DeviceKind::Phone,
            position_m: node.true_position_m,
            trajectory: None,
            clock: Some(ClockConfig::Fixed { offset_s: node.true_clock_offset_s, drift_ppm: node.clock_drift_fraction * PARTS_PER_MILLION }),
            mic_response: Some(point_mic_response()),
            second_mic: None,
            speaker: Default::default(),
            limiter: None,
        })
        .collect();
    let sources = sources
        .iter()
        .enumerate()
        .map(|(index, source)| SourceConfig {
            individual_id: format!("source-{index}"),
            species: format!("species-{index}"),
            label: None,
            label_category: "bird".into(),
            position_m: source.call.position_m,
            trajectory: None,
            calls: vec![CallConfig {
                start_s: source.call.start_s,
                level_db_spl_at_1m: source.level_db_spl_at_1m,
                generator: source.generator.clone(),
                event_id: Some(format!("call-{index}")),
            }],
        })
        .collect();
    let mut scenario = Scenario {
        name: "point_network_bakeoff".into(),
        seed,
        duration_s,
        sample_rate_hz,
        devices,
        sources,
        environment: EnvironmentConfig { temperature_c: SCENE_TEMPERATURE_C, humidity_fraction: SCENE_HUMIDITY_FRACTION, ..EnvironmentConfig::default() },
        ..Scenario::default()
    };
    scenario.ambient_noise.level_db_spl = ambient_noise_db_spl;
    scenario
}
