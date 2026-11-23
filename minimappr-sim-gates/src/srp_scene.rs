//! Layer 1 scenes for SRP trials: random phone layouts rendered by
//! `minimappr-sim` (ambient noise, ground reflection, ISO absorption,
//! random trunks), one call per scene. Mirrors the helpers in
//! `minimappr-core/tests/srp_layout_support` (test modules cannot be
//! imported across crates).

use minimappr_core::srp_layout::{IslandAnalysisWindow, LayoutNode, LayoutSrpConfig, NodeClockModel};
use minimappr_sim::render::render_world;
use minimappr_sim::scenario::devices::{ClockConfig, DeviceConfig, DeviceKind};
use minimappr_sim::scenario::sources::{CallConfig, SourceConfig};
use minimappr_sim::scenario::Scenario;
use minimappr_sim::signals::calls::{CallGenerator, FmSweepParams, HarshParams, TrillParams};
use minimappr_sim::world::{plan_world, WorldPlan};

use crate::trial_geometry::{distance, uniform_in};

const PARTS_PER_MILLION: f64 = 1.0e6;

/// Scene constants (the core SRP tests' values).
#[derive(Debug, Clone, PartialEq)]
pub struct SrpSceneConfig {
    /// Loud-ish songbird; keeps 150 m sources above the 35 dB ambient floor.
    pub call_level_db_spl_at_1m: f64,
    pub call_start_s: f64,
    /// Scene length after the call starts (covers the longest call + flight).
    pub scene_tail_s: f64,
    pub window_padding_s: f64,
    pub device_clock_offset_range_s: [f64; 2],
    pub device_clock_drift_range_ppm: [f64; 2],
    pub random_trunks: bool,
}

impl Default for SrpSceneConfig {
    fn default() -> Self {
        Self {
            call_level_db_spl_at_1m: 92.0,
            call_start_s: 0.5,
            scene_tail_s: 1.6,
            window_padding_s: 0.05,
            device_clock_offset_range_s: [-0.006, 0.006],
            device_clock_drift_range_ppm: [-20.0, 20.0],
            random_trunks: true,
        }
    }
}

fn phone(device_id: String, position_m: [f64; 3], clock: ClockConfig) -> DeviceConfig {
    DeviceConfig {
        device_id,
        kind: DeviceKind::Phone,
        position_m,
        trajectory: None,
        clock: Some(clock),
        mic_response: None,
        second_mic: None,
        speaker: Default::default(),
        limiter: None,
    }
}

/// Broadband-ish call types (tonal whistles need the unbuilt envelope path).
fn call_generator(variant: u64) -> CallGenerator {
    match variant % 3 {
        0 => CallGenerator::Trill(TrillParams { duration_s: 0.5, ..TrillParams::default() }),
        1 => CallGenerator::FmSweep(FmSweepParams { duration_s: 0.3, ..FmSweepParams::default() }),
        _ => CallGenerator::Harsh(HarshParams::default()),
    }
}

/// Scene with phones at `node_positions_m` and one call at `source_m`.
pub fn build_scene(seed: u64, node_positions_m: &[[f64; 3]], source_m: [f64; 3], config: &SrpSceneConfig) -> Scenario {
    let devices = node_positions_m
        .iter()
        .enumerate()
        .map(|(index, position)| {
            let key = index as u64;
            let clock = ClockConfig::Fixed {
                offset_s: uniform_in(seed, "scene_clock_offset", key, config.device_clock_offset_range_s),
                drift_ppm: uniform_in(seed, "scene_clock_drift", key, config.device_clock_drift_range_ppm),
            };
            phone(format!("phone-{index}"), *position, clock)
        })
        .collect();
    let mut scenario = Scenario {
        name: "sim_gates_srp".into(),
        seed,
        duration_s: config.call_start_s + config.scene_tail_s,
        devices,
        sources: vec![SourceConfig {
            individual_id: "bird".into(),
            species: "test".into(),
            label: None,
            label_category: "bird".into(),
            position_m: source_m,
            trajectory: None,
            calls: vec![CallConfig {
                start_s: config.call_start_s,
                level_db_spl_at_1m: config.call_level_db_spl_at_1m,
                generator: call_generator(seed),
                event_id: None,
            }],
        }],
        ..Scenario::default()
    };
    scenario.ground.height_m = 0.0;
    if config.random_trunks {
        scenario.reflectors.random_trunks = Some(Default::default());
    }
    scenario
}

pub struct RenderedScene {
    pub plan: WorldPlan,
    pub channels_per_device: Vec<Vec<Vec<f32>>>,
}

pub fn render(scenario: &Scenario) -> Result<RenderedScene, String> {
    let plan = plan_world(scenario).map_err(|error| format!("plan: {error:?}"))?;
    let rendered = render_world(&plan);
    Ok(RenderedScene { channels_per_device: rendered.into_iter().map(|device| device.channels).collect(), plan })
}

/// Nodes with each device's true position and clock (callers perturb).
pub fn true_layout_nodes(plan: &WorldPlan) -> Vec<LayoutNode> {
    plan.devices
        .iter()
        .map(|device| LayoutNode {
            position_m: device.trajectory.position_at(0.0),
            mic_offsets_m: device.mic_offsets_m.clone(),
            clock: NodeClockModel {
                anchor_sample_index: 0.0,
                anchor_island_time_s: device.clock.offset_s,
                rate_ratio: 1.0 + device.clock.drift_ppm / PARTS_PER_MILLION,
            },
            pair_weight: 1.0,
            position_std_m: 0.0,
            clock_offset_std_s: 0.0,
        })
        .collect()
}

/// Island window covering the call's arrival at every node.
pub fn call_window(plan: &WorldPlan, padding_s: f64) -> IslandAnalysisWindow {
    let emission = &plan.emissions[0];
    let source = emission.trajectory.position_at(emission.start_time_s);
    let distances: Vec<f64> = plan.devices.iter().map(|device| distance(device.trajectory.position_at(0.0), source)).collect();
    let nearest = distances.iter().copied().fold(f64::INFINITY, f64::min);
    let farthest = distances.iter().copied().fold(0.0, f64::max);
    let start = emission.start_time_s + nearest / plan.speed_of_sound_mps - padding_s;
    let end =
        emission.start_time_s + emission.duration_s(plan.sample_rate_hz) + farthest / plan.speed_of_sound_mps + padding_s;
    IslandAnalysisWindow { start_island_time_s: start, sample_count: ((end - start) * plan.sample_rate_hz) as usize }
}

/// Shipping SRP defaults with the scene's sample rate, sound speed, ground.
pub fn srp_config_for(plan: &WorldPlan) -> LayoutSrpConfig {
    LayoutSrpConfig {
        sample_rate_hz: plan.sample_rate_hz as u32,
        sound_speed_mps: plan.speed_of_sound_mps,
        ground_height_m: Some(0.0),
        ..LayoutSrpConfig::default()
    }
}
