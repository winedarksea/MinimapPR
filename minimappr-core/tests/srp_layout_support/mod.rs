//! Shared helpers for layout-SRP scenario tests: build sim scenes, render
//! them, turn planned devices into `LayoutNode`s with their true clocks
//! (standing in for a converged clock filter), and score results.
#![allow(dead_code)]

use minimappr_core::srp_layout::{
    IslandAnalysisWindow, LayoutNode, LayoutSrpConfig, LayoutSrpResult, LayoutSrpSearcher, NodeClip, NodeClockModel,
};
use minimappr_sim::render::render_world;
use minimappr_sim::scenario::devices::{ClockConfig, DeviceConfig, DeviceKind};
use minimappr_sim::scenario::sources::{CallConfig, SourceConfig};
use minimappr_sim::scenario::Scenario;
use minimappr_sim::signals::calls::{CallGenerator, FmSweepParams, HarshParams, TrillParams};
use minimappr_sim::world::{plan_world, WorldPlan};

pub const PARTS_PER_MILLION: f64 = 1.0e6;
/// Loud-ish songbird; keeps 150 m sources above the 35 dB ambient floor.
pub const TEST_CALL_LEVEL_DB_SPL_AT_1M: f64 = 92.0;
pub const CALL_START_S: f64 = 1.0;
/// Window padding around the earliest/latest arrival.
pub const WINDOW_PADDING_S: f64 = 0.05;

pub fn phone(device_id: &str, position_m: [f64; 3], clock: ClockConfig) -> DeviceConfig {
    DeviceConfig {
        device_id: device_id.into(),
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

pub fn tetra(device_id: &str, position_m: [f64; 3]) -> DeviceConfig {
    DeviceConfig {
        device_id: device_id.into(),
        kind: DeviceKind::SirithTetra,
        position_m,
        trajectory: None,
        clock: None,
        mic_response: None,
        second_mic: None,
        speaker: Default::default(),
        limiter: None,
    }
}

pub fn fixed_clock(offset_s: f64, drift_ppm: f64) -> ClockConfig {
    ClockConfig::Fixed { offset_s, drift_ppm }
}

/// Rotate through broadband-ish call types (tonal whistles are a separate,
/// harder case: cycle skipping needs the envelope path, not built here).
pub fn call_generator(variant: usize) -> CallGenerator {
    match variant % 3 {
        0 => CallGenerator::Trill(TrillParams { duration_s: 0.5, ..TrillParams::default() }),
        1 => CallGenerator::FmSweep(FmSweepParams { duration_s: 0.3, ..FmSweepParams::default() }),
        _ => CallGenerator::Harsh(HarshParams::default()),
    }
}

/// Forest-ish scene: ambient noise, ground reflection, air absorption.
pub fn scene(devices: Vec<DeviceConfig>, source_position_m: [f64; 3], call_variant: usize, seed: u64) -> Scenario {
    scene_with_call_start(devices, source_position_m, call_variant, seed, CALL_START_S)
}

pub fn scene_with_call_start(
    devices: Vec<DeviceConfig>,
    source_position_m: [f64; 3],
    call_variant: usize,
    seed: u64,
    call_start_s: f64,
) -> Scenario {
    let mut scenario = Scenario {
        name: "srp_layout".into(),
        seed,
        duration_s: call_start_s + 2.5,
        devices,
        sources: vec![SourceConfig {
            individual_id: "bird".into(),
            species: "test".into(),
            label: None,
            label_category: "bird".into(),
            position_m: source_position_m,
            trajectory: None,
            calls: vec![CallConfig {
                start_s: call_start_s,
                level_db_spl_at_1m: TEST_CALL_LEVEL_DB_SPL_AT_1M,
                generator: call_generator(call_variant),
                event_id: None,
            }],
        }],
        ..Scenario::default()
    };
    scenario.ground.height_m = 0.0;
    scenario
}

pub struct RenderedScene {
    pub plan: WorldPlan,
    pub channels_per_device: Vec<Vec<Vec<f32>>>,
}

pub fn render(scenario: &Scenario) -> RenderedScene {
    let plan = plan_world(scenario).expect("plan");
    let rendered = render_world(&plan);
    RenderedScene {
        channels_per_device: rendered.into_iter().map(|device| device.channels).collect(),
        plan,
    }
}

/// Nodes with each device's true clock as its synced clock model.
pub fn layout_nodes(plan: &WorldPlan) -> Vec<LayoutNode> {
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
        })
        .collect()
}

/// Island window covering the first call's arrival at every node.
pub fn call_window(plan: &WorldPlan) -> IslandAnalysisWindow {
    let emission = &plan.emissions[0];
    let source = emission.trajectory.position_at(emission.start_time_s);
    let distances: Vec<f64> = plan
        .devices
        .iter()
        .map(|device| {
            let position = device.trajectory.position_at(0.0);
            ((position[0] - source[0]).powi(2) + (position[1] - source[1]).powi(2) + (position[2] - source[2]).powi(2)).sqrt()
        })
        .collect();
    let nearest = distances.iter().copied().fold(f64::INFINITY, f64::min);
    let farthest = distances.iter().copied().fold(0.0, f64::max);
    let start = emission.start_time_s + nearest / plan.speed_of_sound_mps - WINDOW_PADDING_S;
    let end = emission.start_time_s
        + emission.duration_s(plan.sample_rate_hz)
        + farthest / plan.speed_of_sound_mps
        + WINDOW_PADDING_S;
    IslandAnalysisWindow {
        start_island_time_s: start,
        sample_count: ((end - start) * plan.sample_rate_hz) as usize,
    }
}

pub fn config_for(plan: &WorldPlan) -> LayoutSrpConfig {
    LayoutSrpConfig {
        sample_rate_hz: plan.sample_rate_hz as u32,
        sound_speed_mps: plan.speed_of_sound_mps,
        ground_height_m: Some(0.0),
        ..LayoutSrpConfig::default()
    }
}

pub fn localize_with(rendered: &RenderedScene, nodes: &[LayoutNode], config: &LayoutSrpConfig) -> LayoutSrpResult {
    let clips: Vec<NodeClip<'_>> = rendered
        .channels_per_device
        .iter()
        .map(|channels| NodeClip {
            channels: channels.iter().map(Vec::as_slice).collect(),
            first_sample_index: 0,
        })
        .collect();
    LayoutSrpSearcher::new(nodes, config)
        .expect("searcher")
        .localize(&clips, call_window(&rendered.plan))
        .expect("localize")
}

pub fn localize(rendered: &RenderedScene) -> LayoutSrpResult {
    localize_with(rendered, &layout_nodes(&rendered.plan), &config_for(&rendered.plan))
}

pub fn horizontal_error_m(estimate_m: [f64; 3], truth_m: [f64; 3]) -> f64 {
    (estimate_m[0] - truth_m[0]).hypot(estimate_m[1] - truth_m[1])
}

/// Bearing error (degrees) of `estimate_azimuth_rad` against the truth's
/// azimuth from `origin_xy_m`.
pub fn bearing_error_deg(estimate_azimuth_rad: f64, origin_xy_m: [f64; 2], truth_m: [f64; 3]) -> f64 {
    let truth_azimuth = (truth_m[1] - origin_xy_m[1]).atan2(truth_m[0] - origin_xy_m[0]);
    let difference = (estimate_azimuth_rad - truth_azimuth + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
        - std::f64::consts::PI;
    difference.abs().to_degrees()
}

pub fn percentile(values: &[f64], fraction: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted[index]
}

/// Irregular five-phone group, ~24 m across, phones at hand height.
pub fn near_phone_layout(seed_offset: u64) -> Vec<DeviceConfig> {
    let positions = [[0.0, 0.0], [14.0, 3.0], [18.0, 15.0], [6.0, 21.0], [-5.0, 11.0]];
    positions
        .iter()
        .enumerate()
        .map(|(index, xy)| {
            let offset_s = 0.003 * ((index as u64 + seed_offset) % 5) as f64 - 0.006;
            phone(&format!("phone-{index}"), [xy[0], xy[1], 1.3 + 0.1 * index as f64], fixed_clock(offset_s, 0.0))
        })
        .collect()
}
