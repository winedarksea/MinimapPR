//! Layout description, clips, and the sensor/pair bookkeeping built from them.

use super::clock_resampling::{resample_onto_island_timeline, NodeClockModel};
use super::config::LayoutSrpConfig;
use super::pair_correlation::SensorPairSpec;
use super::polar_grid::{norm, sub};

/// One node: a single-mic phone (`mic_offsets_m = [[0,0,0]]`) or an array
/// (tetra: four offsets, world-frame, i.e. orientation already applied).
#[derive(Clone, Debug, PartialEq)]
pub struct LayoutNode {
    pub position_m: [f64; 3],
    pub mic_offsets_m: Vec<[f64; 3]>,
    pub clock: NodeClockModel,
    /// Relative trust in this node (e.g. from readiness SNR rank); pair
    /// weights multiply both nodes' values. 1.0 when unknown.
    pub pair_weight: f64,
}

/// A node's clip: one slice per mic, all starting at local sample
/// `first_sample_index` of that node's own stream.
#[derive(Clone, Debug)]
pub struct NodeClip<'a> {
    pub channels: Vec<&'a [f32]>,
    pub first_sample_index: i64,
}

/// The island-time span every clip is mapped onto before correlation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IslandAnalysisWindow {
    pub start_island_time_s: f64,
    pub sample_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayoutSrpError {
    /// Fewer than two sensors in total, so no pair exists.
    TooFewSensors,
    ClipCountMismatch { nodes: usize, clips: usize },
    ChannelCountMismatch { node_index: usize, mics: usize, channels: usize },
    EmptyAnalysisWindow,
    /// Every pair's peak is at its noise-only level: nothing to localize.
    NoCoherentPairs,
    /// A node position, mic offset, clock or pair weight is non-finite, or
    /// the sample rate / sound speed is not positive. Rejected up front:
    /// NaN geometry would otherwise silently burn the whole search budget.
    InvalidInput { node_index: Option<usize> },
}

/// Reject inputs that would make every interval bound NaN.
pub fn validate_layout_inputs(nodes: &[LayoutNode], config: &LayoutSrpConfig) -> Result<(), LayoutSrpError> {
    if config.sample_rate_hz == 0 || !(config.sound_speed_mps.is_finite() && config.sound_speed_mps > 0.0) {
        return Err(LayoutSrpError::InvalidInput { node_index: None });
    }
    for (node_index, node) in nodes.iter().enumerate() {
        let clock = &node.clock;
        let is_finite = node.position_m.iter().all(|value| value.is_finite())
            && node.mic_offsets_m.iter().flatten().all(|value| value.is_finite())
            && clock.anchor_sample_index.is_finite()
            && clock.anchor_island_time_s.is_finite()
            && clock.rate_ratio.is_finite()
            && clock.rate_ratio > 0.0
            && node.pair_weight.is_finite()
            && node.pair_weight >= 0.0;
        if !is_finite {
            return Err(LayoutSrpError::InvalidInput { node_index: Some(node_index) });
        }
    }
    Ok(())
}

impl std::fmt::Display for LayoutSrpError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for LayoutSrpError {}

/// A mic with its world position and owning node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayoutSensor {
    pub node_index: usize,
    pub mic_index: usize,
    pub position_m: [f64; 3],
}

/// A pair plus its geometry-only weight (before signal significance).
#[derive(Clone, Copy, Debug)]
pub struct PlannedPair {
    pub spec: SensorPairSpec,
    pub base_weight: f64,
}

pub fn layout_sensors(nodes: &[LayoutNode]) -> Vec<LayoutSensor> {
    nodes
        .iter()
        .enumerate()
        .flat_map(|(node_index, node)| {
            node.mic_offsets_m.iter().enumerate().map(move |(mic_index, offset)| LayoutSensor {
                node_index,
                mic_index,
                position_m: [
                    node.position_m[0] + offset[0],
                    node.position_m[1] + offset[1],
                    node.position_m[2] + offset[2],
                ],
            })
        })
        .collect()
}

/// Intra-node pairs for every multi-mic node (sharing the node's group
/// weight), plus one inter-node pair per node pair between their first mics.
/// Using only first mics across nodes keeps pair count O(nodes²) instead of
/// O(mics²): the extra 5 cm offsets add no independent inter-node range.
pub fn plan_sensor_pairs(nodes: &[LayoutNode], sensors: &[LayoutSensor], config: &LayoutSrpConfig) -> Vec<PlannedPair> {
    let sample_period_s = 1.0 / config.sample_rate_hz.max(1) as f64;
    let maximum_lag = |a: usize, b: usize| {
        norm(sub(sensors[a].position_m, sensors[b].position_m)) / config.sound_speed_mps
            + config.correlation.lag_margin_s
            + sample_period_s
    };
    let mut pairs = Vec::new();
    for (node_index, node) in nodes.iter().enumerate() {
        let node_sensors: Vec<usize> = (0..sensors.len()).filter(|&s| sensors[s].node_index == node_index).collect();
        let intra_pair_count = node_sensors.len() * node_sensors.len().saturating_sub(1) / 2;
        if intra_pair_count == 0 {
            continue;
        }
        let weight = config.scoring.intra_node_pair_group_weight * node.pair_weight / intra_pair_count as f64;
        for (position, &a) in node_sensors.iter().enumerate() {
            for &b in &node_sensors[position + 1..] {
                pairs.push(PlannedPair {
                    spec: SensorPairSpec { sensor_a: a, sensor_b: b, maximum_lag_s: maximum_lag(a, b) },
                    base_weight: weight,
                });
            }
        }
    }
    let first_sensor_of_node: Vec<Option<usize>> = (0..nodes.len())
        .map(|node_index| sensors.iter().position(|sensor| sensor.node_index == node_index))
        .collect();
    for node_a in 0..nodes.len() {
        for node_b in (node_a + 1)..nodes.len() {
            let (Some(a), Some(b)) = (first_sensor_of_node[node_a], first_sensor_of_node[node_b]) else {
                continue;
            };
            pairs.push(PlannedPair {
                spec: SensorPairSpec { sensor_a: a, sensor_b: b, maximum_lag_s: maximum_lag(a, b) },
                base_weight: nodes[node_a].pair_weight * nodes[node_b].pair_weight,
            });
        }
    }
    pairs
}

/// Map every clip channel onto the island window (one output per sensor,
/// in `layout_sensors` order).
pub fn align_clips_to_island_window(
    nodes: &[LayoutNode],
    clips: &[NodeClip<'_>],
    window: IslandAnalysisWindow,
    config: &LayoutSrpConfig,
) -> Result<Vec<Vec<f32>>, LayoutSrpError> {
    if clips.len() != nodes.len() {
        return Err(LayoutSrpError::ClipCountMismatch { nodes: nodes.len(), clips: clips.len() });
    }
    if window.sample_count == 0 {
        return Err(LayoutSrpError::EmptyAnalysisWindow);
    }
    let mut aligned = Vec::new();
    for (node_index, (node, clip)) in nodes.iter().zip(clips).enumerate() {
        if clip.channels.len() != node.mic_offsets_m.len() {
            return Err(LayoutSrpError::ChannelCountMismatch {
                node_index,
                mics: node.mic_offsets_m.len(),
                channels: clip.channels.len(),
            });
        }
        let clock = if config.apply_clock_rate_correction {
            node.clock
        } else {
            node.clock.without_rate_correction()
        };
        for channel in &clip.channels {
            aligned.push(
                resample_onto_island_timeline(
                    channel,
                    clip.first_sample_index,
                    &clock,
                    window.start_island_time_s,
                    window.sample_count,
                    config.sample_rate_hz as f64,
                )
                .samples,
            );
        }
    }
    Ok(aligned)
}
