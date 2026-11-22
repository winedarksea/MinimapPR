//! Optional record of a window's intermediate stages (onsets, frames, frame
//! peaks, clusters after the echo checks). Parity tests compare each stage
//! against the Python pipeline so a divergence is pinned to its first stage;
//! hosts can also log it. Collected only when requested.

use super::numeric::argmax;
use super::onset_detection::DetectedOnset;
use super::types::{FramePeak, OnsetCluster, OnsetFrame};

#[derive(Debug, Clone, PartialEq)]
pub struct CurveSummary {
    pub sensor_b: String,
    pub weight_sum: f64,
    pub used_envelope: bool,
    pub effective_bandwidth_hz: f64,
    pub centre_frequency_hz: f64,
    pub peak_lag_s: f64,
    pub peak_value: f64,
    pub value_sum: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrameSummary {
    pub frame_index: usize,
    pub template_node_id: String,
    pub reference_sensor: String,
    pub onset_sample: usize,
    pub weight: f64,
    pub curves: Vec<CurveSummary>,
}

impl FrameSummary {
    pub fn from_frame(frame: &OnsetFrame) -> Self {
        Self {
            frame_index: frame.frame_index,
            template_node_id: frame.template_node_id.clone(),
            reference_sensor: frame.reference_sensor.clone(),
            onset_sample: frame.onset_sample,
            weight: frame.weight,
            curves: frame
                .curves
                .iter()
                .map(|curve| {
                    let peak = argmax(&curve.values);
                    CurveSummary {
                        sensor_b: curve.sensor_b.clone(),
                        weight_sum: curve.weight_sum,
                        used_envelope: curve.used_envelope,
                        effective_bandwidth_hz: curve.effective_bandwidth_hz,
                        centre_frequency_hz: curve.centre_frequency_hz,
                        peak_lag_s: curve.lags_s[peak],
                        peak_value: curve.values[peak],
                        value_sum: curve.values.iter().sum(),
                    }
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClusterSummary {
    pub frame_indices: Vec<usize>,
    pub status: &'static str,
    pub status_reasons: Vec<String>,
    pub independent_onset_count: usize,
    pub weight: f64,
}

impl ClusterSummary {
    pub fn from_cluster(cluster: &OnsetCluster) -> Self {
        Self {
            frame_indices: cluster.frame_indices(),
            status: cluster.status.as_str(),
            status_reasons: cluster.status_reasons.clone(),
            independent_onset_count: cluster.independent_onset_count,
            weight: cluster.weight(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct OnsetClusteringTrace {
    /// (node id, onsets detected on its mix), nodes in id order.
    pub onsets_by_node: Vec<(String, Vec<DetectedOnset>)>,
    pub frames: Vec<FrameSummary>,
    pub peaks: Vec<FramePeak>,
    pub is_direction_grid: bool,
    pub grid_point_count: usize,
    /// Clusters after the echo / persistence checks, before refinement.
    pub clusters: Vec<ClusterSummary>,
}
