//! Frame assembly for one window: sensors grouped by node, one onset frame
//! (template-vs-search correlations) per onset, and each frame's SRP peaks.
//! Split from `pipeline.rs` (port of `_build_frame` and the peak loop in
//! `onset_srp_clustering.py`).

use std::collections::BTreeMap;

use super::config::OnsetClusteringConfig;
use super::frame_correlation::{
    correlate_template_against_search, CorrelationSettings, SensorNoiseSpectrum, TemplateSearchPair,
};
use super::geometry_prior::NodePlaneGroundPrior;
use super::geometry_resolution::cap_intra_node_weight_share;
use super::numeric::{python_round_to_i64, SpectralTransforms};
use super::onset_detection::DetectedOnset;
use super::pipeline::OnsetClusteringInput;
use super::srp_grid::{
    distance, pick_frame_peaks, steered_response_power, CandidateGrid, LagCache,
    PeakPickingSettings, Point3,
};
use super::types::{FramePeak, OnsetFrame};
use crate::tdoa_solver::far_field_model_selection::COMPACT_ARRAY_MAX_APERTURE_M;

/// Extra lag (samples) beyond the geometric maximum when sizing search windows.
pub const SEARCH_LAG_MARGIN_SAMPLES: usize = 2;

pub(super) fn group_sensors_by_node(input: &OnsetClusteringInput) -> BTreeMap<String, Vec<String>> {
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for sensor in input.sensor_signals.keys() {
        let node = input
            .sensor_node_ids
            .get(sensor)
            .cloned()
            .unwrap_or_else(|| sensor.clone());
        groups.entry(node).or_default().push(sensor.clone());
    }
    groups
}

pub(super) fn mean_point(points: &[Point3]) -> Point3 {
    let mut sum = [0.0; 3];
    for point in points {
        for axis in 0..3 {
            sum[axis] += point[axis];
        }
    }
    [
        sum[0] / points.len() as f64,
        sum[1] / points.len() as f64,
        sum[2] / points.len() as f64,
    ]
}

pub(super) fn build_frame(
    frame_index: usize,
    node_id: &str,
    node_sensors: &[String],
    onset: &DetectedOnset,
    input: &OnsetClusteringInput,
    noise_spectra: &BTreeMap<String, SensorNoiseSpectrum>,
    config: &OnsetClusteringConfig,
    transforms: &mut SpectralTransforms,
) -> Option<OnsetFrame> {
    let sample_rate = input.sample_rate_hz as f64;
    let frame_samples = python_round_to_i64(config.frame_seconds * sample_rate) as usize;
    let start = onset.sample_index as i64
        - python_round_to_i64(config.frame_pre_roll_seconds * sample_rate);
    let total = input.sensor_signals.values().next()?.len();
    if start < 0 || start as usize + frame_samples > total {
        return None;
    }
    let start = start as usize;
    let energy = |sensor: &String| {
        input.sensor_signals[sensor][start..start + frame_samples]
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
    };
    let mut reference_sensor = &node_sensors[0];
    for sensor in node_sensors {
        if energy(sensor) > energy(reference_sensor) {
            reference_sensor = sensor;
        }
    }
    let template = &input.sensor_signals[reference_sensor][start..start + frame_samples];
    let reference_position = input.sensor_positions_m[reference_sensor];
    let mut curves = Vec::new();
    for (sensor, signal) in &input.sensor_signals {
        if sensor == reference_sensor {
            continue;
        }
        let baseline_m = distance(&input.sensor_positions_m[sensor], &reference_position);
        let max_lag = (baseline_m / input.sound_speed_mps * sample_rate).ceil() as usize
            + SEARCH_LAG_MARGIN_SAMPLES;
        if start < max_lag || start + frame_samples + max_lag > total {
            continue;
        }
        let pair = TemplateSearchPair {
            sensor_a: reference_sensor,
            sensor_b: sensor,
            template,
            search: &signal[start - max_lag..start + frame_samples + max_lag],
            max_lag_samples: max_lag,
            template_noise: &noise_spectra[reference_sensor],
            search_noise: &noise_spectra[sensor],
        };
        let settings = CorrelationSettings {
            sample_rate_hz: input.sample_rate_hz,
            band_min_hz: onset.band_min_hz,
            band_max_hz: onset.band_max_hz,
            interpolation_factor: input.interpolation_factor,
            narrowband_bandwidth_hz: config.narrowband_bandwidth_hz,
            allow_envelope: baseline_m > COMPACT_ARRAY_MAX_APERTURE_M,
        };
        if let Some(curve) = correlate_template_against_search(&pair, &settings, transforms) {
            curves.push(curve);
        }
    }
    if curves.is_empty() {
        return None;
    }
    Some(OnsetFrame {
        frame_index,
        template_node_id: node_id.to_string(),
        reference_sensor: reference_sensor.clone(),
        onset_sample: onset.sample_index,
        strength: onset.strength,
        band_min_hz: onset.band_min_hz,
        band_max_hz: onset.band_max_hz,
        curves: cap_intra_node_weight_share(
            curves,
            node_sensors,
            config.frame_max_intra_node_weight_share,
        ),
        weight: 0.0,
    })
}

/// Per-frame SRP peaks (planar layouts fold peaks above the node plane so a
/// bird's twins cluster together; LM + height resolution pick z afterwards).
pub(super) fn frame_peaks(
    frames: &mut BTreeMap<usize, OnsetFrame>,
    grid: &CandidateGrid,
    prior: &NodePlaneGroundPrior,
    input: &OnsetClusteringInput,
    config: &OnsetClusteringConfig,
) -> Vec<FramePeak> {
    let fold = !grid.is_direction_grid && prior.is_planar;
    let mut lag_cache = LagCache::new(grid, &input.sensor_positions_m, input.sound_speed_mps);
    let max_strength = frames
        .values()
        .map(|frame| frame.strength)
        .fold(f64::NEG_INFINITY, f64::max);
    let picking = PeakPickingSettings {
        max_peaks: config.srp_peaks_per_frame,
        min_relative_height: config.srp_peak_min_relative_height,
        suppression_radius_m: config.cluster_radius_m,
        suppression_range_fraction: config.cluster_range_fraction,
        suppression_angle_deg: config.cluster_angle_deg,
    };
    let mut peaks = Vec::new();
    for frame in frames.values_mut() {
        frame.weight = frame.strength / max_strength;
        let power = steered_response_power(&frame.curves, &mut lag_cache);
        for (rank, grid_peak) in pick_frame_peaks(&power, grid, &picking)
            .into_iter()
            .enumerate()
        {
            let point = grid.points_m[grid_peak.point_index];
            let direction = grid.directions[grid_peak.point_index];
            peaks.push(FramePeak {
                frame_index: frame.frame_index,
                position_m: if fold {
                    prior.fold_above_plane(&point)
                } else {
                    point
                },
                direction: if fold {
                    prior.fold_direction_above_plane(&direction)
                } else {
                    direction
                },
                range_m: grid.ranges_m[grid_peak.point_index],
                value: grid_peak.value,
                weight: frame.weight * grid_peak.value * grid_peak.contrast,
                is_frame_primary: rank == 0,
            });
        }
    }
    peaks
}
