//! Onset-weighted short-frame SRP with spatial clustering for one analysis
//! window (port of `localize_onset_clusters` in `onset_srp_clustering.py`).
//!
//! 1. per node: sub-band flux history -> note onsets (noise-floor relative);
//! 2. per onset: template on the node's loudest channel vs delay-compensated
//!    search windows on every other sensor -> band-limited soft-SNR GCC-PHAT;
//! 3. per frame: SRP over a direction grid (one compact array) or a polar
//!    spatial grid (multi-node) -> a few peaks weighted by onset strength,
//!    peak height and map contrast;
//! 4. greedy weighted clustering -> candidate birds;
//! 5. echo / persistence checks, per-cluster refine, height / range verdicts,
//!    level checks, window-level duplicate holds.
//!
//! Pure and synchronous: hosts run it off their audio hot path.

use std::collections::BTreeMap;

use super::cluster_finalization::{finalize_cluster, WindowContext};
use super::config::OnsetClusteringConfig;
use super::echo_checks::{apply_onset_echo_and_persistence_checks, cluster_frame_peaks};
use super::frame_assembly::{build_frame, frame_peaks, group_sensors_by_node, mean_point};
use super::frame_correlation::{estimate_noise_spectrum, SensorNoiseSpectrum};
use super::geometry_prior::build_node_plane_ground_prior;
use super::geometry_resolution::{
    apply_window_level_holds, build_window_spatial_grid, ground_prior_settings,
};
use super::numeric::{python_round_to_i64, SpectralTransforms};
use super::onset_detection::{
    build_subband_history, detect_onsets, OnsetDetectionSettings, SubbandHistorySettings,
};
use super::srp_grid::{build_direction_grid, distance, Point3};
use super::trace::{ClusterSummary, FrameSummary, OnsetClusteringTrace};
use super::types::{LocalizedOnsetCluster, OnsetClusteringResult, OnsetFrame};
use crate::tdoa_solver::far_field_model_selection::COMPACT_ARRAY_MAX_APERTURE_M;

/// One analysis window: every sensor on the shared sample timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct OnsetClusteringInput {
    /// Sensor id -> samples (all the same length).
    pub sensor_signals: BTreeMap<String, Vec<f64>>,
    pub sensor_positions_m: BTreeMap<String, Point3>,
    /// Sensor id -> node id (a missing entry means the sensor is its own node).
    pub sensor_node_ids: BTreeMap<String, String>,
    pub sample_rate_hz: u32,
    pub sound_speed_mps: f64,
    pub interpolation_factor: usize,
    pub far_field_default_range_m: f64,
    /// Onsets already localized by an overlapping earlier window, per node.
    pub consumed_onset_samples_by_node: BTreeMap<String, Vec<usize>>,
    /// Set (attribution on) to attach per-node level evidence to every cluster.
    pub attribution_audible_snr_db: Option<f64>,
}

/// Localize every onset cluster in one window. The window's strongest cluster
/// is never suppressed (only annotated); tentative / echo clusters are kept
/// with their reasons so hosts can log them but should not surface them.
pub fn localize_onset_clusters(
    input: &OnsetClusteringInput,
    config: &OnsetClusteringConfig,
) -> OnsetClusteringResult {
    localize_onset_clusters_traced(input, config, None)
}

/// [`localize_onset_clusters`], also filling `trace` with the intermediate stages.
pub fn localize_onset_clusters_traced(
    input: &OnsetClusteringInput,
    config: &OnsetClusteringConfig,
    mut trace: Option<&mut OnsetClusteringTrace>,
) -> OnsetClusteringResult {
    let mut transforms = SpectralTransforms::default();
    let sample_rate = input.sample_rate_hz as f64;
    let nodes = group_sensors_by_node(input);
    let frame_samples = python_round_to_i64(config.frame_seconds * sample_rate) as usize;
    let tolerance = python_round_to_i64(config.min_separation_seconds * sample_rate);
    let noise_spectra: BTreeMap<String, SensorNoiseSpectrum> = input
        .sensor_signals
        .iter()
        .map(|(sensor, signal)| {
            (
                sensor.clone(),
                estimate_noise_spectrum(
                    signal,
                    input.sample_rate_hz,
                    frame_samples,
                    &mut transforms,
                ),
            )
        })
        .collect();
    let history_settings = SubbandHistorySettings {
        sample_rate_hz: input.sample_rate_hz,
        hop_seconds: config.hop_seconds,
        band_min_hz: config.band_min_hz,
        band_max_hz: config.band_max_hz,
        band_count: config.subband_count,
    };
    let detection_settings = OnsetDetectionSettings {
        sample_rate_hz: input.sample_rate_hz,
        flux_threshold_mad_multiplier: config.flux_threshold_mad_multiplier,
        min_separation_seconds: config.min_separation_seconds,
        max_onsets: config.max_onsets_per_window,
        active_band_snr_db: config.active_band_snr_db,
        frame_seconds: config.frame_seconds,
    };
    let mut histories = BTreeMap::new();
    let mut frames: BTreeMap<usize, OnsetFrame> = BTreeMap::new();
    let mut onset_count = 0;
    for (node_id, node_sensors) in &nodes {
        let length = input.sensor_signals[&node_sensors[0]].len();
        let node_mix: Vec<f64> = (0..length)
            .map(|index| {
                node_sensors
                    .iter()
                    .map(|s| input.sensor_signals[s][index])
                    .sum::<f64>()
                    / node_sensors.len() as f64
            })
            .collect();
        let history = build_subband_history(&node_mix, &history_settings, &mut transforms);
        let onsets = detect_onsets(&history, &detection_settings);
        if let Some(trace) = trace.as_deref_mut() {
            trace.onsets_by_node.push((node_id.clone(), onsets.clone()));
        }
        histories.insert(node_id.clone(), history);
        onset_count += onsets.len();
        let consumed = input
            .consumed_onset_samples_by_node
            .get(node_id)
            .cloned()
            .unwrap_or_default();
        for onset in &onsets {
            if consumed
                .iter()
                .any(|other| (onset.sample_index as i64 - *other as i64).abs() <= tolerance)
            {
                continue;
            }
            if let Some(frame) = build_frame(
                frames.len(),
                node_id,
                node_sensors,
                onset,
                input,
                &noise_spectra,
                config,
                &mut transforms,
            ) {
                frames.insert(frame.frame_index, frame);
            }
        }
    }
    let consumed_now = nodes
        .keys()
        .map(|node| {
            let mut samples: Vec<usize> = frames
                .values()
                .filter(|f| &f.template_node_id == node)
                .map(|f| f.onset_sample)
                .collect();
            samples.sort_unstable();
            (node.clone(), samples)
        })
        .collect();
    if frames.is_empty() {
        return OnsetClusteringResult {
            clusters: Vec::new(),
            onset_count,
            frame_count: 0,
            consumed_onset_samples_by_node: consumed_now,
        };
    }
    let node_centroids: BTreeMap<String, Point3> = nodes
        .iter()
        .map(|(node, members)| {
            (
                node.clone(),
                mean_point(
                    &members
                        .iter()
                        .map(|s| input.sensor_positions_m[s])
                        .collect::<Vec<_>>(),
                ),
            )
        })
        .collect();
    let centroid_list: Vec<Point3> = node_centroids.values().copied().collect();
    let layout_centroid = mean_point(&centroid_list);
    let positions: Vec<&Point3> = input.sensor_positions_m.values().collect();
    let aperture = positions
        .iter()
        .flat_map(|a| positions.iter().map(move |b| distance(a, b)))
        .fold(0.0, f64::max);
    let prior = build_node_plane_ground_prior(&centroid_list, &ground_prior_settings(config));
    let grid = if aperture <= COMPACT_ARRAY_MAX_APERTURE_M {
        build_direction_grid(
            layout_centroid,
            config.srp_azimuth_step_deg,
            config.srp_elevation_step_deg,
            input.far_field_default_range_m,
        )
    } else {
        build_window_spatial_grid(layout_centroid, &centroid_list, config)
    };
    let peaks = frame_peaks(&mut frames, &grid, &prior, input, config);
    let mut clusters = cluster_frame_peaks(&peaks, &grid, config);
    apply_onset_echo_and_persistence_checks(
        &mut clusters,
        &frames,
        input.sample_rate_hz,
        config,
        grid.is_direction_grid,
    );
    if let Some(trace) = trace.as_deref_mut() {
        trace.frames = frames.values().map(FrameSummary::from_frame).collect();
        trace.peaks = peaks.clone();
        trace.is_direction_grid = grid.is_direction_grid;
        trace.grid_point_count = grid.points_m.len();
        trace.clusters = clusters.iter().map(ClusterSummary::from_cluster).collect();
    }
    let frame_count = frames.len();
    let context = WindowContext {
        input,
        config,
        frames,
        nodes,
        node_centroids,
        histories,
        grid,
        prior,
        frame_samples,
    };
    let mut localized: Vec<LocalizedOnsetCluster> = clusters
        .into_iter()
        .enumerate()
        .map(|(rank, cluster)| finalize_cluster(cluster, &context, rank == 0))
        .collect();
    apply_window_level_holds(
        &mut localized,
        &context.grid,
        &centroid_list,
        input.sound_speed_mps,
        config,
    );
    OnsetClusteringResult {
        clusters: localized,
        onset_count,
        frame_count,
        consumed_onset_samples_by_node: consumed_now,
    }
}
