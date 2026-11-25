//! The shadow onset-clustering localizer, as the sidecar would run it live.
//!
//! One window per trigger, long enough that every node's copy of an onset
//! and its search span fit (`frame_assembly.rs::build_frame` skips a pair
//! whose search window leaves the buffer): from one layout lag plus a
//! pre-roll before the trigger to two layout lags plus a post-roll after.
//! `localize_onset_clusters` finds every source in it; accepted clusters
//! with at least three nodes become fixes, after the same noise-onset gate
//! the Field Mode host applies (the harness then applies the server's
//! reportability gates to every candidate alike).
//!
//! Onset clustering knows nothing of clocks or surveyed-position error, so
//! adopting it live means carrying the R5 policy into what it publishes: the
//! cluster covariance plus every node's clock σ (class default) and position
//! σ / c, propagated through the reference-TDOA Jacobian.

use std::collections::BTreeMap;
use std::time::Instant;

use minimappr_core::onset_clustering::attribution_decision::AttributionThresholds;
use minimappr_core::onset_clustering::{localize_onset_clusters, ClusterStatus, LocalizedOnsetCluster, OnsetClusteringConfig, OnsetClusteringInput};

use crate::bakeoff_candidate_fix::{CandidateFix, CandidateSolveOutcome};
use crate::bakeoff_config::OnsetCandidateConfig;
use crate::bakeoff_covariance_propagation::{add_covariances, node_offset_position_covariance_m2, ReferenceTdoaGeometry};
use crate::bakeoff_production_candidate::sensor_id;
use crate::bakeoff_scenes::BakeoffScene;
use crate::bakeoff_windows::{cut_window, BakeoffTrigger};

/// Layout lags spanned after the trigger: the far node's onset (one lag)
/// plus its search span (another lag).
const LAYOUT_LAGS_AFTER_TRIGGER: f64 = 2.0;

pub fn onset_clustering_config(config: &OnsetCandidateConfig, aperture_m: f64) -> OnsetClusteringConfig {
    let mut onset = OnsetClusteringConfig::default();
    if config.grid_range_aperture_multiple > 0.0 {
        onset.srp_grid_max_range_m = onset.srp_grid_max_range_m.max(config.grid_range_aperture_multiple * aperture_m);
        onset.range_profile_max_range_m = onset.range_profile_max_range_m.max(config.grid_range_aperture_multiple * aperture_m);
    }
    onset
}

fn node_index_of(sensor: &str, node_count: usize) -> Option<usize> {
    (0..node_count).find(|node| sensor_id(*node) == sensor)
}

fn audible_somewhere(cluster: &LocalizedOnsetCluster, config: &OnsetCandidateConfig) -> bool {
    cluster.attribution_evidence.as_ref().is_none_or(|evidence| evidence.node_band_snr_db.values().any(|snr| *snr >= config.minimum_loudest_node_band_snr_db))
}

fn published_covariance(scene: &BakeoffScene, cluster: &LocalizedOnsetCluster, cluster_covariance: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let node_count = scene.nodes.len();
    let reference_node = node_index_of(&cluster.reference_sensor, node_count)?;
    let measured_nodes: Vec<usize> =
        cluster.tdoa_s.keys().filter_map(|sensor| node_index_of(sensor, node_count)).filter(|node| *node != reference_node).collect();
    let node_positions_m = scene.measured_positions_m();
    let node_offset_std_s: Vec<f64> =
        scene.nodes.iter().map(|node| node.clock_class.declared_offset_std_s().hypot(node.stated_position_std_m / scene.sound_speed_mps)).collect();
    let geometry = ReferenceTdoaGeometry {
        position_m: cluster.position_m,
        reference_node,
        measured_nodes: &measured_nodes,
        node_positions_m: &node_positions_m,
        sound_speed_mps: scene.sound_speed_mps,
    };
    let timing_covariance = node_offset_position_covariance_m2(&geometry, &node_offset_std_s)?;
    Some(add_covariances(cluster_covariance, &timing_covariance))
}

fn cluster_fix(scene: &BakeoffScene, cluster: &LocalizedOnsetCluster, window_start_s: f64, config: &OnsetCandidateConfig) -> Option<CandidateFix> {
    if cluster.status != ClusterStatus::Accepted || !audible_somewhere(cluster, config) {
        return None;
    }
    // Nodes with a TDOA: the reference plus every measured sensor (as `OnsetClipSetLocalizer::refine_cluster` counts them).
    let tdoa_node_count = 1 + cluster.tdoa_s.keys().filter(|sensor| **sensor != cluster.reference_sensor).count();
    if tdoa_node_count < config.minimum_contributing_nodes {
        return None;
    }
    let covariance = published_covariance(scene, cluster, cluster.covariance_m2.as_ref()?)?;
    let rate = f64::from(scene.sample_rate_hz);
    let attribution_cues = cluster
        .template_node_ids
        .iter()
        .zip(&cluster.onset_samples)
        .filter_map(|(sensor, sample)| Some((node_index_of(sensor, scene.nodes.len())?, window_start_s + *sample as f64 / rate)))
        .collect();
    Some(CandidateFix {
        position_m: cluster.position_m,
        covariance_m2: Some(covariance),
        range_projection_mode: cluster.range_projection_mode.clone(),
        attribution_cues,
    })
}

pub fn run_onset_candidate(scene: &BakeoffScene, trigger: &BakeoffTrigger, config: &OnsetCandidateConfig) -> CandidateSolveOutcome {
    let layout_lag_s = scene.largest_layout_lag_s();
    let start_s = trigger.stream_time_s - layout_lag_s - config.window_pre_roll_s;
    let end_s = trigger.stream_time_s + LAYOUT_LAGS_AFTER_TRIGGER * layout_lag_s + config.window_post_roll_s;
    let mut input = OnsetClusteringInput {
        sensor_signals: BTreeMap::new(),
        sensor_positions_m: BTreeMap::new(),
        sensor_node_ids: BTreeMap::new(),
        sample_rate_hz: scene.sample_rate_hz,
        sound_speed_mps: scene.sound_speed_mps,
        interpolation_factor: config.interpolation_factor,
        far_field_default_range_m: config.far_field_default_range_m,
        consumed_onset_samples_by_node: BTreeMap::new(),
        attribution_audible_snr_db: Some(AttributionThresholds::default().audible_snr_db),
    };
    for (node, stream) in scene.streams.iter().enumerate() {
        let id = sensor_id(node);
        let window = cut_window(stream, scene.sample_rate_hz, start_s, end_s);
        input.sensor_signals.insert(id.clone(), window.into_iter().map(f64::from).collect());
        input.sensor_positions_m.insert(id.clone(), scene.nodes[node].measured_position_m);
        input.sensor_node_ids.insert(id.clone(), id);
    }
    let started = Instant::now();
    let result = localize_onset_clusters(&input, &onset_clustering_config(config, scene.aperture_m));
    let fixes: Vec<CandidateFix> = result.clusters.iter().filter_map(|cluster| cluster_fix(scene, cluster, start_s, config)).collect();
    let wall_time_s = started.elapsed().as_secs_f64();
    let failure_reason = fixes
        .is_empty()
        .then(|| format!("no publishable cluster ({} onsets, {} frames, {} clusters)", result.onset_count, result.frame_count, result.clusters.len()));
    CandidateSolveOutcome { fixes, wall_time_s, failure_reason, gate_rejections: Vec::new() }
}
