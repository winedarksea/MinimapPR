//! Peak clustering plus telling a second bird from an echo or sidelobe (port
//! of `cluster_frame_peaks` in `onset_srp_clustering.py`,
//! `onset_cluster_echo_checks.py` and `onset_near_node_level_check.py`).
//!
//! 1. Independent onsets: an echo is time-locked 3-30 ms behind its source's
//!    onset in the same band; a secondary cluster not led by onsets of its own
//!    is an echo.
//! 2. Level consistency (multi-node): nearer nodes must hold more
//!    noise-normalised in-band energy; a failure holds a secondary cluster
//!    tentative and only annotates the strongest (a faint far bird is never
//!    rejected on level). The near-node check adds the spreading prediction.
//! 3. Persistence (single compact array): a secondary cluster needs several
//!    independent onsets.

use std::collections::BTreeMap;

use super::config::OnsetClusteringConfig;
use super::numeric::{floor_divide, mean, median, python_round_to_i64};
use super::onset_detection::SubbandHistory;
use super::srp_grid::{
    angular_separation_deg, distance, spatial_cluster_radius_m, CandidateGrid, Point3,
};
use super::types::{ClusterStatus, FramePeak, OnsetCluster, OnsetFrame};

/// Bands must overlap by at least this fraction of the narrower one to be "same band".
pub const SAME_BAND_MIN_OVERLAP_FRACTION: f64 = 0.5;
/// Free-field spreading: 20 log10(d_far / d_near) dB.
const SPREADING_DECIBELS_PER_DECADE: f64 = 20.0;

fn peak_belongs(
    peak: &FramePeak,
    cluster: &OnsetCluster,
    grid: &CandidateGrid,
    config: &OnsetClusteringConfig,
) -> bool {
    if grid.is_direction_grid || peak.range_m >= grid.max_range_m {
        return angular_separation_deg(&peak.direction, &cluster.weighted_direction())
            <= config.cluster_angle_deg;
    }
    let centroid = cluster.weighted_position_m();
    let radius = spatial_cluster_radius_m(
        peak.range_m.min(distance(&centroid, &grid.centroid_m)),
        config.cluster_radius_m,
        config.cluster_range_fraction,
    );
    distance(&peak.position_m, &centroid) <= radius
}

/// Greedy weighted clustering, heaviest peak first; strongest cluster first.
pub fn cluster_frame_peaks(
    peaks: &[FramePeak],
    grid: &CandidateGrid,
    config: &OnsetClusteringConfig,
) -> Vec<OnsetCluster> {
    let mut ordered: Vec<&FramePeak> = peaks.iter().collect();
    ordered.sort_by(|a, b| b.weight.total_cmp(&a.weight));
    let mut clusters: Vec<OnsetCluster> = Vec::new();
    for peak in ordered {
        match clusters
            .iter_mut()
            .find(|cluster| peak_belongs(peak, cluster, grid, config))
        {
            Some(home) => home.peaks.push(*peak),
            None => clusters.push(OnsetCluster::with_peak(*peak)),
        }
    }
    clusters.sort_by(|a, b| b.weight().total_cmp(&a.weight()));
    let Some(strongest) = clusters.first() else {
        return clusters;
    };
    let floor = config.cluster_min_weight_fraction * strongest.weight();
    clusters
        .into_iter()
        .filter(|cluster| cluster.weight() >= floor)
        .take(config.cluster_max_count)
        .collect()
}

pub fn band_overlap_fraction(frame_a: &OnsetFrame, frame_b: &OnsetFrame) -> f64 {
    let low = frame_a.band_min_hz.max(frame_b.band_min_hz);
    let high = frame_a.band_max_hz.min(frame_b.band_max_hz);
    let narrower =
        (frame_a.band_max_hz - frame_a.band_min_hz).min(frame_b.band_max_hz - frame_b.band_min_hz);
    if narrower <= 0.0 {
        return 0.0;
    }
    (high - low).max(0.0) / narrower
}

fn frame_trails_stronger_onset(
    frame: &OnsetFrame,
    stronger_led_frames: &[&OnsetFrame],
    min_lag: i64,
    max_lag: i64,
) -> bool {
    stronger_led_frames.iter().any(|leader| {
        let lag = frame.onset_sample as i64 - leader.onset_sample as i64;
        leader.template_node_id == frame.template_node_id
            && (min_lag..=max_lag).contains(&lag)
            && band_overlap_fraction(frame, leader) >= SAME_BAND_MIN_OVERLAP_FRACTION
    })
}

/// Mutates statuses in place; `clusters` strongest first. Persistence applies
/// to single compact arrays only (multi-node windows use the level check).
pub fn apply_onset_echo_and_persistence_checks(
    clusters: &mut [OnsetCluster],
    frames: &BTreeMap<usize, OnsetFrame>,
    sample_rate_hz: u32,
    config: &OnsetClusteringConfig,
    apply_persistence: bool,
) {
    let sample_rate = sample_rate_hz as f64;
    let min_lag = python_round_to_i64(config.echo_min_lag_seconds * sample_rate);
    let max_lag = python_round_to_i64(config.echo_max_lag_seconds * sample_rate);
    for rank in 0..clusters.len() {
        let stronger_led_frames: Vec<&OnsetFrame> = clusters[..rank]
            .iter()
            .flat_map(|other| other.led_frame_indices())
            .map(|index| &frames[&index])
            .collect();
        let independent = clusters[rank]
            .led_frame_indices()
            .into_iter()
            .filter(|index| {
                !frame_trails_stronger_onset(&frames[index], &stronger_led_frames, min_lag, max_lag)
            })
            .count();
        let cluster = &mut clusters[rank];
        cluster.independent_onset_count = independent;
        if rank == 0 {
            continue;
        }
        let independent_fraction = independent as f64 / cluster.frame_indices().len().max(1) as f64;
        if independent_fraction < config.echo_min_independent_fraction {
            cluster.status = ClusterStatus::Echo;
            cluster
                .status_reasons
                .push("no_independent_onsets".to_string());
        } else if apply_persistence && independent < config.persistence_min_independent_onsets {
            cluster.status = ClusterStatus::Tentative;
            cluster.status_reasons.push("not_persistent".to_string());
        }
    }
}

/// Mean in-band dB above the node's own noise floor over one frame.
pub fn node_band_snr_db(
    history: &SubbandHistory,
    start_sample: i64,
    frame_samples: usize,
    band_min_hz: f64,
    band_max_hz: f64,
) -> Option<f64> {
    let hop = history.hop_samples as i64;
    let first_hop = floor_divide(start_sample, hop).max(0);
    let last_hop = (history.hop_count() as i64)
        .min(floor_divide(start_sample + frame_samples as i64, hop) + 1);
    if last_hop <= first_hop {
        return None;
    }
    let bands: Vec<usize> = history
        .band_centers_hz()
        .iter()
        .enumerate()
        .filter(|(_, centre)| **centre >= band_min_hz && **centre <= band_max_hz)
        .map(|(band, _)| band)
        .collect();
    if bands.is_empty() {
        return None;
    }
    let mut values = Vec::with_capacity(bands.len() * (last_hop - first_hop) as usize);
    for row in first_hop as usize..last_hop as usize {
        for band in &bands {
            values.push(history.snr_db(row, *band));
        }
    }
    Some(mean(&values))
}

/// Inputs shared by the per-cluster level checks.
pub struct LevelCheckContext<'window> {
    pub led_frames: &'window [&'window OnsetFrame],
    pub node_histories: &'window BTreeMap<String, SubbandHistory>,
    pub node_positions_m: &'window BTreeMap<String, Point3>,
    pub sound_speed_mps: f64,
    pub sample_rate_hz: u32,
    pub frame_samples: usize,
}

/// (nearest node, farthest node, distances) for a fix; first-min / first-max like Python.
fn nearest_and_farthest(
    position_m: &Point3,
    nodes: &BTreeMap<String, Point3>,
) -> (String, String, BTreeMap<String, f64>) {
    let distances: BTreeMap<String, f64> = nodes
        .iter()
        .map(|(node, centre)| (node.clone(), distance(position_m, centre)))
        .collect();
    let mut nearest = distances.keys().next().cloned().unwrap_or_default();
    let mut farthest = nearest.clone();
    for (node, value) in &distances {
        if *value < distances[&nearest] {
            nearest = node.clone();
        }
        if *value > distances[&farthest] {
            farthest = node.clone();
        }
    }
    (nearest, farthest, distances)
}

/// Median over led frames of (nearest - farthest) band SNR at predicted arrivals.
fn median_level_difference(
    context: &LevelCheckContext<'_>,
    nearest: &str,
    farthest: &str,
    distances: &BTreeMap<String, f64>,
) -> Option<f64> {
    let mut differences = Vec::new();
    for frame in context.led_frames {
        let template_distance = distances
            .get(&frame.template_node_id)
            .copied()
            .unwrap_or(distances[nearest]);
        let reading = |node: &str| {
            let delay_samples = (distances[node] - template_distance) / context.sound_speed_mps
                * context.sample_rate_hz as f64;
            let arrival = frame.onset_sample as i64 + python_round_to_i64(delay_samples);
            node_band_snr_db(
                &context.node_histories[node],
                arrival,
                context.frame_samples,
                frame.band_min_hz,
                frame.band_max_hz,
            )
        };
        if let (Some(near), Some(far)) = (reading(nearest), reading(farthest)) {
            differences.push(near - far);
        }
    }
    (!differences.is_empty()).then(|| median(&differences))
}

/// Nearest node's band SNR must not trail the farthest's by more than `margin_db`.
pub fn level_consistency_holds(
    position_m: &Point3,
    context: &LevelCheckContext<'_>,
    margin_db: f64,
) -> bool {
    if context.node_histories.len() < 2 || context.led_frames.is_empty() {
        return true;
    }
    let (nearest, farthest, distances) = nearest_and_farthest(position_m, context.node_positions_m);
    if nearest == farthest {
        return true;
    }
    median_level_difference(context, &nearest, &farthest, &distances)
        .is_none_or(|difference| difference >= -margin_db)
}

/// Within `near_field_radius_m` of a node, the level gradient must reach the
/// spreading prediction minus `tolerance_db`.
pub fn near_node_level_gradient_holds(
    position_m: &Point3,
    context: &LevelCheckContext<'_>,
    near_field_radius_m: f64,
    tolerance_db: f64,
) -> bool {
    if context.node_positions_m.len() < 2 || context.led_frames.is_empty() {
        return true;
    }
    let (nearest, farthest, distances) = nearest_and_farthest(position_m, context.node_positions_m);
    if distances[&nearest] > near_field_radius_m
        || nearest == farthest
        || distances[&nearest] <= 0.0
    {
        return true;
    }
    let expected_db =
        SPREADING_DECIBELS_PER_DECADE * (distances[&farthest] / distances[&nearest]).log10();
    median_level_difference(context, &nearest, &farthest, &distances)
        .is_none_or(|difference| difference >= expected_db - tolerance_db)
}

/// A near-field fix whose every leading onset was templated on that node: the
/// shells, not the data, chose the range.
pub fn near_node_fix_led_only_by_that_node(
    position_m: &Point3,
    led_frames: &[&OnsetFrame],
    nodes: &BTreeMap<String, Point3>,
    near_field_radius_m: f64,
) -> bool {
    if nodes.len() < 2 || led_frames.is_empty() {
        return false;
    }
    let (nearest, _, distances) = nearest_and_farthest(position_m, nodes);
    distances[&nearest] <= near_field_radius_m
        && led_frames
            .iter()
            .all(|frame| frame.template_node_id == nearest)
}
