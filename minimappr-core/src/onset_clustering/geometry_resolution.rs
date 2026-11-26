//! Window grid with node shells, intra-node weight cap, per-cluster height /
//! range verdicts and the window-level duplicate holds (port of
//! `minimappr/core/onset_cluster_geometry_resolution.py`).

use std::collections::BTreeMap;

use super::config::OnsetClusteringConfig;
use super::frame_correlation::PairCorrelationCurve;
use super::geometry_prior::{
    node_near_field_points, GroundPriorSettings, NodePlaneGroundPrior, NodeShellSettings,
};
use super::height_and_range::{
    add_vertical_variance, apply_vertical_support, judge_range_from_profile,
    range_support_along_bearing, resolve_node_plane_mirror, vertical_support_at, Covariance,
    RangeProfileSettings, HEIGHT_NOT_APPLICABLE,
};
use super::refinement::RefinedClusterPosition;
use super::srp_grid::{
    angular_separation_deg, augment_spatial_grid, build_spatial_grid, distance, norm, subtract,
    CandidateGrid, Point3, SpatialGridSettings,
};
use super::types::{ClusterStatus, LocalizedOnsetCluster};
use crate::range_projection::is_unobservable;

pub const SAME_SOURCE_RANGE_AMBIGUOUS_REASON: &str = "same_bearing_range_ambiguous";
pub const BEARING_ONLY_NOT_PERSISTENT_REASON: &str = "bearing_only_not_persistent";
pub const TDOA_EQUIVALENT_DUPLICATE_REASON: &str = "tdoa_equivalent_duplicate";

/// Rescale a multi-node frame's intra-node pair weights to at most `max_share`.
/// ML weights grow ~linearly with SNR, so a source 2 m from one node let that
/// node's range-blind compact pairs take ~95% of the frame and the far end of
/// its bearing ray won (S gate: a wren 2.3 m from tetra-c placed 52 m out).
pub fn cap_intra_node_weight_share(
    mut curves: Vec<PairCorrelationCurve>,
    node_sensor_ids: &[String],
    max_share: f64,
) -> Vec<PairCorrelationCurve> {
    let is_intra = |curve: &PairCorrelationCurve| node_sensor_ids.contains(&curve.sensor_b);
    let intra: f64 = curves
        .iter()
        .filter(|c| is_intra(c))
        .map(|c| c.weight_sum)
        .sum();
    let cross: f64 = curves
        .iter()
        .filter(|c| !is_intra(c))
        .map(|c| c.weight_sum)
        .sum();
    if intra <= 0.0 || cross <= 0.0 || intra / (intra + cross) <= max_share {
        return curves;
    }
    let scale = max_share * cross / ((1.0 - max_share) * intra);
    for curve in curves.iter_mut() {
        if node_sensor_ids.contains(&curve.sensor_b) {
            curve.weight_sum *= scale;
        }
    }
    curves
}

pub fn ground_prior_settings(config: &OnsetClusteringConfig) -> GroundPriorSettings {
    GroundPriorSettings {
        enabled: config.ground_prior_enabled,
        node_height_above_ground_m: config.node_height_above_ground_m,
        source_max_height_above_ground_m: config.source_max_height_above_ground_m,
        planar_layout_max_flatness_ratio: config.planar_layout_max_flatness_ratio,
    }
}

pub fn build_window_spatial_grid(
    layout_centroid_m: Point3,
    node_centroids: &[Point3],
    config: &OnsetClusteringConfig,
) -> CandidateGrid {
    let grid = build_spatial_grid(
        layout_centroid_m,
        &SpatialGridSettings {
            azimuth_step_deg: config.srp_azimuth_step_deg,
            elevation_step_deg: config.srp_elevation_step_deg,
            min_range_m: config.srp_grid_min_range_m,
            max_range_m: config.srp_grid_max_range_m,
            ring_count: config.srp_grid_ring_count,
        },
    );
    let shells = node_near_field_points(
        node_centroids,
        &NodeShellSettings {
            min_range_m: config.srp_node_grid_min_range_m,
            max_range_m: config.srp_node_grid_max_range_m,
            ring_count: config.srp_node_grid_ring_count,
            azimuth_step_deg: config.srp_node_grid_azimuth_step_deg,
            elevation_step_deg: config.srp_node_grid_elevation_step_deg,
        },
    );
    augment_spatial_grid(grid, &shells)
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedClusterGeometry {
    pub position_m: Point3,
    pub covariance_m2: Covariance,
    pub range_projection_mode: Option<String>,
    pub range_observability: Option<f64>,
    pub height_resolution: &'static str,
    pub range_interval_m: Option<(f64, f64)>,
    pub range_lower_bound_m: Option<f64>,
}

fn layout_aperture_m(sensor_positions: &BTreeMap<String, Point3>) -> f64 {
    let points: Vec<&Point3> = sensor_positions.values().collect();
    let mut aperture: f64 = 0.0;
    for a in &points {
        for b in &points {
            aperture = aperture.max(distance(a, b));
        }
    }
    aperture
}

/// Height twin first (it moves z only), then the range verdict along the bearing.
pub fn resolve_cluster_geometry(
    refined: &RefinedClusterPosition,
    score_points: &dyn Fn(&[Point3]) -> Vec<f64>,
    sensor_positions: &BTreeMap<String, Point3>,
    grid: &CandidateGrid,
    prior: &NodePlaneGroundPrior,
    config: &OnsetClusteringConfig,
) -> ResolvedClusterGeometry {
    if grid.is_direction_grid {
        return ResolvedClusterGeometry {
            position_m: refined.position_m,
            covariance_m2: refined.covariance_m2,
            range_projection_mode: Some(refined.range_projection_mode.clone()),
            range_observability: refined.range_observability,
            height_resolution: HEIGHT_NOT_APPLICABLE,
            range_interval_m: None,
            range_lower_bound_m: None,
        };
    }
    let height = resolve_node_plane_mirror(
        &refined.position_m,
        prior,
        score_points,
        config.height_mirror_min_score_ratio,
    );
    let mut covariance =
        add_vertical_variance(refined.covariance_m2, height.added_vertical_variance_m2);
    let mut height_position = height.position_m;
    if prior.is_planar && prior.enabled {
        if let Some(z_bounds) = prior.vertical_bounds_m(&height_position) {
            let vertical = vertical_support_at(
                &height_position,
                z_bounds,
                score_points,
                config.range_profile_sample_count,
                config.range_profile_support_ratio,
            );
            (height_position, covariance) =
                apply_vertical_support(height_position, covariance, vertical);
        }
    }
    if !config.range_profile_enabled {
        return ResolvedClusterGeometry {
            position_m: height_position,
            covariance_m2: covariance,
            range_projection_mode: Some(refined.range_projection_mode.clone()),
            range_observability: refined.range_observability,
            height_resolution: height.status,
            range_interval_m: None,
            range_lower_bound_m: None,
        };
    }
    let settings = RangeProfileSettings {
        min_range_m: config.range_profile_min_range_m,
        max_range_m: config.range_profile_max_range_m,
        sample_count: config.range_profile_sample_count,
        cone_half_angle_deg: config.range_profile_cone_half_angle_deg,
        cone_step_deg: config.range_profile_cone_step_deg,
        support_ratio: config.range_profile_support_ratio,
        max_interval_ratio: config.range_observable_max_interval_ratio,
        interval_floor_m: config.range_observable_interval_floor_aperture_fraction
            * layout_aperture_m(sensor_positions),
    };
    let admissible = |point: &Point3| prior.is_admissible(point);
    let support = range_support_along_bearing(
        &height_position,
        &grid.centroid_m,
        score_points,
        &settings,
        if prior.enabled {
            Some(&admissible)
        } else {
            None
        },
    );
    let judged = judge_range_from_profile(
        &height_position,
        covariance,
        Some(&refined.range_projection_mode),
        refined.range_observability,
        &grid.centroid_m,
        support,
    );
    ResolvedClusterGeometry {
        position_m: judged.position_m,
        covariance_m2: judged.covariance_m2,
        range_projection_mode: judged.range_projection_mode,
        range_observability: judged.range_observability,
        height_resolution: height.status,
        range_interval_m: judged.range_interval_m,
        range_lower_bound_m: judged.range_lower_bound_m,
    }
}

fn hold(cluster: &mut LocalizedOnsetCluster, reason: &str) {
    cluster.status = ClusterStatus::Tentative;
    cluster.status_reasons.push(reason.to_string());
}

fn range_interval_or_point(cluster: &LocalizedOnsetCluster, centroid_m: &Point3) -> (f64, f64) {
    cluster.range_interval_m.unwrap_or_else(|| {
        let range = distance(&cluster.position_m, centroid_m);
        (range, range)
    })
}

/// A weaker accepted cluster on a stronger one's bearing with overlapping range
/// intervals (one range unobservable) is the same source split along its range.
pub fn suppress_same_bearing_range_ambiguous_duplicates(
    clusters: &mut [LocalizedOnsetCluster],
    centroid_m: &Point3,
    angle_deg: f64,
) {
    for weaker_index in 1..clusters.len() {
        if clusters[weaker_index].status != ClusterStatus::Accepted {
            continue;
        }
        let weaker = &clusters[weaker_index];
        let mut duplicate = false;
        for stronger in &clusters[..weaker_index] {
            if stronger.status != ClusterStatus::Accepted {
                continue;
            }
            if !(is_unobservable(weaker.range_projection_mode.as_deref())
                || is_unobservable(stronger.range_projection_mode.as_deref()))
            {
                continue;
            }
            let bearing_weaker = subtract(&weaker.position_m, centroid_m);
            let bearing_stronger = subtract(&stronger.position_m, centroid_m);
            let (length_weaker, length_stronger) = (norm(&bearing_weaker), norm(&bearing_stronger));
            if length_weaker * length_stronger <= 0.0 {
                continue;
            }
            let unit = |v: &Point3, l: f64| [v[0] / l, v[1] / l, v[2] / l];
            let separation = angular_separation_deg(
                &unit(&bearing_weaker, length_weaker),
                &unit(&bearing_stronger, length_stronger),
            );
            let (near_w, far_w) = range_interval_or_point(weaker, centroid_m);
            let (near_s, far_s) = range_interval_or_point(stronger, centroid_m);
            if separation <= angle_deg && near_w <= far_s && near_s <= far_w {
                duplicate = true;
                break;
            }
        }
        if duplicate {
            hold(
                &mut clusters[weaker_index],
                SAME_SOURCE_RANGE_AMBIGUOUS_REASON,
            );
        }
    }
}

/// Single-node, range-unobservable secondary clusters need persistent onsets.
pub fn hold_unpersistent_bearing_only_clusters(
    clusters: &mut [LocalizedOnsetCluster],
    min_independent_onsets: usize,
) {
    for cluster in clusters.iter_mut().skip(1) {
        let mut template_nodes = cluster.template_node_ids.clone();
        template_nodes.sort();
        template_nodes.dedup();
        if cluster.status == ClusterStatus::Accepted
            && is_unobservable(cluster.range_projection_mode.as_deref())
            && template_nodes.len() == 1
            && cluster.independent_onset_count < min_independent_onsets
        {
            hold(cluster, BEARING_ONLY_NOT_PERSISTENT_REASON);
        }
    }
}

fn cross_node_tdoas_s(
    position_m: &Point3,
    node_centroids: &[Point3],
    sound_speed_mps: f64,
) -> Vec<f64> {
    let distances: Vec<f64> = node_centroids
        .iter()
        .map(|centroid| distance(position_m, centroid))
        .collect();
    let mut tdoas = Vec::new();
    for first in 0..distances.len() {
        for second in first + 1..distances.len() {
            tdoas.push((distances[first] - distances[second]) / sound_speed_mps);
        }
    }
    tdoas
}

/// Planar layouts pin a TDOA curve; a weaker accepted cluster predicting every
/// cross-node TDOA within `max_tdoa_difference_s` of a stronger one is that source.
pub fn hold_tdoa_equivalent_duplicates(
    clusters: &mut [LocalizedOnsetCluster],
    node_centroids: &[Point3],
    sound_speed_mps: f64,
    max_tdoa_difference_s: f64,
) {
    for index in 1..clusters.len() {
        if clusters[index].status != ClusterStatus::Accepted {
            continue;
        }
        let weaker =
            cross_node_tdoas_s(&clusters[index].position_m, node_centroids, sound_speed_mps);
        let duplicate = clusters[..index]
            .iter()
            .filter(|s| s.status == ClusterStatus::Accepted)
            .any(|stronger| {
                let stronger_tdoas =
                    cross_node_tdoas_s(&stronger.position_m, node_centroids, sound_speed_mps);
                weaker
                    .iter()
                    .zip(&stronger_tdoas)
                    .map(|(a, b)| (a - b).abs())
                    .fold(f64::NEG_INFINITY, f64::max)
                    <= max_tdoa_difference_s
            });
        if duplicate {
            hold(&mut clusters[index], TDOA_EQUIVALENT_DUPLICATE_REASON);
        }
    }
}

/// S-fixes window-level duplicate/persistence holds.
pub fn apply_window_level_holds(
    clusters: &mut [LocalizedOnsetCluster],
    grid: &CandidateGrid,
    node_centroids: &[Point3],
    sound_speed_mps: f64,
    config: &OnsetClusteringConfig,
) {
    suppress_same_bearing_range_ambiguous_duplicates(
        clusters,
        &grid.centroid_m,
        config.cluster_angle_deg,
    );
    if grid.is_direction_grid || node_centroids.len() < 2 {
        return;
    }
    hold_tdoa_equivalent_duplicates(
        clusters,
        node_centroids,
        sound_speed_mps,
        config.same_source_max_tdoa_difference_seconds,
    );
    hold_unpersistent_bearing_only_clusters(clusters, config.persistence_min_independent_onsets);
}
