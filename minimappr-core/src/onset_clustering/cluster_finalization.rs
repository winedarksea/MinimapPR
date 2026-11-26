//! Per-cluster finalisation: refine, height / range verdicts, level checks
//! and the classification node (port of `_finalize_cluster` and
//! `_classification_node_for_cluster` in `onset_srp_clustering.py`). Split
//! from `pipeline.rs`.

use std::collections::BTreeMap;

use super::attribution_evidence::{build_cluster_attribution_evidence, AttributionEvidenceContext};
use super::config::OnsetClusteringConfig;
use super::echo_checks::{
    level_consistency_holds, near_node_fix_led_only_by_that_node, near_node_level_gradient_holds,
    LevelCheckContext,
};
use super::geometry_prior::NodePlaneGroundPrior;
use super::geometry_resolution::resolve_cluster_geometry;
use super::onset_detection::SubbandHistory;
use super::pipeline::OnsetClusteringInput;
use super::refinement::{refine_cluster_position, RefinementSettings};
use super::srp_grid::{add_scaled, distance, CandidateGrid, Point3};
use super::tdoa_votes::ClusterGeometry;
use super::types::{ClusterStatus, LocalizedOnsetCluster, OnsetCluster, OnsetFrame};
use crate::range_projection::is_unobservable;

/// Everything the per-cluster finalisation reads.
pub(super) struct WindowContext<'window> {
    pub(super) input: &'window OnsetClusteringInput,
    pub(super) config: &'window OnsetClusteringConfig,
    pub(super) frames: BTreeMap<usize, OnsetFrame>,
    pub(super) nodes: BTreeMap<String, Vec<String>>,
    pub(super) node_centroids: BTreeMap<String, Point3>,
    pub(super) histories: BTreeMap<String, SubbandHistory>,
    pub(super) grid: CandidateGrid,
    pub(super) prior: NodePlaneGroundPrior,
    pub(super) frame_samples: usize,
}

pub(super) fn finalize_cluster(
    mut cluster: OnsetCluster,
    context: &WindowContext<'_>,
    is_window_strongest: bool,
) -> LocalizedOnsetCluster {
    let input = context.input;
    let config = context.config;
    let grid = &context.grid;
    let bearing_only = grid.is_direction_grid
        || cluster
            .peaks
            .iter()
            .all(|peak| peak.range_m >= grid.max_range_m);
    let seed = if bearing_only {
        add_scaled(
            &grid.centroid_m,
            &cluster.weighted_direction(),
            input.far_field_default_range_m,
        )
    } else {
        cluster.weighted_position_m()
    };
    let geometry = ClusterGeometry {
        frames: &context.frames,
        sensor_positions: &input.sensor_positions_m,
        sound_speed_mps: input.sound_speed_mps,
    };
    let minimum_std_m = input.sound_speed_mps
        / (input.sample_rate_hz as f64 * input.interpolation_factor.max(1) as f64);
    let refined = refine_cluster_position(
        &cluster,
        &geometry,
        &seed,
        &RefinementSettings {
            centroid_m: grid.centroid_m,
            seed_is_bearing_only: bearing_only,
            sample_rate_hz: input.sample_rate_hz,
            interpolation_factor: input.interpolation_factor,
            far_field_default_range_m: input.far_field_default_range_m,
            minimum_position_std_m: minimum_std_m,
            vertical_bounds_m: if context.prior.is_planar {
                None
            } else {
                context.prior.vertical_bounds_m(&seed)
            },
        },
    );
    let score_points = |points: &[Point3]| geometry.cluster_srp_scores(points, &cluster);
    let resolved = resolve_cluster_geometry(
        &refined,
        &score_points,
        &input.sensor_positions_m,
        grid,
        &context.prior,
        config,
    );
    let range_unobservable =
        bearing_only || is_unobservable(resolved.range_projection_mode.as_deref());
    let led = {
        let led = cluster.led_frame_indices();
        if led.is_empty() {
            cluster.frame_indices()
        } else {
            led
        }
    };
    let led_frames: Vec<&OnsetFrame> = led.iter().map(|index| &context.frames[index]).collect();
    let level_context = LevelCheckContext {
        led_frames: &led_frames,
        node_histories: &context.histories,
        node_positions_m: &context.node_centroids,
        sound_speed_mps: input.sound_speed_mps,
        sample_rate_hz: input.sample_rate_hz,
        frame_samples: context.frame_samples,
    };
    let position = resolved.position_m;
    // Level is a hint, not a gate: the strongest cluster stays accepted (annotated).
    let flag = |cluster: &mut OnsetCluster, failed: bool, reason: &str| {
        if cluster.status == ClusterStatus::Accepted && failed {
            cluster.status_reasons.push(reason.to_string());
            if !is_window_strongest {
                cluster.status = ClusterStatus::Tentative;
            }
        }
    };
    let level_failed = cluster.status == ClusterStatus::Accepted
        && context.nodes.len() > 1
        && !level_consistency_holds(
            &position,
            &level_context,
            config.level_consistency_margin_db,
        );
    flag(&mut cluster, level_failed, "level_inconsistent");
    let near_level_failed = cluster.status == ClusterStatus::Accepted
        && !grid.is_direction_grid
        && !near_node_level_gradient_holds(
            &position,
            &level_context,
            config.srp_node_grid_max_range_m,
            config.near_node_level_tolerance_db,
        );
    flag(
        &mut cluster,
        near_level_failed,
        "near_node_level_inconsistent",
    );
    let single_template_failed = cluster.status == ClusterStatus::Accepted
        && !grid.is_direction_grid
        && near_node_fix_led_only_by_that_node(
            &position,
            &led_frames,
            &context.node_centroids,
            config.srp_node_grid_max_range_m,
        );
    flag(
        &mut cluster,
        single_template_failed,
        "near_node_single_template_node",
    );
    let mut contributing: Vec<String> = led_frames
        .iter()
        .map(|frame| frame.template_node_id.clone())
        .collect();
    contributing.sort();
    contributing.dedup();
    let loudest = classification_node_for_cluster(
        &led_frames,
        &context.node_centroids,
        &position,
        range_unobservable,
    );
    let attribution_evidence = input.attribution_audible_snr_db.map(|audible_snr_db| {
        build_cluster_attribution_evidence(
            &position,
            &led_frames,
            &AttributionEvidenceContext {
                node_histories: &context.histories,
                node_positions_m: &context.node_centroids,
                sound_speed_mps: input.sound_speed_mps,
                sample_rate_hz: input.sample_rate_hz,
                frame_samples: context.frame_samples,
                audible_snr_db,
            },
        )
    });
    let frame_indices = cluster.frame_indices();
    let envelope_flags: Vec<bool> = frame_indices
        .iter()
        .flat_map(|index| {
            context.frames[index]
                .curves
                .iter()
                .map(|curve| curve.used_envelope)
        })
        .collect();
    LocalizedOnsetCluster {
        position_m: position,
        covariance_m2: resolved.covariance_m2,
        confidence: refined.confidence,
        gdop: refined.gdop,
        range_projection_mode: resolved.range_projection_mode,
        range_observability: resolved.range_observability,
        residual_rms_seconds: refined.residual_rms_seconds,
        reference_sensor: refined.reference_sensor,
        tdoa_s: refined.tdoa_s,
        status: cluster.status,
        status_reasons: cluster.status_reasons.clone(),
        onset_samples: led
            .iter()
            .map(|index| context.frames[index].onset_sample)
            .collect(),
        template_node_ids: led
            .iter()
            .map(|index| context.frames[index].template_node_id.clone())
            .collect(),
        band_min_hz: led_frames
            .iter()
            .map(|f| f.band_min_hz)
            .fold(f64::INFINITY, f64::min),
        band_max_hz: led_frames
            .iter()
            .map(|f| f.band_max_hz)
            .fold(f64::NEG_INFINITY, f64::max),
        weight: cluster.weight(),
        frame_count: frame_indices.len(),
        independent_onset_count: cluster.independent_onset_count,
        refined_by_least_squares: refined.refined_by_least_squares,
        used_envelope_fraction: if envelope_flags.is_empty() {
            0.0
        } else {
            envelope_flags.iter().filter(|flag| **flag).count() as f64 / envelope_flags.len() as f64
        },
        contributing_node_ids: contributing,
        loudest_node_id: loudest,
        height_resolution: Some(resolved.height_resolution.to_string()),
        range_interval_m: resolved.range_interval_m,
        range_lower_bound_m: resolved.range_lower_bound_m,
        attribution_evidence,
    }
}

/// Nearest node when range is known, else the most frequent template node.
pub(super) fn classification_node_for_cluster(
    led_frames: &[&OnsetFrame],
    node_centroids: &BTreeMap<String, Point3>,
    position_m: &Point3,
    bearing_only: bool,
) -> String {
    if !bearing_only && node_centroids.len() > 1 {
        let mut best: Option<(&String, f64)> = None;
        for (node, centroid) in node_centroids {
            let range = distance(centroid, position_m);
            if best.is_none_or(|(_, current)| range < current) {
                best = Some((node, range));
            }
        }
        return best.map(|(node, _)| node.clone()).unwrap_or_default();
    }
    let mut counts: Vec<(String, usize)> = Vec::new();
    for frame in led_frames {
        match counts
            .iter_mut()
            .find(|(node, _)| *node == frame.template_node_id)
        {
            Some((_, count)) => *count += 1,
            None => counts.push((frame.template_node_id.clone(), 1)),
        }
    }
    let mut best = counts[0].clone();
    for entry in &counts {
        if entry.1 > best.1 {
            best = entry.clone();
        }
    }
    best.0
}
