"""Field Mode S-fixes: window grid with node near-field shells + source prior, and the
per-cluster height / range verdicts applied after LM refinement.

Kept out of ``onset_srp_clustering`` so that module stays an orchestrator. See
``onset_srp_geometry_prior`` (why the prior and the shells), ``onset_height_resolution``
(mirror twins) and ``onset_range_observability`` (range interval).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from minimappr.config import OnsetSrpClusteringConfig
from minimappr.core.onset_cluster_refinement import RefinedClusterPosition, cluster_srp_scores
from minimappr.core.onset_height_resolution import (
    HEIGHT_NOT_APPLICABLE,
    add_vertical_variance,
    apply_vertical_support,
    resolve_node_plane_mirror,
    vertical_support_at,
)
from minimappr.core.onset_range_observability import (
    RangeProfileSettings,
    judge_range_from_profile,
    range_support_along_bearing,
)
from minimappr.core.onset_srp_geometry_prior import (
    NodePlaneGroundPrior,
    build_node_plane_ground_prior,
    node_near_field_points,
)
from minimappr.core.onset_srp_grid import (
    CandidateGrid,
    angular_separation_deg,
    augment_spatial_grid,
    build_spatial_grid,
)
from minimappr.core.onset_srp_types import (
    CLUSTER_STATUS_ACCEPTED,
    CLUSTER_STATUS_TENTATIVE,
    LocalizedOnsetCluster,
    OnsetCluster,
    OnsetFrame,
)
from minimappr.core.range_projection import range_mode_is_unobservable

SAME_SOURCE_RANGE_AMBIGUOUS_REASON = "same_bearing_range_ambiguous"


def cap_intra_node_weight_share(curves: list, node_sensor_ids: set[str], max_share: float) -> list:
    """Rescale a multi-node frame's intra-node pair weights to at most `max_share` of the total.

    ML pair weights grow ~linearly with SNR, so for a source 2 m from one node that node's
    compact pairs (both mics ~20 dB hotter than any cross-node pair) took ~95% of the frame
    weight. Compact pairs are range-blind, so every point on the node's bearing ray scored the
    same and the far end won (S gate: a wren 2.3 m from tetra-c placed 52 m out). Normal frames
    (intra share ~10-20%) are untouched; frames with no cross-node weight stay as they are.
    """
    intra = sum(curve.weight_sum for curve in curves if curve.sensor_b in node_sensor_ids)
    cross = sum(curve.weight_sum for curve in curves if curve.sensor_b not in node_sensor_ids)
    if intra <= 0.0 or cross <= 0.0 or intra / (intra + cross) <= max_share:
        return curves
    scale = max_share * cross / ((1.0 - max_share) * intra)
    return [
        replace(curve, weight_sum=curve.weight_sum * scale) if curve.sensor_b in node_sensor_ids else curve
        for curve in curves
    ]


def window_ground_prior(node_centroids: dict[str, np.ndarray], config: OnsetSrpClusteringConfig) -> NodePlaneGroundPrior:
    return build_node_plane_ground_prior(
        list(node_centroids.values()),
        enabled=config.ground_prior_enabled,
        node_height_above_ground_m=config.node_height_above_ground_m,
        source_max_height_above_ground_m=config.source_max_height_above_ground_m,
        planar_layout_max_flatness_ratio=config.planar_layout_max_flatness_ratio,
    )


def build_window_spatial_grid(
    layout_centroid_m: np.ndarray,
    node_centroids: dict[str, np.ndarray],
    config: OnsetSrpClusteringConfig,
) -> CandidateGrid:
    grid = build_spatial_grid(
        layout_centroid_m,
        azimuth_step_deg=config.srp_azimuth_step_deg,
        elevation_step_deg=config.srp_elevation_step_deg,
        min_range_m=config.srp_grid_min_range_m,
        max_range_m=config.srp_grid_max_range_m,
        ring_count=config.srp_grid_ring_count,
    )
    shells = node_near_field_points(
        list(node_centroids.values()),
        min_range_m=config.srp_node_grid_min_range_m,
        max_range_m=config.srp_node_grid_max_range_m,
        ring_count=config.srp_node_grid_ring_count,
        azimuth_step_deg=config.srp_node_grid_azimuth_step_deg,
        elevation_step_deg=config.srp_node_grid_elevation_step_deg,
    )
    return augment_spatial_grid(grid, extra_points_m=shells)


@dataclass(frozen=True, slots=True)
class ResolvedClusterGeometry:
    position_m: np.ndarray
    covariance_m2: list[list[float]] | None
    range_projection_mode: str | None
    range_observability: float | None
    height_resolution: str
    range_interval_m: tuple[float, float] | None
    range_lower_bound_m: float | None


def _layout_aperture_m(sensor_positions: dict[str, np.ndarray]) -> float:
    points = np.vstack(list(sensor_positions.values()))
    return float(np.max(np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)))


def _range_profile_settings(config: OnsetSrpClusteringConfig, layout_aperture_m: float) -> RangeProfileSettings:
    return RangeProfileSettings(
        min_range_m=config.range_profile_min_range_m,
        max_range_m=config.range_profile_max_range_m,
        sample_count=config.range_profile_sample_count,
        cone_half_angle_deg=config.range_profile_cone_half_angle_deg,
        cone_step_deg=config.range_profile_cone_step_deg,
        support_ratio=config.range_profile_support_ratio,
        max_interval_ratio=config.range_observable_max_interval_ratio,
        interval_floor_m=config.range_observable_interval_floor_aperture_fraction * layout_aperture_m,
    )


def resolve_cluster_geometry(
    refined: RefinedClusterPosition,
    *,
    cluster: OnsetCluster,
    frames: dict[int, OnsetFrame],
    sensor_positions: dict[str, np.ndarray],
    sound_speed_mps: float,
    grid: CandidateGrid,
    prior: NodePlaneGroundPrior,
    config: OnsetSrpClusteringConfig,
) -> ResolvedClusterGeometry:
    """Height twin first (it moves z only), then the range verdict along the bearing."""
    unchanged = ResolvedClusterGeometry(
        refined.position_m, refined.covariance_m2, refined.range_projection_mode, refined.range_observability,
        HEIGHT_NOT_APPLICABLE, None, None,
    )
    if grid.is_direction_grid:
        return unchanged

    def score_points(points_m: np.ndarray) -> np.ndarray:
        return cluster_srp_scores(points_m, cluster, frames, sensor_positions, sound_speed_mps)

    height = resolve_node_plane_mirror(
        refined.position_m, prior=prior, score_points=score_points, min_score_ratio=config.height_mirror_min_score_ratio
    )
    covariance = add_vertical_variance(refined.covariance_m2, height.added_vertical_variance_m2)
    height_position = height.position_m
    if prior.is_planar and prior.enabled:
        z_bounds = prior.vertical_bounds_m(height_position)
        if z_bounds is not None:
            vertical = vertical_support_at(
                height_position, z_bounds_m=z_bounds, score_points=score_points,
                sample_count=config.range_profile_sample_count, support_ratio=config.range_profile_support_ratio,
            )
            height_position, covariance = apply_vertical_support(height_position, covariance, vertical)
    if not config.range_profile_enabled:
        return ResolvedClusterGeometry(
            height_position, covariance, refined.range_projection_mode, refined.range_observability, height.status,
            None, None,
        )
    support = range_support_along_bearing(
        height_position,
        grid.centroid_m,
        score_points=score_points,
        settings=_range_profile_settings(config, _layout_aperture_m(sensor_positions)),
        admissible_mask=prior.admissible_mask if prior.enabled else None,
    )
    judged = judge_range_from_profile(
        position_m=height_position,
        covariance_m2=covariance,
        range_projection_mode=refined.range_projection_mode,
        range_observability=refined.range_observability,
        origin_m=grid.centroid_m,
        support=support,
    )
    return ResolvedClusterGeometry(
        judged.position_m, judged.covariance_m2, judged.range_projection_mode, judged.range_observability,
        height.status, judged.range_interval_m, judged.range_lower_bound_m,
    )


def _range_interval_or_point(cluster: LocalizedOnsetCluster, centroid_m: np.ndarray) -> tuple[float, float]:
    if cluster.range_interval_m is not None:
        return cluster.range_interval_m
    range_m = float(np.linalg.norm(np.asarray(cluster.position_m) - centroid_m))
    return range_m, range_m


def suppress_same_bearing_range_ambiguous_duplicates(
    clusters: list[LocalizedOnsetCluster], *, centroid_m: np.ndarray, angle_deg: float
) -> list[LocalizedOnsetCluster]:
    """A weaker accepted cluster on the same bearing as a stronger one, whose range
    intervals overlap while at least one range is unobservable, is the same source
    split along its (unresolved) range: it becomes tentative, not a second bird.
    `clusters` is strongest first."""
    result = list(clusters)
    for weaker_index in range(1, len(result)):
        weaker = result[weaker_index]
        if weaker.status != CLUSTER_STATUS_ACCEPTED:
            continue
        for stronger in result[:weaker_index]:
            if stronger.status != CLUSTER_STATUS_ACCEPTED:
                continue
            if not (
                range_mode_is_unobservable(weaker.range_projection_mode)
                or range_mode_is_unobservable(stronger.range_projection_mode)
            ):
                continue
            bearing_weaker = np.asarray(weaker.position_m) - centroid_m
            bearing_stronger = np.asarray(stronger.position_m) - centroid_m
            norms = float(np.linalg.norm(bearing_weaker)) * float(np.linalg.norm(bearing_stronger))
            if norms <= 0.0:
                continue
            separation = angular_separation_deg(
                bearing_weaker / np.linalg.norm(bearing_weaker), bearing_stronger / np.linalg.norm(bearing_stronger)
            )
            near_w, far_w = _range_interval_or_point(weaker, centroid_m)
            near_s, far_s = _range_interval_or_point(stronger, centroid_m)
            if separation <= angle_deg and near_w <= far_s and near_s <= far_w:
                result[weaker_index] = _with_status(weaker, CLUSTER_STATUS_TENTATIVE, SAME_SOURCE_RANGE_AMBIGUOUS_REASON)
                break
    return result


def _with_status(cluster: LocalizedOnsetCluster, status: str, reason: str) -> LocalizedOnsetCluster:
    return replace(cluster, status=status, status_reasons=(*cluster.status_reasons, reason))


BEARING_ONLY_NOT_PERSISTENT_REASON = "bearing_only_not_persistent"


def hold_unpersistent_bearing_only_clusters(
    clusters: list[LocalizedOnsetCluster], *, min_independent_onsets: int
) -> list[LocalizedOnsetCluster]:
    """W2 holds a single compact array's secondary clusters to >= `min_independent_onsets`
    independent onsets, because a bearing-only cluster has no level/position cross-check.
    A multi-node cluster whose range proved unobservable and whose onsets all came from one
    node is in the same position (flock scene: one-onset ghosts 40 m out, 25 m up), so it
    gets the same hold. The window's strongest cluster is never held. Strongest first."""
    result = list(clusters)
    for index in range(1, len(result)):
        cluster = result[index]
        if (
            cluster.status == CLUSTER_STATUS_ACCEPTED
            and range_mode_is_unobservable(cluster.range_projection_mode)
            and len(set(cluster.template_node_ids)) == 1
            and cluster.independent_onset_count < min_independent_onsets
        ):
            result[index] = _with_status(cluster, CLUSTER_STATUS_TENTATIVE, BEARING_ONLY_NOT_PERSISTENT_REASON)
    return result


def peak_twin_folds(prior: NodePlaneGroundPrior, grid: CandidateGrid):
    """(position fold, direction fold) for frame peaks: identity unless the layout is planar
    and spatial, where both fold onto the upper side of the node plane (the layout centroid
    lies on that plane, so ranges are unchanged). Without it a bird's peaks split into a
    z > plane and a z < plane cluster (same_band_pair: a 3 m twin duplicate)."""
    if grid.is_direction_grid or not prior.is_planar:
        return (lambda point: point), (lambda direction: direction)
    return prior.fold_above_plane, prior.fold_direction_above_plane


TDOA_EQUIVALENT_DUPLICATE_REASON = "tdoa_equivalent_duplicate"


def _cross_node_tdoas_s(position_m: np.ndarray, node_centroids: list[np.ndarray], sound_speed_mps: float) -> np.ndarray:
    distances = np.asarray([np.linalg.norm(position_m - centroid) for centroid in node_centroids])
    return (distances[:, None] - distances[None, :])[np.triu_indices(len(node_centroids), k=1)] / sound_speed_mps


def hold_tdoa_equivalent_duplicates(
    clusters: list[LocalizedOnsetCluster],
    *,
    node_centroids: dict[str, np.ndarray],
    sound_speed_mps: float,
    max_tdoa_difference_s: float,
) -> list[LocalizedOnsetCluster]:
    """In a planar layout the cross-node TDOAs pin a 3-D *curve*, not a point, so one bird's
    clusters can refine to different points of it (same_band_pair: (24.8, 13.5, 10.3) and
    (21.9, 11.8, 0.0), 1 ms apart in every cross-node TDOA). A weaker accepted cluster whose
    fix predicts every cross-node TDOA within `max_tdoa_difference_s` of a stronger one's is
    that same source. Strongest first."""
    centroids = list(node_centroids.values())
    result = list(clusters)
    for index in range(1, len(result)):
        weaker = result[index]
        if weaker.status != CLUSTER_STATUS_ACCEPTED:
            continue
        weaker_tdoas = _cross_node_tdoas_s(np.asarray(weaker.position_m), centroids, sound_speed_mps)
        for stronger in result[:index]:
            if stronger.status != CLUSTER_STATUS_ACCEPTED:
                continue
            stronger_tdoas = _cross_node_tdoas_s(np.asarray(stronger.position_m), centroids, sound_speed_mps)
            if float(np.max(np.abs(weaker_tdoas - stronger_tdoas))) <= max_tdoa_difference_s:
                result[index] = _with_status(weaker, CLUSTER_STATUS_TENTATIVE, TDOA_EQUIVALENT_DUPLICATE_REASON)
                break
    return result


def apply_window_level_holds(
    clusters: list[LocalizedOnsetCluster],
    *,
    grid: CandidateGrid,
    node_centroids: dict[str, np.ndarray],
    sound_speed_mps: float,
    config: OnsetSrpClusteringConfig,
) -> list[LocalizedOnsetCluster]:
    """Field Mode S-fixes window-level duplicate/persistence holds (multi-node windows only)."""
    clusters = suppress_same_bearing_range_ambiguous_duplicates(
        clusters, centroid_m=grid.centroid_m, angle_deg=config.cluster_angle_deg
    )
    if grid.is_direction_grid or len(node_centroids) < 2:
        return clusters
    clusters = hold_tdoa_equivalent_duplicates(
        clusters, node_centroids=node_centroids, sound_speed_mps=sound_speed_mps,
        max_tdoa_difference_s=config.same_source_max_tdoa_difference_seconds,
    )
    return hold_unpersistent_bearing_only_clusters(clusters, min_independent_onsets=config.persistence_min_independent_onsets)
