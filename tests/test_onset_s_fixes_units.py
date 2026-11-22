"""Unit tests for the Field Mode S-fixes onset modules (docs/field_mode_progress/s_fixes_onset.md)."""

from __future__ import annotations

import numpy as np

from minimappr.core.onset_classification_node import classification_capable_cluster
from minimappr.core.onset_cluster_geometry_resolution import (
    SAME_SOURCE_RANGE_AMBIGUOUS_REASON,
    TDOA_EQUIVALENT_DUPLICATE_REASON,
    cap_intra_node_weight_share,
    hold_tdoa_equivalent_duplicates,
    suppress_same_bearing_range_ambiguous_duplicates,
)
from minimappr.core.onset_frame_correlation import PairCorrelationCurve
from minimappr.core.onset_height_resolution import (
    HEIGHT_AMBIGUOUS,
    HEIGHT_NOT_APPLICABLE,
    HEIGHT_RESOLVED_BY_PRIOR,
    HEIGHT_RESOLVED_BY_SRP,
    resolve_node_plane_mirror,
)
from minimappr.core.onset_range_observability import (
    RangeProfileSettings,
    judge_range_from_profile,
    range_support_along_bearing,
)
from minimappr.core.onset_srp_geometry_prior import build_node_plane_ground_prior, node_near_field_points
from minimappr.core.onset_srp_types import CLUSTER_STATUS_ACCEPTED, CLUSTER_STATUS_TENTATIVE, LocalizedOnsetCluster
from minimappr.core.range_projection import RANGE_BEARING_PROJECTED, RANGE_REFINED

PLANAR_NODES = [np.array([0.0, 0.0, 1.5]), np.array([20.0, 0.0, 1.5]), np.array([10.0, 18.0, 1.5])]
PROFILE = RangeProfileSettings(
    min_range_m=0.5, max_range_m=1000.0, sample_count=96, cone_half_angle_deg=0.0, cone_step_deg=1.0,
    support_ratio=0.9, max_interval_ratio=2.0, interval_floor_m=6.0,
)


def _prior(nodes=PLANAR_NODES, enabled=True):
    return build_node_plane_ground_prior(
        nodes, enabled=enabled, node_height_above_ground_m=1.5, source_max_height_above_ground_m=50.0,
        planar_layout_max_flatness_ratio=0.1,
    )


def _curve(sensor_b: str, weight: float) -> PairCorrelationCurve:
    lags = np.linspace(-1e-3, 1e-3, 5)
    return PairCorrelationCurve("near:ch0", sensor_b, lags, np.zeros(5), 1000.0, False, weight)


def test_intra_node_weight_share_is_capped_only_when_it_dominates() -> None:
    node = {"near:ch0", "near:ch1", "near:ch2"}
    dominated = [_curve("near:ch1", 95.0), _curve("near:ch2", 95.0), _curve("far:ch0", 5.0), _curve("far:ch1", 5.0)]
    capped = cap_intra_node_weight_share(dominated, node, 0.35)
    intra = sum(c.weight_sum for c in capped if c.sensor_b in node)
    assert np.isclose(intra / sum(c.weight_sum for c in capped), 0.35)
    assert capped[2].weight_sum == 5.0
    normal = [_curve("near:ch1", 1.0), _curve("far:ch0", 5.0)]
    assert cap_intra_node_weight_share(normal, node, 0.35) == normal
    single_node = [_curve("near:ch1", 1.0)]
    assert cap_intra_node_weight_share(single_node, node, 0.35) == single_node


def test_node_plane_prior_geometry() -> None:
    prior = _prior()
    assert prior.is_planar
    assert np.allclose(prior.plane_normal, [0, 0, 1])
    assert np.allclose(prior.height_above_ground_m(np.array([5.0, 5.0, 0.0])), [0.0])
    assert prior.admissible_mask(np.array([[5, 5, 4.0], [5, 5, -1.0], [5, 5, 60.0]])).tolist() == [True, False, False]
    assert np.allclose(prior.mirror_across_node_plane(np.array([3.0, 4.0, 4.0])), [3.0, 4.0, -1.0])
    assert np.allclose(prior.vertical_bounds_m(np.array([3.0, 4.0, 9.0])), (0.0, 50.0))
    tetrahedral_layout = [*PLANAR_NODES, np.array([10.0, 6.0, 12.0])]
    assert not _prior(tetrahedral_layout).is_planar
    assert _prior(enabled=False).admissible_mask(np.array([[0, 0, -100.0]])).all()


def test_node_near_field_shells_surround_every_node() -> None:
    points = node_near_field_points(
        PLANAR_NODES[:2], min_range_m=0.5, max_range_m=8.0, ring_count=3, azimuth_step_deg=90.0,
        elevation_step_deg=90.0,
    )
    distances = np.min(np.linalg.norm(points[:, None, :] - np.vstack(PLANAR_NODES[:2])[None], axis=2), axis=1)
    assert np.isclose(distances.min(), 0.5) and np.isclose(distances.max(), 8.0)
    assert node_near_field_points([], min_range_m=1, max_range_m=2, ring_count=2, azimuth_step_deg=10,
                                  elevation_step_deg=10).shape == (0, 3)


def test_mirror_twin_below_ground_is_folded_up_by_the_prior() -> None:
    resolution = resolve_node_plane_mirror(
        np.array([4.0, 6.0, -2.0]), prior=_prior(), score_points=lambda p: np.ones(len(p)), min_score_ratio=1.05
    )
    assert resolution.status == HEIGHT_RESOLVED_BY_PRIOR
    assert np.allclose(resolution.position_m, [4.0, 6.0, 5.0])


def test_mirror_twins_both_plausible_use_srp_or_report_uncertain_height() -> None:
    position = np.array([4.0, 6.0, 2.5])  # mirror at 0.5 m: both above ground
    by_srp = resolve_node_plane_mirror(
        position, prior=_prior(), score_points=lambda p: np.where(p[:, 2] > 1.5, 2.0, 1.0), min_score_ratio=1.05
    )
    assert by_srp.status == HEIGHT_RESOLVED_BY_SRP and by_srp.position_m[2] == 2.5
    tie = resolve_node_plane_mirror(position, prior=_prior(), score_points=lambda p: np.ones(len(p)), min_score_ratio=1.05)
    assert tie.status == HEIGHT_AMBIGUOUS
    assert np.isclose(tie.position_m[2], 1.5) and np.isclose(tie.added_vertical_variance_m2, 1.0)
    non_planar = _prior([*PLANAR_NODES, np.array([10.0, 6.0, 12.0])])
    assert resolve_node_plane_mirror(position, prior=non_planar, score_points=len, min_score_ratio=1.05).status == HEIGHT_NOT_APPLICABLE


def _ridge_score(true_range_m: float, flat_beyond: bool):
    origin = np.zeros(3)

    def score(points: np.ndarray) -> np.ndarray:
        ranges = np.linalg.norm(points - origin, axis=1)
        if flat_beyond:
            return np.clip(ranges / true_range_m, 0.0, 1.0)  # rises, then flat: far field
        return np.exp(-0.5 * ((ranges - true_range_m) / 1.0) ** 2)  # sharp curvature peak

    return score


def test_flat_far_profile_becomes_bearing_projected_with_honest_radial_spread() -> None:
    origin = np.zeros(3)
    claimed = np.array([391.0, 0.0, 0.0])  # LM's confident far range
    support = range_support_along_bearing(claimed, origin, score_points=_ridge_score(70.0, True), settings=PROFILE)
    assert not support.observable and support.interval_m[1] == 1000.0
    judged = judge_range_from_profile(
        position_m=claimed, covariance_m2=np.eye(3).tolist(), range_projection_mode=RANGE_REFINED,
        range_observability=0.9, origin_m=origin, support=support,
    )
    assert judged.range_projection_mode == RANGE_BEARING_PROJECTED
    assert np.isclose(np.linalg.norm(judged.position_m), support.peak_range_m)
    assert support.interval_m[0] <= support.peak_range_m <= support.interval_m[1]
    radial_std = np.sqrt(np.asarray(judged.covariance_m2)[0][0])
    assert radial_std > 300.0


def test_peaked_profile_keeps_range_and_pulls_runaway_lm_back() -> None:
    origin = np.zeros(3)
    support = range_support_along_bearing(np.array([12.0, 0, 0]), origin, score_points=_ridge_score(12.0, False), settings=PROFILE)
    assert support.observable
    judged = judge_range_from_profile(
        position_m=np.array([40.0, 0, 0]), covariance_m2=(np.eye(3) * 0.01).tolist(),
        range_projection_mode=RANGE_REFINED, range_observability=1.0, origin_m=origin, support=support,
    )
    assert judged.range_projection_mode == RANGE_REFINED
    assert abs(np.linalg.norm(judged.position_m) - 12.0) < 1.0


def _cluster(position, mode, interval=None, status=CLUSTER_STATUS_ACCEPTED, node="n1"):
    return LocalizedOnsetCluster(
        position_m=tuple(position), covariance_m2=None, confidence=0.5, gdop=1.0, range_projection_mode=mode,
        range_observability=None, residual_rms_seconds=None, reference_sensor="s", tdoa_s={}, status=status,
        status_reasons=(), onset_samples=(0,), template_node_ids=(node,), band_min_hz=1.0, band_max_hz=2.0,
        weight=1.0, frame_count=1, independent_onset_count=1, refined_by_least_squares=True,
        used_envelope_fraction=0.0, contributing_node_ids=(node,), loudest_node_id=node, range_interval_m=interval,
    )


def test_same_bearing_split_with_unobservable_range_is_held_back() -> None:
    strong = _cluster((50.0, 0, 0), RANGE_BEARING_PROJECTED, (30.0, 1000.0))
    weak = _cluster((300.0, 5.0, 0), RANGE_REFINED)
    elsewhere = _cluster((0.0, 50.0, 0), RANGE_REFINED)
    result = suppress_same_bearing_range_ambiguous_duplicates([strong, weak, elsewhere], centroid_m=np.zeros(3), angle_deg=10.0)
    assert result[1].status == CLUSTER_STATUS_TENTATIVE and SAME_SOURCE_RANGE_AMBIGUOUS_REASON in result[1].status_reasons
    assert result[0].status == result[2].status == CLUSTER_STATUS_ACCEPTED


def test_single_mic_classification_node_is_replaced_by_nearest_array() -> None:
    node_ids = {"phone:ch0": "phone", **{f"tetra:ch{i}": "tetra" for i in range(4)}, **{f"far:ch{i}": "far" for i in range(4)}}
    positions = {"phone:ch0": np.array([10.0, 18.0, 1.5])}
    positions.update({f"tetra:ch{i}": np.array([0.0, 0.0, 1.5]) + 0.01 * i for i in range(4)})
    positions.update({f"far:ch{i}": np.array([100.0, 0.0, 1.5]) + 0.01 * i for i in range(4)})
    cluster = _cluster((11.0, 20.0, 2.0), RANGE_REFINED, node="phone")
    assert classification_capable_cluster(cluster, node_ids, positions, 2).loudest_node_id == "tetra"
    assert classification_capable_cluster(cluster, node_ids, positions, 1).loudest_node_id == "phone"
    assert classification_capable_cluster(cluster, {"phone:ch0": "phone"}, positions, 2).loudest_node_id == "phone"


def test_planar_ridge_twins_with_equal_cross_node_tdoas_are_one_source() -> None:
    nodes = {"a": PLANAR_NODES[0], "b": PLANAR_NODES[1], "c": PLANAR_NODES[2]}
    strong = _cluster((22.0, 12.0, 3.0), RANGE_REFINED)
    twin = _cluster((22.0, 12.0, 0.0), RANGE_REFINED)  # node-plane mirror: identical TDOAs
    other_bird = _cluster((2.0, 8.0, 3.0), RANGE_REFINED)
    result = hold_tdoa_equivalent_duplicates(
        [strong, twin, other_bird], node_centroids=nodes, sound_speed_mps=343.0, max_tdoa_difference_s=0.0015
    )
    assert result[1].status == CLUSTER_STATUS_TENTATIVE and TDOA_EQUIVALENT_DUPLICATE_REASON in result[1].status_reasons
    assert result[2].status == CLUSTER_STATUS_ACCEPTED
