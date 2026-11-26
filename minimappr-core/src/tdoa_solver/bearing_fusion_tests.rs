//! Unit tests for `bearing_fusion` (kept in step with
//! `tests/test_multi_node_bearing_fusion.py`).

use super::*;

fn observation(node_id: &str, origin: [f64; 3], toward: [f64; 3]) -> BearingObservation {
    let offset = vector(&toward) - vector(&origin);
    let direction = offset.normalize();
    BearingObservation {
        node_id: node_id.into(),
        origin_m: origin,
        direction: [direction.x, direction.y, direction.z],
        lateral_std_m: 0.5,
        confidence: 0.8,
        event_time_ns: 0,
        range_prior_m: offset.norm(),
        position_std_m: 0.0,
    }
}

#[test]
fn two_crossing_rays_intersect_at_source() {
    let source = [20.0, 30.0, 5.0];
    let fused = fuse_bearings(
        &[
            observation("a", [0.0, 0.0, 0.0], source),
            observation("b", [40.0, 0.0, 0.0], source),
        ],
        &BearingFusionConfig::default(),
    )
    .unwrap();
    assert!((fused.position_m - vector(&source)).norm() < 1e-9);
    // Exact intersection: agreement factor is 1, so only the cap applies.
    assert!((fused.confidence - 0.95).abs() < 1e-12);
}

#[test]
fn partial_agreement_lowers_confidence_below_noisy_or() {
    let fused = fuse_bearings(
        &[
            observation("a", [0.0, 0.0, 0.0], [20.0, 30.0, 5.0]),
            observation("b", [40.0, 0.0, 0.0], [20.0, 30.0, 7.0]),
        ],
        &BearingFusionConfig::default(),
    )
    .unwrap();
    assert!(fused.confidence < 0.95 && fused.confidence > 0.0);
}

#[test]
fn contradictory_rays_are_rejected() {
    let config = BearingFusionConfig {
        max_range_m: 6000.0,
        ..BearingFusionConfig::default()
    };
    let behind = [
        observation("a", [0.0, 0.0, 0.0], [-10.0, 0.0, 0.0]),
        observation("b", [10.0, 10.0, 0.0], [10.0, 20.0, 0.0]),
    ];
    assert_eq!(
        fuse_bearings(&behind, &config),
        Err(BearingFusionRejection::BehindArray)
    );
    let vertically_skewed = [
        observation("a", [0.0, 0.0, 0.0], [20.0, 0.0, 0.0]),
        observation("b", [20.0, -20.0, 30.0], [20.0, 0.0, 30.0]),
    ];
    assert_eq!(
        fuse_bearings(&vertically_skewed, &config),
        Err(BearingFusionRejection::RaysDisagree)
    );
}

#[test]
fn parallel_and_single_node_inputs_are_rejected() {
    let far = [0.0, 1000.0, 0.0];
    let rays = [
        observation("a", [0.0, 0.0, 0.0], far),
        observation("b", [1.0, 0.0, 0.0], far),
    ];
    assert_eq!(
        fuse_bearings(&rays, &BearingFusionConfig::default()),
        Err(BearingFusionRejection::NearParallel)
    );
    let same_node = [
        observation("a", [0.0, 0.0, 0.0], far),
        observation("a", [9.0, 0.0, 0.0], [5.0, 5.0, 0.0]),
    ];
    assert_eq!(
        fuse_bearings(&same_node, &BearingFusionConfig::default()),
        Err(BearingFusionRejection::InsufficientNodes)
    );
}

fn wide_config() -> BearingFusionConfig {
    BearingFusionConfig {
        max_range_m: 6000.0,
        ..BearingFusionConfig::default()
    }
}

fn cone(node_id: &str, origin: [f64; 3], toward: [f64; 3], confidence: f64) -> BearingObservation {
    BearingObservation {
        lateral_std_m: 1.0,
        confidence,
        ..observation(node_id, origin, toward)
    }
}

#[test]
fn one_node_hearing_another_bird_does_not_veto_a_consistent_pair() {
    let (bird_one, bird_two) = ([100.0, 100.0, 5.0], [-60.0, 140.0, 5.0]);
    let pair = [
        cone("a", [0.0, 0.0, 1.5], bird_one, 0.8),
        cone("b", [60.0, 0.0, 1.5], bird_one, 0.8),
    ];
    assert!(fuse_bearings(&pair, &wide_config()).is_ok());
    let mut with_outlier = pair.to_vec();
    with_outlier.push(cone("c", [120.0, 20.0, 1.5], bird_two, 0.8));
    let fused = fuse_bearings(&with_outlier, &wide_config()).unwrap();
    assert_eq!(fused.contributor_node_ids, vec!["a", "b"]);
    assert!((fused.position_m - vector(&bird_one)).norm() < 1.0);
}

#[test]
fn per_node_cone_is_chosen_by_agreement_not_confidence() {
    let (bird_one, bird_two) = ([100.0, 100.0, 5.0], [150.0, -80.0, 5.0]);
    let fused = fuse_bearings(
        &[
            cone("a", [0.0, 0.0, 1.5], bird_one, 0.8),
            cone("b", [60.0, 0.0, 1.5], bird_two, 0.9),
            cone("b", [60.0, 0.0, 1.5], bird_one, 0.4),
            cone("c", [-40.0, 60.0, 1.5], bird_one, 0.8),
        ],
        &wide_config(),
    )
    .unwrap();
    assert_eq!(fused.contributor_node_ids, vec!["a", "b", "c"]);
    assert!((fused.position_m - vector(&bird_one)).norm() < 1.0);
}

#[test]
fn zero_width_cone_with_zero_floor_is_degenerate() {
    let source = [20.0, 30.0, 5.0];
    let zero_width = |node_id: &str, origin: [f64; 3]| BearingObservation {
        lateral_std_m: 0.0,
        ..observation(node_id, origin, source)
    };
    let config = BearingFusionConfig {
        min_lateral_std_m: 0.0,
        ..BearingFusionConfig::default()
    };
    assert_eq!(
        fuse_bearings(
            &[zero_width("a", [0.0, 0.0, 0.0]), zero_width("b", [40.0, 0.0, 0.0])],
            &config
        ),
        Err(BearingFusionRejection::Degenerate)
    );
}

#[test]
fn only_the_angular_part_scales_with_range() {
    let mut stated = observation("a", [0.0, 0.0, 0.0], [50.0, 0.0, 0.0]);
    stated.lateral_std_m = 1.0;
    stated.position_std_m = 3.0;
    let at_one_km = lateral_std_at_ray_parameter_m(&stated, 1000.0, 1.0).unwrap();
    assert!((at_one_km - 20.0_f64.hypot(3.0)).abs() < 1e-12);
    let at_prior = lateral_std_at_ray_parameter_m(&stated, 50.0, 1.0).unwrap();
    assert!((at_prior - 1.0_f64.hypot(3.0)).abs() < 1e-12);
}
