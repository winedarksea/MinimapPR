use std::collections::BTreeMap;

use minimappr_core::localization_sensor_entry::{NodePlacement, NodePlacementSource};
use minimappr_core::site_frame_projection::{geodetic_to_local_enu_m, local_enu_to_geodetic};

use super::*;

const SITE_ORIGIN: GeodeticOrigin = GeodeticOrigin {
    lat: 45.0,
    lon: -93.0,
    alt_m: 250.0,
};

fn gps_fix_at_site_enu(position_enu_m: [f64; 3]) -> ReportedNodeFix {
    let point = local_enu_to_geodetic(
        &SITE_ORIGIN,
        CoordinateProjection::FlatWgs84Series,
        position_enu_m,
    );
    ReportedNodeFix {
        lat_deg: point.lat,
        lon_deg: point.lon,
        alt_m: point.alt_m,
        position_source: SensorPositionSource::ReportedGps,
        reported_accuracy_m: Some(3.0),
        reported_body_to_world_quaternion_wxyz: None,
    }
}

fn push_with_placements(
    revision: u64,
    projection: CoordinateProjection,
    placements: &[(&str, [f64; 3], NodePlacementSource)],
) -> SiteGeometryPush {
    SiteGeometryPush {
        site_frame_id: "site-7".to_string(),
        geometry_push_revision: revision,
        site_origin: SITE_ORIGIN,
        projection,
        node_placements: placements
            .iter()
            .map(|(node_id, position_m, placement_source)| {
                (
                    node_id.to_string(),
                    NodePlacement {
                        position_m: Some(*position_m),
                        position_std_m: Some(0.05),
                        placement_source: *placement_source,
                        body_to_world_quaternion_wxyz: None,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>(),
    }
}

fn placed_position_m(resolution: &NodePlacementResolution) -> [f64; 3] {
    match resolution {
        NodePlacementResolution::Placed { position_m, .. } => *position_m,
        other => panic!("expected a placement, got {other:?}"),
    }
}

fn assert_near(actual: [f64; 3], expected: [f64; 3], tolerance_m: f64) {
    for axis in 0..3 {
        assert!(
            (actual[axis] - expected[axis]).abs() < tolerance_m,
            "{actual:?} vs {expected:?}"
        );
    }
}

#[test]
fn automatic_frame_adopts_the_first_trusted_gps_fix_with_the_flat_projection() {
    let mut state = SiteGeometryState::default();
    assert_eq!(state.coordinate_frame().origin, None);
    assert_eq!(state.coordinate_frame().projection, None);

    let fallback = ReportedNodeFix {
        position_source: SensorPositionSource::FallbackStatic,
        ..gps_fix_at_site_enu([500.0, 0.0, 0.0])
    };
    assert_eq!(
        state.resolve_node_placement("static-node", Some(&fallback)),
        NodePlacementResolution::Untrusted {
            reason: SolveExclusionReason::NodePositionUntrustedNoOverride
        },
        "a fallback position never becomes the origin"
    );
    assert_eq!(state.coordinate_frame().origin, None);

    let first = state.resolve_node_placement("a", Some(&gps_fix_at_site_enu([0.0, 0.0, 0.0])));
    assert_near(placed_position_m(&first), [0.0; 3], 1e-9);
    let second = state.resolve_node_placement("b", Some(&gps_fix_at_site_enu([20.0, 7.0, 1.0])));
    assert_near(placed_position_m(&second), [20.0, 7.0, 1.0], 1e-6);

    let frame = state.coordinate_frame();
    assert_eq!(frame.kind, CoordinateFrameKind::SidecarAutomaticEnu);
    assert_eq!(frame.frame_id, None);
    assert_eq!(
        frame.projection,
        Some(CoordinateProjection::FlatWgs84Series)
    );
    let origin = frame.origin.expect("origin adopted");
    assert!((origin.lat - SITE_ORIGIN.lat).abs() < 1e-12);
    assert_eq!(state.geometry_push_revision(), None);
    assert!(matches!(
        second,
        NodePlacementResolution::Placed {
            position_source: SensorPositionSource::ReportedGps,
            position_std_m: Some(std_m),
            ..
        } if std_m == 3.0
    ));
}

#[test]
fn pushed_placement_beats_gps_and_gps_fills_unplaced_nodes_in_the_pushed_frame() {
    let push = push_with_placements(
        3,
        CoordinateProjection::GeodeticEcefEnu,
        &[("a", [1.0, 2.0, 1.5], NodePlacementSource::OperatorOverride)],
    );
    let mut state = SiteGeometryState::from_push(push).unwrap();

    let gps_far_from_override = gps_fix_at_site_enu([300.0, 300.0, 0.0]);
    assert_eq!(
        state.resolve_node_placement("a", Some(&gps_far_from_override)),
        NodePlacementResolution::Placed {
            position_m: [1.0, 2.0, 1.5],
            position_std_m: Some(0.05),
            position_source: SensorPositionSource::OperatorOverride,
            body_to_world_quaternion_wxyz: None,
        }
    );

    let gps_node = gps_fix_at_site_enu([40.0, -10.0, 0.0]);
    let expected = geodetic_to_local_enu_m(
        &SITE_ORIGIN,
        CoordinateProjection::GeodeticEcefEnu,
        gps_node.lat_deg,
        gps_node.lon_deg,
        gps_node.alt_m,
    );
    assert_eq!(
        placed_position_m(&state.resolve_node_placement("b", Some(&gps_node))),
        expected,
        "unplaced node uses raw GPS projected with the pushed projection"
    );
    assert_eq!(
        state.resolve_node_placement("c", None),
        NodePlacementResolution::Untrusted {
            reason: SolveExclusionReason::NodePositionUntrustedNoOverride
        }
    );

    let frame = state.coordinate_frame();
    assert_eq!(frame.kind, CoordinateFrameKind::SiteEnu);
    assert_eq!(frame.frame_id.as_deref(), Some("site-7"));
    assert_eq!(frame.origin, Some(SITE_ORIGIN));
    assert_eq!(
        frame.projection,
        Some(CoordinateProjection::GeodeticEcefEnu)
    );
    assert_eq!(state.geometry_push_revision(), Some(3));
}

#[test]
fn operator_placement_admits_a_fallback_static_node() {
    let push = push_with_placements(
        1,
        CoordinateProjection::FlatWgs84Series,
        &[(
            "bench",
            [4.0, 0.0, 1.0],
            NodePlacementSource::OperatorOverride,
        )],
    );
    let mut state = SiteGeometryState::from_push(push).unwrap();
    let fallback = ReportedNodeFix {
        position_source: SensorPositionSource::FallbackStatic,
        ..gps_fix_at_site_enu([6.0, 0.0, 2.0])
    };
    assert_eq!(
        placed_position_m(&state.resolve_node_placement("bench", Some(&fallback))),
        [4.0, 0.0, 1.0]
    );
    assert!(matches!(
        state.resolve_node_placement("other", Some(&fallback)),
        NodePlacementResolution::Untrusted { .. }
    ));
}

#[test]
fn stale_revision_is_refused_and_equal_revision_replaces() {
    let mut state = SiteGeometryState::default();
    let first = push_with_placements(
        5,
        CoordinateProjection::FlatWgs84Series,
        &[(
            "a",
            [1.0, 0.0, 0.0],
            NodePlacementSource::StationaryEstimate,
        )],
    );
    let acknowledgement = state.apply_push(first).unwrap();
    assert_eq!(acknowledgement.applied_geometry_push_revision, 5);
    assert_eq!(acknowledgement.node_placement_count, 1);

    let stale = push_with_placements(4, CoordinateProjection::FlatWgs84Series, &[]);
    assert_eq!(
        state.apply_push(stale),
        Err(SiteGeometryError::StaleGeometryPushRevision {
            pushed_revision: 4,
            applied_revision: 5
        })
    );
    assert_eq!(state.acknowledgement().unwrap().node_placement_count, 1);

    // Full snapshot: node a is absent from the re-push, so it falls back to GPS.
    let same_revision = push_with_placements(5, CoordinateProjection::FlatWgs84Series, &[]);
    state.apply_push(same_revision).unwrap();
    assert!(matches!(
        state.resolve_node_placement("a", Some(&gps_fix_at_site_enu([9.0, 0.0, 0.0]))),
        NodePlacementResolution::Placed {
            position_source: SensorPositionSource::ReportedGps,
            ..
        }
    ));
}

#[test]
fn malformed_pushes_are_rejected() {
    let valid = push_with_placements(
        1,
        CoordinateProjection::FlatWgs84Series,
        &[("a", [0.0; 3], NodePlacementSource::OperatorOverride)],
    );
    let mut empty_frame = valid.clone();
    empty_frame.site_frame_id = " ".to_string();
    assert_eq!(
        SiteGeometryState::from_push(empty_frame).unwrap_err(),
        SiteGeometryError::EmptySiteFrameId
    );

    let mut bad_origin = valid.clone();
    bad_origin.site_origin.lat = 91.0;
    assert!(matches!(
        SiteGeometryState::from_push(bad_origin),
        Err(SiteGeometryError::InvalidSiteOrigin { .. })
    ));

    let mutations: [fn(&mut NodePlacement); 5] = [
        |placement| placement.position_m = Some([0.0, f64::NAN, 0.0]),
        |placement| placement.position_std_m = Some(-0.1),
        |placement| placement.body_to_world_quaternion_wxyz = Some([0.0; 4]),
        // Neither a position nor an orientation: nothing is placed.
        |placement| {
            placement.position_m = None;
            placement.position_std_m = None;
        },
        // An accuracy with no position to describe.
        |placement| {
            placement.position_m = None;
            placement.body_to_world_quaternion_wxyz = Some([1.0, 0.0, 0.0, 0.0]);
        },
    ];
    for mutate in mutations {
        let mut push = valid.clone();
        mutate(push.node_placements.get_mut("a").unwrap());
        assert!(matches!(
            SiteGeometryState::from_push(push),
            Err(SiteGeometryError::InvalidNodePlacement { .. })
        ));
    }
}

#[test]
fn node_context_parsing_reads_source_and_orientation() {
    let node = serde_json::json!({
        "id": "n1",
        "position_geo": {"lat": 45.001, "lon": -93.0, "alt_m": 251.0},
        "orientation": {"yaw_deg": 90.0, "pitch_deg": 0.0, "roll_deg": 0.0},
        "metadata": {"gps": {"position_source": "fallback_static"}}
    });
    let fix = reported_node_fix_from_node_context(&node).unwrap();
    assert_eq!(fix.position_source, SensorPositionSource::FallbackStatic);
    let quaternion = fix.reported_body_to_world_quaternion_wxyz.unwrap();
    assert!((quaternion[3] - std::f64::consts::FRAC_PI_4.sin()).abs() < 1e-15);

    let legacy =
        serde_json::json!({"position_geo": {"lat": 45.0, "lon": -93.0}, "orientation": null});
    let fix = reported_node_fix_from_node_context(&legacy).unwrap();
    assert_eq!(fix.position_source, SensorPositionSource::ReportedGps);
    assert_eq!(fix.alt_m, 0.0);
    assert_eq!(fix.reported_body_to_world_quaternion_wxyz, None);
    assert!(reported_node_fix_from_node_context(&serde_json::json!({"id": "x"})).is_none());
}

#[test]
fn orientation_only_placement_keeps_gps_position_and_pins_the_rotation() {
    const PINNED_YAW_QUATERNION_WXYZ: [f64; 4] = [0.9537, 0.0, 0.0, 0.3007];
    const REPORTED_YAW_QUATERNION_WXYZ: [f64; 4] = [0.0, 0.0, 0.0, 1.0];
    let mut push = push_with_placements(2, CoordinateProjection::FlatWgs84Series, &[]);
    push.node_placements.insert(
        "planar".to_string(),
        NodePlacement {
            position_m: None,
            position_std_m: None,
            placement_source: NodePlacementSource::OperatorOverride,
            body_to_world_quaternion_wxyz: Some(PINNED_YAW_QUATERNION_WXYZ),
        },
    );
    let mut state = SiteGeometryState::from_push(push).unwrap();
    let gps = ReportedNodeFix {
        reported_body_to_world_quaternion_wxyz: Some(REPORTED_YAW_QUATERNION_WXYZ),
        ..gps_fix_at_site_enu([37.0, -22.0, 1.5])
    };
    let NodePlacementResolution::Placed {
        position_m,
        position_std_m,
        position_source,
        body_to_world_quaternion_wxyz,
    } = state.resolve_node_placement("planar", Some(&gps))
    else {
        panic!("a trusted GPS fix places an orientation-only node");
    };
    assert_near(position_m, [37.0, -22.0, 1.5], 1e-6);
    assert_eq!(position_std_m, Some(3.0), "GPS accuracy, not a pushed one");
    assert_eq!(position_source, SensorPositionSource::ReportedGps);
    assert_eq!(
        body_to_world_quaternion_wxyz,
        Some(PINNED_YAW_QUATERNION_WXYZ),
        "the pushed orientation beats the reported one"
    );

    let fallback = ReportedNodeFix {
        position_source: SensorPositionSource::FallbackStatic,
        ..gps
    };
    assert_eq!(
        state.resolve_node_placement("planar", Some(&fallback)),
        NodePlacementResolution::Untrusted {
            reason: SolveExclusionReason::NodePositionUntrustedNoOverride
        },
        "an orientation does not make an untrusted position trusted"
    );
}
