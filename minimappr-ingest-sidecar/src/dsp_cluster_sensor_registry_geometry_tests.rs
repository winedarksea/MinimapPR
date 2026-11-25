//! Registry routing under the site geometry: a push re-places every
//! automatic-cluster member (rotation included) and an untrusted node solves
//! node-local.

use super::dsp_cluster_sensor_registry_tests::*;
use super::*;
use minimappr_core::localization_sensor_entry::{
    CoordinateProjection, NodePlacement, NodePlacementSource, SiteGeometryPush,
};
use minimappr_core::sirith_array_geometry::SIRITH_PLANAR_MIC_OFFSETS_M;

fn site_push(revision: u64, placements: &[(&str, NodePlacement)]) -> SiteGeometryPush {
    SiteGeometryPush {
        site_frame_id: "registry-site".to_string(),
        geometry_push_revision: revision,
        site_origin: SITE_ORIGIN,
        projection: CoordinateProjection::FlatWgs84Series,
        node_placements: placements
            .iter()
            .map(|(node_id, placement)| (node_id.to_string(), placement.clone()))
            .collect(),
    }
}

#[test]
fn geometry_push_replaces_every_member_in_the_site_frame_and_bumps_the_revision() {
    const PUSHED_POINT_POSITION_M: [f64; 3] = [7.0, 1.0, 2.0];
    // +90 degree yaw about +Z, (w, x, y, z).
    let quarter_turn_wxyz = [
        std::f64::consts::FRAC_PI_4.cos(),
        0.0,
        0.0,
        std::f64::consts::FRAC_PI_4.sin(),
    ];
    let mut registry = registry();
    route_point(&mut registry, "point-b", 3.0, RATE_HZ, SECOND_NS);
    let before_push = route_planar(&mut registry, 0.0, SECOND_NS);
    assert_eq!(
        before_push.layout.coordinate_frame.kind,
        CoordinateFrameKind::SidecarAutomaticEnu
    );

    registry
        .site_geometry
        .apply_push(site_push(
            4,
            &[
                (
                    "point-b",
                    NodePlacement {
                        position_m: Some(PUSHED_POINT_POSITION_M),
                        position_std_m: Some(0.05),
                        placement_source: NodePlacementSource::OperatorOverride,
                        body_to_world_quaternion_wxyz: None,
                    },
                ),
                (
                    "planar-a",
                    NodePlacement {
                        position_m: None,
                        position_std_m: None,
                        placement_source: NodePlacementSource::OperatorOverride,
                        body_to_world_quaternion_wxyz: Some(quarter_turn_wxyz),
                    },
                ),
            ],
        ))
        .unwrap();
    // Only the planar node reports; the point member moves with the push anyway.
    let after_push = route_planar(&mut registry, 0.0, 2 * SECOND_NS).layout;
    assert!(after_push.layout_revision > before_push.layout.layout_revision);
    assert_eq!(
        after_push.coordinate_frame.kind,
        CoordinateFrameKind::SiteEnu
    );
    assert_eq!(
        after_push.coordinate_frame.frame_id.as_deref(),
        Some("registry-site")
    );
    assert_eq!(after_push.geometry_push_revision, Some(4));
    let point = &after_push.sensors[5];
    assert_eq!(point.position_m, PUSHED_POINT_POSITION_M);
    assert_eq!(
        point.position_source,
        SensorPositionSource::OperatorOverride
    );
    // Planar: GPS position (site origin), offsets rotated by the pushed yaw.
    let corner_body = SIRITH_PLANAR_MIC_OFFSETS_M[0].map(f64::from);
    let corner_world = after_push.sensors[0].position_m;
    assert!(
        (corner_world[0] + corner_body[1]).abs() < 1e-6,
        "{corner_world:?}"
    );
    assert!(
        (corner_world[1] - corner_body[0]).abs() < 1e-6,
        "{corner_world:?}"
    );
    assert_eq!(
        after_push.sensors[0].position_source,
        SensorPositionSource::ReportedGps
    );
}

fn fallback_point(north_m: f64) -> serde_json::Value {
    let mut node = gps_node(&[[0.0; 3]], north_m);
    node["metadata"] = serde_json::json!({"gps": {"position_source": "gps_fallback"}});
    node
}

fn route_point_with(
    registry: &mut RegistryUnderTest,
    node: Option<serde_json::Value>,
    now_ns: u128,
) -> SensorRoute {
    route(registry, "point-b", &manifest(node), 1, RATE_HZ, now_ns).route
}

#[test]
fn never_trusted_fallback_node_solves_node_local() {
    let mut registry = registry();
    route_planar(&mut registry, 0.0, SECOND_NS);
    let untrusted = route_point_with(&mut registry, Some(fallback_point(3.0)), SECOND_NS);
    assert_eq!(untrusted.layout.buffer_scope, SolveBufferScope::NodeLocal);
    assert_eq!(
        untrusted.layout.node_local_routing_reason,
        Some(NodeLocalRoutingReason::NodePositionUntrustedNoOverride)
    );
}

#[test]
fn member_flapping_to_untrusted_or_no_fix_keeps_its_place_for_the_stale_window() {
    let stale_after_ns = (STALE_SECONDS as u128) * SECOND_NS;
    let mut registry = registry();
    route_planar(&mut registry, 0.0, SECOND_NS);
    let joined = route_point(&mut registry, "point-b", 3.0, RATE_HZ, 2 * SECOND_NS);
    assert_eq!(ids(&joined), PLANAR_AND_POINT_IDS);

    // gps_fallback on one packet, no position_geo on the next, then trusted
    // again: the shared buffer never changes, so it is never reordered.
    let no_fix_point = serde_json::json!({"sensor_offsets_m": [[0.0, 0.0, 0.0]]});
    for (step, node) in [Some(fallback_point(3.0)), Some(no_fix_point)]
        .into_iter()
        .enumerate()
    {
        let held = route_point_with(&mut registry, node, (3 + step as u128) * SECOND_NS);
        assert_eq!(
            held.layout.buffer_scope,
            SolveBufferScope::AutomaticSiteCluster
        );
        assert_eq!(ids(&held), PLANAR_AND_POINT_IDS);
        assert_eq!(held.layout.layout_revision, joined.layout.layout_revision);
        assert_eq!(held.layout.positions_m, joined.layout.positions_m);
    }
    let planar_while_held = route_planar(&mut registry, 0.0, 5 * SECOND_NS);
    assert_eq!(ids(&planar_while_held), PLANAR_AND_POINT_IDS);
    assert_eq!(
        planar_while_held.layout.layout_revision,
        joined.layout.layout_revision
    );
    let trusted_again = route_point(&mut registry, "point-b", 3.0, RATE_HZ, 6 * SECOND_NS);
    assert_eq!(
        trusted_again.layout.layout_revision,
        joined.layout.layout_revision
    );

    // Untrusted beyond the stale window after its last trusted fix: it leaves.
    let last_trusted_ns = 6 * SECOND_NS;
    let expired_ns = last_trusted_ns + stale_after_ns + SECOND_NS;
    route_planar(&mut registry, 0.0, expired_ns - 2 * SECOND_NS);
    let demoted = route_point_with(&mut registry, Some(fallback_point(3.0)), expired_ns);
    assert_eq!(demoted.layout.buffer_scope, SolveBufferScope::NodeLocal);
    let planar_after = route_planar(&mut registry, 0.0, expired_ns);
    assert_eq!(ids(&planar_after), &PLANAR_AND_POINT_IDS[..5]);
    assert!(planar_after.layout.layout_revision > joined.layout.layout_revision);
}

#[test]
fn member_that_routes_elsewhere_leaves_the_automatic_cluster_at_once() {
    let mut registry = registry();
    route_planar(&mut registry, 0.0, SECOND_NS);
    let joined = route_point(&mut registry, "point-b", 3.0, RATE_HZ, SECOND_NS);
    assert_eq!(ids(&joined), PLANAR_AND_POINT_IDS);
    // A cluster id without positions is an incomplete explicit declaration.
    let mut incomplete_explicit = manifest(Some(gps_node(&[[0.0; 3]], 3.0)));
    incomplete_explicit.cluster_id = Some("operator-cluster".to_string());
    let elsewhere = route(
        &mut registry,
        "point-b",
        &incomplete_explicit,
        1,
        RATE_HZ,
        2 * SECOND_NS,
    )
    .route;
    assert_eq!(elsewhere.layout.buffer_scope, SolveBufferScope::NodeLocal);
    let planar_after = route_planar(&mut registry, 0.0, 2 * SECOND_NS);
    assert_eq!(ids(&planar_after), &PLANAR_AND_POINT_IDS[..5]);
}
