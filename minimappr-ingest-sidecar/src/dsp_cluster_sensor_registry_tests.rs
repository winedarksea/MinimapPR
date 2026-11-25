//! Registry unit tests: routing, deterministic order, eviction, revisions,
//! the solve-sensor cap, the rate guard and buffer reconcile plans.

use super::*;
use crate::dsp_geometry_config::SiteGeometryState;
use crate::localization_solve_config::LocalizationSolveConfig;
use minimappr_core::localization_sensor_entry::{
    CoordinateProjection, GeodeticOrigin, SolveExclusionReason,
};
use minimappr_core::sirith_array_geometry::SIRITH_PLANAR_MIC_OFFSETS_M;
use minimappr_core::site_frame_projection::local_enu_to_geodetic;

pub(super) const SECOND_NS: u128 = 1_000_000_000;
pub(super) const STALE_SECONDS: f64 = 120.0;
pub(super) const RATE_HZ: u32 = 16_000;
pub(super) const SITE_ORIGIN: GeodeticOrigin = GeodeticOrigin {
    lat: 45.0,
    lon: -93.0,
    alt_m: 0.0,
};

/// The registry plus the site geometry the worker would hand it.
pub(super) struct RegistryUnderTest {
    pub(super) registry: ClusterSensorRegistry,
    pub(super) site_geometry: SiteGeometryState,
}

impl RegistryUnderTest {
    fn tracked_sensor_count(&self) -> usize {
        self.registry.tracked_sensor_count()
    }

    fn evict_stale_sensors(&mut self, now_ns: u128) {
        self.registry.evict_stale_sensors(now_ns);
    }
}

fn registry_with_cap(max_solve_sensors: usize) -> RegistryUnderTest {
    RegistryUnderTest {
        registry: ClusterSensorRegistry::new(&LocalizationSolveConfig {
            localization_max_solve_sensors: max_solve_sensors,
            localization_sensor_registry_stale_seconds: STALE_SECONDS,
            ..LocalizationSolveConfig::default()
        }),
        site_geometry: SiteGeometryState::default(),
    }
}

pub(super) fn registry() -> RegistryUnderTest {
    registry_with_cap(LocalizationSolveConfig::default().localization_max_solve_sensors)
}

pub(super) fn manifest(node: Option<serde_json::Value>) -> DspManifest {
    let mut manifest: DspManifest = serde_json::from_value(serde_json::json!({
        "manifest_id": "m", "manifest_type": "raw_journal_append", "created_ns": 0,
        "source_handles": [], "promotion_ready": false
    }))
    .unwrap();
    manifest.node_context = node.map(|node| serde_json::json!({ "node": node }));
    manifest
}

/// Node JSON with a GPS fix `north_m` north of the site point.
pub(super) fn gps_node(offsets: &[[f32; 3]], north_m: f64) -> serde_json::Value {
    let fix = local_enu_to_geodetic(
        &SITE_ORIGIN,
        CoordinateProjection::FlatWgs84Series,
        [0.0, north_m, 0.0],
    );
    serde_json::json!({
        "sensor_offsets_m": offsets,
        "position_geo": {"lat": fix.lat, "lon": fix.lon, "alt_m": fix.alt_m}
    })
}

fn planar_gps_node(north_m: f64) -> serde_json::Value {
    let mut node = gps_node(&SIRITH_PLANAR_MIC_OFFSETS_M, north_m);
    node["half_space"] = "upper".into();
    node
}

pub(super) fn route(
    registry: &mut RegistryUnderTest,
    node_id: &str,
    manifest: &DspManifest,
    channel_count: usize,
    sample_rate_hz: u32,
    now_ns: u128,
) -> NodeRoutingOutcome {
    let stream_key = format!("{node_id}__audio_main__0001");
    registry.registry.route_node(
        &NodeRoutingRequest {
            stream_key: &stream_key,
            manifest,
            channel_count,
            sample_rate_hz,
            now_ns,
        },
        &mut registry.site_geometry,
    )
}

pub(super) fn route_planar(
    registry: &mut RegistryUnderTest,
    north_m: f64,
    now_ns: u128,
) -> SensorRoute {
    let manifest = manifest(Some(planar_gps_node(north_m)));
    route(registry, "planar-a", &manifest, 5, RATE_HZ, now_ns).route
}

pub(super) fn route_point(
    registry: &mut RegistryUnderTest,
    node_id: &str,
    north_m: f64,
    sample_rate_hz: u32,
    now_ns: u128,
) -> SensorRoute {
    let manifest = manifest(Some(gps_node(&[[0.0; 3]], north_m)));
    route(registry, node_id, &manifest, 1, sample_rate_hz, now_ns).route
}

pub(super) fn ids(route: &SensorRoute) -> Vec<&str> {
    route.buffer_sensor_ids.iter().map(String::as_str).collect()
}

pub(super) const PLANAR_AND_POINT_IDS: [&str; 6] = [
    "planar-a:ch0",
    "planar-a:ch1",
    "planar-a:ch2",
    "planar-a:ch3",
    "planar-a:ch4",
    "point-b:ch0",
];

#[test]
fn planar_node_joins_the_automatic_cluster_validated_on_its_own_offsets() {
    let mut registry = registry();
    route_point(&mut registry, "point-b", 3.0, RATE_HZ, SECOND_NS);
    let manifest = manifest(Some(planar_gps_node(0.0)));
    let outcome = route(&mut registry, "planar-a", &manifest, 5, RATE_HZ, SECOND_NS);
    assert_eq!(outcome.large_array_omni_fallback_reason, None);
    let route = outcome.route;
    assert_eq!(route.buffer_key, "cluster::auto-gps::audio_main");
    assert_eq!(
        route.layout.buffer_scope,
        SolveBufferScope::AutomaticSiteCluster
    );
    assert_eq!(route.layout.node_local_routing_reason, None);
    assert_eq!(ids(&route), PLANAR_AND_POINT_IDS);
    assert_eq!(route.buffer_channel_indices, [0, 1, 2, 3, 4]);
    // Point origin was adopted first, so the planar node sits 3 m south.
    let planar_ch4 = route.layout.sensors[4].position_m;
    assert!((planar_ch4[1] + 3.0).abs() < 1e-3, "{planar_ch4:?}");
}

#[test]
fn sensor_order_is_the_same_for_both_arrival_orders() {
    let mut planar_first = registry();
    route_planar(&mut planar_first, 0.0, SECOND_NS);
    let planar_first_route = route_point(&mut planar_first, "point-b", 3.0, RATE_HZ, SECOND_NS);

    let mut point_first = registry();
    route_point(&mut point_first, "point-b", 3.0, RATE_HZ, SECOND_NS);
    let point_first_route = route_planar(&mut point_first, 0.0, SECOND_NS);

    assert_eq!(ids(&planar_first_route), PLANAR_AND_POINT_IDS);
    assert_eq!(ids(&point_first_route), PLANAR_AND_POINT_IDS);
    assert_eq!(planar_first_route.buffer_channel_indices, [5]);
    assert_eq!(point_first_route.buffer_channel_indices, [0, 1, 2, 3, 4]);
}

#[test]
fn explicit_cluster_order_ignores_arrival_and_listing_order() {
    let declared = |reverse: bool| {
        let mut positions: Vec<(String, [f32; 3])> = ["tetra:ch0", "tetra:ch1", "tetra:ch2"]
            .iter()
            .chain(&["tetra:ch3", "point:ch0"])
            .enumerate()
            .map(|(index, id)| (id.to_string(), [index as f32, 0.0, 1.0]))
            .collect();
        if reverse {
            positions.reverse();
        }
        let mut manifest = manifest(None);
        manifest.cluster_id = Some("c".to_string());
        manifest.cluster_sensor_positions = Some(positions);
        manifest
    };
    let mut tetra_first = registry();
    route(
        &mut tetra_first,
        "tetra",
        &declared(false),
        4,
        RATE_HZ,
        SECOND_NS,
    );
    let tetra_first_route = route(
        &mut tetra_first,
        "point",
        &declared(false),
        1,
        RATE_HZ,
        SECOND_NS,
    )
    .route;
    let mut point_first = registry();
    route(
        &mut point_first,
        "point",
        &declared(true),
        1,
        RATE_HZ,
        SECOND_NS,
    );
    let point_first_route = route(
        &mut point_first,
        "tetra",
        &declared(true),
        4,
        RATE_HZ,
        SECOND_NS,
    )
    .route;

    let expected = [
        "point:ch0",
        "tetra:ch0",
        "tetra:ch1",
        "tetra:ch2",
        "tetra:ch3",
    ];
    assert_eq!(ids(&tetra_first_route), expected);
    assert_eq!(ids(&point_first_route), expected);
    assert_eq!(point_first_route.buffer_channel_indices, [1, 2, 3, 4]);
    assert_eq!(
        tetra_first_route.layout.buffer_scope,
        SolveBufferScope::ExplicitCluster
    );
    assert_eq!(
        tetra_first_route.layout.sensors[0].position_m,
        [4.0, 0.0, 1.0]
    );
}

#[test]
fn node_without_a_position_solves_node_local_with_its_own_offsets_and_a_reason() {
    let mut registry = registry();
    let mut node = serde_json::json!({ "sensor_offsets_m": SIRITH_PLANAR_MIC_OFFSETS_M });
    node["half_space"] = "upper".into();
    let route = route(
        &mut registry,
        "planar-a",
        &manifest(Some(node)),
        5,
        RATE_HZ,
        SECOND_NS,
    )
    .route;
    assert_eq!(route.buffer_key, "planar-a__audio_main__0001");
    assert_eq!(route.layout.buffer_scope, SolveBufferScope::NodeLocal);
    assert_eq!(
        route.layout.node_local_routing_reason,
        Some(NodeLocalRoutingReason::NodePositionUntrustedNoOverride)
    );
    let expected_offsets: Vec<[f32; 3]> = SIRITH_PLANAR_MIC_OFFSETS_M.to_vec();
    assert_eq!(route.layout.positions_m, expected_offsets);
    assert!(route.layout.excluded_sensors.is_empty());
    assert_eq!(
        route.layout.solve_half_space(&[0, 1, 2, 3, 4]),
        crate::srp_phat::HalfSpace::Upper
    );
}

#[test]
fn explicit_cluster_that_omits_a_node_channel_routes_that_node_locally() {
    let mut registry = registry();
    let mut manifest = manifest(None);
    manifest.cluster_id = Some("c".to_string());
    manifest.cluster_sensor_positions = Some(
        ["a:ch0", "b:ch0", "c:ch0", "tetra:ch0"]
            .iter()
            .map(|id| (id.to_string(), [0.0, 0.0, 0.0]))
            .collect(),
    );
    let route = route(&mut registry, "tetra", &manifest, 4, RATE_HZ, SECOND_NS).route;
    assert_eq!(route.layout.buffer_scope, SolveBufferScope::NodeLocal);
    assert_eq!(
        route.layout.node_local_routing_reason,
        Some(NodeLocalRoutingReason::NodeChannelsMissingFromExplicitCluster)
    );
    assert_eq!(route.layout.positions_m, SIRITH_MIC_POSITIONS_M.to_vec());
    assert_eq!(
        NodeLocalRoutingReason::NodeChannelsMissingFromExplicitCluster.as_str(),
        "node_channels_missing_from_explicit_cluster"
    );
}

#[test]
fn unusable_large_array_takes_the_omni_path_and_is_excluded_from_any_solve() {
    let mut registry = registry();
    let outcome = route(
        &mut registry,
        "wide",
        &manifest(None),
        6,
        RATE_HZ,
        SECOND_NS,
    );
    assert_eq!(
        outcome.large_array_omni_fallback_reason,
        Some(LARGE_ARRAY_REASON_GEOMETRY_UNUSABLE)
    );
    let layout = &outcome.route.layout;
    assert_eq!(layout.sensor_count(), 6);
    assert!(layout
        .excluded_sensors
        .iter()
        .all(|excluded| excluded.reason == SolveExclusionReason::NodeArrayGeometryUnusable));
    assert_eq!(layout.excluded_sensors.len(), 6);
    assert_eq!(
        registry.tracked_sensor_count(),
        0,
        "omni path retains nothing"
    );
}

#[test]
fn layout_revision_bumps_on_join_and_gps_update_but_not_on_a_repeat() {
    let mut registry = registry();
    let first = route_planar(&mut registry, 0.0, SECOND_NS)
        .layout
        .layout_revision;
    let repeat = route_planar(&mut registry, 0.0, 2 * SECOND_NS)
        .layout
        .layout_revision;
    assert_eq!(first, repeat, "an unchanged report keeps the revision");
    let joined = route_point(&mut registry, "point-b", 3.0, RATE_HZ, 2 * SECOND_NS)
        .layout
        .layout_revision;
    assert!(joined > repeat);
    let moved = route_point(&mut registry, "point-b", 3.4, RATE_HZ, 3 * SECOND_NS)
        .layout
        .layout_revision;
    assert!(moved > joined, "a GPS update is a position change");
}

#[test]
fn stale_sensors_are_evicted_so_the_registry_stays_bounded() {
    let mut registry = registry();
    let stale_after_ns = (STALE_SECONDS as u128) * SECOND_NS;
    route_point(&mut registry, "point-b", 3.0, RATE_HZ, SECOND_NS);
    let before_eviction = route_planar(&mut registry, 0.0, SECOND_NS);
    assert_eq!(ids(&before_eviction), PLANAR_AND_POINT_IDS);

    let later_ns = SECOND_NS + stale_after_ns + SECOND_NS;
    let after_eviction = route_planar(&mut registry, 0.0, later_ns);
    assert_eq!(ids(&after_eviction), &PLANAR_AND_POINT_IDS[..5]);
    assert!(after_eviction.layout.layout_revision > before_eviction.layout.layout_revision);
    assert_eq!(registry.tracked_sensor_count(), 5);

    registry.evict_stale_sensors(later_ns + stale_after_ns + SECOND_NS);
    assert_eq!(registry.tracked_sensor_count(), 0, "empty sets are dropped");
    let rejoined = route_planar(&mut registry, 0.0, later_ns + 2 * stale_after_ns);
    assert!(
        rejoined.layout.layout_revision > after_eviction.layout.layout_revision,
        "a rebuilt set never reuses a revision"
    );
}

#[test]
fn mismatched_rate_sensors_are_excluded_from_the_reporters_solve() {
    let mut registry = registry();
    const POINT_RATE_HZ: u32 = 48_000;
    route_point(&mut registry, "point-b", 3.0, POINT_RATE_HZ, SECOND_NS);
    let planar_reports = route_planar(&mut registry, 0.0, SECOND_NS).layout;
    assert_eq!(planar_reports.solve_sample_rate_hz, RATE_HZ);
    assert_eq!(planar_reports.sensors[5].sample_rate_hz, POINT_RATE_HZ);
    assert_eq!(planar_reports.excluded_sensors.len(), 1);
    assert_eq!(planar_reports.excluded_sensors[0].sensor_index, 5);
    assert_eq!(
        planar_reports.excluded_sensors[0].reason,
        SolveExclusionReason::SampleRateMismatch
    );

    let point_reports = route_point(&mut registry, "point-b", 3.0, POINT_RATE_HZ, SECOND_NS).layout;
    let excluded: Vec<usize> = point_reports
        .excluded_sensors
        .iter()
        .map(|excluded| excluded.sensor_index)
        .collect();
    assert_eq!(
        excluded,
        [0, 1, 2, 3, 4],
        "the planar node is the mismatch now"
    );
}

#[test]
fn solve_sensor_cap_keeps_the_reporting_node_and_then_sensor_order() {
    let mut registry = registry_with_cap(6);
    for (node_id, north_m) in [("point-b", 3.0), ("point-c", 6.0)] {
        route_point(&mut registry, node_id, north_m, RATE_HZ, SECOND_NS);
    }
    let layout = route_planar(&mut registry, 0.0, SECOND_NS).layout;
    assert_eq!(layout.sensor_count(), 7);
    assert_eq!(layout.excluded_sensors.len(), 1);
    assert_eq!(layout.excluded_sensors[0].sensor_id, "point-c:ch0");
    assert_eq!(
        layout.excluded_sensors[0].reason,
        SolveExclusionReason::MaxSolveSensorsExceeded
    );

    let point_c_reports = route_point(&mut registry, "point-c", 6.0, RATE_HZ, SECOND_NS).layout;
    let excluded: Vec<&str> = point_c_reports
        .excluded_sensors
        .iter()
        .map(|excluded| excluded.sensor_id.as_str())
        .collect();
    assert_eq!(
        excluded,
        ["point-b:ch0"],
        "the reporter's own sensor is kept"
    );
}

#[test]
fn buffer_reconcile_appends_in_place_and_reorders_only_when_channels_move() {
    let owned = |ids: &[&str]| ids.iter().map(|id| id.to_string()).collect::<Vec<_>>();
    assert_eq!(
        plan_buffer_channel_reconcile(&owned(&["a:ch0"]), &owned(&["a:ch0"])),
        BufferChannelReconcile::Unchanged
    );
    assert_eq!(
        plan_buffer_channel_reconcile(&owned(&["a:ch0"]), &owned(&["a:ch0", "b:ch0"])),
        BufferChannelReconcile::AppendChannels
    );
    assert_eq!(
        plan_buffer_channel_reconcile(&owned(&["b:ch0"]), &owned(&["a:ch0", "b:ch0"])),
        BufferChannelReconcile::Reorder {
            source_channel_by_new_channel: vec![None, Some(0)]
        }
    );
    assert_eq!(
        plan_buffer_channel_reconcile(
            &owned(&["a:ch0", "b:ch0", "c:ch0"]),
            &owned(&["a:ch0", "c:ch0"])
        ),
        BufferChannelReconcile::Reorder {
            source_channel_by_new_channel: vec![Some(0), Some(2)]
        }
    );
}
