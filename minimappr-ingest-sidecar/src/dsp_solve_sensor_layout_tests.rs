use super::*;
use minimappr_core::localization_sensor_entry::{
    CoordinateFrameKind, NodeTimeQuality, SensorPositionSource,
};

const RATE_HZ: u32 = 16_000;

fn sensor(node_id: &str, channel: usize, sample_rate_hz: u32) -> SolveSensorEntry {
    SolveSensorEntry {
        sensor_id: sensor_id_for_node_channel(node_id, channel),
        node_id: node_id.to_string(),
        node_channel_index: channel,
        position_m: [channel as f64, 0.0, 0.0],
        sample_rate_hz,
        position_source: SensorPositionSource::ReportedGps,
        position_std_m: None,
        clock_source: NodeTimeQuality::Unknown,
        clock_domain: "utc".to_string(),
        clock_offset_std_s: None,
    }
}

/// planar-a ch0..1 (declares `upper`) + point-b ch0 at `point_rate_hz`.
fn two_node_layout(point_rate_hz: u32) -> SolveSensorLayout {
    SolveSensorLayout::from_parts(SolveSensorLayoutParts {
        sensors: vec![
            sensor("planar-a", 0, RATE_HZ),
            sensor("planar-a", 1, RATE_HZ),
            sensor("point-b", 0, point_rate_hz),
        ],
        layout_revision: 3,
        geometry_push_revision: None,
        coordinate_frame: SolveCoordinateFrame {
            frame_id: None,
            kind: CoordinateFrameKind::SidecarAutomaticEnu,
            origin: None,
            projection: None,
        },
        buffer_scope: SolveBufferScope::AutomaticSiteCluster,
        cluster_id: Some("auto-gps".to_string()),
        reporting_node_id: "planar-a".to_string(),
        node_local_routing_reason: None,
        solve_sample_rate_hz: RATE_HZ,
        half_space_by_node: BTreeMap::from([("planar-a".to_string(), HalfSpace::Upper)]),
        geometry_unusable_node_id: None,
        max_solve_sensors: 16,
    })
}

#[test]
fn sensor_ids_split_into_node_and_channel() {
    assert_eq!(split_sensor_id("planar-a:ch4"), ("planar-a".to_string(), 4));
    assert_eq!(split_sensor_id("odd:id"), ("odd:id".to_string(), 0));
    assert!(sensor_order_key("n:ch2") < sensor_order_key("n:ch10"));
}

#[test]
fn half_space_applies_only_when_every_active_sensor_is_on_the_declaring_node() {
    let layout = two_node_layout(RATE_HZ);
    assert_eq!(layout.solve_half_space(&[0, 1]), HalfSpace::Upper);
    assert_eq!(layout.solve_half_space(&[0, 1, 2]), HalfSpace::None);
    assert_eq!(layout.solve_half_space(&[2]), HalfSpace::None);
    assert_eq!(layout.solve_half_space(&[]), HalfSpace::None);
}

#[test]
fn cross_node_pairs_are_identified_by_sensor_node() {
    let layout = two_node_layout(RATE_HZ);
    assert!(!layout.is_cross_node_pair(0, 1));
    assert!(layout.is_cross_node_pair(1, 2));
}

#[test]
fn rate_mismatched_channel_is_cleared_before_solve_and_render() {
    let layout = two_node_layout(48_000);
    let mut windows = vec![vec![1.0_f32; 4], vec![1.0; 4], vec![1.0; 12]];
    layout.clear_solve_excluded_channels(&mut windows);
    assert_eq!(windows[2], Vec::<f32>::new());
    assert_eq!(windows[0].len(), 4);
}
