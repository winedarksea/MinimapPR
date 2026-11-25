//! Test-only `SolveSensorLayout` builder: tetra-like nodes placed in metres.

use std::collections::BTreeMap;

use super::*;
use minimappr_core::localization_sensor_entry::{
    CoordinateFrameKind, NodeTimeQuality, SensorPositionSource, SolveBufferScope,
    SolveCoordinateFrame, SolveSensorEntry,
};

pub(crate) const FIXTURE_SAMPLE_RATE_HZ: u32 = 16_000;
const FIXTURE_CHANNELS_PER_NODE: usize = 4;
const FIXTURE_MIC_SPACING_M: f64 = 0.05;

/// One 4-channel node per `(node_id, centre)`, channels ordered node by node.
/// More than one node makes an automatic site cluster.
pub(crate) fn tetra_nodes_layout(nodes: &[(&str, [f64; 3])]) -> SolveSensorLayout {
    let sensors = nodes
        .iter()
        .flat_map(|(node_id, centre)| {
            (0..FIXTURE_CHANNELS_PER_NODE).map(move |channel| SolveSensorEntry {
                sensor_id: sensor_id_for_node_channel(node_id, channel),
                node_id: node_id.to_string(),
                node_channel_index: channel,
                position_m: [
                    centre[0] + channel as f64 * FIXTURE_MIC_SPACING_M,
                    centre[1],
                    centre[2],
                ],
                sample_rate_hz: FIXTURE_SAMPLE_RATE_HZ,
                position_source: SensorPositionSource::ReportedGps,
                position_std_m: None,
                clock_source: NodeTimeQuality::Unknown,
                clock_domain: "utc".to_string(),
                clock_offset_std_s: None,
            })
        })
        .collect();
    let is_cluster = nodes.len() > 1;
    SolveSensorLayout::from_parts(SolveSensorLayoutParts {
        sensors,
        layout_revision: 1,
        geometry_push_revision: None,
        coordinate_frame: SolveCoordinateFrame {
            frame_id: None,
            kind: CoordinateFrameKind::SidecarAutomaticEnu,
            origin: None,
            projection: None,
        },
        buffer_scope: if is_cluster {
            SolveBufferScope::AutomaticSiteCluster
        } else {
            SolveBufferScope::NodeLocal
        },
        cluster_id: is_cluster.then(|| AUTOMATIC_GPS_CLUSTER_ID.to_string()),
        reporting_node_id: nodes[0].0.to_string(),
        node_local_routing_reason: None,
        solve_sample_rate_hz: FIXTURE_SAMPLE_RATE_HZ,
        half_space_by_node: BTreeMap::new(),
        geometry_unusable_node_id: None,
        max_solve_sensors: 16,
    })
}
