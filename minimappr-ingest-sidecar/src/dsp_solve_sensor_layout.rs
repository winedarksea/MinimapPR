//! The ordered sensor set one solve runs on, snapshotted when the manifest is
//! ingested so a parked or queued solve keeps the geometry it triggered with.
//!
//! Index `i` of `sensors` / `positions_m` is buffer channel `i`; SRP, render
//! and the published provenance all index this one list.

use super::*;
use minimappr_core::localization_sensor_entry::{
    CoordinateFrameKind, GeodeticOrigin, NodeTimeQuality, SensorPositionSource, SolveBufferScope,
    SolveCoordinateFrame, SolveSensorEntry,
};

/// Timebase of every packet timestamp the sidecar decodes today.
const PACKET_CLOCK_DOMAIN_UTC: &str = "utc";
/// Sensor ids are `<node_id>:ch<channel_index>` on both sides of the wire.
const SENSOR_ID_CHANNEL_SEPARATOR: &str = ":ch";

#[derive(Clone, Debug)]
#[expect(
    dead_code,
    reason = "provenance fields are emitted by milestone 1 (R6)"
)]
pub(crate) struct SolveSensorLayout {
    pub(crate) sensors: Vec<SolveSensorEntry>,
    /// f32 copy of `sensors[].position_m`, the form SRP and render consume.
    pub(crate) positions_m: Vec<[f32; 3]>,
    /// Milestone 1 (R1 registry) assigns real revisions; until then every
    /// layout reports revision 0.
    pub(crate) layout_revision: u64,
    pub(crate) coordinate_frame: SolveCoordinateFrame,
    pub(crate) buffer_scope: SolveBufferScope,
    pub(crate) cluster_id: Option<String>,
    pub(crate) reporting_node_id: String,
    reporting_node_half_space: crate::srp_phat::HalfSpace,
}

impl SolveSensorLayout {
    pub(crate) fn sensor_count(&self) -> usize {
        self.positions_m.len()
    }

    /// Coplanar up/down constraint for this solve. Today the reporting node's
    /// declaration applies to the whole buffer (R1 restricts it to solves whose
    /// active sensors all belong to that node).
    pub(crate) fn solve_half_space(&self) -> crate::srp_phat::HalfSpace {
        self.reporting_node_half_space
    }
}

/// Layout for one ingested manifest, from the same sources the solve used
/// before the seam existed: explicit/automatic cluster positions first, then
/// the node's own offsets, then the tetra reference geometry.
pub(crate) fn solve_sensor_layout_from_manifest(
    manifest: &DspManifest,
    reporting_node_id: &str,
    sample_rate_hz: u32,
    automatic_cluster_origin: Option<GeodeticOrigin>,
) -> SolveSensorLayout {
    let positions_m = mic_positions_from_manifest(manifest);
    let cluster_positions = manifest
        .cluster_sensor_positions
        .as_ref()
        .filter(|positions| !positions.is_empty());
    let is_automatic_cluster = manifest.cluster_id.as_deref() == Some(AUTOMATIC_GPS_CLUSTER_ID);
    let (buffer_scope, position_source, coordinate_frame) = match cluster_positions {
        Some(_) if is_automatic_cluster => (
            SolveBufferScope::AutomaticSiteCluster,
            SensorPositionSource::ReportedGps,
            SolveCoordinateFrame {
                frame_id: None,
                kind: CoordinateFrameKind::SidecarAutomaticEnu,
                origin: automatic_cluster_origin,
                // The legacy spherical projection is not a contract projection.
                projection: None,
            },
        ),
        Some(_) => (
            SolveBufferScope::ExplicitCluster,
            SensorPositionSource::ExplicitClusterManifest,
            frame_without_geodetic_anchor(CoordinateFrameKind::ExplicitClusterPositions),
        ),
        None => (
            SolveBufferScope::NodeLocal,
            SensorPositionSource::NodeBodyOffset,
            frame_without_geodetic_anchor(CoordinateFrameKind::NodeBody),
        ),
    };
    let sensor_ids: Vec<String> = match cluster_positions {
        Some(positions) => positions.iter().map(|(id, _)| id.clone()).collect(),
        None => (0..positions_m.len())
            .map(|channel_index| {
                format!("{reporting_node_id}{SENSOR_ID_CHANNEL_SEPARATOR}{channel_index}")
            })
            .collect(),
    };
    let sensors = sensor_ids
        .into_iter()
        .zip(&positions_m)
        .map(|(sensor_id, position_m)| {
            let (node_id, node_channel_index) = split_sensor_id(&sensor_id);
            SolveSensorEntry {
                node_id,
                node_channel_index,
                sensor_id,
                position_m: position_m.map(f64::from),
                sample_rate_hz,
                position_source,
                position_std_m: None,
                clock_source: NodeTimeQuality::Unknown,
                clock_domain: PACKET_CLOCK_DOMAIN_UTC.to_string(),
                clock_offset_std_s: None,
            }
        })
        .collect();
    SolveSensorLayout {
        sensors,
        positions_m,
        layout_revision: 0,
        coordinate_frame,
        buffer_scope,
        cluster_id: manifest.cluster_id.clone(),
        reporting_node_id: reporting_node_id.to_string(),
        reporting_node_half_space: half_space_from_node_context(&manifest.node_context),
    }
}

fn frame_without_geodetic_anchor(kind: CoordinateFrameKind) -> SolveCoordinateFrame {
    SolveCoordinateFrame {
        frame_id: None,
        kind,
        origin: None,
        projection: None,
    }
}

/// `node-a:ch3` -> (`node-a`, 3). An id without the channel suffix is a
/// single-sensor node: (id, 0).
fn split_sensor_id(sensor_id: &str) -> (String, usize) {
    sensor_id
        .rsplit_once(SENSOR_ID_CHANNEL_SEPARATOR)
        .and_then(|(node_id, channel)| Some((node_id.to_string(), channel.parse().ok()?)))
        .unwrap_or_else(|| (sensor_id.to_string(), 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensor_ids_split_into_node_and_channel() {
        assert_eq!(split_sensor_id("planar-a:ch4"), ("planar-a".to_string(), 4));
        assert_eq!(split_sensor_id("odd:id"), ("odd:id".to_string(), 0));
    }
}
