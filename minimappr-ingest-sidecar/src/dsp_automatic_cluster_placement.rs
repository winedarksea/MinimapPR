//! World placement of automatic-cluster members (`auto-gps` buffers).
//!
//! Each member's sensors sit at node position + R(body->world) x body offset,
//! with the node position and rotation from
//! [`SiteGeometryState::resolve_node_placement`] (pushed placement > trusted
//! GPS > untrusted). Every route re-places EVERY member, not only the
//! reporter: a geometry push switches the whole frame (automatic -> site), and
//! a buffer whose members were placed in different frames would hand SRP a
//! meaningless geometry until each node happened to report again.

use super::*;
use crate::dsp_geometry_config::{NodePlacementResolution, ReportedNodeFix, SiteGeometryState};
use minimappr_core::localization_sensor_entry::{SensorPositionSource, SolveExclusionReason};
use minimappr_core::sensor_world_position::sensor_world_position_m;

/// What the registry keeps per member so any later route can re-place it.
#[derive(Clone, Debug)]
pub(crate) struct AutomaticClusterMember {
    pub(crate) body_offsets_m: Vec<[f32; 3]>,
    /// The member's latest packet fix; `None` when it reported no position.
    pub(crate) reported_fix: Option<ReportedNodeFix>,
}

/// One node's sensors in the frame in effect, in channel order.
pub(crate) struct PlacedNodeSensors {
    pub(crate) sensor_positions_m: Vec<[f64; 3]>,
    pub(crate) position_source: SensorPositionSource,
    pub(crate) position_std_m: Option<f64>,
}

pub(crate) fn place_automatic_cluster_member(
    site_geometry: &mut SiteGeometryState,
    node_id: &str,
    member: &AutomaticClusterMember,
) -> Result<PlacedNodeSensors, SolveExclusionReason> {
    match site_geometry.resolve_node_placement(node_id, member.reported_fix.as_ref()) {
        NodePlacementResolution::Untrusted { reason } => Err(reason),
        NodePlacementResolution::Placed {
            position_m,
            position_std_m,
            position_source,
            body_to_world_quaternion_wxyz,
        } => Ok(PlacedNodeSensors {
            sensor_positions_m: member
                .body_offsets_m
                .iter()
                .map(|offset| {
                    sensor_world_position_m(
                        position_m,
                        body_to_world_quaternion_wxyz,
                        offset.map(f64::from),
                    )
                })
                .collect(),
            position_source,
            position_std_m,
        }),
    }
}

/// Re-places every member except `reporting_node_id` (already placed by the
/// caller) in the frame in effect now. A member that no longer resolves to a
/// trusted position leaves the shared buffer. True when anything moved.
pub(crate) fn replace_other_automatic_cluster_members(
    set: &mut BufferSensorSet,
    site_geometry: &mut SiteGeometryState,
    reporting_node_id: &str,
) -> bool {
    let members = std::mem::take(&mut set.automatic_members);
    let mut changed = false;
    for (node_id, member) in members {
        if node_id == reporting_node_id {
            set.automatic_members.insert(node_id, member);
            continue;
        }
        match place_automatic_cluster_member(site_geometry, &node_id, &member) {
            Ok(placed) => {
                changed |= set.reposition_node_sensors(&node_id, &placed);
                set.automatic_members.insert(node_id, member);
            }
            Err(_) => changed |= set.remove_node(&node_id),
        }
    }
    changed
}
