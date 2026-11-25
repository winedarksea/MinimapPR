//! The sensors registered behind one solve buffer, and the immutable
//! [`SolveSensorLayout`] snapshot a manifest takes of them.

use std::collections::BTreeMap;

use super::*;
use crate::srp_phat::HalfSpace;
use minimappr_core::localization_sensor_entry::{
    CoordinateFrameKind, NodeTimeQuality, SensorPositionSource, SolveBufferScope,
    SolveCoordinateFrame, SolveSensorEntry,
};

/// Timebase of every packet timestamp the sidecar decodes today.
const PACKET_CLOCK_DOMAIN_UTC: &str = "utc";

pub(crate) type SensorOrderKey = (String, usize, String);

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RegisteredSensor {
    pub(crate) position_m: [f64; 3],
    pub(crate) position_source: SensorPositionSource,
    pub(crate) position_std_m: Option<f64>,
    /// `None` until the sensor's own node streams (declared-only members).
    pub(crate) sample_rate_hz: Option<u32>,
    pub(crate) last_refreshed_ns: u128,
}

pub(crate) struct SensorObservation {
    pub(crate) position_m: [f64; 3],
    pub(crate) position_source: SensorPositionSource,
    pub(crate) position_std_m: Option<f64>,
    pub(crate) sample_rate_hz: Option<u32>,
}

/// The sensors behind one buffer key.
pub(crate) struct BufferSensorSet {
    pub(crate) buffer_scope: SolveBufferScope,
    pub(crate) cluster_id: Option<String>,
    pub(crate) sensors: BTreeMap<SensorOrderKey, RegisteredSensor>,
    pub(crate) half_space_by_node: BTreeMap<String, HalfSpace>,
    /// Automatic-cluster sets only: what re-places each member per route.
    pub(crate) automatic_members: BTreeMap<String, AutomaticClusterMember>,
    /// Frame the positions were last expressed in; a change (geometry push,
    /// automatic origin adoption) is a layout change even if no number moved.
    pub(crate) coordinate_frame: Option<SolveCoordinateFrame>,
    pub(crate) layout_revision: u64,
}

impl BufferSensorSet {
    pub(crate) fn new(buffer_scope: SolveBufferScope, cluster_id: Option<String>) -> Self {
        Self {
            buffer_scope,
            cluster_id,
            sensors: BTreeMap::new(),
            half_space_by_node: BTreeMap::new(),
            automatic_members: BTreeMap::new(),
            coordinate_frame: None,
            layout_revision: 0,
        }
    }

    /// True when the sensor is new or anything a solve reads about it changed.
    pub(crate) fn observe(
        &mut self,
        sensor_id: &str,
        observation: SensorObservation,
        now_ns: u128,
    ) -> bool {
        let key = sensor_order_key(sensor_id);
        let existing = self.sensors.get(&key);
        let sample_rate_hz = observation
            .sample_rate_hz
            .or_else(|| existing.and_then(|sensor| sensor.sample_rate_hz));
        let updated = RegisteredSensor {
            position_m: observation.position_m,
            position_source: observation.position_source,
            position_std_m: observation.position_std_m,
            sample_rate_hz,
            last_refreshed_ns: now_ns,
        };
        let changed = existing.is_none_or(|existing| {
            RegisteredSensor {
                last_refreshed_ns: now_ns,
                ..existing.clone()
            } != updated
        });
        self.sensors.insert(key, updated);
        changed
    }

    /// Drops channels a node no longer reports (channel count shrank).
    pub(crate) fn retain_node_channels(&mut self, node_id: &str, channel_count: usize) -> bool {
        let before = self.sensors.len();
        self.sensors
            .retain(|(sensor_node_id, channel_index, _), _| {
                sensor_node_id != node_id || *channel_index < channel_count
            });
        self.sensors.len() != before
    }

    pub(crate) fn evict_stale(&mut self, now_ns: u128, stale_after_ns: u128) -> bool {
        let before = self.sensors.len();
        self.sensors
            .retain(|_, sensor| now_ns.saturating_sub(sensor.last_refreshed_ns) <= stale_after_ns);
        let sensors = &self.sensors;
        let node_has_sensors = |node_id: &String| {
            sensors
                .keys()
                .any(|(sensor_node_id, _, _)| sensor_node_id == node_id)
        };
        self.half_space_by_node
            .retain(|node_id, _| node_has_sensors(node_id));
        self.automatic_members
            .retain(|node_id, _| node_has_sensors(node_id));
        self.sensors.len() != before
    }

    /// Exactly channels `0..channel_count` of the node are registered, each
    /// refreshed within `stale_after_ns`.
    pub(crate) fn node_sensors_fresh(
        &self,
        node_id: &str,
        channel_count: usize,
        now_ns: u128,
        stale_after_ns: u128,
    ) -> bool {
        let mut registered_channel_count = 0;
        for ((sensor_node_id, channel_index, _), sensor) in &self.sensors {
            if sensor_node_id != node_id {
                continue;
            }
            if *channel_index >= channel_count
                || now_ns.saturating_sub(sensor.last_refreshed_ns) > stale_after_ns
            {
                return false;
            }
            registered_channel_count += 1;
        }
        registered_channel_count == channel_count
    }

    /// Drops every trace of a node (it lost its trusted position).
    pub(crate) fn remove_node(&mut self, node_id: &str) -> bool {
        let before = self.sensors.len();
        self.sensors
            .retain(|(sensor_node_id, _, _), _| sensor_node_id != node_id);
        self.half_space_by_node.remove(node_id);
        self.automatic_members.remove(node_id);
        self.sensors.len() != before
    }

    /// Moves a node's registered sensors without refreshing them: only the
    /// node's own packets keep it from going stale.
    pub(crate) fn reposition_node_sensors(
        &mut self,
        node_id: &str,
        placed: &PlacedNodeSensors,
    ) -> bool {
        let mut changed = false;
        for ((sensor_node_id, channel_index, _), sensor) in &mut self.sensors {
            if sensor_node_id != node_id {
                continue;
            }
            let Some(position_m) = placed.sensor_positions_m.get(*channel_index) else {
                continue;
            };
            let moved = sensor.position_m != *position_m
                || sensor.position_source != placed.position_source
                || sensor.position_std_m != placed.position_std_m;
            if moved {
                sensor.position_m = *position_m;
                sensor.position_source = placed.position_source;
                sensor.position_std_m = placed.position_std_m;
                changed = true;
            }
        }
        changed
    }

    /// Records the frame of this route; true when it differs from the last one.
    pub(crate) fn adopt_coordinate_frame(&mut self, frame: &SolveCoordinateFrame) -> bool {
        if self.coordinate_frame.as_ref() == Some(frame) {
            return false;
        }
        self.coordinate_frame = Some(frame.clone());
        true
    }

    pub(crate) fn set_node_half_space(&mut self, node_id: &str, half_space: HalfSpace) {
        if half_space == HalfSpace::None {
            self.half_space_by_node.remove(node_id);
        } else {
            self.half_space_by_node
                .insert(node_id.to_string(), half_space);
        }
    }
}

pub(crate) struct RouteSnapshotContext {
    pub(crate) coordinate_frame: SolveCoordinateFrame,
    pub(crate) geometry_push_revision: Option<u64>,
    pub(crate) node_local_routing_reason: Option<NodeLocalRoutingReason>,
    pub(crate) geometry_unusable_node_id: Option<String>,
    pub(crate) max_solve_sensors: usize,
}

pub(crate) fn node_body_snapshot_context(
    node_local_routing_reason: Option<NodeLocalRoutingReason>,
    geometry_unusable_node_id: Option<String>,
    max_solve_sensors: usize,
) -> RouteSnapshotContext {
    RouteSnapshotContext {
        coordinate_frame: SolveCoordinateFrame {
            frame_id: None,
            kind: CoordinateFrameKind::NodeBody,
            origin: None,
            projection: None,
        },
        geometry_push_revision: None,
        node_local_routing_reason,
        geometry_unusable_node_id,
        max_solve_sensors,
    }
}

pub(crate) fn snapshot_route(
    set: &BufferSensorSet,
    buffer_key: String,
    request: &NodeRoutingRequest,
    node_id: &str,
    context: RouteSnapshotContext,
) -> SensorRoute {
    let sensors: Vec<SolveSensorEntry> = set
        .sensors
        .iter()
        .map(
            |((sensor_node_id, node_channel_index, sensor_id), sensor)| SolveSensorEntry {
                sensor_id: sensor_id.clone(),
                node_id: sensor_node_id.clone(),
                node_channel_index: *node_channel_index,
                position_m: sensor.position_m,
                // A declared member that never streamed has no audio; it is
                // listed at the solve rate rather than flagged as a mismatch.
                sample_rate_hz: sensor.sample_rate_hz.unwrap_or(request.sample_rate_hz),
                position_source: sensor.position_source,
                position_std_m: sensor.position_std_m,
                clock_source: NodeTimeQuality::Unknown,
                clock_domain: PACKET_CLOCK_DOMAIN_UTC.to_string(),
                clock_offset_std_s: None,
            },
        )
        .collect();
    let buffer_sensor_ids: Vec<String> = sensors
        .iter()
        .map(|sensor| sensor.sensor_id.clone())
        .collect();
    let buffer_channel_indices = (0..request.channel_count)
        .map(|channel_index| {
            let sensor_id = sensor_id_for_node_channel(node_id, channel_index);
            buffer_sensor_ids
                .iter()
                .position(|candidate| *candidate == sensor_id)
                .expect("every routed node channel is registered in its buffer set")
        })
        .collect();
    let layout = SolveSensorLayout::from_parts(SolveSensorLayoutParts {
        sensors,
        layout_revision: set.layout_revision,
        geometry_push_revision: context.geometry_push_revision,
        coordinate_frame: context.coordinate_frame,
        buffer_scope: set.buffer_scope,
        cluster_id: set.cluster_id.clone(),
        reporting_node_id: node_id.to_string(),
        node_local_routing_reason: context.node_local_routing_reason,
        solve_sample_rate_hz: request.sample_rate_hz,
        half_space_by_node: set.half_space_by_node.clone(),
        geometry_unusable_node_id: context.geometry_unusable_node_id,
        max_solve_sensors: context.max_solve_sensors,
    });
    SensorRoute {
        buffer_key,
        buffer_sensor_ids: Arc::new(buffer_sensor_ids),
        buffer_channel_indices,
        layout: Arc::new(layout),
    }
}
