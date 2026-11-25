//! The ordered sensor set one solve runs on, snapshotted by the sensor
//! registry when the manifest is ingested so a parked or queued solve keeps
//! the geometry it triggered with (it holds the `Arc`, the registry moves on).
//!
//! Index `i` of `sensors` / `positions_m` is buffer channel `i`; SRP, render
//! and the published provenance all index this one list.

use std::collections::BTreeMap;

use crate::srp_phat::HalfSpace;
pub(crate) use minimappr_core::localization_sensor_entry::NodeLocalRoutingReason;
use minimappr_core::localization_sensor_entry::{
    ExcludedSensor, SolveBufferScope, SolveCoordinateFrame, SolveExclusionReason, SolveSensorEntry,
};

/// Sensor ids are `<node_id>:ch<channel_index>` on both sides of the wire.
const SENSOR_ID_CHANNEL_SEPARATOR: &str = ":ch";

#[derive(Clone, Debug)]
pub(crate) struct SolveSensorLayout {
    pub(crate) sensors: Vec<SolveSensorEntry>,
    /// f32 copy of `sensors[].position_m`, the form SRP and render consume.
    pub(crate) positions_m: Vec<[f32; 3]>,
    pub(crate) layout_revision: u64,
    pub(crate) geometry_push_revision: Option<u64>,
    pub(crate) coordinate_frame: SolveCoordinateFrame,
    pub(crate) buffer_scope: SolveBufferScope,
    pub(crate) cluster_id: Option<String>,
    pub(crate) reporting_node_id: String,
    pub(crate) node_local_routing_reason: Option<NodeLocalRoutingReason>,
    /// Sensors in the buffer that this solve must not use, sorted by index.
    pub(crate) excluded_sensors: Vec<ExcludedSensor>,
    /// The reporting node's rate; every active sensor runs at it.
    pub(crate) solve_sample_rate_hz: u32,
    half_space_by_node: BTreeMap<String, HalfSpace>,
}

/// Everything the registry knows when it snapshots one layout.
pub(crate) struct SolveSensorLayoutParts {
    pub(crate) sensors: Vec<SolveSensorEntry>,
    pub(crate) layout_revision: u64,
    pub(crate) geometry_push_revision: Option<u64>,
    pub(crate) coordinate_frame: SolveCoordinateFrame,
    pub(crate) buffer_scope: SolveBufferScope,
    pub(crate) cluster_id: Option<String>,
    pub(crate) reporting_node_id: String,
    pub(crate) node_local_routing_reason: Option<NodeLocalRoutingReason>,
    pub(crate) solve_sample_rate_hz: u32,
    pub(crate) half_space_by_node: BTreeMap<String, HalfSpace>,
    /// Set only for the large-array omni path: that node's sensors are listed
    /// for provenance but excluded from any solve.
    pub(crate) geometry_unusable_node_id: Option<String>,
    pub(crate) max_solve_sensors: usize,
}

impl SolveSensorLayout {
    pub(crate) fn from_parts(parts: SolveSensorLayoutParts) -> Self {
        let excluded_sensors = solve_sensor_exclusions(&parts);
        let positions_m = parts
            .sensors
            .iter()
            .map(|sensor| sensor.position_m.map(|axis| axis as f32))
            .collect();
        Self {
            sensors: parts.sensors,
            positions_m,
            layout_revision: parts.layout_revision,
            geometry_push_revision: parts.geometry_push_revision,
            coordinate_frame: parts.coordinate_frame,
            buffer_scope: parts.buffer_scope,
            cluster_id: parts.cluster_id,
            reporting_node_id: parts.reporting_node_id,
            node_local_routing_reason: parts.node_local_routing_reason,
            excluded_sensors,
            solve_sample_rate_hz: parts.solve_sample_rate_hz,
            half_space_by_node: parts.half_space_by_node,
        }
    }

    pub(crate) fn sensor_count(&self) -> usize {
        self.positions_m.len()
    }

    /// Sensors this solve may use (the buffer's minus every exclusion).
    pub(crate) fn solve_eligible_sensor_count(&self) -> usize {
        self.sensor_count() - self.excluded_sensors.len()
    }

    /// Key of the dispatch gates this layout's solves share (`dsp_dispatch_gate_key`).
    pub(crate) fn dispatch_gate_key(&self, buffer_key: &str) -> String {
        super::dsp_dispatch_gate_key::dispatch_gate_key(buffer_key, self.solve_sample_rate_hz)
    }

    /// Resets the per-channel entries of excluded sensors (empty window, no
    /// coverage), so a wrong-rate or capped channel reaches neither SRP nor
    /// the render.
    pub(crate) fn clear_solve_excluded_channels<T: Default>(&self, per_channel: &mut [T]) {
        for excluded in &self.excluded_sensors {
            if let Some(entry) = per_channel.get_mut(excluded.sensor_index) {
                *entry = T::default();
            }
        }
    }

    /// The coplanar up/down prior only holds when every active sensor is on
    /// the node that declared it; a cross-node solve has no shared plane.
    pub(crate) fn solve_half_space(&self, active_sensor_indices: &[usize]) -> HalfSpace {
        let mut active_nodes = active_sensor_indices
            .iter()
            .filter_map(|index| self.sensors.get(*index))
            .map(|sensor| sensor.node_id.as_str());
        let Some(first_node) = active_nodes.next() else {
            return HalfSpace::None;
        };
        if active_nodes.any(|node_id| node_id != first_node) {
            return HalfSpace::None;
        }
        self.half_space_by_node
            .get(first_node)
            .copied()
            .unwrap_or(HalfSpace::None)
    }

    pub(crate) fn is_cross_node_pair(&self, sensor_index_a: usize, sensor_index_b: usize) -> bool {
        match (
            self.sensors.get(sensor_index_a),
            self.sensors.get(sensor_index_b),
        ) {
            (Some(a), Some(b)) => a.node_id != b.node_id,
            _ => false,
        }
    }

    /// `(sensor_id, position)` in buffer order, for the legacy
    /// `cluster_sensor_positions` manifest field.
    pub(crate) fn cluster_sensor_positions(&self) -> Vec<(String, [f32; 3])> {
        self.sensors
            .iter()
            .zip(&self.positions_m)
            .map(|(sensor, position)| (sensor.sensor_id.clone(), *position))
            .collect()
    }
}

/// Geometry-unusable node first, then any rate other than the solve rate (no
/// resampling this round), then the solve-sensor cap. The cap keeps the
/// reporting node's own sensors first so its own channels always solve, then
/// the rest in buffer (sensor) order.
fn solve_sensor_exclusions(parts: &SolveSensorLayoutParts) -> Vec<ExcludedSensor> {
    let mut excluded = Vec::new();
    let mut eligible_indices = Vec::new();
    for (sensor_index, sensor) in parts.sensors.iter().enumerate() {
        let reason = if parts.geometry_unusable_node_id.as_deref() == Some(&sensor.node_id) {
            Some(SolveExclusionReason::NodeArrayGeometryUnusable)
        } else if sensor.sample_rate_hz != parts.solve_sample_rate_hz {
            Some(SolveExclusionReason::SampleRateMismatch)
        } else {
            None
        };
        match reason {
            Some(reason) => excluded.push(ExcludedSensor {
                sensor_index,
                sensor_id: sensor.sensor_id.clone(),
                reason,
            }),
            None => eligible_indices.push(sensor_index),
        }
    }
    eligible_indices.sort_by_key(|index| parts.sensors[*index].node_id != parts.reporting_node_id);
    for sensor_index in eligible_indices.into_iter().skip(parts.max_solve_sensors) {
        excluded.push(ExcludedSensor {
            sensor_index,
            sensor_id: parts.sensors[sensor_index].sensor_id.clone(),
            reason: SolveExclusionReason::MaxSolveSensorsExceeded,
        });
    }
    excluded.sort_by_key(|entry| entry.sensor_index);
    excluded
}

pub(crate) fn sensor_id_for_node_channel(node_id: &str, channel_index: usize) -> String {
    format!("{node_id}{SENSOR_ID_CHANNEL_SEPARATOR}{channel_index}")
}

/// `node-a:ch3` -> (`node-a`, 3). An id without the channel suffix is a
/// single-sensor node: (id, 0).
pub(crate) fn split_sensor_id(sensor_id: &str) -> (String, usize) {
    sensor_id
        .rsplit_once(SENSOR_ID_CHANNEL_SEPARATOR)
        .and_then(|(node_id, channel)| Some((node_id.to_string(), channel.parse().ok()?)))
        .unwrap_or_else(|| (sensor_id.to_string(), 0))
}

/// Deterministic buffer order: node id, then numeric channel (so `ch10`
/// follows `ch2`), then the raw id to separate malformed duplicates.
pub(crate) fn sensor_order_key(sensor_id: &str) -> (String, usize, String) {
    let (node_id, channel_index) = split_sensor_id(sensor_id);
    (node_id, channel_index, sensor_id.to_string())
}

#[cfg(test)]
#[path = "dsp_solve_sensor_layout_tests.rs"]
mod dsp_solve_sensor_layout_tests;
